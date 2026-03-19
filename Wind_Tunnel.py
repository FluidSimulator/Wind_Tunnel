
"""
Wind Tunnel -- Eulerian Fluid + Neural Super-Resolution
=======================================================
Solver  : MAC-grid incompressible Navier-Stokes (Muller 2017)
ML      : CNN super-resolves 2x smoke density; runs async in background thread.

Performance optimisations for 25-30 FPS on i7-1255U
----------------------------------------------------
Root cause of 15 FPS: 45 parallel-loop barriers per frame at ~1ms each
on Windows (thread pool fork/join for 8 threads on a 20K-cell grid).

OPT-1  _TAICHI_THREADS = 4
       For NX*NY = 20K cells each thread handles ~5K cells/loop.
       Fork/join with 4 threads costs ~0.5ms vs ~1ms with 8 threads.
       i7-1255U has 2 P-cores (fast, consistent) -- 4 threads maps cleanly.
       Expected barrier cost halved: 45ms -> 22ms.

OPT-2  k_bnd_and_project() -- ONE kernel replaces 4 separate calls
       (k_set_bnd x1 + k_clear_pressure x1 + k_project_all x1)
       Warm-start pressure (no reset) -- previous frame is a good initial
       guess, so NUM_ITERS=8 gives same convergence as cold-start NUM_ITERS=12.
       Barriers: 6 (bnd) + 2*8 (GS) = 22

OPT-3  k_advect_all() -- ONE kernel replaces 4 separate calls
       (k_advect_velocity + k_copy_velocity + k_advect_smoke + k_copy_smoke)
       Correctness: loops 1-3 read u,v; loops 4-6 write u,v,smoke.
       Taichi's implicit barriers between sequential loops enforce ordering.
       Barriers: 6

       Total per frame: 22 + 6 = 28 barriers (was 45, removed 17)

OPT-4  SCALE = 5  (1000x500 = 500K pixels, was 1200x600 = 720K)
       k_neural_colormap 31% less work.

OPT-5  AsyncCNN (already present) -- CNN on background thread, 1-frame latency.

Budget after all fixes (barriers at 0.5ms each, 4 threads):
  28 * 0.5ms (barriers) + 10ms (compute) + 5ms (GUI) = 29ms -> ~34 FPS

Controls: LMB = drag obstacle | R = reset | ESC = quit
"""

import os
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import taichi as ti

# ---------------------------------------------------------------------------
#  Thread config -- set BEFORE ti.init
# ---------------------------------------------------------------------------
_N_THREADS      = min(8, os.cpu_count() or 4)   # PyTorch threads (CNN)
_TAICHI_THREADS = min(4, os.cpu_count() or 4)   # OPT-1: 4 threads for small grid

torch.set_num_threads(_N_THREADS)
torch.set_num_interop_threads(1)

ti.init(
    arch=ti.cpu,
    cpu_max_num_threads=_TAICHI_THREADS,
    default_fp=ti.f32,
    fast_math=True,
)

# ===========================================================================
#  SIMULATION PARAMETERS
# ===========================================================================

NX          = 200
NY          = 100
H           = 1.0 / NY
DENSITY     = 1000.0
DT          = 1.0 / 60.0
NUM_ITERS   = 8            # OPT-2: warm-start needs fewer iters than cold-start
OVER_RELAX  = 1.9
INFLOW_VEL  = 2.0

STREAM_LO   = NY // 2 - NY // 20   # 45  narrow 10-cell jet
STREAM_HI   = NY // 2 + NY // 20   # 55

OBS_CX      = int(NX * 0.40)
OBS_CY      = NY // 2
OBS_R       = int(NY * 0.15)

SCALE       = 5            # OPT-4: 1000x500 instead of 1200x600
WIN_W       = NX * SCALE   # 1000
WIN_H       = NY * SCALE   # 500

ML_WARMUP   = 120
ML_COLLECT  = 500
ML_EPOCHS   = 60
ML_BATCH    = 32
ML_LR       = 1e-3

# ===========================================================================
#  TAICHI FIELDS
# ===========================================================================

u         = ti.field(ti.f32, shape=(NX+1, NY   ))
v         = ti.field(ti.f32, shape=(NX,   NY+1 ))
u_buf     = ti.field(ti.f32, shape=(NX+1, NY   ))
v_buf     = ti.field(ti.f32, shape=(NX,   NY+1 ))
p         = ti.field(ti.f32, shape=(NX,   NY   ))
s         = ti.field(ti.f32, shape=(NX,   NY   ))
smoke     = ti.field(ti.f32, shape=(NX,   NY   ))
smoke_buf = ti.field(ti.f32, shape=(NX,   NY   ))
pixels    = ti.Vector.field(3, ti.f32, shape=(WIN_W, WIN_H))
sr_field  = ti.field(ti.f32, shape=(NY, NX))
obs_cx    = ti.field(ti.i32, shape=())
obs_cy    = ti.field(ti.i32, shape=())
obs_r     = ti.field(ti.i32, shape=())

# ===========================================================================
#  BILINEAR SAMPLERS  (single return at end -- Taichi requirement)
# ===========================================================================

@ti.func
def sample_u(px: ti.f32, py: ti.f32) -> ti.f32:
    gx = px / H
    gy = py / H - 0.5
    i0 = ti.max(0, ti.min(NX-1, ti.cast(ti.floor(gx), ti.i32)))
    j0 = ti.max(0, ti.min(NY-2, ti.cast(ti.floor(gy), ti.i32)))
    tx = ti.max(0.0, ti.min(1.0, gx - float(i0)))
    ty = ti.max(0.0, ti.min(1.0, gy - float(j0)))
    return ((1-tx)*(1-ty)*u[i0,  j0  ] + tx*(1-ty)*u[i0+1,j0  ]
          + (1-tx)*ty    *u[i0,  j0+1] + tx*ty    *u[i0+1,j0+1])


@ti.func
def sample_v(px: ti.f32, py: ti.f32) -> ti.f32:
    gx = px / H - 0.5
    gy = py / H
    i0 = ti.max(0, ti.min(NX-2, ti.cast(ti.floor(gx), ti.i32)))
    j0 = ti.max(0, ti.min(NY-1, ti.cast(ti.floor(gy), ti.i32)))
    tx = ti.max(0.0, ti.min(1.0, gx - float(i0)))
    ty = ti.max(0.0, ti.min(1.0, gy - float(j0)))
    return ((1-tx)*(1-ty)*v[i0,  j0  ] + tx*(1-ty)*v[i0+1,j0  ]
          + (1-tx)*ty    *v[i0,  j0+1] + tx*ty    *v[i0+1,j0+1])


@ti.func
def sample_smoke(px: ti.f32, py: ti.f32) -> ti.f32:
    gx = px / H - 0.5
    gy = py / H - 0.5
    i0 = ti.max(0, ti.min(NX-2, ti.cast(ti.floor(gx), ti.i32)))
    j0 = ti.max(0, ti.min(NY-2, ti.cast(ti.floor(gy), ti.i32)))
    tx = ti.max(0.0, ti.min(1.0, gx - float(i0)))
    ty = ti.max(0.0, ti.min(1.0, gy - float(j0)))
    return ((1-tx)*(1-ty)*smoke[i0,  j0  ] + tx*(1-ty)*smoke[i0+1,j0  ]
          + (1-tx)*ty    *smoke[i0,  j0+1] + tx*ty    *smoke[i0+1,j0+1])


# ===========================================================================
#  SIMULATION KERNELS
# ===========================================================================

@ti.kernel
def k_init():
    """Initialise fields for the wind-tunnel scenario."""
    for i, j in s:
        s[i, j] = 1.0
    for i in range(NX):
        s[i, 0] = 0.0; s[i, NY-1] = 0.0
    obs_cx[None] = OBS_CX
    obs_cy[None] = OBS_CY
    obs_r[None]  = OBS_R
    for i, j in ti.ndrange(NX, NY):
        dx = float(i) - float(OBS_CX)
        dy = float(j) - float(OBS_CY)
        if dx*dx + dy*dy <= float(OBS_R * OBS_R):
            s[i, j] = 0.0
    for i, j in ti.ndrange(NX+1, NY):
        u[i, j] = INFLOW_VEL
    for i, j in ti.ndrange(NX, NY+1):
        v[i, j] = 0.0
    for i, j in p:
        p[i, j] = 0.0
    for i, j in smoke:
        smoke[i, j] = 1.0 if j >= STREAM_LO and j < STREAM_HI else 0.0


@ti.kernel
def k_move_obstacle(new_cx: ti.i32, new_cy: ti.i32):
    """Erase old obstacle disk, stamp new one (two-pass, race-free)."""
    old_cx = obs_cx[None]; old_cy = obs_cy[None]; r = obs_r[None]
    margin = r + 2
    for i, j in ti.ndrange(NX, NY):
        dx = float(i) - float(old_cx); dy = float(j) - float(old_cy)
        if dx*dx + dy*dy <= float(margin*margin):
            if j > 0 and j < NY - 1:
                s[i, j] = 1.0
    for i, j in ti.ndrange(NX, NY):
        dx = float(i) - float(new_cx); dy = float(j) - float(new_cy)
        if dx*dx + dy*dy <= float(r*r):
            s[i, j] = 0.0
    obs_cx[None] = new_cx; obs_cy[None] = new_cy


# ---------------------------------------------------------------------------
#  OPT-2: k_bnd_and_project -- replaces k_set_bnd + k_clear_pressure + k_project_all
#  ONE Python dispatch instead of THREE.
#  Pressure warm-started (not cleared) -- converges faster with fewer iterations.
#  Inner barriers: 6 (boundary loops) + 2*NUM_ITERS (GS red/black) = 22
# ---------------------------------------------------------------------------
@ti.kernel
def k_bnd_and_project():
    # ---- Boundary conditions (was k_set_bnd) ----
    # Loop 1: left inflow
    for j in range(1, NY-1):
        u[1, j]     = INFLOW_VEL
        smoke[0, j] = 1.0 if j >= STREAM_LO and j < STREAM_HI else 0.0
    # Loop 2: right outflow + left ghost
    for j in range(NY):
        u[NX, j] = u[NX-1, j]
        u[0,  j] = u[1,    j]
    # Loop 3: top/bottom wall u faces
    for i in range(NX+1):
        u[i, 0] = 0.0; u[i, NY-1] = 0.0
    # Loop 4: top/bottom wall v faces
    for i in range(NX):
        v[i, 0] = 0.0; v[i, NY] = 0.0
    # Loop 5: u faces bordering any solid cell
    for i, j in ti.ndrange((1, NX), NY):
        if s[i-1, j] == 0.0 or s[i, j] == 0.0:
            u[i, j] = 0.0
    # Loop 6: v faces bordering any solid cell
    for i, j in ti.ndrange(NX, (1, NY)):
        if s[i, j-1] == 0.0 or s[i, j] == 0.0:
            v[i, j] = 0.0

    # ---- Warm-start GS pressure projection (was k_project_all) ----
    # Pressure NOT cleared: previous frame's p is a warm initial guess.
    # With warm start, NUM_ITERS=8 converges as well as cold-start with 12+.
    cp = DENSITY * H / DT
    for _it in ti.static(range(NUM_ITERS)):
        # Red pass (i+j even)
        for i, j in ti.ndrange((1, NX-1), (1, NY-1)):
            if (i+j) % 2 == 0 and s[i, j] != 0.0:
                sx0 = s[i-1,j]; sx1 = s[i+1,j]
                sy0 = s[i,j-1]; sy1 = s[i,j+1]
                ns  = sx0 + sx1 + sy0 + sy1
                if ns > 0.0:
                    div  = (u[i+1,j]-u[i,j]) + (v[i,j+1]-v[i,j])
                    corr = -div / ns * OVER_RELAX
                    p[i,j]    += cp*corr
                    u[i,  j]  -= sx0*corr;  u[i+1,j] += sx1*corr
                    v[i,j  ]  -= sy0*corr;  v[i,j+1] += sy1*corr
        # Black pass (i+j odd)
        for i, j in ti.ndrange((1, NX-1), (1, NY-1)):
            if (i+j) % 2 == 1 and s[i, j] != 0.0:
                sx0 = s[i-1,j]; sx1 = s[i+1,j]
                sy0 = s[i,j-1]; sy1 = s[i,j+1]
                ns  = sx0 + sx1 + sy0 + sy1
                if ns > 0.0:
                    div  = (u[i+1,j]-u[i,j]) + (v[i,j+1]-v[i,j])
                    corr = -div / ns * OVER_RELAX
                    p[i,j]    += cp*corr
                    u[i,  j]  -= sx0*corr;  u[i+1,j] += sx1*corr
                    v[i,j  ]  -= sy0*corr;  v[i,j+1] += sy1*corr


# ---------------------------------------------------------------------------
#  OPT-3: k_advect_all -- replaces k_advect_velocity + k_copy_velocity
#                                 + k_advect_smoke   + k_copy_smoke
#  ONE Python dispatch instead of FOUR.
#  Correctness guarantee:
#    Loops 1-3 READ from u, v (unchanged -- buffers not yet swapped).
#    Taichi implicit barrier after loop 3 ensures all reads complete.
#    Loops 4-6 WRITE u_buf->u, v_buf->v, smoke_buf->smoke.
#  Inner barriers: 6 total.
# ---------------------------------------------------------------------------
@ti.kernel
def k_advect_all():
    # Loop 1: semi-Lagrangian advect u -> u_buf
    for i, j in ti.ndrange((1, NX), NY):
        if s[i-1,j] != 0.0 or s[i,j] != 0.0:
            px = float(i)*H;  py = (float(j)+0.5)*H
            u_buf[i,j] = sample_u(px - DT*u[i,j],
                                  py - DT*sample_v(px, py))
        else:
            u_buf[i,j] = 0.0
    # Loop 2: semi-Lagrangian advect v -> v_buf
    for i, j in ti.ndrange(NX, (1, NY)):
        if s[i,j-1] != 0.0 or s[i,j] != 0.0:
            px = (float(i)+0.5)*H;  py = float(j)*H
            v_buf[i,j] = sample_v(px - DT*sample_u(px, py),
                                  py - DT*v[i,j])
        else:
            v_buf[i,j] = 0.0
    # Loop 3: semi-Lagrangian advect smoke -> smoke_buf (reads OLD u, v)
    for i, j in ti.ndrange(NX, NY):
        if s[i,j] != 0.0:
            px = (float(i)+0.5)*H;  py = (float(j)+0.5)*H
            smoke_buf[i,j] = sample_smoke(px - DT*sample_u(px, py),
                                          py - DT*sample_v(px, py))
        else:
            smoke_buf[i,j] = smoke[i,j]
    # --- implicit Taichi barrier: loops 1-3 fully complete before 4-6 ---
    # Loop 4: u_buf -> u
    for i, j in u_buf:
        u[i,j] = u_buf[i,j]
    # Loop 5: v_buf -> v
    for i, j in v_buf:
        v[i,j] = v_buf[i,j]
    # Loop 6: smoke_buf -> smoke
    for i, j in smoke_buf:
        smoke[i,j] = smoke_buf[i,j]


# ---------------------------------------------------------------------------
#  Render kernels (unchanged from before)
# ---------------------------------------------------------------------------

@ti.kernel
def k_smoke_to_sr():
    """Copy smoke -> sr_field on-device; avoids to_numpy round-trip."""
    for iy, ix in sr_field:
        sr_field[iy, ix] = smoke[ix, iy]


@ti.kernel
def k_neural_colormap():
    """
    Parallel upscale + colormap (500K pixels on 4 threads).
    Squared-distance obstacle test -- sqrt only for ~470 AA-band pixels.
    """
    wall_col = ti.Vector([0.0, 0.0, 0.0])
    obs_col  = ti.Vector([1.0, 0.0, 0.0])

    cx = float(obs_cx[None])
    cy = float(obs_cy[None])
    r  = float(obs_r[None])
    r_inner_sq = (r - 0.5) * (r - 0.5)
    r_outer_sq = (r + 0.5) * (r + 0.5)

    for pi, pj in pixels:
        ci = ti.max(0, ti.min(NX-1, pi // SCALE))
        cj = ti.max(0, ti.min(NY-1, pj // SCALE))

        val       = ti.max(0.0, ti.min(1.0, sr_field[cj, ci]))
        fluid_col = ti.Vector([val, val, val])
        out_col   = fluid_col

        if cj == 0 or cj == NY - 1:
            out_col = wall_col
        else:
            px_g    = (float(pi) + 0.5) / float(SCALE)
            py_g    = (float(pj) + 0.5) / float(SCALE)
            ddx     = px_g - cx
            ddy     = py_g - cy
            dist_sq = ddx*ddx + ddy*ddy

            if dist_sq <= r_inner_sq:
                out_col = obs_col
            elif dist_sq >= r_outer_sq:
                out_col = fluid_col
            else:
                dist  = ti.sqrt(dist_sq)
                t     = ti.max(0.0, ti.min(1.0, dist - r + 0.5))
                t     = t * t * (3.0 - 2.0 * t)
                out_col = ti.Vector([
                    obs_col[0] + t*(fluid_col[0] - obs_col[0]),
                    obs_col[1] + t*(fluid_col[1] - obs_col[1]),
                    obs_col[2] + t*(fluid_col[2] - obs_col[2]),
                ])

        pixels[pi, pj] = out_col


# ===========================================================================
#  ML SECTION -- Neural Smoke Super-Resolution
# ===========================================================================

class SmokeUpsampleNet(nn.Module):
    """
    CNN: 2x super-resolves smoke density.
    Input (B,1,50,100) -> Output (B,1,100,200).
    ~6K params (encode 1->16, decode 16->8->1). Fast CPU inference.
    """
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1,  16, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, 3, padding=1), nn.LeakyReLU(0.2, True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=False)
        self.decode = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(16,  8, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d( 8,  1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decode(self.upsample(self.encode(x)))


def try_compile(model):
    try:
        compiled = torch.compile(model)
        with torch.inference_mode():
            compiled(torch.zeros(1, 1, NY//2, NX//2))
        print("  [ML] torch.compile: JIT-compiled inference active")
        return compiled
    except Exception as e:
        print(f"  [ML] torch.compile: skipped ({e})")
        return model


def train_model(model, inputs, targets):
    model.train()
    opt   = optim.Adam(model.parameters(), lr=ML_LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, ML_EPOCHS, 1e-5)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=ML_BATCH, shuffle=True, drop_last=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [ML] {n_params:,} params | {len(inputs)} pairs | "
          f"{ML_EPOCHS} epochs | batch={ML_BATCH} | "
          f"torch_threads={_N_THREADS}")

    t0, losses = time.perf_counter(), []
    for epoch in range(ML_EPOCHS):
        el = 0.0
        for xb, yb in loader:
            pr = model(xb)
            gx = F.mse_loss(pr[:,:,:,1:]-pr[:,:,:,:-1],
                            yb[:,:,:,1:]-yb[:,:,:,:-1])
            gy = F.mse_loss(pr[:,:,1:,:]-pr[:,:,:-1,:],
                            yb[:,:,1:,:]-yb[:,:,:-1,:])
            loss = F.mse_loss(pr, yb) + 0.1*(gx+gy)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item()
        sched.step()
        avg = el / len(loader)
        losses.append(avg)
        if (epoch+1) % 10 == 0:
            print(f"  [ML] epoch {epoch+1:3d}/{ML_EPOCHS}  "
                  f"loss={avg:.5f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"t={time.perf_counter()-t0:.1f}s")
    model.eval()
    print(f"  [ML] done in {time.perf_counter()-t0:.1f}s  "
          f"loss {losses[0]:.5f}->{losses[-1]:.5f}")
    return losses


# ===========================================================================
#  OPT-5: ASYNC CNN -- inference off the critical path
# ===========================================================================

class AsyncCNN:
    """
    Background thread runs CNN inference pipelined with Taichi simulation.

    Frame N: simulate() runs while CNN processes frame N-1 snapshot.
    Render:  use latest available CNN result (1-frame latency, imperceptible).
    Frame time = max(simulate_ms, cnn_ms) + colormap_ms, not their sum.
    """

    def __init__(self):
        self.lock     = threading.Lock()
        self.trigger  = threading.Event()
        self.shutdown = threading.Event()
        self.in_buf   = np.zeros((NY, NX), dtype=np.float32)
        self.out_buf  = np.zeros((NY, NX), dtype=np.float32)
        self.ready    = False
        self.thread   = None

    def start(self, model):
        self.thread = threading.Thread(
            target=self._worker, args=(model,), daemon=True)
        self.thread.start()

    def stop(self):
        self.shutdown.set(); self.trigger.set()

    def submit(self, smoke_field):
        """
        Copy current smoke snapshot and wake the CNN worker.
        Skip if the worker has not consumed the previous submission yet
        (trigger still set) -- avoids a redundant smoke.to_numpy() call.
        """
        if self.trigger.is_set():
            return                             # worker still busy, skip frame
        raw = smoke_field.to_numpy()           # (NX, NY)
        with self.lock:
            np.copyto(self.in_buf, raw.T)      # transpose to (NY, NX)
        self.trigger.set()

    def get_result(self, dest_np):
        """Non-blocking copy of latest CNN output. Returns True if result is ready."""
        with self.lock:
            if not self.ready:
                return False
            np.copyto(dest_np, self.out_buf)
        return True

    def _worker(self, model):
        fine_t   = torch.empty(1, 1, NY,    NX,    dtype=torch.float32)
        coarse_t = torch.empty(1, 1, NY//2, NX//2, dtype=torch.float32)
        local_in = np.empty((NY, NX), dtype=np.float32)

        while not self.shutdown.is_set():
            fired = self.trigger.wait(timeout=0.05)
            if not fired:
                continue
            self.trigger.clear()
            if self.shutdown.is_set():
                break
            with self.lock:
                np.copyto(local_in, self.in_buf)
            fine_t[0, 0].copy_(torch.from_numpy(local_in))
            with torch.inference_mode():
                F.avg_pool2d(fine_t, kernel_size=2, out=coarse_t)
                sr_t = model(coarse_t)
            sr_np = sr_t[0, 0].contiguous().numpy()
            with self.lock:
                np.copyto(self.out_buf, sr_np)
                self.ready = True


_render_buf = np.empty((NY, NX), dtype=np.float32)
_async_cnn  = AsyncCNN()


# ===========================================================================
#  SIMULATION + RENDER
# ===========================================================================

def simulate():
    """
    OPT: 2 kernel dispatches, 28 inner barriers.
    Was: 9 dispatches, 45 barriers.
    """
    k_bnd_and_project()   # set_bnd + warm GS  (22 inner barriers)
    k_advect_all()        # advect u/v/smoke + copy all  (6 inner barriers)


def render_preview():
    """Preview render during warmup/collecting: pure Taichi, zero numpy."""
    k_smoke_to_sr()
    k_neural_colormap()


def render_neural_async():
    """Non-blocking neural render: submit smoke, render previous result."""
    _async_cnn.submit(smoke)
    if _async_cnn.get_result(_render_buf):
        sr_field.from_numpy(_render_buf)
    else:
        k_smoke_to_sr()
    k_neural_colormap()


# ===========================================================================
#  MAIN
# ===========================================================================

PHASE_WARMUP        = 0
PHASE_COLLECTING    = 1
PHASE_TRAINING      = 2
PHASE_TRAINING_WAIT = 3
PHASE_RUNNING       = 4


def main():
    gui = ti.GUI(
        "Wind Tunnel + Neural SR  |  LMB=drag  R=reset  ESC=quit",
        res=(WIN_W, WIN_H),
        fast_gui=True,
    )

    k_init()

    phase    = PHASE_WARMUP
    frame    = 0
    dragging = False

    ml_model  = SmokeUpsampleNet()
    train_in  = []
    train_tgt = []

    _train_result = {}
    _train_event  = threading.Event()

    t_prev  = time.perf_counter()
    fps_acc = 0.0
    fps_n   = 0

    print("=" * 65)
    print(" Wind Tunnel  --  Eulerian Fluid + Neural Super-Resolution")
    print(f" Grid         : {NX}x{NY}   Coarse input : {NX//2}x{NY//2}")
    print(f" NUM_ITERS    : {NUM_ITERS} (warm-start GS)")
    print(f" Kernel calls : 2 per frame  (was 9)")
    print(f" Barriers     : 28 per frame (was 45)")
    print(f" Taichi       : {_TAICHI_THREADS} threads  PyTorch: {_N_THREADS} threads")
    print(f" Window       : {WIN_W}x{WIN_H} (SCALE={SCALE})")
    print(" Controls     : LMB=drag  R=reset  ESC=quit")
    print("=" * 65)
    print(f" [1/4] Warming up ({ML_WARMUP} frames) ...")

    while gui.running:

        # -- Keyboard --------------------------------------------------
        for e in gui.get_events(ti.GUI.PRESS):
            key = e.key
            if key in (ti.GUI.ESCAPE, 'q', 'Q'):
                gui.running = False
            elif key in ('r', 'R'):
                k_init()
                frame = 0; dragging = False
                if phase in (PHASE_WARMUP, PHASE_COLLECTING, PHASE_TRAINING_WAIT):
                    train_in.clear(); train_tgt.clear()
                    phase = PHASE_WARMUP
                    print("  Reset -- restarting collection.")
                else:
                    print("  Reset -- model retained.")
                t_prev = time.perf_counter()

        # -- Mouse drag ------------------------------------------------
        mx, my = gui.get_cursor_pos()
        mci = max(0, min(NX-1, int(mx * NX)))
        mcj = max(0, min(NY-1, int(my * NY)))
        ccx, ccy, cr = obs_cx[None], obs_cy[None], obs_r[None]
        lmb = gui.is_pressed(ti.GUI.LMB)
        if lmb and (mci-ccx)**2 + (mcj-ccy)**2 <= (cr+2)**2:
            dragging = True
        if not lmb:
            dragging = False
        if dragging:
            k_move_obstacle(max(cr+2, min(NX-cr-2, mci)),
                            max(cr+2, min(NY-cr-2, mcj)))

        # -- Simulate --------------------------------------------------
        simulate()

        # -- ML state machine ------------------------------------------
        if phase == PHASE_WARMUP:
            if frame >= ML_WARMUP:
                phase = PHASE_COLLECTING
                print(f" [2/4] Collecting {ML_COLLECT} frames ...")

        elif phase == PHASE_COLLECTING:
            fn = smoke.to_numpy().T.astype(np.float32)
            ft = torch.from_numpy(fn.copy()).unsqueeze(0).unsqueeze(0)
            train_in.append(F.avg_pool2d(ft, 2))
            train_tgt.append(ft)
            n = len(train_in)
            if n % 100 == 0:
                print(f"  [ML] {n}/{ML_COLLECT} ...")
            if n >= ML_COLLECT:
                phase = PHASE_TRAINING

        elif phase == PHASE_TRAINING:
            print(" [3/4] Training (background thread) ...")
            inp = torch.cat(train_in, 0)
            tgt = torch.cat(train_tgt, 0)
            train_in.clear(); train_tgt.clear()
            _train_result.clear(); _train_event.clear()

            def _do_train():
                _train_result['losses'] = train_model(ml_model, inp, tgt)
                _train_result['model']  = try_compile(ml_model)
                _train_event.set()

            threading.Thread(target=_do_train, daemon=True).start()
            phase = PHASE_TRAINING_WAIT

        elif phase == PHASE_TRAINING_WAIT:
            if _train_event.is_set():
                ml_model = _train_result['model']
                losses   = _train_result['losses']
                _async_cnn.start(ml_model)
                phase = PHASE_RUNNING
                print(f" [4/4] Neural SR active  [async CNN running]")
                print(f"       loss {losses[0]:.5f} -> {losses[-1]:.5f}")

        # -- Render ----------------------------------------------------
        if phase == PHASE_RUNNING:
            render_neural_async()
        else:
            render_preview()

        gui.set_image(pixels)
        gui.show()

        # -- FPS -------------------------------------------------------
        frame  += 1
        t_now   = time.perf_counter()
        fps_acc += 1.0 / max(t_now - t_prev, 1e-9)
        fps_n   += 1
        t_prev   = t_now

        if frame % 60 == 0 and phase not in (PHASE_TRAINING, PHASE_TRAINING_WAIT):
            avg = fps_acc / fps_n
            fps_acc = fps_n = 0
            tag = {PHASE_WARMUP:     "warmup",
                   PHASE_COLLECTING: f"collecting {len(train_in)}/{ML_COLLECT}",
                   PHASE_RUNNING:    "neural-SR [async]"}.get(phase, "?")
            print(f"  Frame {frame:5d}  {avg:5.1f} FPS  [{tag}]")

    _async_cnn.stop()
    gui.close()
    print("Done.")


if __name__ == "__main__":
    main()

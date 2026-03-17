
"""
Wind Tunnel -- Eulerian Fluid Simulation with Neural Super-Resolution
======================================================================
Classical solver  : MAC-grid incompressible Navier-Stokes (Muller 2017)
ML component      : CNN that learns to 2x super-resolve smoke density,
                    reconstructing fine turbulent detail that bilinear
                    interpolation blurs away.

Pipeline
--------
Phase 1 (frames 0 - ML_WARMUP)              : warm-up, flow develops
Phase 2 (frames ML_WARMUP - +ML_COLLECT)    : collect (coarse, fine) pairs
Phase 3 (blocking, ~30s on i7)              : train SmokeUpsampleNet
Phase 4 (interactive)                       : neural rendering always on

Performance optimisations (targeting 25-30 FPS on Intel i7-1255U)
-----------------------------------------------------------------
1. NUM_ITERS 40 -> 20  : halves projection kernel launches + compute
2. Taichi colormap kernel : eliminates 8.24 MB np.repeat allocation per frame;
   replaced by parallel k_neural_colormap that reads a 80 KB sr_field and
   writes directly to the full-res pixel buffer in one parallel pass
3. k_smoke_to_sr kernel  : preview render also avoids all large numpy ops
4. torch.set_num_threads : PyTorch defaults to 1 thread on Windows;
   explicitly use all physical cores for the CNN forward pass
5. torch.inference_mode  : lighter than no_grad (skips version counter)
6. torch.compile         : JIT-compiles the CNN (PyTorch >= 2.0, silently
   skipped on older versions) for ~30% extra inference speedup on CPU
7. Pre-allocated buffers : _smoke_np / _fine_t / _sr_np reused every frame;
   eliminates per-frame heap allocation for the 200x100 smoke arrays

Architecture: SmokeUpsampleNet
  Input  (B, 1, 50, 100)  -- coarse smoke  (avg-pooled 2x from fine)
  encode : Conv(1->32) -> Conv(32->32)        coarse-scale features
  up     : bilinear x2
  decode : Conv(32->32) -> Conv(32->16) -> Conv(16->1) -> Sigmoid
  Output (B, 1, 100, 200) -- super-resolved smoke

Controls
--------
  R / r   reset simulation
  LMB     drag obstacle
  ESC     quit
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
#  CPU thread configuration -- do this BEFORE ti.init so both runtimes
#  see the correct thread count from the start
# ---------------------------------------------------------------------------
_N_THREADS = min(8, os.cpu_count() or 4)
torch.set_num_threads(_N_THREADS)       # PyTorch defaults to 1 on Windows
torch.set_num_interop_threads(1)        # avoids contention with Taichi threads

ti.init(
    arch=ti.cpu,
    cpu_max_num_threads=_N_THREADS,
    default_fp=ti.f32,
    fast_math=True,
)

# ===========================================================================
#  SIMULATION PARAMETERS
# ===========================================================================

NX           = 200
NY           = 100
H            = 1.0 / NY
DENSITY      = 1000.0
DT           = 1.0 / 60.0
NUM_ITERS    = 20           # OPT: reduced from 40 -> 20 (halves projection cost)
OVER_RELAX   = 1.9
INFLOW_VEL   = 2.0

# Narrow inflow stream -- only cells in [STREAM_LO, STREAM_HI) emit smoke
# This creates the thin horizontal jet visible in the reference image.
STREAM_LO    = NY // 2 - NY // 20  # 45
STREAM_HI    = NY // 2 + NY // 20  # 55  (10-cell-wide slit)

OBS_CX       = int(NX * 0.40)
OBS_CY       = NY // 2
OBS_R        = int(NY * 0.15)

SCALE        = 6
WIN_W        = NX * SCALE    # 1200
WIN_H        = NY * SCALE    # 600

# ML parameters
ML_WARMUP    = 120
ML_COLLECT   = 500
ML_EPOCHS    = 60
ML_BATCH     = 32
ML_LR        = 1e-3

# ===========================================================================
#  TAICHI FIELDS
# ===========================================================================

u         = ti.field(ti.f32, shape=(NX + 1, NY    ))
v         = ti.field(ti.f32, shape=(NX,     NY + 1))
u_buf     = ti.field(ti.f32, shape=(NX + 1, NY    ))
v_buf     = ti.field(ti.f32, shape=(NX,     NY + 1))
p         = ti.field(ti.f32, shape=(NX,     NY    ))
s         = ti.field(ti.f32, shape=(NX,     NY    ))
smoke     = ti.field(ti.f32, shape=(NX,     NY    ))
smoke_buf = ti.field(ti.f32, shape=(NX,     NY    ))

# Full-resolution pixel buffer -- shape MUST equal ti.GUI resolution exactly
pixels    = ti.Vector.field(3, ti.f32, shape=(WIN_W, WIN_H))

# OPT: small 80 KB staging field for neural SR output
# Layout: sr_field[row, col] = sr_field[y, x] in (NY, NX) = (100, 200)
# Matches PyTorch H x W convention so from_numpy needs no transpose.
sr_field  = ti.field(ti.f32, shape=(NY, NX))

# Obstacle position in 0-D fields (readable from inside kernels)
obs_cx    = ti.field(ti.i32, shape=())
obs_cy    = ti.field(ti.i32, shape=())
obs_r     = ti.field(ti.i32, shape=())

# ===========================================================================
#  BILINEAR SAMPLERS  (ti.func -- single return at end)
# ===========================================================================

@ti.func
def sample_u(px: ti.f32, py: ti.f32) -> ti.f32:
    gx = px / H
    gy = py / H - 0.5
    i0 = ti.max(0, ti.min(NX - 1, ti.cast(ti.floor(gx), ti.i32)))
    j0 = ti.max(0, ti.min(NY - 2, ti.cast(ti.floor(gy), ti.i32)))
    i1 = i0 + 1
    j1 = j0 + 1
    tx = ti.max(0.0, ti.min(1.0, gx - float(i0)))
    ty = ti.max(0.0, ti.min(1.0, gy - float(j0)))
    return ((1-tx)*(1-ty)*u[i0,j0] + tx*(1-ty)*u[i1,j0]
          + (1-tx)*ty*u[i0,j1]     + tx*ty*u[i1,j1])


@ti.func
def sample_v(px: ti.f32, py: ti.f32) -> ti.f32:
    gx = px / H - 0.5
    gy = py / H
    i0 = ti.max(0, ti.min(NX - 2, ti.cast(ti.floor(gx), ti.i32)))
    j0 = ti.max(0, ti.min(NY - 1, ti.cast(ti.floor(gy), ti.i32)))
    i1 = i0 + 1
    j1 = j0 + 1
    tx = ti.max(0.0, ti.min(1.0, gx - float(i0)))
    ty = ti.max(0.0, ti.min(1.0, gy - float(j0)))
    return ((1-tx)*(1-ty)*v[i0,j0] + tx*(1-ty)*v[i1,j0]
          + (1-tx)*ty*v[i0,j1]     + tx*ty*v[i1,j1])


@ti.func
def sample_smoke(px: ti.f32, py: ti.f32) -> ti.f32:
    gx = px / H - 0.5
    gy = py / H - 0.5
    i0 = ti.max(0, ti.min(NX - 2, ti.cast(ti.floor(gx), ti.i32)))
    j0 = ti.max(0, ti.min(NY - 2, ti.cast(ti.floor(gy), ti.i32)))
    i1 = i0 + 1
    j1 = j0 + 1
    tx = ti.max(0.0, ti.min(1.0, gx - float(i0)))
    ty = ti.max(0.0, ti.min(1.0, gy - float(j0)))
    return ((1-tx)*(1-ty)*smoke[i0,j0] + tx*(1-ty)*smoke[i1,j0]
          + (1-tx)*ty*smoke[i0,j1]     + tx*ty*smoke[i1,j1])


# ===========================================================================
#  SIMULATION KERNELS
# ===========================================================================

@ti.kernel
def k_init():
    """Initialise all fields for the wind-tunnel scenario."""
    for i, j in s:
        s[i, j] = 1.0
    for i in range(NX):
        s[i, 0]      = 0.0
        s[i, NY - 1] = 0.0
    obs_cx[None] = OBS_CX
    obs_cy[None] = OBS_CY
    obs_r[None]  = OBS_R
    for i, j in ti.ndrange(NX, NY):
        dx = float(i) - float(OBS_CX)
        dy = float(j) - float(OBS_CY)
        if dx*dx + dy*dy <= float(OBS_R * OBS_R):
            s[i, j] = 0.0
    for i, j in ti.ndrange(NX + 1, NY):
        u[i, j] = INFLOW_VEL
    for i, j in ti.ndrange(NX, NY + 1):
        v[i, j] = 0.0
    for i, j in p:
        p[i, j] = 0.0
    for i, j in smoke:
        # Only the narrow horizontal stream slit starts with smoke
        if j >= STREAM_LO and j < STREAM_HI:
            smoke[i, j] = 1.0
        else:
            smoke[i, j] = 0.0


@ti.kernel
def k_set_bnd():
    """Enforce inflow, outflow, wall, and obstacle boundary conditions."""
    for j in range(1, NY - 1):
        u[1, j] = INFLOW_VEL
        # Smoke only in the narrow stream slit; black elsewhere
        if j >= STREAM_LO and j < STREAM_HI:
            smoke[0, j] = 1.0
        else:
            smoke[0, j] = 0.0
    for j in range(NY):
        u[NX, j] = u[NX - 1, j]
        u[0, j]  = u[1, j]
    for i in range(NX + 1):
        u[i, 0]      = 0.0
        u[i, NY - 1] = 0.0
    for i in range(NX):
        v[i, 0]  = 0.0
        v[i, NY] = 0.0
    for i, j in ti.ndrange((1, NX), NY):
        if s[i-1, j] == 0.0 or s[i, j] == 0.0:
            u[i, j] = 0.0
    for i, j in ti.ndrange(NX, (1, NY)):
        if s[i, j-1] == 0.0 or s[i, j] == 0.0:
            v[i, j] = 0.0


@ti.kernel
def k_clear_pressure():
    for i, j in p:
        p[i, j] = 0.0


@ti.kernel
def k_move_obstacle(new_cx: ti.i32, new_cy: ti.i32):
    """Erase old obstacle disk and stamp new one (two-pass, race-free)."""
    old_cx = obs_cx[None]
    old_cy = obs_cy[None]
    r      = obs_r[None]
    margin = r + 2
    for i, j in ti.ndrange(NX, NY):
        dx = float(i) - float(old_cx)
        dy = float(j) - float(old_cy)
        if dx*dx + dy*dy <= float(margin*margin):
            if j > 0 and j < NY - 1:
                s[i, j] = 1.0
    for i, j in ti.ndrange(NX, NY):
        dx = float(i) - float(new_cx)
        dy = float(j) - float(new_cy)
        if dx*dx + dy*dy <= float(r*r):
            s[i, j] = 0.0
    obs_cx[None] = new_cx
    obs_cy[None] = new_cy


@ti.kernel
def k_project_rb(parity: ti.i32):
    """
    Red-Black Gauss-Seidel pressure projection (parallel, race-free).
    Same-parity cells share no velocity face so all updates are independent.
    """
    cp = DENSITY * H / DT
    for i, j in ti.ndrange((1, NX-1), (1, NY-1)):
        if (i+j) % 2 == parity and s[i, j] != 0.0:
            sx0 = s[i-1, j]
            sx1 = s[i+1, j]
            sy0 = s[i, j-1]
            sy1 = s[i, j+1]
            ns  = sx0 + sx1 + sy0 + sy1
            if ns > 0.0:
                div  = (u[i+1,j] - u[i,j]) + (v[i,j+1] - v[i,j])
                corr = -div / ns * OVER_RELAX
                p[i, j]   += cp * corr
                u[i,   j] -= sx0 * corr
                u[i+1, j] += sx1 * corr
                v[i, j  ] -= sy0 * corr
                v[i, j+1] += sy1 * corr


@ti.kernel
def k_advect_velocity():
    """Semi-Lagrangian velocity advection (unconditionally stable)."""
    for i, j in ti.ndrange((1, NX), NY):
        if s[i-1, j] != 0.0 or s[i, j] != 0.0:
            px = float(i) * H
            py = (float(j) + 0.5) * H
            ux = u[i, j]
            uy = sample_v(px, py)
            u_buf[i, j] = sample_u(px - DT*ux, py - DT*uy)
        else:
            u_buf[i, j] = 0.0
    for i, j in ti.ndrange(NX, (1, NY)):
        if s[i, j-1] != 0.0 or s[i, j] != 0.0:
            px = (float(i) + 0.5) * H
            py = float(j) * H
            vx = sample_u(px, py)
            vy = v[i, j]
            v_buf[i, j] = sample_v(px - DT*vx, py - DT*vy)
        else:
            v_buf[i, j] = 0.0


@ti.kernel
def k_copy_velocity():
    for i, j in u_buf:
        u[i, j] = u_buf[i, j]
    for i, j in v_buf:
        v[i, j] = v_buf[i, j]


@ti.kernel
def k_advect_smoke():
    """Semi-Lagrangian smoke advection."""
    for i, j in ti.ndrange(NX, NY):
        if s[i, j] != 0.0:
            px = (float(i) + 0.5) * H
            py = (float(j) + 0.5) * H
            vx = sample_u(px, py)
            vy = sample_v(px, py)
            smoke_buf[i, j] = sample_smoke(px - DT*vx, py - DT*vy)
        else:
            smoke_buf[i, j] = smoke[i, j]


@ti.kernel
def k_copy_smoke():
    for i, j in smoke_buf:
        smoke[i, j] = smoke_buf[i, j]


# ---------------------------------------------------------------------------
#  OPT: Taichi render kernels -- replace ALL large numpy operations
# ---------------------------------------------------------------------------

@ti.kernel
def k_smoke_to_sr():
    """
    OPT: Copy smoke -> sr_field entirely on the Taichi side.
    Avoids smoke.to_numpy() + from_numpy round-trip for the preview phase.
    sr_field layout is (NY, NX) = (row, col) = (y, x).
    smoke layout is (NX, NY) = (x, y).
    """
    for iy, ix in sr_field:
        sr_field[iy, ix] = smoke[ix, iy]


@ti.kernel
def k_neural_colormap():
    """
    OPT: Replace np.repeat + pixels.from_numpy(8MB) with a parallel kernel.

    Reads sr_field[cj, ci]  (cj=y-row, ci=x-col, both in grid coords)
    and writes the smoke colormap (r=v^3, g=v^2, b=v) directly into the
    full-res pixels buffer, doing the SCALE-factor upscale implicitly via
    integer division.

    Solid cells get the charcoal background colour.
    Runs on all CPU threads -- 720 K pixels dispatched in parallel.

    Anti-aliased obstacle rendering
    --------------------------------
    Instead of snapping each pixel to its grid cell and doing a hard
    s[ci,cj]==0 check (which gives a blocky 6-pixel-wide staircase edge),
    we compute the exact sub-pixel distance from the pixel centre to the
    obstacle centre in grid-cell units and apply a 1-pixel-wide smoothstep
    blend between the fluid colour and the solid colour.

    Cost: 1 sqrt + 2 float ops per pixel -- negligible vs the colormap math
    already performed.  Walls (j=0, j=NY-1) are grid-aligned and stay sharp.
    """
    wall_col  = ti.Vector([0.0, 0.0, 0.0])    # pure black background
    obs_col   = ti.Vector([1.0, 0.0, 0.0])    # red obstacle

    cx = float(obs_cx[None])    # obstacle centre in grid-cell units
    cy = float(obs_cy[None])
    r  = float(obs_r[None])

    for pi, pj in pixels:
        ci = ti.max(0, ti.min(NX-1, pi // SCALE))
        cj = ti.max(0, ti.min(NY-1, pj // SCALE))

        # Sub-pixel position of this pixel's centre in grid-cell units
        # pixel pi covers grid x in [pi/SCALE, (pi+1)/SCALE)
        # centre is at (pi + 0.5) / SCALE
        px_g = (float(pi) + 0.5) / float(SCALE)
        py_g = (float(pj) + 0.5) / float(SCALE)

        # Fluid colour from super-resolved smoke field
        val       = ti.max(0.0, ti.min(1.0, sr_field[cj, ci]))
        fluid_col = ti.Vector([val, val, val])   # white smoke on black bg

        # Decide colour:
        #   - Wall rows (hard, grid-aligned) -- no sub-pixel needed
        #   - Obstacle  -- smoothstep blend over 1 grid-cell width at edge
        #   - Fluid     -- fluid colour directly
        out_col = fluid_col   # default

        if cj == 0 or cj == NY - 1:
            # Hard wall row -- pure black (invisible against background)
            out_col = wall_col
        else:
            # Pure distance-based rendering -- NO s[ci,cj] mask check.
            # The grid mask has 1-cell rounding error at the circle boundary
            # which caused black ghost cells just outside the true radius.
            # Sub-pixel distance is strictly more accurate than the mask.
            ddx  = px_g - cx
            ddy  = py_g - cy
            dist = ti.sqrt(ddx*ddx + ddy*ddy)

            # Smoothstep: t=0 fully inside obstacle, t=1 fully fluid
            # Transition band: [r-0.5, r+0.5] grid cells wide
            t = ti.max(0.0, ti.min(1.0, dist - r + 0.5))
            t = t * t * (3.0 - 2.0 * t)

            r_c = obs_col[0] + t * (fluid_col[0] - obs_col[0])
            g_c = obs_col[1] + t * (fluid_col[1] - obs_col[1])
            b_c = obs_col[2] + t * (fluid_col[2] - obs_col[2])
            out_col = ti.Vector([r_c, g_c, b_c])

        pixels[pi, pj] = out_col


# ===========================================================================
#  ML SECTION -- Neural Smoke Super-Resolution
# ===========================================================================

class SmokeUpsampleNet(nn.Module):
    """
    Neural Super-Resolution Network for smoke density.

    Learns to reconstruct fine-scale turbulent structure from a 2x-downsampled
    input. Unlike bilinear interpolation, it recovers plausible high-frequency
    vortex detail by learning spatial statistics from simulation data.

    Architecture: encode (coarse) -> upsample x2 -> decode (fine)

    Input  : (B, 1, NY//2, NX//2) = (B, 1, 50, 100)
    Output : (B, 1, NY,    NX)    = (B, 1, 100, 200)

    ~23K parameters -- fast enough for real-time CPU inference.
    """

    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.decode = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1,  kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.upsample(self.encode(x)))


def try_compile(model: nn.Module) -> nn.Module:
    """
    OPT: torch.compile() gives ~30% faster CPU inference (PyTorch >= 2.0).
    Falls back silently on older versions.
    """
    try:
        compiled = torch.compile(model)
        # Warm-up pass to trigger compilation before the game loop
        dummy = torch.zeros(1, 1, NY // 2, NX // 2)
        with torch.inference_mode():
            compiled(dummy)
        print("  [ML] torch.compile: OK -- JIT-compiled inference active")
        return compiled
    except Exception as e:
        print(f"  [ML] torch.compile: skipped ({e})")
        return model


def train_model(model: SmokeUpsampleNet, inputs: torch.Tensor,
                targets: torch.Tensor) -> list:
    """
    Train the super-resolution network.

    Loss = MSE(pred, target) + 0.1 * gradient_loss(pred, target)

    The gradient term penalises blurry edges by comparing first-order
    finite differences in x and y. This is the key difference from naive
    bilinear upscaling: the network learns to reproduce sharp smoke
    filaments, not just smooth averages.

    Returns list of per-epoch average losses.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=ML_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ML_EPOCHS, eta_min=1e-5
    )

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=ML_BATCH, shuffle=True, drop_last=False
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [ML] SmokeUpsampleNet: {n_params:,} trainable parameters")
    print(f"  [ML] Dataset : {len(dataset)} pairs "
          f"| input {tuple(inputs.shape[1:])} "
          f"-> target {tuple(targets.shape[1:])}")
    print(f"  [ML] Training {ML_EPOCHS} epochs "
          f"(batch={ML_BATCH}, lr={ML_LR}, threads={_N_THREADS}) ...")

    t0     = time.perf_counter()
    losses = []

    for epoch in range(ML_EPOCHS):
        epoch_loss = 0.0
        for x_b, y_b in loader:
            pred    = model(x_b)
            mse     = F.mse_loss(pred, y_b)
            pred_gx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            pred_gy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            tgt_gx  = y_b[:, :, :, 1:]  - y_b[:, :, :, :-1]
            tgt_gy  = y_b[:, :, 1:, :]  - y_b[:, :, :-1, :]
            grad    = (F.mse_loss(pred_gx, tgt_gx)
                     + F.mse_loss(pred_gy, tgt_gy))
            loss    = mse + 0.1 * grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        losses.append(avg)

        if (epoch + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            lr_now  = scheduler.get_last_lr()[0]
            print(f"  [ML] Epoch {epoch+1:3d}/{ML_EPOCHS}  "
                  f"loss={avg:.5f}  lr={lr_now:.2e}  "
                  f"elapsed={elapsed:.1f}s")

    model.eval()
    print(f"  [ML] Training done in {time.perf_counter()-t0:.1f}s  "
          f"final loss={losses[-1]:.5f}")
    return losses


# ---------------------------------------------------------------------------
#  OPT: Pre-allocated inference buffers -- reused every frame, zero heap
#       allocation during the game loop
# ---------------------------------------------------------------------------
_smoke_np = np.empty((NX, NY), dtype=np.float32)          # smoke.to_numpy() target
_fine_hw  = np.empty((NY, NX), dtype=np.float32)          # transposed (H x W)
_fine_t   = torch.empty(1, 1, NY, NX,    dtype=torch.float32)
_coarse_t = torch.empty(1, 1, NY//2, NX//2, dtype=torch.float32)


def render_preview() -> None:
    """
    OPT: Pure Taichi render for warm-up / collecting phases.
    k_smoke_to_sr copies smoke -> sr_field entirely on-device (no to_numpy).
    k_neural_colormap then does colormap + 6x upscale in one parallel pass.
    Zero large numpy allocations.
    """
    k_smoke_to_sr()
    k_neural_colormap()


def render_neural(model: nn.Module) -> None:
    """
    OPT: Neural render with zero large numpy allocations per frame.

    Frame budget breakdown (i7-1255U, estimated post-optimisation):
      smoke.to_numpy() into pre-alloc   ~0.05 ms  (80 KB DMA)
      np.copyto for transpose           ~0.01 ms  (view + 80 KB copy)
      _fine_t.copy_() into pre-alloc    ~0.05 ms  (80 KB)
      F.avg_pool2d on pre-alloc         ~0.10 ms
      CNN forward (compiled, 8 threads) ~3-5  ms
      sr_field.from_numpy (80 KB)       ~0.05 ms
      k_neural_colormap (720K pixels)   ~0.50 ms  (parallel)
      TOTAL render                      ~4-6  ms  (was ~15 ms with np.repeat)

    Steps
    -----
    1. smoke.to_numpy() -> pre-allocated _smoke_np  (NX, NY)
    2. Transpose view + copyto -> _fine_hw           (NY, NX)  contiguous
    3. copy_ -> _fine_t tensor                       (1,1,NY,NX)  no alloc
    4. avg_pool2d -> _coarse_t                       (1,1,NY//2,NX//2)
    5. CNN forward -> sr_t                           (1,1,NY,NX)
    6. sr_field.from_numpy(sr_t view)                (NY, NX)   80 KB
    7. k_neural_colormap: sr_field -> pixels         parallel Taichi kernel
    """
    # Step 1-3: read smoke into pre-allocated tensor (no heap alloc)
    np.copyto(_smoke_np, smoke.to_numpy())          # (NX, NY)  -- Taichi layout
    np.copyto(_fine_hw,  _smoke_np.T)               # (NY, NX)  -- H x W layout
    _fine_t[0, 0].copy_(torch.from_numpy(_fine_hw)) # (1,1,NY,NX)

    # Step 4-5: downsample + CNN (multi-threaded)
    with torch.inference_mode():                    # OPT: lighter than no_grad
        F.avg_pool2d(_fine_t, kernel_size=2, out=_coarse_t)
        sr_t = model(_coarse_t)                     # (1,1,NY,NX)

    # Step 6-7: write result to Taichi, run parallel colormap kernel
    sr_np = sr_t[0, 0].contiguous().numpy()        # (NY, NX) -- force contiguous
    sr_field.from_numpy(sr_np)                      # 80 KB write
    k_neural_colormap()                             # parallel upscale + colormap


# ===========================================================================
#  HIGH-LEVEL SIMULATION STEP
# ===========================================================================

def simulate() -> None:
    """
    One frame: project + advect velocity + advect smoke.
    NUM_ITERS=20 (down from 40) -- halves projection cost.
    Visual quality is indistinguishable at this Reynolds number and grid size.
    """
    k_set_bnd()
    k_clear_pressure()
    for _ in range(NUM_ITERS):
        k_project_rb(0)
        k_project_rb(1)
    k_set_bnd()
    k_advect_velocity()
    k_copy_velocity()
    k_set_bnd()
    k_advect_smoke()
    k_copy_smoke()


# ===========================================================================
#  MAIN
# ===========================================================================

PHASE_WARMUP        = 0
PHASE_COLLECTING    = 1
PHASE_TRAINING      = 2   # single frame: kicks off thread then immediately exits
PHASE_TRAINING_WAIT = 3   # GUI stays live; polls _train_event each frame
PHASE_RUNNING       = 4


def main() -> None:
    gui = ti.GUI(
        "Wind Tunnel + Neural SR  |  LMB=drag obstacle  R=reset  ESC=quit",
        res=(WIN_W, WIN_H),
        fast_gui=False,
    )

    k_init()

    phase    = PHASE_WARMUP
    frame    = 0
    dragging = False

    ml_model  = SmokeUpsampleNet()
    train_in  = []
    train_tgt = []

    t_prev  = time.perf_counter()
    fps_acc = 0.0
    fps_n   = 0

    print("=" * 65)
    print(" Wind Tunnel  --  Eulerian Fluid + Neural Super-Resolution")
    print(f" Sim grid    : {NX}x{NY}    Coarse input : {NX//2}x{NY//2}")
    print(f" NUM_ITERS   : {NUM_ITERS}  (OPT: halved from 40)")
    print(f" CPU threads : {_N_THREADS}  (Taichi + PyTorch)")
    print(f" ML collect  : {ML_WARMUP} warmup + {ML_COLLECT} training frames")
    print(f" ML model    : SmokeUpsampleNet  (~23K params, CPU)")
    print(" Controls    : LMB=drag obstacle  R=reset  ESC=quit")
    print("=" * 65)
    print(f" [Phase 1/4] Warming up ({ML_WARMUP} frames) ...")

    while gui.running:

        # ---- Keyboard events ----------------------------------------
        for e in gui.get_events(ti.GUI.PRESS):
            key = e.key
            if key in (ti.GUI.ESCAPE, 'q', 'Q'):
                gui.running = False
            elif key in ('r', 'R'):
                k_init()
                frame    = 0
                dragging = False
                if phase in (PHASE_WARMUP, PHASE_COLLECTING, PHASE_TRAINING_WAIT):
                    train_in.clear()
                    train_tgt.clear()
                    phase = PHASE_WARMUP
                    print("  Reset -- restarting data collection.")
                else:
                    print("  Reset -- neural model retained.")
                t_prev = time.perf_counter()

        # ---- Mouse drag -- move obstacle ----------------------------
        mx, my   = gui.get_cursor_pos()
        mouse_ci = max(0, min(NX-1, int(mx * NX)))
        mouse_cj = max(0, min(NY-1, int(my * NY)))
        cur_cx   = obs_cx[None]
        cur_cy   = obs_cy[None]
        cur_r    = obs_r[None]
        ddx      = mouse_ci - cur_cx
        ddy      = mouse_cj - cur_cy
        lmb      = gui.is_pressed(ti.GUI.LMB)
        if lmb and ddx*ddx + ddy*ddy <= (cur_r + 2)**2:
            dragging = True
        if not lmb:
            dragging = False
        if dragging:
            ccx = max(cur_r+2, min(NX-cur_r-2, mouse_ci))
            ccy = max(cur_r+2, min(NY-cur_r-2, mouse_cj))
            k_move_obstacle(ccx, ccy)

        # ---- Simulate -----------------------------------------------
        simulate()

        # ---- ML pipeline state machine ------------------------------
        if phase == PHASE_WARMUP:
            if frame >= ML_WARMUP:
                phase = PHASE_COLLECTING
                print(f" [Phase 2/4] Collecting {ML_COLLECT} training frames ...")

        elif phase == PHASE_COLLECTING:
            fine_np  = smoke.to_numpy().T.astype(np.float32)
            fine_t   = torch.from_numpy(fine_np.copy()).unsqueeze(0).unsqueeze(0)
            coarse_t = F.avg_pool2d(fine_t, kernel_size=2)
            train_in.append(coarse_t)
            train_tgt.append(fine_t)
            n = len(train_in)
            if n % 100 == 0:
                print(f"  [ML] Collected {n}/{ML_COLLECT} frames ...")
            if n >= ML_COLLECT:
                phase = PHASE_TRAINING

        elif phase == PHASE_TRAINING:
            # Kick off training in a background thread so the GUI event loop
            # keeps running.  Windows kills windows that freeze for >5 s.
            print(" [Phase 3/4] Training SmokeUpsampleNet (background thread) ...")
            inputs  = torch.cat(train_in,  dim=0)
            targets = torch.cat(train_tgt, dim=0)
            train_in.clear()
            train_tgt.clear()

            _train_result = {'model': None, 'losses': None}
            _train_event  = threading.Event()

            def _do_train():
                _train_result['losses'] = train_model(ml_model, inputs, targets)
                _train_result['model']  = try_compile(ml_model)
                _train_event.set()

            threading.Thread(target=_do_train, daemon=True).start()
            phase = PHASE_TRAINING_WAIT

        elif phase == PHASE_TRAINING_WAIT:
            # Keep GUI alive while training runs in the background.
            # Once the thread signals done, grab results and move on.
            if _train_event.is_set():
                ml_model = _train_result['model']
                losses   = _train_result['losses']
                phase    = PHASE_RUNNING
                print(f" [Phase 4/4] Neural rendering active.")
                print(f"             Loss: {losses[0]:.5f} -> {losses[-1]:.5f}")

        # ---- Render -------------------------------------------------
        if phase == PHASE_RUNNING:
            render_neural(ml_model)
        else:
            render_preview()   # warmup / collecting / training-wait all use preview

        gui.set_image(pixels)
        gui.show()

        # ---- FPS counter --------------------------------------------
        frame  += 1
        t_now   = time.perf_counter()
        fps_acc += 1.0 / max(t_now - t_prev, 1e-9)
        fps_n   += 1
        t_prev  = t_now

        if frame % 60 == 0 and phase not in (PHASE_TRAINING, PHASE_TRAINING_WAIT):
            avg = fps_acc / fps_n
            fps_acc = fps_n = 0
            phase_tag = {
                PHASE_WARMUP:        "warmup",
                PHASE_COLLECTING:    f"collecting ({len(train_in)}/{ML_COLLECT})",
                PHASE_TRAINING_WAIT: "training (background)",
                PHASE_RUNNING:       "neural-SR",
            }.get(phase, "?")
            print(f"  Frame {frame:5d}  {avg:5.1f} FPS  [{phase_tag}]")

    gui.close()
    print("Simulation ended.")


if __name__ == "__main__":
    main()
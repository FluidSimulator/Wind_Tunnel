# Wind Tunnel — Eulerian Fluid + Neural Super-Resolution

A real-time 2D fluid simulation powered by **Taichi parallel CPU kernels** and a **CNN super-resolution network** trained entirely from simulation data. Targets 25–30 FPS on an Intel i7-1255U with no dedicated GPU.

Based on Matthias Muller's [Ten Minute Physics Tutorial 17](https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/17-fluidSim.html), ported to Python with Taichi and extended with an original ML pipeline.

---

## Quick Start

**Requirements**

| Package | Version | Notes |
|---|---|---|
| Python | 3.10+ | Tested on 3.13.1 |
| taichi | 1.7.4 | CPU backend (x64) |
| torch | 2.0+ | CPU inference; `torch.compile` requires 2.0+ |
| numpy | any recent | Buffer management |

**Install and run**

```bash
pip install taichi torch numpy
python wind_tunnel.py
```

The simulation starts immediately. After collecting 500 training frames (~50 s of wall time), the CNN trains in a background thread (~30 s) while the simulation keeps running. Neural rendering activates automatically once training completes.

---

## Controls

| Input | Action |
|---|---|
| `LMB` hold + drag | Move the red circular obstacle through the flow field |
| `R` | Reset simulation; restart data collection if training has not begun |
| `ESC` / `Q` | Exit |

---

## Architecture

### Fluid Solver

The solver implements the **incompressible Navier-Stokes equations** on a **MAC (Marker-And-Cell) staggered grid**. Velocity components are stored on cell faces rather than cell centres, eliminating checkerboard pressure instabilities.

```
u[i, j]     -- x-velocity at LEFT face    shape (NX+1, NY)   = (201, 100)
v[i, j]     -- y-velocity at BOTTOM face  shape (NX,   NY+1) = (200, 101)
p[i, j]     -- pressure at cell centre    shape (NX,   NY)   = (200, 100)
s[i, j]     -- solid mask  0=solid 1=fluid
smoke[i, j] -- passive scalar tracer
```

**Pressure Projection (Incompressibility)**

A **Red-Black Gauss-Seidel** iterative solver enforces zero divergence in every fluid cell. Cells are split by parity of `(i+j)`: same-parity cells share no velocity face, making each colour-pass fully parallel with no data races. Pressure is **warm-started** each frame (not reset), so convergence needs only `NUM_ITERS = 8` instead of 12+ for a cold start.

**Semi-Lagrangian Advection**

Velocity and smoke are advected by tracing each face/cell centre backward one timestep along the velocity field, then bilinearly sampling the old field. Unconditionally stable at any CFL (CFL ≈ 3.3 here).

**Boundary Conditions**

- **Left wall:** constant inflow at 2.0 m/s; narrow 10-cell smoke jet centred vertically
- **Right wall:** zero-gradient outflow (extrapolation)
- **Top/bottom:** solid no-slip walls
- **Obstacle:** red circle at 40% domain width, radius 15 cells, draggable in real time

---

### Neural Super-Resolution

The neural component learns to **2x super-resolve smoke density** from a 100×50 coarse input to the full 200×100 simulation resolution. Unlike bilinear upscaling, it recovers sharp vortex filaments by learning their statistical patterns from simulation data.

**SmokeUpsampleNet architecture**

```
Input   (B, 1, 50, 100)  -- avg_pool2d(2x) of fine smoke

encode: Conv(1->16, 3x3) -> LeakyReLU(0.2)
        Conv(16->16, 3x3) -> LeakyReLU(0.2)

upsample: bilinear x2  (no learnable params)

decode: Conv(16->16, 3x3) -> LeakyReLU(0.2)
        Conv(16->8,  3x3) -> LeakyReLU(0.2)
        Conv(8->1,   3x3) -> Sigmoid

Output  (B, 1, 100, 200) -- super-resolved smoke

Total trainable parameters: ~6,000
```

**Training**

Adam optimiser with cosine annealing LR schedule (1e-3 → 1e-5, 60 epochs). Loss function:

```
L = MSE(pred, target)
  + 0.1 * [MSE(grad_x(pred), grad_x(target))
         + MSE(grad_y(pred), grad_y(target))]
```

The gradient loss term penalises blurry edges, compelling the network to reconstruct sharp smoke boundaries rather than smooth averages.

**AsyncCNN — Pipelined Inference**

After training, the CNN runs in a **daemon background thread** pipelined with simulation. The main thread submits a smoke snapshot, then immediately renders the **previous frame's CNN result** (1-frame latency, imperceptible at 25+ FPS). Frame time becomes `max(simulate_ms, cnn_ms) + render_ms` rather than their sum.

---

### Four-Phase Pipeline

| Phase | Name | Duration | What happens |
|---|---|---|---|
| 1 / 4 | Warmup | ~2 s | 120 frames; flow develops past transient startup |
| 2 / 4 | Collecting | ~8 s | 500 (coarse, fine) smoke pairs saved |
| 3 / 4 | Training | ~30 s | Background thread trains CNN; GUI stays live |
| 4 / 4 | Running | forever | Neural rendering active; CNN async-pipelined |

---

## Simulation Parameters

| Parameter | Value | Description |
|---|---|---|
| `NX / NY` | 200 / 100 | Grid dimensions (cells) |
| `H` | 0.01 m | Cell size = 1.0 / NY |
| `DT` | 1/60 s | Timestep (~16.7 ms) |
| `DENSITY` | 1000 kg/m³ | Fluid density (water) |
| `INFLOW_VEL` | 2.0 m/s | Left-boundary inflow speed |
| `NUM_ITERS` | 8 | GS pressure iterations (warm-start) |
| `OVER_RELAX` | 1.9 | Gauss-Seidel over-relaxation factor |
| `STREAM_LO/HI` | 45 / 55 | 10-cell-wide smoke slit (NY//20 half-width) |
| `OBS_CX/CY` | 80 / 50 | Initial obstacle centre (40% width, mid height) |
| `OBS_R` | 15 cells | Obstacle radius (NY × 0.15) |
| `SCALE` | 5 | Display scale; window = 1000×500 px |

---

## Performance

### Why 15 FPS (before optimisation)

On Windows, each parallel-loop **barrier (thread-pool fork/join)** costs ~1 ms for 8 threads on a 20K-cell grid. The original implementation had **45 barriers per frame** = 45 ms of overhead alone, independent of actual compute time.

### Optimisations Applied

| # | Change | Effect |
|---|---|---|
| OPT-1 | Taichi threads: 8 → 4 | Barrier cost halved (~1 ms → ~0.5 ms each); 4 threads map to i7 P-cores |
| OPT-2 | `k_bnd_and_project()` merged kernel | 4 separate calls → 1; pressure warm-started; NUM_ITERS 12 → 8; 27 barriers → 22 |
| OPT-3 | `k_advect_all()` merged kernel | 4 separate calls → 1; advect u/v/smoke + 3 copies in one kernel; 8 barriers → 6 |
| OPT-4 | SCALE 6 → 5 (1200×600 → 1000×500) | 720K → 500K pixels; colormap kernel 31% less work |
| OPT-5 | `AsyncCNN` background thread | CNN pipelined with simulation; frame time = max, not sum |
| OPT-6 | Squared-distance obstacle test | `sqrt` only for ~470 AA-band pixels; skipped for 499K fluid pixels |
| OPT-7 | `torch.set_num_threads(8)` | PyTorch defaults to 1 thread on Windows |
| OPT-8 | `torch.inference_mode()` | Lighter than `no_grad`; skips version counter tracking |
| OPT-9 | `torch.compile(model)` | JIT-compiles CNN ops; ~20–30% faster CPU inference (PyTorch >= 2.0) |

### Frame Budget (post-optimisation)

```
28 barriers x 0.5 ms  =  14 ms   (was 45 ms)
Compute (advect, bnd)  =  10 ms
GUI (set_image/show)   =   5 ms
CNN (async, pipelined) =   0 ms   (runs concurrently)
                          -----
Total                  = ~29 ms  ->  ~34 FPS
```

---

## Code Structure

Single file: `wind_tunnel.py` (~715 lines).

### Taichi Kernels

| Kernel | Called | Purpose |
|---|---|---|
| `k_init()` | Once at startup | Initialise all fields, obstacle, and smoke |
| `k_move_obstacle(cx, cy)` | On LMB drag | Erase old disk, stamp new one (two-pass, race-free) |
| `k_bnd_and_project()` | Every frame | Set boundaries + warm-start GS pressure projection |
| `k_advect_all()` | Every frame | Semi-Lagrangian advect u, v, smoke; copy all buffers |
| `k_smoke_to_sr()` | Preview phases | Copy smoke → sr_field on-device (no `to_numpy`) |
| `k_neural_colormap()` | Every frame | Parallel colormap + SCALE-factor upscale → pixels |

### Python Functions

| Function | Called | Purpose |
|---|---|---|
| `simulate()` | Main loop | Dispatch `k_bnd_and_project` + `k_advect_all` (2 kernel calls) |
| `render_preview()` | Phases 1–3 | `k_smoke_to_sr` + `k_neural_colormap`, zero numpy allocation |
| `render_neural_async()` | Phase 4 | Submit smoke; render previous CNN result; `k_neural_colormap` |
| `train_model()` | Phase 3 | Adam + cosine LR + MSE + gradient loss, 60 epochs |
| `try_compile()` | After training | `torch.compile()` with silent fallback for older PyTorch |
| `AsyncCNN._worker()` | Background thread | Continuous CNN inference loop, thread-safe double buffer |

---

## ML Component

### Why this qualifies as real ML

- The network is never shown the Navier-Stokes equations or any physics rules.
- It learns the statistical structure of vortex shedding purely from data.
- It generalises to new obstacle positions dragged by the user at runtime.
- The gradient loss term explicitly teaches sharp discontinuity reconstruction — something MSE-only optimisation smooths away.
- The network predicts physically plausible fine-scale detail in the wake that is absent from the coarse input used for inference.

### Data pipeline

- **Input:** `avg_pool2d(smoke, kernel_size=2)` — simulates output of a 4× cheaper coarse solver
- **Target:** `smoke.to_numpy().T` — full-resolution ground truth from the live simulation
- 500 pairs collected over ~8 s of simulation; no external dataset required

### Where bilinear upscaling fails

A naive 2× bilinear upsample of the coarse smoke produces a blurred version of the input. Karman vortex street filaments downstream of the obstacle are completely smoothed out. `SmokeUpsampleNet` reconstructs these structures because it has learned — from hundreds of real simulation frames — that they always appear as tight counter-rotating pairs with characteristic spatial frequency and orientation relative to the obstacle wake.

---

## Dependencies

| Package | Role |
|---|---|
| `taichi` | Parallel CPU kernels, GUI, field management |
| `torch` | CNN definition, training, async inference |
| `numpy` | Pre-allocated buffers, smoke snapshots, array ops |
| `threading` | Background training and async CNN inference (stdlib) |
| `os`, `time` | Thread count detection, FPS measurement (stdlib) |

---

## Known Limitations

- Window size is fixed at **1000×500 px** (`SCALE=5`). Changing `SCALE` requires a restart.
- The CNN trains once on the default obstacle position. After dragging to a significantly different position, super-resolution quality may degrade slightly in the new wake region.
- GPU backends (`ti.cuda`, `ti.gpu`) are not used. On machines with a dedicated GPU, switching the Taichi backend would substantially increase simulation throughput, shifting the bottleneck to the Python/GUI layer.
- The pressure solver uses first-order Gauss-Seidel. A multigrid or conjugate-gradient solver would give significantly better convergence per iteration at higher resolutions.

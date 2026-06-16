"""M3 FEASIBILITY SPIKE (throwaway-OK): couple a baroclinic vorticity source
into the v1.6 jupiter_vorticity 2D-turbulence solver and produce comparison
renders for a human go/no-go.

Pipeline:
  1. v1.6 jupiter_vorticity default render (no source) = baseline.
  2. Spin the validated 2-layer baroclinic CPU config to ~step 9000, extract
     LOWER-layer eddy relative vorticity (zonal mean removed).
  3. Resample (H,W) -> equirect grid (W_grid, W_grid//2); normalize to unit std.
  4. Inject as a STATIC per-step source at 3 gains; render each.
  5. Save PNGs + a grayscale dump of the resampled source.

Opt-in & reversible: uses Simulation.set_external_vorticity_source(); the
default path (never call it) is byte-identical to the unmodified solver.

Usage:  py -3 scripts/sw_m3_spike_coupling.py [RENDER_RES] [GRID_W] [SRC_STEPS]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim import shallow_water_ref as ref

OUT = Path("out/audit/m3/spike")
RENDER_RES = int(sys.argv[1]) if len(sys.argv) >= 2 else 2048
GRID_W = int(sys.argv[2]) if len(sys.argv) >= 3 else RENDER_RES
SRC_STEPS = int(sys.argv[3]) if len(sys.argv) >= 4 else 9000

# Baroclinic source config (the validated unstable config from the brief).
SRC_W, SRC_H = 192, 96
GP1, GP2, XI = 0.05, 0.3, 3.0

# Gains as fractions of the solver Coriolis scale f0 (coriolis_f0=3.0). The
# source is normalized to unit std, so gain == peak ~few * std in solver-q units.
GAINS = {"lowgain": 0.5, "midgain": 1.5, "highgain": 4.0}


def _u8(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def build_baroclinic_source(grid_w: int, grid_h: int) -> np.ndarray:
    """Spin the 2-layer baroclinic CPU solver, extract LOWER-layer eddy
    vorticity (zonal-mean removed), resample to (grid_h, grid_w), unit-std."""
    print(f"  building baroclinic source: {SRC_W}x{SRC_H}, {SRC_STEPS} steps "
          f"(gp1={GP1}, gp2={GP2}, xi={XI})")
    st = ref.baroclinic_test_state(
        W=SRC_W, H=SRC_H, unstable=True, seed=0,
        gp1=GP1, gp2=GP2, xi_unstable=XI,
        pert_amp_frac=1e-3, dt_safety=0.3, nu4=0.0,
    )
    t0 = time.perf_counter()
    for s in range(SRC_STEPS):
        ref.step_2layer(st)
        if (s + 1) % 1500 == 0:
            zc = ref.vorticity(st.u2, st.v2, st.g)          # corners (H+1,W)
            eddy = zc - zc.mean(axis=1, keepdims=True)
            f0 = 2.0 * st.omega * np.sin(np.radians(45.0))
            print(f"    step {s+1:5d}  lower-eddy zeta std={np.std(eddy):.3e}  "
                  f"Ro={np.std(eddy)/abs(f0):.3f}  ({time.perf_counter()-t0:.0f}s)")

    # LOWER-layer eddy relative vorticity at cell centers.
    zc = ref.vorticity(st.u2, st.v2, st.g)                   # (H+1, W) corners
    zcen = 0.5 * (zc[:-1] + zc[1:])                          # (H, W) centers
    eddy = zcen - zcen.mean(axis=1, keepdims=True)           # remove zonal mean
    f0 = 2.0 * st.omega * np.sin(np.radians(45.0))
    print(f"  source eddy zeta std = {np.std(eddy):.3e}  (Ro ~ {np.std(eddy)/abs(f0):.3f})")

    # Resample (SRC_H, SRC_W) -> (grid_h, grid_w). cv2 expects (W,H).
    resamp = cv2.resize(eddy.astype(np.float32), (grid_w, grid_h),
                        interpolation=cv2.INTER_LINEAR)
    # Normalize to unit std so `gain` is directly in solver-q units.
    std = float(np.std(resamp))
    if std > 0:
        resamp = resamp / std
    print(f"  resampled to {grid_w}x{grid_h}, normalized to unit std "
          f"(range [{resamp.min():.2f},{resamp.max():.2f}])")
    return resamp.astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    gpu = GpuContext.headless()
    gpu.make_current()
    print("GPU:", gpu.ctx.info.get("GL_RENDERER"))

    p = load_factory_preset("jupiter_vorticity")
    p = p.model_copy(update={
        "sim": p.sim.model_copy(update={"resolution": GRID_W}),
    })
    grid_w, grid_h = GRID_W, GRID_W // 2
    print(f"preset jupiter_vorticity  grid={grid_w}x{grid_h}  dev_steps={p.sim.dev_steps}  "
          f"coriolis_f0={p.solver.coriolis_f0}  render={RENDER_RES}")

    # ---- baseline (no source) ----
    print("\n=== v1.6 default (no source) ===")
    sim = Simulation(p, gpu)
    base = sim.render_maps(RENDER_RES)["color"]
    base_rgb = np.clip(base[..., :3], 0, 1).astype(np.float32)
    base_path = (OUT / "spike_v16_default.png").resolve()
    cv2.imwrite(str(base_path), _u8(base_rgb))
    print(f"  wrote {base_path}")
    sim._release_sim()

    # ---- baroclinic source field ----
    print("\n=== baroclinic source ===")
    src = build_baroclinic_source(grid_w, grid_h)
    # Grayscale dump (symmetric about 0).
    a = float(np.abs(src).max()) or 1.0
    gray = ((src / a) * 0.5 + 0.5)
    src_path = (OUT / "spike_source.png").resolve()
    cv2.imwrite(str(src_path), (np.clip(gray, 0, 1) * 255).astype(np.uint8))
    print(f"  wrote {src_path}")

    # ---- coupled renders at 3 gains ----
    for name, gain in GAINS.items():
        print(f"\n=== coupled {name} (gain={gain}) ===")
        sim = Simulation(p, gpu)
        sim.set_external_vorticity_source(src, gain=gain)
        rgb = sim.render_maps(RENDER_RES)["color"]
        rgb = np.clip(rgb[..., :3], 0, 1).astype(np.float32)
        path = (OUT / f"spike_coupled_{name}.png").resolve()
        cv2.imwrite(str(path), _u8(rgb))
        d = float(np.abs(rgb - base_rgb).mean())
        print(f"  wrote {path}  (mean abs-diff vs baseline = {d:.4f})")
        sim._release_sim()

    print(f"\nDONE  {time.perf_counter()-wall0:.0f}s total")


if __name__ == "__main__":
    main()

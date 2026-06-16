"""M3 FEASIBILITY SPIKE ITERATION 2 (throwaway-OK): couple a COHERENT baroclinic
vorticity source into the v1.6 jupiter_vorticity solver.

Iteration-1 lesson: the LOWER-layer eddy RELATIVE VORTICITY field is dominated by a
C-grid 2dx vorticity checkerboard (m~44-51 vertical stripes) -- NOT the coherent
baroclinic eddies. The coherent signal (m~5) lives in the INTERFACE THICKNESS h2.

This driver derives the source from the eddy interface thickness
    h2e = h2 - zonalmean(h2)
(the clean m~5 field), Gaussian-smooths it, and converts it to a geostrophic
vorticity proxy
    zeta_src = (gp2 / f) * laplacian(smooth(h2e))
with f = 2*omega*sin(phi) (guarded near the equator). This keeps the source
coherent AND concentrated at the baroclinically-active mid-latitudes (the band
envelope confines h2e there).

Opt-in & reversible: uses Simulation.set_external_vorticity_source(); the default
path (never call it) is byte-identical to the unmodified solver.

Usage:  py -3 scripts/sw_m3_spike_coupling2.py [RENDER_RES] [GRID_W] [SRC_STEPS]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim import shallow_water_ref as ref

OUT = Path("out/audit/m3/spike")
RENDER_RES = int(sys.argv[1]) if len(sys.argv) >= 2 else 2048
GRID_W = int(sys.argv[2]) if len(sys.argv) >= 3 else RENDER_RES
SRC_STEPS = int(sys.argv[3]) if len(sys.argv) >= 4 else 11500

# Validated unstable baroclinic config (from the brief).
SRC_W, SRC_H = 192, 96
GP1, GP2, XI = 0.05, 0.3, 3.0
SMOOTH_SIGMA = 2.5  # Gaussian sigma (cells) applied to h2e before the Laplacian.

# Gains as fractions of the solver Coriolis scale (coriolis_f0 = 3.0). The source
# is normalized to unit std, so gain ~= peak (few*std) in solver-q units.
GAINS = {"lowgain": 0.3, "midgain": 0.8, "highgain": 1.5}


def _u8(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _dominant_zonal_m(field2d: np.ndarray) -> tuple[int, np.ndarray]:
    """Dominant zonal wavenumber of a (H,W) field: FFT a band of mid-latitude rows,
    average the power spectra, return argmax m (excluding m=0) and the spectrum."""
    H, W = field2d.shape
    # Mid-band rows: the baroclinic band is centred near +/-45 deg. Use the upper
    # mid-latitudes (rows ~25-40% from the top, descending phi -> northern band).
    r0, r1 = int(0.20 * H), int(0.42 * H)
    rows = field2d[r0:r1]
    rows = rows - rows.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(rows, axis=1)) ** 2
    spec = spec.mean(axis=0)
    m = int(np.argmax(spec[1:]) + 1)  # exclude DC
    return m, spec


def _gaussian_smooth_periodic(field2d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth with wrap in longitude (axis=1) and reflect in latitude."""
    return gaussian_filter(field2d, sigma=sigma, mode=("reflect", "wrap"))


def build_coherent_source(grid_w: int, grid_h: int):
    """Spin the unstable 2-layer baroclinic solver to a nonlinear roll-up, derive a
    COHERENT geostrophic vorticity proxy from the eddy interface thickness h2e."""
    print(f"  building coherent source: {SRC_W}x{SRC_H}, target {SRC_STEPS} steps "
          f"(gp1={GP1}, gp2={GP2}, xi={XI})")
    st = ref.baroclinic_test_state(
        W=SRC_W, H=SRC_H, unstable=True, seed=0,
        gp1=GP1, gp2=GP2, xi_unstable=XI,
        pert_amp_frac=1e-3, dt_safety=0.30, nu4=0.0,
    )
    t0 = time.perf_counter()
    last_good = ref.eddy_interface_var(st)
    steps_done = 0
    for s in range(SRC_STEPS):
        try:
            ref.step_2layer(st)
        except ValueError as e:
            print(f"    OUTCROP (ValueError) at step {s+1}: {e}")
            print(f"    keeping last good state at step {steps_done}")
            break
        steps_done = s + 1
        if steps_done % 1500 == 0:
            ev = ref.eddy_interface_var(st)
            ro = ref.local_rossby_number(st)
            print(f"    step {steps_done:5d}  eddy h2 var={ev:.4e}  Ro={ro:.3f}  "
                  f"({time.perf_counter()-t0:.0f}s)")
            last_good = ev
    print(f"  stepped {steps_done} steps; final eddy h2 var={ref.eddy_interface_var(st):.4e}")

    # --- Coherent source from eddy interface thickness ---
    g = st.g
    h2e = st.h2 - st.h2.mean(axis=1, keepdims=True)        # clean m~5 eddy field
    m_raw, _ = _dominant_zonal_m(h2e)
    print(f"  raw h2e dominant zonal m = {m_raw}")

    # Gaussian smooth (suppress any residual grid-scale content before Laplacian).
    h2e_s = _gaussian_smooth_periodic(h2e, SMOOTH_SIGMA)

    # Geostrophic vorticity proxy: zeta = (gp2/f) * laplacian(h2e_s).
    # f = 2 omega sin(phi), guarded near the equator.
    f_c = 2.0 * st.omega * np.sin(g.phi_c)
    f_safe = np.where(np.abs(f_c) < 1e-12, np.sign(f_c + 1e-30) * 1e-12, f_c)[:, None]
    # Spherical Laplacian (lat-lon), simplified: d2/dphi2 + (1/cos^2) d2/dlam2.
    d2_dphi2 = np.gradient(np.gradient(h2e_s, g.phi_c, axis=0), g.phi_c, axis=0)
    d2_dlam2 = np.gradient(np.gradient(h2e_s, g.dlam, axis=1), g.dlam, axis=1)
    cos2 = (g.cos_c[:, None] ** 2 + 1e-12)
    lap = (d2_dphi2 + d2_dlam2 / cos2) / (g.a ** 2)
    zeta_src = (GP2 / f_safe) * lap

    # Smooth once more lightly to clean up the differentiated field.
    zeta_src = _gaussian_smooth_periodic(zeta_src, 1.0)

    # The physical signal (h2e) is band-confined; everything near the poles is
    # pure numerical artifact (np.gradient edge terms + 1/cos^2 blow-up in the
    # zonal Laplacian). Mask to zero outside |lat| in [10,80] deg with a cosine
    # taper, so the dynamic range concentrates on the active mid-latitudes.
    latdeg = np.degrees(g.phi_c)
    lat_lo, lat_hi, taper = 10.0, 80.0, 8.0
    a_lat = np.clip((np.abs(latdeg) - lat_lo) / taper, 0.0, 1.0)
    b_lat = np.clip((lat_hi - np.abs(latdeg)) / taper, 0.0, 1.0)
    band_mask = (0.5 - 0.5 * np.cos(np.pi * a_lat)) * (0.5 - 0.5 * np.cos(np.pi * b_lat))
    zeta_src = zeta_src * band_mask[:, None]

    m_src, spec = _dominant_zonal_m(zeta_src)
    print(f"  derived zeta_src dominant zonal m = {m_src}")
    top = np.argsort(spec[1:])[::-1][:5] + 1
    print(f"  top-5 zonal modes (m): {top.tolist()}  "
          f"powers={[f'{spec[k]:.2e}' for k in top]}")

    # Resample (SRC_H, SRC_W) -> (grid_h, grid_w). cv2 expects (W, H).
    resamp = cv2.resize(zeta_src.astype(np.float32), (grid_w, grid_h),
                        interpolation=cv2.INTER_LINEAR)
    std = float(np.std(resamp))
    if std > 0:
        resamp = resamp / std
    print(f"  resampled to {grid_w}x{grid_h}, unit std "
          f"(range [{resamp.min():.2f},{resamp.max():.2f}])")
    return resamp.astype(np.float32), m_src, top.tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    gpu = GpuContext.headless()
    gpu.make_current()
    print("GPU:", gpu.ctx.info.get("GL_RENDERER"))

    p = load_factory_preset("jupiter_vorticity")
    p = p.model_copy(update={"sim": p.sim.model_copy(update={"resolution": GRID_W})})
    grid_w, grid_h = GRID_W, GRID_W // 2
    print(f"preset jupiter_vorticity  grid={grid_w}x{grid_h}  "
          f"coriolis_f0={p.solver.coriolis_f0}  render={RENDER_RES}")

    # ---- baseline (no source), re-rendered for a clean comparison ----
    print("\n=== v1.6 default (no source) ===")
    sim = Simulation(p, gpu)
    base = sim.render_maps(RENDER_RES)["color"]
    base_rgb = np.clip(base[..., :3], 0, 1).astype(np.float32)
    base_path = (OUT / "spike_v16_default.png").resolve()
    cv2.imwrite(str(base_path), _u8(base_rgb))
    print(f"  wrote {base_path}")
    sim._release_sim()

    # ---- coherent baroclinic source ----
    print("\n=== coherent baroclinic source ===")
    src, m_src, top_modes = build_coherent_source(grid_w, grid_h)
    a = float(np.abs(src).max()) or 1.0
    gray = (src / a) * 0.5 + 0.5
    src_path = (OUT / "spike2_source.png").resolve()
    cv2.imwrite(str(src_path), (np.clip(gray, 0, 1) * 255).astype(np.uint8))
    print(f"  wrote {src_path}")

    # ---- coupled renders at 3 gains ----
    diffs = {}
    for name, gain in GAINS.items():
        print(f"\n=== coupled {name} (gain={gain}) ===")
        sim = Simulation(p, gpu)
        sim.set_external_vorticity_source(src, gain=gain)
        rgb = sim.render_maps(RENDER_RES)["color"]
        rgb = np.clip(rgb[..., :3], 0, 1).astype(np.float32)
        path = (OUT / f"spike2_coupled_{name}.png").resolve()
        cv2.imwrite(str(path), _u8(rgb))
        d = float(np.abs(rgb - base_rgb).mean())
        diffs[name] = d
        print(f"  wrote {path}  (mean abs-diff vs baseline = {d:.4f})")
        sim._release_sim()

    print(f"\nSOURCE dominant m = {m_src}  top modes = {top_modes}")
    print(f"DIFFS vs baseline = {diffs}")
    print(f"DONE  {time.perf_counter()-wall0:.0f}s total")


if __name__ == "__main__":
    main()

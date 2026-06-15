"""M0 kill-gate harness: SW spike vs v1.6 render.

Produces:
  out/audit/m0/sw_vs_v16.png   — vertical stack: v1.6 belt crop | SW spike belt crop
  out/audit/m0/report.txt      — coher values, R1/R3 gates, eddy_vorticity_std, ms/step

Usage:
    uv run python scripts/sw_spike_killgate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make scripts/ importable for measure_morphology helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_morphology import _belt_crop_from_rgb, _lum, coher  # noqa: E402

from gasgiant.engine.facade import Simulation  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402

# SW spike imports.
from gasgiant.sim.sw_spike.init import emergent_init  # noqa: E402
from gasgiant.sim.sw_spike.solver import (  # noqa: E402
    checkerboard_amplitude,
    eddy_vorticity_std,
    step,
)
from gasgiant.sim.sw_spike.encode import to_tracer  # noqa: E402

OUT = Path("out/audit/m0")
RES = 4096          # GPU render resolution for both v1.6 and SW
BELT_WIDTH = 640    # crop width for the comparison strip
SW_N_STEPS = 8000   # target steps for the SW spike
SW_CHECK_INTERVAL = 500  # NaN check every N steps

# Validated config for the SW spike.
SW_W, SW_H = 256, 128
SW_FALLBACK_W, SW_FALLBACK_H = 192, 96
SW_F0 = 4.0
SW_GP = (1.0, 0.05)
SW_N_BANDS = 14
SW_BAND_CONTRAST = 0.5


def _label(img_u8, text):
    out = img_u8.copy()
    cv2.rectangle(out, (0, 0), (img_u8.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _u8(rgb01):
    return cv2.cvtColor(
        (np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )


def _run_sw_spike(W, H):
    """Run the SW spike to completion.  Returns (state, ms_per_step)."""
    print(f"  Initialising SW spike ({W}x{H})...")
    st = emergent_init(W, H, SW_F0, SW_GP, SW_N_BANDS, SW_BAND_CONTRAST)

    t0 = time.perf_counter()
    for chunk_start in range(0, SW_N_STEPS, SW_CHECK_INTERVAL):
        chunk_end = min(chunk_start + SW_CHECK_INTERVAL, SW_N_STEPS)
        for _ in range(chunk_end - chunk_start):
            st = step(st, st.dt)
        if not (np.isfinite(st.h1).all() and np.isfinite(st.u1).all()):
            raise RuntimeError(
                f"SW spike NaN at step {chunk_end} ({W}x{H})"
            )
        print(f"    step {chunk_end}/{SW_N_STEPS}  "
              f"h1_range=[{st.h1.min():.3f}, {st.h1.max():.3f}]")
    elapsed = time.perf_counter() - t0
    ms_per_step = 1000.0 * elapsed / SW_N_STEPS
    return st, ms_per_step


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. v1.6 jupiter_vorticity baseline render
    # ------------------------------------------------------------------ #
    print("=== v1.6 jupiter_vorticity baseline ===")
    gpu = GpuContext.headless()
    gpu.make_current()

    p = load_factory_preset("jupiter_vorticity")
    sim = Simulation(p, gpu)
    print(f"  Running v1.6 sim to completion ({p.sim.dev_steps} steps)...")
    rgb_v16 = sim.render_maps(RES)["color"]

    # Get belt crop AND save the belt box for reuse on SW render.
    v16_crop, belt_box = _belt_crop_from_rgb(rgb_v16, sim, BELT_WIDTH)
    coher_v16 = coher(_lum(v16_crop))
    print(f"  v1.6 coher = {coher_v16:.4f}")
    sim._release_sim()

    # ------------------------------------------------------------------ #
    # 2. SW spike spin-up
    # ------------------------------------------------------------------ #
    print("\n=== SW spike spin-up ===")
    try:
        sw_st, ms_per_step = _run_sw_spike(SW_W, SW_H)
        sw_config = f"{SW_W}x{SW_H}"
    except RuntimeError as e:
        print(f"  WARNING: {e}  -- falling back to {SW_FALLBACK_W}x{SW_FALLBACK_H}")
        sw_st, ms_per_step = _run_sw_spike(SW_FALLBACK_W, SW_FALLBACK_H)
        sw_config = f"{SW_FALLBACK_W}x{SW_FALLBACK_H}"

    # ------------------------------------------------------------------ #
    # 3. Compute R1 / R3 gates
    # ------------------------------------------------------------------ #
    r1_amp = checkerboard_amplitude(sw_st.h1)
    r1_pass = r1_amp < 0.05
    r3_min = float(np.min(sw_st.h1))
    r3_pass = (r3_min >= sw_st.h_floor - 1e-6) and np.isfinite(sw_st.h1).all()
    ev_std = eddy_vorticity_std(sw_st)
    print(f"  R1 (checker amp={r1_amp:.5f} < 0.05): {'PASS' if r1_pass else 'FAIL'}")
    print(f"  R3 (min h1={r3_min:.4f} >= floor={sw_st.h_floor:.4f}): "
          f"{'PASS' if r3_pass else 'FAIL'}")
    print(f"  eddy_vorticity_std = {ev_std:.5f}")
    print(f"  ms/step = {ms_per_step:.2f}")

    # ------------------------------------------------------------------ #
    # 4. Feed SW tracer through the GPU render
    # ------------------------------------------------------------------ #
    print("\n=== SW spike GPU render ===")
    sw_tracer = to_tracer(sw_st)   # (H, W, 4) float32
    print(f"  tracer shape: {sw_tracer.shape}  dtype: {sw_tracer.dtype}")

    # Reuse the same gpu context — ONE GpuContext per process.
    # Use the v1.6 preset's appearance so colour palette is jupiter-like.
    from gasgiant.render.maps import MapDeriver
    deriver = MapDeriver(gpu)
    rgb_sw = deriver.derive_from_tracer(
        sw_tracer, RES, p.appearance, seed=p.seed
    )
    print(f"  SW render shape: {rgb_sw.shape}")

    # ------------------------------------------------------------------ #
    # 5. Belt crop on the SW render — use the SAME belt box as v1.6
    # ------------------------------------------------------------------ #
    from measure_morphology import _crop_deg, _fit_width
    import cv2 as _cv2

    # Match width to the reference image (as _belt_crop_from_rgb does).
    ref_w = _cv2.imread("refs/PIA07782.jpg").shape[1]
    rgb_sw_matched = _fit_width(
        np.clip(rgb_sw[..., :3], 0, 1).astype(np.float32), ref_w
    )
    sw_crop = _fit_width(_crop_deg(rgb_sw_matched, *belt_box), BELT_WIDTH)
    coher_sw = coher(_lum(sw_crop))
    print(f"  SW coher = {coher_sw:.4f}  (ref 0.62)")

    # ------------------------------------------------------------------ #
    # 6. Write comparison image
    # ------------------------------------------------------------------ #
    strip = np.vstack([
        _label(_u8(v16_crop),
               f"v1.6 jupiter_vorticity  coher={coher_v16:.3f}"),
        _label(_u8(sw_crop),
               f"SW spike ({sw_config})  coher={coher_sw:.3f}"),
    ])
    img_path = OUT / "sw_vs_v16.png"
    cv2.imwrite(str(img_path), strip)
    print(f"  wrote {img_path}")

    # ------------------------------------------------------------------ #
    # 7. Write report
    # ------------------------------------------------------------------ #
    report_lines = [
        "M0 KILL-GATE REPORT",
        "=" * 60,
        f"v1.6 jupiter_vorticity coher : {coher_v16:.4f}",
        f"SW spike coher               : {coher_sw:.4f}  (reference target: 0.62)",
        "",
        f"R1 (checkerboard_amplitude)  : amp={r1_amp:.5f}  {'PASS' if r1_pass else 'FAIL'}  (gate: < 0.05)",
        f"R3 (min h1 >= floor, no NaN) : min_h1={r3_min:.4f}  floor={sw_st.h_floor}  {'PASS' if r3_pass else 'FAIL'}",
        "",
        f"eddy_vorticity_std           : {ev_std:.5f}",
        f"SW config                    : {sw_config}",
        f"SW steps                     : {SW_N_STEPS}",
        f"ms/step                      : {ms_per_step:.2f}",
        f"GPU render res               : {RES}",
        "",
        "NOTE: The BLIND JUDGE PANEL on sw_vs_v16.png is the binding gate",
        "(MD-1).  coher is a necessary-not-sufficient morphology metric.",
        "The SW spike runs at coarse CPU resolution (a handicap vs v1.6's",
        f"4096-cell GPU sim).  A WIN is strong; a narrow LOSS may be",
        "resolution-bound and not a fundamental physics failure.",
    ]
    report_path = OUT / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  wrote {report_path}")

    print("\n=== SUMMARY ===")
    print(f"  v1.6 coher={coher_v16:.4f}  SW coher={coher_sw:.4f}")
    print(f"  R1={'PASS' if r1_pass else 'FAIL'}  R3={'PASS' if r3_pass else 'FAIL'}")
    print(f"  eddy_vorticity_std={ev_std:.5f}  ms/step={ms_per_step:.2f}")


if __name__ == "__main__":
    main()

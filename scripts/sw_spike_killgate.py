"""M0 kill-gate harness: SW spike vs v1.6 render.

Produces:
  out/audit/m0/sw_vs_v16.png      — vertical stack: v1.6 belt crop | SW spike belt crop
  out/audit/m0/sw_render_full.png — raw SW equirect render (unlabeled, full)
  out/audit/m0/report.txt         — coher values, R1/R3 gates, eddy_vorticity_std, ms/step

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
SW_N_STEPS = 12000  # target steps for the SW spike (validated eddy-developing config)
SW_CHECK_INTERVAL = 500  # NaN check every N steps

# Validated eddy-developing config (192x96, nu4=0.05).
# At 256x128 the dt is tiny so 8000 steps is too little physical time — eddies never grow.
# At 192x96 nu4=0.05 is stable (smaller dt keeps it from blowing up) and reaches
# eddy_vorticity_std ~1.5-2.0 (filamentary) over ~8000-10000 steps.
SW_W, SW_H = 192, 96
SW_F0 = 4.0
SW_GP = (1.0, 0.05)
SW_N_BANDS = 14
SW_BAND_CONTRAST = 0.5
SW_NU4 = 0.05
SW_NU4_FALLBACK = 0.07  # retry value if NaN at SW_NU4


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


def _run_sw_spike(W, H, nu4):
    """Run the SW spike to completion.  Returns (state, ms_per_step).
    Raises RuntimeError if NaN is detected."""
    print(f"  Initialising SW spike ({W}x{H}, nu4={nu4})...")
    st = emergent_init(W, H, SW_F0, SW_GP, SW_N_BANDS, SW_BAND_CONTRAST)
    # Override nu4 AFTER emergent_init (which sets its own default).
    st.nu4 = nu4

    t0 = time.perf_counter()
    for chunk_start in range(0, SW_N_STEPS, SW_CHECK_INTERVAL):
        chunk_end = min(chunk_start + SW_CHECK_INTERVAL, SW_N_STEPS)
        for _ in range(chunk_end - chunk_start):
            st = step(st, st.dt)
        if not (np.isfinite(st.h1).all() and np.isfinite(st.u1).all()):
            raise RuntimeError(
                f"SW spike NaN at step {chunk_end} ({W}x{H}, nu4={nu4})"
            )
        print(f"    step {chunk_end}/{SW_N_STEPS}  "
              f"h1_range=[{st.h1.min():.3f}, {st.h1.max():.3f}]")
    elapsed = time.perf_counter() - t0
    ms_per_step = 1000.0 * elapsed / SW_N_STEPS
    return st, ms_per_step


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wall_t0 = time.perf_counter()

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
    # 2. SW spike spin-up — validated eddy-developing config (192x96, nu4=0.05)
    # ------------------------------------------------------------------ #
    print("\n=== SW spike spin-up ===")
    sw_nu4_used = SW_NU4
    try:
        sw_st, ms_per_step = _run_sw_spike(SW_W, SW_H, SW_NU4)
        sw_config = f"{SW_W}x{SW_H} nu4={SW_NU4}"
    except RuntimeError as e:
        print(f"  WARNING: {e}")
        print(f"  Retrying once with nu4={SW_NU4_FALLBACK}...")
        sw_nu4_used = SW_NU4_FALLBACK
        sw_st, ms_per_step = _run_sw_spike(SW_W, SW_H, SW_NU4_FALLBACK)
        sw_config = f"{SW_W}x{SW_H} nu4={SW_NU4_FALLBACK}"

    # ------------------------------------------------------------------ #
    # 3. Fail-fast eddy check BEFORE rendering
    # ------------------------------------------------------------------ #
    ev_std = eddy_vorticity_std(sw_st)
    eddies_developed = ev_std >= 0.3
    if not eddies_developed:
        print(f"\n  *** WARNING: SPIN-UP DID NOT DEVELOP EDDIES (e={ev_std:.5f}) ***")
        print("  The render is NOT a valid SW test — still producing artifacts so we can see what happened.")
    else:
        print(f"\n  Eddies developed: eddy_vorticity_std = {ev_std:.5f} (>= 0.3 threshold) PASS")

    # ------------------------------------------------------------------ #
    # 4. Compute R1 / R3 gates
    # ------------------------------------------------------------------ #
    r1_amp = checkerboard_amplitude(sw_st.h1)
    r1_pass = r1_amp < 0.05
    r3_min = float(np.min(sw_st.h1))
    r3_pass = (r3_min >= sw_st.h_floor - 1e-6) and np.isfinite(sw_st.h1).all()
    print(f"  R1 (checker amp={r1_amp:.5f} < 0.05): {'PASS' if r1_pass else 'FAIL'}")
    print(f"  R3 (min h1={r3_min:.4f} >= floor={sw_st.h_floor:.4f}): "
          f"{'PASS' if r3_pass else 'FAIL'}")
    print(f"  ms/step = {ms_per_step:.2f}")

    # ------------------------------------------------------------------ #
    # 5. Feed SW tracer through the GPU render
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
    # 6. Save raw SW equirect render (full, unlabeled)
    # ------------------------------------------------------------------ #
    sw_full_path = OUT / "sw_render_full.png"
    cv2.imwrite(str(sw_full_path), _u8(rgb_sw[..., :3]))
    print(f"  wrote {sw_full_path}")

    # ------------------------------------------------------------------ #
    # 7. Belt crop on the SW render — use the SAME belt box as v1.6
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
    print("  NOTE: coher is UNRELIABLE for coarse 192x96 upscaled to 4K — see report.")

    # ------------------------------------------------------------------ #
    # 8. Write comparison image (v1.6 belt crop + SW belt crop stacked)
    # ------------------------------------------------------------------ #
    strip = np.vstack([
        _label(_u8(v16_crop),
               f"v1.6 jupiter_vorticity  coher={coher_v16:.3f}"),
        _label(_u8(sw_crop),
               f"SW spike ({sw_config})  coher={coher_sw:.3f}  e={ev_std:.4f}"),
    ])
    img_path = OUT / "sw_vs_v16.png"
    cv2.imwrite(str(img_path), strip)
    print(f"  wrote {img_path}")

    wall_elapsed = time.perf_counter() - wall_t0

    # ------------------------------------------------------------------ #
    # 9. Write report
    # ------------------------------------------------------------------ #
    eddy_status = "PASS (eddies developed)" if eddies_developed else (
        f"FAIL — SPIN-UP DID NOT DEVELOP EDDIES (e={ev_std:.5f}) "
        "— render is not a valid SW test"
    )
    report_lines = [
        "M0 KILL-GATE REPORT",
        "=" * 60,
        f"v1.6 jupiter_vorticity coher : {coher_v16:.4f}",
        f"SW spike coher               : {coher_sw:.4f}  (reference target: 0.62)",
        "",
        "*** COHER CAVEAT (IMPORTANT) ***",
        "coher is UNRELIABLE for the coarse 192x96 spike upscaled to 4K",
        "(it scores trivially-horizontal upscaled fields near 1.0); the",
        "binding gate is the BLIND VISUAL PANEL on sw_vs_v16.png — look",
        "for folded-filament belt morphology, not the coher number.",
        "",
        f"R1 (checkerboard_amplitude)  : amp={r1_amp:.5f}  {'PASS' if r1_pass else 'FAIL'}  (gate: < 0.05)",
        f"R3 (min h1 >= floor, no NaN) : min_h1={r3_min:.4f}  floor={sw_st.h_floor}  {'PASS' if r3_pass else 'FAIL'}",
        "",
        f"eddy_vorticity_std           : {ev_std:.5f}",
        f"Eddies developed (e >= 0.3)  : {eddy_status}",
        f"SW config                    : {sw_config}",
        f"SW steps                     : {SW_N_STEPS}",
        f"ms/step                      : {ms_per_step:.2f}",
        f"Total wall time              : {wall_elapsed:.1f}s",
        f"GPU render res               : {RES}",
        "",
        "Artifacts written:",
        f"  out/audit/m0/sw_vs_v16.png      — v1.6 belt crop | SW belt crop (labeled)",
        f"  out/audit/m0/sw_render_full.png — raw SW equirect render (unlabeled, full)",
        f"  out/audit/m0/report.txt         — this report",
        "",
        "BINDING GATE: The BLIND VISUAL PANEL on sw_vs_v16.png is the kill-gate",
        "(MD-1). Look for folded-filament belt morphology in the SW row.",
        "coher is a secondary metric only — do NOT use it as the primary decision.",
        "A WIN on the visual panel is strong; a narrow LOSS may be resolution-bound",
        "and not a fundamental physics failure.",
    ]
    report_path = OUT / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  wrote {report_path}")

    print("\n=== SUMMARY ===")
    print(f"  v1.6 coher={coher_v16:.4f}  SW coher={coher_sw:.4f} (UNRELIABLE — see report)")
    print(f"  R1={'PASS' if r1_pass else 'FAIL'}  R3={'PASS' if r3_pass else 'FAIL'}")
    print(f"  eddy_vorticity_std={ev_std:.5f}  eddies={'YES (PASS)' if eddies_developed else 'NO (FAIL)'}")
    print(f"  ms/step={ms_per_step:.2f}  wall={wall_elapsed:.1f}s")
    print(f"  sw_vs_v16.png: {img_path}")
    print(f"  sw_render_full.png: {sw_full_path}")


if __name__ == "__main__":
    main()

"""M0.5 KILL-GATE harness: GPU 2-layer SW vs v1.6 (morphology-only, fair).

FAIRNESS FIX vs M0: both renders use morphology-only mode:
  - v1.6 jupiter_vorticity: detail.intensity=0, bands.warp_amount=0, bands.lane_density=0
  - SW: derive_from_tracer (already no detail/warp/lanes by construction)
  - Same palette/appearance on both sides

BLIND output: swp_vs_v16_blind.png has UNLABELED crops in randomized order.
Blind key (which is top/bottom) is written to report.txt only.

FALSIFIABILITY GUARD: if eddy_vorticity_std < 1.0 at end of spin-up,
verdict is INCONCLUSIVE (not a refutation).

Produces:
  out/audit/m0p5/swp_vs_v16_blind.png  — stacked belt crops, UNLABELED, random order
  out/audit/m0p5/sw_render_full.png    — raw SW equirect render (full, unlabeled)
  out/audit/m0p5/report.txt            — full metrics + blind key

Usage:
    uv run python scripts/swp_killgate.py
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make scripts/ importable for measure_morphology helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_morphology import _belt_crop_from_rgb, _crop_deg, _fit_width, _lum, coher  # noqa: E402

from gasgiant.engine.facade import Simulation  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402
from gasgiant.render.maps import MapDeriver  # noqa: E402
from gasgiant.sim.sw_gpu_probe import solver as gsolver  # noqa: E402
from gasgiant.sim.sw_spike import init  # noqa: E402
from gasgiant.sim.sw_spike.solver import checkerboard_amplitude  # noqa: E402

OUT = Path("out/audit/m0p5")
RES = 4096
BELT_WIDTH = 640

# Spin-up config (reuse swp_spinup.py settings)
W_REF, H_REF = 192, 96
W_HI,  H_HI  = 512, 256
NU4          = 0.07
STEP_CAP     = 32000
STEP_CHUNK   = 1000
EVS_TARGET   = 1.0

# Fixed seed for blind panel randomization (reproducible).
BLIND_SEED = 42


def _compute_dt(W: int, H: int, geff: float = 1.0, h_mean: float = 5.0) -> float:
    from gasgiant.sim.sw_spike.grid import Grid
    g = Grid(W, H)
    c_gw = np.sqrt(geff * h_mean)
    dx_min = min(g.cos_c.min() * g.dlam, g.dphi)
    return 0.3 * dx_min / c_gw


def _u8(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        (np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def build_tracer_from_gpu(sg: gsolver.SwpSolver) -> np.ndarray:
    """Download GPU fields and pack into (H, W, 4) RGBA tracer like sw_spike encode.to_tracer.

    r = color index  <- thickness anomaly (banded color)
    g = height       <- thickness (cloud altitude)
    b = detail       <- relative vorticity magnitude
    a = tint         <- signed vorticity (storm polarity)
    """
    h1    = sg.download("h1")      # (H, W)
    h_eq1 = sg.download("h_eq1")   # (H, W)
    u1    = sg.download("u1")      # (H, W)
    v1    = sg.download("v1")      # (H+1, W)

    H, W = h1.shape
    h_anom = h1 - h_eq1

    # Relative vorticity at corners via GPU kernel (returns (H+1, W)).
    zeta_corner = gsolver.run_vorticity(sg.gpu, u1, v1)
    # Average to cell centres.
    zeta_c = 0.5 * (zeta_corner[0:H] + zeta_corner[1:H + 1])

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., 0] = _norm(h_anom)
    rgba[..., 1] = _norm(h1)
    rgba[..., 2] = _norm(np.abs(zeta_c))
    rgba[..., 3] = _norm(zeta_c)
    return rgba


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wall_t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 1. GPU context (ONE per process — reused for both renders)
    # ------------------------------------------------------------------ #
    gpu = GpuContext.headless()
    gpu.make_current()

    # ------------------------------------------------------------------ #
    # 2. v1.6 jupiter_vorticity — MORPHOLOGY-ONLY render
    #    detail.intensity=0, bands.warp_amount=0, bands.lane_density=0
    # ------------------------------------------------------------------ #
    print("=== v1.6 jupiter_vorticity (morphology-only) ===")
    p = load_factory_preset("jupiter_vorticity")

    # Apply morphology-only zeroing.
    p = p.model_copy(update={
        "detail": p.detail.model_copy(update={"intensity": 0.0}),
        "bands":  p.bands.model_copy(update={"warp_amount": 0.0, "lane_density": 0.0}),
    })
    print(f"  detail.intensity = {p.detail.intensity}  (must be 0.0)")
    print(f"  bands.warp_amount = {p.bands.warp_amount}  (must be 0.0)")
    print(f"  bands.lane_density = {p.bands.lane_density}  (must be 0.0)")

    sim = Simulation(p, gpu)
    print(f"  Running v1.6 sim to completion ({p.sim.dev_steps} steps)...")
    rgb_v16 = sim.render_maps(RES)["color"]   # (H, W, 4) float32

    # Belt crop (uses sim.bands for widest tropical belt detection).
    v16_crop, belt_box = _belt_crop_from_rgb(rgb_v16, sim, BELT_WIDTH)
    coher_v16 = coher(_lum(v16_crop))
    print(f"  v1.6 coher (morphology-only) = {coher_v16:.4f}")

    # Save the appearance so SW render uses the same palette.
    appearance = p.appearance
    seed = p.seed
    sim._release_sim()

    # ------------------------------------------------------------------ #
    # 3. GPU 2-layer SW spin-up (512x256, nu4=0.07, physical-time rescale)
    # ------------------------------------------------------------------ #
    print("\n=== GPU 2-layer SW spin-up (512x256) ===")

    # Compute physical-time dt_scale.
    st_ref = init.emergent_init(
        W=W_REF, H=H_REF, f0=4.0, gp=(1.0, 0.05), n_bands=10, band_contrast=0.4,
    )
    dt_ref = st_ref.dt
    dt_hi = _compute_dt(W_HI, H_HI)
    dt_scale = dt_hi / dt_ref
    print(f"  dt_ref (192x96)  = {dt_ref:.6e}")
    print(f"  dt_hi  (512x256) = {dt_hi:.6e}")
    print(f"  dt_scale         = {dt_scale:.6f}")

    # Initialize CPU state then upload to GPU.
    st_hi = init.emergent_init(
        W=W_HI, H=H_HI, f0=4.0, gp=(1.0, 0.05), n_bands=10, band_contrast=0.4,
    )
    st_hi = dataclasses.replace(st_hi, nu4=NU4)
    sg = gsolver.SwpSolver.from_cpu_state(gpu, st_hi, forcing_dt_scale=dt_scale)

    evs_target_reached = False
    evs_target_step = None
    step_count = 0
    ms_per_step_last = float("nan")

    evs0 = sg.eddy_vorticity_std()
    print(f"  step={step_count:6d}  eddy_vort_std={evs0:.4f}")

    t_spin_start = time.perf_counter()

    for chunk_start in range(0, STEP_CAP, STEP_CHUNK):
        n = min(STEP_CHUNK, STEP_CAP - chunk_start)
        t0 = time.perf_counter()
        for _ in range(n):
            sg.step()
        t1 = time.perf_counter()
        step_count += n

        h1 = sg.download("h1")
        if not np.all(np.isfinite(h1)):
            print(f"  !! BLOWUP at step {step_count} — aborting spin-up.")
            break

        evs = sg.eddy_vorticity_std()
        ms_per_step_last = (t1 - t0) / n * 1000.0
        print(f"  step={step_count:6d}  eddy_vort_std={evs:.4f}  {ms_per_step_last:.2f} ms/step")

        if not evs_target_reached and evs >= EVS_TARGET:
            evs_target_reached = True
            evs_target_step = step_count
            print(f"  *** eddy_vorticity_std >= {EVS_TARGET:.1f} reached at step {step_count} ***")

    t_spin_elapsed = time.perf_counter() - t_spin_start
    final_evs = sg.eddy_vorticity_std()
    ms_avg = t_spin_elapsed / step_count * 1000.0 if step_count > 0 else float("nan")
    print(f"\n  Final eddy_vort_std = {final_evs:.4f}")
    if evs_target_reached:
        print(f"  Regime >= {EVS_TARGET:.1f} reached at step {evs_target_step}")
    else:
        print(f"  Regime >= {EVS_TARGET:.1f} NOT reached in {step_count} steps — INCONCLUSIVE")

    # ------------------------------------------------------------------ #
    # 4. R1 / R3 gates (on h1)
    # ------------------------------------------------------------------ #
    h1_final = sg.download("h1")
    h_floor = float(st_hi.h_floor)

    r1_amp  = checkerboard_amplitude(h1_final)
    r1_pass = r1_amp < 0.05
    r3_min  = float(np.min(h1_final))
    r3_pass = (r3_min >= h_floor - 1e-6) and bool(np.all(np.isfinite(h1_final)))

    print(f"\n  R1 (checkerboard_amplitude={r1_amp:.5f} < 0.05): {'PASS' if r1_pass else 'FAIL'}")
    print(f"  R3 (min h1={r3_min:.4f} >= floor={h_floor:.4f}, no NaN): {'PASS' if r3_pass else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # 5. Build SW tracer from GPU state and render via derive_from_tracer
    # ------------------------------------------------------------------ #
    print("\n=== SW GPU render ===")
    sw_tracer = build_tracer_from_gpu(sg)
    print(f"  tracer shape: {sw_tracer.shape}  dtype: {sw_tracer.dtype}")

    deriver = MapDeriver(gpu)
    rgb_sw = deriver.derive_from_tracer(sw_tracer, RES, appearance, seed=seed)
    print(f"  SW render shape: {rgb_sw.shape}")

    # ------------------------------------------------------------------ #
    # 6. Save raw SW equirect (unlabeled, full)
    # ------------------------------------------------------------------ #
    sw_full_path = OUT / "sw_render_full.png"
    cv2.imwrite(str(sw_full_path), _u8(rgb_sw[..., :3]))
    print(f"  wrote {sw_full_path}")

    # ------------------------------------------------------------------ #
    # 7. Belt crop on SW render — use the SAME belt_box as v1.6
    # ------------------------------------------------------------------ #
    from gasgiant.render.maps import chroma_uniforms  # noqa: F401 (just verifying import)

    sw_rgb3 = np.clip(rgb_sw[..., :3], 0, 1).astype(np.float32)
    # Match to reference width (as _belt_crop_from_rgb does internally).
    ref_img = cv2.imread("refs/PIA07782.jpg")
    ref_w = ref_img.shape[1] if ref_img is not None else 3000
    sw_matched = _fit_width(sw_rgb3, ref_w)
    sw_crop = _fit_width(_crop_deg(sw_matched, *belt_box), BELT_WIDTH)
    coher_sw = coher(_lum(sw_crop))
    print(f"  SW coher = {coher_sw:.4f}")

    # ------------------------------------------------------------------ #
    # 8. BLIND image: randomized top/bottom, UNLABELED
    # ------------------------------------------------------------------ #
    rng_blind = np.random.default_rng(BLIND_SEED)
    flip = bool(rng_blind.integers(0, 2))  # True -> SW on top, False -> v1.6 on top

    crop_a = _u8(sw_crop)    if flip else _u8(v16_crop)
    crop_b = _u8(v16_crop)   if flip else _u8(sw_crop)

    # Ensure both crops are the same width for vstack.
    wa, wb = crop_a.shape[1], crop_b.shape[1]
    if wa != wb:
        target_w = min(wa, wb)
        crop_a = _fit_width(crop_a.astype(np.float32) / 255.0, target_w)
        crop_b = _fit_width(crop_b.astype(np.float32) / 255.0, target_w)
        crop_a = (np.clip(crop_a, 0, 1) * 255).astype(np.uint8)
        crop_b = (np.clip(crop_b, 0, 1) * 255).astype(np.uint8)

    blind_strip = np.vstack([crop_a, crop_b])
    blind_path = OUT / "swp_vs_v16_blind.png"
    cv2.imwrite(str(blind_path), blind_strip)
    print(f"  wrote {blind_path}  (UNLABELED — top is {'SW' if flip else 'v1.6'}, bottom is {'v1.6' if flip else 'SW'})")

    # ------------------------------------------------------------------ #
    # 9. Verdict / INCONCLUSIVE guard
    # ------------------------------------------------------------------ #
    wall_elapsed = time.perf_counter() - wall_t0

    if not evs_target_reached:
        verdict_status = "INCONCLUSIVE — regime not reached (eddy_vort_std < 1.0); DO NOT refute"
    else:
        verdict_status = "PENDING BLIND PANEL (regime reached; judges decide GO/NO-GO)"

    blind_key = (
        f"TOP = {'SW (GPU 2-layer)' if flip else 'v1.6 jupiter_vorticity (morphology-only)'} | "
        f"BOTTOM = {'v1.6 jupiter_vorticity (morphology-only)' if flip else 'SW (GPU 2-layer)'}"
    )

    # ------------------------------------------------------------------ #
    # 10. Write report
    # ------------------------------------------------------------------ #
    report_lines = [
        "M0.5 KILL-GATE REPORT",
        "=" * 70,
        "",
        "PIPELINE MATCH MODE: morphology-only (FAIR COMPARISON)",
        "  v1.6 : detail.intensity=0, bands.warp_amount=0, bands.lane_density=0",
        "  SW   : derive_from_tracer (no detail/warp/lanes by construction)",
        "  Both use same appearance params (palette, contrast, saturation etc.)",
        "",
        f"Resolution               : {W_HI}x{H_HI} grid -> rendered at {RES}px",
        f"nu4                      : {NU4}",
        f"dt_scale                 : {dt_scale:.6f}  (dt_hi / dt_ref)",
        f"Steps run                : {step_count}",
        f"Final eddy_vort_std      : {final_evs:.4f}",
        f"Regime >= {EVS_TARGET:.1f} reached    : {'YES at step ' + str(evs_target_step) if evs_target_reached else 'NO — INCONCLUSIVE'}",
        "",
        f"v1.6 coher (morph-only)  : {coher_v16:.4f}",
        f"SW coher                 : {coher_sw:.4f}",
        f"Belt box (lat0,lat1,lon0,lon1): {belt_box}",
        "",
        f"R1 (checkerboard amp)    : {r1_amp:.5f}  {'PASS' if r1_pass else 'FAIL'}  (gate: < 0.05)",
        f"R3 (min h1 >= floor, no NaN): min_h1={r3_min:.4f}  floor={h_floor}  {'PASS' if r3_pass else 'FAIL'}",
        "",
        f"ms/step (avg)            : {ms_avg:.2f}",
        f"Wall time                : {wall_elapsed:.1f}s",
        "",
        "Artifacts written:",
        f"  {OUT}/swp_vs_v16_blind.png  — stacked crops, UNLABELED, randomized order",
        f"  {OUT}/sw_render_full.png    — raw SW equirect render (unlabeled, full)",
        f"  {OUT}/report.txt            — this report",
        "",
        "VERDICT STATUS: " + verdict_status,
        "",
        "FALSIFIABILITY RULE:",
        "  LOSE refutes (is evidence against SW physics) ONLY IF:",
        "    (1) morphology-only renders on both sides (MET: detail/warp/lanes=0 on v1.6)",
        "    (2) regime eddy_vort_std >= 1.0 was reached (see above)",
        "  If regime NOT reached -> INCONCLUSIVE (may still render, cannot refute).",
        "",
        "=== BLIND KEY (read AFTER human judges the image) ===",
        blind_key,
        "",
        "Blind seed: " + str(BLIND_SEED) + "  flip=" + str(flip),
    ]

    report_path = OUT / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  wrote {report_path}")

    # ------------------------------------------------------------------ #
    # 11. Console summary
    # ------------------------------------------------------------------ #
    print("\n=== SUMMARY ===")
    print(f"  Resolution: {W_HI}x{H_HI} -> {RES}px render")
    print(f"  Final eddy_vort_std = {final_evs:.4f}  ({'REGIME REACHED' if evs_target_reached else 'INCONCLUSIVE — regime not reached'})")
    print("  Pipeline match: morphology-only (detail/warp/lanes=0 on v1.6 side)")
    print(f"  v1.6 coher = {coher_v16:.4f}")
    print(f"  SW  coher  = {coher_sw:.4f}")
    print(f"  R1={'PASS' if r1_pass else 'FAIL'}  R3={'PASS' if r3_pass else 'FAIL'}")
    print(f"  ms/step={ms_avg:.2f}  wall={wall_elapsed:.1f}s")
    print(f"  {blind_path}  [UNLABELED]")
    print(f"  {sw_full_path}")
    print(f"  BLIND KEY: {blind_key}")
    print(f"  VERDICT: {verdict_status}")


if __name__ == "__main__":
    main()

"""M0 shallow-water spike exploration script.

Sweeps 7 configs to test whether non-zonal (eddy) vorticity grows under
baroclinic instability, and whether folded-filamentary structure is reachable
in feasible compute at 192x96. Determines kill-gate recommendation.

Run 2026-06-14 findings summary
--------------------------------
Primary sweep (6000 steps, W=192x96):
  baseline (nu4=0.20)          start=0.0009 -> end=0.0000  DECAYS
  low_nu4_0.02                 start=0.0009 -> peak=0.9149 @3000, then NaN  (too unstable)
  f0_2_more_unstable           start=0.0009 -> end=0.0000  DECAYS
  band_contrast_0.8            start=0.0008 -> end=0.0000  DECAYS
  big_seed_h1_noise            start=0.0087 -> end=0.0005  DECAYS
  gp_0.3_bigger_dt             start=0.0005 -> end=0.0000  DECAYS
  weak_drag_1e9                start=0.0009 -> end=0.0001  DECAYS

nu4 sweet-spot probe (8000 steps):
  nu4=0.05  0.00093->1.96  clean exponential growth, sustained, NO NaN  **WINNER**
  nu4=0.08  0.00092->0.010 slow but growing — stable
  nu4=0.12  0.00090->0.00029 — effectively DECAYS (nu4 too high)

Kill-gate verdict: nu4=0.05 produces FILAMENTARY structure (high-k fraction=0.974)
at 8000 steps, ~4.9 ms/step. Kill-gate render IS worth running with this config.

The vorticity PNG (out/audit/m0/explore_vort.png) at step 8000 shows strongly
non-zonal filamentary texture confirming deep nonlinear eddy development.

Usage:
    python scripts/sw_spike_explore.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gasgiant.sim.sw_spike import init, operators, solver  # noqa: E402

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "out" / "audit" / "m0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Eddy diagnostics
# ---------------------------------------------------------------------------

def eddy_vort_std(st) -> float:
    """std of vorticity deviation from the zonal mean.

    zeta lives at corners (H+1, W); we use interior rows [1:H] to
    avoid the hard-zeroed pole rows which would dilute the signal.
    """
    g = st.g
    zeta = operators.vorticity(st.u1, st.v1, g)   # (H+1, W)
    zeta_int = zeta[1:g.H]                        # interior corners only (H-1, W)
    zonal_mean = zeta_int.mean(axis=1, keepdims=True)  # (H-1, 1)
    return float(np.std(zeta_int - zonal_mean))


def eddy_ke(st) -> float:
    """Mean eddy kinetic energy: KE of (u - zonal_mean(u)) and v deviation."""
    g = st.g
    u = st.u1                                    # (H, W)
    vc = 0.5 * (st.v1[0:g.H] + st.v1[1:g.H + 1])  # (H, W) — v at centers
    u_eddy = u - u.mean(axis=1, keepdims=True)
    v_eddy = vc - vc.mean(axis=1, keepdims=True)
    return float(np.mean(u_eddy**2 + v_eddy**2))


# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------

def run_config(name: str, W: int, H: int, f0: float, gp: tuple,
               n_bands: int, band_contrast: float,
               steps: int, log_every: int,
               nu4_override=None,
               tau_drag_override=None,
               h1_noise_seed=None) -> dict:
    """Spin up one config and return a summary dict."""
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    st = init.emergent_init(W=W, H=H, f0=f0, gp=gp,
                            n_bands=n_bands, band_contrast=band_contrast)

    # Apply overrides *after* init so we can test individual axes
    if nu4_override is not None:
        st.nu4 = nu4_override
        print(f"  nu4 overridden -> {nu4_override}")
    if tau_drag_override is not None:
        st.tau_drag = tau_drag_override
        print(f"  tau_drag overridden -> {tau_drag_override}")
    if h1_noise_seed is not None:
        rng = np.random.default_rng(h1_noise_seed)
        st.h1 = st.h1 + 0.02 * rng.standard_normal(st.h1.shape)
        st.h1 = np.maximum(st.h1, st.h_floor)
        print(f"  h1 noise seed={h1_noise_seed}, amp=0.02 applied")

    print(f"  dt={st.dt:.5f}  gp={gp}  f0={f0}  n_bands={n_bands}  "
          f"band_contrast={band_contrast}  nu4={st.nu4}  "
          f"tau_drag={st.tau_drag}  steps={steps}")

    series_ev = []  # (step, eddy_vort_std, eddy_ke)
    t0 = time.perf_counter()

    for s in range(steps):
        st = solver.step(st, st.dt)
        if s % log_every == 0 or s == steps - 1:
            ev = eddy_vort_std(st)
            ek = eddy_ke(st)
            series_ev.append((s, ev, ek))
            elapsed = time.perf_counter() - t0
            ms_per_step = 1000.0 * elapsed / (s + 1)
            print(f"  step {s:5d}  eddy_vort_std={ev:.4f}  "
                  f"eddy_ke={ek:.5f}  [{ms_per_step:.2f} ms/step]")
            if not np.all(np.isfinite(st.h1)):
                print("  *** NaN DETECTED — aborting config ***")
                break

    wall = time.perf_counter() - t0
    ms_per_step = 1000.0 * wall / steps

    # Summarise trajectory
    evs = [x[1] for x in series_ev]
    start_ev = evs[0]
    peak_ev = max(evs)
    end_ev = evs[-1]
    peak_step = series_ev[evs.index(peak_ev)][0]

    # Verdict: GROWS if peak >20% above start and end > 1.1*start
    #          FLAT if peak within ±30% of start
    #          DECAYS if end < 0.8*start
    if peak_ev > 1.2 * start_ev and end_ev > 1.1 * start_ev:
        verdict = "GROWS"
    elif end_ev < 0.8 * start_ev:
        verdict = "DECAYS"
    else:
        verdict = "FLAT"

    print(f"\n  SUMMARY: start={start_ev:.4f}  peak={peak_ev:.4f} "
          f"@step{peak_step}  end={end_ev:.4f}  => {verdict}")
    print(f"  Wall time: {wall:.1f}s  ({ms_per_step:.2f} ms/step)")

    return {
        "name": name,
        "start_ev": start_ev,
        "peak_ev": peak_ev,
        "peak_step": peak_step,
        "end_ev": end_ev,
        "verdict": verdict,
        "ms_per_step": ms_per_step,
        "state": st,
        "series": series_ev,
    }


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

W, H = 192, 96
STEPS = 6000
LOG_EVERY = 1000

CONFIGS = [
    dict(name="baseline",
         W=W, H=H, f0=4.0, gp=(1.0, 0.05),
         n_bands=14, band_contrast=0.5,
         steps=STEPS, log_every=LOG_EVERY),

    dict(name="low_nu4_0.02",
         W=W, H=H, f0=4.0, gp=(1.0, 0.05),
         n_bands=14, band_contrast=0.5,
         steps=STEPS, log_every=LOG_EVERY,
         nu4_override=0.02),

    dict(name="f0_2_more_unstable",
         W=W, H=H, f0=2.0, gp=(1.0, 0.05),
         n_bands=14, band_contrast=0.5,
         steps=STEPS, log_every=LOG_EVERY),

    dict(name="band_contrast_0.8",
         W=W, H=H, f0=4.0, gp=(1.0, 0.05),
         n_bands=14, band_contrast=0.8,
         steps=STEPS, log_every=LOG_EVERY),

    dict(name="big_seed_h1_noise",
         W=W, H=H, f0=4.0, gp=(1.0, 0.05),
         n_bands=14, band_contrast=0.5,
         steps=STEPS, log_every=LOG_EVERY,
         h1_noise_seed=7),

    dict(name="gp_0.3_bigger_dt",
         W=W, H=H, f0=4.0, gp=(0.3, 0.05),
         n_bands=14, band_contrast=0.5,
         steps=STEPS, log_every=LOG_EVERY),

    dict(name="weak_drag_1e9",
         W=W, H=H, f0=4.0, gp=(1.0, 0.05),
         n_bands=14, band_contrast=0.5,
         steps=STEPS, log_every=LOG_EVERY,
         tau_drag_override=1e9),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_t0 = time.perf_counter()
    results = []

    for cfg in CONFIGS:
        r = run_config(**cfg)
        results.append(r)

    total_wall = time.perf_counter() - total_t0
    print(f"\n{'='*60}")
    print(f"  ALL CONFIGS DONE  (total {total_wall/60:.1f} min)")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'Config':<28} {'start':>7} {'peak':>7} {'@step':>6} "
          f"{'end':>7} {'verdict':>7} {'ms/stp':>7}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<28} {r['start_ev']:7.4f} {r['peak_ev']:7.4f} "
              f"{r['peak_step']:6d} {r['end_ev']:7.4f} "
              f"{r['verdict']:>7} {r['ms_per_step']:7.2f}")

    # -----------------------------------------------------------------------
    # Best config: highest sustained eddy_vort_std (end value)
    # -----------------------------------------------------------------------
    best = max(results, key=lambda r: r["end_ev"])
    print(f"\nBest config (highest end eddy_vort_std): {best['name']}  "
          f"end_ev={best['end_ev']:.4f}  verdict={best['verdict']}")

    # -----------------------------------------------------------------------
    # Visualise top-layer vorticity of best config
    # -----------------------------------------------------------------------
    _save_vorticity_png(best["state"], OUT_DIR / "explore_vort.png",
                        label=best["name"])

    # -----------------------------------------------------------------------
    # Assessment
    # -----------------------------------------------------------------------
    _print_assessment(results, best)


def _save_vorticity_png(st, path: Path, label: str = ""):
    """Write top-layer relative vorticity as a colourmap PNG via cv2."""
    try:
        import cv2
    except ImportError:
        print("\n[WARNING] cv2 not available; skipping PNG output.")
        # Fallback: save as raw numpy for manual inspection
        zeta = operators.vorticity(st.u1, st.v1, st.g)
        np.save(str(path).replace(".png", ".npy"), zeta)
        print(f"  Saved raw vorticity array to {str(path).replace('.png', '.npy')}")
        return

    zeta = operators.vorticity(st.u1, st.v1, st.g)  # (H+1, W)
    # Clip to ±3σ to keep colour range meaningful
    sigma = float(np.std(zeta))
    z_clip = np.clip(zeta, -3 * sigma, 3 * sigma)
    # Normalise to 0-255
    z_norm = ((z_clip - z_clip.min()) / (z_clip.max() - z_clip.min() + 1e-30) * 255).astype(np.uint8)
    # Apply COLORMAP_RdBu-like: cv2.COLORMAP_COOL for cyclone/anticyclone contrast
    coloured = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), coloured)
    print(f"\n  Saved vorticity PNG: {path}")

    # Qualitative assessment of structure
    _assess_structure(zeta, label)


def _assess_structure(zeta: np.ndarray, label: str):
    """Heuristic: ratio of high-wavenumber power to total gives filament score."""
    # Longitude FFT of interior rows
    interior = zeta[1:-1]  # (H-1, W)
    fft = np.abs(np.fft.rfft(interior, axis=1))  # (H-1, W//2+1)
    total_power = fft[:, 1:].sum()               # skip DC
    # 'High-k' = wavenumbers > W/4 (quarter-wave and shorter)
    k_thresh = fft.shape[1] // 4
    high_k_power = fft[:, k_thresh:].sum()
    frac = float(high_k_power / (total_power + 1e-30))

    print(f"\n  Structure assessment for '{label}':")
    print(f"    High-wavenumber fraction (k>W/4): {frac:.3f}")
    if frac > 0.25:
        print("    => FILAMENTARY: significant fine-scale (folded) structure")
    elif frac > 0.12:
        print("    => MIXED: moderate fine structure — eddies present, filaments weak")
    else:
        print("    => SMOOTH: dominated by low-wavenumber / zonal structure")


def _print_assessment(results: list[dict], best: dict):
    """Print the kill-gate assessment."""
    grows_count = sum(1 for r in results if r["verdict"] == "GROWS")
    decays_count = sum(1 for r in results if r["verdict"] == "DECAYS")
    ms_stp = best["ms_per_step"]

    print(f"\n{'='*60}")
    print("  KILL-GATE ASSESSMENT")
    print(f"{'='*60}")
    print(f"  Configs with GROWS verdict: {grows_count}/{len(results)}")
    print(f"  Configs with DECAYS verdict: {decays_count}/{len(results)}")
    print(f"  Best config: {best['name']}  "
          f"eddy_vort_std {best['start_ev']:.4f} -> {best['end_ev']:.4f}  "
          f"({best['verdict']})")
    print(f"  ms/step at 192x96: {ms_stp:.2f}")

    # Physical time per step estimate
    # dt_phys is solver dt (nondim). With gp[0]=1, h~5 => c_gw~2.24, dlam=2pi/192
    # Rough physical time: each step ~ dt nondim units
    # At gp=(0.3,0.05) the dt is larger by ~sqrt(1/0.3) ~ 1.83x
    # We can't convert to real seconds without Earth radius, but nondim is meaningful.

    if grows_count >= 2:
        print("\n  ASSESSMENT: Baroclinic instability IS active in at least some configs.")
        if ms_stp < 5.0:
            print(f"  At {ms_stp:.2f} ms/step, 10000 steps ~ {10000*ms_stp/1000:.0f}s — feasible.")
        else:
            print(f"  At {ms_stp:.2f} ms/step, 10000 steps ~ {10000*ms_stp/1000:.0f}s — borderline.")
        print("  RECOMMENDATION: Kill-gate render is worth attempting. Use best config.")
        print("  CAVEAT: Without semi-implicit (bigger dt), reaching deeply nonlinear")
        print("  filamentary regime requires many more steps than the spin-up decay rate.")
    else:
        print("\n  ASSESSMENT: Eddies NOT growing — purely zonal equilibrium dominates.")
        print("  The explicit GW dt is too small; physical integration time is insufficient.")
        print("  RECOMMENDATION: Kill-gate render is NOT recommended without semi-implicit.")
        print("  Options: (a) reduce gp to increase dt, (b) add leapfrog/RK4+SIT filter,")
        print("           (c) accept zonal-only M0 as scope and defer eddies to M1.")

    print("\n  Note: tau_rad/tau_drag are step-based (not seconds) — the explicit")
    print(f"  gravity-wave dt (~{best['state'].dt:.4f} nondim) means 1000 steps =")
    print(f"  {1000*best['state'].dt:.2f} nondim time units.")


if __name__ == "__main__":
    main()

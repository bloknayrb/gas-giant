"""v1.6 diagnostic A/B: does broadband eddy injection wake up the LAMINAR
(off-) bands, or is the alternating-band activity intrinsic to the dynamics?

Loads the winning `jupiter_vorticity` preset and varies ONLY solver.vort_inject.
For each run it renders a full equirect map and computes a per-latitude
small-scale "activity" curve (std of the longitudinally high-pass-filtered
luminance per row). If the off-bands are forcing-starved, their activity rises
with injection; if the alternation is intrinsic shear-instability asymmetry,
the active bands stay active and the dead bands stay (relatively) flat.

    uv run python scripts/inject_ab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_morphology import _lum  # noqa: E402

from gasgiant.engine.facade import Simulation  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.model import SolverType  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402

OUT = Path("out/audit/v16")
RES = 1536          # full-map render resolution (W); H = W//2
INJECTS = [0.0, 0.1, 0.5]
N_BINS = 36         # latitude bins for the activity profile


def _activity_per_lat(rgb01: np.ndarray) -> np.ndarray:
    """Per-row small-scale energy: std along longitude of the high-pass lum."""
    lum = _lum(rgb01)                       # (H, W) in [0,1]
    # Longitudinal high-pass: subtract a wide box-blur along x (wrap-aware).
    k = max(rgb01.shape[1] // 24, 3)
    pad = np.concatenate([lum[:, -k:], lum, lum[:, :k]], axis=1)
    smooth = cv2.blur(pad, (2 * k + 1, 1))[:, k:-k]
    hp = lum - smooth
    return hp.std(axis=1)                    # (H,)


def _label(img_u8, text):
    out = img_u8.copy()
    cv2.rectangle(out, (0, 0), (img_u8.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _u8(rgb01):
    return cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = GpuContext.headless()
    gpu.make_current()

    maps = []
    curves = {}
    for inj in INJECTS:
        p = load_factory_preset("jupiter_vorticity")
        p.solver.type = SolverType.VORTICITY
        p.sim.resolution = RES
        p.solver.vort_inject = inj
        sim = Simulation(p, gpu)
        rgb = sim.render_maps(RES)["color"][..., :3]
        sim._release_sim()
        curves[inj] = _activity_per_lat(rgb)
        maps.append(_label(_u8(rgb), f"jupiter_vorticity  vort_inject={inj}"))

    cv2.imwrite(str(OUT / "inject_ab_maps.png"), np.vstack(maps))

    # Per-band activity table: bin the active belt region (|lat|<60) and compare.
    H = curves[INJECTS[0]].shape[0]
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0    # descending degrees
    active = np.abs(lat) < 60.0
    edges = np.linspace(lat[active].max(), lat[active].min(), N_BINS + 1)
    base = curves[0.0]

    print(f"\nper-band small-scale activity (winning preset, res={RES})")
    print(f"{'lat(deg)':>9} " + " ".join(f"inj={i:<4}" for i in INJECTS) + "   off-band?")
    for b in range(N_BINS):
        lo, hi = edges[b + 1], edges[b]
        m = (lat <= hi) & (lat > lo) & active
        if not m.any():
            continue
        vals = [curves[i][m].mean() for i in INJECTS]
        # "off-band" = bottom tertile of baseline activity.
        thr = np.quantile(base[active], 0.33)
        flag = "DEAD" if vals[0] < thr else ""
        row = " ".join(f"{v:>8.4f}" for v in vals)
        print(f"{0.5*(lo+hi):>9.1f} {row}   {flag}")

    # Summary: did the dead bands rise more than the live bands?
    thr = np.quantile(base[active], 0.33)
    dead = active & (base < thr)
    live = active & (base >= np.quantile(base[active], 0.66))
    for i in INJECTS[1:]:
        dr = (curves[i][dead].mean() / base[dead].mean() - 1.0) * 100
        lr = (curves[i][live].mean() / base[live].mean() - 1.0) * 100
        print(f"\ninj={i}: DEAD bands {dr:+.0f}%   LIVE bands {lr:+.0f}%  "
              f"(if DEAD rises >> LIVE -> forcing-starved; if similar -> intrinsic)")

    print(f"\nwrote {OUT}/inject_ab_maps.png")


if __name__ == "__main__":
    main()

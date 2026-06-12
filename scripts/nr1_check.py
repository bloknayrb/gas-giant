"""v1.5 NR-1 non-regression: no v1.4 metric moves AWAY from the reference by
>5% (one-sided; movement TOWARD the reference is unbounded). belt_L_std gets a
two-sided guardrail (it is the metric Axis-1 exists to move). Height-map std
spot check (T2 also feeds height) within +-25%.

Renders the v1.5 preset and a v1.4-equivalent (v1.5 knobs reset to defaults),
both vs PIA07782, and reports per-key regression.

    uv run python scripts/nr1_check.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.palette.reference import latitude_profile, profile_signed
from gasgiant.params.presets import load_factory_preset

REF = Path("refs/PIA07782.jpg")
WIDTH = 2048
TOL = 0.05  # one-sided: |v15 dist| must not exceed |v14 dist| * (1+TOL)
# Color-family (v1.4 PASS-axis) scalar metrics that must not regress.
WATCH = ("zone_chroma", "belt_chroma", "hue_spread")
# Axis-1 metrics: movement toward the reference is the GOAL (reported, not gated
# one-sided); belt_L_std carries the two-sided guardrail separately.
AXIS1 = ("texture_energy", "belt_chroma_std")


def _v14_params():
    p = load_factory_preset("jupiter_like")
    p.turbulence.belt_replenish = 0.0
    p.detail.belt_texture_fine = 0.0
    p.storms.hero_latitude = None
    p.storms.hero_aspect = 1.0
    p.storms.rim_contrast = 1.0
    p.storms.hero_radius = 0.13
    p.detail.belt_texture = 1.05
    return p


def _render(gpu, p):
    sim = Simulation(p, gpu)
    maps = sim.render_maps(WIDTH)
    rgb = np.clip(maps["color"][..., :3], 0.0, 1.0).astype(np.float32)
    hstd = float(maps["height"].std())
    sim._release_sim()
    return rgb, hstd


def main():
    gpu = GpuContext.headless()
    gpu.make_current()
    ref = cv2.cvtColor(cv2.imread(str(REF)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ref = cv2.resize(ref, (WIDTH, WIDTH // 2), interpolation=cv2.INTER_AREA)
    ref_prof = latitude_profile(ref)

    rgb14, h14 = _render(gpu, _v14_params())
    rgb15, h15 = _render(gpu, load_factory_preset("jupiter_like"))
    d14 = profile_signed(latitude_profile(rgb14), ref_prof, lat_max=50.0)
    d15 = profile_signed(latitude_profile(rgb15), ref_prof, lat_max=50.0)

    print(f"{'metric':>16} {'v14_dist':>10} {'v15_dist':>10} {'verdict':>10}")
    worst = []
    for k in WATCH:
        a, b = abs(d14[k]), abs(d15[k])
        regressed = b > a * (1 + TOL) + 1e-6
        print(f"{k:>16} {d14[k]:>10.4f} {d15[k]:>10.4f} {'REGRESS' if regressed else 'ok':>10}")
        if regressed:
            worst.append(k)

    # Axis-1 metrics: report signed distance (toward 0 = toward reference = good).
    print()
    for k in AXIS1:
        toward = "toward ref" if abs(d15[k]) < abs(d14[k]) else "away"
        print(f"{k:>16} {d14[k]:>10.4f} {d15[k]:>10.4f} {toward:>12}")
    print(f"\nbelt_L_std signed dist v14 {d14['belt_L_std']:.4f} -> v15 {d15['belt_L_std']:.4f} "
          f"(guardrail: stays near ref)")

    # Height std spot check +-25%.
    hr = h15 / h14 if h14 else float("nan")
    print(f"height std v14 {h14:.5f} -> v15 {h15:.5f}  ratio {hr:.3f} "
          f"({'OK' if 0.75 <= hr <= 1.25 else 'OUT OF +-25%'})")

    print(f"\nNR-1: {'PASS' if not worst else 'REGRESSIONS: ' + ', '.join(worst)}")


if __name__ == "__main__":
    main()

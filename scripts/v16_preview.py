"""v1.6 visual preview: belt-crop comparison (reference | v1.5 kinematic |
vorticity) + a full vorticity equirect map. Honest snapshot of the current
(clamp-leaning) state. Writes PNGs to out/audit/v16/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_morphology import _belt_crop_from_rgb, _lum, coher  # noqa: E402

from gasgiant.engine.facade import Simulation  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.model import SolverType  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402

REF = Path("refs/PIA07782.jpg")
OUT = Path("out/audit/v16")


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

    # v1.5 kinematic (default).
    p = load_factory_preset("jupiter_like")
    sim = Simulation(p, gpu)
    rgb_kin = sim.render_maps(8192)["color"]
    kin_crop, box = _belt_crop_from_rgb(rgb_kin, sim, 640)
    ck = coher(_lum(kin_crop))
    sim._release_sim()

    # Vorticity (current fold-friendly config; honest = the one that shows folds).
    p = load_factory_preset("jupiter_like")
    p.solver.type = SolverType.VORTICITY
    p.sim.dev_steps = 600
    p.turbulence.relax_tau = 2000.0
    p.turbulence.replenish_rate = 0.0
    p.solver.vort_relax_tau = 250.0
    p.solver.vort_hypervisc = 1.0
    p.solver.coriolis_f0 = 6.0
    p.solver.vort_inject = 0.0
    sim = Simulation(p, gpu)
    rgb_vort_full = sim.render_maps(2048)["color"]
    rgb_vort = sim.render_maps(8192)["color"]
    vort_crop, _ = _belt_crop_from_rgb(rgb_vort, sim, 640)
    cv = coher(_lum(vort_crop))
    sim._release_sim()

    # Reference belt crop, same box.
    ref = cv2.cvtColor(cv2.imread(str(REF)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    from measure_morphology import _crop_deg, _fit_width
    ref_crop = _fit_width(_crop_deg(ref, *box), 640)

    # Stack belt crops with labels.
    strip = np.vstack([
        _label(_u8(ref_crop), "REFERENCE (PIA07782)  coher=0.62"),
        _label(_u8(kin_crop), f"v1.5 KINEMATIC  coher={ck:.3f}"),
        _label(_u8(vort_crop), f"v1.6 VORTICITY  coher={cv:.3f}  (belt clean; ~5% polar clamp)"),
    ])
    cv2.imwrite(str(OUT / "belt_compare.png"), strip)

    # Full vorticity map (downscaled).
    full = _u8(rgb_vort_full[..., :3])
    full = cv2.resize(full, (1600, 800), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "vorticity_fullmap.png"), _label(full, "v1.6 VORTICITY full equirect"))

    print(f"v1.5 coher={ck:.4f}  vorticity coher={cv:.4f}")
    print(f"wrote {OUT}/belt_compare.png and {OUT}/vorticity_fullmap.png")


if __name__ == "__main__":
    main()

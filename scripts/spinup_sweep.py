"""v1.6 P4 spin-up characterization: how the folded-filament morphology (coher)
develops vs dev_steps and vort_relax_tau in vorticity mode. The vorticity field
starts as the smooth analytic target and must DEVELOP folds by advection, so the
dev_steps the proof render needs is not the kinematic 500 — it must be measured.
Reports coher (the P4.5 gate metric) on the matched belt crop per config, plus
wall time. The v1.5 isotropic baseline is coher~0.14; the reference ~0.62;
pre-registered MD-2 bar coher>=0.30.

    uv run python scripts/spinup_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_morphology import _belt_crop_from_rgb, _lum, coher  # noqa: E402

from gasgiant.engine.facade import Simulation  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.model import SolverType  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402

# Diagnostic configs: does the vorticity VELOCITY fold the tracers once the
# v1.5 re-smoothing forcing is relaxed? Each dict overrides preset fields.
# label -> overrides (dotted onto the params tree).
# Vary the ACTUAL morphology knobs: solver.vort_relax_tau (LONG = let eddies
# fold before re-pinning) and solver.coriolis_f0 (sets the PV-gradient / whether
# jets are barotropically unstable and shed folds). Tracer forcing reduced
# (weak relax + no isotropic replenish) so a folding velocity can imprint.
# Eddy-injection amplitude sweep in a fold-friendly regime (long vort_relax_tau,
# low hypervisc, weak tracer re-smoothing so the folding velocity imprints).
_BASE = {"sim.dev_steps": 800, "turbulence.relax_tau": 2000.0,
         "turbulence.replenish_rate": 0.0, "solver.vort_relax_tau": 800.0,
         "solver.vort_hypervisc": 0.3, "solver.coriolis_f0": 6.0,
         "solver.vort_inject_scale": 0.5}
CONFIGS = [
    ("inject_0.0", {**_BASE, "solver.vort_inject": 0.0}),
    ("inject_0.5", {**_BASE, "solver.vort_inject": 0.5}),
    ("inject_1.5", {**_BASE, "solver.vort_inject": 1.5}),
    ("inject_3.0", {**_BASE, "solver.vort_inject": 3.0}),
]


def _apply(p, overrides):
    for dotted, val in overrides.items():
        obj = p
        parts = dotted.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)


def main():
    gpu = GpuContext.headless()
    gpu.make_current()
    print(f"{'label':>14} {'coher':>8} {'sec':>7}")
    for label, overrides in CONFIGS:
        p = load_factory_preset("jupiter_like")
        p.solver.type = SolverType.VORTICITY
        _apply(p, overrides)
        t0 = time.perf_counter()
        sim = Simulation(p, gpu)
        rgb = sim.render_maps(8192)["color"]
        crop, _ = _belt_crop_from_rgb(rgb, sim, 640)
        c = coher(_lum(crop))
        dt = time.perf_counter() - t0
        sim._release_sim()
        print(f"{label:>14} {c:>8.4f} {dt:>7.1f}")


if __name__ == "__main__":
    main()

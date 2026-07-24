"""Emergence-ON render-hash gate: the variant p05 cannot see.

`scripts/p05_baseline_hash.py` hashes GPU output for DEFAULT-program configs only
(model defaults + jupiter_like, both `hero_emergence == 0`), and
`tests/unit/test_kinematic_kernels_pinned.py` hashes shader FILE TEXT. Neither
can see a numeric drift introduced INSIDE a preprocessor variant: every line of
the hero-emergence anatomy, `heroRelaxWeight` and the per-storm CastLevers reads
sits behind `#ifdef HERO_EMERGENCE` / `#ifdef CAST_LEVERS`, so p05 passing is
guaranteed by construction for any edit confined to those arms.

This is that gate. It captures SHA1s of developed float32 tracers for a set of
emergence-ON KINEMATIC configs -- kinematic because only that path is byte-exact
(CLAUDE.md: the vorticity path carries ~1e-3/~0.004 SOR noise, so a stored
cross-process hash there would be flake, not a gate). Like p05, the baseline is
MACHINE-LOCAL: capture it before your edit, re-check after.

    uv run python scripts/m2b_emergence_hash.py            # write/print baseline
    uv run python scripts/m2b_emergence_hash.py --check    # assert vs baseline

The first run writes out/audit/m2b_emergence_hash.json if absent; --check asserts.

`two_heroes` is the load-bearing config: it is the only one where
heroRelaxWeight's cross-hero max() actually CONTENDS (measured ~5.9k px with two
heroes inside q<=4.2 and ~2.2k inside both wake windows). With a single hero the
max never contends and a restructure of it is trivially safe.

The omega side is deliberately NOT here: vorticity output cannot carry a stored
byte-exact hash, so its equivalence is asserted same-process in
tests/gpu/test_cast_levers.py against the dev-0 omega texture (CLAUDE.md's one
byte-exact carve-out for vorticity mode).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gasgiant.engine import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import PlanetParams, SolverType, WakeDir

BASELINE = Path("out/audit/m2b_emergence_hash.json")


def _p(**storms) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 60
    p.storms.hero_count = 1
    p.storms.hero_latitude = -22.5
    p.storms.hero_emergence = 0.9
    for k, v in storms.items():
        setattr(p.storms, k, v)
    assert p.solver.type == SolverType.KINEMATIC
    return p


CONFIGS = {
    # bare emergence: anatomy + heroRelaxWeight rim/flush, no shape/taper
    "bare": lambda: _p(),
    # shape + taper: exercises the twr erosion hold + wedge flush + deflect
    "shape_taper": lambda: _p(hero_shape=1.0, hero_taper=0.8),
    # rim levers on top (the M2-A hoisted set) + wake direction pinned
    "rim_wake": lambda: _p(hero_rim_tint=0.7, hero_rim_warp=0.5,
                           hero_wake_dir=WakeDir.WEST, hero_wake_detail=0.6),
    # TWO heroes: the ONLY config where heroRelaxWeight's cross-hero max()
    # combine actually contends -- the site M2-B restructures.
    "two_heroes": lambda: _p(hero_count=2, hero_shape=1.0, hero_taper=0.8),
    # aspect + mottle/tint_var: the rest of the per-hero anatomy block
    "aspect_interior": lambda: _p(hero_aspect=2.2, hero_mottle=0.7,
                                  hero_tint_var=0.6, hero_shape=0.8),
}


def _hash(p: PlanetParams, gpu) -> str:
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    arr = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)
    return hashlib.sha1(np.ascontiguousarray(arr, np.float32).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    gpu = GpuContext.headless()
    out = {name: _hash(mk(), gpu) for name, mk in CONFIGS.items()}
    for k, v in out.items():
        print(f"{k}: {v}")

    if args.check:
        old = json.loads(BASELINE.read_text(encoding="utf-8"))
        bad = [k for k in out if old.get(k) != out[k]]
        if bad:
            raise SystemExit(f"MISMATCH in {bad}\n  was {[old.get(k) for k in bad]}")
        print(f"OK: all {len(out)} emergence hashes match baseline.")
    else:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {BASELINE}")


if __name__ == "__main__":
    main()

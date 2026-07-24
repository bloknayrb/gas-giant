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

`render_*` configs hash render_maps() with detail ON, covering detail.comp's own
HERO_EMERGENCE arm — the quiet-storm remap and the collar/spiral/wake windows,
which the tracer configs never touch and which are LIVE in the flagship preset.

The omega side is deliberately NOT here: vorticity output cannot carry a stored
byte-exact hash, so its equivalence is asserted same-process in
tests/gpu/test_cast_levers.py against the dev-0 omega texture (CLAUDE.md's one
byte-exact carve-out for vorticity mode) — and for the STEPPED path, which that
carve-out cannot reach, by scripts/m2c_omega_equiv.py's two-tree .npy diff.
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
from gasgiant.render.detail import detail_fx_enabled

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


_RENDER_DETAIL = 0.6   # > 0 so detail.comp actually runs
_RENDER_WIDTH = 512
# detail.comp's hero-emergence read sites are split across its variants: the two
# cross-hero ones are in the BASE HERO_EMERGENCE path, but the spiral-pitch,
# spiral-window and collar-window ones are inside the DETAIL_FX-only block. With
# every fx pfield at its 0 default, DETAIL_FX does not compile and those three
# sites are not in the program at all -- a hash that leaves them off is green by
# construction, the same blindness p05 has toward HERO_EMERGENCE. So drive the
# fx levers that own them.
_RENDER_FX = {"hero_spiral": 0.8, "hero_collar_wrap": 0.7, "hero_wake_braid": 0.6,
              "intermittency": 0.5}

# The RENDER-side companions (rationale in the module docstring): deliberately the
# SAME scene builders as the tracer configs, not copies -- only the hashed artifact
# differs, and a copy would let a scene retune leave the two families describing
# different planets. Add a render-only scene as an explicit extra entry rather than
# forking one of these three.
RENDER_SCENES = ("bare", "shape_taper", "two_heroes")
RENDER_CONFIGS = {f"render_{name}": CONFIGS[name] for name in RENDER_SCENES}


def _hash(p: PlanetParams, gpu) -> str:
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    arr = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)
    return hashlib.sha1(np.ascontiguousarray(arr, np.float32).tobytes()).hexdigest()


def _render_hash(p: PlanetParams, gpu) -> str:
    p.detail.intensity = _RENDER_DETAIL
    for name, value in _RENDER_FX.items():
        setattr(p.detail, name, value)
    assert detail_fx_enabled(p.detail), \
        "DETAIL_FX must compile or the fx-only sites are unhashed"
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    maps = sim.render_maps(_RENDER_WIDTH)
    h = hashlib.sha1()
    for key in sorted(maps):
        h.update(key.encode())
        h.update(np.ascontiguousarray(maps[key], np.float32).tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--expect-moved", nargs="*", default=[], metavar="KEY",
                    help="config(s) this change is SUPPOSED to move; --check then "
                         "passes only if they move and nothing else does")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing baseline (re-baseline everything)")
    args = ap.parse_args()

    gpu = GpuContext.headless()
    out = {name: _hash(mk(), gpu) for name, mk in CONFIGS.items()}
    out.update({name: _render_hash(mk(), gpu) for name, mk in RENDER_CONFIGS.items()})
    for k, v in out.items():
        print(f"{k}: {v}")

    if args.check:
        old = json.loads(BASELINE.read_text(encoding="utf-8"))
        missing = [k for k in out if k not in old]
        if missing:
            raise SystemExit(
                f"NO BASELINE for {missing}: these configs are newer than the stored "
                f"baseline. Re-capture (--force) once the shared configs are green -- "
                f"this is NOT a numeric regression."
            )
        bad = [k for k in out if old[k] != out[k] and k not in args.expect_moved]
        stale = [k for k in args.expect_moved if k in old and old[k] == out[k]]
        if bad:
            raise SystemExit(f"MISMATCH in {bad}\n  was {[old[k] for k in bad]}")
        if stale:
            raise SystemExit(
                f"--expect-moved named {stale}, but they did NOT move. Either the "
                f"change did not do what you think, or the flag is stale."
            )
        moved = [k for k in args.expect_moved if k in old]
        print(f"OK: all {len(out) - len(moved)} emergence hashes match baseline"
              + (f" ({moved} moved as declared)." if moved else "."))
    else:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        # p05's guard, for the reason M2-C ran into: --check is all-or-nothing, so
        # a DELIBERATE mover (render_two_heroes) makes it exit non-zero, and the
        # natural remedy -- rerun without --check -- silently overwrote every OTHER
        # key too, destroying the very baseline that proves the unmoved ones did
        # not move. Acknowledge an expected mover with --expect-moved instead; only
        # --force overwrites, and then deliberately.
        if BASELINE.exists() and not args.force:
            raise SystemExit(
                f"{BASELINE} exists -- refusing to overwrite (that would erase the "
                f"evidence for every UNCHANGED config). Use --check, or --check "
                f"--expect-moved KEY to acknowledge a deliberate mover, or --force "
                f"if you really mean to re-baseline all {len(out)}."
            )
        BASELINE.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {BASELINE}")


if __name__ == "__main__":
    main()

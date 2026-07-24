"""Old-vs-new equivalence probe for the omega_force hero sites — the gate no
other tool provides.

`scripts/p05_baseline_hash.py` is KINEMATIC, so it never runs omega_force at
all. `scripts/m2b_emergence_hash.py` is kinematic for the same reason CLAUDE.md
gives: the vorticity path's SOR solve carries ~1e-3/~0.004 LSB noise, so a
stored cross-process hash there would be flake rather than a gate. And the one
byte-exact carve-out for vorticity — the dev-0 omega texture — is read back
BEFORE any force pass, so it cannot see omega_force either.

That leaves the hero anchor (the 60x nudge boost) and the wake eddy injection
with no automated protection whatsoever. This script supplies it differently:
instead of storing a hash, it captures the developed state to .npy so the SAME
config can be run on two trees and diffed. Cross-process reproducibility is not
assumed — capture a SAME-TREE control and judge against its floor. On a fresh
session this box reproduces exactly (floor 0); after enough GL work in one
session, CLAUDE.md's documented LSB noise appears (observed: q exactly 0, tr
2.1e-06 on 8% of pixels) and an absolute-zero verdict would call that DIVERGED.

    uv run python scripts/m2c_omega_equiv.py out/audit/omega_new
    uv run python scripts/m2c_omega_equiv.py out/audit/omega_ctl   # control
    git checkout <pre-change-rev> -- src/gasgiant/sim/kernels/vortex_omega.glsl \
                                     src/gasgiant/sim/kernels/omega_force.comp
    uv run python scripts/m2c_omega_equiv.py out/audit/omega_old
    git checkout HEAD -- src/gasgiant/sim/kernels/vortex_omega.glsl \
                         src/gasgiant/sim/kernels/omega_force.comp
    uv run python scripts/m2c_omega_equiv.py out/audit/omega_old \
                     --diff out/audit/omega_new --control out/audit/omega_ctl

(Commit or copy first: `git checkout --` discards UNCOMMITTED edits to those
files, which has already cost one round of work here.)

It earned its place by measuring the M2-C anchor restructure at 1 ULP after one
step, amplified past GPU_NOISE_ATOL by step 40 — the reason the per-storm form
rides CAST_LEVERS instead of replacing the legacy lines. Full derivation at
vortex_omega.glsl::heroAnchorBoost.

Both hero sites must be live or the comparison is vacuous: hero_emergence > 0
drives the anchor, wake_turbulence > 0 drives the injection. Asserted below.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gasgiant.engine import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import PlanetParams, SolverType

STEPS = 40
RESOLUTION = 512


def build(steps: int = STEPS) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.solver.type = SolverType.VORTICITY   # kinematic never reaches omega_force
    p.sim.resolution = RESOLUTION
    p.sim.dev_steps = steps
    p.storms.hero_count = 1
    p.storms.hero_latitude = -22.5
    p.storms.hero_emergence = 0.9          # the anchor site
    p.storms.hero_solid_core = 0.8         # ring branch, so the anchor has a body
    p.storms.wake_turbulence = 0.7         # -> u_hero_wake_turb, the injection site
    assert p.storms.hero_emergence > 0.0, "anchor site would not run"
    assert p.storms.wake_turbulence > 0.0, "wake-injection site would not run"
    return p


def capture(out: Path, steps: int) -> None:
    gpu = GpuContext.headless()
    sim = Simulation(build(steps), gpu)
    sim.run_to_completion(chunk=20)
    q = np.squeeze(np.asarray(gpu.read_texture(sim.solver._omega_state.cur)))
    tr = np.asarray(gpu.read_texture(sim.solver.equirect.tracers.cur))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{out}_q.npy", np.ascontiguousarray(q, np.float32))
    np.save(f"{out}_tr.npy", np.ascontiguousarray(tr, np.float32))
    print(f"{out}: q absmax={np.abs(q).max():.6f} mean={q.mean():.6f}")


def _maxdiff(a: Path, b: Path) -> dict[str, float]:
    out = {}
    for tag in ("q", "tr"):
        x = np.load(f"{a}_{tag}.npy").astype(np.float64)
        y = np.load(f"{b}_{tag}.npy").astype(np.float64)
        out[tag] = float(np.abs(x - y).max())
    return out


def diff(old: Path, new: Path, control: Path | None = None) -> None:
    """Compare two captures, and judge against a same-tree CONTROL if given.

    The verdict is control-RELATIVE, not absolute, because the honest floor is
    not always 0. This box reproduces exactly cross-process on a fresh session
    (measured), but after enough GL work in one session CLAUDE.md's documented
    LSB noise appears -- observed here as q maxdiff 0 with tr maxdiff 2.1e-06 on
    8% of pixels, in BOTH the control and the cross-tree diff. Comparing to a
    hard 0 in that state reports DIVERGED for a change that contributes nothing,
    which is how a real regression later gets waved through as "just noise".
    So: capture the control, and require the cross-tree diff to be no worse."""
    worst = 0.0
    for tag in ("q", "tr"):
        a = np.load(f"{old}_{tag}.npy").astype(np.float64)
        b = np.load(f"{new}_{tag}.npy").astype(np.float64)
        d = np.abs(a - b)
        worst = max(worst, float(d.max()))
        print(f"{tag}: maxdiff={d.max():.6g} mean={d.mean():.6g} "
              f"nonzero={(d > 0).mean():.4%}")

    if control is None:
        if worst == 0.0:
            print("EQUIVALENT (bit-exact)")
        else:
            print(f"DIVERGED: {worst:.6g} -- but with NO --control this cannot "
                  f"distinguish the change from session LSB noise. Re-run with "
                  f"a same-tree control capture before believing it.")
        return

    floor = max(_maxdiff(new, control).values())
    print(f"same-tree control floor: {floor:.6g}")
    if worst <= floor:
        print("EQUIVALENT" + (" (bit-exact)" if worst == 0.0 else
                              " (at or below the control floor)"))
    else:
        print(f"DIVERGED: {worst:.6g} exceeds the control floor {floor:.6g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", type=Path, help="path prefix for the .npy pair")
    ap.add_argument("--diff", type=Path, metavar="OTHER_PREFIX",
                    help="compare PREFIX against OTHER_PREFIX instead of capturing")
    ap.add_argument("--control", type=Path, metavar="CONTROL_PREFIX",
                    help="a SAME-TREE second capture of OTHER_PREFIX; the verdict "
                         "is judged against its floor instead of against 0")
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    if args.diff is not None:
        diff(args.prefix, args.diff, args.control)
    else:
        capture(args.prefix, args.steps)


if __name__ == "__main__":
    main()

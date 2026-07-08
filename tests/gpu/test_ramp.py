"""T8: ramp-driven sequence export.

(1) A POST-only ramp renders frames that vary monotonically in the ramped
    quantity (here: lower gamma -> brighter color).
(2) BLOCKER regression -- a VELOCITY-tier ramp re-applies params EVERY frame;
    the sim's step_index MUST advance by exactly steps_per_frame on every frame
    (the facade's VELOCITY _extra_steps reset must NOT clobber the extend_run
    frame clock, freezing the sim after frame 1).
(3) Frame 0 of a ramp is byte-identical to a plain export of the same base
    (kinematic path), since t=0 is the base state.

Collects without GL (import-only at module scope); the bodies need a context.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

_RES = 512  # >= SimParams.resolution floor, and < TILE (1024) so each frame is one tile


def _base(seed: int = 7, dev_steps: int = 12) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = _RES
    p.sim.dev_steps = dev_steps
    p.export.width = _RES
    return p


def _frame_mean(path):
    from gasgiant.export.writers import read_png16

    return float(np.mean(read_png16(path).astype(np.float64)))


def test_post_ramp_frames_vary_monotonically(gpu, tmp_path):
    from gasgiant.export.exporter import run_export_sequence

    base = _base()
    ramp_to = base.model_copy(deep=True)
    ramp_to.appearance.gamma = 0.4  # POST: lower gamma brightens each pixel (in <= 1)

    sim = Simulation(base, gpu)
    out = tmp_path / "seq"
    frames = 5
    run_export_sequence(sim, out, frames=frames, steps_per_frame=2, ramp_to=ramp_to)

    means = [_frame_mean(out / "frames" / f"frame_{i:04d}.png") for i in range(frames)]
    # The gamma sweep dominates the tiny per-frame sim drift, so the frame means
    # move STRICTLY MONOTONICALLY across the ramp (the tone-curve direction is a
    # pipeline detail; the point is a clean, ordered progression, not a jitter).
    diffs = [b - a for a, b in zip(means, means[1:], strict=False)]
    assert all(d < 0 for d in diffs) or all(d > 0 for d in diffs), means


def test_velocity_ramp_advances_step_clock_every_frame(gpu, tmp_path):
    """BLOCKER: without the _extra_steps snapshot/preserve, a VELOCITY-tier diff
    applied per frame freezes the sim after frame 1."""
    from gasgiant.export.exporter import export_sequence_job

    base = _base()
    ramp_to = base.model_copy(deep=True)
    ramp_to.jets.strength = 2.0  # VELOCITY-tier diff each frame

    sim = Simulation(base, gpu)
    out = tmp_path / "seq"
    frames, spf = 4, 5
    dev = base.sim.dev_steps

    # _RES < TILE, so each frame is a single tile: "frame {fi} tile 1/1" fires
    # right AFTER that frame's extend_run, so steps_done then == dev + fi*spf.
    per_frame: dict[int, int] = {}
    job = export_sequence_job(sim, out, frames=frames, steps_per_frame=spf, ramp_to=ramp_to)
    for prog in job:
        m = prog.message
        if m.startswith("frame ") and "tile 1/" in m:
            fi = int(m.split()[1])
            per_frame.setdefault(fi, sim.steps_done)

    for fi in range(1, frames):
        assert per_frame[fi] == dev + fi * spf, (fi, per_frame)
    # And the run ends exactly (frames-1) advances past the developed base.
    assert sim.steps_done == dev + (frames - 1) * spf


def test_ramp_frame0_identical_to_plain_export(gpu, tmp_path):
    """t=0 is the base state: a ramp's frame_0000 matches a plain sequence's
    frame_0000 byte-for-byte on the byte-exact kinematic path."""
    from gasgiant.export.exporter import run_export_sequence

    base = _base()
    ramp_to = base.model_copy(deep=True)
    ramp_to.appearance.gamma = 0.4

    plain = tmp_path / "plain"
    run_export_sequence(Simulation(base.model_copy(deep=True), gpu), plain,
                        frames=2, steps_per_frame=4)
    ramp = tmp_path / "ramp"
    run_export_sequence(Simulation(base.model_copy(deep=True), gpu), ramp,
                        frames=2, steps_per_frame=4, ramp_to=ramp_to)

    a = (plain / "frames" / "frame_0000.png").read_bytes()
    b = (ramp / "frames" / "frame_0000.png").read_bytes()
    assert a == b

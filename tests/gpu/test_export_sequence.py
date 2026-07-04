"""Sequence export: Simulation.extend_run + export_sequence_job.

Determinism scope (per testing policy): the kinematic path is byte-exact, so
the golden A/B test hash-compares kinematic sequence files. The vorticity
path carries SOR LSB noise that COMPOUNDS across frames, so vorticity gets
STRUCTURAL assertions only (frame count/naming, frames differ pairwise,
manifest schema-valid, cancellation cleanup) — never hash comparisons.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _kin_params(dev_steps: int = 20, width: int = 512) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = dev_steps
    p.export.width = width
    return p


def _vort_params() -> PlanetParams:
    p = _kin_params()
    p.solver.type = "vorticity"
    return p


# -- Simulation.extend_run ----------------------------------------------------


def test_extend_run_advances_exact_steps(gpu):
    p = _kin_params()
    sim = Simulation(p, gpu)
    sim.run_to_completion()
    assert sim.steps_done == p.sim.dev_steps
    sim.extend_run(7)
    assert sim.steps_done == p.sim.dev_steps + 7
    assert sim.is_developed


def test_extend_run_accumulates(gpu):
    p = _kin_params()
    sim = Simulation(p, gpu)
    sim.run_to_completion()
    sim.extend_run(3)
    sim.extend_run(4)
    assert sim.steps_done == p.sim.dev_steps + 7
    assert sim.steps_target == p.sim.dev_steps + 7


def test_extend_run_rejects_negative(gpu):
    sim = Simulation(_kin_params(), gpu)
    with pytest.raises(ValueError):
        sim.extend_run(-1)


def test_extend_run_zero_is_noop(gpu):
    p = _kin_params()
    sim = Simulation(p, gpu)
    sim.run_to_completion()
    sim.extend_run(0)
    assert sim.steps_done == p.sim.dev_steps

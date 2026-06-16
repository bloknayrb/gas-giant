"""Ship-path: driver lifecycle, off byte-identity, preview==export, seed determinism."""
from __future__ import annotations

import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu


def _baro_params(seed: int = 7, dev_steps: int = 40, warmup: int = 600,
                 enabled: bool = True) -> PlanetParams:
    p = load_factory_preset("jupiter_vorticity")
    p = p.model_copy(update={"seed": seed})
    p.sim.resolution = 512
    p.sim.dev_steps = dev_steps
    p.solver.type = SolverType.VORTICITY
    p.solver.baroclinic = p.solver.baroclinic.model_copy(update={
        "enabled": enabled, "gain": 0.5, "warmup_steps": warmup,
        "baro_steps_per_update": 60, "update_every": 16,
    })
    return p


def test_enabled_builds_driver(gpu):
    sim = Simulation(_baro_params(), gpu)
    try:
        assert sim._baro_driver is not None
    finally:
        sim._release_sim()


def test_disabled_has_no_driver(gpu):
    sim = Simulation(_baro_params(enabled=False), gpu)
    try:
        assert sim._baro_driver is None
    finally:
        sim._release_sim()

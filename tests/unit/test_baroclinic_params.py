"""BaroclinicParams: defaults off, validator requires vorticity, presets load."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset


def test_baroclinic_defaults_off():
    p = PlanetParams()
    assert p.solver.baroclinic.enabled is False
    assert p.solver.baroclinic.gain == 0.5


def test_enabled_requires_vorticity():
    with pytest.raises(ValidationError):
        PlanetParams.model_validate(
            {"solver": {"type": "kinematic", "baroclinic": {"enabled": True}}}
        )


def test_enabled_with_vorticity_ok():
    p = PlanetParams.model_validate(
        {"solver": {"type": "vorticity", "baroclinic": {"enabled": True}}}
    )
    assert p.solver.baroclinic.enabled is True


def test_existing_preset_without_block_loads():
    p = load_factory_preset("jupiter_vorticity")
    assert p.solver.baroclinic.enabled is False

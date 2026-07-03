"""BaroclinicParams: defaults off, validator requires vorticity, presets load."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import load_factory_preset


def test_baroclinic_defaults_off():
    p = PlanetParams()
    assert p.solver.baroclinic.enabled is False
    assert p.solver.baroclinic.gain == 2.0


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


def test_baroclinic_fields_have_no_rand():
    # randomize() must never flip these on; lock the trap against future edits.
    from gasgiant.params.model import BaroclinicParams, field_meta
    for f in ("enabled", "gain", "warmup_steps", "baro_steps_per_update", "update_every"):
        assert field_meta(BaroclinicParams, f).rand is None, f

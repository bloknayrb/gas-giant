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
    """NO baroclinic field is randomized, and the reason differs by group.

    The enablement and cadence fields: randomize() must never silently switch the
    feature on. The storm-band shape levers: they are inert while `enabled` is
    False -- which is every shipped preset -- so randomizing them would vary
    nothing visible, while still consuming draws. The randomize walk pulls one
    value per rand-bearing field in DECLARATION order, so seven extra draws here
    would shift the roll of every field declared after them and silently change
    the output of randomize(seed) for every preset. Not worth it for levers that
    are switched off.
    """
    from gasgiant.params.model import BaroclinicParams, field_meta
    for f in BaroclinicParams.model_fields:
        assert field_meta(BaroclinicParams, f).rand is None, f


def test_storm_band_levers_default_to_the_previous_hardcoded_values():
    """Every promoted constant must default to the number it replaced, or
    enabling baroclinic would render differently than it did before."""
    from gasgiant.sim import baroclinic_source as bsrc
    b = PlanetParams().solver.baroclinic
    assert (b.latitude, b.width) == (bsrc.PHI_TEST_DEG, bsrc.BAND_HALFWIDTH_DEG)
    assert (b.eddy_scale, b.zonal_count, b.smooth) == (
        bsrc.GP2, bsrc.M_ZONAL, bsrc.SMOOTH_SIGMA)
    assert (b.phase_jitter, b.spectrum_width) == (0.0, 0)


def test_storm_band_levers_are_range_checked():
    """lo/hi are pydantic ge/le on a validate_assignment model, so an
    out-of-range eddy_scale is rejected rather than silently outcropping the
    solver mid-run."""
    import pytest
    from pydantic import ValidationError
    b = PlanetParams().solver.baroclinic
    for field, bad in (("eddy_scale", 0.9), ("zonal_count", 40),
                       ("latitude", 89.0), ("width", 0.5),
                       ("smooth", 0.0), ("phase_jitter", -1.0),
                       ("spectrum_width", -1)):
        with pytest.raises(ValidationError):
            setattr(b, field, bad)

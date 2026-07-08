"""T8: parameter interpolation (lerp_params) + ramp validation (validate_ramp)."""

from __future__ import annotations

import pytest

from gasgiant.params.interp import RampError, lerp_params, validate_ramp
from gasgiant.params.model import (
    GradientStop,
    PaletteRow,
    PlanetParams,
    SolverType,
)

# ---------------------------------------------------------------- endpoints

def test_t0_returns_a_bit_exact():
    a = PlanetParams(seed=0)
    a.appearance.contrast = 1.234567
    a.appearance.haze_color = (0.11, 0.22, 0.33)
    b = PlanetParams(seed=0)
    b.appearance.contrast = 2.0
    b.appearance.haze_color = (0.9, 0.8, 0.7)
    # t == 0 must be a BIT-EXACT (no float round-trip through the lerp arithmetic).
    assert lerp_params(a, b, 0.0).model_dump() == a.model_dump()


def test_t1_returns_b_bit_exact():
    a = PlanetParams(seed=0)
    b = PlanetParams(seed=0)
    b.appearance.contrast = 2.0
    b.export.png_compression = 8
    assert lerp_params(a, b, 1.0).model_dump() == b.model_dump()


# ---------------------------------------------------------------- leaf lerps

def test_float_int_color_lerp_at_half():
    a = PlanetParams(seed=0)  # contrast 1.0, png_compression 2, haze (0.85,0.78,0.62)
    b = PlanetParams(seed=0)
    b.appearance.contrast = 2.0          # float POST
    b.export.png_compression = 8         # int POST
    b.appearance.haze_color = (0.05, 0.18, 0.42)  # color POST

    mid = lerp_params(a, b, 0.5)
    assert mid.appearance.contrast == pytest.approx(1.5)              # float lerp
    assert mid.export.png_compression == 5                            # rounded int lerp
    assert mid.appearance.haze_color == pytest.approx((0.45, 0.48, 0.52))  # RGB lerp


# ---------------------------------------------------------------- rejections

def _vorticity(**over) -> PlanetParams:
    p = PlanetParams(seed=0)
    p.solver.type = SolverType.VORTICITY
    for k, v in over.items():
        setattr(p.solver.baroclinic, k, v)
    return p


def test_bool_differs_raises():
    a = _vorticity(enabled=False)
    b = _vorticity(enabled=True)
    with pytest.raises(RampError, match="boolean"):
        lerp_params(a, b, 0.5)


def test_enum_differs_raises():
    a = PlanetParams(seed=0)  # solver.type kinematic
    b = PlanetParams(seed=0)
    b.solver.type = SolverType.VORTICITY
    with pytest.raises(RampError, match="solver.type"):
        lerp_params(a, b, 0.5)


def test_str_differs_raises():
    a = PlanetParams(seed=0, name="alpha")
    b = PlanetParams(seed=0, name="beta")
    with pytest.raises(RampError, match="name"):
        lerp_params(a, b, 0.5)


def test_none_to_value_raises():
    a = PlanetParams(seed=0)          # storms.hero_latitude None
    b = PlanetParams(seed=0)
    b.storms.hero_latitude = 30.0
    with pytest.raises(RampError, match="hero_latitude"):
        lerp_params(a, b, 0.5)


def test_stop_length_mismatch_raises():
    a = PlanetParams(seed=0)  # default palette row has 4 stops
    b = PlanetParams(seed=0)
    b.appearance.palette_rows = [
        PaletteRow(latitude=0.0, stops=[
            GradientStop(pos=0.0, color=(0.0, 0.0, 0.0)),
            GradientStop(pos=0.5, color=(0.5, 0.5, 0.5)),
            GradientStop(pos=1.0, color=(1.0, 1.0, 1.0)),
        ])
    ]
    with pytest.raises(RampError, match="length mismatch"):
        lerp_params(a, b, 0.5)


def test_stop_pos_set_mismatch_raises():
    a = PlanetParams(seed=0)
    stops_a = a.appearance.palette_rows[0].stops
    b = PlanetParams(seed=0)
    # Same COUNT, different anchor positions -> a reshaped gradient, not a lerp.
    b.appearance.palette_rows = [
        PaletteRow(latitude=0.0, stops=[
            GradientStop(pos=min(1.0, s.pos + 0.05), color=s.color) for s in stops_a
        ])
    ]
    with pytest.raises(RampError, match="stop positions differ"):
        lerp_params(a, b, 0.5)


# ---------------------------------------------------------------- validate_ramp

def test_validate_ramp_rejects_restart_diff_naming_path():
    a = PlanetParams(seed=0)
    b = PlanetParams(seed=0)
    b.bands.count = 20  # RESTART
    with pytest.raises(RampError, match="bands.count"):
        validate_ramp(a, b)


def test_validate_ramp_rejects_seed_diff_naming_path():
    a = PlanetParams(seed=0)
    b = PlanetParams(seed=1)
    with pytest.raises(RampError, match="seed"):
        validate_ramp(a, b)


def test_validate_ramp_accepts_post_only():
    a = PlanetParams(seed=0)
    b = PlanetParams(seed=0)
    b.appearance.contrast = 1.5  # POST
    validate_ramp(a, b)  # no raise


def test_validate_ramp_accepts_velocity_only():
    a = PlanetParams(seed=0)
    b = PlanetParams(seed=0)
    b.jets.strength = 2.0  # VELOCITY
    validate_ramp(a, b)  # no raise

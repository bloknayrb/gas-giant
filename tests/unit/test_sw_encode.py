"""Unit tests for the M3 production top-of-atmosphere encoder (sw_encode)."""

import numpy as np

from gasgiant.sim import shallow_water_ref as ref
from gasgiant.sim import sw_encode


def _make_state(W=32, H=16):
    """A small balanced 2-layer state with a top-layer jet (nonzero vorticity)."""
    return ref.balanced_2layer_state(
        W=W, H=H, a=6.4e6, omega=7.292e-5, gp1=0.5, gp2=0.3, u0=20.0,
    )


def test_to_tracer_shape_and_dtype():
    st = _make_state()
    rgba = sw_encode.to_tracer(st)
    assert rgba.shape == (st.g.H, st.g.W, 4)
    assert rgba.dtype == np.float32


def test_to_tracer_all_channels_in_unit_range():
    st = _make_state()
    rgba = sw_encode.to_tracer(st)
    assert np.all(np.isfinite(rgba)), "tracer must be finite"
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0, "tracer must be in [0,1]"


def test_to_tracer_channels_are_nondegenerate():
    """A jet has real structure: each channel must span a meaningful range,
    not collapse to a constant (a degenerate encode renders a flat disc)."""
    st = _make_state()
    rgba = sw_encode.to_tracer(st)
    for c, name in enumerate(("h_anom", "h1", "|zeta1|", "zeta1")):
        spread = float(rgba[..., c].max() - rgba[..., c].min())
        assert spread > 0.1, f"channel {name} degenerate (spread={spread:.3f})"


def test_to_tracer_fields_matches_wrapper():
    st = _make_state()
    a = sw_encode.to_tracer(st)
    b = sw_encode.to_tracer_fields(st.h1, st.u1, st.v1, st.g, st.h_eq1)
    assert np.array_equal(a, b)


def test_to_tracer_handles_missing_heq():
    """h_eq1 None -> anomaly vs global mean, still a sane [0,1] RGBA."""
    st = _make_state()
    rgba = sw_encode.to_tracer_fields(st.h1, st.u1, st.v1, st.g, None)
    assert rgba.shape == (st.g.H, st.g.W, 4)
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0

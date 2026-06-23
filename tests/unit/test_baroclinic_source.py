"""Coherent baroclinic source module: wavenumber detection, coherence gate,
geostrophic proxy on synthetic states, resampling."""
from __future__ import annotations

import types

import numpy as np
import pytest

from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim.shallow_water_ref import Grid


def _synthetic_state(m_zonal: int, W: int = 192, H: int = 96):
    """Minimal duck-typed baroclinic state with a single-wavenumber interface
    eddy h2 = cos(m*lambda) * lat-envelope, plus the attributes the source
    module reads (h2, g, omega, gp2)."""
    g = Grid(W=W, H=H, a=6.4e6)
    lam = (np.arange(W) + 0.5) * g.dlam
    latdeg = np.degrees(g.phi_c)
    env = np.exp(-((np.abs(latdeg) - 45.0) ** 2) / (2 * 12.0 ** 2))  # band near +/-45
    h2 = 1.0 + 0.01 * (env[:, None] * np.cos(m_zonal * lam)[None, :])
    return types.SimpleNamespace(h2=h2.astype(np.float64), g=g,
                                 omega=7.292e-5, gp2=0.3)


def test_dominant_zonal_m_recovers_wavenumber():
    g = Grid(W=192, H=96, a=6.4e6)
    lam = (np.arange(192) + 0.5) * g.dlam
    field = np.cos(8 * lam)[None, :] * np.ones((96, 1))
    m, _ = bsrc.dominant_zonal_m(field)
    assert m == 8


def test_geostrophic_source_is_coherent():
    st = _synthetic_state(m_zonal=5)
    zeta = bsrc.geostrophic_vorticity_source(st)
    m = bsrc.assert_coherent(zeta)         # must not raise
    assert m <= 8                          # near the seeded m=5


def test_assert_coherent_rejects_checkerboard():
    st = _synthetic_state(m_zonal=44)      # C-grid-like high wavenumber
    zeta = bsrc.geostrophic_vorticity_source(st)
    with pytest.raises(ValueError, match="coherence gate"):
        bsrc.assert_coherent(zeta)


def test_coherence_gate_boundary():
    """The gate was widened 15->20 for the m~14 production mode. Enforce the new
    band: a coherent m=18 source (REJECTED at the old gate of 15) is now ACCEPTED
    and stays in band, while m=25 is still rejected. Guards an accidental revert
    of M_GATE_MAX in either direction."""
    accepted = bsrc.geostrophic_vorticity_source(_synthetic_state(m_zonal=18))
    m = bsrc.assert_coherent(accepted)            # must not raise
    assert 16 <= m <= bsrc.M_GATE_MAX             # in the newly-opened [16,20] band
    rejected = bsrc.geostrophic_vorticity_source(_synthetic_state(m_zonal=25))
    with pytest.raises(bsrc.IncoherentSourceError, match="coherence gate"):
        bsrc.assert_coherent(rejected)


def test_gate_error_is_valueerror_subclass():
    """IncoherentSourceError must subclass ValueError so every existing
    `except ValueError` / pytest.raises(ValueError) caller keeps working."""
    assert issubclass(bsrc.IncoherentSourceError, ValueError)


def test_band_mask_zeros_poles():
    st = _synthetic_state(m_zonal=5)
    zeta = bsrc.geostrophic_vorticity_source(st)
    latdeg = np.degrees(st.g.phi_c)
    polar = np.abs(latdeg) > 85.0
    assert np.allclose(zeta[polar], 0.0)


def test_resample_unit_std_and_shape():
    st = _synthetic_state(m_zonal=5)
    zeta = bsrc.geostrophic_vorticity_source(st)
    out = bsrc.resample_to_equirect(zeta, 128, 64)
    assert out.shape == (64, 128)
    assert out.dtype == np.float32
    assert abs(float(out.std()) - 1.0) < 1e-3

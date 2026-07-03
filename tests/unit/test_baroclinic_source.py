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


# -- A2-3: scipy removed from the production path ---------------------------------
#
# scipy was declared test-only (pyproject "Test-only." comment) but
# baroclinic_source imported scipy.ndimage at module top, production-reachable via
# baroclinic.enabled -> a hard ImportError on a plain install. The two scipy call
# sites are replaced with numpy/cv2 equivalents; these tests prove numerical parity
# against scipy (still available in the dev/test env) and pin scipy out of the
# production import graph.


def test_smooth_periodic_matches_scipy_gaussian_filter():
    """_smooth_periodic must be numerically equivalent to
    scipy.ndimage.gaussian_filter(mode=("reflect", "wrap")) at the production
    sigmas (1.0 mid-pipeline, SMOOTH_SIGMA driver, 2.5 default)."""
    ndi = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(42)
    field = rng.standard_normal((96, 192))
    for sigma in (1.0, bsrc.SMOOTH_SIGMA, 2.5):
        ours = bsrc._smooth_periodic(field, sigma)
        ref = ndi.gaussian_filter(field, sigma=sigma, mode=("reflect", "wrap"))
        np.testing.assert_allclose(ours, ref, atol=1e-9, rtol=1e-9,
                                   err_msg=f"sigma={sigma}")


def test_zoom_bilinear_matches_scipy_zoom():
    """The bilinear resampler must match scipy.ndimage.zoom(order=1,
    mode="nearest", grid_mode=False) on the production float32 path, for both
    up- and same-size resampling (SRC 192x96 -> equirect grids)."""
    ndi = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(7)
    field = rng.standard_normal((96, 192)).astype(np.float32)
    for out_h, out_w in ((256, 512), (96, 192), (64, 128)):
        ours = bsrc._zoom_bilinear(field, out_h, out_w)
        ref = ndi.zoom(field, (out_h / 96, out_w / 192), order=1, mode="nearest")
        assert ours.shape == ref.shape == (out_h, out_w)
        np.testing.assert_allclose(ours, ref, atol=1e-6, rtol=1e-6,
                                   err_msg=f"out=({out_h},{out_w})")


def test_resample_to_equirect_matches_legacy_scipy_pipeline():
    """End-to-end parity of resample_to_equirect against the exact legacy scipy
    implementation (zoom -> crop -> unit-std normalize -> float32)."""
    ndi = pytest.importorskip("scipy.ndimage")
    st = _synthetic_state(m_zonal=5)
    zeta = bsrc.geostrophic_vorticity_source(st)
    out = bsrc.resample_to_equirect(zeta, 512, 256)

    zy, zx = 256 / zeta.shape[0], 512 / zeta.shape[1]
    legacy = ndi.zoom(zeta.astype(np.float32), (zy, zx), order=1, mode="nearest")
    legacy = legacy[:256, :512]
    std = float(np.std(legacy))
    if std > 0:
        legacy = legacy / std
    np.testing.assert_allclose(out, legacy.astype(np.float32), atol=1e-5, rtol=1e-5)


def test_production_import_graph_is_scipy_free():
    """Importing the baroclinic production chain must not pull in scipy (a plain
    install has no scipy; the module-top import was the A2-3 crash). Checked in a
    subprocess so this test env's already-imported modules can't mask it."""
    import subprocess
    import sys
    code = (
        "import sys; "
        "import gasgiant.sim.baroclinic_source, gasgiant.sim.baroclinic_driver; "
        "bad = [m for m in sys.modules if m == 'scipy' or m.startswith('scipy.')]; "
        "sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"scipy leaked into the production import graph: {proc.stdout} {proc.stderr}"
    )

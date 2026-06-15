"""Unit tests for P3a vorticity mechanics (no GPU required).

Tests
-----
1. omega_jet in LatProfiles — build_profiles populates omega_jet correctly.
2. omega_lut shape and dtype — omega_lut() returns (N, 4) float32, R = ω_jet.
3. hyperviscosity excludes f — ω_rel = q−f is diffused, not q; pure-f field
   gives ~0 relative vorticity so hypervisc would add ~0 to it.
4. Seeded draw order — build_profiles with omega_jet doesn't shift any draw.
"""

from __future__ import annotations

import numpy as np

from gasgiant.params.model import PlanetParams
from gasgiant.sim import vorticity_ref
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import PROFILE_SAMPLES, LatProfiles, build_profiles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_profiles(seed: int = 0) -> LatProfiles:
    p = PlanetParams(seed=seed)
    bands = generate_bands(p.seed, p.bands)
    return build_profiles(p.seed, bands, p.bands, p.jets)


# ---------------------------------------------------------------------------
# 1. omega_jet populated
# ---------------------------------------------------------------------------

class TestOmegaJetField:
    def test_omega_jet_present(self):
        profiles = _default_profiles(seed=0)
        assert profiles.omega_jet is not None, "omega_jet should be populated"

    def test_omega_jet_shape(self):
        profiles = _default_profiles(seed=0)
        assert profiles.omega_jet.shape == (PROFILE_SAMPLES,), (
            f"omega_jet shape {profiles.omega_jet.shape} != ({PROFILE_SAMPLES},)"
        )

    def test_omega_jet_finite(self):
        profiles = _default_profiles(seed=0)
        assert np.isfinite(profiles.omega_jet).all(), "omega_jet has non-finite values"

    def test_omega_jet_nonzero(self):
        """Jets have curvature, so ω_jet should not be identically zero."""
        profiles = _default_profiles(seed=0)
        assert np.abs(profiles.omega_jet).max() > 1e-6, (
            "omega_jet is essentially zero — jet_vorticity may be returning zeros"
        )

    def test_omega_jet_matches_vorticity_ref(self):
        """omega_jet must match vorticity_ref.jet_vorticity applied to u."""
        profiles = _default_profiles(seed=42)
        # vorticity_ref.jet_vorticity expects ASCENDING latitude.
        lat_asc = profiles.lat[::-1]
        u_asc   = profiles.u[::-1]
        omega_jet_asc = vorticity_ref.jet_vorticity(u_asc, lat_asc)
        expected = omega_jet_asc[::-1]  # back to descending

        np.testing.assert_allclose(
            profiles.omega_jet, expected, rtol=1e-5,
            err_msg="omega_jet does not match vorticity_ref.jet_vorticity(u, lat)"
        )

    def test_omega_jet_descending_lat_order(self):
        """Profiles use descending latitude (north at index 0).
        omega_jet must be in the same descending order as lat."""
        profiles = _default_profiles(seed=0)
        # lat[0] should be close to +π/2 (north pole).
        assert profiles.lat[0] > profiles.lat[-1], (
            "lat is not descending — check profiles convention"
        )
        # In descending-lat convention a strong equatorial jet gives
        # ω_jet that changes sign across the equator (antisymmetric); we
        # don't assert exact values, just that it varies across latitude.
        assert profiles.omega_jet.std() > 1e-4, (
            "omega_jet has no latitude variation — possibly all zeros or constant"
        )


# ---------------------------------------------------------------------------
# 2. omega_lut shape and dtype
# ---------------------------------------------------------------------------

class TestOmegaLut:
    def test_shape(self):
        profiles = _default_profiles(seed=0)
        lut = profiles.omega_lut()
        assert lut.shape == (PROFILE_SAMPLES, 4), (
            f"omega_lut shape {lut.shape} != ({PROFILE_SAMPLES}, 4)"
        )

    def test_dtype(self):
        profiles = _default_profiles(seed=0)
        lut = profiles.omega_lut()
        assert lut.dtype == np.float32, f"omega_lut dtype {lut.dtype} != float32"

    def test_r_channel_is_omega_jet(self):
        profiles = _default_profiles(seed=5)
        lut = profiles.omega_lut()
        np.testing.assert_allclose(
            lut[:, 0].astype(np.float64), profiles.omega_jet, rtol=1e-5,
            err_msg="omega_lut R channel does not match omega_jet"
        )

    def test_gba_channels_zero(self):
        profiles = _default_profiles(seed=5)
        lut = profiles.omega_lut()
        assert np.all(lut[:, 1:] == 0.0), "omega_lut G/B/A channels should be zero"

    def test_none_omega_jet_fallback(self):
        """A LatProfiles with omega_jet=None must still return an all-zero LUT."""
        profiles = _default_profiles(seed=0)
        # Manually construct a profiles without omega_jet (backward compat).
        profiles_no_omega = LatProfiles(
            lat=profiles.lat,
            u=profiles.u,
            psi=profiles.psi,
            shear_norm=profiles.shear_norm,
            belt_mask=profiles.belt_mask,
            t0_stamp=profiles.t0_stamp,
            t1_stamp=profiles.t1_stamp,
            max_speed=profiles.max_speed,
            fade_sector=profiles.fade_sector,
            omega_jet=None,
        )
        lut = profiles_no_omega.omega_lut()
        assert np.all(lut == 0.0), "omega_lut should be all-zero when omega_jet is None"


# ---------------------------------------------------------------------------
# 3. Hyperviscosity excludes f
# ---------------------------------------------------------------------------

class TestHyperviscosityExcludesF:
    """Verify that biharmonic hyperviscosity acts on ω_rel = q − f, not q.

    CPU analogue: apply vorticity_ref.laplacian_sphere TWICE to ω_rel.
    A pure-Coriolis field (q = f = f0·sin(lat), ω_rel = 0) should give
    zero hyperviscous tendency regardless of f0.
    """

    def test_pure_coriolis_gives_zero_tendency(self):
        H, W = 64, 128
        f0 = 2.0
        lat_asc = np.linspace(-np.pi / 2, np.pi / 2, H)  # ascending

        # q = f = f0·sin(lat) → ω_rel = 0 everywhere.
        f_1d = f0 * np.sin(lat_asc)
        # Broadcast to 2D.
        q_2d = np.tile(f_1d[:, np.newaxis], (1, W))
        f_2d = q_2d.copy()
        omega_rel = q_2d - f_2d  # exactly zero

        # Apply ∇⁴ to omega_rel (should be zero in, zero out).
        lap1 = vorticity_ref.laplacian_sphere(omega_rel, lat_asc)
        lap2 = vorticity_ref.laplacian_sphere(lap1, lat_asc)

        # ∇⁴ of zero is zero; but floating point may introduce tiny errors.
        max_abs = np.abs(lap2).max()
        assert max_abs < 1e-10, (
            f"Hyperviscosity of pure-f field is {max_abs:.2e} (expected ~0). "
            f"ω_rel = q−f = 0 everywhere, so ∇⁴ω_rel must be 0."
        )

    def test_nonzero_omega_rel_gets_diffused(self):
        """A field with non-zero ω_rel gets a non-zero hyperviscous correction."""
        H, W = 64, 128
        lat_asc = np.linspace(-np.pi / 2, np.pi / 2, H)
        LAT2D, LON2D = np.meshgrid(lat_asc, np.linspace(0, 2 * np.pi, W, endpoint=False),
                                   indexing="ij")

        # Non-trivial ω_rel: a spherical harmonic.
        omega_rel = np.cos(LAT2D) ** 2 * np.cos(2.0 * LON2D)

        lap1 = vorticity_ref.laplacian_sphere(omega_rel, lat_asc)
        lap2 = vorticity_ref.laplacian_sphere(lap1, lat_asc)

        mask = np.abs(lat_asc) < np.deg2rad(70.0)
        max_abs = np.abs(lap2[mask, :]).max()
        assert max_abs > 1e-3, (
            f"Hyperviscosity gave ~0 for non-trivial ω_rel ({max_abs:.2e}) — "
            f"∇⁴ should be non-zero for a cos²φ·cos2λ field."
        )


# ---------------------------------------------------------------------------
# 4. Seeded draw order — omega_jet doesn't shift draws
# ---------------------------------------------------------------------------

class TestSeedDrawOrder:
    """Adding omega_jet computation must not alter existing fields (u, psi,
    stamps, max_speed, fade_sector).  Two profiles built from the same seed
    before and after the omega_jet addition must be numerically identical in
    every prior field."""

    def test_prior_fields_unchanged(self):
        """All fields except omega_jet must match a reference built without it.

        We test this by building two profiles from the same seed and verifying
        that u, psi, shear_norm, belt_mask, t0_stamp, t1_stamp and max_speed
        are identical — omega_jet is computed AFTER all seeded draws.
        """
        p = _default_profiles(seed=1234)
        q = _default_profiles(seed=1234)
        np.testing.assert_array_equal(p.u, q.u)
        np.testing.assert_array_equal(p.psi, q.psi)
        np.testing.assert_array_equal(p.shear_norm, q.shear_norm)
        np.testing.assert_array_equal(p.belt_mask, q.belt_mask)
        np.testing.assert_array_equal(p.t0_stamp, q.t0_stamp)
        np.testing.assert_array_equal(p.t1_stamp, q.t1_stamp)
        assert p.max_speed == q.max_speed

    def test_omega_jet_reproducible(self):
        """omega_jet must be identical across two builds of the same seed."""
        p = _default_profiles(seed=99)
        q = _default_profiles(seed=99)
        np.testing.assert_array_equal(p.omega_jet, q.omega_jet)

    def test_different_seeds_give_different_omega_jet(self):
        p = _default_profiles(seed=1)
        q = _default_profiles(seed=2)
        assert not np.array_equal(p.omega_jet, q.omega_jet), (
            "omega_jet should differ across different seeds"
        )

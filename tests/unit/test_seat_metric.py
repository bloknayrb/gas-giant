"""seat_quality: a diagnostic proxy for how well the NATURAL jets give a hero a
two-sided anticyclonic bearing at a candidate latitude. Pure function of the
bracket-off profile; higher = better bearing. Used by the GUI seat meter."""
from __future__ import annotations

import numpy as np

from gasgiant.params.model import PlanetParams
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles, seat_band, seat_quality, seat_scan


def _warm_like_profile(seed=4201):
    p = PlanetParams(seed=seed)
    bands = generate_bands(seed, p.bands)
    return build_profiles(seed, bands, p.bands, p.jets)


def test_seat_quality_best_seat_is_not_the_iconic_latitude():
    """Design premise: on warm the natural best bearing is NOT at the iconic
    hero latitude (-22). The scan's argmax must be a DISTINCT latitude, not -22
    itself -- a `max(scan) >= q(-22)` check would be tautological because the
    scan grid contains -22. (This is exactly why the bracket override exists.)"""
    prof = _warm_like_profile()
    q22 = seat_quality(prof, -22.0, 3.0)
    # scan grid deliberately EXCLUDES -22 so the comparison is non-vacuous
    scan = seat_scan(prof, [ld for ld in np.arange(-14.0, -44.0, -1.0)
                            if abs(ld - (-22.0)) > 0.5], 3.0)
    best_lat, best_q = max(scan, key=lambda t: t[1])
    assert best_q > q22, f"no natural seat beats -22 (q22={q22}, best={best_q})"
    assert abs(best_lat - (-22.0)) > 3.0, f"best seat {best_lat} too close to -22"


def test_seat_quality_sign_flips_with_spin():
    """A seat that is good for an anticyclone (spin +1) is bad for a cyclone
    (spin -1) at the same latitude: quality changes sign of its two_sided term."""
    prof = _warm_like_profile()
    lat = -19.0
    qa = seat_quality(prof, lat, 3.0, spin_sign=1.0)
    qc = seat_quality(prof, lat, 3.0, spin_sign=-1.0)
    assert qa != qc


def test_seat_band_thresholds():
    assert seat_band(0.3) == "green"
    assert seat_band(0.05) == "amber"
    assert seat_band(-0.2) == "red"


def test_seat_scan_returns_lat_quality_pairs():
    prof = _warm_like_profile()
    scan = seat_scan(prof, [-20.0, -30.0, -40.0], 3.0)
    assert len(scan) == 3
    assert all(len(t) == 2 for t in scan)
    assert scan[0][0] == -20.0

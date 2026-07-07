"""GPU tier: field-driven detail activity pass + FIELD_DRIVE variant.

Determinism note: the activity field is deterministic on the baked velocity;
kinematic mode is byte-exact, vorticity mode is within the documented SOR LSB
floors. These tests use synthetic/analytic velocity fields, so they are exact.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

from gasgiant.render.activity import ActivitySynth, new_activity_texture


def _shear_velocity(gpu, w, h):
    """A pure zonal jet u = sin(2*lat) (strong midlat shear) as an RG f4
    equirect velocity, repeat_x wrapped. du/dlat = 2 cos(2 lat): max at the
    equator, zero at +-45 deg."""
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi  # +pi/2 .. -pi/2
    u = np.sin(2.0 * lat)[:, None] * np.ones((1, w))
    vel = np.zeros((h, w, 2), np.float32)
    vel[:, :, 0] = u
    return gpu.texture2d((w, h), 2, "f4", data=vel, linear=True)


def test_activity_is_finite_including_poles(gpu):
    w, h = 256, 128
    synth = ActivitySynth(gpu)
    vel = _shear_velocity(gpu, w, h)
    act = new_activity_texture(gpu, (w, h))
    means = synth.build(vel, act)
    arr = gpu.read_texture(act)
    assert np.all(np.isfinite(arr)), "activity has NaN/Inf (pole 1/cos blowup?)"
    assert means.mean_eddy >= 0.0
    assert means.mean_vort >= 0.0
    means.release()
    synth.release()


def test_activity_strain_peaks_at_jet_shear(gpu):
    w, h = 256, 128
    synth = ActivitySynth(gpu)
    vel = _shear_velocity(gpu, w, h)
    act = new_activity_texture(gpu, (w, h))
    synth.build(vel, act)
    strain = gpu.read_texture(act)[:, :, 0]
    eq_row = h // 2       # ~equator (max |du/dlat|)
    q_row = h // 4        # ~+45 deg (min |du/dlat|)
    assert strain[eq_row].mean() > strain[q_row].mean()
    synth.release()

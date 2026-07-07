"""GPU tier: field-driven detail activity pass + FIELD_DRIVE variant.

Determinism note: the activity field is deterministic on the baked velocity;
kinematic mode is byte-exact, vorticity mode is within the documented SOR LSB
floors. These tests use synthetic/analytic velocity fields, so they are exact.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

from gasgiant.params.model import DetailParams  # noqa: E402
from gasgiant.render.activity import ActivitySynth, new_activity_texture  # noqa: E402
from gasgiant.render.detail import DetailSynth  # noqa: E402


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


# -- FIELD_DRIVE variant wiring ------------------------------------------------
def _synth_detail(gpu, params, activity=None, means=None):
    """Minimal detail synthesis into a 1-channel out texture (no heroes, no
    polar route) -> (H, W) float32."""
    w, h = 128, 64
    vel = gpu.texture2d((w, h), 2, "f4", linear=True)
    tracers = gpu.texture2d((w, h), 4, "f4", linear=True)
    prof = gpu.lut_texture(np.zeros((h, 4), np.float32))
    out = gpu.texture2d((w, h), 1, "f4", linear=True)
    DetailSynth(gpu).synthesize(
        7, vel, tracers, prof, out, params, activity=activity, means=means,
    )
    return gpu.read_texture(out)[:, :, 0]


def test_field_drive_zero_is_byte_identical_to_default(gpu):
    """field_drive=0 selects the non-variant program -> byte-for-byte today."""
    base = _synth_detail(gpu, DetailParams(intensity=0.55))
    off = _synth_detail(gpu, DetailParams(intensity=0.55, field_drive=0.0))
    np.testing.assert_array_equal(base, off)


def test_field_drive_zero_byte_identical_with_fx_on(gpu):
    """The base-path re-gate lands in BOTH fx and non-fx programs, so a
    DETAIL_FX build with field_drive=0 must also be byte-identical."""
    fx = DetailParams(intensity=0.55, belt_texture=1.0, mottle=0.8)
    base = _synth_detail(gpu, fx)
    off = _synth_detail(gpu, fx.model_copy(update={"field_drive": 0.0}))
    np.testing.assert_array_equal(base, off)


def test_field_drive_forced_variant_is_near_default(gpu):
    """field_drive=1e-6 compiles the FIELD_DRIVE binary; output ~ default within
    the cross-binary FP-reschedule tolerance (NOT array_equal)."""
    w, h = 128, 64
    velfield = gpu.texture2d((w, h), 2, "f4", linear=True)
    act = new_activity_texture(gpu, (w, h))
    means = ActivitySynth(gpu).build(velfield, act)
    base = _synth_detail(gpu, DetailParams(intensity=0.55))
    tiny = _synth_detail(
        gpu, DetailParams(intensity=0.55, field_drive=1e-6), activity=act, means=means,
    )
    np.testing.assert_allclose(base, tiny, atol=1e-3)
    means.release()


def _warm_sim(gpu):
    import json
    import pathlib

    from gasgiant.engine.facade import Simulation
    from gasgiant.params.presets import load_preset_doc

    doc = json.loads(
        pathlib.Path("src/gasgiant/presets/gas_giant_warm.json").read_text()
    )
    params = load_preset_doc(doc, "test")
    return Simulation(params, gpu=gpu)


def test_facade_preview_field_drive_builds_and_differs(gpu):
    sim = _warm_sim(gpu)
    sim.run_to_completion()
    base_color, _ = sim.ensure_preview(256)
    base = gpu.read_texture(base_color).copy()
    p2 = sim.params.model_copy(deep=True)
    p2.detail.field_drive = 1.0  # POST edit -> re-derive
    sim.update_params(p2)
    fd_color, _ = sim.ensure_preview(256)
    fd = gpu.read_texture(fd_color)
    assert not np.allclose(base, fd, atol=1e-2), "field_drive=1 did not change preview"


def test_field_drive_forced_variant_near_default_with_fx(gpu):
    """The (DETAIL_FX + FIELD_DRIVE) program: forced tiny drive ~ the DETAIL_FX
    default. Exercises the BOTH-variant vort tripwire path."""
    w, h = 128, 64
    velfield = gpu.texture2d((w, h), 2, "f4", linear=True)
    act = new_activity_texture(gpu, (w, h))
    means = ActivitySynth(gpu).build(velfield, act)
    fx = DetailParams(intensity=0.55, belt_texture=1.0, mottle=0.8)
    base = _synth_detail(gpu, fx)
    tiny = _synth_detail(
        gpu, fx.model_copy(update={"field_drive": 1e-6}), activity=act, means=means,
    )
    np.testing.assert_allclose(base, tiny, atol=1e-3)
    means.release()

"""GPU tier: uniform detail coverage (`detail.spread`) SPREAD variant.

spread=0 selects the non-variant program and is byte-identical to today's
band-gated detail; spread>0 applies the flow-folded detail at even density
across latitude. Determinism: kinematic byte-exact, vorticity within the SOR
LSB floors (these tests use analytic inputs, so they are exact).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

from gasgiant.params.model import DetailParams  # noqa: E402
from gasgiant.render.detail import DetailSynth  # noqa: E402


def _synth_detail(gpu, params):
    """Minimal detail synthesis into a 1-channel out texture -> (H, W) float32."""
    w, h = 128, 64
    vel = gpu.texture2d((w, h), 2, "f4", linear=True)
    tracers = gpu.texture2d((w, h), 4, "f4", linear=True)
    prof = gpu.lut_texture(np.zeros((h, 4), np.float32))
    out = gpu.texture2d((w, h), 1, "f4", linear=True)
    DetailSynth(gpu).synthesize(7, vel, tracers, prof, out, params)
    return gpu.read_texture(out)[:, :, 0]


def test_spread_zero_is_byte_identical_to_default(gpu):
    """spread=0 selects the non-variant program -> byte-for-byte today."""
    base = _synth_detail(gpu, DetailParams(intensity=0.55))
    off = _synth_detail(gpu, DetailParams(intensity=0.55, spread=0.0))
    np.testing.assert_array_equal(base, off)


def test_spread_zero_byte_identical_with_fx_on(gpu):
    """The base-path re-gate lands in BOTH fx and non-fx programs, so a DETAIL_FX
    build with spread=0 must also be byte-identical."""
    fx = DetailParams(intensity=0.55, belt_texture=1.0, mottle=0.8)
    base = _synth_detail(gpu, fx)
    off = _synth_detail(gpu, fx.model_copy(update={"spread": 0.0}))
    np.testing.assert_array_equal(base, off)


def test_spread_variant_changes_output(gpu):
    """spread>0 intentionally changes the look (it is a coverage level, NOT a
    blend-from-default): the SPREAD binary must differ from the non-variant one.
    (The byte-identity contract is spread=0 -> non-variant program, above.)"""
    base = _synth_detail(gpu, DetailParams(intensity=0.55, belt_texture=1.0))
    spread = _synth_detail(gpu, DetailParams(intensity=0.55, belt_texture=1.0, spread=0.36))
    assert not np.allclose(base, spread, atol=1e-3)


def _warm_sim(gpu, res=None):
    import json
    import pathlib

    from gasgiant.engine.facade import Simulation
    from gasgiant.params.presets import load_preset_doc

    doc = json.loads(
        pathlib.Path("src/gasgiant/presets/gas_giant_warm.json").read_text()
    )
    params = load_preset_doc(doc, "test")
    if res is not None:  # smaller sim for fast structural tests
        params = params.model_copy(deep=True)
        params.sim.resolution = res
    return Simulation(params, gpu=gpu)


def test_facade_preview_spread_differs(gpu):
    sim = _warm_sim(gpu)
    sim.run_to_completion()
    base_color, _ = sim.ensure_preview(256)
    base = gpu.read_texture(base_color).copy()
    p2 = sim.params.model_copy(deep=True)
    p2.detail.spread = 0.36  # POST edit -> re-derive
    sim.update_params(p2)
    fd_color, _ = sim.ensure_preview(256)
    fd = gpu.read_texture(fd_color)
    assert not np.allclose(base, fd, atol=1e-2), "spread=0.36 did not change preview"


def test_export_tiled_matches_full_at_spread(gpu):
    """Tiled export must equal a single-tile render at spread>0: the SPREAD
    weights are functions of absolute lon/lat, so they are seam-safe."""
    from gasgiant.export.exporter import _derive_tile

    sim = _warm_sim(gpu, res=512)
    p2 = sim.params.model_copy(deep=True)
    p2.detail.spread = 0.36
    sim.update_params(p2)
    sim.run_to_completion()
    snap = sim.create_snapshot()
    w, h = 256, 128
    full_c = gpu.texture2d((w, h), 4, "f4")
    full_hh = gpu.texture2d((w, h), 1, "f4")
    full_d = gpu.texture2d((w, h), 1, "f4", linear=True)
    _derive_tile(sim, snap, snap.params, 0, 0, w, h, full_c, full_hh, full_d, None)
    full = gpu.read_texture(full_c).copy()
    tile_c = gpu.texture2d((w, h // 2), 4, "f4")
    tile_hh = gpu.texture2d((w, h // 2), 1, "f4")
    tile_d = gpu.texture2d((w, h // 2), 1, "f4", linear=True)
    _derive_tile(sim, snap, snap.params, 0, 0, w, h, tile_c, tile_hh, tile_d, None)
    top = gpu.read_texture(tile_c)[: h // 2].copy()
    np.testing.assert_allclose(full[: h // 2], top, atol=1e-3)
    snap.release()

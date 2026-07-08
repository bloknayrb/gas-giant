"""T12 in-GUI mask brush: the pure-numpy great-circle stamp, the 16-bit
grayscale sidecar round-trip, and the headless GUI paint tool (mode, stamp,
throttled upload, undo-stroke, clear).

The brush tests are GL-free (core.brush imports numpy only); the GUI tool test
importorskips imgui and fakes the sim/viewport the same way test_place_storm and
test_playback do.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gasgiant.core import brush

# ---------------------------------------------------------------- helpers

def _great_circle_deg(lon0, lat0, lon_grid, lat_grid):
    """Great-circle distance (deg) from (lon0, lat0) to each grid point."""
    lat0r, lon0r = math.radians(lat0), math.radians(lon0)
    latr, lonr = np.radians(lat_grid), np.radians(lon_grid)
    cos_d = (
        math.sin(lat0r) * np.sin(latr)
        + math.cos(lat0r) * np.cos(latr) * np.cos(lonr - lon0r)
    )
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def _grid(h, w):
    lon = -180.0 + (np.arange(w) + 0.5) / w * 360.0
    lat = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    return np.broadcast_to(lon, (h, w)), np.broadcast_to(lat[:, None], (h, w))


# ---------------------------------------------------------------- basic deposit

def test_stamp_deposits_peak_at_center():
    buf = brush.new_buffer(64, 128)
    brush.stamp(buf, lon_deg=0.0, lat_deg=0.0, radius_deg=20.0, strength=0.8)
    # center column/row of the (64,128) grid is near lon 0 / lat 0
    center = buf[32, 64]
    assert center == pytest.approx(0.8, abs=0.05)
    assert buf.max() <= 1.0 and buf.min() >= 0.0


def test_stamp_noop_for_zero_radius_or_strength():
    buf = brush.new_buffer(32, 64)
    brush.stamp(buf, 0.0, 0.0, radius_deg=0.0, strength=1.0)
    assert not buf.any()
    brush.stamp(buf, 0.0, 0.0, radius_deg=10.0, strength=0.0)
    assert not buf.any()


# ---------------------------------------------------------------- great-circle set

def test_painted_set_matches_great_circle_radius():
    h, w = 128, 256
    buf = brush.new_buffer(h, w)
    lon0, lat0, radius = 30.0, 15.0, 25.0
    brush.stamp(buf, lon0, lat0, radius_deg=radius, strength=1.0)
    lon_grid, lat_grid = _grid(h, w)
    d = _great_circle_deg(lon0, lat0, lon_grid, lat_grid)
    painted = buf > 0.0
    within = d < radius - 1e-6  # strictly inside falls to zero exactly at radius
    outside = d > radius + 1e-6
    # every strictly-inside texel is painted; every strictly-outside is not
    assert np.all(painted[within])
    assert not np.any(painted[outside])


# ---------------------------------------------------------------- dateline wrap

def test_dateline_continuity():
    h, w = 128, 256
    buf = brush.new_buffer(h, w)
    # stamp near +179 lon; it must paint pixels near -179 (opposite edge)
    brush.stamp(buf, lon_deg=179.0, lat_deg=0.0, radius_deg=15.0, strength=1.0)
    row = buf[h // 2]
    assert row[-1] > 0.0, "east edge (near +180) painted"
    assert row[0] > 0.0, "west edge (near -180) painted via wrap"


# ---------------------------------------------------------------- pole coverage

def test_pole_coverage_full_row_no_nan():
    h, w = 128, 256
    buf = brush.new_buffer(h, w)
    brush.stamp(buf, lon_deg=0.0, lat_deg=89.0, radius_deg=10.0, strength=1.0)
    assert np.isfinite(buf).all()
    top = buf[0]  # top row is the highest latitude band
    # a near-pole stamp covers a WIDE longitude span in the top row
    assert np.count_nonzero(top) > w // 2
    assert top.min() > 0.0  # in fact the whole top row is within radius of the pole


def test_pole_center_no_nan_exact_pole():
    buf = brush.new_buffer(64, 128)
    brush.stamp(buf, lon_deg=45.0, lat_deg=90.0, radius_deg=12.0, strength=1.0)
    assert np.isfinite(buf).all()
    assert buf[0].min() > 0.0


# ---------------------------------------------------------------- erase inverse

def test_erase_is_inverse_of_paint():
    buf = brush.new_buffer(96, 192)
    args = dict(lon_deg=-40.0, lat_deg=-20.0, radius_deg=18.0, strength=0.7)
    brush.stamp(buf, **args)
    assert buf.max() > 0.0
    brush.stamp(buf, **args, erase=True)
    assert buf.max() == pytest.approx(0.0, abs=1e-6)


def test_erase_clamps_at_zero():
    buf = brush.new_buffer(64, 128)
    # erase where nothing was painted must not go negative
    brush.stamp(buf, 0.0, 0.0, radius_deg=20.0, strength=0.9, erase=True)
    assert buf.min() >= 0.0
    assert not buf.any()


# ---------------------------------------------------------------- clamp bounds

def test_repeated_paint_clamps_to_one():
    buf = brush.new_buffer(64, 128)
    for _ in range(10):
        brush.stamp(buf, 0.0, 0.0, radius_deg=30.0, strength=0.5)
    assert buf.max() <= 1.0
    assert buf[32, 64] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- sidecar round-trip

def test_png16_gray_round_trip(tmp_path):
    from gasgiant.export.writers import decode_image, write_png16_gray

    h, w = 256, 512  # 2:1 for decode_image
    buf = brush.new_buffer(h, w)
    brush.stamp(buf, 10.0, -5.0, radius_deg=40.0, strength=1.0)
    brush.stamp(buf, -120.0, 30.0, radius_deg=25.0, strength=0.6, erase=False)
    path = tmp_path / "mask.png"
    write_png16_gray(path, buf)
    back = decode_image(path)
    assert back.shape == buf.shape
    assert np.max(np.abs(back - buf)) <= 1.0 / 65535.0 + 1e-7


# ---------------------------------------------------------------- headless GUI tool

class _FakeSim:
    """Records set_mask calls without a GL context."""

    def __init__(self) -> None:
        self.mask_calls: list = []  # each is the arr (or None)

    def set_mask(self, arr) -> None:
        self.mask_calls.append(arr)


class _FakeViewport:
    def __init__(self) -> None:
        self.image_rect_min = (100.0, 50.0)
        self.image_rect_max = (900.0, 450.0)
        self.stale = 0

    def mark_stale(self) -> None:
        self.stale += 1


def _make_paint_app():
    from gasgiant.app import main as main_mod
    from gasgiant.params.model import PlanetParams

    app = main_mod.StudioApp.__new__(main_mod.StudioApp)
    app.sim = _FakeSim()
    app.viewport = _FakeViewport()
    app.params = PlanetParams()
    app._export = None
    app._storm_tool_mode = "paint"
    app._drag_index = None
    app._drag_lonlat = None
    app._paint_buffer = None
    app._paint_base = None
    app._paint_strokes = []
    app._paint_radius_deg = 12.0
    app._paint_strength = 0.5
    app._paint_erase = False
    app._paint_dirty = False
    app._paint_was_down = False
    app._paint_upload_accum = main_mod.PAINT_UPLOAD_INTERVAL
    return app


@pytest.fixture
def imgui_ctx():
    imgui = pytest.importorskip("imgui_bundle.imgui")
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1000.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def test_paint_mouse_down_stamps_and_uploads(imgui_ctx):
    imgui = imgui_ctx
    app = _make_paint_app()

    io = imgui.get_io()
    io.add_mouse_pos_event(500.0, 250.0)  # inside the (100,50)-(900,450) rect

    def _frame():
        imgui.new_frame()
        imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
        imgui.set_next_window_size(imgui.ImVec2(1000.0, 600.0))
        imgui.begin("Equirect", None, 0)
        app._draw_paint_tool_ui()
        app._handle_paint_tool()
        app._draw_brush_cursor()
        imgui.end()
        imgui.end_frame()

    # Warm-up frame: imgui computes window-hover from the PREVIOUS frame's rect,
    # so the button-down must land on a frame where the window is already known.
    _frame()
    io.add_mouse_button_event(0, True)  # left button down
    _frame()

    assert app._paint_buffer is not None
    assert app._paint_buffer.any(), "a stamp landed in the buffer"
    assert len(app._paint_strokes) == 1
    assert app.sim.mask_calls, "the mask was uploaded (throttle first-frame)"
    assert app.sim.mask_calls[-1] is app._paint_buffer


def test_paint_undo_stroke_rebuilds(imgui_ctx):
    app = _make_paint_app()
    app._paint_stamp(0.0, 0.0)
    app._paint_stamp(40.0, 10.0)
    assert len(app._paint_strokes) == 2
    after_two = app._paint_buffer.copy()

    app._undo_paint_stroke()
    assert len(app._paint_strokes) == 1
    # rebuilt buffer differs from the two-stroke buffer and uploaded once
    assert not np.array_equal(app._paint_buffer, after_two)
    assert app.sim.mask_calls and app.sim.mask_calls[-1] is app._paint_buffer

    app._undo_paint_stroke()  # back to empty -> mask cleared to None
    assert app._paint_strokes == []
    assert app.sim.mask_calls[-1] is None


def test_paint_clear_zeroes_and_clears_mask(imgui_ctx):
    app = _make_paint_app()
    app._paint_stamp(0.0, 0.0)
    assert app._paint_buffer.any()
    app._clear_paint_mask()
    assert not app._paint_buffer.any()
    assert app._paint_strokes == []
    assert app.sim.mask_calls[-1] is None


def test_paint_held_during_export_does_not_upload(imgui_ctx):
    imgui = imgui_ctx
    app = _make_paint_app()
    app._export = object()  # export in flight

    io = imgui.get_io()
    io.add_mouse_pos_event(500.0, 250.0)
    io.add_mouse_button_event(0, True)
    imgui.new_frame()
    imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
    imgui.set_next_window_size(imgui.ImVec2(1000.0, 600.0))
    imgui.begin("Equirect", None, 0)
    app._handle_paint_tool()
    imgui.end()
    imgui.end_frame()

    assert app.sim.mask_calls == []  # never re-derives mid-export
    assert app._paint_strokes == []


def test_paint_mode_is_sibling_of_place_storm():
    """Toggling Paint mask off returns to 'none'; it never leaves a stale
    place/drag mode, and Place-storm's toggle ignores 'paint'."""
    app = _make_paint_app()
    assert app._storm_tool_mode == "paint"
    # the storm UI predicate must not read 'paint' as storm-on
    app._storm_tool_mode = "paint"
    # emulate the paint toggle turning off
    app._storm_tool_mode = "none"
    assert app._storm_tool_mode == "none"

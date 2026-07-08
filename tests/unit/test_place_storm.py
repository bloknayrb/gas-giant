"""T4 click-to-place / drag-to-move storms: the pure equirect<->screen mapping
helpers, the wrap-aware marker picker, and a headless click->append test."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from gasgiant.app.viewport import (
    lonlat_to_screen,
    nearest_cast_index,
    screen_to_lonlat,
)

# A displayed image rect: 800x400 px, offset so tests exercise a non-zero origin.
RMIN = (100.0, 50.0)
RMAX = (900.0, 450.0)  # width 800, height 400 (equirect 2:1)


@dataclass
class FakeCast:
    lat_deg: float
    lon_deg: float


# ---------------------------------------------------------------- screen_to_lonlat

def test_screen_to_lonlat_center():
    cx = (RMIN[0] + RMAX[0]) / 2
    cy = (RMIN[1] + RMAX[1]) / 2
    lon, lat = screen_to_lonlat(cx, cy, RMIN, RMAX)
    assert lon == pytest.approx(0.0)
    assert lat == pytest.approx(0.0)


def test_screen_to_lonlat_corners():
    lon, lat = screen_to_lonlat(RMIN[0], RMIN[1], RMIN, RMAX)
    assert (lon, lat) == pytest.approx((-180.0, 90.0))
    lon, lat = screen_to_lonlat(RMAX[0], RMAX[1], RMIN, RMAX)
    assert (lon, lat) == pytest.approx((180.0, -90.0))


def test_screen_to_lonlat_clamps_outside():
    # far below/right of the rect clamps into range
    lon, lat = screen_to_lonlat(5000.0, 5000.0, RMIN, RMAX)
    assert (lon, lat) == pytest.approx((180.0, -90.0))
    # far above/left clamps to the top-left corner
    lon, lat = screen_to_lonlat(-5000.0, -5000.0, RMIN, RMAX)
    assert (lon, lat) == pytest.approx((-180.0, 90.0))


def test_screen_to_lonlat_known_interior():
    # 1/4 across, 1/4 down -> lon -90, lat +45
    mx = RMIN[0] + 0.25 * 800
    my = RMIN[1] + 0.25 * 400
    lon, lat = screen_to_lonlat(mx, my, RMIN, RMAX)
    assert (lon, lat) == pytest.approx((-90.0, 45.0))


def test_screen_to_lonlat_zero_size_rect():
    # degenerate rect must not divide by zero
    lon, lat = screen_to_lonlat(10.0, 10.0, (10.0, 10.0), (10.0, 10.0))
    assert (lon, lat) == pytest.approx((-180.0, 90.0))


# ---------------------------------------------------------------- round trip

@pytest.mark.parametrize(
    "lon,lat",
    [(0.0, 0.0), (-90.0, 45.0), (137.0, -22.5), (179.0, 12.3), (-45.0, -60.0)],
)
def test_lonlat_screen_round_trip(lon, lat):
    sx, sy = lonlat_to_screen(lon, lat, RMIN, RMAX)
    # interior points are inside the rect
    assert RMIN[0] <= sx <= RMAX[0]
    assert RMIN[1] <= sy <= RMAX[1]
    lon2, lat2 = screen_to_lonlat(sx, sy, RMIN, RMAX)
    assert (lon2, lat2) == pytest.approx((lon, lat))


def test_lonlat_to_screen_wrapped_maps_outside():
    # a wrapped duplicate longitude falls outside the rect (for edge drawing)
    sx, _ = lonlat_to_screen(179.0 - 360.0, 0.0, RMIN, RMAX)
    assert sx < RMIN[0]


# ---------------------------------------------------------------- nearest_cast_index

def test_nearest_picks_true_nearest():
    cast = [FakeCast(0.0, 0.0), FakeCast(40.0, 40.0), FakeCast(-30.0, 120.0)]
    assert nearest_cast_index(38.0, 42.0, cast) == 1
    assert nearest_cast_index(2.0, -3.0, cast) == 0


def test_nearest_is_dateline_wrap_aware():
    # a point just east of +170 is nearer a marker at -175 (wrap) than +120
    cast = [FakeCast(0.0, -175.0), FakeCast(0.0, 120.0)]
    assert nearest_cast_index(172.0, 0.0, cast) == 0


def test_nearest_empty_is_none():
    assert nearest_cast_index(10.0, 10.0, []) is None


def test_nearest_max_deg_filters():
    cast = [FakeCast(0.0, 0.0)]
    assert nearest_cast_index(3.0, 4.0, cast, max_deg=10.0) == 0  # dist 5 <= 10
    assert nearest_cast_index(30.0, 40.0, cast, max_deg=10.0) is None  # dist 50 > 10


# ------------------------------------------------- headless: click -> append entry

def _make_app_headless():
    """Construct StudioApp without GL and neutralise the engine-touching parts of
    the discrete-action commit so _place_storm_at is exercisable in isolation."""
    from gasgiant.app import main as main_mod

    app = main_mod.StudioApp()
    # _commit normally calls self.sim.update_params(); here just adopt the params.
    app._commit = lambda new: setattr(app, "params", new)
    app._push_history = lambda params: None
    app._reset_working_copy = lambda: None
    return app


def test_place_storm_appends_clamped_entry():
    from gasgiant.params.model import CastKind, hero_latitude_cap

    app = _make_app_headless()
    app.params.storms.cast = []
    app._storm_tool_mode = "place"
    app._storm_tool_kind = CastKind.OVAL.value
    app._storm_tool_radius = 0.03

    before = len(app.params.storms.cast)
    # a latitude far above the radius-coupled cap must clamp down
    app._place_storm_at(37.5, 88.0)

    cast = app.params.storms.cast
    assert len(cast) == before + 1
    entry = cast[-1]
    assert entry.kind == CastKind.OVAL
    assert entry.lon_deg == pytest.approx(37.5)
    cap = hero_latitude_cap(0.03)
    assert entry.lat_deg == pytest.approx(cap)  # clamped to the cap, not 88
    assert abs(entry.lat_deg) <= cap


def test_place_storm_respects_cap_of_16():
    from gasgiant.params.model import CastKind, StormOverride

    app = _make_app_headless()
    app.params.storms.cast = [
        StormOverride(kind=CastKind.OVAL, lat_deg=0.0, lon_deg=float(i), radius=0.03)
        for i in range(16)
    ]
    app._storm_tool_mode = "place"
    app._place_storm_at(10.0, 5.0)
    assert len(app.params.storms.cast) == 16  # no-op when full


def test_place_storm_held_during_export():
    app = _make_app_headless()
    app.params.storms.cast = []
    app._export = object()  # export in flight
    app._place_storm_at(10.0, 5.0)
    assert app.params.storms.cast == []  # never commits mid-export


def test_commit_cast_move_writes_clamped_position():
    from gasgiant.params.model import CastKind, StormOverride, hero_latitude_cap

    app = _make_app_headless()
    app.params.storms.cast = [
        StormOverride(kind=CastKind.OVAL, lat_deg=0.0, lon_deg=0.0, radius=0.03)
    ]
    app._commit_cast_move(0, 200.0, 90.0)  # both out of range
    entry = app.params.storms.cast[0]
    assert entry.lon_deg == pytest.approx(180.0)  # clamped to +180
    assert entry.lat_deg == pytest.approx(hero_latitude_cap(0.03))


# ---------------------------------------------------------------- headless imgui


@pytest.fixture
def imgui_ctx():
    imgui = pytest.importorskip("imgui_bundle.imgui")
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def test_draw_markers_and_tool_ui_run_in_a_frame(imgui_ctx):
    """Exercise the imgui rendering paths (marker overlay + tool toolbar +
    per-frame handler) inside a real frame; they must not raise. Without a
    simulated click the handler leaves the cast list untouched."""
    imgui = imgui_ctx
    from gasgiant.app.viewport import EquirectViewport
    from gasgiant.params.model import CastKind, StormOverride

    vp = EquirectViewport.__new__(EquirectViewport)  # no GL context needed
    vp.image_rect_min = (100.0, 50.0)
    vp.image_rect_max = (900.0, 450.0)

    app = _make_app_headless()
    app.viewport = vp
    app.params.storms.cast = [
        StormOverride(kind=CastKind.HERO, lat_deg=0.0, lon_deg=0.0, radius=0.10),
        StormOverride(kind=CastKind.PEARL, lat_deg=10.0, lon_deg=179.0, radius=0.03),
    ]
    app._storm_tool_mode = "place"

    before = len(app.params.storms.cast)
    imgui.new_frame()
    imgui.begin("Equirect", None, 0)
    vp.draw_markers(
        app.params.storms.cast, drag_index=0, drag_lonlat=(20.0, 5.0)
    )
    app._draw_storm_tool_ui()
    app._handle_storm_tool()  # no mouse click this frame -> no placement
    imgui.end()
    imgui.end_frame()

    assert len(app.params.storms.cast) == before

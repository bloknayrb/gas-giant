"""T4 click-to-place / drag-to-move storms: the pure equirect<->screen mapping
helpers, the wrap-aware marker picker, and a headless click->append test."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from gasgiant.app.viewport import (
    drag_is_noop,
    lonlat_to_screen,
    marker_hit_test,
    nearest_cast_index,
    screen_to_lonlat,
    selection_after_delete,
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


# ---------------------------------------------------------------- marker_hit_test

def test_marker_hit_test_hits_within_threshold():
    cast = [FakeCast(0.0, 0.0)]
    sx, sy = lonlat_to_screen(0.0, 0.0, RMIN, RMAX)  # marker at (lon,lat)=(0,0)
    assert marker_hit_test(sx + 3.0, sy - 3.0, cast, RMIN, RMAX) == 0


def test_marker_hit_test_misses_when_far():
    cast = [FakeCast(0.0, 0.0)]
    sx, sy = lonlat_to_screen(0.0, 0.0, RMIN, RMAX)
    assert marker_hit_test(sx + 60.0, sy, cast, RMIN, RMAX) is None


def test_marker_hit_test_empty_is_none():
    assert marker_hit_test(500.0, 250.0, [], RMIN, RMAX) is None


def test_marker_hit_test_grabs_wrapped_marker_from_far_edge():
    # a marker at lon +179 has a wrapped screen twin near the LEFT edge; a click
    # just off the left edge must still grab it.
    cast = [FakeCast(0.0, 179.0)]
    sx, sy = lonlat_to_screen(179.0 - 360.0, 0.0, RMIN, RMAX)  # wrapped twin
    assert marker_hit_test(sx + 2.0, sy, cast, RMIN, RMAX) == 0


# ---------------------------------------------------------------- drag_is_noop

def test_drag_is_noop_zero_move():
    assert drag_is_noop((12.0, -3.0), (12.0, -3.0)) is True


def test_drag_is_noop_real_move():
    assert drag_is_noop((12.0, -3.0), (12.0, -1.0)) is False  # 2 deg latitude
    assert drag_is_noop((12.0, -3.0), (20.0, -3.0)) is False  # 8 deg longitude


def test_drag_is_noop_wrap_aware():
    # +179.9995 to -179.9995 is a ~0.001 deg step across the dateline, not ~360
    assert drag_is_noop((179.9995, 0.0), (-179.9995, 0.0), eps=0.01) is True


# ---------------------------------------------------------------- selection_after_delete

def test_selection_after_delete_none_selected():
    assert selection_after_delete(None, 2) is None


def test_selection_after_delete_deleting_selected_clears():
    assert selection_after_delete(3, 3) is None


def test_selection_after_delete_before_selection_shifts_down():
    assert selection_after_delete(4, 1) == 3


def test_selection_after_delete_after_selection_unchanged():
    assert selection_after_delete(1, 4) == 1


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


def test_show_markers_defaults_on():
    """The placement-marker overlay is shown by default; the equirect toggle only
    hides it as view state (never committed)."""
    app = _make_app_headless()
    assert app._show_markers is True


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


# ---------------------------------------------------------------- delete + selection

def _cast_of(n):
    from gasgiant.params.model import CastKind, StormOverride

    return [
        StormOverride(kind=CastKind.OVAL, lat_deg=0.0, lon_deg=float(i), radius=0.03)
        for i in range(n)
    ]


def test_delete_selected_removes_and_reconciles():
    app = _make_app_headless()
    app.params.storms.cast = _cast_of(3)
    app._selected_cast = 1
    app._delete_selected_cast()
    cast = app.params.storms.cast
    assert len(cast) == 2
    assert [e.lon_deg for e in cast] == [0.0, 2.0]  # entry 1 gone
    assert app._selected_cast is None  # deleting the selected clears selection


def test_delete_is_noop_mid_export():
    app = _make_app_headless()
    app.params.storms.cast = _cast_of(2)
    app._selected_cast = 0
    app._export = object()
    app._delete_selected_cast()
    assert len(app.params.storms.cast) == 2  # never mutates mid-export


def test_delete_is_noop_when_nothing_selected():
    app = _make_app_headless()
    app.params.storms.cast = _cast_of(2)
    app._selected_cast = None
    app._delete_selected_cast()
    assert len(app.params.storms.cast) == 2


def test_reset_working_copy_clamps_stale_selection():
    # the real reconcile point: after a discrete action shrinks the cast, an
    # out-of-range selection is dropped (undo/redo/preset-load path).
    from gasgiant.app import main as main_mod

    app = main_mod.StudioApp()
    app.params.storms.cast = _cast_of(2)
    app._selected_cast = 5  # stale (e.g. undo swapped in a shorter list)
    app._reset_working_copy()
    assert app._selected_cast is None


def test_reset_working_copy_keeps_valid_selection():
    from gasgiant.app import main as main_mod

    app = main_mod.StudioApp()
    app.params.storms.cast = _cast_of(3)
    app._selected_cast = 2  # in range
    app._reset_working_copy()
    assert app._selected_cast == 2


# ---------------------------------------------------------------- panel bridge


def test_bridge_panel_selection_round_trips(monkeypatch):
    """The panel<->app selection bridge publishes the app's selection into
    panel_state (scroll-requesting it only when the change came from the app/
    viewport), then reads a panel-side click back onto the app. Guards the
    read-back line whose loss would silently break panel->marker selection."""
    from gasgiant.app import main as main_mod

    app = _make_app_headless()
    app._process_edit = lambda draft, changed, committed: None

    # Case 1: an app-side (viewport) selection change scrolls the row into view.
    app._selected_cast = 2
    app.panel_state.selected_cast = None
    app.panel_state.cast_scroll_requested = False
    monkeypatch.setattr(main_mod, "draw_params_panel",
                        lambda live, ps, sim=None: ({}, False, False))
    app._bridge_panel_selection()
    assert app.panel_state.cast_scroll_requested is True  # external change -> scroll
    assert app._selected_cast == 2

    # Case 2: a panel-side row click writes back to the app, WITHOUT a re-scroll.
    app.panel_state.cast_scroll_requested = False

    def panel_clicks_row(live, ps, sim=None):
        ps.selected_cast = 0  # user clicked row 0 in the panel this frame
        return {}, False, False
    monkeypatch.setattr(main_mod, "draw_params_panel", panel_clicks_row)
    app._bridge_panel_selection()
    assert app._selected_cast == 0  # read back from the panel
    assert app.panel_state.cast_scroll_requested is False  # panel-origin -> no scroll


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

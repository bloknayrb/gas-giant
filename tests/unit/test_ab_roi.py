"""T5 A/B compare + ROI export-res inspector: the pure ROI origin math, the
app-owned snapshot-A capture/release contract, the compare-mode predicate, and
the export-gated inspector -- all exercised headlessly (no GL context, and only
the last test needs a headless imgui frame)."""

from __future__ import annotations

import pytest

from gasgiant.export.exporter import TILE, roi_tile_origin

# ---------------------------------------------------------------- roi_tile_origin

def test_roi_origin_centered():
    # 4096x2048 map, center (0.5, 0.5): a 1024 tile centered is at (1536, 512).
    assert roi_tile_origin(0.5, 0.5, 4096, 2048, TILE) == (1536, 512)


def test_roi_origin_clamps_low():
    # center at the top-left corner clamps the origin to (0, 0).
    assert roi_tile_origin(0.0, 0.0, 4096, 2048, TILE) == (0, 0)


def test_roi_origin_clamps_high():
    # center at the bottom-right corner clamps to keep the tile inside the map.
    assert roi_tile_origin(1.0, 1.0, 4096, 2048, TILE) == (4096 - TILE, 2048 - TILE)


def test_roi_origin_map_smaller_than_tile_is_zero():
    # a map narrower/shorter than a tile pins the origin to 0 on that axis.
    assert roi_tile_origin(0.5, 0.5, 512, 256, TILE) == (0, 0)


def test_roi_origin_is_within_bounds_for_any_center():
    for cx in (0.0, 0.13, 0.5, 0.87, 1.0):
        for cy in (0.0, 0.5, 1.0):
            x0, y0 = roi_tile_origin(cx, cy, 8192, 4096, TILE)
            assert 0 <= x0 <= 8192 - TILE
            assert 0 <= y0 <= 4096 - TILE


# ---------------------------------------------------------------- fakes / helpers

class _FakeTexture:
    """A stand-in preview clone that records its own release (leak detection)."""

    _counter = 0

    def __init__(self) -> None:
        _FakeTexture._counter += 1
        self.glo = _FakeTexture._counter
        self.released = False

    def release(self) -> None:
        self.released = True


class _FakeSim:
    def __init__(self) -> None:
        self.snapshots: list[_FakeTexture] = []
        self.create_snapshot_calls = 0

    def snapshot_preview_color(self) -> _FakeTexture:
        tex = _FakeTexture()
        self.snapshots.append(tex)
        return tex

    def create_snapshot(self):  # must never be called while gated
        self.create_snapshot_calls += 1
        raise AssertionError("create_snapshot must not run when export-gated")


def _make_app_headless():
    """Construct StudioApp without GL (mirrors test_place_storm's helper)."""
    from gasgiant.app import main as main_mod

    app = main_mod.StudioApp()
    app.sim = _FakeSim()
    return app


# ---------------------------------------------------------------- snapshot A state

def test_take_snapshot_a_sets_app_state():
    app = _make_app_headless()
    assert app._snapshot_a is None
    app._take_snapshot_a()
    assert app._snapshot_a is not None
    assert app._snapshot_a is app.sim.snapshots[0]


def test_retake_releases_previous_snapshot():
    app = _make_app_headless()
    app._take_snapshot_a()
    first = app._snapshot_a
    app._take_snapshot_a()
    second = app._snapshot_a
    assert first is not second
    assert first.released is True  # old snapshot freed -> no leak on retake
    assert second.released is False


def test_snapshot_failure_keeps_previous():
    app = _make_app_headless()
    app._take_snapshot_a()
    held = app._snapshot_a

    def _boom():
        raise RuntimeError("no preview yet")

    app.sim.snapshot_preview_color = _boom
    app._take_snapshot_a()
    assert app._snapshot_a is held  # unchanged
    assert held.released is False   # not freed on failure


# ---------------------------------------------------------------- compare predicate

def test_compare_inactive_without_snapshot():
    app = _make_app_headless()
    app._compare_mode = "split"
    assert app._snapshot_a is None
    assert app._compare_active() is False  # a mode with no snapshot A is inert


def test_compare_mode_transitions():
    app = _make_app_headless()
    app._take_snapshot_a()
    for mode, expected in (("off", False), ("flash", True), ("split", True)):
        app._compare_mode = mode
        assert app._compare_active() is expected


# ---------------------------------------------------------------- inspector gating

def test_inspect_gated_during_export():
    app = _make_app_headless()
    app._export = object()  # export in flight
    app._run_inspect()
    assert app._inspect_tile is None                 # nothing rendered
    assert app.sim.create_snapshot_calls == 0        # snapshot never taken


# ---------------------------------------------------------------- headless imgui

@pytest.fixture
def imgui_ctx():
    imgui = pytest.importorskip("imgui_bundle.imgui")
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(900.0, 700.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def test_compare_and_inspect_controls_run_in_a_frame(imgui_ctx):
    """The compare toolbar and the inspector window draw inside a real frame
    without raising, in every compare mode, and with no snapshot A held (the
    disabled path). No GL is touched: no button is clicked."""
    imgui = imgui_ctx
    app = _make_app_headless()
    for mode in ("off", "flash", "split"):
        app._compare_mode = mode
        imgui.new_frame()
        imgui.begin("Equirect", None, 0)
        app._draw_compare_controls()
        imgui.end()
        app._show_inspect = True
        app._draw_inspect_window()
        imgui.end_frame()

"""B1-8: preset overwrite/delete affordances.

Ctrl+S overwrites the active USER preset (after a confirm modal) instead of
always opening a dialog; user presets can be deleted in-app (confirm first,
params stay loaded). Driven headlessly against StudioApp with the
test_shortcuts.py fake-app pattern.
"""

from __future__ import annotations

from collections import deque

import pytest

from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import PresetSource, load_preset

main = pytest.importorskip("gasgiant.app.main")
panels = pytest.importorskip("gasgiant.app.panels")
imgui = pytest.importorskip("imgui_bundle.imgui")


@pytest.fixture
def imgui_ctx():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    """Point the app's user-preset namespace at a temp dir and keep
    _refresh_presets off the real home directory."""
    monkeypatch.setattr(main, "USER_PRESET_DIR", tmp_path)
    monkeypatch.setattr(main, "available_presets", lambda: [])
    return tmp_path


def _make_app(active=None):
    app = main.StudioApp.__new__(main.StudioApp)
    app.params = PlanetParams()
    app._live = app.params
    app._gesture_base = None
    app._undo_stack = deque(maxlen=64)
    app._redo_stack = deque(maxlen=64)
    app._active_preset = active
    app._pristine = None
    app._dialog = None
    app._export = None
    app._pending_export = None
    app._pending_overwrite = None
    app._pending_delete = None
    app._preset_cache = []
    app.panel_state = panels.PanelState()
    app.toasts = main.Toasts()
    return app


# -- _active_user_preset_path ---------------------------------------------------


def test_active_user_preset_path_none_for_unnamed_and_factory(user_dir):
    assert _make_app(None)._active_user_preset_path() is None
    factory = _make_app(("gas_giant_warm", PresetSource.FACTORY))
    assert factory._active_user_preset_path() is None
    file_src = _make_app(("somewhere", PresetSource.FILE))
    assert file_src._active_user_preset_path() is None


def test_active_user_preset_path_for_user_preset(user_dir):
    app = _make_app(("mine", PresetSource.USER))
    assert app._active_user_preset_path() == ("mine", user_dir / "mine.json")


# -- _request_overwrite_active ----------------------------------------------------


def test_request_overwrite_falls_back_to_save_as(user_dir, monkeypatch):
    """No active user preset (factory/unsaved): Ctrl+S keeps its old
    Save-As behavior instead of silently no-op'ing."""
    app = _make_app(("gas_giant_warm", PresetSource.FACTORY))
    calls: list[bool] = []
    monkeypatch.setattr(
        main.StudioApp, "_open_save_dialog", lambda self: calls.append(True)
    )
    app._request_overwrite_active()
    assert calls == [True]
    assert app._pending_overwrite is None


def test_request_overwrite_stages_confirm_for_user_preset(user_dir):
    app = _make_app(("mine", PresetSource.USER))
    app._request_overwrite_active()
    assert app._pending_overwrite == ("mine", user_dir / "mine.json")


def test_request_overwrite_gated_during_export(user_dir):
    app = _make_app(("mine", PresetSource.USER))
    app._export = main.ExportJob(object(), object())
    app._request_overwrite_active()
    assert app._pending_overwrite is None


# -- _overwrite_active_preset -------------------------------------------------------


def test_overwrite_saves_file_and_clears_dirty(user_dir):
    app = _make_app(("mine", PresetSource.USER))
    # simulate an edited (dirty) state vs an older on-disk version
    old = PlanetParams()
    old.seed = 1
    (user_dir / "mine.json").write_text("{}")  # stale placeholder to be overwritten
    app._pristine = old
    app.params.seed = 42
    assert app._is_dirty()

    app._pending_overwrite = ("mine", user_dir / "mine.json")
    app._overwrite_active_preset()

    assert app._pending_overwrite is None
    reloaded = load_preset(user_dir / "mine.json")
    assert reloaded.seed == 42
    assert app._active_preset == ("mine", PresetSource.USER)
    assert not app._is_dirty(), "overwrite adopts the saved params as pristine"


def test_overwrite_noop_without_pending(user_dir):
    app = _make_app(("mine", PresetSource.USER))
    app._overwrite_active_preset()  # nothing staged: no crash, no file
    assert not (user_dir / "mine.json").exists()


# -- _delete_active_preset ------------------------------------------------------------


def test_delete_removes_file_and_degrades_identity(user_dir):
    app = _make_app(("mine", PresetSource.USER))
    path = user_dir / "mine.json"
    path.write_text("{}")
    before = app.params
    app._request_delete_active()
    assert app._pending_delete == ("mine", path)

    app._delete_active_preset()
    assert not path.exists()
    assert app._active_preset is None, "identity degrades to 'unsaved'"
    assert app.params is before, "the loaded params are NOT yanked away"
    assert app._pending_delete is None


def test_request_delete_noop_for_factory(user_dir):
    app = _make_app(("gas_giant_warm", PresetSource.FACTORY))
    app._request_delete_active()
    assert app._pending_delete is None


# -- confirm modals draw headlessly ----------------------------------------------------


def test_confirm_modals_draw_without_consuming(imgui_ctx, user_dir):
    app = _make_app(("mine", PresetSource.USER))
    app._pending_overwrite = ("mine", user_dir / "mine.json")
    imgui.new_frame()
    imgui.begin("w", None, 0)
    app._draw_preset_confirm_modals()
    imgui.end()
    imgui.end_frame()
    assert app._pending_overwrite is not None, "drawing alone must not confirm"

    app._pending_overwrite = None
    app._pending_delete = ("mine", user_dir / "mine.json")
    imgui.new_frame()
    imgui.begin("w", None, 0)
    app._draw_preset_confirm_modals()
    imgui.end()
    imgui.end_frame()
    assert app._pending_delete is not None

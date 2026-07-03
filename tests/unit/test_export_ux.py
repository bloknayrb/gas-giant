"""B1-4 + B1-5: export UX — file list, overwrite confirm, last-export
persistence/open-folder, hold-notice wording, pending-hint wording.

Pure helpers are unit-tested directly; the dialog->confirm->start flow is
driven through _poll_dialog/_run_export_slice with fakes (the
test_shortcuts.py headless pattern); the new modal gets a headless-imgui
smoke frame (the test_phase8_ui.py pattern).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

from gasgiant.jobs import Progress
from gasgiant.params.model import PlanetParams, Tier

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


def _make_app(params: PlanetParams | None = None):
    params = params or PlanetParams()
    app = main.StudioApp.__new__(main.StudioApp)
    app.params = params
    app._live = params
    app._gesture_base = None
    app._undo_stack = deque(maxlen=64)
    app._redo_stack = deque(maxlen=64)
    app._active_preset = None
    app._pristine = None
    app._dialog = None
    app._export = None
    app._pending_export = None
    app._pending_overwrite = None
    app._pending_delete = None
    app._last_export_dir = None
    app.init_error = None
    app.sim = object()  # only identity-checked by the flows under test
    app.panel_state = panels.PanelState()
    app.toasts = main.Toasts()
    return app


# -- _export_file_lines -----------------------------------------------------------


def test_export_file_lines_default_no_emission():
    params = PlanetParams()
    params.emission.thermal_strength = 0.0
    params.emission.lightning_strength = 0.0
    params.emission.aurora_strength = 0.0
    lines = main._export_file_lines(params)
    joined = "\n".join(lines)
    assert "color.png" in joined
    assert "height.exr" in joined
    assert "mapset.json" in joined
    assert "Blender" in joined  # the next-step pointer
    assert "emission.exr" not in joined
    w = params.export.width
    assert f"{w}x{w // 2}" in joined


def test_export_file_lines_with_emission():
    params = PlanetParams()
    params.emission.thermal_strength = 0.5
    assert params.emission.enabled
    joined = "\n".join(main._export_file_lines(params))
    assert "emission.exr" in joined


# -- _export_conflicts --------------------------------------------------------------


def test_export_conflicts_empty_and_missing_dir(tmp_path):
    assert main._export_conflicts(tmp_path) == []
    assert main._export_conflicts(tmp_path / "nope") == []


def test_export_conflicts_lists_existing_mapset_files(tmp_path):
    (tmp_path / "color.png").write_bytes(b"x")
    (tmp_path / "mapset.json").write_text("{}")
    (tmp_path / "unrelated.txt").write_text("keep")
    assert main._export_conflicts(tmp_path) == ["color.png", "mapset.json"]


# -- progress label + hold notice (B1-5) --------------------------------------------


def test_export_progress_label_states():
    assert main._export_progress_label(None) == "starting"
    assert main._export_progress_label(Progress(5, 100, "developing")) == "finishing dev run 5/100"
    assert main._export_progress_label(Progress(3, 20, "tile 3/18")) == "tile 3/18"


def test_export_hold_notice_wording():
    plain = main._export_hold_notice(False)
    dev = main._export_hold_notice(True)
    assert "paused" in plain and "pending edits" in plain
    assert dev.startswith("Export is finishing the development run first.")
    assert plain in dev


# -- _pending_hint_text (B1-5: no tier jargon) ---------------------------------------


def test_pending_hint_none_when_nothing_heavy():
    assert main._pending_hint_text(set()) is None
    assert main._pending_hint_text({Tier.POST}) is None


def test_pending_hint_velocity_wording():
    hint = main._pending_hint_text({Tier.VELOCITY})
    assert hint is not None
    assert "release" in hint
    assert "restart/velocity" not in hint  # the old jargon string is gone


def test_pending_hint_restart_wins():
    hint = main._pending_hint_text({Tier.RESTART, Tier.VELOCITY, Tier.POST})
    assert hint is not None
    assert "development run" in hint


# -- last-export persistence ----------------------------------------------------------


def test_load_last_export_dir_roundtrip(tmp_path):
    store: dict[str, str] = {}
    main._save_last_export_dir(tmp_path, save=store.__setitem__)
    assert store[main._LAST_EXPORT_PREF] == str(tmp_path)
    loaded = main._load_last_export_dir(load=store.__getitem__)
    assert loaded == tmp_path


def test_load_last_export_dir_rejects_missing_dir(tmp_path):
    gone = tmp_path / "deleted"
    assert main._load_last_export_dir(load=lambda _k: str(gone)) is None
    assert main._load_last_export_dir(load=lambda _k: "") is None


def test_load_last_export_dir_swallows_pref_errors():
    def boom(_key):
        raise RuntimeError("no runner")

    assert main._load_last_export_dir(load=boom) is None


def test_save_last_export_dir_swallows_pref_errors(tmp_path):
    def boom(_key, _value):
        raise RuntimeError("no runner")

    main._save_last_export_dir(tmp_path, save=boom)  # must not raise


def test_open_folder_returns_false_on_failure(monkeypatch, tmp_path):
    import sys

    if sys.platform == "win32":
        monkeypatch.setattr(main.os, "startfile", _raise_oserror, raising=False)
    else:
        monkeypatch.setattr(main.subprocess, "Popen", _raise_oserror)
    assert main._open_folder(tmp_path) is False


def _raise_oserror(*_a, **_k):
    raise OSError("boom")


# -- dialog -> conflict-confirm -> start flow ------------------------------------------


class _FakeDialog:
    def __init__(self, result):
        self._result = result

    def ready(self):
        return True

    def result(self):
        return self._result


def test_export_dialog_starts_job_when_folder_clean(tmp_path, monkeypatch):
    app = _make_app()
    started: list[Path] = []
    monkeypatch.setattr(
        main.StudioApp, "_start_export", lambda self, out: started.append(out)
    )
    app._dialog = (main.DialogKind.EXPORT, _FakeDialog(str(tmp_path)))
    app._poll_dialog()
    assert started == [tmp_path]
    assert app._pending_export is None
    assert app._dialog is None


def test_export_dialog_holds_on_conflicts(tmp_path, monkeypatch):
    (tmp_path / "mapset.json").write_text("{}")
    app = _make_app()
    started: list[Path] = []
    monkeypatch.setattr(
        main.StudioApp, "_start_export", lambda self, out: started.append(out)
    )
    app._dialog = (main.DialogKind.EXPORT, _FakeDialog(str(tmp_path)))
    app._poll_dialog()
    assert started == [], "job must not start until the overwrite is confirmed"
    assert app._pending_export == (tmp_path, ["mapset.json"])


def test_export_success_records_and_persists_last_dir(tmp_path, monkeypatch):
    app = _make_app()
    saved: list[Path] = []
    monkeypatch.setattr(main, "_save_last_export_dir", lambda p: saved.append(p))

    def _done_job():
        return
        yield  # pragma: no cover - a generator that immediately StopIterations

    app._export = main.ExportJob(_done_job(), tmp_path)
    app._run_export_slice()
    assert app._export is None
    assert app._last_export_dir == tmp_path
    assert saved == [tmp_path]


# -- headless draw smoke: overwrite confirm + last-export line + modal file list -------


def test_overwrite_confirm_modal_draws(imgui_ctx, tmp_path):
    app = _make_app()
    app._pending_export = (tmp_path, ["color.png", "mapset.json"])
    imgui.new_frame()
    imgui.begin("w", None, 0)
    app._draw_export_overwrite_confirm()
    imgui.end()
    imgui.end_frame()
    assert app._pending_export is not None, "drawing alone must not consume the confirm"


def test_last_export_line_draws(imgui_ctx, tmp_path):
    app = _make_app()
    app._last_export_dir = tmp_path
    imgui.new_frame()
    imgui.begin("w", None, 0)
    app._draw_last_export_line()
    imgui.end()
    imgui.end_frame()


def test_export_modal_with_file_list_draws(imgui_ctx):
    app = _make_app()
    app._commit = lambda new_params: setattr(app, "params", new_params)
    app._reset_working_copy = lambda: None
    imgui.new_frame()
    imgui.begin("w", None, 0)
    imgui.open_popup("Export map set")
    app._draw_export_modal()
    imgui.end()
    imgui.end_frame()
    assert app._dialog is None

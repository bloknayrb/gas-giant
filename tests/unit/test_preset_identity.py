"""Phase 6 active-preset / dirty-indicator tracking.

Drives StudioApp's identity logic directly (no GL context / imgui frame): the
combo-load path (``_load_preset_entry``), the Save/Load branches of
``_poll_dialog``, the ``_active_label``/``_is_dirty`` helpers, and undo/redo
restoring the identity + pristine baseline. Mirrors test_undo_redo.py's harness.
"""

from __future__ import annotations

from collections import deque

import pytest

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params import presets as presets_mod
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.params.presets import save_preset

main = pytest.importorskip("gasgiant.app.main")
StudioApp = main.StudioApp


class FakeSim:
    def __init__(self, params: PlanetParams) -> None:
        self.params = params

    def update_params(self, new: PlanetParams) -> set[Tier]:
        tiers = diff_tiers(self.params, new)
        self.params = new
        return tiers


class FakeViewport:
    def mark_stale(self) -> None:
        pass


def _make_app(params: PlanetParams | None = None) -> StudioApp:
    params = params or PlanetParams()
    app = StudioApp.__new__(StudioApp)
    app.params = params
    app._live = params
    app._gesture_base = None
    app._recomputing = False
    app._undo_stack = deque(maxlen=64)
    app._redo_stack = deque(maxlen=64)
    app._active_preset = None
    app._pristine = None
    app._preset_cache = []
    app._dialog = None
    app._export = None
    app.sim = FakeSim(params)
    app.viewport = FakeViewport()
    app.toasts = main.Toasts()
    return app


# -- combo load sets identity + pristine ----------------------------------------


def test_combo_load_sets_identity_and_pristine_clean() -> None:
    app = _make_app()
    app._load_preset_entry("gas_giant_warm", "factory")
    assert app._active_preset == ("gas_giant_warm", "factory")
    assert app._pristine is not None
    assert not app._is_dirty(), "right after a load the state is clean"
    assert app._active_label() == "gas_giant_warm"
    # one undo entry pushed (the pre-load state)
    assert len(app._undo_stack) == 1


def test_edit_after_load_reads_dirty() -> None:
    app = _make_app()
    app._load_preset_entry("gas_giant_warm", "factory")
    app.params = app.params.model_copy(deep=True)
    app.params.appearance.gamma += 0.3
    assert app._is_dirty()
    assert app._active_label() == "gas_giant_warm *"


def test_user_preset_label_prefixed() -> None:
    app = _make_app()
    app._active_preset = ("my_look", "user")
    app._pristine = app.params.model_copy(deep=True)
    assert app._active_label() == "user/my_look"


def test_no_preset_label_is_unsaved_never_starred() -> None:
    app = _make_app()  # _active_preset None, _pristine None
    assert app._active_label() == "unsaved"
    # even after an edit, a None baseline never reads dirty (no star noise)
    app.params = app.params.model_copy(deep=True)
    app.params.appearance.gamma += 0.3
    assert not app._is_dirty()
    assert app._active_label() == "unsaved"


# -- Save sets a fresh clean baseline -------------------------------------------


def test_save_dialog_sets_identity_and_resets_dirty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(presets_mod, "USER_PRESET_DIR", tmp_path)
    monkeypatch.setattr(main, "USER_PRESET_DIR", tmp_path)
    app = _make_app()
    # start dirty relative to some prior baseline
    app._active_preset = ("gas_giant_warm", "factory")
    app._pristine = PlanetParams(seed=123)  # differs from app.params
    assert app._is_dirty()

    save_path = tmp_path / "my_look.json"

    class FakeDialog:
        def ready(self) -> bool:
            return True

        def result(self) -> str:
            return str(save_path)

    app._dialog = ("save", FakeDialog())
    app._poll_dialog()

    assert save_path.is_file()
    assert app._active_preset == ("my_look", "user"), "saved into USER_PRESET_DIR"
    assert not app._is_dirty(), "save adopts current params as the clean baseline"


# -- Load file-dialog sets a file-sourced identity ------------------------------


def test_load_dialog_sets_file_identity(tmp_path) -> None:
    other = PlanetParams(seed=7)
    preset_file = tmp_path / "from_disk.json"
    save_preset(other, preset_file)

    app = _make_app()

    class FakeDialog:
        def ready(self) -> bool:
            return True

        def result(self) -> list[str]:
            return [str(preset_file)]

    app._dialog = ("load", FakeDialog())
    app._poll_dialog()

    assert app._active_preset == ("from_disk", "file")
    assert not app._is_dirty()
    assert app.params.seed == 7


# -- randomize / reroll leave identity alone ------------------------------------


def test_randomize_leaves_identity_but_reads_dirty() -> None:
    from gasgiant.app.panels import PanelState

    app = _make_app()
    app.panel_state = PanelState()
    app._load_preset_entry("gas_giant_warm", "factory")
    assert not app._is_dirty()

    # replicate the Randomize button body (draw_controls): NO identity change
    app._push_history(app.params)
    app._commit(app._randomize(12345))
    app._reset_working_copy()

    assert app._active_preset == ("gas_giant_warm", "factory"), "identity unchanged"
    assert app._is_dirty(), "randomized params differ from the loaded baseline"


# -- undo across a load restores identity + dirty -------------------------------


def test_undo_across_load_restores_identity_and_dirty() -> None:
    app = _make_app()
    # epoch A: load a factory preset, then edit it dirty
    app._load_preset_entry("gas_giant_warm", "factory")
    base_a = app._pristine
    app._push_history(app.params)  # emulate an edit's coalesced push
    edited = app.params.model_copy(deep=True)
    edited.appearance.gamma += 0.5
    app._commit(edited)
    assert app._is_dirty()

    # epoch B: load a different preset (clean baseline)
    app._load_preset_entry("jupiter_like", "factory")
    assert app._active_preset == ("jupiter_like", "factory")
    assert not app._is_dirty()

    # undo the epoch-B load: back to the edited epoch-A state
    app._undo()
    assert app._active_preset == ("gas_giant_warm", "factory"), "restored active name"
    assert app._pristine == base_a, "restored the epoch-A baseline"
    assert app._is_dirty(), "the restored state was dirty vs epoch A"

    # redo re-applies epoch B (clean again)
    app._redo()
    assert app._active_preset == ("jupiter_like", "factory")
    assert not app._is_dirty()


# -- B4-5: Import preset... (dialog flow) -----------------------------------------


class _FakeDialog:
    def __init__(self, path) -> None:
        self._path = path

    def ready(self) -> bool:
        return True

    def result(self) -> list[str]:
        return [str(self._path)]


def test_import_dialog_installs_user_preset_and_adopts_identity(
    tmp_path, monkeypatch
) -> None:
    """The Import... flow: validate, copy into the user preset dir, refresh
    the dropdown, adopt USER identity (clean), push one undo entry."""
    user_dir = tmp_path / "user"
    monkeypatch.setattr(presets_mod, "USER_PRESET_DIR", user_dir)

    other = PlanetParams(seed=321)
    other.bands.count = 9
    src = tmp_path / "somewhere" / "cool_look.json"
    src.parent.mkdir()
    save_preset(other, src)

    app = _make_app()
    pre_count = app.params.bands.count
    app._dialog = (main.DialogKind.IMPORT, _FakeDialog(src))
    app._poll_dialog()

    assert app.params.bands.count == 9, "the imported preset was adopted"
    assert (user_dir / "cool_look.json").is_file(), "installed as a user preset"
    assert app._active_preset == ("cool_look", "user")
    assert not app._is_dirty(), "freshly imported = clean baseline"
    assert ("cool_look", "user") in app._preset_cache, "dropdown refreshed"
    assert len(app._undo_stack) == 1
    assert app._undo_stack[-1][0].bands.count == pre_count


def test_import_dialog_collision_toasts_and_leaves_state_untouched(
    tmp_path, monkeypatch
) -> None:
    user_dir = tmp_path / "user"
    monkeypatch.setattr(presets_mod, "USER_PRESET_DIR", user_dir)
    save_preset(PlanetParams(seed=1), user_dir / "mine.json")
    src = tmp_path / "mine.json"
    save_preset(PlanetParams(seed=2), src)

    app = _make_app()
    before = app.params
    app._dialog = (main.DialogKind.IMPORT, _FakeDialog(src))
    app._poll_dialog()

    assert app.params is before, "failed import commits nothing"
    assert app._active_preset is None
    assert len(app._undo_stack) == 0, "no stray undo entry on failure"
    assert app.toasts._items and app.toasts._items[-1][1] is True, "an error toast"


def test_import_dialog_held_while_export_in_flight(tmp_path, monkeypatch) -> None:
    """Same mid-export hold as Load: the picked file is not consumed while an
    export is running (applying it would commit mid-export)."""
    monkeypatch.setattr(presets_mod, "USER_PRESET_DIR", tmp_path / "user")
    src = tmp_path / "look.json"
    save_preset(PlanetParams(seed=7), src)

    app = _make_app()
    app._export = object()
    dialog = (main.DialogKind.IMPORT, _FakeDialog(src))
    app._dialog = dialog
    app._poll_dialog()

    assert app._dialog is dialog, "dialog result held, not consumed, mid-export"
    assert not (tmp_path / "user").exists(), "nothing installed mid-export"

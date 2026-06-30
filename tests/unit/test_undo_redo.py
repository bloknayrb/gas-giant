"""Phase 2 bounded undo/redo history.

These tests drive StudioApp's history logic directly — the push triggers in
``_process_edit`` (gesture coalescing), the discrete-action pushes, and the
``_undo``/``_redo`` pop/commit methods — without a GL context or imgui frame.
The imgui button-disable wrapper (``_draw_history_buttons``) is covered by
manual verification; its pure logic lives in ``_undo``/``_redo`` and is asserted
here. The ``(draft, any_changed, any_committed)`` panel triple is fed in by hand
exactly as ``test_commit_loop.py`` does.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.params.presets import save_preset

main = pytest.importorskip("gasgiant.app.main")
StudioApp = main.StudioApp


class FakeSim:
    def __init__(self, params: PlanetParams) -> None:
        self.params = params
        self.calls: list[PlanetParams] = []

    def update_params(self, new: PlanetParams) -> set[Tier]:
        tiers = diff_tiers(self.params, new)
        self.params = new
        self.calls.append(new)
        return tiers


class FakeViewport:
    def __init__(self) -> None:
        self.stale = 0

    def mark_stale(self) -> None:
        self.stale += 1


def _make_app(params: PlanetParams | None = None) -> StudioApp:
    params = params or PlanetParams()
    app = StudioApp.__new__(StudioApp)
    app.params = params
    app._live = params
    app._gesture_base = None
    app._recomputing = False
    app._undo_stack = deque(maxlen=64)
    app._redo_stack = deque(maxlen=64)
    app.sim = FakeSim(params)
    app.viewport = FakeViewport()
    return app


def _draft_with(params: PlanetParams, dotted: str, value: object) -> dict:
    draft = params.model_dump()
    node = draft
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node[key]
    node[leaf] = value
    return draft


# -- gesture coalescing: one drag -> exactly one undo entry ----------------------


def test_post_drag_coalesces_into_one_undo_entry() -> None:
    """A POST-tier slider drag commits every frame but must yield ONE undo entry
    captured at gesture start (pins the any_committed-keyed coalescing)."""
    app = _make_app()
    pre = app.params.appearance.gamma

    # many mid-drag frames: each commits live (POST), none release
    for value in (1.1, 1.2, 1.3, 1.4):
        draft = _draft_with(app.params, "appearance.gamma", value)
        app._process_edit(draft, any_changed=True, any_committed=False)
    assert len(app._undo_stack) == 0, "no undo entry mid-drag"

    # release frame
    draft = _draft_with(app.params, "appearance.gamma", 1.4)
    app._process_edit(draft, any_changed=True, any_committed=True)

    assert len(app._undo_stack) == 1, "a whole drag coalesces into one entry"
    record = app._undo_stack[-1]
    assert record[0].appearance.gamma == pytest.approx(pre), "entry is the pre-drag state"
    assert record[1] is None and record[2] is None, "Phase 6 placeholders are None"


def test_heavy_drag_coalesces_into_one_undo_entry() -> None:
    app = _make_app()
    pre = app.params.bands.count
    for value in (15, 16, 17):
        draft = _draft_with(app.params, "bands.count", value)
        app._process_edit(draft, any_changed=True, any_committed=False)
    assert len(app._undo_stack) == 0
    draft = _draft_with(app.params, "bands.count", 17)
    app._process_edit(draft, any_changed=True, any_committed=True)
    assert len(app._undo_stack) == 1
    assert app._undo_stack[-1][0].bands.count == pre


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("bands.count", 22),  # int / RESTART, single-frame discrete
        ("appearance.gamma", 1.7),  # float / POST
        ("solver.sor_omega", 1.5),  # float with cross-field validator
    ],
)
def test_single_frame_edit_yields_one_undo_entry(dotted: str, value: object) -> None:
    """A non-drag edit (checkbox/combo/color/str) reports changed+committed in one
    frame; that must produce exactly one undo entry."""
    app = _make_app()
    draft = _draft_with(app.params, dotted, value)
    app._process_edit(draft, any_changed=True, any_committed=True)
    assert len(app._undo_stack) == 1


def test_composite_palette_edit_yields_one_undo_entry() -> None:
    """A composite edit (add a palette stop) arrives as a single any_committed
    frame and must yield exactly one undo entry, not be absorbed into a neighbor."""
    app = _make_app()
    pre_len = len(app.params.appearance.palette_rows[0].stops)
    draft = app.params.model_dump()
    stops = draft["appearance"]["palette_rows"][0]["stops"]
    stops.append(dict(stops[-1]))  # add a stop (composite mutation)
    app._process_edit(draft, any_changed=True, any_committed=True)
    assert len(app._undo_stack) == 1
    # the stored entry is the PRE-edit state (one fewer stop than now)
    assert len(app._undo_stack[-1][0].appearance.palette_rows[0].stops) == pre_len
    assert len(app.params.appearance.palette_rows[0].stops) == pre_len + 1


# -- no-op gestures push nothing -------------------------------------------------


def test_release_with_no_change_pushes_nothing() -> None:
    """Clicking into a slider and releasing without moving it: any_committed fires
    but self.params == _gesture_base (here _gesture_base is None) -> push nothing."""
    app = _make_app()
    draft = app.params.model_dump()  # identical
    app._process_edit(draft, any_changed=False, any_committed=True)
    assert len(app._undo_stack) == 0


def test_release_equal_value_pushes_nothing() -> None:
    """Even if a gesture base was captured, a release whose committed state equals
    the base (net-zero edit) must not push."""
    app = _make_app()
    base = app.params
    # capture a gesture base by starting an edit, then revert to the same value
    d1 = _draft_with(app.params, "appearance.gamma", base.appearance.gamma + 0.2)
    app._process_edit(d1, any_changed=True, any_committed=False)
    assert app._gesture_base is base
    # revert to the original value on the release frame
    d2 = _draft_with(app.params, "appearance.gamma", base.appearance.gamma)
    app._process_edit(d2, any_changed=True, any_committed=True)
    assert len(app._undo_stack) == 0, "net-zero gesture pushes nothing"


# -- discrete actions push exactly once ------------------------------------------


def test_load_dialog_pushes_one_undo_entry(tmp_path: Path) -> None:
    other = PlanetParams()
    other.bands.count = 7
    preset_file = tmp_path / "loaded.json"
    save_preset(other, preset_file)

    app = _make_app()
    app.toasts = main.Toasts()
    pre_count = app.params.bands.count

    class FakeDialog:
        def ready(self) -> bool:
            return True

        def result(self) -> list[str]:
            return [str(preset_file)]

    app._dialog = ("load", FakeDialog())
    app._poll_dialog()

    assert app.params.bands.count == 7, "the load won"
    assert len(app._undo_stack) == 1, "load pushes exactly one entry"
    assert app._undo_stack[-1][0].bands.count == pre_count, "entry is the pre-load state"


# -- undo / redo round trip ------------------------------------------------------


def test_heavy_edit_then_undo_restores_and_is_redoable() -> None:
    app = _make_app()
    pre = app.params.bands.count
    # a heavy discrete edit
    draft = _draft_with(app.params, "bands.count", 25)
    app._process_edit(draft, any_changed=True, any_committed=True)
    assert app.params.bands.count == 25
    assert len(app._undo_stack) == 1

    app._undo()
    assert app.params.bands.count == pre, "undo restored the pre-edit params"
    assert len(app._undo_stack) == 0
    assert len(app._redo_stack) == 1, "the undone state went onto the redo stack"
    assert app._redo_stack[-1][0].bands.count == 25

    app._redo()
    assert app.params.bands.count == 25, "redo re-applied the edit"
    assert len(app._redo_stack) == 0
    assert len(app._undo_stack) == 1


def test_new_edit_clears_redo_stack() -> None:
    app = _make_app()
    app._process_edit(_draft_with(app.params, "bands.count", 25), True, True)
    app._undo()
    assert len(app._redo_stack) == 1
    # a fresh edit invalidates the redo future
    app._process_edit(_draft_with(app.params, "bands.count", 30), True, True)
    assert len(app._redo_stack) == 0, "a new edit clears redo"
    assert len(app._undo_stack) == 1


# -- bounded stack ---------------------------------------------------------------


def test_undo_stack_evicts_oldest_past_maxlen() -> None:
    app = _make_app()
    assert app._undo_stack.maxlen == 64
    # push 70 distinct records via the real push path (seed varies, stays in range)
    for i in range(70):
        snapshot = app.params.model_copy(deep=True)
        snapshot.seed = i
        app._push_history(snapshot)
    assert len(app._undo_stack) == 64, "deque caps at maxlen, evicting the oldest"
    # entries 0..5 (seeds 0..5) were evicted; the oldest survivor is seed 6
    assert app._undo_stack[0][0].seed == 6, "the six oldest entries were evicted"


# -- entries are deep copies, not shared references ------------------------------


def test_history_entries_are_deep_copies() -> None:
    app = _make_app()
    pre = app.params.bands.count
    app._process_edit(_draft_with(app.params, "bands.count", 25), True, True)
    record_params = app._undo_stack[-1][0]
    assert record_params is not app.params
    assert record_params.bands.count == pre, "entry is the pre-edit snapshot"
    # mutating the live params must not leak into the stored snapshot
    app.params.bands.count = 40
    assert record_params.bands.count == pre

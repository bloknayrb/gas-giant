"""Phase 1 commit-loop state machine: one engine commit per gesture.

These tests exercise StudioApp's commit-decision logic (``_process_edit``),
the working-copy lifecycle (``_live`` / ``_gesture_base``), and the
discrete-action reset directly, without spinning up a GL context or an imgui
frame. The panel's imgui-driven half (reading
``is_item_deactivated_after_edit``) is not unit-testable headless and is
covered by manual verification; what the panel *produces* — the
``(draft, any_changed, any_committed)`` triple — is fed in by hand here so the
whole decision table is asserted.
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
    """Records commits and returns the real invalidation tiers, so _commit's
    mark_stale / _recomputing branches see the same tiers the engine would."""

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
    """A StudioApp with the GL/imgui collaborators faked out — enough to drive
    the commit loop. Bypasses __init__ (which reads the session file)."""
    params = params or PlanetParams()
    app = StudioApp.__new__(StudioApp)
    app.params = params
    app._live = params
    app._gesture_base = None
    app._recomputing = False
    # Phase 2 history stacks: the real __init__ creates these, so the hand-rolled
    # mock must too (the gesture/discrete push paths append to them).
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


# -- POST tier: commits every changed frame, stays frame-live -------------------


def test_post_edit_commits_on_changed_frame_without_release() -> None:
    app = _make_app()
    draft = _draft_with(app.params, "appearance.gamma", 1.4)
    # mid-drag: changed but not yet released
    app._process_edit(draft, any_changed=True, any_committed=False)
    assert len(app.sim.calls) == 1, "POST edit should commit live, not wait for release"
    assert app.params.appearance.gamma == pytest.approx(1.4)
    assert app.params == app._live
    assert app._recomputing is False, "POST commit must not flag a heavy recompute"


def test_post_edit_no_commit_when_value_unchanged() -> None:
    app = _make_app()
    draft = app.params.model_dump()  # identical
    app._process_edit(draft, any_changed=True, any_committed=False)
    assert app.sim.calls == [], "no diff -> no commit"


# -- VELOCITY/RESTART tier: defers the rebuild to release -----------------------


def test_heavy_edit_defers_commit_until_release() -> None:
    app = _make_app()
    base = app.params

    # frames mid-drag: changed, never released -> no commit, edit lives in _live
    for value in (15, 16, 17):
        draft = _draft_with(app.params, "bands.count", value)
        app._process_edit(draft, any_changed=True, any_committed=False)
    assert app.sim.calls == [], "heavy edit must not rebuild every frame"
    assert app.params is base, "committed state untouched until release"
    assert app._live.bands.count == 17, "working copy tracks the in-progress value"
    assert app._gesture_base is base, "gesture base = the pre-edit committed state"

    # release frame
    draft = _draft_with(app.params, "bands.count", 17)
    app._process_edit(draft, any_changed=True, any_committed=True)
    assert len(app.sim.calls) == 1, "exactly one rebuild on release"
    assert app.params.bands.count == 17
    assert app.params == app._live
    assert app._recomputing is True, "a restart commit flags the recompute hint"
    assert app._gesture_base is None, "gesture base cleared once released"


def test_heavy_combo_reselect_same_value_does_not_commit() -> None:
    # any_committed fires but the diff is empty (re-selecting the current value).
    app = _make_app()
    draft = app.params.model_dump()  # unchanged
    app._process_edit(draft, any_changed=False, any_committed=True)
    assert app.sim.calls == [], "empty diff on release must not rebuild"
    assert app._gesture_base is None


def test_heavy_discrete_edit_commits_on_same_frame() -> None:
    # a combo/checkbox pick reports changed+committed in one frame.
    app = _make_app()
    draft = _draft_with(app.params, "bands.count", 20)
    app._process_edit(draft, any_changed=True, any_committed=True)
    assert len(app.sim.calls) == 1
    assert app.params.bands.count == 20
    assert app._gesture_base is None


# -- invalid mid-drag states are silent and keep the last valid _live -----------


def test_invalid_mid_drag_keeps_last_valid_live_no_crash() -> None:
    app = _make_app()
    # sor_omega must be strictly in (1, 2); 2.0 trips the cross-field validator.
    good = _draft_with(app.params, "solver.sor_omega", 1.5)
    app._process_edit(good, any_changed=True, any_committed=False)
    last_valid = app._live
    assert app._live.solver.sor_omega == pytest.approx(1.5)

    bad = _draft_with(app.params, "solver.sor_omega", 2.0)
    app._process_edit(bad, any_changed=True, any_committed=False)
    assert app._live is last_valid, "invalid value must not replace the working copy"


# -- gesture base lifecycle -----------------------------------------------------


def test_gesture_base_captured_once_per_gesture() -> None:
    app = _make_app()
    base = app.params
    d1 = _draft_with(app.params, "bands.count", 15)
    app._process_edit(d1, any_changed=True, any_committed=False)
    assert app._gesture_base is base
    # a later frame of the SAME gesture must not re-capture
    d2 = _draft_with(app.params, "bands.count", 16)
    app._process_edit(d2, any_changed=True, any_committed=False)
    assert app._gesture_base is base


# -- discrete-action reset wins over a pending heavy edit -----------------------


def test_reset_working_copy_drops_pending_edit() -> None:
    app = _make_app()
    # leave a heavy edit pending (not released)
    draft = _draft_with(app.params, "bands.count", 30)
    app._process_edit(draft, any_changed=True, any_committed=False)
    assert app._live.bands.count == 30
    assert app._gesture_base is not None

    app._reset_working_copy()
    assert app._live is app.params, "working copy snaps back to committed state"
    assert app._gesture_base is None
    # the abandoned edit must not resurrect on the next idle frame
    idle = app.params.model_dump()
    app._process_edit(idle, any_changed=False, any_committed=False)
    assert app.sim.calls == [], "no stale commit after a discrete reset"


def test_load_dialog_resets_working_copy(tmp_path: Path) -> None:
    """The Load path runs in _poll_dialog (pre_frame) — the branch most likely
    to miss the reset. A pending heavy edit must not survive a file load."""
    other = PlanetParams()
    other.bands.count = 7
    preset_file = tmp_path / "loaded.json"
    save_preset(other, preset_file)

    app = _make_app()
    app.toasts = main.Toasts()
    # a heavy edit pending before the load
    draft = _draft_with(app.params, "bands.count", 30)
    app._process_edit(draft, any_changed=True, any_committed=False)
    assert app._live.bands.count == 30
    assert app._gesture_base is not None

    class FakeDialog:
        def ready(self) -> bool:
            return True

        def result(self) -> list[str]:
            return [str(preset_file)]

    app._dialog = ("load", FakeDialog())
    app._poll_dialog()

    assert app.params.bands.count == 7, "the load won"
    assert app._live is app.params, "_live reset to the loaded params"
    assert app._gesture_base is None, "gesture base cleared by the load"


# -- export resolution combo writes export.width --------------------------------


# -- Phase 4 escape hatches commit through the same pipeline --------------------


def test_bands_template_clear_commits_through_pipeline() -> None:
    """The panel's 'Clear template' button (panels._draw_bands_template_escape)
    mutates the draft and reports (changed=True, committed=True), same as a
    composite-editor add/remove-row click; this exercises the resulting
    _process_edit call the way main.py's draw_controls would drive it."""
    from gasgiant.params.model import BandTemplate

    base = PlanetParams()
    base.bands.template = BandTemplate(
        edges_deg=[90.0, 0.0, -90.0], values=[0.2, 0.8], heights=[0.5, 0.5]
    )
    app = _make_app(base)

    draft = _draft_with(app.params, "bands.template", None)
    app._process_edit(draft, any_changed=True, any_committed=True)

    assert app.params.bands.template is None, "the escape hatch's clear must commit"
    assert len(app.sim.calls) == 1
    assert app._gesture_base is None, "a discrete click, not a lingering gesture"


def test_hero_latitude_unpin_commits_through_pipeline() -> None:
    """The panel's 'Unpin latitude' button (panels._draw_hero_latitude_escape)
    resets storms.hero_latitude to None; must satisfy _validate_hero_latitude
    (which only checks a non-None value) and commit through the same pipeline
    as any other discrete panel action."""
    base = PlanetParams()
    base.storms.hero_latitude = 12.0
    app = _make_app(base)

    draft = _draft_with(app.params, "storms.hero_latitude", None)
    app._process_edit(draft, any_changed=True, any_committed=True)

    assert app.params.storms.hero_latitude is None, "the escape hatch's unpin must commit"
    assert len(app.sim.calls) == 1
    assert app._gesture_base is None


def test_export_resolutions_within_bounds() -> None:
    info = PlanetParams.model_fields["export"].annotation.model_fields["width"]
    lo = next(m.ge for m in info.metadata if hasattr(m, "ge"))
    hi = next(m.le for m in info.metadata if hasattr(m, "le"))
    for width, _label in main.EXPORT_RESOLUTIONS:
        assert lo <= width <= hi, f"{width} outside export.width bounds [{lo}, {hi}]"

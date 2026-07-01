"""Phase 3 panel-navigation tests: PanelState, the shared search predicate,
the reset-to-default value computation, the search-forced-open id-stack
scoping, and the locked-set threading into ``StudioApp._randomize``.

The pure-logic pieces (``_leaf_visible``, ``_subtree_has_match``,
``_default_value``) need no GUI. The header push_id/SetNextItemOpen scoping
(guard test: search type-then-clear preserves header open/closed state) is
exercised against a real, headless imgui context (``imgui.create_context()``
+ ``new_frame``/``end_frame``, no window/renderer backend needed) so the
behavior is verified mechanically rather than asserted by inspection alone.
"""

from __future__ import annotations

from collections import deque

import pytest

from gasgiant.params.model import PlanetParams
from gasgiant.params.randomize import randomize

panels = pytest.importorskip("gasgiant.app.panels")
main = pytest.importorskip("gasgiant.app.main")
imgui = pytest.importorskip("imgui_bundle.imgui")

PanelState = panels.PanelState
StudioApp = main.StudioApp


# -- a real (headless) imgui context, for the id-stack/storage tests ------------


@pytest.fixture
def imgui_ctx():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)  # don't litter the repo root with an imgui.ini
    # Dear ImGui (the renderer-has-textures backend flag) lets NewFrame run
    # without a real font-atlas upload -- we only need widget *logic*, never
    # actually render or rasterize anything.
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def _frame(label: str, *, searching: bool, force_open: bool | None = None):
    """One full imgui frame: optionally prime SetNextItemOpen under the
    NORMAL (non-search) id scope, then draw the section header exactly the
    way _draw_model does via panels._section_header. Returns the header's
    open/closed return value for that frame."""
    imgui.new_frame()
    imgui.begin("panel_state_test", None, 0)
    if force_open is not None:
        imgui.set_next_item_open(force_open, imgui.Cond_.always)
    opened = panels._section_header(label, 0, searching)
    imgui.end()
    imgui.end_frame()
    return opened


# -- guard test #4: search type-then-clear preserves header open/closed state ---


def test_search_forced_open_does_not_clobber_manual_header_state(imgui_ctx):
    # User manually closes the section (simulated by priming SetNextItemOpen
    # under the normal, non-search id scope).
    assert _frame("Bands", searching=False, force_open=False) is False
    assert _frame("Bands", searching=False) is False, "stays closed with no search active"

    # A search starts: several frames force it open under push_id("search").
    for _ in range(3):
        assert _frame("Bands", searching=True, force_open=False) is True

    # Clearing the search reverts to the normal id scope with no forcing --
    # must read back the user's manually-closed state, not the search-forced
    # open state.
    assert _frame("Bands", searching=False) is False, (
        "clearing the search must restore the pre-search header state"
    )


def test_search_forced_open_when_manually_open_stays_open_after_clear(imgui_ctx):
    assert _frame("Jets", searching=False, force_open=True) is True
    for _ in range(2):
        assert _frame("Jets", searching=True) is True
    assert _frame("Jets", searching=False) is True


# -- _leaf_visible: name / label / description, case-insensitive ---------------


def _gamma_info():
    return PlanetParams.model_fields["appearance"].annotation.model_fields["gamma"]


def test_leaf_visible_no_query_matches_everything():
    state = PanelState()
    assert panels._leaf_visible("gamma", _gamma_info(), {}, state) is True


def test_leaf_visible_matches_field_name_case_insensitive():
    state = PanelState(search="GAM")
    assert panels._leaf_visible("gamma", _gamma_info(), {}, state) is True


def test_leaf_visible_matches_label_with_underscore_replaced():
    info = PlanetParams.model_fields["bands"].annotation.model_fields["width_jitter"]
    state = PanelState(search="width jitter")
    assert panels._leaf_visible("width_jitter", info, {}, state) is True


def test_leaf_visible_matches_description():
    info = PlanetParams.model_fields["seed"]
    assert info.description, "seed must carry a description for this test to be meaningful"
    state = PanelState(search="deterministic")
    assert panels._leaf_visible("seed", info, {}, state) is True


def test_leaf_visible_no_match_returns_false():
    state = PanelState(search="zzz_no_such_thing")
    assert panels._leaf_visible("gamma", _gamma_info(), {}, state) is False


# -- _subtree_has_match: shares the SAME predicate as the leaf gate (guard #8) --


def test_subtree_has_match_delegates_to_leaf_visible(monkeypatch):
    """Spies on the real panels._leaf_visible to confirm _subtree_has_match
    calls it (rather than re-implementing the match logic) -- the two call
    sites must share one predicate, not drift via separate copies."""
    calls: list[str] = []
    original = panels._leaf_visible

    def spy(name, info, doc, state):
        calls.append(name)
        return original(name, info, doc, state)

    monkeypatch.setattr(panels, "_leaf_visible", spy)

    params = PlanetParams()
    doc = params.model_dump()
    state = PanelState(search="gamma")
    appearance_model = type(params).model_fields["appearance"].annotation
    assert panels._subtree_has_match(appearance_model, doc["appearance"], state) is True
    assert "gamma" in calls


def test_subtree_has_match_false_when_nothing_matches():
    params = PlanetParams()
    doc = params.model_dump()
    state = PanelState(search="zzz_no_such_thing")
    physical_model = type(params).model_fields["physical"].annotation
    assert panels._subtree_has_match(physical_model, doc["physical"], state) is False


# -- _default_value: scalar AND composite (palette/stops), deep-copy isolated ---


def test_default_value_scalar():
    baseline = panels._defaults_baseline()
    assert panels._default_value("gamma", baseline["appearance"]) == baseline["appearance"]["gamma"]


def test_default_value_composite_is_an_independent_deep_copy():
    """Mutating the value handed back by a reset must NOT corrupt the cached
    baseline -- otherwise every subsequent 'Reset to default' for a
    palette/stops field would read back a poisoned baseline."""
    baseline = panels._defaults_baseline()
    pre_len = len(baseline["appearance"]["palette_rows"][0]["stops"])

    reset_value = panels._default_value("palette_rows", baseline["appearance"])
    reset_value[0]["stops"].append(dict(reset_value[0]["stops"][-1]))

    assert len(baseline["appearance"]["palette_rows"][0]["stops"]) == pre_len, (
        "the cached baseline must be untouched by mutating a returned reset value"
    )
    assert len(reset_value[0]["stops"]) == pre_len + 1


def test_default_value_storm_tints_composite():
    baseline = panels._defaults_baseline()
    reset_value = panels._default_value("storm_tints", baseline["appearance"])
    assert reset_value == baseline["appearance"]["storm_tints"]
    assert reset_value is not baseline["appearance"]["storm_tints"]


# -- leaf_kind stays unchanged (signature/behavior) ------------------------------


def test_leaf_kind_signature_unchanged():
    """Phase 3 must not touch leaf_kind's signature/behavior -- the static
    coverage test (test_panels_coverage.py) calls it directly without imgui."""
    info = _gamma_info()
    assert panels.leaf_kind("gamma", info, 1.0) == "float"


# -- StudioApp._randomize: threads panel_state.locked into randomize() ----------


def _make_app(params: PlanetParams | None = None) -> StudioApp:
    params = params or PlanetParams()
    app = StudioApp.__new__(StudioApp)
    app.params = params
    app._live = params
    app._gesture_base = None
    app._recomputing = False
    app._undo_stack = deque(maxlen=64)
    app._redo_stack = deque(maxlen=64)
    app.panel_state = PanelState()
    return app


def test_randomize_honors_locked_dotted_path():
    app = _make_app()
    app.params.bands.count = 33
    app._live = app.params
    app.panel_state.locked = {"bands.count"}

    result = app._randomize(5)
    unlocked = randomize(5, base=app.params)

    assert result.bands.count == 33, "locked field must not reroll"
    assert result.appearance.haze_amount == unlocked.appearance.haze_amount, (
        "the lock consumes its draw so unlocked fields are unaffected"
    )


def test_randomize_locked_seed_keeps_master_seed_pinned():
    app = _make_app()
    pre_seed = app.params.seed
    app.panel_state.locked = {"seed"}

    result = app._randomize(123456)

    assert result.seed == pre_seed, "locking 'seed' must keep the stored master seed pinned"
    # but the rest of the params still reroll from the fresh seed's RNG stream
    unlocked = randomize(123456, base=app.params)
    unlocked.seed = pre_seed
    assert result == unlocked


def test_randomize_without_locks_matches_bare_randomize():
    app = _make_app()
    result = app._randomize(77)
    assert result == randomize(77, base=app.params)

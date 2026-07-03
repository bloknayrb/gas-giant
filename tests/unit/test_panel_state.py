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


# -- Phase 4: Basic/Advanced gating -----------------------------------------------


def _hero_solid_core_info():
    """An adv=True leaf (a byte-identical-off-by-default hero cosmetic lever)."""
    return PlanetParams.model_fields["storms"].annotation.model_fields["hero_solid_core"]


def test_advanced_visible_true_for_non_adv_field_regardless_of_toggle():
    info = _gamma_info()  # adv=False (a Basic headline knob)
    assert panels._advanced_visible(info, PanelState(show_advanced=False)) is True
    assert panels._advanced_visible(info, PanelState(show_advanced=True)) is True


def test_advanced_visible_gated_by_toggle_for_adv_field():
    info = _hero_solid_core_info()
    assert panels._advanced_visible(info, PanelState(show_advanced=False)) is False
    assert panels._advanced_visible(info, PanelState(show_advanced=True)) is True


def test_leaf_visible_hides_adv_field_when_basic_and_not_searching():
    info = _hero_solid_core_info()
    state = PanelState(search="", show_advanced=False)
    assert panels._leaf_visible("hero_solid_core", info, {}, state) is False


def test_leaf_visible_shows_adv_field_when_advanced_on():
    info = _hero_solid_core_info()
    state = PanelState(search="", show_advanced=True)
    assert panels._leaf_visible("hero_solid_core", info, {}, state) is True


def test_leaf_visible_search_overrides_advanced_gate():
    """A search match on an adv=True field's name must find it even with
    Advanced off -- the gate is bypassed entirely while searching."""
    info = _hero_solid_core_info()
    state = PanelState(search="solid core", show_advanced=False)
    assert panels._leaf_visible("hero_solid_core", info, {}, state) is True


def test_leaf_visible_search_still_respects_no_match_for_adv_field():
    info = _hero_solid_core_info()
    state = PanelState(search="zzz_no_such_thing", show_advanced=False)
    assert panels._leaf_visible("hero_solid_core", info, {}, state) is False


def test_default_panel_state_is_basic_mode():
    """Phase 4 flips the Phase-3-plumbed-but-inert default: the app now
    actually gates on show_advanced, so it should land newcomers in Basic."""
    assert PanelState().show_advanced is False


# -- Phase 4: hidden-advanced-settings-differ hint count -------------------------


def test_count_differs_from_default_zero_at_defaults():
    baseline = panels._defaults_baseline()
    params = PlanetParams()
    doc = params.model_dump()
    solver_model = type(params).model_fields["solver"].annotation
    assert panels._count_differs_from_default(solver_model, doc["solver"], baseline["solver"]) == 0


def test_count_differs_from_default_counts_changed_leaves():
    baseline = panels._defaults_baseline()
    params = PlanetParams()
    params.solver.vort_inject = 1.5  # 1 leaf differs
    params.solver.baroclinic.enabled = False  # baroclinic.enabled is already False by
    # default -- flip a DIFFERENT nested leaf instead so the count is deterministic.
    params.solver.baroclinic.gain = 5.0  # nested leaf differs too
    doc = params.model_dump()
    solver_model = type(params).model_fields["solver"].annotation
    n = panels._count_differs_from_default(solver_model, doc["solver"], baseline["solver"])
    assert n == 2, "expected exactly the two changed leaves (one nested)"


def test_hidden_advanced_hint_only_fires_when_section_fully_hidden(imgui_ctx, monkeypatch):
    """Solver is fully adv=True: in Basic mode with a non-default solver
    field, the hint text must be drawn (H4 + MED-3)."""
    params = PlanetParams()
    params.solver.vort_inject = 2.0
    doc = params.model_dump()
    baseline = panels._defaults_baseline()
    state = PanelState(show_advanced=False)

    seen: list[str] = []
    monkeypatch.setattr(
        imgui, "text_colored", lambda color, text: seen.append(text)
    )

    imgui.new_frame()
    imgui.begin("hint_test", None, 0)
    solver_model = type(params).model_fields["solver"].annotation
    has_match = panels._subtree_has_match(solver_model, doc["solver"], state)
    assert has_match is False, "Basic mode + fully-advanced section = zero visible leaves"
    panels._draw_hidden_advanced_hint(solver_model, doc["solver"], baseline["solver"])
    imgui.end()
    imgui.end_frame()

    assert any("1 advanced setting" in s for s in seen)


def test_hidden_advanced_hint_silent_when_at_defaults(imgui_ctx, monkeypatch):
    params = PlanetParams()
    doc = params.model_dump()
    baseline = panels._defaults_baseline()

    calls = []
    monkeypatch.setattr(imgui, "text_colored", lambda color, text: calls.append(text))

    imgui.new_frame()
    imgui.begin("hint_test2", None, 0)
    solver_model = type(params).model_fields["solver"].annotation
    panels._draw_hidden_advanced_hint(solver_model, doc["solver"], baseline["solver"])
    imgui.end()
    imgui.end_frame()

    assert calls == [], "no hint text when the section is untouched from defaults"


# -- Phase 4: bands.template / hero_latitude Basic-visible escape hatches --------


def test_bands_template_escape_noop_when_unset(imgui_ctx):
    params = PlanetParams()
    doc = params.model_dump()
    imgui.new_frame()
    imgui.begin("escape_test", None, 0)
    changed, committed = panels._draw_bands_template_escape(doc["bands"])
    imgui.end()
    imgui.end_frame()
    assert (changed, committed) == (False, False)


def test_bands_template_escape_draws_banner_when_set(imgui_ctx, monkeypatch):
    from gasgiant.params.model import BandTemplate

    params = PlanetParams()
    params.bands.template = BandTemplate(
        edges_deg=[90.0, 0.0, -90.0], values=[0.2, 0.8], heights=[0.5, 0.5]
    )
    doc = params.model_dump()

    seen = []
    monkeypatch.setattr(imgui, "text_colored", lambda color, text: seen.append(text))

    imgui.new_frame()
    imgui.begin("escape_test2", None, 0)
    changed, committed = panels._draw_bands_template_escape(doc["bands"])
    imgui.end()
    imgui.end_frame()

    assert (changed, committed) == (False, False), "drawing the banner alone is not a click"
    assert any("template is set" in s for s in seen)


def test_hero_latitude_escape_noop_when_unpinned(imgui_ctx):
    params = PlanetParams()
    doc = params.model_dump()
    imgui.new_frame()
    imgui.begin("escape_test3", None, 0)
    changed, committed = panels._draw_hero_latitude_escape(doc["storms"])
    imgui.end()
    imgui.end_frame()
    assert (changed, committed) == (False, False)


def test_hero_latitude_escape_draws_when_pinned(imgui_ctx, monkeypatch):
    params = PlanetParams()
    params.storms.hero_latitude = 12.0
    doc = params.model_dump()

    seen = []
    monkeypatch.setattr(imgui, "text_colored", lambda color, text: seen.append(text))

    imgui.new_frame()
    imgui.begin("escape_test4", None, 0)
    changed, committed = panels._draw_hero_latitude_escape(doc["storms"])
    imgui.end()
    imgui.end_frame()

    assert (changed, committed) == (False, False)
    assert any("pinned to 12.0" in s for s in seen)


# -- Phase 4: Storms sub-grouping (ui labels + declaration order) ----------------


def test_storms_fields_grouped_into_named_subsections():
    from gasgiant.params.model import StormsParams

    expected_groups = {
        "Hero", "Ovals", "Barges", "Pearls", "Outbreaks", "Small storms", "Mergers",
    }
    seen_groups = {
        info.json_schema_extra.get("ui")
        for info in StormsParams.model_fields.values()
        if isinstance(info.json_schema_extra, dict)
    }
    assert seen_groups == expected_groups


def test_storms_subgroup_fields_are_contiguous_by_declaration_order():
    """_draw_model's separator_text logic assumes same-ui fields are
    consecutive in declaration order -- verify StormsParams actually
    satisfies that (a group label appearing, disappearing, then
    reappearing would draw two separators for one logical group)."""
    from gasgiant.params.model import StormsParams

    seen_once: list[str] = []
    for info in StormsParams.model_fields.values():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        ui = extra.get("ui")
        if not seen_once or seen_once[-1] != ui:
            assert ui not in seen_once, f"ui label {ui!r} is not contiguous"
            seen_once.append(ui)


def test_other_sections_have_constant_ui_no_separators_expected():
    """Backward-safety: every section besides Storms must still have a
    single constant ui value across all its (non-empty-ui) leaves, so
    _draw_model's separator_text logic never fires for them."""
    from pydantic import BaseModel as _BaseModel

    from gasgiant.params.model import PlanetParams as _PlanetParams

    def walk(model, prefix=""):
        for name, info in model.model_fields.items():
            ann = info.annotation
            path = f"{prefix}{name}"
            if isinstance(ann, type) and issubclass(ann, _BaseModel):
                yield from walk(ann, f"{path}.")
                continue
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
            yield path, extra.get("ui")

    from collections import defaultdict

    by_section: dict[str, set[str]] = defaultdict(set)
    for path, ui in walk(_PlanetParams):
        if not ui:
            continue
        top = path.split(".", 1)[0]
        by_section[top].add(ui)

    for section, labels in by_section.items():
        if section == "storms":
            continue
        assert len(labels) == 1, f"section {section!r} has multiple ui labels: {labels}"


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


# -- Minor: the modified (*) marker must not false-positive on colors -----------
#
# model_dump() emits colors as lists; color_edit3 writes back tuple(rgb). A plain
# `current != default` then reads a colour dragged back to its default as still
# modified (tuple != list), so the `*` sticks and the "N advanced differ" count
# is inflated. _leaf_changed normalizes list-vs-tuple.


def test_leaf_changed_treats_equal_color_list_and_tuple_as_unchanged():
    assert panels._leaf_changed((0.1, 0.2, 0.3), [0.1, 0.2, 0.3]) is False


def test_leaf_changed_detects_real_color_difference():
    assert panels._leaf_changed((0.1, 0.2, 0.9), [0.1, 0.2, 0.3]) is True


def test_leaf_changed_scalars_unaffected():
    assert panels._leaf_changed(0.5, 0.5) is False
    assert panels._leaf_changed(0.5, 0.6) is True


# -- B4-2: Clear-template confirm + hero_latitude pin widget ---------------------


def test_clear_template_button_stages_confirm_not_clear(imgui_ctx, monkeypatch):
    """Clicking "Clear template" must NOT clear the template outright -- it
    opens a confirm modal (the cleared skeleton is recoverable only via Undo,
    and the startup preset ships with a template engaged)."""
    from gasgiant.params.model import BandTemplate

    params = PlanetParams()
    params.bands.template = BandTemplate(
        edges_deg=[90.0, 0.0, -90.0], values=[0.2, 0.8], heights=[0.5, 0.5]
    )
    doc = params.model_dump()

    opened = []
    monkeypatch.setattr(panels.imgui, "small_button", lambda label: True)
    monkeypatch.setattr(panels.imgui, "open_popup", lambda title: opened.append(title))

    imgui.new_frame()
    imgui.begin("confirm_test", None, 0)
    changed, committed = panels._draw_bands_template_escape(doc["bands"])
    imgui.end()
    imgui.end_frame()

    assert (changed, committed) == (False, False), "the click alone must not clear"
    assert doc["bands"]["template"] is not None, "template survives until the confirm"
    assert opened == ["Clear band template?"], "the confirm modal was staged"


def test_clear_template_confirm_clears_and_commits(imgui_ctx, monkeypatch):
    """Confirming inside the modal clears the template and reports a committed
    change (one undo entry via the normal panel pipeline)."""
    from gasgiant.params.model import BandTemplate

    params = PlanetParams()
    params.bands.template = BandTemplate(
        edges_deg=[90.0, 0.0, -90.0], values=[0.2, 0.8], heights=[0.5, 0.5]
    )
    doc = params.model_dump()

    # Simulate: modal is open, its "Clear##template" button is pressed.
    monkeypatch.setattr(panels.imgui, "small_button", lambda label: False)
    monkeypatch.setattr(
        panels.imgui, "begin_popup_modal", lambda title, p, flags: (True, True)
    )
    monkeypatch.setattr(
        panels.imgui, "button", lambda label: label == "Clear##template"
    )
    monkeypatch.setattr(panels.imgui, "close_current_popup", lambda: None)
    monkeypatch.setattr(panels.imgui, "end_popup", lambda: None)

    imgui.new_frame()
    imgui.begin("confirm_test2", None, 0)
    changed, committed = panels._draw_bands_template_escape(doc["bands"])
    imgui.end()
    imgui.end_frame()

    assert (changed, committed) == (True, True)
    assert doc["bands"]["template"] is None
    assert "Undo" in panels._CLEAR_TEMPLATE_CONFIRM, "the copy names the way back"


def test_clear_template_cancel_keeps_template(imgui_ctx, monkeypatch):
    from gasgiant.params.model import BandTemplate

    params = PlanetParams()
    params.bands.template = BandTemplate(
        edges_deg=[90.0, 0.0, -90.0], values=[0.2, 0.8], heights=[0.5, 0.5]
    )
    doc = params.model_dump()

    monkeypatch.setattr(panels.imgui, "small_button", lambda label: False)
    monkeypatch.setattr(
        panels.imgui, "begin_popup_modal", lambda title, p, flags: (True, True)
    )
    monkeypatch.setattr(
        panels.imgui, "button", lambda label: label == "Cancel##template"
    )
    monkeypatch.setattr(panels.imgui, "close_current_popup", lambda: None)
    monkeypatch.setattr(panels.imgui, "end_popup", lambda: None)

    imgui.new_frame()
    imgui.begin("confirm_test3", None, 0)
    changed, committed = panels._draw_bands_template_escape(doc["bands"])
    imgui.end()
    imgui.end_frame()

    assert (changed, committed) == (False, False)
    assert doc["bands"]["template"] is not None

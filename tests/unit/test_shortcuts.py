"""Phase 7 help window + keyboard shortcuts.

The app-level logic (Randomize extraction, the save-dialog opener, the
shortcut dispatch table) is driven directly against ``StudioApp`` methods --
the same headless pattern ``test_undo_redo.py``/``test_preset_identity.py``
use, no GL context or real window needed.

The ``want_text_input`` guard and the ``/`` search-focus request ARE
mechanically verified here against a real (headless) imgui context
(``imgui.create_context()`` + ``new_frame``/``end_frame``, the same pattern
``test_panel_state.py`` uses for the header id-stack tests) rather than just
asserted by inspection: ``io.want_text_input`` only reflects a widget's
active state one frame after it gains focus (verified empirically while
building this test), so the fixtures below drive a few warm-up frames before
asserting.
"""

from __future__ import annotations

from collections import deque

import pytest

from gasgiant.params.model import PlanetParams

main = pytest.importorskip("gasgiant.app.main")
panels = pytest.importorskip("gasgiant.app.panels")
imgui = pytest.importorskip("imgui_bundle.imgui")

StudioApp = main.StudioApp
PanelState = panels.PanelState


# -- a real (headless) imgui context, for the want_text_input/focus tests -------


@pytest.fixture
def imgui_ctx():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)  # don't litter the repo root with an imgui.ini
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def _frame(after_new_frame=None, draw_fn=None) -> None:
    """One full imgui frame. ``after_new_frame`` runs right after
    ``new_frame()`` -- the same point in the frame ``StudioApp._handle_shortcuts``
    is wired to via hello_imgui's ``post_new_frame`` callback (before any
    window is drawn this frame)."""
    imgui.new_frame()
    if after_new_frame is not None:
        after_new_frame()
    imgui.begin("w", None, 0)
    if draw_fn is not None:
        draw_fn()
    imgui.end()
    imgui.end_frame()


def _draw_text_input(click: bool = False) -> None:
    """Draw a text input, optionally simulating a mouse-down over it on
    THIS frame only -- re-firing the mouse-down event every frame (rather
    than once, then just redrawing) was tried first and did not reliably
    propagate to ``io.want_text_input`` in this headless setup; a single
    click followed by plain redraws matches real click-to-focus behavior
    and was verified to work empirically while building this test."""
    imgui.input_text("##t", "hello")
    if click:
        io = imgui.get_io()
        rmin, rmax = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        cx, cy = (rmin.x + rmax.x) / 2, (rmin.y + rmax.y) / 2
        io.add_mouse_pos_event(cx, cy)
        io.add_mouse_button_event(0, True)


def _focus_text_input() -> None:
    """Draw the text input once (warm-up -- clicking on the very first-ever
    frame a widget exists doesn't register a hit, since imgui's mouse hover
    test needs the item's previous-frame layout rect), click it on the next
    frame, then drive a few more redraw-only frames for the click to
    propagate into an active item and then into ``io.want_text_input``
    (empirically ~3 frames after the click in this headless setup -- see
    module docstring)."""
    _frame(draw_fn=_draw_text_input)
    _frame(draw_fn=lambda: _draw_text_input(click=True))
    for _ in range(3):
        _frame(draw_fn=_draw_text_input)
    assert imgui.get_io().want_text_input, "setup failed to focus the text input"


def _press_ctrl_key(io, key, down: bool) -> None:
    """Queue a Ctrl+<key> chord. A headless context needs BOTH the physical
    modifier key (left_ctrl) AND the synthetic mod_ctrl key event for
    ``io.key_ctrl`` to update -- a real platform backend (GLFW/SDL, as used
    by the real app via hello_imgui) submits both automatically; verified
    empirically that omitting either leaves ``io.key_ctrl`` False."""
    io.add_key_event(imgui.Key.mod_ctrl, down)
    io.add_key_event(imgui.Key.left_ctrl, down)
    io.add_key_event(key, down)


# -- _shortcuts_enabled(): the want_text_input guard, mechanically verified -----


def test_shortcuts_enabled_true_with_no_focused_widget(imgui_ctx):
    _frame()
    assert main._shortcuts_enabled() is True


def test_shortcuts_enabled_false_while_text_input_focused(imgui_ctx):
    _focus_text_input()
    assert main._shortcuts_enabled() is False


# -- _handle_shortcuts: the guard actually suppresses a real shortcut -----------


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
    app._dialog = None
    app._export = None
    app._show_help = False
    app.panel_state = PanelState()
    app.toasts = main.Toasts()
    return app


def test_r_shortcut_suppressed_while_typing(imgui_ctx):
    """The scenario the brief calls out explicitly: typing "r" while the
    search box is focused (e.g. filtering for "rim_contrast") must NOT
    trigger Randomize."""
    app = _make_app()
    calls: list[bool] = []
    app._do_randomize = lambda: calls.append(True)  # type: ignore[method-assign]

    _focus_text_input()
    io = imgui.get_io()
    io.add_key_event(imgui.Key.r, True)
    _frame(after_new_frame=app._handle_shortcuts, draw_fn=_draw_text_input)
    io.add_key_event(imgui.Key.r, False)

    assert calls == [], "'r' while a text input is focused must not fire Randomize"


def test_r_shortcut_fires_when_not_typing(imgui_ctx):
    app = _make_app()
    calls: list[bool] = []
    app._do_randomize = lambda: calls.append(True)  # type: ignore[method-assign]

    _frame()  # no text input drawn/focused
    io = imgui.get_io()
    io.add_key_event(imgui.Key.r, True)
    _frame(after_new_frame=app._handle_shortcuts)
    io.add_key_event(imgui.Key.r, False)

    assert calls == [True]


def test_r_shortcut_suppressed_during_export(imgui_ctx):
    """Randomize's button is disabled outright while an export is in
    flight (M5 / Round 2 LOW-5); the R shortcut must honor the same gate."""
    app = _make_app()
    app._export = (object(), object())
    calls: list[bool] = []
    app._do_randomize = lambda: calls.append(True)  # type: ignore[method-assign]

    _frame()
    io = imgui.get_io()
    io.add_key_event(imgui.Key.r, True)
    _frame(after_new_frame=app._handle_shortcuts)
    io.add_key_event(imgui.Key.r, False)

    assert calls == [], "R must not fire while an export is in flight"


def test_a_shortcut_toggles_advanced(imgui_ctx):
    app = _make_app()
    assert app.panel_state.show_advanced is False

    _frame()
    io = imgui.get_io()
    io.add_key_event(imgui.Key.a, True)
    _frame(after_new_frame=app._handle_shortcuts)
    io.add_key_event(imgui.Key.a, False)

    assert app.panel_state.show_advanced is True


def test_f1_shortcut_toggles_help(imgui_ctx):
    app = _make_app()
    assert app._show_help is False

    _frame()
    io = imgui.get_io()
    io.add_key_event(imgui.Key.f1, True)
    _frame(after_new_frame=app._handle_shortcuts)
    io.add_key_event(imgui.Key.f1, False)

    assert app._show_help is True


def test_slash_shortcut_requests_search_focus(imgui_ctx):
    app = _make_app()
    assert app.panel_state.focus_search_requested is False

    _frame()
    io = imgui.get_io()
    io.add_key_event(imgui.Key.slash, True)
    _frame(after_new_frame=app._handle_shortcuts)
    io.add_key_event(imgui.Key.slash, False)

    assert app.panel_state.focus_search_requested is True


def test_ctrl_z_shortcut_calls_undo(imgui_ctx):
    app = _make_app()
    calls: list[bool] = []
    app._undo = lambda: calls.append(True)  # type: ignore[method-assign]

    _frame()
    io = imgui.get_io()
    _press_ctrl_key(io, imgui.Key.z, True)
    _frame(after_new_frame=app._handle_shortcuts)
    _press_ctrl_key(io, imgui.Key.z, False)

    assert calls == [True]


def test_ctrl_y_shortcut_calls_redo(imgui_ctx):
    app = _make_app()
    calls: list[bool] = []
    app._redo = lambda: calls.append(True)  # type: ignore[method-assign]

    _frame()
    io = imgui.get_io()
    _press_ctrl_key(io, imgui.Key.y, True)
    _frame(after_new_frame=app._handle_shortcuts)
    _press_ctrl_key(io, imgui.Key.y, False)

    assert calls == [True]


def test_ctrl_s_shortcut_opens_save_dialog(imgui_ctx):
    app = _make_app()
    calls: list[bool] = []
    app._open_save_dialog = lambda: calls.append(True)  # type: ignore[method-assign]

    _frame()
    io = imgui.get_io()
    _press_ctrl_key(io, imgui.Key.s, True)
    _frame(after_new_frame=app._handle_shortcuts)
    _press_ctrl_key(io, imgui.Key.s, False)

    assert calls == [True]


def test_undo_redo_save_suppressed_during_export(imgui_ctx):
    app = _make_app()
    app._export = (object(), object())
    undo_calls: list[bool] = []
    save_calls: list[bool] = []
    app._undo = lambda: undo_calls.append(True)  # type: ignore[method-assign]
    app._open_save_dialog = lambda: save_calls.append(True)  # type: ignore[method-assign]

    _frame()
    io = imgui.get_io()
    io.add_key_event(imgui.Key.mod_ctrl, True)
    io.add_key_event(imgui.Key.left_ctrl, True)
    io.add_key_event(imgui.Key.z, True)
    io.add_key_event(imgui.Key.s, True)
    _frame(after_new_frame=app._handle_shortcuts)
    io.add_key_event(imgui.Key.z, False)
    io.add_key_event(imgui.Key.s, False)
    io.add_key_event(imgui.Key.left_ctrl, False)
    io.add_key_event(imgui.Key.mod_ctrl, False)

    assert undo_calls == []
    assert save_calls == []


# -- / focus request is actually consumed by _draw_search_box -------------------


def test_draw_search_box_consumes_focus_request(imgui_ctx, monkeypatch):
    calls: list[bool] = []
    original = imgui.set_keyboard_focus_here

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(imgui, "set_keyboard_focus_here", spy)

    state = PanelState(focus_search_requested=True)
    imgui.new_frame()
    imgui.begin("search_test", None, 0)
    panels._draw_search_box(state)
    imgui.end()
    imgui.end_frame()

    assert calls == [True], "the / shortcut's flag must call set_keyboard_focus_here"
    assert state.focus_search_requested is False, "the flag is consumed, not re-armed every frame"


def test_draw_search_box_no_focus_call_when_not_requested(imgui_ctx, monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(imgui, "set_keyboard_focus_here", lambda *a, **k: calls.append(True))

    state = PanelState(focus_search_requested=False)
    imgui.new_frame()
    imgui.begin("search_test2", None, 0)
    panels._draw_search_box(state)
    imgui.end()
    imgui.end_frame()

    assert calls == []


def test_slash_focus_request_actually_focuses_the_search_box(imgui_ctx):
    """End-to-end (within the headless imgui context): after the flag is
    set, the search box genuinely becomes the focused text-input widget --
    not just "a focus call was made" (the two tests above), but imgui's own
    ``want_text_input`` signal confirms it a few frames later (empirically
    3 frames in this headless setup, matching ``_focus_text_input`` above)."""
    state = PanelState(focus_search_requested=True)
    for _ in range(3):
        imgui.new_frame()
        imgui.begin("search_test3", None, 0)
        panels._draw_search_box(state)
        imgui.end()
        imgui.end_frame()
    assert imgui.get_io().want_text_input is True


# -- _do_randomize: shared by the button and the R shortcut ---------------------


def test_do_randomize_pushes_history_and_commits():
    app = _make_app()
    pre = app.params.model_copy(deep=True)

    app._commit = lambda new_params: setattr(app, "params", new_params)  # type: ignore[method-assign]
    app._push_history = lambda p: app._undo_stack.append((p, None, None))  # type: ignore[method-assign]
    reset_calls: list[bool] = []
    app._reset_working_copy = lambda: reset_calls.append(True)  # type: ignore[method-assign]

    app._do_randomize()

    assert len(app._undo_stack) == 1
    assert app._undo_stack[-1][0] == pre, "history entry is the pre-randomize state"
    assert app.params != pre, "randomize actually changed something"
    assert reset_calls == [True], "the pending working copy is reset (discrete action wins)"


def test_do_randomize_matches_button_body_via_real_app_wiring():
    """Drive _do_randomize through the REAL (non-stubbed) _push_history/
    _commit/_reset_working_copy -- the same integration test_undo_redo.py
    uses for discrete actions -- to confirm the extracted method behaves
    exactly like the former inline button body (test_preset_identity.py's
    test_randomize_leaves_identity_but_reads_dirty replicated that body by
    hand; this exercises the shared method itself)."""
    from gasgiant.engine.invalidation import diff_tiers

    class FakeSim:
        def __init__(self, params: PlanetParams) -> None:
            self.params = params

        def update_params(self, new):
            tiers = diff_tiers(self.params, new)
            self.params = new
            return tiers

    class FakeViewport:
        def mark_stale(self) -> None:
            pass

    app = _make_app()
    app.sim = FakeSim(app.params)
    app.viewport = FakeViewport()
    app._active_preset = ("gas_giant_warm", "factory")
    app._pristine = app.params.model_copy(deep=True)

    pre = app.params
    app._do_randomize()

    assert len(app._undo_stack) == 1
    assert app._undo_stack[-1][0] == pre
    assert app._active_preset == ("gas_giant_warm", "factory"), "identity unchanged"
    assert app._live is app.params, "pending working copy reset to the new committed params"
    assert app._gesture_base is None


# -- _open_save_dialog: same path as the Save button -----------------------------


def test_open_save_dialog_sets_dialog(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "USER_PRESET_DIR", tmp_path)

    calls: list[tuple] = []

    class FakeDialog:
        pass

    def fake_save_file(title, default, filters):
        calls.append((title, default, filters))
        return FakeDialog()

    monkeypatch.setattr(main.pfd, "save_file", fake_save_file)

    app = _make_app()
    app._open_save_dialog()

    assert app._dialog is not None
    kind, dlg = app._dialog
    assert kind == "save"
    assert isinstance(dlg, FakeDialog)
    assert len(calls) == 1
    title, default, _filters = calls[0]
    assert default == str(tmp_path / "preset.json")


def test_open_save_dialog_noop_if_already_open(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "USER_PRESET_DIR", tmp_path)
    calls: list[bool] = []
    monkeypatch.setattr(main.pfd, "save_file", lambda *a, **k: calls.append(True))

    app = _make_app()
    sentinel = ("load", object())
    app._dialog = sentinel
    app._open_save_dialog()

    assert app._dialog is sentinel, "an already-open dialog must not be clobbered"
    assert calls == [], "save_file must not be invoked while a dialog is already open"


# -- Regression: input_int seed widgets + begin_popup_context_item() ------------
#
# imgui.input_int()'s default +/- stepper buttons wrap the widget in a
# BeginGroup()/EndGroup() pair. EndGroup() finishes with ItemAdd(bb, id=0), so
# g.LastItemData.ID is 0 immediately after an input_int() call.
# begin_popup_context_item() called with no explicit str_id falls back to
# that last-item id and hits imgui's IM_ASSERT(id != 0) -- which fired every
# frame, unconditionally, the instant the Controls panel drew (frame 1 of the
# real app). Both call sites (the header seed control in main.py and the
# panel's seed leaf in panels.py) now pass an explicit str_id. These tests
# drive a REAL headless imgui frame (not a mock) through both, the exact
# thing no test did before this bug shipped.


def test_seed_header_control_survives_real_frame(imgui_ctx):
    app = _make_app()
    _frame(draw_fn=app._draw_seed_header_control)
    _frame(draw_fn=app._draw_seed_header_control)  # a second frame: no stale-id carryover


def test_panel_seed_leaf_survives_real_frame(imgui_ctx):
    """Drives the full reflected panel tree (draw_params_panel), which
    reaches the ``seed`` field -- the only ``input_int`` leaf in the tree
    (panels._draw_leaf, kind == "int" with a >1e6 range) -- via the same code
    path the live Controls panel uses."""
    state = PanelState()
    _frame(draw_fn=lambda: panels.draw_params_panel(PlanetParams(), state))
    _frame(draw_fn=lambda: panels.draw_params_panel(PlanetParams(), state))


# -- #3 refutation: input_int DOES report is_item_deactivated_after_edit --------
#
# Two reviewers suspected the seed input_int never commits: input_int wraps the
# widget in BeginGroup/EndGroup, EndGroup leaves LastItemData.ID == 0, and they
# reasoned that 0 id would also defeat is_item_deactivated_after_edit() (the
# `committed` signal panels._draw_leaf reads for _SINGLE_ITEM_KINDS, which
# includes "int"). It does NOT: EndGroup forwards an explicit Deactivated status
# flag, so the id==0 quirk only ever affected begin_popup_context_item (the real,
# already-fixed crash). This drives a real focus -> type -> Enter cycle through an
# input_int and pins that the commit signal fires -- the precondition the seed
# leaf's RESTART rebuild depends on. Guards against a future imgui_bundle upgrade
# silently breaking it.


class _IntBox:
    def __init__(self, value: int) -> None:
        self.value = value
        self.committed_ever = False


def _draw_input_int(box: _IntBox, *, click: bool = False) -> None:
    changed, box.value = imgui.input_int("seed", box.value)
    # exactly what panels._draw_leaf does for _SINGLE_ITEM_KINDS
    if imgui.is_item_deactivated_after_edit():
        box.committed_ever = True
    if click:
        io = imgui.get_io()
        rmin, rmax = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        cx = rmin.x + (rmax.x - rmin.x) * 0.15  # the editable text field, clear of +/-
        io.add_mouse_pos_event(cx, (rmin.y + rmax.y) / 2)
        io.add_mouse_button_event(0, True)


def test_input_int_reports_commit_on_type_then_enter(imgui_ctx):
    box = _IntBox(100)
    _frame(draw_fn=lambda: _draw_input_int(box))              # warm-up (layout rect)
    _frame(draw_fn=lambda: _draw_input_int(box, click=True))  # focus the text field
    for _ in range(3):
        _frame(draw_fn=lambda: _draw_input_int(box))
    assert imgui.get_io().want_text_input, "setup failed to focus the input_int"
    imgui.get_io().add_input_character(ord("9"))              # edit the value
    _frame(draw_fn=lambda: _draw_input_int(box))
    imgui.get_io().add_key_event(imgui.Key.enter, True)       # commit
    imgui.get_io().add_key_event(imgui.Key.enter, False)
    for _ in range(3):
        _frame(draw_fn=lambda: _draw_input_int(box))
    assert box.committed_ever, "input_int must report is_item_deactivated_after_edit on Enter"

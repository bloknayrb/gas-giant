"""Tooltips must wrap, and must never hide text behind a scrollbar.

The width assertions are the obvious half. The ``scroll_max`` assertions are
the ones that actually pin the behaviour -- see the sizing invariant in
``gasgiant.app.tooltips``: a width-only fix passes every width test while
silently truncating on the other axis.
"""

from __future__ import annotations

import pytest

from gasgiant.params.model import PlanetParams, StormsParams, iter_pfields

panels = pytest.importorskip("gasgiant.app.panels")
tooltips = pytest.importorskip("gasgiant.app.tooltips")
imgui = pytest.importorskip("imgui_bundle.imgui")
imgui_internal = pytest.importorskip("imgui_bundle.imgui.internal")

# The real worst case in the corpus, not a synthetic string: if this field is
# ever shortened the test still holds, it just stops being the tightest probe.
LONGEST = StormsParams.model_fields["hero_emergence"].description or ""


def _all_descriptions():
    """``(dotted_path, description)`` for every pfield leaf that has one."""
    return [(leaf.path, leaf.description) for leaf in iter_pfields() if leaf.description]


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


def _draw(text: str, size: tuple[float, float] = (800.0, 600.0), frames: int = 4):
    """Submit ``text`` as a tooltip for a few frames; return its window.

    Several frames because an auto-resize window reports a placeholder size on
    the frame it is created (a single-frame read gives a bogus ~16 px and has
    fooled a reviewer already).
    """
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(*size)
    for _ in range(frames):
        imgui.new_frame()
        imgui.begin("host", None, 0)
        tooltips.tooltip(text)
        imgui.end()
        imgui.end_frame()
        imgui.render()
    return _live_tooltip_window()


def _live_tooltip_window():
    """The tooltip window drawn THIS frame, or None.

    Both guards are load-bearing. ``find_window_by_name`` returns a window that
    persists across frames carrying its LAST geometry, so liveness has to be
    checked or a stale size reads as current. And a superseded tooltip still
    reports ``last_frame_active == frame_count`` while holding stale geometry
    -- it is hidden, not retired -- so ``hidden_frames_can_skip_items`` is the
    property that actually identifies the visible one.

    Only ``##Tooltip_00`` is ever created here: ``tooltips.tooltip`` is the
    app's sole tooltip opener (enforced by ``test_tooltip_guard.py``) and each
    ``_draw`` frame calls it once. Higher indices appear only when a frame
    submits two tooltips, which these tests never do.
    """
    win = imgui_internal.find_window_by_name("##Tooltip_00")
    if (
        win is not None
        and win.last_frame_active == imgui.get_frame_count()
        and win.hidden_frames_can_skip_items == 0
    ):
        return win
    return None


def test_long_tooltip_wraps_instead_of_running_off_screen(imgui_ctx):
    win = _draw(LONGEST)
    assert win is not None, "no live tooltip window"
    expected = tooltips._layout(LONGEST)[0]
    padding = imgui.get_style().window_padding.x
    assert win.size.x <= expected + 2.0 * padding, "wider than the wrap position"
    assert win.size.x < imgui.get_io().display_size.x * 0.6, "still eating the screen"
    assert win.size.y > 3.0 * imgui.get_font_size(), "did not actually wrap to many lines"


def test_unwrapped_would_have_run_off_screen(imgui_ctx):
    """States the regression the wrap exists to prevent, so the test above
    reads as a fix rather than an arbitrary bound."""
    imgui.new_frame()
    assert imgui.calc_text_size(LONGEST).x > imgui.get_io().display_size.x * 3.0
    imgui.end_frame()


@pytest.mark.parametrize("size", [(1700.0, 980.0), (800.0, 600.0), (520.0, 400.0)])
def test_tooltip_never_hides_text_behind_an_unreachable_scrollbar(imgui_ctx, size):
    """The core guarantee. A tooltip window is NoInputs, so any scrollable
    overflow is unreachable and therefore invisible -- on EITHER axis.

    All three sizes exercise a different branch of the sizing solve, so none is
    padding: 1700x980 is em-limited (455 px, no widening); 800x600 is
    viewport-fraction-limited (320 px, no widening); 520x400 is where the
    minimum-width floor binds AND the widen loop actually runs (260 -> 455).
    520x400 is also the discriminating case -- a fixed-width design that
    ignores height produces scroll_max == (14, 298) there, i.e. 298 px of
    silently dropped text.
    """
    win = _draw(LONGEST, size=size)
    assert win is not None
    assert win.scroll_max.y == 0.0, f"{win.scroll_max.y:.0f}px of text hidden vertically"
    assert win.scroll_max.x == 0.0, f"{win.scroll_max.x:.0f}px of text hidden horizontally"


def test_text_too_long_for_any_wrap_is_trimmed_not_hidden(imgui_ctx):
    """When even the widest allowed wrap overflows, trim visibly rather than
    let imgui hide the tail behind a scrollbar nobody can move.

    Asserts the trim itself rather than re-checking scroll_max: the scroll
    assertions above already cover that, and on their own they would also pass
    an implementation that rendered nothing at all.
    """
    huge = LONGEST * 4
    imgui.get_io().display_size = imgui.ImVec2(520.0, 400.0)
    imgui.new_frame()
    width, budget = tooltips._layout(huge)
    fitted = tooltips._fit(huge, width, budget)
    fits = imgui.calc_text_size(fitted, wrap_width=width).y
    imgui.end_frame()
    assert fitted.endswith("...") and len(fitted) < len(huge), "should have been trimmed"
    assert huge.startswith(fitted[:-3].rstrip()), "trim must be a prefix, not mangled"
    assert fits <= budget, "trimmed text still overflows the height budget"


class _TooManyMeasurements(RuntimeError):
    pass


def test_widen_loop_cannot_spin_forever(imgui_ctx, monkeypatch):
    """A minimized window reports a ~0-height viewport, where an unbounded
    widen-until-it-fits loop never exits and freezes the GUI inside a frame.

    Budgets the MEASUREMENT rather than asserting on the result: there is no
    pytest-timeout in this repo, so a non-terminating loop would hang the job
    rather than fail it -- a red test is worth far more than a stalled runner.
    """
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(600.0, 12.0)
    real, calls = imgui.calc_text_size, {"n": 0}

    def budgeted(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 64:
            raise _TooManyMeasurements("widen loop did not terminate")
        return real(*args, **kwargs)

    imgui.new_frame()
    monkeypatch.setattr(tooltips.imgui, "calc_text_size", budgeted)
    try:
        width = tooltips._layout(LONGEST)[0]
    finally:
        monkeypatch.undo()
        imgui.end_frame()
    assert width > 0.0, "a negative wrap disables wrapping entirely"


@pytest.mark.parametrize("size", [(400.0, 300.0), (320.0, 240.0), (200.0, 150.0)])
def test_the_guarantee_holds_down_to_the_smallest_usable_window(imgui_ctx, size):
    """Nothing hidden, on either axis, at every viewport a person could
    plausibly shrink the studio to. 200x150 is the measured floor for a clean
    (0, 0); below that the window is smaller than the text it must show and no
    layout can win (see the degenerate test)."""
    win = _draw(LONGEST, size=size)
    assert win is not None
    assert win.scroll_max.x == 0.0, f"{win.scroll_max.x:.0f}px hidden horizontally"
    assert win.scroll_max.y == 0.0, f"{win.scroll_max.y:.0f}px hidden vertically"


@pytest.mark.parametrize("size", [(16.0, 600.0), (600.0, 12.0)])
def test_degenerate_viewports_degrade_gracefully(imgui_ctx, size):
    """Below any usable size the text cannot fit and some overflow is
    unavoidable -- the point is that it stays BOUNDED.

    Unclamped, the arithmetic goes negative here, and imgui reads a negative
    wrap as "do not wrap at all": measured 9128 px of a single unwrapped line
    behind an unreachable scrollbar, i.e. the original bug reinstated by the
    fix's own edge case, with ``_fit`` measuring at the same negative value and
    reporting a fit. The positive clamps hold it to tens of pixels.
    """
    win = _draw(LONGEST, size=size)
    assert win is not None
    assert win.scroll_max.x < 200.0, f"{win.scroll_max.x:.0f}px -- wrapping is disabled"


@pytest.mark.parametrize("size", [(1700.0, 980.0), (800.0, 600.0), (520.0, 400.0)])
def test_overlong_text_is_trimmed_by_the_drawn_tooltip(imgui_ctx, size):
    """The trim must be reached THROUGH ``tooltip()``, not only by calling
    ``_fit`` directly.

    Mutation-checked: dropping the ``_fit`` call from the draw path left the
    rest of this file green while hiding 636 px at 800x600, because no other
    test drives the render path with text long enough to need trimming.
    """
    win = _draw(LONGEST * 4, size=size)
    assert win is not None
    assert win.scroll_max.y == 0.0, f"{win.scroll_max.y:.0f}px of text hidden vertically"
    assert win.scroll_max.x == 0.0, f"{win.scroll_max.x:.0f}px of text hidden horizontally"


@pytest.mark.parametrize("size", [(1700.0, 980.0), (800.0, 600.0), (520.0, 400.0)])
def test_nothing_is_dropped_from_a_description_that_fits(imgui_ctx, size):
    """A description that CAN be shown whole must be shown whole.

    ``scroll_max == 0`` is trivially satisfied by rendering LESS text, so on
    its own it passes a draw path that truncates -- and it also leaves the
    widen loop's purpose unpinned (deleting the loop keeps every other test
    green, because ``_fit`` then quietly absorbs the overflow as a trim).
    Derives the expectation from the DRAWN window, not from ``_layout``, so a
    mis-sized layout cannot validate itself.
    """
    win = _draw(LONGEST, size=size)
    assert win is not None
    imgui.new_frame()
    pad = imgui.get_style().window_padding
    used_wrap = win.size.x - 2.0 * pad.x
    needed = imgui.calc_text_size(LONGEST, wrap_width=used_wrap).y
    imgui.end_frame()
    assert win.size.y >= needed + 2.0 * pad.y - 2.0, (
        f"window is {win.size.y:.0f}px but the whole text needs "
        f"{needed + 2.0 * pad.y:.0f}px at wrap {used_wrap:.0f} -- text was dropped"
    )


def test_no_shipped_description_ever_needs_trimming(imgui_ctx):
    """``_fit`` is a last resort for degenerate viewports, NOT a licence to
    ship descriptions too long to display. If a real pfield ever needs it that
    is an authoring bug, and it should fail here rather than silently lose its
    tail in the GUI (where the warning goes to a logger the app pins at INFO).

    Green at 0/226 today, so it lands as a pure ratchet.
    """
    imgui.get_io().display_size = imgui.ImVec2(640.0, 360.0)
    imgui.new_frame()
    over = [
        name
        for name, desc in _all_descriptions()
        if tooltips._fit(desc, *tooltips._layout(desc)) != desc
    ]
    imgui.end_frame()
    assert not over, f"descriptions too long to display at 640x360: {over}"


def test_layout_is_safe_before_the_first_frame():
    """``calc_text_size`` segfaults the interpreter before the first
    ``new_frame()`` (exit 139, no traceback). ``get_font_size()`` reports 0.0
    there, which is the signal the guard keys on."""
    ctx = imgui.create_context()
    imgui.get_io().set_ini_filename(None)
    imgui.get_io().display_size = imgui.ImVec2(800.0, 600.0)
    try:
        assert imgui.get_font_size() == 0.0, "premise: no font metrics yet"
        assert tooltips._layout(LONGEST)[0] > 0.0
    finally:
        imgui.destroy_context(ctx)


def test_empty_and_none_draw_nothing(imgui_ctx):
    for text in ("", None):
        assert _draw(text) is None, f"{text!r} should not open a tooltip"


def test_long_description_reaches_the_wrapping_helper(imgui_ctx, monkeypatch):
    """Every parameter tooltip must route through the helper -- otherwise it is
    an unwrapped tooltip. Spying works only because ``item_tooltip`` owns the
    hover test: headless there is no mouse, so a call-site hover check would
    make this unobservable."""
    seen: list[str | None] = []
    monkeypatch.setattr(panels, "item_tooltip", seen.append)
    state = panels.PanelState(search="hero_emergence", show_advanced=True)
    imgui.new_frame()
    imgui.begin("host", None, 0)
    panels.draw_params_panel(PlanetParams(), state)
    imgui.end()
    imgui.end_frame()
    assert LONGEST in seen


def test_help_marker_delegates_to_the_helper(imgui_ctx, monkeypatch):
    seen: list[str | None] = []
    monkeypatch.setattr(panels, "item_tooltip", seen.append)
    imgui.new_frame()
    imgui.begin("host", None, 0)
    panels._draw_help_marker("section blurb")
    imgui.end()
    imgui.end_frame()
    assert seen == ["section blurb"]


@pytest.mark.parametrize("search", ["", "hero", "color"])
def test_full_panel_walk_does_not_crash(imgui_ctx, search):
    """``render()`` matters: an unbalanced begin_tooltip/end_tooltip surfaces
    there as a clean RuntimeError rather than silently mis-nesting."""
    state = panels.PanelState(search=search, show_advanced=True)
    imgui.new_frame()
    imgui.begin("host", None, 0)
    panels.draw_params_panel(PlanetParams(), state)
    imgui.end()
    imgui.end_frame()
    imgui.render()

"""Tooltips must wrap, and must never hide text behind a scrollbar.

The width assertions are the obvious half. The ``scroll_max`` assertions are
the ones that actually pin the behaviour -- see the sizing invariant in
``gasgiant.app.tooltips``: a width-only fix passes every width test while
silently truncating on the other axis.
"""

from __future__ import annotations

import pytest

from gasgiant.params.model import PlanetParams, StormsParams

panels = pytest.importorskip("gasgiant.app.panels")
tooltips = pytest.importorskip("gasgiant.app.tooltips")
imgui = pytest.importorskip("imgui_bundle.imgui")
imgui_internal = pytest.importorskip("imgui_bundle.imgui.internal")

# The real worst case in the corpus, not a synthetic string: if this field is
# ever shortened the test still holds, it just stops being the tightest probe.
LONGEST = StormsParams.model_fields["hero_emergence"].description or ""


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
    expected = tooltips.wrap_width(LONGEST)
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


def test_wrap_width_terminates_on_a_degenerate_viewport(imgui_ctx):
    """A minimized window reports a ~0-height viewport. An unbounded
    widen-until-it-fits loop never exits there and freezes the GUI inside a
    frame -- no exception, no timeout. The cap makes termination structural.
    """
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(600.0, 12.0)
    imgui.new_frame()
    width = tooltips.wrap_width(LONGEST)
    imgui.end_frame()
    assert width > 0.0


def test_wrap_width_is_safe_before_the_first_frame():
    """``calc_text_size`` segfaults the interpreter before the first
    ``new_frame()`` (exit 139, no traceback). ``get_font_size()`` reports 0.0
    there, which is the signal the guard keys on."""
    ctx = imgui.create_context()
    imgui.get_io().set_ini_filename(None)
    imgui.get_io().display_size = imgui.ImVec2(800.0, 600.0)
    try:
        assert imgui.get_font_size() == 0.0, "premise: no font metrics yet"
        assert tooltips.wrap_width(LONGEST) > 0.0
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

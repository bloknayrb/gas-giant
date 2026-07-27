"""Shared tooltip primitives for the GUI layer.

Every tooltip in the studio is long-form prose -- the ``pfield`` descriptions
reach 1303 characters -- and imgui's default tooltip is a single unwrapped
line, so a long description ran clear off the screen. These helpers are the
ONE place a tooltip window is opened, so the wrap applies everywhere by
construction (enforced by ruff's banned-api rule plus a guard test).

``imgui.set_tooltip`` CANNOT be wrapped from the outside. ``push_text_wrap_pos``
writes ``window->DC.TextWrapPos``, and ``Begin()`` resets that to -1 for any
non-child window -- and ``SetTooltip`` opens its own window. Measured headless:
a 1200-char tooltip under an outer ``push_text_wrap_pos(300.0)`` still came out
1594 px wide on a 1600 px viewport, byte-identical to no push at all. So the
tooltip window has to be opened by hand with the wrap pushed INSIDE it, which
is Dear ImGui's own ``HelpMarker`` demo idiom.

THE SIZING INVARIANT, which the rest of this module exists to hold: a tooltip
window carries ``NoInputs``, so an overflow scrollbar can never be scrolled --
any text imgui pushes out of view is silently unreachable. Width alone does not
achieve that, because a NARROWER wrap makes a TALLER tooltip: measured, the
1303-char description at a 520x400 viewport lost 298 px behind such a
scrollbar. So ``_layout`` widens until the text fits the height budget, never
past the viewport width (or the overflow just moves to the horizontal axis),
and ``_fit`` trims as a last resort when even the widest allowed wrap overflows.
"""

from __future__ import annotations

import logging

from imgui_bundle import imgui

log = logging.getLogger(__name__)

# Target line length in font-size units -- the same 35.0 Dear ImGui's own
# HelpMarker demo uses. At the default 13 px font that is 455 px == 65 chars
# per line (the font is fixed-width 7 px), inside the 45-75 chars/line range
# typography calls readable. Em-relative, NOT a pixel constant, so the tooltip
# tracks hello_imgui's DPI font scaling instead of collapsing to a thin ribbon
# on a HiDPI display.
WRAP_EM = 35.0
# ...but never wider than this fraction of the viewport (455 px of tooltip over
# an 800 px window is a wall), and never narrower than this many ems.
WRAP_VIEWPORT_FRAC = 0.40
WRAP_MIN_EM = 20.0
# Fraction of the viewport height a tooltip may occupy before it is widened.
HEIGHT_BUDGET_FRAC = 0.85
# Growth factor per widening step. 1.15 converges in <= 7 steps from the
# narrowest start to the widest cap at every viewport size measured.
_WIDEN_STEP = 1.15


def _layout(text: str) -> tuple[float, float]:
    """``(wrap_px, height_budget_px)`` for ``text`` this frame.

    Both numbers come from one viewport read so they cannot describe different
    frames -- and so the degenerate-font fallback below applies to the pair
    rather than to the width alone.

    MUST be called inside a frame: ``imgui.calc_text_size`` segfaults the
    process before the first ``new_frame()`` (exit 139, no traceback).
    ``get_font_size()`` reports 0.0 there, which is the signal the guard uses.
    """
    font_size = imgui.get_font_size()
    view = imgui.get_main_viewport().work_size
    pad = imgui.get_style().window_padding
    # -2.0 keeps the window itself (wrap + 2*padding) strictly inside the
    # viewport, so imgui never has to clamp it and grow a horizontal scrollbar.
    max_wrap = view.x - 2.0 * pad.x - 2.0
    budget = HEIGHT_BUDGET_FRAC * view.y - 2.0 * pad.y
    if font_size <= 0.0:
        return WRAP_EM * 13.0, budget  # no font metrics yet; 13 px is imgui's default
    preferred = min(WRAP_EM * font_size, WRAP_VIEWPORT_FRAC * view.x)
    # A floor past the cap would push the wrap back outside the viewport and
    # re-create the horizontal overflow this function exists to prevent.
    floor = min(WRAP_MIN_EM * font_size, max_wrap)
    width = min(max(preferred, floor), max_wrap)
    # Terminates structurally: width climbs monotonically toward max_wrap and
    # the loop is bounded by `width < max_wrap`, so a viewport too short to fit
    # even one line exits at the cap instead of spinning forever (a bare
    # "while it doesn't fit" loop hangs the GUI when the window is minimized,
    # which reports a 0-height viewport).
    while width < max_wrap and imgui.calc_text_size(text, wrap_width=width).y > budget:
        width = min(width * _WIDEN_STEP, max_wrap)
    return width, budget


def wrap_width(text: str) -> float:
    """Wrap position (px) for ``text``, widened to fit the viewport height."""
    return _layout(text)[0]


def _fit(text: str, width: float, budget: float) -> str:
    """``text`` trimmed to the longest prefix that fits ``budget`` at ``width``.

    Only reached when even the widest allowed wrap overflows. Headroom is
    thinner than it looks -- at 520x400 the widest wrap fits ~1729 chars
    against a corpus worst case of 1303 -- so this is rare, not dead.
    Trimming is visibly lossy, but the alternative is the unreachable
    scrollbar described in the module docstring.
    """
    if imgui.calc_text_size(text, wrap_width=width).y <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if imgui.calc_text_size(text[:mid].rstrip() + "...", wrap_width=width).y <= budget:
            lo = mid
        else:
            hi = mid - 1
    log.debug("tooltip trimmed to %d of %d chars at wrap %.0f", lo, len(text), width)
    return text[:lo].rstrip() + "..."


def tooltip(text: str | None) -> None:
    """Draw ``text`` as a word-wrapped tooltip, unconditionally.

    The caller owns the show/hide decision. Reach for this whenever the test
    cannot be ``is_item_hovered()`` *at this point in the frame* -- either the
    trigger is not a widget at all (the storm-placement hint uses a manual rect
    hit-test) or the hover was latched earlier (see ``item_tooltip``'s note on
    containers). Otherwise use ``item_tooltip``.
    """
    if not text:
        return
    width, budget = _layout(text)
    if imgui.begin_tooltip():
        imgui.push_text_wrap_pos(width)
        imgui.text_unformatted(_fit(text, width, budget))
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def item_tooltip(text: str | None) -> None:
    """Wrapped tooltip for the item just drawn.

    The hover test lives HERE, not at the call site, for two reasons: the call
    sites collapse to one line, and a headless panel walk (no mouse, so nothing
    is ever hovered) can spy this one function and observe every tooltip the
    panel offers -- with the hover test left at the call site that wiring is
    untestable without synthesising mouse input.

    CONTAINERS: this reads the item drawn immediately before it, so after an
    ``end_combo``/``end_popup``/``end_child`` that is an item *inside* the
    container, not the container itself. Capture ``is_item_hovered()`` before
    the container body and pass it to ``tooltip()`` instead -- see the
    Scenarios combo in ``main.py``.

    Plain ``is_item_hovered()``, NOT ``imgui.begin_item_tooltip()``: the latter
    implies ``ImGuiHoveredFlags_ForTooltip`` (stationary + short delay +
    allow-when-disabled), which would silently start showing tooltips on the
    export-disabled preset buttons. Worth considering, but as its own decision.
    """
    if text and imgui.is_item_hovered():
        tooltip(text)

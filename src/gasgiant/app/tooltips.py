"""Shared tooltip primitives for the GUI layer.

Every tooltip in the studio is long-form prose -- the ``pfield`` descriptions
reach 1303 characters -- and imgui's default tooltip is a single unwrapped
line, so a long description ran clear off the screen. These helpers are the
ONE place a tooltip window is opened, so the wrap applies everywhere by
construction (enforced by ruff's banned-api rule plus a guard test).

``imgui.set_tooltip`` CANNOT be wrapped from the outside. ``push_text_wrap_pos``
writes ``window->DC.TextWrapPos``, ``Begin()`` resets that to -1, and
``SetTooltip`` opens its own window. Measured headless: a 1200-char tooltip
under an outer ``push_text_wrap_pos(300.0)`` still came out 1594 px wide on a
1600 px viewport, byte-identical to no push at all. So the tooltip window has
to be opened by hand with the wrap pushed INSIDE it, which is Dear ImGui's own
``HelpMarker`` demo idiom.

Measurements in this module were taken against imgui_bundle 1.92.8 with the
default style (``window_padding.x`` 8.0). Character-per-line figures assume
imgui's built-in fixed-width font, which is what the headless tests see; the
shipped app loads hello_imgui's proportional DroidSans, where the same wrap
gives roughly 90 chars/line (see ``WRAP_EM``).

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
# HelpMarker demo uses. Em-relative, NOT a pixel constant, so the tooltip tracks
# hello_imgui's DPI font scaling instead of collapsing to a thin ribbon on a
# HiDPI display.
#
# Chars per line depends on the font and is NOT settled by this number: with
# imgui's built-in fixed-width font (13 px, 7 px/char -- the headless tests)
# 35 em == 455 px == 65 chars, inside the 45-75 range typography calls
# readable; with the DroidSans the app actually loads (avg advance 0.389 em)
# the same 35 em is ~90 chars, above that range. Whether to drop toward ~29 em
# for the shipped font is an open question, deliberately not settled here.
WRAP_EM = 35.0
# ...but never wider than this fraction of the viewport (455 px of tooltip over
# an 800 px window is a wall), and never narrower than this many ems.
WRAP_VIEWPORT_FRAC = 0.40
WRAP_MIN_EM = 20.0
# Fraction of the viewport height a tooltip may occupy before it is widened.
HEIGHT_BUDGET_FRAC = 0.85
# Growth factor per widening step. Step COUNT is not bounded by this constant
# -- traversing floor to cap takes ~5 steps at 520x400 but ~16 at 3840-wide.
# Termination comes from the `width < max_wrap` bound in _layout, not from a
# step budget.
_WIDEN_STEP = 1.15
# Absolute floor for max_wrap, in ems -- distinct from WRAP_MIN_EM above, which
# floors the CHOSEN wrap. This one only binds on a viewport far too narrow to be
# usable, and its job is to keep the cap POSITIVE (see _layout).
_ABSOLUTE_MIN_EM = 4.0
# Texts already reported as truncated. Truncation is a per-frame condition --
# it re-fires ~60x/s while the pointer rests -- so the warning is deduplicated
# by content rather than logged every frame.
_TRIMMED: set[str] = set()


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
    line = imgui.get_text_line_height()
    # Both bounds are clamped POSITIVE, and that is load-bearing rather than
    # defensive. imgui reads a negative wrap as "do not wrap at all", so an
    # unclamped max_wrap on a very narrow viewport reinstates the exact bug
    # this module exists to fix -- measured at 16 px wide: wrap -2.0, and
    # 9128 px of text behind an unreachable horizontal scrollbar, with _fit
    # measuring at the same negative value and cheerfully reporting a fit.
    # A negative budget is the same story on the other axis: _fit trims to a
    # bare "..." that still overflows. Clamping turns both into an ugly but
    # bounded tooltip that _fit can legitimately trim.
    # The -8.0 keeps the window (wrap + 2*padding) strictly inside the viewport
    # so imgui never clamps it and grows a scrollbar. Measured: it holds the
    # guarantee at every viewport from 1700x980 down to 200x150; a tighter
    # margin leaked 1-4 px of horizontal scroll at some sizes.
    max_wrap = max(view.x - 2.0 * pad.x - 8.0, _ABSOLUTE_MIN_EM * max(font_size, 1.0))
    budget = max(HEIGHT_BUDGET_FRAC * view.y - 2.0 * pad.y, line)
    if font_size <= 0.0:
        # Reachable ONLY before the first new_frame(), i.e. only by misuse.
        # Falling back rather than raising because calc_text_size segfaults
        # there (exit 139, no traceback) -- but say so, or the next reader
        # takes this for a supported mode.
        log.warning("tooltip layout requested outside a frame; using default font metrics")
        return min(WRAP_EM * 13.0, max_wrap), budget
    preferred = min(WRAP_EM * font_size, WRAP_VIEWPORT_FRAC * view.x)
    # The outer min is what keeps a floor past the cap from pushing the wrap
    # back outside the viewport and re-creating the horizontal overflow this
    # function exists to prevent.
    width = min(max(preferred, WRAP_MIN_EM * font_size), max_wrap)
    # Terminates structurally: width climbs monotonically toward max_wrap and
    # the loop is bounded by `width < max_wrap`, so a viewport too short to fit
    # even one line exits at the cap instead of spinning forever (a bare
    # "while it doesn't fit" loop hangs the GUI when the window is minimized,
    # which reports a 0-height viewport).
    while width < max_wrap and imgui.calc_text_size(text, wrap_width=width).y > budget:
        width = min(width * _WIDEN_STEP, max_wrap)
    return width, budget


def _fit(text: str, width: float, budget: float) -> str:
    """``text`` trimmed to the longest prefix that fits ``budget`` at ``width``.

    Only reached when even the widest allowed wrap overflows. Headroom is
    thinner than it looks -- at 520x400 this keeps ~1650 chars (arithmetic
    ceiling 1704: 71 cols x 24 lines) against a corpus worst case of 1303 --
    so this is rare, not dead. Trimming is visibly lossy, but the alternative
    is the unreachable scrollbar described in the module docstring.
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
    out = text[:lo].rstrip() + "..."
    if text not in _TRIMMED:
        _TRIMMED.add(text)
        # WARNING, not debug: this drops content the user asked to read, and
        # `configure_logging` pins the root logger at INFO with no --verbose in
        # the GUI, so a debug record would be discarded before it reached any
        # handler -- a silent report of a silent failure.
        log.warning(
            "tooltip truncated: %d of %d chars dropped at wrap %.0f / budget %.0f "
            "(full text is in docs/sliders.md)",
            len(text) - lo, len(text), width, budget,
        )
        # Postcondition, checked under the same dedup guard: the binary search
        # assumes SOME prefix fits, and on a viewport too small for even one
        # line that premise fails and the bare "..." still overflows. Inside the
        # guard because truncation is a per-frame condition -- outside it, both
        # this warning AND its calc_text_size re-fired ~60x/s for as long as the
        # pointer rested. _TRIMMED is content-keyed and lives for the process,
        # so the check now runs at most once per distinct text EVER, not once
        # per hover; that is intended, since the condition depends on the
        # viewport and a later resize will not re-report it.
        if imgui.calc_text_size(out, wrap_width=width).y > budget:
            log.warning("tooltip cannot fit any prefix at wrap %.0f / budget %.0f", width, budget)
    return out


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
        # push_text_wrap_pos takes a window-local POSITION, not a width, and
        # text starts at window_padding.x -- so passing `width` bare would wrap
        # 8 px narrower than everything above measured, rendering up to a line
        # taller than the budget just checked. Measured at wrap 320: 403 px
        # predicted vs 416 px actual. Offsetting by the cursor makes the
        # measurement and the render agree exactly.
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + width)
        imgui.text_unformatted(_fit(text, width, budget))
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def item_tooltip(text: str | None, *, when_disabled: bool = False) -> None:
    """Wrapped tooltip for the item just drawn.

    Pass ``when_disabled=True`` for a tooltip that EXPLAINS why its item is
    greyed out. A disabled item is not hovered by the default flags, so such a
    message is otherwise unreachable exactly when the user needs it -- the
    "ffmpeg not found on PATH" hint on the mp4 checkbox is the live case. It is
    opt-in rather than the default because the preset row goes disabled during
    an export, and those tooltips are descriptions that SHOULD go quiet.

    The hover test lives HERE, not at the call site, so 27 call sites collapse
    to one line and a headless panel walk can spy this one function to observe
    every tooltip the panel offers. (The walk needs no mouse to do that because
    the spy replaces this function outright; driving a real hover headlessly is
    also possible via ``io.add_mouse_pos_event``.)

    Safe after a container's ``end_combo``/``end_popup``/``end_child``:
    ``End()`` restores ``ParentLastItemDataBackup``, so the last item is the
    container (or, for a popup, the item before it) rather than something drawn
    inside. Measured on the Scenarios combo -- hovering it closed reads True
    both before ``begin_combo`` and after ``end_combo``.

    Plain ``is_item_hovered()``, NOT ``imgui.begin_item_tooltip()``: the latter
    implies ``ImGuiHoveredFlags_ForTooltip`` (stationary + short delay +
    allow-when-disabled), which would silently start showing tooltips on the
    export-disabled preset buttons. Worth considering, but as its own decision.
    """
    flags = imgui.HoveredFlags_.allow_when_disabled if when_disabled else 0
    if text and imgui.is_item_hovered(flags):
        tooltip(text)

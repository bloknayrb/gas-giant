"""gasgiant-studio: the live-preview GUI.

Architecture: imgui-bundle's native hello_imgui runner owns the window, GL
context (4.3 core requested), and event/backend handling — version-matched to
imgui by construction. ModernGL attaches to that context in post_init; all
sim/preview GL work happens inside the gui callback on the single GL thread.
Frame loop: handle param edits (validate -> invalidation tiers -> engine),
re-render preview when dirty, draw UI. Phase 4's tiled export will run as
frame-budgeted job slices in this loop.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
from imgui_bundle import hello_imgui, imgui
from imgui_bundle import portable_file_dialogs as pfd
from pydantic import ValidationError

from gasgiant.app.panels import _TIER_GLYPHS, PanelState, draw_params_panel
from gasgiant.app.sphere_preview import SpherePreview
from gasgiant.app.viewport import (
    EquirectViewport,
    lonlat_to_screen,
    nearest_cast_index,
    screen_to_lonlat,
)
from gasgiant.core import brush
from gasgiant.diagnostics import PerfCounter, configure_logging
from gasgiant.engine import Simulation
from gasgiant.engine.invalidation import diff_tiers
from gasgiant.export.exporter import export_job
from gasgiant.export.manifest import MANIFEST_FILENAME
from gasgiant.gl import GpuContext
from gasgiant.jobs import Progress
from gasgiant.params.model import (
    CastKind,
    PlanetParams,
    StormOverride,
    Tier,
    hero_latitude_cap,
)
from gasgiant.params.presets import (
    USER_PRESET_DIR,
    PresetError,
    PresetSource,
    available_presets,
    import_preset,
    load_factory_preset,
    load_preset,
    load_user_preset,
    save_preset,
)
from gasgiant.params.randomize import randomize

log = logging.getLogger(__name__)

class DialogKind(StrEnum):
    """Which native file dialog ``self._dialog`` is currently awaiting. A
    ``StrEnum`` so it stays ``==``-comparable with the bare strings it replaced."""

    LOAD = "load"
    SAVE = "save"
    EXPORT = "export"
    IMPORT = "import"
    PALETTE_FIT = "palette_fit"


@dataclass
class ExportJob:
    """One export in flight: the running job generator, its output directory, and
    the latest ``Progress`` (None until the first slice runs). Bundling these into
    one object means the 'an export is running' predicate (``self._export is not
    None``) and its progress can never disagree the way two separate Optionals
    could."""

    job: object
    out_dir: Path
    progress: Progress | None = None


class UndoRecord(NamedTuple):
    """A history entry: the committed params snapshot plus the preset identity in
    effect when it was captured -- the active-preset ``(name, source)`` pair and
    the pristine baseline (the params as last loaded/saved) that ``dirty`` is
    measured against. ``identity``/``pristine`` are None when no named preset is
    active (e.g. a restored session). The pristine baseline is a full
    PlanetParams rather than a bool because undoing back across a preset load must
    restore the actual baseline so later edits recompute dirty correctly -- a bool
    "was dirty at capture" can't reconstruct that baseline (see _undo). Named
    fields, but still tuple-compatible (index + unpack) for existing call sites."""

    params: PlanetParams
    identity: tuple[str, PresetSource] | None
    pristine: PlanetParams | None

PREVIEW_WIDTH = 2048
SESSION_PATH = Path.home() / ".gasgiant" / "session.json"

# Export-resolution presets for the combo next to the Export button. Widths are
# within ExportParams.width's (512, 16384) bounds; height is always width // 2.
EXPORT_RESOLUTIONS: list[tuple[int, str]] = [
    (1024, "1K"),
    (2048, "2K"),
    (4096, "4K"),
    (8192, "8K"),
    (16384, "16K"),
]

# Playback speed options for draw_equirect's steps-per-frame. "Max" is bounded
# (not "tick until developed" / not unbounded) so a single frame can't stall
# for multiple seconds on a large dev_steps target -- 256 steps/frame is well
# within the per-frame budget the export path already uses in slices (8/frame
# there; playback can afford more since it isn't also deriving+encoding PNGs).
MAX_STEPS_PER_FRAME = 256
SPEED_OPTIONS: list[tuple[int, str]] = [
    (1, "1"),
    (2, "2"),
    (4, "4"),
    (8, "8"),
    (16, "16"),
    (MAX_STEPS_PER_FRAME, "Max"),
]
# Default steps-per-frame for the dev-run preview. Measured (W10a, 2026-07-02,
# RTX 3070, gas_giant_warm): GUI dev-run throughput is FLAT across this value
# (~3.0 steps/s at 2 AND at 8 steps/frame; frame time grows 653 ms -> 2379 ms
# to match), so raising it buys no wall-time and only worsens input latency
# while a dev run is in flight. Kept at the historical 2. NOTE the open perf
# question this measurement exposed: the same preset steps at ~85 ms/step in a
# headless context vs ~330 ms/step under the GUI loop -- root-causing that
# ~4x per-step gap (not chunking) is where the first-launch minutes actually
# are. tick() is chunk-invariant (see Simulation.tick) either way.
DEFAULT_STEPS_PER_FRAME = 2

# T12 in-GUI mask brush: cap live mask re-uploads (and the POST re-derive they
# trigger) to ~30 Hz while dragging, so a fast stroke doesn't re-derive the
# preview every single frame. Gated by accumulated imgui io.delta_time on the
# single GL thread -- never a blocking sleep.
PAINT_UPLOAD_INTERVAL = 1.0 / 30.0


# -- B1-2: friendly GL-failure message ---------------------------------------

def _gl_failure_message(detail: str) -> str:
    """Actionable message for any GL-init failure (context attach, version
    check, first compute-shader compile). Names the hard requirement (the sim
    runs in GLSL 4.3 compute shaders) and the two realistic ways out."""
    return (
        "Gas Giant Studio could not initialize OpenGL 4.3.\n"
        f"  cause: {detail}\n"
        "The simulation runs in GLSL 4.3 compute shaders, so it needs a GPU/driver\n"
        "with OpenGL 4.3 support. Try updating your GPU driver; on a headless or\n"
        "virtual machine, Mesa's llvmpipe software renderer works\n"
        "(set LIBGL_ALWAYS_SOFTWARE=1)."
    )


# -- B1-4/B1-5: export UX helpers (pure, unit-tested) -------------------------

# Every file the exporter can write into the chosen folder -- the overwrite
# check scans for exactly these (the exporter's own cancellation cleanup list).
_EXPORT_FILENAMES = ("color.png", "height.exr", "emission.exr", MANIFEST_FILENAME)

_LAST_EXPORT_PREF = "GasGiantStudio_LastExportDir"


def _export_file_lines(params: PlanetParams) -> list[str]:
    """The files the export will write, for the confirm modal (B1-4: the modal
    never named its outputs). emission.exr appears only when emission is on --
    mirrors export_job's own emission_on gate."""
    w = params.export.width
    h = w // 2
    lines = [
        f"color.png — 16-bit sRGB color map, {w}x{h}",
        f"height.exr — 32-bit float height map, {w}x{h}",
    ]
    if params.emission.enabled:
        lines.append(f"emission.exr — HDR emission map (aurora in alpha), {w}x{h}")
    lines.append(f"{MANIFEST_FILENAME} — manifest; import this in Blender (docs/blender_addon.md)")
    return lines


def _export_conflicts(out_dir: Path) -> list[str]:
    """Map-set files already present in ``out_dir`` (B1-4: the exporter
    silently overwrites -- `mkdir exist_ok=True` + unconditional writes -- so
    the GUI must ask first)."""
    return [name for name in _EXPORT_FILENAMES if (out_dir / name).exists()]


def _export_progress_label(prog: Progress | None) -> str:
    """Progress-bar label for an in-flight export. Phase A of export_job
    finishes the development run first (its Progress says just "developing");
    surface that hold state in plain words (B1-5) instead of an unexplained
    long 'developing' stall before the first tile."""
    if prog is None:
        return "starting"
    if prog.message == "developing":
        return f"finishing dev run {prog.done}/{prog.total}"
    return prog.message


def _export_hold_notice(developing: bool) -> str:
    """The export-hold explanation drawn in the Controls pane -- the SAME pane
    whose sliders are frozen (B1-5: the old notice lived only in the Playback
    pane, a different pane than the frozen controls)."""
    base = (
        "Export in progress — controls are paused; pending edits apply "
        "automatically when it finishes."
    )
    if developing:
        return "Export is finishing the development run first. " + base
    return base


def _pending_hint_text(tiers: set[Tier]) -> str | None:
    """Plain-language pending-edit hint (B1-5: 'release to apply
    (restart/velocity)' was tier jargon). None when nothing heavy is pending.
    RESTART wins the wording when both heavy tiers are pending -- it is the
    costlier consequence."""
    heavy = set(tiers) - {Tier.POST}
    if not heavy:
        return None
    if Tier.RESTART in heavy:
        return "pending change — applies when you release (restarts the development run)"
    return "pending change — applies when you release (rebuilds the flow; the run continues)"


def _load_last_export_dir(load=None) -> Path | None:
    """Last-export folder from hello_imgui's user prefs -- the same ini
    mechanism that already persists window geometry and docking layout, so no
    new config file (B1-4). ``load`` is injectable for tests: the real
    ``hello_imgui.load_user_pref`` requires a live runner (it segfaults
    outside one), so this must only be called from inside the app run
    (init_gl / frame callbacks)."""
    load = load or hello_imgui.load_user_pref
    try:
        raw = load(_LAST_EXPORT_PREF)
    except Exception:  # noqa: BLE001 - prefs are best-effort, never fatal
        return None
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


def _save_last_export_dir(path: Path, save=None) -> None:
    """Persist the last-export folder (see _load_last_export_dir for the
    mechanism/injection rationale). Best-effort: a pref failure must never
    break the export success path."""
    save = save or hello_imgui.save_user_pref
    try:
        save(_LAST_EXPORT_PREF, str(path))
    except Exception:  # noqa: BLE001 - prefs are best-effort, never fatal
        log.warning("could not persist last-export folder %s", path)


def _open_folder(path: Path) -> bool:
    """Open ``path`` in the OS file browser (the post-export 'open folder'
    affordance, B1-4). Returns False on failure so the caller can toast."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as exc:
        log.warning("could not open folder %s: %s", path, exc)
        return False
    return True


# -- B1-3: Help window copy (module constants so tests can pin the content) ---

_HELP_DEV_RUN = (
    "Every planet is grown by a development run: the simulation advects the "
    "cloud tracers for a fixed number of steps (sim.dev_steps) before the "
    "image is final. While it runs you'll see 'developing N/M' with a time "
    "estimate in the Playback pane and overlaid on the viewport — the picture "
    "keeps changing until it completes. Pause/Play/Step and the Speed control "
    "live in the Playback pane. Edits marked R (RESTART) re-run the "
    "development run from step 0; V (VELOCITY) rebuilds the flow and the run "
    "continues; P (POST) recolors instantly."
)

_HELP_DIRTY = (
    "The preset selector shows where the current settings came from. A '*' "
    "after the name means they differ from the loaded preset — save to keep "
    "them. 'unsaved' means this session was restored from your last run and "
    "is not tied to a named preset; use Save... to give it a name."
)

# B2-4: the recurring unglossed vocabulary, collected in one Help block.
_HELP_GLOSSARY = (
    "GRS = Great Red Spot, Jupiter's giant storm; 'hero' storms are its "
    "class. Anticyclone = high-pressure vortex (bright/red ovals like the "
    "GRS); cyclonic = the opposite spin (dark, stormy belts). Retrograde = "
    "flowing against the planet's rotation; prograde = with it. Zones are "
    "the bright bands, belts the dark ones. Some fields take latitudes/"
    "widths in radians: 1 rad = 57.3 degrees (tooltips give the degree "
    "equivalent)."
)

# B4-6: what Undo covers and what it deliberately does not -- the exclusions
# were previously undiscoverable ("I pressed Ctrl+Z and nothing came back").
_HELP_UNDO = (
    "Undo/redo (Ctrl+Z / Ctrl+Y) covers every committed parameter edit — a "
    "whole slider drag counts as one step — plus preset loads, Randomize, "
    "Reroll, and the export settings in the Export... dialog (resolution, "
    "PNG compression). NOT undoable: Restart dev and the playback controls "
    "(Play/Pause/Step/Speed — they steer the run, not the parameters), "
    "per-field locks, saving/overwriting/deleting preset files on disk, "
    "viewport channel/AgX and sun-angle choices, and a finished export's "
    "files. History keeps the most recent 64 steps; older steps fall off "
    "the end."
)


def _shortcuts_enabled() -> bool:
    """Global keyboard shortcuts (Phase 7) fire only when no text-input
    widget currently holds focus -- typing "r" while searching for a field
    like "rim_contrast", or editing the seed field, must not trigger
    Randomize. ``io.want_text_input`` is exactly this signal, already
    maintained by imgui for every text-entry widget (search box, seed
    ``input_int``, any ``input_text``/``input_float`` in the panel).
    Extracted as a standalone predicate (rather than inlined into
    ``StudioApp._handle_shortcuts``) so it can be exercised against a
    headless imgui context without driving a full app frame."""
    return not imgui.get_io().want_text_input


def _format_eta(seconds: float) -> str:
    """Human ETA for the dev run: seconds under a minute, whole minutes above.
    Rounds up so the estimate never under-promises ("~60s" not "~59s")."""
    if seconds < 59.5:
        return f"~{math.ceil(seconds)}s left"
    return f"~{math.ceil(seconds / 60.0)}m left"


def _dev_progress_label(
    done: int,
    target: int,
    playing: bool,
    recomputing: bool,
    spinner: str,
    eta_seconds: float | None = None,
) -> str:
    """Label for the development-progress bar (drawn only while ``done < target``).

    ``tick()`` no-ops once developed, so Pause deliberately freezes the dev-run
    animation. When paused we must NOT keep animating the "recomputing..."
    spinner -- a frozen-but-spinning bar reads as a hang (#1). Instead say so
    plainly and point at the way out. The playing branch leads with a verb
    (B1-1: a bare "N/M" next to a churning half-formed planet reads as
    working, broken, or done with equal probability), plus an ETA once
    DevRateSampler has enough span. It deliberately promises no speed-combo
    speedup -- measured throughput is flat across steps-per-frame (see
    DEFAULT_STEPS_PER_FRAME). Extracted as a pure function so
    the label logic is unit-testable without an imgui frame or the
    ``time.monotonic`` spinner clock."""
    if not playing:
        return f"paused {done}/{target} (Play/Step to develop)"
    if recomputing:
        return f"{spinner} recomputing... {done}/{target}"
    eta = f" ({_format_eta(eta_seconds)})" if eta_seconds is not None else ""
    return f"developing {done}/{target}{eta}"


def _dev_overlay_text(
    done: int, target: int, playing: bool, eta_seconds: float | None
) -> str | None:
    """One-line viewport overlay while the dev run is visibly evolving (B1-1:
    the progress bar lives in a 10%-height side pane a new user never opens).
    Hidden once developed; hidden while paused because the Performance-pane
    label already says "paused ... (Play/Step to develop)" and a paused overlay
    on top of the image would nag."""
    if done >= target or not playing:
        return None
    eta = f" ({_format_eta(eta_seconds)})" if eta_seconds is not None else ""
    return f"Developing planet — {done}/{target}{eta}"


class DevRateSampler:
    """Rolling steps/sec over the last few seconds of the dev run, for the ETA.

    Pure (caller supplies ``now``) so it is unit-testable without patching the
    clock. ETA is withheld until the window spans >= 2 s of wall time with real
    progress -- a first-frame estimate from one giant chunk would be noise. A
    steps_done that moves backwards (Restart dev, RESTART-tier commit) resets
    the window automatically."""

    _WINDOW_S = 5.0
    _MIN_SPAN_S = 2.0

    def __init__(self) -> None:
        self._samples: deque[tuple[float, int]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def add(self, now: float, steps_done: int) -> None:
        if self._samples and steps_done < self._samples[-1][1]:
            self._samples.clear()  # dev run restarted
        self._samples.append((now, steps_done))
        while self._samples and now - self._samples[0][0] > self._WINDOW_S:
            self._samples.popleft()

    def eta_seconds(self, now: float, remaining: int) -> float | None:
        if remaining <= 0 or len(self._samples) < 2:
            return None
        t0, s0 = self._samples[0]
        t1, s1 = self._samples[-1]
        span, steps = t1 - t0, s1 - s0
        if span < self._MIN_SPAN_S or steps <= 0:
            return None
        return remaining / (steps / span)


class Toasts:
    def __init__(self) -> None:
        self._items: list[tuple[str, bool, float]] = []  # (message, is_error, expiry)

    def info(self, message: str) -> None:
        self._items.append((message, False, time.monotonic() + 4.0))

    def error(self, message: str) -> None:
        log.error("%s", message)
        self._items.append((message, True, time.monotonic() + 8.0))

    def draw(self) -> None:
        now = time.monotonic()
        self._items = [t for t in self._items if t[2] > now]
        if not self._items:
            return
        vp = imgui.get_main_viewport()
        imgui.set_next_window_pos(
            imgui.ImVec2(vp.pos.x + vp.size.x - 12.0, vp.pos.y + vp.size.y - 12.0),
            imgui.Cond_.always,
            imgui.ImVec2(1.0, 1.0),
        )
        imgui.set_next_window_bg_alpha(0.75)
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_focus_on_appearing
            | imgui.WindowFlags_.no_nav
        )
        if imgui.begin("##toasts", None, flags)[0]:
            for message, is_error, _ in self._items[-6:]:
                color = (1.0, 0.45, 0.4, 1.0) if is_error else (0.7, 0.9, 0.7, 1.0)
                imgui.text_colored(imgui.ImVec4(*color), message)
        imgui.end()


class StudioApp:
    """All state behind the hello_imgui callbacks. GL members are created in
    init_gl (post_init), never at construction."""

    def __init__(self) -> None:
        self.gpu: GpuContext | None = None
        self.sim: Simulation | None = None
        self.viewport: EquirectViewport | None = None
        self.sphere: SpherePreview | None = None
        self.params = self._load_session_or_default()
        # Working copy the panel edits each frame. Distinct from self.params
        # (the committed/engine state): a heavy (velocity/restart) edit lives in
        # _live until the gesture is released, so a drag rebuilds once instead of
        # once per frame. _gesture_base is the committed snapshot captured at the
        # start of a gesture (consumed by Phase 2 undo coalescing).
        self._live: PlanetParams = self.params
        self._gesture_base: PlanetParams | None = None
        # Phase 3 panel-navigation state (search/locked/show_advanced),
        # threaded into draw_params_panel and the header seed control.
        self.panel_state = PanelState()
        self.toasts = Toasts()
        # Phase 6 preset identity. _active_preset is the (name, source) pair of
        # the currently-loaded preset (source in {"factory","user","file"}), or
        # None for an unnamed state (a restored session). _pristine is the params
        # as last loaded/saved; dirty = self.params != self._pristine. Both are
        # stored into every undo record so undo across a load restores them.
        if self._session_restored:
            self._active_preset: tuple[str, PresetSource] | None = None
            self.toasts.info("restored previous session")
        else:
            self._active_preset = ("gas_giant_warm", PresetSource.FACTORY)
            self.toasts.info(
                f"started from gas_giant_warm — developing {self.params.sim.dev_steps} steps"
                " (Speed control in the Playback pane)"
            )
        self._pristine: PlanetParams | None = self.params.model_copy(deep=True)
        # Cached merged (factory + user) dropdown list; refreshed on save/load and
        # by the explicit "Refresh presets" button, never re-enumerated per frame.
        self._preset_cache: list[tuple[str, str]] = available_presets()
        self.frame_perf = PerfCounter()
        self.render_perf = PerfCounter()
        self._recomputing = False  # a heavy commit reset the dev run; show progress
        # Phase 5 playback state -- a pure presentation layer over Simulation.tick():
        # these only decide how often/with what max_steps draw_equirect calls tick(),
        # never touch tick()'s own (chunk-invariant) stepping logic.
        self._playing = True  # default matches the old always-ticking behavior
        self._steps_per_frame = DEFAULT_STEPS_PER_FRAME  # measured choice, see constant
        self._dev_rate = DevRateSampler()  # feeds the ETA in label + overlay
        self._single_step_requested = False  # consumed once, the frame after Step
        # Bounded undo/redo history (Phase 2). Each entry is an UndoRecord — a
        # deep copy of a committed params snapshot plus Phase 6 preset-identity
        # placeholders. maxlen=64 evicts the oldest entry automatically.
        self._undo_stack: deque[UndoRecord] = deque(maxlen=64)
        self._redo_stack: deque[UndoRecord] = deque(maxlen=64)
        self._dialog: tuple[DialogKind, object] | None = None
        self._export: ExportJob | None = None
        # B1-2: friendly GL-failure message, set by init_gl on failure and
        # printed by main() after the runner exits (never a raw traceback).
        self.init_error: str | None = None
        # B1-4: last successful export folder. Loaded from hello_imgui user
        # prefs in init_gl (prefs need a live runner), saved on every export
        # success; feeds the folder-picker default, the persistent
        # "last export" line, and the Open-folder button.
        self._last_export_dir: Path | None = None
        # B1-4: a picked export folder that already contains a map set, held
        # here until the user confirms/cancels the overwrite modal.
        self._pending_export: tuple[Path, list[str]] | None = None
        # B1-8: (name, path) of a user preset awaiting overwrite/delete
        # confirmation in their respective modals.
        self._pending_overwrite: tuple[str, Path] | None = None
        self._pending_delete: tuple[str, Path] | None = None
        self._frame_count = 0
        self._smoke_frames = int(os.environ.get("GASGIANT_SMOKE_FRAMES", "0"))
        self._smoke_shot = os.environ.get("GASGIANT_SMOKE_SCREENSHOT", "")
        # Phase 7 help window -- a plain floating imgui window (the same
        # idiom Toasts.draw already uses in this file), toggled by F1 or the
        # "Help (F1)" button, drawn from gui_overlays each frame while visible.
        self._show_help = False
        # A2-2: last observed Simulation.baroclinic_status, so the per-frame
        # check toasts exactly once per transition INTO 'degraded'.
        self._baro_status_seen = "off"
        # T4: click-to-place / drag-to-move storm tool. Mode is a small state
        # machine over the equirect viewport: "none" = tool off, "place" =
        # armed for click-to-place, "drag" = a marker is being dragged (its live
        # position tracks the cursor; the texture only re-develops on release).
        self._storm_tool_mode = "none"
        self._storm_tool_kind = CastKind.OVAL.value  # placement archetype
        self._storm_tool_radius = 0.03  # placement radius (StormOverride default)
        self._drag_index: int | None = None  # cast index being dragged
        self._drag_lonlat: tuple[float, float] | None = None  # live cursor lon/lat
        # T12: in-GUI mask brush. The "paint" tool mode shares T4's single
        # _storm_tool_mode state machine (so paint and Place-storm can't both be
        # armed). _paint_buffer is the CPU-side single-channel equirect mask,
        # lazily allocated on first stamp; it is uploaded via facade.set_mask and
        # re-derived (POST) on a throttle. _paint_base holds a mask adopted from a
        # loaded preset (the canvas painting continues from) so an Undo-stroke
        # rebuild replays onto it rather than wiping it. _paint_strokes is a
        # stroke-replay history SEPARATE from the params undo stack: the mask is
        # sidecar pixel data, not a params-field value, so entangling it with
        # _push_history (which snapshots PlanetParams) would be a category error.
        self._paint_buffer: np.ndarray | None = None
        self._paint_base: np.ndarray | None = None
        self._paint_strokes: list[tuple[float, float, float, float, bool]] = []
        self._paint_radius_deg = 12.0
        self._paint_strength = 0.5
        self._paint_erase = False
        self._paint_dirty = False  # buffer changed since the last mask upload
        self._paint_was_down = False  # left button was painting last frame
        self._paint_upload_accum = PAINT_UPLOAD_INTERVAL  # so the first stamp uploads at once
        # T5: A/B compare + ROI export-res inspector. _snapshot_a is an
        # app-OWNED clone of the color preview (released on retake / never leaked);
        # the ROI tile is likewise app-owned (released on close/retake).
        self._snapshot_a = None  # moderngl.Texture | None
        self._compare_mode = "off"  # "off" | "flash" | "split"
        self._flash_show_a = False  # flash mode: True shows A, False shows live
        self._show_inspect = False
        self._inspect_center = (0.5, 0.5)  # normalized (x, y) ROI tile center
        self._inspect_tile = None  # moderngl.Texture | None (ROI color tile)
        self._inspect_label = ""

    # -- lifecycle ------------------------------------------------------------

    def init_gl(self) -> None:
        """post_init: attach ModernGL to the runner's GL context and build the
        sim. On ANY failure (attach, an actual < 4.3 context the window system
        silently fell back to, the first compute-shader compile) this stores a
        friendly message and asks the runner to exit instead of re-raising --
        an exception thrown through the native callback surfaced as a raw
        traceback, the first minute of the first-run journey (B1-2)."""
        try:
            # Build EVERYTHING into locals first: a partial self-assignment
            # (sim set, then viewport/sphere construction failing -- both
            # compile shaders, exactly the failure class handled here) would
            # slip past the draw callbacks' `if self.sim is None` guards and
            # crash on self.viewport next frame, through the native callback.
            # self.* is assigned only after every constructor succeeded.
            gpu = GpuContext.attach()
            version = gpu.ctx.version_code
            if version < 430:
                raise RuntimeError(
                    f"the created OpenGL context is {version // 100}.{(version % 100) // 10}, "
                    "but 4.3 is required"
                )
            sim = Simulation(self.params, gpu)
            viewport = EquirectViewport(gpu)
            sphere = SpherePreview(gpu)
        except Exception as exc:  # noqa: BLE001 - translated, never swallowed (logged + shown)
            log.exception("GL initialization failed")
            self.init_error = _gl_failure_message(str(exc) or type(exc).__name__)
            hello_imgui.get_runner_params().app_shall_exit = True
            return
        self.gpu = gpu
        self.sim = sim
        self.viewport = viewport
        self.sphere = sphere
        log.info("GL ready: %s", self.gpu.ctx.info.get("GL_RENDERER", "?"))
        self._last_export_dir = _load_last_export_dir()  # prefs need a live runner

    def shutdown(self) -> None:
        self._save_session()

    def _load_session_or_default(self) -> PlanetParams:
        """Restore the last session if present, else the default preset. Sets
        ``self._session_restored`` so ``__init__`` can toast which happened and
        seed the right preset identity (Phase 6)."""
        self._session_restored = False
        if SESSION_PATH.is_file():
            self._backup_old_format_session()
            try:
                params = load_preset(SESSION_PATH)
                self._session_restored = True
                return params
            except (PresetError, OSError) as exc:
                log.warning("session restore failed (%s); using default preset", exc)
        return load_factory_preset("gas_giant_warm")

    def _backup_old_format_session(self) -> None:
        """Before the first load that would migrate (and later overwrite) an
        older-format session, keep the original next to it."""
        import json

        from gasgiant.params.migrations import CURRENT_PRESET_FORMAT

        backup = SESSION_PATH.with_suffix(".json.bak")
        try:
            raw = SESSION_PATH.read_bytes()
            doc = json.loads(raw)
        except (OSError, ValueError):
            return  # unreadable session: the load below reports it
        if doc.get("preset_format", 1) < CURRENT_PRESET_FORMAT and not backup.exists():
            # Separate the WRITE from the read above: a failed backup write is
            # NOT the "unreadable session" case, and swallowing it silently would
            # let migration proceed and shutdown overwrite the user's only
            # pre-migration copy with no trace (#6). Migrate anyway (the load
            # still works) but leave a warning.
            try:
                backup.write_bytes(raw)
                log.info("backed up pre-migration session to %s", backup)
            except OSError as exc:
                log.warning("could not back up pre-migration session (%s); migrating anyway", exc)

    def _save_session(self) -> None:
        try:
            # The session is machine-local state: keep the mask's ABSOLUTE path
            # (do NOT relativize/copy a sidecar next to session.json).
            save_preset(self.params, SESSION_PATH, name="session", relativize_mask=False)
        except OSError as exc:
            log.warning("session autosave failed: %s", exc)

    # -- parameter commits -------------------------------------------------------

    def _commit(self, new_params: PlanetParams) -> None:
        tiers = self.sim.update_params(new_params)
        self.params = new_params
        if tiers:
            self.viewport.mark_stale()
        if tiers - {Tier.POST}:  # velocity/restart reset the dev run
            self._recomputing = True

    def _record(self, params: PlanetParams) -> UndoRecord:
        """Snapshot params into a history record. Stores a deep copy of params
        (never a shared reference) because in-place assignment on params is a
        supported pattern (validate_assignment=True), plus the preset identity
        (_active_preset / _pristine) in effect right now. _pristine is shared by
        reference -- it is replaced wholesale on identity changes, never mutated
        in place -- so it is safe (and cheaper) not to deep-copy it here."""
        return UndoRecord(params.model_copy(deep=True), self._active_preset, self._pristine)

    def _set_identity(self, active: tuple[str, PresetSource] | None, params: PlanetParams) -> None:
        """Adopt ``active`` as the current preset identity and ``params`` as the
        new pristine baseline (deep-copied so later in-place edits to self.params
        can't leak into it). Called on every authoritative path: preset combo,
        Save, Load file-dialog, Reset-to-default."""
        self._active_preset = active
        self._pristine = params.model_copy(deep=True)

    def _is_dirty(self) -> bool:
        return self._pristine is not None and self.params != self._pristine

    def _active_label(self) -> str:
        """Preview label for the preset combo: the active name (``user/`` prefixed
        for user presets), or ``unsaved`` when no preset is active, plus a ``*``
        when the working params differ from the pristine baseline."""
        if self._active_preset is None:
            name = "unsaved"
        else:
            pname, source = self._active_preset
            name = f"user/{pname}" if source == PresetSource.USER else pname
        return f"{name}{' *' if self._is_dirty() else ''}"

    def _refresh_presets(self) -> None:
        self._preset_cache = available_presets()

    def _load_preset_entry(self, name: str, source: PresetSource) -> None:
        """Load a factory/user preset by (name, source) through the shared
        push-history -> commit -> set-identity -> reset-working-copy path. Load
        failures toast and leave state untouched (no stray undo entry)."""
        if self._export is not None:
            return  # defense-in-depth: never commit mid-export (see draw_controls)
        try:
            params = (
                load_user_preset(name) if source == PresetSource.USER else load_factory_preset(name)
            )
        except (PresetError, OSError) as exc:
            self.toasts.error(str(exc))
            return
        if self.params != params:  # re-selecting the active preset is a no-op undo
            self._push_history(self.params)
        self._commit(params)
        self._set_identity((name, source), params)
        self._reset_working_copy()  # discrete action wins over pending edit
        label = f"user/{name}" if source == PresetSource.USER else name
        self.toasts.info(f"preset: {label}")
        self._toast_param_warnings(params)

    def _toast_param_warnings(self, params: PlanetParams) -> None:
        """B5-6: surface model-level cross-field WARNINGS (valid-but-inert
        configurations, e.g. a vorticity-only storm lever under the kinematic
        solver) once on each preset/file load -- never per frame or per edit."""
        for message in params.validation_warnings():
            self.toasts.info(f"warning: {message}")

    def _push_history(self, params: PlanetParams) -> None:
        """Push the pre-edit/pre-action state onto the undo stack and clear the
        redo stack (a new edit invalidates the redo future). Used by the gesture
        path and every discrete action."""
        self._undo_stack.append(self._record(params))
        self._redo_stack.clear()

    def _reset_working_copy(self) -> None:
        """Drop any pending working-copy edit after a discrete action (preset /
        load / randomize / reroll / undo / redo). Without this a heavy edit left
        pending before the action could resurrect itself the next frame, because
        diff_tiers(self.params, self._live) would still report the abandoned
        change. Phase 2's redo path must call this too."""
        self._live = self.params
        self._gesture_base = None

    def _commit_export_field(self, name: str, value: object) -> None:
        """B4-4: the export modal's editors (resolution combo, PNG-compression
        slider) commit through the SAME ``_process_edit`` pipeline as every
        panel widget -- ONE undo policy for the whole app. That makes output
        edits ordinary undoable history entries, clears redo only via the
        standard a-new-edit-clears-redo rule (no more silently destroyed Redo,
        B4-6), honors the export-in-flight hold (the draft is kept, not
        dropped, and flushes as one undo step), and RELEASES rather than
        discards a pending heavy edit (the old ``_commit_output_setting``
        reset the working copy wholesale, dropping pending drafts)."""
        draft = self._live.model_dump()
        draft["export"][name] = value
        self._process_edit(draft, any_changed=True, any_committed=True)

    def _process_edit(self, draft: dict, any_changed: bool, any_committed: bool) -> None:
        """One frame of panel editing → at most one engine commit.

        POST-tier edits commit every changed frame (cheap derive, stays
        frame-live). Velocity/restart edits commit only on release
        (``any_committed``), so a slider drag rebuilds once. The commit goes
        straight through ``_commit`` — never ``_try_commit_draft`` — because
        ``_live`` is already validated, and re-validating would re-toast.

        Phase 5 (M5 / Round 2 LOW-5): while an export is in flight
        (``self._export is not None``), ALL commits are held back — POST
        included — because export Phase A ticks the live sim then snapshots
        it; committing (even a cheap POST re-derive) or rebuilding mid-export
        can corrupt the in-flight run or the exported color. ``_live`` still
        updates above so the widget the user is dragging doesn't visually snap
        back; the draft is just not applied to the engine yet. It is not
        dropped: ``_flush_pending_edit`` applies it, coalesced into one undo
        entry, the moment the export clears (see ``_run_export_slice`` /
        ``_cancel_export``)."""
        if any_changed:
            try:
                self._live = PlanetParams.model_validate(draft)
            except ValidationError as exc:
                # Transient mid-drag invalid states stay silent -- the active
                # widget just freezes at its last valid _live. But a COMMIT-frame
                # invalid draft (any_committed: Enter/release of a Ctrl+click
                # typed value past the slider bounds) is a deliberate entry, so
                # surface why nothing happened (#5 -- restores the toast the old
                # _try_commit_draft gave before Phase 1 swallowed it).
                if any_committed:
                    self.toasts.error(f"invalid value: {exc.errors()[0]['msg']}")
            if self._gesture_base is None:
                self._gesture_base = self.params  # pre-edit committed state (Phase 2)

        if self._export is not None:
            return  # export in flight: hold the draft, don't touch the engine

        tiers = diff_tiers(self.params, self._live)
        heavy = bool(tiers - {Tier.POST})
        if tiers and (not heavy or any_committed):
            # POST-only → commit live; heavy → only on the release frame. Either
            # way a commit fires only when the diff is non-empty, so a combo
            # re-select of the same value (any_committed, empty diff) is a no-op.
            self._commit(self._live)

        if any_committed:
            # Gesture released: coalesce the whole drag (incl. live POST
            # per-frame commits) into one undo entry. Read _gesture_base BEFORE
            # clearing it. Push only if the gesture actually changed the
            # committed state (guards the same-value combo-reselect no-op, where
            # _gesture_base is None or equals self.params).
            if self._gesture_base is not None and self.params != self._gesture_base:
                self._push_history(self._gesture_base)
            self._gesture_base = None  # gesture released; base consumed

    def _flush_pending_edit(self) -> None:
        """Apply a draft that the export gate held back, the moment the export
        clears. Mirrors a normal gesture release (commit + coalesced undo entry)
        so a drag that finished mid-export lands as exactly one undo step, not
        zero (silently dropped) or many. No-op if nothing was pending."""
        if self._live == self.params:
            self._gesture_base = None
            return
        if self._gesture_base is None:
            self._gesture_base = self.params
        self._commit(self._live)
        if self.params != self._gesture_base:
            self._push_history(self._gesture_base)
        self._gesture_base = None

    def _randomize(self, seed: int) -> PlanetParams:
        """``randomize()`` honoring the panel's lock set (UX G1). The walk over
        ``rand``-bearing fields already skips locked dotted paths; the
        top-level ``seed`` field is special-cased because ``randomize()``
        always overwrites it with the new master seed regardless of locks (it
        isn't reached by the per-field ``rand`` walk at all) -- locking
        "seed" here keeps the stored seed pinned across Randomize/Reroll while
        every other unlocked field still re-rolls from the fresh seed's RNG
        stream."""
        result = randomize(seed, base=self.params, locked=self.panel_state.locked)
        if "seed" in self.panel_state.locked:
            result.seed = self.params.seed
        return result

    def _do_randomize(self) -> None:
        """The Randomize action: push history, roll a fresh time-seeded
        snapshot (honoring locks via ``_randomize``), commit, and drop any
        pending working-copy edit. Shared verbatim by the Randomize button
        and the ``R`` keyboard shortcut (Phase 7) so the two can't drift.
        Time-seeded randomize is captured as a concrete params snapshot
        immediately -- never re-rolled on redo (the module's determinism
        contract)."""
        if self._export is not None:
            return  # defense-in-depth: never commit mid-export (see draw_controls)
        self._push_history(self.params)
        seed = int(time.time_ns() % (2**31 - 1))
        self._commit(self._randomize(seed))
        self._reset_working_copy()  # discrete action wins over pending edit
        self.toasts.info(f"randomized (seed {seed})")

    def _do_reroll(self) -> None:
        """Reroll to the next seed (Reroll button). Extracted from draw_controls
        so it carries the same export guard as the other discrete commits."""
        if self._export is not None:
            return  # defense-in-depth: never commit mid-export (see draw_controls)
        self._push_history(self.params)
        self._commit(self._randomize(self.params.seed + 1))
        self._reset_working_copy()  # discrete action wins over pending edit

    # -- B1-8: preset overwrite/delete lifecycle -------------------------------

    def _active_user_preset_path(self) -> tuple[str, Path] | None:
        """(name, path) of the active USER preset, or None. Only USER presets
        are overwritable/deletable in-app: factory presets are package data,
        and a FILE identity only stored the stem (its full path is gone)."""
        if self._active_preset is None:
            return None
        name, source = self._active_preset
        if source != PresetSource.USER:
            return None
        return name, USER_PRESET_DIR / f"{name}.json"

    def _request_overwrite_active(self) -> None:
        """Ctrl+S / the contextual Save button: stage the overwrite-confirm
        modal for the active user preset, or fall back to the Save-As dialog
        when no user preset is active (factory/file/unsaved) -- Ctrl+S always
        does SOMETHING useful, never a silent no-op (B1-8)."""
        if self._export is not None:
            return  # same gate as every other Ctrl-shortcut commit path
        entry = self._active_user_preset_path()
        if entry is None:
            self._open_save_dialog()
            return
        self._pending_overwrite = entry

    def _params_for_save(self, dest_dir: Path, stem: str) -> PlanetParams:
        """Params to hand ``save_preset``. With a painted mask buffer present,
        write it as a 16-bit grayscale sidecar (``<stem>_mask.png``) next to the
        preset and return a copy whose ``mask.file`` points at it, so T11's
        relativize/sidecar path stores just the portable filename. Without a
        painted buffer this is ``self.params`` unchanged (a preset that already
        references an imported mask keeps flowing through T11's copy path)."""
        # getattr guards the partial StudioApp instances some tests build via
        # __new__ (they never touch the paint tool, so the attrs may be absent).
        buf = getattr(self, "_paint_buffer", None)
        strokes = getattr(self, "_paint_strokes", None)
        base = getattr(self, "_paint_base", None)
        painted = buf is not None and (bool(strokes) or base is not None)
        if not painted:
            return self.params
        dest_dir.mkdir(parents=True, exist_ok=True)
        from gasgiant.export.writers import write_png16_gray
        png = dest_dir / f"{stem}_mask.png"
        write_png16_gray(png, buf)
        out = self.params.model_copy(deep=True)
        out.mask.file = str(png.resolve())
        return out

    def _overwrite_active_preset(self) -> None:
        """Confirmed overwrite: save the current params over the active user
        preset's file and adopt them as the new pristine baseline (clears the
        dirty ``*``)."""
        entry = self._pending_overwrite
        self._pending_overwrite = None
        if entry is None:
            return
        name, path = entry
        try:
            save_preset(self._params_for_save(path.parent, path.stem), path)
        except OSError as exc:
            self.toasts.error(str(exc))
            return
        self._set_identity((name, PresetSource.USER), self.params)
        self._refresh_presets()
        self.toasts.info(f"saved user/{name}")

    def _request_delete_active(self) -> None:
        if self._active_user_preset_path() is not None:
            self._pending_delete = self._active_user_preset_path()

    def _delete_active_preset(self) -> None:
        """Confirmed delete: remove the file, drop the named identity (the
        working params stay loaded -- deleting the file must not yank the
        planet out from under the user -- so the combo reads 'unsaved')."""
        entry = self._pending_delete
        self._pending_delete = None
        if entry is None:
            return
        name, path = entry
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            self.toasts.error(str(exc))
            return
        self._active_preset = None
        self._refresh_presets()
        self.toasts.info(f"deleted user/{name} (current settings kept, now unsaved)")

    def _draw_preset_confirm_modals(self) -> None:
        """The overwrite/delete confirm modals (B1-8), staged by
        ``_pending_overwrite``/``_pending_delete``. Same held-state modal
        idiom as ``_draw_export_overwrite_confirm``."""
        if self._pending_overwrite is not None:
            title = "Overwrite preset?"
            if not imgui.is_popup_open(title):
                imgui.open_popup(title)
            center = imgui.get_main_viewport().get_center()
            imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
            if imgui.begin_popup_modal(title, None, imgui.WindowFlags_.always_auto_resize)[0]:
                name, path = self._pending_overwrite
                imgui.text_wrapped(
                    f"Overwrite user preset '{name}' with the current settings?"
                )
                imgui.text_disabled(str(path))
                imgui.separator()
                if imgui.button("Overwrite##preset"):
                    self._overwrite_active_preset()
                    imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Cancel##overwrite_preset"):
                    self._pending_overwrite = None
                    imgui.close_current_popup()
                imgui.end_popup()
        if self._pending_delete is not None:
            title = "Delete preset?"
            if not imgui.is_popup_open(title):
                imgui.open_popup(title)
            center = imgui.get_main_viewport().get_center()
            imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
            if imgui.begin_popup_modal(title, None, imgui.WindowFlags_.always_auto_resize)[0]:
                name, path = self._pending_delete
                imgui.text_wrapped(
                    f"Delete user preset '{name}'? This removes the file from disk; "
                    "the current settings stay loaded (as 'unsaved')."
                )
                imgui.text_disabled(str(path))
                imgui.separator()
                if imgui.button("Delete##preset"):
                    self._delete_active_preset()
                    imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Cancel##delete_preset"):
                    self._pending_delete = None
                    imgui.close_current_popup()
                imgui.end_popup()

    def _open_save_dialog(self) -> None:
        """Open the preset-save (Save As...) file dialog -- the "Save..."
        button's path, and Ctrl+S's fallback when no user preset is active
        (``_request_overwrite_active``). No-op if a dialog is already open
        (mirrors the button's ``self._dialog is None`` guard)."""
        if self._dialog is not None:
            return
        # Default the save dialog into USER_PRESET_DIR so saved presets land
        # where user_preset_names() enumerates them (created if absent).
        USER_PRESET_DIR.mkdir(parents=True, exist_ok=True)
        default = str(USER_PRESET_DIR / "preset.json")
        self._dialog = (DialogKind.SAVE, pfd.save_file("Save preset", default, ["JSON", "*.json"]))

    def _handle_shortcuts(self) -> None:
        """Global keyboard shortcuts (Phase 7), registered as hello_imgui's
        ``post_new_frame`` callback -- it runs right after ImGui::NewFrame(),
        so ``io.want_text_input`` already reflects whether a text-input
        widget is currently focused, and BEFORE any window is drawn this
        frame, so a flag set here (the search-focus request) is consumed
        later this same frame by ``_draw_search_box``.

        Undo/Redo/Randomize/Save additionally check ``self._export is
        None``: ``draw_controls`` disables their buttons outright while an
        export is in flight (M5 / Round 2 LOW-5), and a shortcut must honor
        that same gate rather than bypass it."""
        if not _shortcuts_enabled():
            return
        io = imgui.get_io()
        ctrl = io.key_ctrl
        exporting = self._export is not None
        # F1/A are TOGGLES: with imgui's default key-repeat, holding the key
        # flips the toggle many times per second and the final state depends on
        # repeat-count parity. repeat=False fires once per physical press. (Undo/
        # Redo/Randomize below deliberately KEEP repeat so a held key fast-repeats
        # through history / rerolls.)
        if imgui.is_key_pressed(imgui.Key.f1, False):
            self._show_help = not self._show_help
        if not ctrl and imgui.is_key_pressed(imgui.Key.slash):
            self.panel_state.focus_search_requested = True
        if not ctrl and imgui.is_key_pressed(imgui.Key.a, False):
            self.panel_state.show_advanced = not self.panel_state.show_advanced
        if not ctrl and imgui.is_key_pressed(imgui.Key.r) and not exporting:
            self._do_randomize()
        if ctrl and imgui.is_key_pressed(imgui.Key.z) and not exporting:
            self._undo()
        if ctrl and imgui.is_key_pressed(imgui.Key.y) and not exporting:
            self._redo()
        if ctrl and imgui.is_key_pressed(imgui.Key.s) and not exporting:
            # B1-8: Ctrl+S overwrites the active USER preset (after a confirm
            # modal); with no user preset active it opens Save As, as before.
            self._request_overwrite_active()

    # -- dialogs --------------------------------------------------------------------

    def _poll_dialog(self) -> None:
        if self._dialog is None:
            return
        kind, dlg = self._dialog
        if not dlg.ready():
            return
        if (
            kind in (DialogKind.LOAD, DialogKind.IMPORT, DialogKind.PALETTE_FIT)
            and self._export is not None
        ):
            # A file was already picked (dialog opened before the export
            # started, or a race with the disabled-button gate below) but
            # applying it now would commit mid-export. Hold the dialog --
            # don't consume its result -- and retry next frame.
            return
        self._dialog = None
        result = dlg.result()
        if not result:
            return
        try:
            if kind == DialogKind.LOAD:
                path = Path(result[0])
                loaded = load_preset(path)  # push only AFTER a successful load
                if self.params != loaded:  # reloading the current file is a no-op undo
                    self._push_history(self.params)
                self._commit(loaded)
                # Loaded-from-file becomes the active identity + dirty baseline.
                self._set_identity((path.stem, "file"), loaded)
                self._reset_working_copy()  # discrete action wins over pending edit
                self.toasts.info(f"loaded {path.name}")
                self._toast_param_warnings(loaded)
            elif kind == DialogKind.IMPORT:
                # B4-5 "Import preset...": validate + install as a durable
                # user preset (vs Load's transient FILE identity), then adopt
                # it as the active preset through the same history path.
                path = Path(result[0])
                name, imported = import_preset(path)
                if self.params != imported:
                    self._push_history(self.params)
                self._commit(imported)
                self._set_identity((name, PresetSource.USER), imported)
                self._reset_working_copy()
                self._refresh_presets()
                self.toasts.info(f"imported {path.name} as user/{name}")
                self._toast_param_warnings(imported)
            elif kind == DialogKind.PALETTE_FIT:
                self._apply_palette_fit(Path(result[0]))
            elif kind == DialogKind.SAVE:
                path = Path(result if isinstance(result, str) else result[0])
                if path.suffix != ".json":
                    path = path.with_suffix(".json")
                save_preset(self._params_for_save(path.parent, path.stem), path)
                # The just-saved preset is the new active/pristine baseline (so
                # dirty resets); refresh so a save into USER_PRESET_DIR appears.
                source = PresetSource.USER if path.parent == USER_PRESET_DIR else PresetSource.FILE
                self._set_identity((path.stem, source), self.params)
                self._refresh_presets()
                self.toasts.info(f"saved {path.name}")
            elif kind == DialogKind.EXPORT:
                out = Path(result)
                conflicts = _export_conflicts(out)
                if conflicts:
                    # B1-4: the exporter itself overwrites silently, so ask
                    # first. Held here; draw_controls draws the confirm modal.
                    self._pending_export = (out, conflicts)
                else:
                    self._start_export(out)
        except (PresetError, OSError, ValueError) as exc:
            self.toasts.error(str(exc))

    # -- UI ----------------------------------------------------------------------------

    def draw_controls(self) -> None:
        if self.sim is None:  # init_gl failed; the runner is already exiting
            imgui.text_wrapped(self.init_error or "initializing...")
            return
        # Every action in this block (Phase 6 preset combo, Load/Save, seed
        # header, Randomize/Reroll, Undo/Redo) commits straight through
        # `self._commit`, bypassing `_process_edit`'s gate -- so while an
        # export is in flight (M5 / Round 2 LOW-5) they're disabled outright
        # rather than silently no-op'ing (a no-op commit here would still push
        # a bogus before==after undo entry). `_poll_dialog` additionally holds
        # a "load" dialog's result if it resolves mid-export via the narrow
        # race where the dialog was opened just before Export was clicked.
        if imgui.button("? Help (F1)"):
            self._show_help = not self._show_help
        imgui.separator()

        exporting = self._export is not None
        imgui.begin_disabled(exporting)
        # Merged factory+user dropdown. Preview shows the active preset + dirty
        # marker; entries render from the cached list (self._preset_cache), never
        # re-enumerating the filesystem per frame.
        imgui.set_next_item_width(160.0)
        if imgui.begin_combo("##preset", self._active_label()):
            for name, source in self._preset_cache:
                label = f"user/{name}" if source == PresetSource.USER else name
                if imgui.selectable(label, False)[0]:
                    self._load_preset_entry(name, source)
            imgui.end_combo()
        imgui.same_line()
        if imgui.button("Refresh presets"):
            self._refresh_presets()
            self.toasts.info("preset list refreshed")
        imgui.same_line()
        if imgui.button("Load...") and self._dialog is None:
            self._dialog = (DialogKind.LOAD, pfd.open_file("Load preset", "", ["JSON", "*.json"]))
        if imgui.is_item_hovered():
            imgui.set_tooltip("load a preset .json for this session (not added to the dropdown)")
        imgui.same_line()
        if imgui.button("Import...") and self._dialog is None:
            # B4-5: pick a preset .json from anywhere; it is validated and
            # installed into the user preset dir, so it joins the dropdown.
            self._dialog = (
                DialogKind.IMPORT,
                pfd.open_file("Import preset", "", ["JSON", "*.json"]),
            )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "copy a preset .json into your user presets (~/.gasgiant/presets) "
                "and load it"
            )
        imgui.same_line()
        if imgui.button("Save..."):
            self._open_save_dialog()
        imgui.same_line()
        if imgui.button("Reset to gas_giant_warm"):
            self._load_preset_entry("gas_giant_warm", PresetSource.FACTORY)

        # T13: fit the latitude palette rows from a reference photo. The fitted
        # colors are baked into appearance.palette_rows (POST tier, undoable);
        # no image path is kept on the params.
        if imgui.button("Fit palette from reference image...") and self._dialog is None:
            self._dialog = (
                DialogKind.PALETTE_FIT,
                pfd.open_file(
                    "Fit palette from reference image", "",
                    ["Images", "*.png *.jpg *.jpeg *.tif *.tiff", "All files", "*"],
                ),
            )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "fit the per-latitude palette rows from a cylindrical true-color "
                "reference photo (values are baked in; undoable)"
            )

        # B1-8: contextual overwrite/delete row, shown only while a USER
        # preset is active (factory presets are package data; a FILE identity
        # lost its full path) -- both confirm before touching the disk.
        user_entry = self._active_user_preset_path()
        if user_entry is not None:
            name, _path = user_entry
            if imgui.button(f"Save '{name}'"):
                self._request_overwrite_active()
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"overwrite user/{name} with the current settings (Ctrl+S)")
            imgui.same_line()
            if imgui.button("Delete##preset_row"):
                self._request_delete_active()
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"delete user/{name} from disk (asks first)")

        self._draw_seed_header_control()

        if imgui.button("Randomize"):
            self._do_randomize()
        imgui.same_line()
        if imgui.button("Reroll seed"):
            self._do_reroll()
        imgui.same_line()
        self._draw_history_buttons()
        imgui.end_disabled()

        # Confirm modals live OUTSIDE the begin_disabled block: a modal's
        # buttons must stay clickable (they're only staged while no export is
        # running anyway -- every request path is export-gated).
        self._draw_preset_confirm_modals()

        if self._export is None:
            if imgui.button("Export..."):
                imgui.open_popup("Export map set")
            self._draw_export_modal()
            self._draw_export_overwrite_confirm()
            self._draw_last_export_line()
        else:
            prog = self._export.progress
            frac = prog.fraction if prog else 0.0
            imgui.progress_bar(frac, imgui.ImVec2(180.0, 0.0), _export_progress_label(prog))
            imgui.same_line()
            if imgui.button("Cancel"):
                self._cancel_export()
            # B1-5: the hold notice in the SAME pane as the frozen sliders.
            developing = prog is not None and prog.message == "developing"
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.8, 0.3, 1.0))
            imgui.text_wrapped(_export_hold_notice(developing))
            imgui.pop_style_color()

        self._draw_pending_hint()
        imgui.separator()
        draft, any_changed, any_committed = draw_params_panel(self._live, self.panel_state)
        self._process_edit(draft, any_changed, any_committed)

    def _draw_seed_header_control(self) -> None:
        """Seed editor in the controls header, next to Randomize/Reroll (UX
        G5). Goes through the same draft -> _process_edit commit/gesture
        pipeline as every panel field (not a direct self.params mutation), so
        a seed edit here coalesces into undo exactly like dragging any other
        field. Also lockable via the same ``state.locked`` mechanism the
        per-field right-click menu uses (dotted path "seed")."""
        draft = self._live.model_dump()
        imgui.text("seed")
        imgui.same_line()
        imgui.set_next_item_width(110.0)
        changed, value = imgui.input_int("##header_seed", draft["seed"])
        committed = imgui.is_item_deactivated_after_edit()
        if changed:
            draft["seed"] = max(0, min(2**31 - 1, value))
        # Explicit str_id: input_int's +/- stepper wraps the widget in a
        # BeginGroup/EndGroup, and EndGroup's closing ItemAdd registers id=0
        # as the last item -- begin_popup_context_item() with no str_id falls
        # back to that last-item id and hits imgui's IM_ASSERT(id != 0) every
        # frame. An explicit id sidesteps the last-item lookup entirely.
        if imgui.begin_popup_context_item("seed_header_context"):
            locked = "seed" in self.panel_state.locked
            clicked, now_locked = imgui.menu_item("Lock for randomize", "", locked)
            if clicked:
                if now_locked:
                    self.panel_state.locked.add("seed")
                else:
                    self.panel_state.locked.discard("seed")
            imgui.end_popup()
        imgui.same_line()
        if "seed" in self.panel_state.locked:
            imgui.text_disabled("(locked)")
        if changed or committed:
            self._process_edit(draft, changed, committed)

    def _undo(self) -> None:
        """Pop the most recent pre-edit record, push the CURRENT state onto redo,
        then commit the popped params. Goes through _commit + _reset_working_copy
        so a pending working-copy edit can't resurrect. No-op if nothing to undo."""
        if not self._undo_stack:
            return
        params, active_preset, pristine = self._undo_stack.pop()
        # _record reads the CURRENT identity, so capture redo BEFORE overwriting.
        self._redo_stack.append(self._record(self.params))
        self._commit(params)
        self._active_preset = active_preset
        self._pristine = pristine
        self._reset_working_copy()

    def _redo(self) -> None:
        """Mirror of _undo: pop from redo, push current onto undo, commit."""
        if not self._redo_stack:
            return
        params, active_preset, pristine = self._redo_stack.pop()
        self._undo_stack.append(self._record(self.params))
        self._commit(params)
        self._active_preset = active_preset
        self._pristine = pristine
        self._reset_working_copy()

    def _draw_history_buttons(self) -> None:
        """Undo/Redo buttons, greyed out when their stack is empty."""
        imgui.begin_disabled(not self._undo_stack)
        if imgui.button("Undo"):
            self._undo()
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(not self._redo_stack)
        if imgui.button("Redo"):
            self._redo()
        imgui.end_disabled()

    def _draw_export_resolution(self) -> None:
        """Resolution combo in the export modal -- THE editor for export.width
        (B4-4: the auto-panel renders it read-only, so there is exactly one
        live editor and one undo policy). Reads the working copy so a held
        mid-export draft can't visually snap back; a width outside the snap
        list (from a preset) previews as "<N> px (custom)" instead of blank."""
        current = self._live.export.width
        preview = next(
            (lbl for w, lbl in EXPORT_RESOLUTIONS if w == current), f"{current} px (custom)"
        )
        imgui.set_next_item_width(110.0)
        if imgui.begin_combo("##exportres", preview):
            for width, label in EXPORT_RESOLUTIONS:
                if imgui.selectable(label, width == current)[0] and width != current:
                    self._commit_export_field("width", width)
            imgui.end_combo()
        imgui.same_line()
        # Clarify that the export map size is independent of the sim grid (the
        # two were easy to confuse when export.width sat among the sliders).
        imgui.text_disabled(
            f"Export map {current}x{current // 2} px "
            f"· Sim grid {self.params.sim.resolution} px (independent)"
        )

    def _draw_export_modal(self) -> None:
        """Confirm-step modal in front of the folder picker: resolution +
        PNG-compression + an emission indicator + the sim-vs-export clarifier.
        This modal is the ONE live editor for the output settings (B4-4); both
        commit POST-tier through the shared _process_edit pipeline, so they
        are ordinary undoable edits. The final "Export..." opens the SAME
        folder dialog the bare button used to open directly, then closes the
        modal; "Cancel" closes it with no side effect. Nothing fires on the
        first click that merely opened the modal (an explicit confirm step)."""
        center = imgui.get_main_viewport().get_center()
        imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
        if not imgui.begin_popup_modal(
            "Export map set", None, imgui.WindowFlags_.always_auto_resize
        )[0]:
            return
        # Resolution combo + the "map size is independent of the sim grid"
        # clarifier text, reused verbatim from the old inline placement.
        self._draw_export_resolution()
        # PNG compression (0-9). Reads the working copy and commits through
        # the shared pipeline (same rationale as _draw_seed_header_control):
        # POST-tier drags stay live, the release coalesces into one undo entry.
        current = self._live.export.png_compression
        changed, value = imgui.slider_int("PNG compression", current, 0, 9)
        committed = imgui.is_item_deactivated_after_edit()
        if changed or committed:
            draft = self._live.model_dump()
            if changed:
                draft["export"]["png_compression"] = value
            self._process_edit(draft, changed, committed)
        # B1-4: name what will actually be written (the modal never listed its
        # outputs) and where the map set goes next (Blender). The file list
        # subsumes the old bare "Emission: enabled/disabled" indicator --
        # emission.exr appears in it exactly when params.emission.enabled.
        imgui.separator()
        imgui.text("Files written to the chosen folder:")
        for line in _export_file_lines(self.params):
            imgui.bullet_text(line)
        imgui.separator()
        if imgui.button("Export...") and self._dialog is None:
            default_dir = str(self._last_export_dir) if self._last_export_dir else ""
            self._dialog = (
                DialogKind.EXPORT,
                pfd.select_folder("Export map set to folder", default_dir),
            )
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel"):
            imgui.close_current_popup()
        imgui.end_popup()

    def _start_export(self, out: Path) -> None:
        """Kick off the tiled export job into ``out`` (folder already picked
        and, if it contained a map set, overwrite-confirmed)."""
        self._export = ExportJob(export_job(self.sim, out), out)

    def _draw_export_overwrite_confirm(self) -> None:
        """B1-4: confirm modal shown when the picked folder already contains
        map-set files (``self._pending_export``). Overwrite starts the job;
        Cancel drops it with no side effect."""
        if self._pending_export is None:
            return
        title = "Overwrite existing export?"
        if not imgui.is_popup_open(title):
            imgui.open_popup(title)
        center = imgui.get_main_viewport().get_center()
        imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
        if not imgui.begin_popup_modal(title, None, imgui.WindowFlags_.always_auto_resize)[0]:
            return
        out, conflicts = self._pending_export
        imgui.text_wrapped(f"{out} already contains a map set:")
        for name in conflicts:
            imgui.bullet_text(name)
        imgui.text_wrapped("Exporting here will overwrite these files.")
        imgui.separator()
        if imgui.button("Overwrite"):
            self._pending_export = None
            self._start_export(out)
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel##overwrite_export"):
            self._pending_export = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _draw_last_export_line(self) -> None:
        """B1-4: persistent 'last exported to...' line + Open-folder button
        (the success toast vanished after 4 s with no way back to the files).
        Drawn only while no export is in flight."""
        if self._last_export_dir is None:
            return
        if imgui.small_button("Open folder") and not _open_folder(self._last_export_dir):
            self.toasts.error(f"could not open {self._last_export_dir}")
        imgui.same_line()
        imgui.text_disabled(f"last export: {self._last_export_dir}")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"{self._last_export_dir}\nImport mapset.json in Blender "
                "(File > Import > Gas Giant Map Set)"
            )

    def _draw_pending_hint(self) -> None:
        """While a heavy (velocity/restart) edit waits for release, tell the user
        the rebuild is deferred so the absence of a live update isn't confusing.
        Wording is plain-language, worst-tier-first (B1-5)."""
        hint = _pending_hint_text(diff_tiers(self.params, self._live))
        if hint is not None:
            imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.3, 1.0), hint)

    def draw_equirect(self) -> None:
        if self.sim is None:  # init_gl failed; the runner is already exiting
            return
        self.render_perf.begin()
        # Advance the development run a little each frame so the user watches
        # the clouds evolve; the viewport re-derives whenever tracers moved.
        # Playback (_playing/_steps_per_frame/_single_step_requested, driven by
        # _draw_playback) is purely app-level: it only changes how often / with
        # what max_steps this calls Simulation.tick() -- tick() itself is
        # unmodified and stays chunk-invariant. A single-step request ORs into
        # the same one tick() call a "playing" frame would make, so pressing
        # Step while already playing can't cause a double tick this frame.
        step_now = self._playing or self._single_step_requested
        self._single_step_requested = False
        if step_now and self.sim.tick(self._steps_per_frame):
            self.viewport.mark_stale()
        self.viewport.draw(
            self.sim, PREVIEW_WIDTH,
            compare_mode=self._compare_mode,
            snapshot_a=self._snapshot_a,
            flash_show_a=self._flash_show_a,
        )
        # imgui-drawing block for the Equirect window. Everything here uses imgui
        # widgets / the window draw list, so it must run right after the image was
        # blitted and its rect captured. The guard keeps draw_equirect a pure
        # tick/blit body when driven headlessly (no image -> image_rect_min is
        # None -> no imgui calls), preserving the invariant its playback tests
        # rely on. The T5 A/B compare + ROI controls always draw here; the T4
        # storm tool additionally needs a hit-testable (non-compare) equirect.
        if self.viewport.image_rect_min is not None:
            self._draw_compare_controls()
            if self.viewport.hit_testable:
                self.viewport.draw_markers(
                    self.params.storms.cast,
                    drag_index=self._drag_index,
                    drag_lonlat=self._drag_lonlat,
                )
                self._draw_storm_tool_ui()
                self._draw_paint_tool_ui()
                # The two tools share one mode machine; dispatch the mouse layer
                # to whichever is armed (paint takes over place/drag when on).
                if self._storm_tool_mode == "paint":
                    self._handle_paint_tool()
                    self._draw_brush_cursor()
                else:
                    self._handle_storm_tool()
        self.render_perf.end()

    # -- T4: click-to-place / drag-to-move storms --------------------------------

    def _draw_storm_tool_ui(self) -> None:
        """Tool block under the equirect image: a 'Place storm' toggle plus a
        kind combo and radius slider while armed, with an honest caption about
        the restart-tier rebuild on placement/drag-release."""
        exporting = self._export is not None
        # "paint" is a sibling mode in the same state machine (T12); it is not a
        # storm-tool-on state, so the Place-storm toggle keys off place/drag only.
        tool_on = self._storm_tool_mode in ("place", "drag")
        imgui.begin_disabled(exporting)
        if imgui.button("Place storm: ON" if tool_on else "Place storm"):
            self._storm_tool_mode = "none" if tool_on else "place"
            self._drag_index = None
            self._drag_lonlat = None
        imgui.end_disabled()
        if not tool_on:
            return
        imgui.same_line()
        imgui.set_next_item_width(110.0)
        kinds = [k.value for k in CastKind]
        cur = kinds.index(self._storm_tool_kind) if self._storm_tool_kind in kinds else 0
        changed, idx = imgui.combo("kind##placetool", cur, kinds)
        if changed:
            self._storm_tool_kind = kinds[idx]
        imgui.same_line()
        imgui.set_next_item_width(150.0)
        rchanged, rv = imgui.slider_float(
            "radius##placetool", self._storm_tool_radius, 0.01, 0.15
        )
        if rchanged:
            self._storm_tool_radius = rv
        n = len(self.params.storms.cast)
        if self._storm_tool_mode == "drag":
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.8, 0.3, 1.0),
                "marker moves live; the planet re-develops on release (restart-tier)",
            )
        else:
            imgui.text_disabled(
                f"click the map to place a storm ({n}/16); drag a marker to move it "
                "(re-develops on placement/release)"
            )

    def _handle_storm_tool(self) -> None:
        """Per-frame mouse handling for the storm tool. Hit-tests the stored
        image rect manually (imgui.is_item_clicked can't be used after the tool
        widgets), gated by is_window_hovered + rect containment so clicks on the
        panels never place storms. Placement/drag-release both commit exactly
        once through the discrete-action idiom (one restart-tier rebuild)."""
        vp = self.viewport
        if vp is None or vp.image_rect_min is None or vp.image_rect_max is None:
            return
        rmin, rmax = vp.image_rect_min, vp.image_rect_max
        mouse = imgui.get_mouse_pos()
        mx, my = mouse.x, mouse.y
        inside = rmin[0] <= mx <= rmax[0] and rmin[1] <= my <= rmax[1]
        lon, lat = screen_to_lonlat(mx, my, rmin, rmax)
        tool_on = self._storm_tool_mode != "none"

        if not tool_on:
            # Subtle hint when hovering the planet with the tool off (cheap:
            # no adopt-seeded, that's deferred).
            if inside and imgui.is_window_hovered():
                imgui.set_tooltip("add a cast storm here to direct the seeded storms")
            return

        if self._export is not None:
            # Never commit mid-export; abandon any in-flight drag.
            self._drag_index = None
            self._drag_lonlat = None
            if self._storm_tool_mode == "drag":
                self._storm_tool_mode = "place"
            return

        if self._storm_tool_mode == "drag":
            self._drag_lonlat = (lon, lat)  # marker follows the cursor, no commit
            if not imgui.is_mouse_down(imgui.MouseButton_.left):
                idx = self._drag_index
                self._storm_tool_mode = "place"
                self._drag_index = None
                self._drag_lonlat = None
                if idx is not None:
                    self._commit_cast_move(idx, lon, lat)
            return

        # place mode: a click either grabs a nearby marker (drag) or places one.
        if inside and imgui.is_window_hovered() and imgui.is_mouse_clicked(
            imgui.MouseButton_.left
        ):
            idx = self._marker_under_cursor(mx, my, rmin, rmax)
            if idx is not None:
                self._storm_tool_mode = "drag"
                self._drag_index = idx
                self._drag_lonlat = (lon, lat)
            else:
                self._place_storm_at(lon, lat)

    def _marker_under_cursor(
        self, mx: float, my: float, rmin, rmax, threshold: float = 14.0
    ) -> int | None:
        """Cast index whose marker is within ``threshold`` pixels of the cursor,
        or None. Uses the wrap-aware nearest_cast_index to pick a candidate, then
        confirms in pixel space (checking the +-360 wrapped screen positions so a
        dateline marker is grabbable from either edge)."""
        cast = self.params.storms.cast
        if not cast:
            return None
        lon, lat = screen_to_lonlat(mx, my, rmin, rmax)
        idx = nearest_cast_index(lon, lat, cast)
        if idx is None:
            return None
        entry = cast[idx]
        for lon_v in (entry.lon_deg, entry.lon_deg - 360.0, entry.lon_deg + 360.0):
            sx, sy = lonlat_to_screen(lon_v, entry.lat_deg, rmin, rmax)
            if (sx - mx) ** 2 + (sy - my) ** 2 <= threshold * threshold:
                return idx
        return None

    def _place_storm_at(self, lon_deg: float, lat_deg: float) -> None:
        """Append a cast storm at (clamped) lon/lat via the discrete-action
        commit idiom (push_history -> commit -> reset_working_copy). No-op (with
        a toast) when the 16-entry cap is reached or an export is in flight."""
        if self._export is not None:
            return  # defense-in-depth: never commit mid-export
        if len(self.params.storms.cast) >= 16:
            self.toasts.info("cast list is full (16 storms)")
            return
        radius = self._storm_tool_radius
        cap = hero_latitude_cap(radius)
        lat = max(-cap, min(cap, lat_deg))
        lon = max(-180.0, min(180.0, lon_deg))
        self._push_history(self.params)
        new = self.params.model_copy(deep=True)
        new.storms.cast.append(
            StormOverride(
                kind=CastKind(self._storm_tool_kind),
                lat_deg=lat,
                lon_deg=lon,
                radius=radius,
            )
        )
        self._commit(new)
        self._reset_working_copy()  # discrete action wins over pending edit
        self.toasts.info(f"placed {self._storm_tool_kind} at lon {lon:.1f}, lat {lat:.1f}")

    def _commit_cast_move(self, index: int, lon_deg: float, lat_deg: float) -> None:
        """Write a dragged marker's final position into cast[index] via one
        discrete-action commit (one restart-tier rebuild). Lat is clamped to the
        entry's radius-coupled cap; lon to +-180."""
        if self._export is not None:
            return  # defense-in-depth: never commit mid-export
        cast = self.params.storms.cast
        if not (0 <= index < len(cast)):
            return
        radius = cast[index].radius
        cap = hero_latitude_cap(radius)
        lat = max(-cap, min(cap, lat_deg))
        lon = max(-180.0, min(180.0, lon_deg))
        self._push_history(self.params)
        new = self.params.model_copy(deep=True)
        new.storms.cast[index].lat_deg = lat
        new.storms.cast[index].lon_deg = lon
        self._commit(new)
        self._reset_working_copy()  # discrete action wins over pending edit

    # -- T13: palette-from-image ------------------------------------------------

    def _apply_palette_fit(self, path: Path) -> None:
        """Fit palette rows from a reference image and BAKE them into
        ``appearance.palette_rows`` via one undoable POST-tier commit. The
        values are baked in; no image path is stored on the params. Decode +
        model conversion happen here (the app layer) so ``palette.fit`` stays
        layer-clean. A bad image just toasts an error and leaves params intact."""
        if self._export is not None:
            return  # defense-in-depth: never commit mid-export
        from gasgiant.export.writers import decode_image
        from gasgiant.palette.fit import calibrate
        from gasgiant.params.model import palette_rows_from_fit
        try:
            img = decode_image(path, color=True)
        except (OSError, ValueError) as exc:
            self.toasts.error(f"could not read {path.name}: {exc}")
            return
        doc = calibrate(img)
        rows = palette_rows_from_fit(doc["palette_rows"])
        self._push_history(self.params)
        new = self.params.model_copy(deep=True)
        new.appearance.palette_rows = rows
        self._commit(new)
        self._reset_working_copy()  # discrete action wins over pending edit
        self.toasts.info(f"fit {len(rows)} palette rows from {path.name}")

    # -- T12: in-GUI mask brush --------------------------------------------------

    def _ensure_paint_buffer(self) -> np.ndarray:
        """Lazily allocate the CPU paint buffer (a zeroed 2:1 equirect) so nothing
        is spent until the user actually paints."""
        if self._paint_buffer is None:
            self._paint_buffer = brush.new_buffer()
        return self._paint_buffer

    def _adopt_mask_into_buffer(self) -> None:
        """When the paint tool opens, seed the canvas from the currently-loaded
        mask (``params.mask.file``) so painting continues from it rather than a
        blank sheet. Best-effort: a missing/invalid file just leaves the existing
        buffer. Skipped once the user has local strokes (their work wins). The
        adopted image is kept as ``_paint_base`` so an Undo-stroke rebuild replays
        onto it instead of wiping it back to zero."""
        if self._paint_strokes or self._paint_buffer is not None:
            return
        f = self.params.mask.file
        if not f:
            return
        from gasgiant.export.writers import decode_image
        try:
            arr = decode_image(Path(f))
        except (OSError, ValueError) as exc:
            log.warning("could not adopt mask into paint buffer: %s", exc)
            return
        arr = np.ascontiguousarray(arr.astype(np.float32))
        self._paint_base = arr.copy()
        self._paint_buffer = arr.copy()

    def _paint_stamp(self, lon_deg: float, lat_deg: float) -> None:
        """Apply one brush stamp to the buffer and record it in the stroke-replay
        history. Marks the buffer dirty for the throttled upload."""
        buf = self._ensure_paint_buffer()
        s = (
            float(lon_deg), float(lat_deg), float(self._paint_radius_deg),
            float(self._paint_strength), bool(self._paint_erase),
        )
        brush.stamp(buf, *s)
        self._paint_strokes.append(s)
        self._paint_dirty = True

    def _rebuild_paint_buffer(self) -> None:
        """Replay the remaining strokes from the adopted base (or zero) after an
        Undo-stroke, then upload the result immediately (a discrete action, so it
        bypasses the drag throttle). Clears the mask entirely when nothing is left
        to replay and no base was adopted."""
        buf = self._ensure_paint_buffer()
        buf[:] = self._paint_base if self._paint_base is not None else 0.0
        for s in self._paint_strokes:
            brush.stamp(buf, *s)
        if self._paint_strokes or self._paint_base is not None:
            self.sim.set_mask(buf)
        else:
            self.sim.set_mask(None)
        self._paint_dirty = False
        self.viewport.mark_stale()

    def _undo_paint_stroke(self) -> None:
        """Pop the most recent stamp and rebuild the buffer from the remaining
        strokes. Deliberately NOT wired to the params undo stack: mask pixels are
        sidecar data, not a PlanetParams field, so Ctrl+Z must not touch them."""
        if not self._paint_strokes:
            return
        self._paint_strokes.pop()
        self._rebuild_paint_buffer()

    def _clear_paint_mask(self) -> None:
        """Zero the buffer, drop every stroke and any adopted base, and clear the
        live mask (``set_mask(None)``) so the planet returns to its no-mask look."""
        self._paint_strokes.clear()
        self._paint_base = None
        if self._paint_buffer is not None:
            self._paint_buffer[:] = 0.0
        self._paint_dirty = False
        self.sim.set_mask(None)
        self.viewport.mark_stale()

    def _maybe_upload_mask(self, force: bool) -> None:
        """Throttled mask upload + POST re-derive. Accumulates imgui frame time
        and uploads at most every ``PAINT_UPLOAD_INTERVAL`` (~30 Hz) while a
        stroke is in progress, or immediately on ``force`` (the mouse-up that ends
        a stroke) so the final pixels are never left unshown. set_mask marks the
        facade's POST/emission previews dirty; mark_stale makes the viewport's
        next ``ensure_preview`` re-derive."""
        self._paint_upload_accum += imgui.get_io().delta_time
        if not self._paint_dirty:
            return
        if force or self._paint_upload_accum >= PAINT_UPLOAD_INTERVAL:
            self.sim.set_mask(self._ensure_paint_buffer())
            self.viewport.mark_stale()
            self._paint_dirty = False
            self._paint_upload_accum = 0.0

    def _draw_paint_tool_ui(self) -> None:
        """Tool block under the equirect image: a 'Paint mask' toggle plus brush
        controls (radius, strength, erase), Undo-stroke and Clear-mask buttons
        while armed. Paint is inert on the final image until a Mask gain > 0 (all
        default 0); the caption says so."""
        exporting = self._export is not None
        paint_on = self._storm_tool_mode == "paint"
        imgui.begin_disabled(exporting)
        if imgui.button("Paint mask: ON" if paint_on else "Paint mask"):
            if paint_on:
                self._storm_tool_mode = "none"
            else:
                self._storm_tool_mode = "paint"
                self._drag_index = None
                self._drag_lonlat = None
                self._adopt_mask_into_buffer()
        imgui.end_disabled()
        if not paint_on:
            return
        imgui.same_line()
        imgui.set_next_item_width(150.0)
        rchanged, rv = imgui.slider_float(
            "radius (deg)##paint", self._paint_radius_deg, 2.0, 60.0
        )
        if rchanged:
            self._paint_radius_deg = rv
        imgui.same_line()
        imgui.set_next_item_width(120.0)
        schanged, sv = imgui.slider_float(
            "strength##paint", self._paint_strength, 0.0, 1.0
        )
        if schanged:
            self._paint_strength = sv
        imgui.same_line()
        echanged, ev = imgui.checkbox("erase##paint", self._paint_erase)
        if echanged:
            self._paint_erase = ev
        if imgui.button("Undo stroke##paint"):
            self._undo_paint_stroke()
        imgui.same_line()
        if imgui.button("Clear mask##paint"):
            self._clear_paint_mask()
        imgui.same_line()
        n = len(self._paint_strokes)
        gains_on = (
            self.params.mask.band_fade > 0.0
            or self.params.mask.emission_gain > 0.0
            or self.params.mask.detail_gain > 0.0
        )
        note = "" if gains_on else " — set a Mask gain (>0) to see it act"
        imgui.text_disabled(f"drag on the map to paint ({n} strokes){note}")

    def _handle_paint_tool(self) -> None:
        """Per-frame mouse handling for the paint tool. Shares T4's mouse layer
        (the stored image rect + screen_to_lonlat). While the left button is held
        inside the image, each frame stamps the buffer at the cursor; the upload
        is throttled and forced once on release. Held off entirely during an
        export (a POST re-derive mid-export would corrupt the in-flight run)."""
        vp = self.viewport
        if vp is None or vp.image_rect_min is None or vp.image_rect_max is None:
            return
        if self._export is not None:
            self._paint_was_down = False
            return
        rmin, rmax = vp.image_rect_min, vp.image_rect_max
        mouse = imgui.get_mouse_pos()
        mx, my = mouse.x, mouse.y
        inside = rmin[0] <= mx <= rmax[0] and rmin[1] <= my <= rmax[1]
        down = imgui.is_mouse_down(imgui.MouseButton_.left)
        painting = down and inside and imgui.is_window_hovered()
        if painting:
            lon, lat = screen_to_lonlat(mx, my, rmin, rmax)
            self._paint_stamp(lon, lat)
        released = self._paint_was_down and not down
        self._maybe_upload_mask(force=released)
        self._paint_was_down = painting

    def _draw_brush_cursor(self) -> None:
        """Draw the brush-radius circle at the cursor via the window draw list
        (T4's draw-list idiom). Uses the meridional deg->pixel scale so the ring
        reads as the brush's angular size. Only inside the captured image rect."""
        vp = self.viewport
        if vp is None or vp.image_rect_min is None or vp.image_rect_max is None:
            return
        rmin, rmax = vp.image_rect_min, vp.image_rect_max
        mouse = imgui.get_mouse_pos()
        mx, my = mouse.x, mouse.y
        if not (rmin[0] <= mx <= rmax[0] and rmin[1] <= my <= rmax[1]):
            return
        px_per_deg = (rmax[1] - rmin[1]) / 180.0
        r = max(2.0, self._paint_radius_deg * px_per_deg)
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(imgui.ImVec2(*rmin), imgui.ImVec2(*rmax), True)
        col = imgui.get_color_u32(
            imgui.ImVec4(1.0, 0.4, 0.4, 0.95) if self._paint_erase
            else imgui.ImVec4(0.4, 0.9, 1.0, 0.95)
        )
        draw_list.add_circle(imgui.ImVec2(mx, my), r, col, 32, 2.0)
        draw_list.pop_clip_rect()

    # -- T5: A/B compare + ROI export-res inspector ------------------------------

    def _compare_active(self) -> bool:
        """Compare draws only when a mode is selected AND snapshot A is held; a
        mode chosen with no snapshot is an inert no-op (viewport shows live)."""
        return self._compare_mode != "off" and self._snapshot_a is not None

    def _take_snapshot_a(self) -> None:
        """Capture the current color preview into snapshot A, RELEASING any
        previously-held snapshot first (no leak on retake). The app owns the
        returned clone (facade contract); on failure the old snapshot is kept."""
        try:
            new = self.sim.snapshot_preview_color()
        except RuntimeError as exc:  # no preview yet (shouldn't happen post-first-frame)
            self.toasts.error(str(exc))
            return
        if self._snapshot_a is not None:
            self._snapshot_a.release()
        self._snapshot_a = new
        self.toasts.info("captured snapshot A")

    def _draw_compare_controls(self) -> None:
        """A/B compare + ROI toolbar under the viewport: take snapshot A, a mode
        selector (off/flash/split; disabled until A is held), the flash A/live
        toggle, and the ROI inspector opener (gated during export)."""
        if imgui.button("Take snapshot A"):
            self._take_snapshot_a()
        imgui.same_line()
        have_a = self._snapshot_a is not None
        imgui.begin_disabled(not have_a)
        modes = ["off", "flash", "split"]
        cur = modes.index(self._compare_mode) if self._compare_mode in modes else 0
        imgui.set_next_item_width(90.0)
        changed, idx = imgui.combo("compare##ab", cur, modes)
        if changed:
            self._compare_mode = modes[idx]
        if self._compare_mode == "flash":
            imgui.same_line()
            if imgui.button("Show live" if self._flash_show_a else "Show A"):
                self._flash_show_a = not self._flash_show_a
        imgui.end_disabled()
        if not have_a:
            imgui.same_line()
            imgui.text_disabled("(take snapshot A to compare)")
        imgui.same_line()
        exporting = self._export is not None
        imgui.begin_disabled(exporting)
        if imgui.button("Inspect region..."):
            self._show_inspect = True
            self._run_inspect()
        imgui.end_disabled()

    def _run_inspect(self) -> None:
        """Render ONE export-resolution tile centered on the picked region from a
        consistent snapshot and hold it for display. GATED while an export is in
        flight (deriver / detail-synth state is shared -- same guard as every
        discrete action). Releases the snapshot immediately and the prior tile on
        retake; the tile is app-owned (released on close). The inspected tile is
        byte-for-byte the corresponding crop of a full export at the same dims/
        origin (kinematic mode)."""
        if self._export is not None:
            return  # defense-in-depth: never touch the shared deriver mid-export
        from gasgiant.export.exporter import TILE, derive_tile, roi_tile_origin

        w = self.params.export.width
        h = w // 2
        cx, cy = self._inspect_center
        x0, y0 = roi_tile_origin(cx, cy, w, h, TILE)
        snap = self.sim.create_snapshot()
        n_step = self.sim.steps_done
        m_step = snap.params.sim.dev_steps
        tile_color = self.gpu.texture2d((TILE, TILE), 4, "f4")
        tile_height = self.gpu.texture2d((TILE, TILE), 1, "f4")
        tile_detail = self.gpu.texture2d((TILE, TILE), 1, "f4", linear=True)
        ok = False
        try:
            derive_tile(
                self.sim, snap, snap.params, x0, y0, w, h,
                tile_color, tile_height, tile_detail, None,
            )
            ok = True
        finally:
            tile_height.release()
            tile_detail.release()
            snap.release()
            # FBO hygiene: create_snapshot's clone_texture (and derive) leave an
            # offscreen FBO bound; rebind the default framebuffer for imgui.
            self.gpu.ctx.screen.use()
            if not ok:
                tile_color.release()
        if self._inspect_tile is not None:
            self._inspect_tile.release()
        self._inspect_tile = tile_color
        self._inspect_label = (
            f"tile origin ({x0}, {y0}) at {w}x{h} · as of step {n_step} / {m_step}"
        )

    def _release_inspect(self) -> None:
        if self._inspect_tile is not None:
            self._inspect_tile.release()
            self._inspect_tile = None
        self._inspect_label = ""

    def _draw_inspect_window(self) -> None:
        """The ROI inspector window (drawn from gui_overlays, like Help). Picks a
        normalized region center, renders one export-resolution tile, and shows
        it 1:1. Render is gated while an export is in flight."""
        if not self._show_inspect:
            return
        imgui.set_next_window_size(imgui.ImVec2(560.0, 660.0), imgui.Cond_.first_use_ever)
        opened, self._show_inspect = imgui.begin("ROI inspector", self._show_inspect)
        if opened:
            imgui.text_wrapped(
                "Renders one export-resolution tile so you can check fine detail "
                "without a full export. Pick a region center, then Render."
            )
            cx, cy = self._inspect_center
            xchg, cx = imgui.slider_float("center x##roi", cx, 0.0, 1.0)
            ychg, cy = imgui.slider_float("center y##roi", cy, 0.0, 1.0)
            if xchg or ychg:
                self._inspect_center = (cx, cy)
            exporting = self._export is not None
            imgui.begin_disabled(exporting)
            if imgui.button("Render region"):
                self._run_inspect()
            imgui.end_disabled()
            if exporting:
                imgui.same_line()
                imgui.text_disabled("paused during export")
            if self._inspect_tile is not None:
                imgui.separator()
                imgui.text(self._inspect_label)
                avail = imgui.get_content_region_avail()
                side = max(min(avail.x, 512.0), 64.0)
                imgui.image(
                    imgui.ImTextureRef(self._inspect_tile.glo),
                    imgui.ImVec2(side, side),
                )
        imgui.end()
        if not self._show_inspect:  # window just closed via its title-bar X
            self._release_inspect()

    def draw_sphere(self) -> None:
        if self.sim is None:  # init_gl failed; the runner is already exiting
            return
        color_tex, _ = self.sim.ensure_preview(PREVIEW_WIDTH)
        self.sphere.draw(color_tex, self.viewport.agx)

    def _draw_playback(self) -> None:
        """Pause/Play, Step, Restart-dev, and the steps-per-frame speed
        selector for the live dev-run preview (Phase 5). Restart-dev is
        disabled while an export is in flight (M5 / Round 2 LOW-5) -- export
        Phase A ticks the live sim then snapshots it, so releasing/rebuilding
        it mid-export would corrupt the in-flight run."""
        if imgui.button("Pause" if self._playing else "Play"):
            self._playing = not self._playing
        imgui.same_line()
        if imgui.button("Step"):
            # Always available (not just while paused): while playing, this is
            # redundant with the tick already happening this frame -- see the
            # single-call OR in draw_equirect -- but disabling it would need to
            # explain why to the user for no real benefit, so it just stays live.
            self._single_step_requested = True
        imgui.same_line()
        exporting = self._export is not None
        imgui.begin_disabled(exporting)
        if imgui.button("Restart dev"):
            self.sim.rebuild()
            self.viewport.mark_stale()
            self._recomputing = True
        imgui.end_disabled()
        imgui.same_line()
        imgui.set_next_item_width(90.0)
        labels = [label for _, label in SPEED_OPTIONS]
        values = [value for value, _ in SPEED_OPTIONS]
        cur_idx = values.index(self._steps_per_frame) if self._steps_per_frame in values else 1
        clicked, idx = imgui.combo("##speed", cur_idx, labels)
        if clicked:
            self._steps_per_frame = values[idx]
        if exporting:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.8, 0.3, 1.0),
                "export in progress — restart & param edits paused",
            )

    def draw_perf(self) -> None:
        if self.sim is None:  # init_gl failed; the runner is already exiting
            return
        imgui.text(f"frame  {self.frame_perf.mean_ms:6.2f} ms")
        imgui.text(f"render {self.render_perf.last_ms:6.2f} ms (last)")
        imgui.text(f"preview {PREVIEW_WIDTH}x{PREVIEW_WIDTH // 2}")
        self._draw_playback()
        done, target = self.sim.steps_done, self.sim.steps_target
        if done < target:
            now = time.monotonic()
            spinner = "|/-\\"[int(now * 8) % 4]
            eta = self._dev_rate.eta_seconds(now, target - done)
            label = _dev_progress_label(
                done, target, self._playing, self._recomputing, spinner, eta
            )
            imgui.progress_bar(done / max(target, 1), imgui.ImVec2(-1.0, 0.0), label)
        else:
            self._recomputing = False  # dev run caught up; back to the plain state
            imgui.text(f"developed ({done} steps)")

    # -- frame -----------------------------------------------------------------------------

    def _run_export_slice(self) -> None:
        if self._export is None:
            return
        job, out = self._export.job, self._export.out_dir
        try:
            self._export.progress = next(job)
        except StopIteration:
            self._export = None
            # B1-4: remember where (this session + persisted via the same
            # hello_imgui prefs that keep window geometry/docking layout).
            self._last_export_dir = out
            _save_last_export_dir(out)
            self.toasts.info(f"exported to {out} — 'Open folder' in Controls")
        except Exception as exc:  # noqa: BLE001 - surface any export failure
            # Record the full traceback (this catch is broad -- a GL error mid
            # derive, an IndexError in the tiler, an OSError on the last tile),
            # and fall back to the type name when str(exc) is empty (a bare
            # KeyError etc.) so the toast is never "export failed: " (#7).
            log.exception("export failed")
            self._export = None
            detail = str(exc) or type(exc).__name__
            self.toasts.error(f"export failed: {detail}")
        if self._export is None:
            self._flush_pending_edit()  # apply anything the export gate held back

    def _cancel_export(self) -> None:
        if self._export is None:
            return
        self._export.job.close()  # finally-block removes partial output
        self._export = None
        self.toasts.info("export cancelled")
        self._flush_pending_edit()  # apply anything the export gate held back

    def _check_baroclinic_status(self) -> None:
        """A2-2: surface the engine's baroclinic graceful degrade in-window.
        The facade degrade paths only log to file; a user who enabled
        solver.baroclinic (RESTART-tier, long CPU warmup) and hit an outcrop
        would otherwise get a plain-v1.6 render with zero in-window signal.
        Toasts once per transition into 'degraded' (never per frame)."""
        if self.sim is None:
            return
        status = self.sim.baroclinic_status
        if status != self._baro_status_seen:
            if status == "degraded":
                reason = self.sim.baroclinic_degraded_reason or "unknown cause"
                self.toasts.error(f"baroclinic coupling degraded: {reason}")
            self._baro_status_seen = status

    def pre_frame(self) -> None:
        """Runs once per frame before imgui NewFrame: dialogs, pacing, smoke exit."""
        if self.gpu is None and self.init_error is None:  # defensive; post_init normally did this
            self.init_gl()
        self.frame_perf.end()
        self.frame_perf.begin()
        self._poll_dialog()
        self._run_export_slice()
        self._check_baroclinic_status()
        self._frame_count += 1
        if self._smoke_frames and self._frame_count >= self._smoke_frames:
            hello_imgui.get_runner_params().app_shall_exit = True

    def gui_overlays(self) -> None:
        self._draw_dev_overlay()
        self.toasts.draw()
        self.draw_help()
        self._draw_inspect_window()

    def _draw_dev_overlay(self) -> None:
        """B1-1: one line over the viewport while the dev run evolves, so the
        state is visible without opening the Playback pane. Drawn from the
        overlay layer (same idiom as Toasts.draw) rather than inside
        draw_equirect, which stays a pure tick/blit body its headless tests
        can drive without an imgui frame. Also the single per-frame feed point
        for the ETA sampler -- gui_overlays runs after draw_equirect ticked."""
        if self.sim is None:
            return
        now = time.monotonic()
        done, target = self.sim.steps_done, self.sim.steps_target
        if self._playing and done < target:
            self._dev_rate.add(now, done)
        eta = self._dev_rate.eta_seconds(now, target - done)
        text = _dev_overlay_text(done, target, self._playing, eta)
        if text is None:
            return
        vp = imgui.get_main_viewport()
        imgui.set_next_window_pos(
            imgui.ImVec2(vp.pos.x + vp.size.x * 0.5, vp.pos.y + 40.0),
            imgui.Cond_.always,
            imgui.ImVec2(0.5, 0.0),
        )
        imgui.set_next_window_bg_alpha(0.6)
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_focus_on_appearing
            | imgui.WindowFlags_.no_nav
            | imgui.WindowFlags_.no_move
        )
        if imgui.begin("##dev-overlay", None, flags)[0]:
            imgui.text(text)
        imgui.end()

    def draw_help(self) -> None:
        """Phase 7 help window: a plain floating imgui window (the same
        idiom ``Toasts.draw`` already uses in this file), toggled by F1 or
        the "Help (F1)" button in ``draw_controls``, drawn from
        ``gui_overlays`` each frame while ``self._show_help`` is set. Not a
        permanent ``DockableWindow`` tab -- Help is occasional reference
        material, not something that should occupy dock space every
        session."""
        if not self._show_help:
            return
        imgui.set_next_window_size(imgui.ImVec2(520.0, 480.0), imgui.Cond_.first_use_ever)
        opened, self._show_help = imgui.begin("Help", self._show_help)
        if opened:
            # B1-3: the app's central concept, first -- a new user opens Help
            # precisely because the planet is still churning.
            imgui.text("How the simulation develops:")
            imgui.text_wrapped(_HELP_DEV_RUN)
            imgui.separator()
            imgui.text("Preset status ('*' and 'unsaved'):")
            imgui.text_wrapped(_HELP_DIRTY)
            imgui.separator()
            imgui.text_wrapped(
                "Type in the search box to filter fields by name, label, or "
                "description. Toggle Advanced to see the full field set -- a "
                "search always overrides the Advanced filter, so a searched-for "
                "advanced field is still findable with Advanced off."
            )
            imgui.separator()
            imgui.text("Tier badges (change cost, shown left of every field):")
            for _key, (glyph, color, full) in _TIER_GLYPHS.items():
                imgui.text_colored(imgui.ImVec4(*color), glyph)
                imgui.same_line()
                imgui.text(full)
            imgui.separator()
            imgui.bullet_text("Lock (right-click a field): exclude it from Randomize/Reroll.")
            imgui.bullet_text("Randomize / Reroll seed: re-roll unlocked fields from a fresh seed.")
            imgui.bullet_text("Undo / Redo: step back/forward through committed edits.")
            imgui.bullet_text("Export...: render the current look to a map set on disk.")
            imgui.separator()
            imgui.text("What Undo covers:")
            imgui.text_wrapped(_HELP_UNDO)
            imgui.separator()
            imgui.text("Glossary:")
            imgui.text_wrapped(_HELP_GLOSSARY)
            imgui.bullet_text(
                "Ctrl-click any slider to type an exact value (built-in imgui "
                "behavior -- already works everywhere, nothing to enable)."
            )
            imgui.separator()
            imgui.text("Keyboard shortcuts:")
            imgui.bullet_text("/         focus the search box")
            imgui.bullet_text("A         toggle Advanced")
            imgui.bullet_text("R         Randomize")
            imgui.bullet_text("F1        toggle this Help window")
            imgui.bullet_text("Ctrl+Z    Undo")
            imgui.bullet_text("Ctrl+Y    Redo")
            imgui.bullet_text(
                "Ctrl+S    Save preset (overwrites the active user preset "
                "after a confirm; otherwise opens Save As...)"
            )
            imgui.separator()
            imgui.text_wrapped(
                "Hover the (?) marker next to a section header for a one-line "
                "summary of that section. See docs/sliders.md for the full "
                "field reference."
            )
        imgui.end()


def main() -> int:
    configure_logging(log_file=Path.home() / ".gasgiant" / "studio.log")
    app = StudioApp()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Gas Giant Studio"
    params.app_window_params.window_geometry.size = (1700, 980)
    # Remember window position/size across sessions; (1700, 980) above is only
    # the first-run default, used until a prior session's geometry exists.
    params.app_window_params.restore_previous_geometry = True
    gl_options = hello_imgui.OpenGlOptions()
    gl_options.major_version = 4
    gl_options.minor_version = 3
    gl_options.use_core_profile = True
    params.renderer_backend_options.open_gl_options = gl_options
    params.callbacks.post_init = app.init_gl
    params.callbacks.pre_new_frame = app.pre_frame
    # Runs right after ImGui::NewFrame() -- io.want_text_input/is_key_pressed
    # are current for this frame, and it fires before any window is drawn, so
    # a shortcut-set flag (search-focus) is consumed later this same frame.
    params.callbacks.post_new_frame = app._handle_shortcuts
    params.callbacks.show_gui = app.gui_overlays
    params.callbacks.before_exit = app.shutdown
    params.fps_idling.enable_idling = False
    # Window/dock layout persists in the OS temp folder, not the project dir.
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder

    # Menu bar for the hello_imgui "View" menu: it lists every dockable window
    # with a show/hide checkbox (so a closed pane -- e.g. Sphere -- can be
    # reopened in-app) plus a "Restore default layout" entry. show_menu_view is
    # on by default; we only need to turn the bar on. The App menu (Quit) rides
    # along as the conventional first menu.
    params.imgui_window_params.show_menu_bar = True

    # Docked layout: controls left, equirect main, sphere right, perf bottom-left.
    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    split_left = hello_imgui.DockingSplit("MainDockSpace", "LeftSpace", imgui.Dir.left, 0.26)
    split_perf = hello_imgui.DockingSplit("LeftSpace", "PerfSpace", imgui.Dir.down, 0.10)
    split_right = hello_imgui.DockingSplit("MainDockSpace", "SphereSpace", imgui.Dir.right, 0.36)
    windows = [
        hello_imgui.DockableWindow("Controls", "LeftSpace", app.draw_controls),
        # "###Performance" keeps the imgui window ID (and thus saved docking
        # layouts / View-menu state from older sessions) stable while the
        # visible title says what the pane actually hosts: transport controls
        # and the dev-run progress, with perf text as a footnote.
        hello_imgui.DockableWindow("Playback###Performance", "PerfSpace", app.draw_perf),
        hello_imgui.DockableWindow("Equirect", "MainDockSpace", app.draw_equirect),
        hello_imgui.DockableWindow("Sphere", "SphereSpace", app.draw_sphere),
    ]
    params.docking_params = hello_imgui.DockingParams(
        docking_splits=[split_left, split_perf, split_right],
        dockable_windows=windows,
    )

    try:
        hello_imgui.run(params)
    except Exception as exc:  # noqa: BLE001 - window/GL-context creation failed in the runner
        # The runner couldn't even create the 4.3 window/context (init_gl
        # never ran). Same friendly translation as the init_gl path (B1-2).
        log.exception("GUI runner failed")
        print(_gl_failure_message(str(exc) or type(exc).__name__), file=sys.stderr)
        return 1
    if app.init_error is not None:
        print(app.init_error, file=sys.stderr)
        return 1

    shot_path = os.environ.get("GASGIANT_SMOKE_SCREENSHOT", "")
    if shot_path:
        import cv2

        image = hello_imgui.final_app_window_screenshot()
        if image is not None and image.size:
            cv2.imwrite(shot_path, image[:, :, 2::-1])  # RGBA -> BGR
            log.info("screenshot written to %s", shot_path)
        else:
            log.warning("final_app_window_screenshot returned nothing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

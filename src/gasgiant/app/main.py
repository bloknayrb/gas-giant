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
import os
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from imgui_bundle import hello_imgui, imgui
from imgui_bundle import portable_file_dialogs as pfd
from pydantic import ValidationError

from gasgiant.app.panels import _TIER_GLYPHS, PanelState, draw_params_panel
from gasgiant.app.sphere_preview import SpherePreview
from gasgiant.app.viewport import EquirectViewport
from gasgiant.diagnostics import PerfCounter, configure_logging
from gasgiant.engine import Simulation
from gasgiant.engine.invalidation import diff_tiers
from gasgiant.export.exporter import export_job
from gasgiant.gl import GpuContext
from gasgiant.jobs import Progress
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.params.presets import (
    USER_PRESET_DIR,
    PresetError,
    PresetSource,
    available_presets,
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


def _dev_progress_label(
    done: int, target: int, playing: bool, recomputing: bool, spinner: str
) -> str:
    """Label for the development-progress bar (drawn only while ``done < target``).

    ``tick()`` no-ops once developed, so Pause deliberately freezes the dev-run
    animation. When paused we must NOT keep animating the "recomputing..."
    spinner -- a frozen-but-spinning bar reads as a hang (#1). Instead say so
    plainly and point at the way out. Extracted as a pure function so the label
    logic is unit-testable without an imgui frame or the ``time.monotonic``
    spinner clock."""
    if not playing:
        return f"paused {done}/{target} (Play/Step to develop)"
    if recomputing:
        return f"{spinner} recomputing... {done}/{target}"
    return f"{done}/{target}"


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
            self.toasts.info("started from gas_giant_warm")
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
        self._steps_per_frame = 2  # matches the old hardcoded tick(2)
        self._single_step_requested = False  # consumed once, the frame after Step
        # Bounded undo/redo history (Phase 2). Each entry is an UndoRecord — a
        # deep copy of a committed params snapshot plus Phase 6 preset-identity
        # placeholders. maxlen=64 evicts the oldest entry automatically.
        self._undo_stack: deque[UndoRecord] = deque(maxlen=64)
        self._redo_stack: deque[UndoRecord] = deque(maxlen=64)
        self._dialog: tuple[DialogKind, object] | None = None
        self._export: ExportJob | None = None
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

    # -- lifecycle ------------------------------------------------------------

    def init_gl(self) -> None:
        try:
            self.gpu = GpuContext.attach()
            self.sim = Simulation(self.params, self.gpu)
            self.viewport = EquirectViewport(self.gpu)
            self.sphere = SpherePreview(self.gpu)
            log.info("GL ready: %s", self.gpu.ctx.info.get("GL_RENDERER", "?"))
        except Exception:
            log.exception("GL initialization failed")
            raise

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
            save_preset(self.params, SESSION_PATH, name="session")
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

    def _commit_output_setting(self, new_params: PlanetParams) -> None:
        """Commit an export/output-only setting (map resolution, PNG
        compression). These are OUTPUT params, not planet-design edits, so they
        are intentionally kept out of undo history -- but they ARE a fresh user
        action, so they must invalidate any pending redo future (#4). Otherwise a
        Redo issued after changing an export setting would replay the stale
        pre-undo snapshot on top of it. Clearing redo here is exactly what
        ``_push_history`` does for history-backed edits; these two sites are the
        only commits outside the gesture/undo/redo paths."""
        self._commit(new_params)
        self._redo_stack.clear()
        self._reset_working_copy()  # keep _live in lockstep with the applied setting

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

    def _open_save_dialog(self) -> None:
        """Open the preset-save file dialog -- the exact path both the
        "Save..." button and the Ctrl+S shortcut trigger. Ctrl+S always opens
        the dialog, identical to the button; it does not silently re-save to
        the active preset's path. No-op if a dialog is already open (mirrors
        the button's ``self._dialog is None`` guard)."""
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
            self._open_save_dialog()

    # -- dialogs --------------------------------------------------------------------

    def _poll_dialog(self) -> None:
        if self._dialog is None:
            return
        kind, dlg = self._dialog
        if not dlg.ready():
            return
        if kind == DialogKind.LOAD and self._export is not None:
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
            elif kind == DialogKind.SAVE:
                path = Path(result if isinstance(result, str) else result[0])
                if path.suffix != ".json":
                    path = path.with_suffix(".json")
                save_preset(self.params, path)
                # The just-saved preset is the new active/pristine baseline (so
                # dirty resets); refresh so a save into USER_PRESET_DIR appears.
                source = PresetSource.USER if path.parent == USER_PRESET_DIR else PresetSource.FILE
                self._set_identity((path.stem, source), self.params)
                self._refresh_presets()
                self.toasts.info(f"saved {path.name}")
            elif kind == DialogKind.EXPORT:
                out = Path(result)
                self._export = ExportJob(export_job(self.sim, out), out)
        except (PresetError, OSError, ValueError) as exc:
            self.toasts.error(str(exc))

    # -- UI ----------------------------------------------------------------------------

    def draw_controls(self) -> None:
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
        imgui.same_line()
        if imgui.button("Save..."):
            self._open_save_dialog()
        imgui.same_line()
        if imgui.button("Reset to gas_giant_warm"):
            self._load_preset_entry("gas_giant_warm", PresetSource.FACTORY)

        self._draw_seed_header_control()

        if imgui.button("Randomize"):
            self._do_randomize()
        imgui.same_line()
        if imgui.button("Reroll seed"):
            self._do_reroll()
        imgui.same_line()
        self._draw_history_buttons()
        imgui.end_disabled()

        if self._export is None:
            if imgui.button("Export..."):
                imgui.open_popup("Export map set")
            self._draw_export_modal()
        else:
            prog = self._export.progress
            frac = prog.fraction if prog else 0.0
            label = prog.message if prog else "starting"
            imgui.progress_bar(frac, imgui.ImVec2(180.0, 0.0), label)
            imgui.same_line()
            if imgui.button("Cancel"):
                self._cancel_export()

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
        """Resolution combo next to Export, writing export.width (POST tier).
        Kept in sync with the working copy so a pending panel edit can't revert
        it on the next frame."""
        widths = [w for w, _ in EXPORT_RESOLUTIONS]
        labels = [lbl for _, lbl in EXPORT_RESOLUTIONS]
        current = self.params.export.width
        cur_idx = widths.index(current) if current in widths else -1
        imgui.set_next_item_width(70.0)
        clicked, idx = imgui.combo("##exportres", cur_idx, labels)
        if clicked and 0 <= idx < len(widths) and widths[idx] != current:
            new_params = self.params.model_copy(deep=True)
            new_params.export.width = widths[idx]
            self._commit_output_setting(new_params)
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
        Both resolution and compression commit POST-tier (cheap re-derive) the
        same way the old inline combo did. The final "Export..." opens the SAME
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
        # PNG compression (0-9), committed POST-tier like the resolution combo.
        current = self.params.export.png_compression
        changed, value = imgui.slider_int("PNG compression", current, 0, 9)
        if changed and value != current:
            new_params = self.params.model_copy(deep=True)
            new_params.export.png_compression = value
            self._commit_output_setting(new_params)
        if self.params.emission.enabled:
            imgui.text("Emission: enabled (will export emission.exr)")
        else:
            imgui.text("Emission: disabled")
        imgui.separator()
        if imgui.button("Export...") and self._dialog is None:
            self._dialog = (DialogKind.EXPORT, pfd.select_folder("Export map set to folder"))
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel"):
            imgui.close_current_popup()
        imgui.end_popup()

    def _draw_pending_hint(self) -> None:
        """While a heavy (velocity/restart) edit waits for release, tell the user
        the rebuild is deferred so the absence of a live update isn't confusing."""
        heavy_pending = bool(diff_tiers(self.params, self._live) - {Tier.POST})
        if heavy_pending:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.8, 0.3, 1.0), "release to apply (restart/velocity)"
            )

    def draw_equirect(self) -> None:
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
        self.viewport.draw(self.sim, PREVIEW_WIDTH)
        self.render_perf.end()

    def draw_sphere(self) -> None:
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
        imgui.text(f"frame  {self.frame_perf.mean_ms:6.2f} ms")
        imgui.text(f"render {self.render_perf.last_ms:6.2f} ms (last)")
        imgui.text(f"preview {PREVIEW_WIDTH}x{PREVIEW_WIDTH // 2}")
        self._draw_playback()
        done, target = self.sim.steps_done, self.sim.steps_target
        if done < target:
            spinner = "|/-\\"[int(time.monotonic() * 8) % 4]
            label = _dev_progress_label(done, target, self._playing, self._recomputing, spinner)
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
            self.toasts.info(f"exported to {out}")
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
        if self.gpu is None:  # defensive; post_init normally did this
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
        self.toasts.draw()
        self.draw_help()

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
            imgui.bullet_text("Ctrl+S    Save preset...")
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
        hello_imgui.DockableWindow("Performance", "PerfSpace", app.draw_perf),
        hello_imgui.DockableWindow("Equirect", "MainDockSpace", app.draw_equirect),
        hello_imgui.DockableWindow("Sphere", "SphereSpace", app.draw_sphere),
    ]
    params.docking_params = hello_imgui.DockingParams(
        docking_splits=[split_left, split_perf, split_right],
        dockable_windows=windows,
    )

    hello_imgui.run(params)

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

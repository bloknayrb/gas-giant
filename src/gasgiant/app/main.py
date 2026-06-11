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
from pathlib import Path

from imgui_bundle import hello_imgui, imgui
from imgui_bundle import portable_file_dialogs as pfd
from pydantic import ValidationError

from gasgiant.app.panels import draw_params_panel
from gasgiant.app.sphere_preview import SpherePreview
from gasgiant.app.viewport import EquirectViewport
from gasgiant.diagnostics import PerfCounter, configure_logging
from gasgiant.engine import Simulation
from gasgiant.export.manifest import export_mapset
from gasgiant.gl import GpuContext
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import (
    PresetError,
    factory_preset_names,
    load_factory_preset,
    load_preset,
    save_preset,
)
from gasgiant.params.randomize import randomize

log = logging.getLogger(__name__)

PREVIEW_WIDTH = 2048
SESSION_PATH = Path.home() / ".gasgiant" / "session.json"


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
        self.toasts = Toasts()
        self.frame_perf = PerfCounter()
        self.render_perf = PerfCounter()
        self._undo_params: PlanetParams | None = None
        self._dialog: tuple[str, object] | None = None
        self._frame_count = 0
        self._smoke_frames = int(os.environ.get("GASGIANT_SMOKE_FRAMES", "0"))
        self._smoke_shot = os.environ.get("GASGIANT_SMOKE_SCREENSHOT", "")

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
        if SESSION_PATH.is_file():
            try:
                return load_preset(SESSION_PATH)
            except (PresetError, OSError) as exc:
                log.warning("session restore failed (%s); using default preset", exc)
        return load_factory_preset("jupiter_like")

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

    def _try_commit_draft(self, draft: dict) -> None:
        try:
            new_params = PlanetParams.model_validate(draft)
        except ValidationError as exc:
            self.toasts.error(f"invalid value: {exc.errors()[0]['msg']}")
            return
        self._commit(new_params)

    # -- dialogs --------------------------------------------------------------------

    def _poll_dialog(self) -> None:
        if self._dialog is None:
            return
        kind, dlg = self._dialog
        if not dlg.ready():
            return
        self._dialog = None
        result = dlg.result()
        if not result:
            return
        try:
            if kind == "load":
                self._undo_params = self.params
                self._commit(load_preset(Path(result[0])))
                self.toasts.info(f"loaded {Path(result[0]).name}")
            elif kind == "save":
                path = Path(result if isinstance(result, str) else result[0])
                if path.suffix != ".json":
                    path = path.with_suffix(".json")
                save_preset(self.params, path)
                self.toasts.info(f"saved {path.name}")
            elif kind == "export":
                out = Path(result)
                started = time.perf_counter()
                export_mapset(self.sim, out)
                self.toasts.info(f"exported to {out} in {time.perf_counter() - started:.1f}s")
        except (PresetError, OSError, ValueError) as exc:
            self.toasts.error(str(exc))

    # -- UI ----------------------------------------------------------------------------

    def draw_controls(self) -> None:
        names = factory_preset_names()
        imgui.set_next_item_width(160.0)
        clicked, idx = imgui.combo("##preset", -1, ["preset..."] + names)
        if clicked and idx > 0:
            try:
                self._undo_params = self.params
                self._commit(load_factory_preset(names[idx - 1]))
                self.toasts.info(f"preset: {names[idx - 1]}")
            except PresetError as exc:
                self.toasts.error(str(exc))
        imgui.same_line()
        if imgui.button("Load...") and self._dialog is None:
            self._dialog = ("load", pfd.open_file("Load preset", "", ["JSON", "*.json"]))
        imgui.same_line()
        if imgui.button("Save...") and self._dialog is None:
            self._dialog = ("save", pfd.save_file("Save preset", "preset.json", ["JSON", "*.json"]))

        if imgui.button("Randomize"):
            self._undo_params = self.params
            seed = int(time.time_ns() % (2**31 - 1))
            self._commit(randomize(seed, base=self.params))
            self.toasts.info(f"randomized (seed {seed})")
        imgui.same_line()
        if imgui.button("Reroll seed"):
            self._undo_params = self.params
            self._commit(randomize(self.params.seed + 1, base=self.params))
        imgui.same_line()
        if self._undo_params is not None and imgui.button("Undo"):
            self._commit(self._undo_params)
            self._undo_params = None

        if imgui.button("Export...") and self._dialog is None:
            self._dialog = ("export", pfd.select_folder("Export map set to folder"))
        imgui.same_line()
        imgui.text_disabled(f"{self.params.export.width}x{self.params.export.width // 2}")

        imgui.separator()
        draft = draw_params_panel(self.params)
        if draft is not None:
            self._try_commit_draft(draft)

    def draw_equirect(self) -> None:
        self.render_perf.begin()
        # Advance the development run a little each frame so the user watches
        # the clouds evolve; the viewport re-derives whenever tracers moved.
        if self.sim.tick(2):
            self.viewport.mark_stale()
        self.viewport.draw(self.sim, PREVIEW_WIDTH)
        self.render_perf.end()

    def draw_sphere(self) -> None:
        color_tex, _ = self.sim.ensure_preview(PREVIEW_WIDTH)
        self.sphere.draw(color_tex, self.viewport.agx)

    def draw_perf(self) -> None:
        imgui.text(f"frame  {self.frame_perf.mean_ms:6.2f} ms")
        imgui.text(f"render {self.render_perf.last_ms:6.2f} ms (last)")
        imgui.text(f"preview {PREVIEW_WIDTH}x{PREVIEW_WIDTH // 2}")
        done, target = self.sim.steps_done, self.sim.steps_target
        if done < target:
            imgui.progress_bar(done / max(target, 1), imgui.ImVec2(-1.0, 0.0), f"{done}/{target}")
        else:
            imgui.text(f"developed ({done} steps)")

    # -- frame -----------------------------------------------------------------------------

    def pre_frame(self) -> None:
        """Runs once per frame before imgui NewFrame: dialogs, pacing, smoke exit."""
        if self.gpu is None:  # defensive; post_init normally did this
            self.init_gl()
        self.frame_perf.end()
        self.frame_perf.begin()
        self._poll_dialog()
        self._frame_count += 1
        if self._smoke_frames and self._frame_count >= self._smoke_frames:
            hello_imgui.get_runner_params().app_shall_exit = True

    def gui_overlays(self) -> None:
        self.toasts.draw()


def main() -> int:
    configure_logging(log_file=Path.home() / ".gasgiant" / "studio.log")
    app = StudioApp()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Gas Giant Studio"
    params.app_window_params.window_geometry.size = (1700, 980)
    gl_options = hello_imgui.OpenGlOptions()
    gl_options.major_version = 4
    gl_options.minor_version = 3
    gl_options.use_core_profile = True
    params.renderer_backend_options.open_gl_options = gl_options
    params.callbacks.post_init = app.init_gl
    params.callbacks.pre_new_frame = app.pre_frame
    params.callbacks.show_gui = app.gui_overlays
    params.callbacks.before_exit = app.shutdown
    params.fps_idling.enable_idling = False
    # Window/dock layout persists in the OS temp folder, not the project dir.
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder

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

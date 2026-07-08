"""GUI preset thumbnails: a tick-sliced, cross-session-cached preview per preset.

A naive "render the preset live when its combo entry is shown" would run a full
development run on the single GL thread -- seconds of frozen UI per preset. This
module instead runs ONE in-flight thumbnail ``Simulation`` at a time, advanced a
small ``tick`` slice PER GUI FRAME (the same ``_recomputing`` idiom the app uses
for its own dev run), so the UI never blocks. When a thumbnail finishes it is
encoded to an 8-bit PNG under ``~/.gasgiant/thumb_cache/`` keyed by a hash of the
preset's canonical JSON, then uploaded as a GL texture for the combo to blit.

Cache: the key is the sha256 of the preset's canonical params JSON. A preset
whose JSON changed misses the cache and re-renders; an unchanged preset reuses
the on-disk PNG across sessions. The key/path helpers are pure (no GL) and
unit-tested directly; the ``ThumbnailManager`` state machine owns the GL work.

Opt-in tooling on top of the facade: no default/shader/params changes, so no
byte-identity risk. Lives in the ``app`` layer (may import ``engine``/``export``);
deliberately imgui-free so its logic is testable without a GUI context.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gasgiant.export.writers import decode_image, write_png8_rgb
from gasgiant.params.model import PlanetParams

if TYPE_CHECKING:
    import moderngl

    from gasgiant.gl import GpuContext

log = logging.getLogger(__name__)

THUMB_CACHE_DIR = Path.home() / ".gasgiant" / "thumb_cache"
# Small, cheap render: 512px cell, a clamped dev budget, advanced 8 steps/frame.
THUMB_RES = 512
THUMB_DEV_STEPS = 64
THUMB_TICK = 8


def thumb_cache_key(params: PlanetParams) -> str:
    """Stable content hash of a preset: sha256 of its canonical params JSON.

    Canonical = the model's JSON round-tripped through ``json.dumps`` with sorted
    keys and no whitespace, so equal params always hash equal (and a changed leaf
    -- any field, seed included -- changes the key -> cache miss -> re-render).
    Pure: no GL, no filesystem; unit-tested for stability and leaf-sensitivity."""
    canonical = json.dumps(
        json.loads(params.to_json()), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def thumb_cache_path(params: PlanetParams, cache_dir: Path = THUMB_CACHE_DIR) -> Path:
    """On-disk PNG path for ``params``' thumbnail (``<cache_dir>/<key>.png``)."""
    return cache_dir / f"{thumb_cache_key(params)}.png"


def is_cached(params: PlanetParams, cache_dir: Path = THUMB_CACHE_DIR) -> bool:
    """True when a rendered thumbnail already exists on disk for ``params``."""
    return thumb_cache_path(params, cache_dir).is_file()


class ThumbnailManager:
    """Owns the ONE in-flight thumbnail sim, the render queue, and the uploaded
    GL textures. Driven by ``advance()`` once per GUI frame (never blocks) and
    queried by ``get_texture()`` while drawing the preset combo.

    Lifecycle per preset: ``request(params)`` enqueues it (or immediately loads a
    cached PNG); ``advance()`` builds a sim for the head of the queue, ticks it a
    slice per frame, then on completion renders at ``res``, caches the PNG, and
    uploads a texture. ``get_texture(params)`` returns the GL texture once ready,
    else ``None`` (the combo shows a placeholder)."""

    def __init__(
        self,
        gpu: GpuContext,
        *,
        cache_dir: Path = THUMB_CACHE_DIR,
        res: int = THUMB_RES,
        dev_steps: int = THUMB_DEV_STEPS,
        tick: int = THUMB_TICK,
    ) -> None:
        self.gpu = gpu
        self.cache_dir = cache_dir
        self.res = res
        self.dev_steps = dev_steps
        self.tick = tick
        self._queue: list[tuple[str, PlanetParams]] = []
        self._queued: set[str] = set()
        self._textures: dict[str, moderngl.Texture] = {}
        self._failed: set[str] = set()  # keys that errored; don't retry this session
        self._sim: Any = None
        self._cur_key: str | None = None

    # -- public API used by the combo ------------------------------------------

    def request(self, params: PlanetParams) -> None:
        """Ensure a thumbnail for ``params`` is ready, queued, or being made.

        A no-op if it is already uploaded, in flight, queued, or known-failed.
        A cached-on-disk thumbnail is decoded + uploaded immediately (cheap);
        otherwise the preset joins the render queue for ``advance()``."""
        try:
            key = thumb_cache_key(params)
        except Exception:  # noqa: BLE001 - a bad preset must never crash the combo
            log.exception("thumbnail key failed")
            return
        if (
            key in self._textures
            or key in self._queued
            or key in self._failed
            or key == self._cur_key
        ):
            return
        path = self.cache_dir / f"{key}.png"
        if path.is_file():
            self._upload_from_disk(key, path)
            return
        self._queue.append((key, params.model_copy(deep=True)))
        self._queued.add(key)

    def get_texture(self, params: PlanetParams) -> moderngl.Texture | None:
        """The uploaded GL texture for ``params``' thumbnail, or None if not
        ready yet (queued / rendering / failed). Never triggers work."""
        try:
            return self._textures.get(thumb_cache_key(params))
        except Exception:  # noqa: BLE001
            return None

    # -- per-frame driver ------------------------------------------------------

    def advance(self) -> None:
        """Do at most one slice of thumbnail work this frame; never blocks on a
        full render. Builds the next sim, OR ticks the in-flight sim a slice, OR
        (once developed) renders + caches + uploads and moves to the next preset.

        The development run -- the expensive part -- is spread across frames a
        ``tick`` slice at a time. Only the final 512px derive+readback happens in
        one frame, and that is milliseconds, not the multi-second dev run."""
        from gasgiant.engine import Simulation

        if self._sim is None:
            if not self._queue:
                return
            key, params = self._queue.pop(0)
            self._queued.discard(key)
            self._cur_key = key
            try:
                tp = params.model_copy(deep=True)
                tp.sim.dev_steps = min(tp.sim.dev_steps, self.dev_steps)
                self._sim = Simulation(tp, self.gpu)
            except Exception:  # noqa: BLE001 - one bad preset shouldn't kill the queue
                log.exception("thumbnail sim build failed for %s", key)
                self._fail_current()
            return

        try:
            if not self._sim.is_developed:
                self._sim.tick(self.tick)
                return
            color = self._sim.render_maps(self.res)["color"][..., :3]
            self._finish(self._cur_key, np.ascontiguousarray(color, dtype=np.float32))
        except Exception:  # noqa: BLE001
            log.exception("thumbnail render failed for %s", self._cur_key)
            self._fail_current()
        finally:
            # A derive/readback can leave an offscreen FBO bound; the imgui
            # backend renders into whatever is current, so restore the default.
            self.gpu.ctx.screen.use()

    # -- internals -------------------------------------------------------------

    def _finish(self, key: str, rgb01: np.ndarray) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_png8_rgb(self.cache_dir / f"{key}.png", rgb01)
        except OSError:
            log.exception("thumbnail cache write failed for %s", key)  # upload anyway
        self._upload(key, rgb01)
        self._release_sim()

    def _upload(self, key: str, rgb01: np.ndarray) -> None:
        h, w = rgb01.shape[:2]
        arr = np.ascontiguousarray(np.clip(rgb01, 0.0, 1.0), dtype=np.float32)
        old = self._textures.pop(key, None)
        if old is not None:
            old.release()
        self._textures[key] = self.gpu.texture2d((w, h), 3, "f4", data=arr, linear=True)

    def _upload_from_disk(self, key: str, path: Path) -> None:
        try:
            rgb = decode_image(path, color=True)  # (H, W, 3) float32 0..1
        except (OSError, ValueError):
            log.warning("thumbnail cache read failed for %s; re-rendering", key)
            path.unlink(missing_ok=True)  # stale/corrupt -> queue a fresh render
            return
        self._upload(key, rgb)

    def _fail_current(self) -> None:
        if self._cur_key is not None:
            self._failed.add(self._cur_key)
        self._release_sim()

    def _release_sim(self) -> None:
        if self._sim is not None:
            self._sim.release()
            self._sim = None
        self._cur_key = None

    def release(self) -> None:
        """Release every GL resource (in-flight sim + all uploaded textures).
        Idempotent; call on app shutdown."""
        self._release_sim()
        for tex in self._textures.values():
            tex.release()
        self._textures.clear()
        self._queue.clear()
        self._queued.clear()

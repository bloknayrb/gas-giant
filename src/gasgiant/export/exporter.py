"""The tiled export job.

A generator that renders the full-resolution map set tile by tile from an
immutable snapshot, yielding Progress after each slice so the GUI keeps its
frame loop (the CLI just drains it). Detail synthesis and map derivation read
ONLY sim-resolution snapshot textures plus analytic noise, so tiles need no
apron and can never disagree at their borders.

Encoding runs in a small thread pool (PNG deflate of a 16K map is seconds of
pure CPU); worker exceptions are re-raised, and cancellation (generator
close) removes partial output files after the pool drains.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gasgiant.export.manifest import MANIFEST_FILENAME, build_manifest, write_manifest
from gasgiant.export.writers import write_exr_gray, write_png16_rgb_u16
from gasgiant.jobs import Progress
from gasgiant.params.presets import to_preset_doc

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

TILE = 1024


def export_job(sim: Any, out_dir: Path, width: int | None = None) -> Iterator[Progress]:
    """sim: engine.Simulation (duck-typed; export sits below engine in the
    layer order, so the engine object arrives as a parameter, never an import)."""
    # Phase A: finish the development run (visible progress in the GUI).
    while sim.tick(8):
        yield Progress(sim.steps_done, sim.steps_target, "developing")

    snap = sim.create_snapshot()
    params = snap.params
    w = width or params.export.width
    h = w // 2
    gpu = sim.gpu

    tiles = [
        (x, y)
        for y in range(0, h, TILE)
        for x in range(0, w, TILE)
    ]
    total = len(tiles) + 2  # + encode + manifest

    color_full = np.empty((h, w, 3), dtype=np.uint16)
    height_full = np.empty((h, w), dtype=np.float32)

    tile_color = gpu.texture2d((TILE, TILE), 4, "f4")
    tile_height = gpu.texture2d((TILE, TILE), 1, "f4")
    tile_detail = gpu.texture2d((TILE, TILE), 1, "f4", linear=True)

    pool = ThreadPoolExecutor(max_workers=2)
    futures: list[Future] = []
    completed = False
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        started = time.perf_counter()
        use_detail = params.detail.intensity > 0.0
        for i, (x0, y0) in enumerate(tiles):
            tw = min(TILE, w - x0)
            th = min(TILE, h - y0)
            if use_detail:
                sim.detail_synth.synthesize(
                    params.seed, snap.vel_eq, snap.tracers_eq, snap.profile_dyn,
                    tile_detail, params.detail, origin=(x0, y0), full_size=(w, h),
                    heroes=snap.heroes,
                )
            sim.deriver.derive(
                snap.tracers_eq, snap.tracers_n, snap.tracers_s,
                snap.patch_rho_max, snap.blend_band,
                tile_color, tile_height, params.appearance,
                detail_tex=tile_detail if use_detail else None,
                detail_intensity=params.detail.intensity,
                origin=(x0, y0), full_size=(w, h),
                lanes=snap.lanes, warp=snap.warp,
            )
            color = gpu.read_texture(tile_color)[:th, :tw, :3]
            height = gpu.read_texture(tile_height)[:th, :tw, 0]
            color_full[y0 : y0 + th, x0 : x0 + tw] = (
                np.clip(color, 0.0, 1.0) * 65535.0 + 0.5
            ).astype(np.uint16)
            height_full[y0 : y0 + th, x0 : x0 + tw] = height
            yield Progress(i + 1, total, f"tile {i + 1}/{len(tiles)}")

        # Encode off-thread; keep yielding so the GUI stays live.
        futures.append(
            pool.submit(
                write_png16_rgb_u16, out_dir / "color.png", color_full,
                params.export.png_compression,
            )
        )
        futures.append(pool.submit(write_exr_gray, out_dir / "height.exr", height_full))
        while not all(f.done() for f in futures):
            yield Progress(len(tiles) + 1, total, "encoding")
            time.sleep(0.01)
        for f in futures:
            f.result()  # surface worker exceptions

        manifest = build_manifest(
            name=params.name,
            seed=params.seed,
            resolution=(w, h),
            maps={
                "color": {
                    "file": "color.png", "format": "png16",
                    "colorspace": "srgb", "channels": 3,
                },
                "height": {
                    "file": "height.exr", "format": "exr32f",
                    "colorspace": "non-color", "channels": 1,
                },
            },
            physical={
                "radius_km": params.physical.radius_km,
                "height_scale": params.physical.height_scale,
                "height_midlevel": params.physical.height_midlevel,
            },
            preset_doc=to_preset_doc(params),
            atmosphere_hint={"rim_color": [0.55, 0.65, 1.0], "rim_strength": 0.4},
        )
        write_manifest(out_dir, manifest)
        completed = True
        log.info("exported %dx%d map set to %s in %.1fs", w, h, out_dir,
                 time.perf_counter() - started)
        yield Progress(total, total, "done")
    finally:
        pool.shutdown(wait=True)
        tile_color.release()
        tile_height.release()
        tile_detail.release()
        snap.release()
        if not completed:
            # Cancellation: remove only the files WE write (the user may have
            # picked a folder containing their own data), after the pool
            # drained so there are no Windows open-handle races.
            for name in ("color.png", "height.exr", MANIFEST_FILENAME):
                (out_dir / name).unlink(missing_ok=True)
            log.info("export cancelled; partial output removed")


def run_export(sim: Any, out_dir: Path, width: int | None = None) -> None:
    """Drain the job synchronously (CLI / tests)."""
    for _ in export_job(sim, out_dir, width):
        pass

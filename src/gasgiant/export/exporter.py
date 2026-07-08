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

import contextlib
import logging
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gasgiant.export.manifest import (
    MANIFEST_FILENAME,
    attach_frames,
    build_manifest,
    read_manifest,
    write_manifest,
)
from gasgiant.export.writers import write_exr_gray, write_exr_rgba, write_png16_rgb_u16
from gasgiant.jobs import Progress
from gasgiant.params.presets import to_preset_doc
from gasgiant.render.detail import PolarRoute

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

TILE = 1024


def _derive_tile(
    sim: Any,
    snap: Any,
    params: Any,
    x0: int,
    y0: int,
    w: int,
    h: int,
    tile_color: Any,
    tile_height: Any,
    tile_detail: Any,
    tile_emission: Any,
) -> None:
    """Synthesize detail + derive color/height(/emission) into the tile
    textures for the TILE-sized tile at (x0, y0) of a (w, h) map, reading only
    the immutable snapshot. Shared by the mapset export and the sequence
    per-frame color render; ``tile_emission=None`` selects the non-EMISSION
    derive variant."""
    use_detail = params.detail.intensity > 0.0
    if use_detail:
        sim.detail_synth.synthesize(
            params.seed, snap.vel_eq, snap.tracers_eq, snap.profile_dyn,
            tile_detail, params.detail, origin=(x0, y0), full_size=(w, h),
            heroes=snap.heroes,
            polar=PolarRoute(
                snap.vel_n, snap.vel_s, snap.tracers_n, snap.tracers_s,
                snap.patch_rho_max,
            ),
            clouds=snap.clouds,
            profile_stamp=snap.profile_stamp,
        )
    sim.deriver.derive(
        snap.tracers_eq, snap.tracers_n, snap.tracers_s,
        snap.patch_rho_max, snap.blend_band,
        tile_color, tile_height, params.appearance,
        detail_tex=tile_detail if use_detail else None,
        detail_intensity=params.detail.intensity,
        origin=(x0, y0), full_size=(w, h),
        lanes=snap.lanes, warp=snap.warp,
        emission_out=tile_emission,
        emission=params.emission if tile_emission is not None else None,
        seed=params.seed,
        profile_dyn=snap.profile_dyn,
        profile_stamp=snap.profile_stamp,
    )


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

    emission_on = params.emission.enabled
    color_full = np.empty((h, w, 3), dtype=np.uint16)
    height_full = np.empty((h, w), dtype=np.float32)
    emission_full = np.empty((h, w, 4), dtype=np.float32) if emission_on else None

    tile_color = gpu.texture2d((TILE, TILE), 4, "f4")
    tile_height = gpu.texture2d((TILE, TILE), 1, "f4")
    tile_detail = gpu.texture2d((TILE, TILE), 1, "f4", linear=True)
    tile_emission = gpu.texture2d((TILE, TILE), 4, "f4") if emission_on else None

    pool = ThreadPoolExecutor(max_workers=3)
    futures: list[Future] = []
    completed = False
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        started = time.perf_counter()
        for i, (x0, y0) in enumerate(tiles):
            tw = min(TILE, w - x0)
            th = min(TILE, h - y0)
            _derive_tile(
                sim, snap, params, x0, y0, w, h,
                tile_color, tile_height, tile_detail, tile_emission,
            )
            color = gpu.read_texture(tile_color)[:th, :tw, :3]
            height = gpu.read_texture(tile_height)[:th, :tw, 0]
            color_full[y0 : y0 + th, x0 : x0 + tw] = (
                np.clip(color, 0.0, 1.0) * 65535.0 + 0.5
            ).astype(np.uint16)
            height_full[y0 : y0 + th, x0 : x0 + tw] = height
            if emission_on:
                emission_full[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(
                    tile_emission
                )[:th, :tw]
            yield Progress(i + 1, total, f"tile {i + 1}/{len(tiles)}")

        # Encode off-thread; keep yielding so the GUI stays live.
        futures.append(
            pool.submit(
                write_png16_rgb_u16, out_dir / "color.png", color_full,
                params.export.png_compression,
            )
        )
        futures.append(pool.submit(write_exr_gray, out_dir / "height.exr", height_full))
        if emission_on:
            futures.append(
                pool.submit(write_exr_rgba, out_dir / "emission.exr", emission_full)
            )
        while not all(f.done() for f in futures):
            yield Progress(len(tiles) + 1, total, "encoding")
            time.sleep(0.01)
        for f in futures:
            f.result()  # surface worker exceptions

        maps = {
            "color": {
                "file": "color.png", "format": "png16",
                "colorspace": "srgb", "channels": 3,
            },
            "height": {
                "file": "height.exr", "format": "exr32f",
                "colorspace": "non-color", "channels": 1,
            },
        }
        if emission_on:
            # RGB = thermal+lightning radiance; A = aurora intensity, hue
            # applied at import (aurora_color travels in the manifest so the
            # importer can tint it / lift it onto a shell).
            maps["emission"] = {
                "file": "emission.exr", "format": "exr32f",
                "colorspace": "non-color", "channels": 4,
                "aurora_color": list(params.emission.aurora_color),
            }
        manifest = build_manifest(
            name=params.name,
            seed=params.seed,
            resolution=(w, h),
            maps=maps,
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
        if tile_emission is not None:
            tile_emission.release()
        snap.release()
        if not completed:
            # Cancellation: remove only the files WE write (the user may have
            # picked a folder containing their own data), after the pool
            # drained so there are no Windows open-handle races.
            for name in ("color.png", "height.exr", "emission.exr", MANIFEST_FILENAME):
                (out_dir / name).unlink(missing_ok=True)
            log.info("export cancelled; partial output removed")


def run_export(sim: Any, out_dir: Path, width: int | None = None) -> None:
    """Drain the job synchronously (CLI / tests)."""
    for _ in export_job(sim, out_dir, width):
        pass


def export_sequence_job(
    sim: Any, out_dir: Path, frames: int, steps_per_frame: int,
    width: int | None = None,
) -> Iterator[Progress]:
    """Animated sequence export.

    Frame 0 is the full existing mapset export (its color map duplicated as
    ``frames/frame_0000.png``); each subsequent frame advances the sim by
    ``steps_per_frame`` via ``Simulation.extend_run`` and renders COLOR ONLY
    through the same per-tile path as the mapset export. The manifest gains an
    optional ``frames`` block, written only once every frame file exists — a
    cancelled/failed sequence removes everything it wrote, so no half-written
    frame is ever counted in a manifest.

    Determinism note: the kinematic path is byte-exact across runs; vorticity
    frames carry compounding SOR LSB noise (structural guarantees only).
    """
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    if steps_per_frame < 1:
        raise ValueError(f"steps_per_frame must be >= 1, got {steps_per_frame}")

    frames_dir = out_dir / "frames"
    written: list[Path] = []
    tile_texs: list[Any] = []
    completed = False
    total = frames + 1  # frames + manifest rewrite
    try:
        # Frame 0: the full mapset export (writes color/height/manifest and
        # cleans up after ITSELF if cancelled inside this phase).
        yield from export_job(sim, out_dir, width)

        params = sim.params
        w = width or params.export.width
        h = w // 2
        gpu = sim.gpu

        frames_dir.mkdir(parents=True, exist_ok=True)
        frame0 = frames_dir / "frame_0000.png"
        written.append(frame0)
        shutil.copyfile(out_dir / "color.png", frame0)
        yield Progress(1, total, "frame 0")

        tile_color = gpu.texture2d((TILE, TILE), 4, "f4")
        tile_height = gpu.texture2d((TILE, TILE), 1, "f4")
        tile_detail = gpu.texture2d((TILE, TILE), 1, "f4", linear=True)
        tile_texs += [tile_color, tile_height, tile_detail]
        tiles = [(x, y) for y in range(0, h, TILE) for x in range(0, w, TILE)]
        color_full = np.empty((h, w, 3), dtype=np.uint16)

        for fi in range(1, frames):
            sim.extend_run(steps_per_frame)
            snap = sim.create_snapshot()
            try:
                for x0, y0 in tiles:
                    tw = min(TILE, w - x0)
                    th = min(TILE, h - y0)
                    _derive_tile(
                        sim, snap, snap.params, x0, y0, w, h,
                        tile_color, tile_height, tile_detail, None,
                    )
                    color = gpu.read_texture(tile_color)[:th, :tw, :3]
                    color_full[y0 : y0 + th, x0 : x0 + tw] = (
                        np.clip(color, 0.0, 1.0) * 65535.0 + 0.5
                    ).astype(np.uint16)
            finally:
                snap.release()
            path = frames_dir / f"frame_{fi:04d}.png"
            written.append(path)  # tracked BEFORE the write so a partial file
            write_png16_rgb_u16(path, color_full, params.export.png_compression)
            yield Progress(fi + 1, total, f"frame {fi}")

        manifest = read_manifest(out_dir)
        attach_frames(
            manifest, count=frames, steps_per_frame=steps_per_frame,
            files=[f"frames/frame_{i:04d}.png" for i in range(frames)],
        )
        write_manifest(out_dir, manifest)
        completed = True
        yield Progress(total, total, "done")
    finally:
        for tex in tile_texs:
            tex.release()
        if not completed:
            # Remove only files WE wrote: the frame PNGs plus the base map set
            # (export_job's own cancellation already covers the frame-0 phase;
            # these unlinks are no-ops then). The user's files are untouched.
            for p in written:
                p.unlink(missing_ok=True)
            if frames_dir.is_dir():
                with contextlib.suppress(OSError):  # user data in frames/: leave it
                    frames_dir.rmdir()
            for name in ("color.png", "height.exr", "emission.exr", MANIFEST_FILENAME):
                (out_dir / name).unlink(missing_ok=True)
            log.info("sequence export cancelled; partial output removed")


def run_export_sequence(
    sim: Any, out_dir: Path, frames: int, steps_per_frame: int,
    width: int | None = None,
) -> None:
    """Drain the sequence job synchronously (CLI / tests)."""
    for _ in export_sequence_job(sim, out_dir, frames, steps_per_frame, width):
        pass

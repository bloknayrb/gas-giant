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
    CUBE_FACE_NAMES,
    MANIFEST_FILENAME,
    PROJECTION_CUBE,
    attach_frames,
    build_manifest,
    read_manifest,
    write_manifest,
)
from gasgiant.export.rings import ring_strip
from gasgiant.export.video import encode_video_job
from gasgiant.export.writers import (
    read_exr_gray,
    write_exr_gray,
    write_exr_rgba,
    write_png16_gray,
    write_png16_rgb_u16,
)
from gasgiant.jobs import Progress
from gasgiant.params.presets import to_preset_doc
from gasgiant.render.detail import PolarRoute

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

TILE = 1024

# Sequence export encodes each finished frame off-thread (PNG deflate / EXR ZIP
# of a full map is seconds of pure CPU). Bound the in-flight encodes so a long
# sequence can't buffer every frame's arrays in memory: when more than this many
# futures are pending, the generator keeps yielding until they drain.
_MAX_PENDING_ENCODES = 6


def roi_tile_origin(
    center_x: float,
    center_y: float,
    full_w: int,
    full_h: int,
    tile: int = TILE,
) -> tuple[int, int]:
    """Top-left origin of a ``tile``-sized ROI centered (as closely as the map
    bounds allow) on the normalized point ``(center_x, center_y)`` -- each in
    [0, 1] -- of a ``(full_w, full_h)`` map. Clamped so the tile stays wholly
    inside the map; when the map is smaller than the tile on an axis the origin
    is 0 there. Pure (no GL) so the ROI inspector's region math is unit-testable
    without a context. The tile it locates is byte-for-byte the corresponding
    crop of a full export at the same dims (see derive_tile's origin/full_size)."""
    def axis(c: float, full: int) -> int:
        if full <= tile:
            return 0
        o = int(round(c * full - tile / 2.0))
        return max(0, min(full - tile, o))
    return axis(center_x, full_w), axis(center_y, full_h)


def derive_tile(
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
        mask=snap.mask,
        mask_params=params.mask,
    )


def _cube_face_size(width: int) -> int:
    """Per-face square size for a cube map derived from the equirect ``width``.

    ``width/4`` matches the equator texel density of the equirect map (equirect
    has ``width/(2*pi)`` texels/radian in longitude; a cube face of size F has
    ``2F/pi`` at its center, so ``F = width/4`` equalizes them). Floored at 64 so
    tiny widths still produce a usable set."""
    return max(width // 4, 64)


def _export_cube_job(
    sim: Any, out_dir: Path, snap: Any, params: Any, width: int, gpu: Any
) -> Iterator[Progress]:
    """Render a 6-face cube map (T17). Each face is a ``face_size`` square derived
    with the PROJECTION_CUBE variant (``cube_face`` 0..5 = +X,-X,+Y,-Y,+Z,-Z),
    tiled exactly like the equirect path so large faces stream. Writes
    ``<map>_<face>.<ext>`` per map plus a v2 faces-manifest.

    The synthesized detail layer is intentionally OMITTED here: detail synthesis
    maps tile pixels through an EQUIRECT lat/lon, so per cube face it would
    produce geometrically-wrong, seam-breaking filaments. The tracer-driven
    detail_gain term still applies (it reads the equirect tracer at the correct
    direction). Flow/rings maps (equirect-space conventions) are also not part of
    the cube set. The cube job owns ``snap``'s release."""
    # A cube export silently drops flow/rings (equirect-space conventions) and
    # the synthesized detail layer -- warn so an artist who enabled those on a
    # cube export isn't surprised by their absence (docs record it, but nothing
    # else user-facing does).
    if params.export.flow_map or params.rings.enabled:
        log.warning(
            "cube projection omits the flow map and rings (equirect-space "
            "features); export a separate equirect map set for those."
        )
    face_size = _cube_face_size(width)
    emission_on = params.emission.enabled
    tiles = [
        (x, y)
        for y in range(0, face_size, TILE)
        for x in range(0, face_size, TILE)
    ]
    total = 6 * len(tiles) + 2  # + encode + manifest

    tile_color = gpu.texture2d((TILE, TILE), 4, "f4")
    tile_height = gpu.texture2d((TILE, TILE), 1, "f4")
    tile_emission = gpu.texture2d((TILE, TILE), 4, "f4") if emission_on else None

    pool = ThreadPoolExecutor(max_workers=3)
    futures: list[Future] = []
    written: list[Path] = []
    completed = False
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        started = time.perf_counter()
        step = 0
        for face in range(6):
            color_full = np.empty((face_size, face_size, 3), dtype=np.uint16)
            height_full = np.empty((face_size, face_size), dtype=np.float32)
            emission_full = (
                np.empty((face_size, face_size, 4), dtype=np.float32) if emission_on else None
            )
            for x0, y0 in tiles:
                tw = min(TILE, face_size - x0)
                th = min(TILE, face_size - y0)
                sim.deriver.derive(
                    snap.tracers_eq, snap.tracers_n, snap.tracers_s,
                    snap.patch_rho_max, snap.blend_band,
                    tile_color, tile_height, params.appearance,
                    detail_tex=None, detail_intensity=0.0,
                    origin=(x0, y0), full_size=(face_size, face_size),
                    lanes=snap.lanes, warp=snap.warp,
                    emission_out=tile_emission,
                    emission=params.emission if tile_emission is not None else None,
                    seed=params.seed,
                    profile_dyn=snap.profile_dyn, profile_stamp=snap.profile_stamp,
                    mask=snap.mask, mask_params=params.mask,
                    projection_cube=True, cube_face=face,
                )
                color = gpu.read_texture(tile_color)[:th, :tw, :3]
                color_full[y0 : y0 + th, x0 : x0 + tw] = (
                    np.clip(color, 0.0, 1.0) * 65535.0 + 0.5
                ).astype(np.uint16)
                height_full[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(tile_height)[
                    :th, :tw, 0
                ]
                if emission_on:
                    emission_full[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(
                        tile_emission
                    )[:th, :tw]
                step += 1
                yield Progress(step, total, f"cube face {face + 1}/6")

            fn = CUBE_FACE_NAMES[face]
            cpath = out_dir / f"color_{fn}.png"
            written.append(cpath)
            futures.append(pool.submit(
                write_png16_rgb_u16, cpath, color_full.copy(), params.export.png_compression,
            ))
            hpath = out_dir / f"height_{fn}.exr"
            written.append(hpath)
            futures.append(pool.submit(write_exr_gray, hpath, height_full.copy()))
            if emission_on:
                epath = out_dir / f"emission_{fn}.exr"
                written.append(epath)
                futures.append(pool.submit(write_exr_rgba, epath, emission_full.copy()))

        while not all(f.done() for f in futures):
            yield Progress(total - 1, total, "encoding")
            time.sleep(0.01)
        for f in futures:
            f.result()  # surface worker exceptions

        def _faces(prefix: str, ext: str) -> dict[str, str]:
            return {fn: f"{prefix}_{fn}.{ext}" for fn in CUBE_FACE_NAMES}

        maps: dict[str, dict[str, Any]] = {
            "color": {
                "faces": _faces("color", "png"), "format": "png16",
                "colorspace": "srgb", "channels": 3,
            },
            "height": {
                "faces": _faces("height", "exr"), "format": "exr32f",
                "colorspace": "non-color", "channels": 1,
            },
        }
        if emission_on:
            maps["emission"] = {
                "faces": _faces("emission", "exr"), "format": "exr32f",
                "colorspace": "non-color", "channels": 4,
                "aurora_color": list(params.emission.aurora_color),
            }
        physical = {
            "radius_km": params.physical.radius_km,
            "height_scale": params.physical.height_scale,
            "height_midlevel": params.physical.height_midlevel,
        }
        manifest = build_manifest(
            name=params.name,
            seed=params.seed,
            resolution=(face_size, face_size),
            maps=maps,
            physical=physical,
            preset_doc=to_preset_doc(params),
            atmosphere_hint={"rim_color": [0.55, 0.65, 1.0], "rim_strength": 0.4},
            projection=PROJECTION_CUBE,
        )
        write_manifest(out_dir, manifest)
        completed = True
        log.info("exported %d-face cube map (%dpx faces) to %s in %.1fs",
                 6, face_size, out_dir, time.perf_counter() - started)
        yield Progress(total, total, "done")
    finally:
        pool.shutdown(wait=True)
        tile_color.release()
        tile_height.release()
        if tile_emission is not None:
            tile_emission.release()
        snap.release()
        if not completed:
            for p in written:
                p.unlink(missing_ok=True)
            (out_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
            log.info("cube export cancelled; partial output removed")


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

    # Cube-map export (T17) is a wholly separate output path (6 square faces, a
    # v2 faces-manifest); the equirect path below is untouched, so a default
    # export is byte-identical. The cube job owns the snapshot's release.
    if str(params.export.projection) == PROJECTION_CUBE:
        yield from _export_cube_job(sim, out_dir, snap, params, w, gpu)
        return

    tiles = [
        (x, y)
        for y in range(0, h, TILE)
        for x in range(0, w, TILE)
    ]
    total = len(tiles) + 2  # + encode + manifest

    emission_on = params.emission.enabled
    flow_on = params.export.flow_map
    rings_on = params.rings.enabled
    color_full = np.empty((h, w, 3), dtype=np.uint16)
    height_full = np.empty((h, w), dtype=np.float32)
    emission_full = np.empty((h, w, 4), dtype=np.float32) if emission_on else None
    flow_full = np.empty((h, w, 4), dtype=np.float32) if flow_on else None

    tile_color = gpu.texture2d((TILE, TILE), 4, "f4")
    tile_height = gpu.texture2d((TILE, TILE), 1, "f4")
    tile_detail = gpu.texture2d((TILE, TILE), 1, "f4", linear=True)
    tile_emission = gpu.texture2d((TILE, TILE), 4, "f4") if emission_on else None
    tile_flow = gpu.texture2d((TILE, TILE), 4, "f4") if flow_on else None

    pool = ThreadPoolExecutor(max_workers=3)
    futures: list[Future] = []
    completed = False
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        started = time.perf_counter()
        for i, (x0, y0) in enumerate(tiles):
            tw = min(TILE, w - x0)
            th = min(TILE, h - y0)
            derive_tile(
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
            if flow_on:
                # Resample the frozen velocity field into (east, north) for this
                # tile. Reads only snapshot velocity textures + analytic feather,
                # so tiles agree at their borders just like the color/height path.
                sim.deriver.resample_flow(
                    snap.vel_eq, snap.vel_n, snap.vel_s,
                    snap.patch_rho_max, snap.blend_band,
                    tile_flow, origin=(x0, y0), full_size=(w, h),
                )
                flow_full[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(
                    tile_flow
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
        if flow_on:
            futures.append(pool.submit(write_exr_rgba, out_dir / "flow.exr", flow_full))
        if rings_on:
            # Rings are a CPU-only radial strip (no GL); build then encode. A
            # separate exported map -- the color/height/emission path above is
            # untouched, so a rings-enabled export is byte-identical there.
            rings_strip = ring_strip(params)
            futures.append(pool.submit(write_exr_rgba, out_dir / "rings.exr", rings_strip))
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
        if flow_on:
            # RG = (eastward, northward) sim per-step velocity; B=0, A=1. The
            # convention string names the channel layout + units for the importer.
            maps["flow"] = {
                "file": "flow.exr", "format": "exr32f",
                "colorspace": "non-color", "channels": 4,
                "convention": "rg_east_north_texel_per_step",
            }
        if rings_on:
            # RGBA radial strip: axis 0 (long) = radius inner->outer, A = coverage.
            # The importer builds an annulus from physical.ring_inner_km/outer_km.
            maps["rings"] = {
                "file": "rings.exr", "format": "exr32f",
                "colorspace": "non-color", "channels": 4,
                "convention": "radial_inner_to_outer_alpha_coverage",
            }
        physical = {
            "radius_km": params.physical.radius_km,
            "height_scale": params.physical.height_scale,
            "height_midlevel": params.physical.height_midlevel,
        }
        if rings_on:
            physical["ring_inner_km"] = params.physical.ring_inner_km
            physical["ring_outer_km"] = params.physical.ring_outer_km
        manifest = build_manifest(
            name=params.name,
            seed=params.seed,
            resolution=(w, h),
            maps=maps,
            physical=physical,
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
        if tile_flow is not None:
            tile_flow.release()
        snap.release()
        if not completed:
            # Cancellation: remove only the files WE write (the user may have
            # picked a folder containing their own data — e.g. a rings.exr from
            # an earlier rings-enabled export), after the pool drained so there
            # are no Windows open-handle races.
            names = ["color.png", "height.exr", MANIFEST_FILENAME]
            if emission_on:
                names.append("emission.exr")
            if flow_on:
                names.append("flow.exr")
            if rings_on:
                names.append("rings.exr")
            for name in names:
                (out_dir / name).unlink(missing_ok=True)
            log.info("export cancelled; partial output removed")


def run_export(sim: Any, out_dir: Path, width: int | None = None) -> None:
    """Drain the job synchronously (CLI / tests)."""
    for _ in export_job(sim, out_dir, width):
        pass


def _reap(f: Future) -> bool:
    """True (and surface any worker exception) once encode future ``f`` is done;
    False while it is still running. Lets the sequence job prune finished encodes
    and fail fast on an encode error."""
    if f.done():
        f.result()  # re-raise a worker exception at the driver
        return True
    return False


def export_sequence_job(
    sim: Any, out_dir: Path, frames: int, steps_per_frame: int,
    width: int | None = None, *, all_maps: bool = False,
    video: bool = False, fps: int = 24, ramp_to: Any | None = None,
) -> Iterator[Progress]:
    """Animated sequence export.

    Frame 0 is the full existing mapset export (its color map duplicated as
    ``frames/frame_0000.png``); each subsequent frame advances the sim by
    ``steps_per_frame`` via ``Simulation.extend_run`` and renders through the
    same per-tile path as the mapset export. The per-frame loop yields a
    ``Progress`` PER TILE (not once per frame) and pushes each finished frame's
    encode onto a bounded thread pool, so a single frame's render+encode never
    blocks the generator for seconds — the GUI stays responsive.

    ``ramp_to`` (a ``PlanetParams``) turns this into a PARAM RAMP: the look
    interpolates from the base state (t=0, frame 0) to ``ramp_to`` (t=1, the
    last frame). Each frame ``fi`` re-applies ``lerp_params(base, ramp_to, t)``
    with ``t = fi/(frames-1)`` before advancing the sim. Applying a VELOCITY-tier
    diff every frame would clobber the ``extend_run`` frame clock (the facade's
    ``_extra_steps`` reset), so the update goes through
    ``update_params(preserve_target=True)`` -- the velocity field still rebuilds,
    but the development target is left for ``extend_run`` to advance by exactly
    ``steps_per_frame``. ``validate_ramp`` runs ONCE up front (fail fast): a
    RESTART-tier or seed diff cannot be ramped mid-sequence. The non-ramp path
    (``ramp_to is None``) is unchanged.

    ``all_maps`` additionally writes ``frames/height_NNNN.png`` (16-bit gray)
    and, when ``emission.enabled``, ``frames/emission_NNNN.exr`` per frame; the
    frame-0 versions are derived from the base ``height.exr`` / ``emission.exr``
    so every per-map list starts at 0000 like color does. ``video`` runs an
    ffmpeg mp4 encode over the color frames after they all exist.

    The manifest gains an optional ``frames`` block (with a ``maps`` sub-block
    when ``all_maps`` and a ``video`` key when ``video``), written only once
    every output file exists — a cancelled/failed sequence removes everything it
    wrote (never pre-existing user data in ``frames/``), so no half-written
    frame is ever counted in a manifest.

    Determinism note: the kinematic path is byte-exact across runs; vorticity
    frames carry compounding SOR LSB noise (structural guarantees only).

    Flow map (T10): when ``export.flow_map`` is on, frame 0's ``export_job``
    already writes the base ``flow.exr`` (and the manifest ``flow`` entry). A
    per-frame flow sequence (``frames/flow_NNNN.exr``) is the natural extension
    but is NOT yet wired here -- resample the velocity of each frame's snapshot
    with ``sim.deriver.resample_flow`` beside the height/emission tiles and add a
    ``flow`` list to ``maps_block`` (the same slot the note below describes).

    Extension point: additional per-frame maps (e.g. per-frame flow) slot in
    beside height/emission — enqueue their encode alongside the others and add
    their file list to ``maps_block``.
    """
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    if steps_per_frame < 1:
        raise ValueError(f"steps_per_frame must be >= 1, got {steps_per_frame}")

    base_params = sim.params
    if str(base_params.export.projection) == PROJECTION_CUBE:
        # Fail fast BEFORE any dev/GL work: frame 0 would take the cube path
        # (six face files, no color.png), so the frames/ phase has nothing to
        # copy or sequence.
        raise ValueError(
            "sequence export requires export.projection 'equirect'; "
            "a cube-map set has no color.png to sequence"
        )
    if ramp_to is not None:
        # Fail fast BEFORE any GL/dev work: a RESTART-tier or seed diff can't ramp.
        from gasgiant.params.interp import validate_ramp

        validate_ramp(base_params, ramp_to)

    frames_dir = out_dir / "frames"
    written: list[Path] = []
    tile_texs: list[Any] = []
    pool = ThreadPoolExecutor(max_workers=3)
    futures: list[Future] = []
    completed = False
    try:
        # Frame 0: the full mapset export (writes color/height/(emission)/
        # manifest and cleans up after ITSELF if cancelled inside this phase).
        yield from export_job(sim, out_dir, width)

        params = sim.params
        w = width or params.export.width
        h = w // 2
        gpu = sim.gpu
        emission_on = all_maps and params.emission.enabled

        tiles = [(x, y) for y in range(0, h, TILE) for x in range(0, w, TILE)]
        # Progress bookkeeping: frame 0 + a slice per (frame>=1, tile) + a final
        # manifest/done slice. Encoding-wait and video yields reuse the current
        # index so the bar never exceeds 1.
        total = 1 + (frames - 1) * len(tiles) + 1
        step = 0

        frames_dir.mkdir(parents=True, exist_ok=True)

        # -- Frame 0 into frames/: color copy plus (all_maps) the base maps
        # re-expressed at the per-frame names/formats so each list starts 0000.
        frame0 = frames_dir / "frame_0000.png"
        written.append(frame0)
        shutil.copyfile(out_dir / "color.png", frame0)
        if all_maps:
            h0 = frames_dir / "height_0000.png"
            written.append(h0)
            # Per-frame height is a 16-bit gray PNG; convert the base float EXR.
            write_png16_gray(h0, np.clip(read_exr_gray(out_dir / "height.exr"), 0.0, 1.0))
            if emission_on:
                e0 = frames_dir / "emission_0000.exr"
                written.append(e0)
                shutil.copyfile(out_dir / "emission.exr", e0)  # same format: copy
        step += 1
        yield Progress(step, total, "frame 0")

        tile_color = gpu.texture2d((TILE, TILE), 4, "f4")
        tile_height = gpu.texture2d((TILE, TILE), 1, "f4")
        tile_detail = gpu.texture2d((TILE, TILE), 1, "f4", linear=True)
        tile_texs += [tile_color, tile_height, tile_detail]
        tile_emission = gpu.texture2d((TILE, TILE), 4, "f4") if emission_on else None
        if tile_emission is not None:
            tile_texs.append(tile_emission)

        color_full = np.empty((h, w, 3), dtype=np.uint16)
        height_full = np.empty((h, w), dtype=np.float32) if all_maps else None
        emission_full = np.empty((h, w, 4), dtype=np.float32) if emission_on else None

        for fi in range(1, frames):
            if ramp_to is not None:
                from gasgiant.params.interp import lerp_params

                # t spans 0 (frame 0 = base) .. 1 (last frame = ramp_to). Apply the
                # lerped look, then advance EXACTLY steps_per_frame: preserve_target
                # keeps the VELOCITY-tier reset from clobbering the extend_run clock.
                t = fi / (frames - 1)
                sim.update_params(lerp_params(base_params, ramp_to, t), preserve_target=True)
            sim.extend_run(steps_per_frame)
            snap = sim.create_snapshot()
            try:
                for ti, (x0, y0) in enumerate(tiles):
                    tw = min(TILE, w - x0)
                    th = min(TILE, h - y0)
                    derive_tile(
                        sim, snap, snap.params, x0, y0, w, h,
                        tile_color, tile_height, tile_detail, tile_emission,
                    )
                    color = gpu.read_texture(tile_color)[:th, :tw, :3]
                    color_full[y0 : y0 + th, x0 : x0 + tw] = (
                        np.clip(color, 0.0, 1.0) * 65535.0 + 0.5
                    ).astype(np.uint16)
                    if all_maps:
                        height_full[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(
                            tile_height
                        )[:th, :tw, 0]
                    if emission_on:
                        emission_full[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(
                            tile_emission
                        )[:th, :tw]
                    step += 1
                    yield Progress(step, total, f"frame {fi} tile {ti + 1}/{len(tiles)}")
            finally:
                snap.release()

            # Frame arrays complete: enqueue the encodes off-thread. Copy the
            # reused buffers so the worker owns a stable snapshot; track paths
            # BEFORE submitting so a partial file is always in the cleanup list.
            cpath = frames_dir / f"frame_{fi:04d}.png"
            written.append(cpath)
            futures.append(pool.submit(
                write_png16_rgb_u16, cpath, color_full.copy(), params.export.png_compression,
            ))
            if all_maps:
                hpath = frames_dir / f"height_{fi:04d}.png"
                written.append(hpath)
                futures.append(pool.submit(write_png16_gray, hpath, height_full.copy()))
            if emission_on:
                epath = frames_dir / f"emission_{fi:04d}.exr"
                written.append(epath)
                futures.append(pool.submit(write_exr_rgba, epath, emission_full.copy()))

            # Bound in-flight encodes: reap finished ones (surfacing errors) and,
            # while too many remain, keep yielding so we don't block or OOM.
            futures = [f for f in futures if not _reap(f)]
            while len(futures) > _MAX_PENDING_ENCODES:
                yield Progress(step, total, f"frame {fi} encoding")
                time.sleep(0.005)
                futures = [f for f in futures if not _reap(f)]

        # Drain remaining encodes before the manifest counts them.
        while not all(f.done() for f in futures):
            yield Progress(step, total, "encoding")
            time.sleep(0.01)
        for f in futures:
            f.result()  # surface worker exceptions
        futures = []

        # Optional mp4 (color frames drive it); tracked for cleanup on cancel.
        maps_block: dict[str, list[str]] | None = None
        if all_maps:
            maps_block = {"height": [f"frames/height_{i:04d}.png" for i in range(frames)]}
            if emission_on:
                maps_block["emission"] = [
                    f"frames/emission_{i:04d}.exr" for i in range(frames)
                ]
        video_name: str | None = None
        if video:
            video_path = out_dir / "sequence.mp4"
            written.append(video_path)
            yield from encode_video_job(frames_dir, video_path, fps, w, h)
            video_name = "sequence.mp4"

        manifest = read_manifest(out_dir)
        attach_frames(
            manifest, count=frames, steps_per_frame=steps_per_frame,
            files=[f"frames/frame_{i:04d}.png" for i in range(frames)],
            maps=maps_block, video=video_name,
        )
        write_manifest(out_dir, manifest)
        completed = True
        yield Progress(total, total, "done")
    finally:
        pool.shutdown(wait=True)
        for tex in tile_texs:
            tex.release()
        if not completed:
            # Remove only files WE wrote: the per-frame maps plus the base map
            # set (export_job's own cancellation already covers the frame-0
            # phase; those unlinks are no-ops then). The user's files are
            # untouched, and a non-empty pre-existing frames/ is left in place.
            for p in written:
                p.unlink(missing_ok=True)
            if frames_dir.is_dir():
                with contextlib.suppress(OSError):  # user data in frames/: leave it
                    frames_dir.rmdir()
            names = ["color.png", "height.exr", MANIFEST_FILENAME]
            if base_params.emission.enabled:
                names.append("emission.exr")
            if base_params.export.flow_map:
                names.append("flow.exr")
            if base_params.rings.enabled:
                names.append("rings.exr")
            for name in names:
                (out_dir / name).unlink(missing_ok=True)
            log.info("sequence export cancelled; partial output removed")


def run_export_sequence(
    sim: Any, out_dir: Path, frames: int, steps_per_frame: int,
    width: int | None = None, *, all_maps: bool = False,
    video: bool = False, fps: int = 24, ramp_to: Any | None = None,
) -> None:
    """Drain the sequence job synchronously (CLI / tests)."""
    for _ in export_sequence_job(
        sim, out_dir, frames, steps_per_frame, width,
        all_maps=all_maps, video=video, fps=fps, ramp_to=ramp_to,
    ):
        pass

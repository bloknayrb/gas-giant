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
from dataclasses import dataclass, field
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
    write_png16_bgr_u16,
    write_png16_gray,
    write_png16_gray_u16,
)
from gasgiant.jobs import Progress
from gasgiant.params.presets import to_preset_doc
from gasgiant.render.detail import PolarRoute

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

# H.264 caps a coded width/height at 16384 px (level-independent), so a video
# encode is refused above it even though a still map set of that size is fine.
H264_MAX_DIM = 16384

TILE = 1024

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


def enumerate_tiles(w: int, h: int, tile: int = TILE) -> list[tuple[int, int]]:
    """Top-left corners of the tiles covering a ``w x h`` map, row-major.

    Row-major matters beyond tidiness: it makes every horizontal band of height
    ``tile`` complete before the next band starts, which is what lets a consumer
    finish with a band (encode it, flush it) instead of holding the whole map.
    The last column/row are short when ``w``/``h`` are not multiples of ``tile``;
    callers clamp against ``w``/``h``, so the corners alone define the cover.
    """
    return [(x, y) for y in range(0, h, tile) for x in range(0, w, tile)]


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
            hero_emergence=snap.hero_emergence,
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


def _tex(gpu: Any, made: list[Any], channels: int, dtype: str, **kw: Any) -> Any:
    """Create a TILE-sized texture and record it, so a later failure in the same
    acquisition block can release what was already acquired."""
    tex = gpu.texture2d((TILE, TILE), channels, dtype, **kw)
    made.append(tex)
    return tex


def _release_all(texs: list[Any]) -> None:
    for tex in texs:
        with contextlib.suppress(Exception):
            tex.release()


def _emission_format(params: Any) -> str:
    """Manifest format token for emission.exr. Half-float is a per-export choice
    (export.emission_half), so it is declared per map rather than implied by the
    schema version -- and it must agree with the buffer dtype at every site."""
    return "exr16f" if params.export.emission_half else "exr32f"


def _emission_dtype(params: Any) -> Any:
    """Assembly-buffer dtype for emission, the partner of ``_emission_format``.

    Kept as a function because there are three assembly sites (map set, cube,
    sequence) and they must all agree: an earlier revision wired half-float into
    two of them, so the base map came out half while every sequence frame stayed
    float32. Nothing raised -- the frames block carries no format token, so no
    reader could notice."""
    return np.float16 if params.export.emission_half else np.float32


def _to_u16(a: np.ndarray) -> np.ndarray:
    """Float in [0, 1] -> uint16, using the ONE rounding the 16-bit PNG writers
    document (``clip(x, 0, 1) * 65535 + 0.5``). ``write_png16_gray_u16`` is
    bit-identical to ``write_png16_gray`` only while this matches; the pin is
    tests/unit/test_writers.py.

    Applied per TILE, never to a whole map: the whole-map form costs 2.00x the
    buffer in temporaries, which on a 32K height map is 4.00 GiB."""
    return (np.clip(a, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def _scatter_color_bgr(dst: np.ndarray, rgb: np.ndarray, x0: int, y0: int) -> None:
    """Quantize a float RGB tile into a whole-map color buffer, STORING BGR.

    BGR is OpenCV's native order, so ``write_png16_bgr_u16`` can hand the whole
    buffer to libpng directly; storing RGB would make cv2 materialise a
    contiguous duplicate of the entire map (3.00 GiB at 32768). This reversal is
    the only reason that writer is correct -- drop it and every exported color
    map has red and blue swapped, silently. The channel-order pins are
    tests/gpu/test_cube_export.py and tests/gpu/test_export_sequence.py.

    ``[..., ::-1]`` is a read-only view of ``read_texture``'s non-writeable
    result; it materialises here, one tile at a time."""
    th, tw = rgb.shape[:2]
    dst[y0 : y0 + th, x0 : x0 + tw] = _to_u16(rgb[..., ::-1])


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
    emission_dtype = _emission_dtype(params)
    tiles = [
        (x, y)
        for y in range(0, face_size, TILE)
        for x in range(0, face_size, TILE)
    ]
    total = 6 * len(tiles) + 2  # + encode + manifest

    made: list[Any] = []
    try:  # outside the try below: a throw here would leak snap (see export_job)
        tile_color = _tex(gpu, made, 4, "f4")
        tile_height = _tex(gpu, made, 1, "f4")
        tile_emission = _tex(gpu, made, 4, "f4") if emission_on else None
        # Double-buffered face assembly (see _FrameSet). The cube job used to
        # allocate a set per face and submit a copy of each, with no bound at
        # all -- up to 6 sets plus copies live at once in the worst case (the
        # measured 32768 peak was 8.60 GiB, ~5 sets, since encodes do retire
        # while later faces render). This is also the allocation most likely to
        # fail, and it comes AFTER the textures -- hence the release below.
        sets = [_alloc_cube_set(face_size, emission_on, emission_dtype) for _ in range(2)]
    except BaseException:
        _release_all(made)
        snap.release()
        raise

    pool = ThreadPoolExecutor(max_workers=3)
    written: list[Path] = []
    completed = False
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        started = time.perf_counter()
        step = 0
        for face in range(6):
            fs = sets[face % len(sets)]
            yield from fs.drain(step, total, f"cube face {face + 1}/6 encoding")
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
                _scatter_color_bgr(
                    fs.color, gpu.read_texture(tile_color)[:th, :tw, :3], x0, y0
                )
                fs.height[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(tile_height)[
                    :th, :tw, 0
                ]
                if emission_on:
                    fs.emission[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(
                        tile_emission
                    )[:th, :tw]
                step += 1
                yield Progress(step, total, f"cube face {face + 1}/6")

            fn = CUBE_FACE_NAMES[face]
            cpath = out_dir / f"color_{fn}.png"
            written.append(cpath)
            fs.futures.append(pool.submit(
                write_png16_bgr_u16, cpath, fs.color, params.export.png_compression,
            ))
            hpath = out_dir / f"height_{fn}.exr"
            written.append(hpath)
            fs.futures.append(pool.submit(write_exr_gray, hpath, fs.height))
            if emission_on:
                epath = out_dir / f"emission_{fn}.exr"
                written.append(epath)
                fs.futures.append(pool.submit(write_exr_rgba, epath, fs.emission))

        for fs in sets:  # every set, before the manifest counts the files
            yield from fs.drain(total - 1, total, "encoding")

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
                "faces": _faces("emission", "exr"), "format": _emission_format(params),
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

    tiles = enumerate_tiles(w, h)
    total = len(tiles) + 2  # + encode + manifest

    emission_on = params.emission.enabled
    flow_on = params.export.flow_map
    rings_on = params.rings.enabled
    emission_dtype = _emission_dtype(params)
    # These allocations sit OUTSIDE the try below, so a failure here would skip
    # its finally and leak the snapshot's cloned GL textures (~400 MB VRAM at
    # sim.resolution 4096). The GUI catches MemoryError and keeps running, so
    # that leak accumulates across retries. Release explicitly and re-raise;
    # moving them inside the try instead would raise UnboundLocalError from the
    # finally's unconditional tile-texture .release() calls.
    made: list[Any] = []  # textures created so far, for the failure path below
    try:
        color_full = np.empty((h, w, 3), dtype=np.uint16)
        height_full = np.empty((h, w), dtype=np.float32)
        emission_full = np.empty((h, w, 4), dtype=emission_dtype) if emission_on else None
        flow_full = np.empty((h, w, 4), dtype=np.float32) if flow_on else None

        tile_color = _tex(gpu, made, 4, "f4")
        tile_height = _tex(gpu, made, 1, "f4")
        tile_detail = _tex(gpu, made, 1, "f4", linear=True)
        tile_emission = _tex(gpu, made, 4, "f4") if emission_on else None
        tile_flow = _tex(gpu, made, 4, "f4") if flow_on else None
    except BaseException:
        # Release EVERYTHING already acquired, not just the snapshot: a throw on
        # a later line here would otherwise strand the earlier tile textures
        # (~24 MB of VRAM). The GUI catches and keeps running, so it accumulates
        # across retries -- the same failure this guard exists to prevent.
        _release_all(made)
        snap.release()
        raise

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
            _scatter_color_bgr(
                color_full, gpu.read_texture(tile_color)[:th, :tw, :3], x0, y0
            )
            height_full[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(tile_height)[
                :th, :tw, 0
            ]
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
                write_png16_bgr_u16, out_dir / "color.png", color_full,
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
        yield from _drain(futures, len(tiles) + 1, total, "encoding")

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
                "file": "emission.exr", "format": _emission_format(params),
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


def _drain(
    futures: list[Future], step: int, total: int, message: str
) -> Iterator[Progress]:
    """Yield until every future is done, then surface its result and clear.

    Yields rather than blocking: the GUI drives the export one ``next()`` per
    frame and cancels between them, so a blocking wait would freeze it for a
    whole encode -- over a minute for a 32K color PNG, which alone accounts for
    at least the 61.5 s that png_compression 0 takes off a 32K map set -- and
    cancel would be unobservable for that long.

    ``f.result()`` is what raises. ``concurrent.futures.wait()`` and a bare
    ``done()`` loop both return silently on a failed future, which would turn a
    failed encode into a reported-success export.
    """
    while not all(f.done() for f in futures):
        yield Progress(step, total, message)
        time.sleep(0.005)
    for f in futures:
        f.result()
    futures.clear()


@dataclass
class _FrameSet:
    """One frame's whole-map assembly buffers, plus the encode futures that own
    them.

    Double-buffered: the renderer fills set ``fi % n_sets`` only after that
    set's previous encodes have completed, so buffers go to the pool with NO
    copy and the host peak is exactly ``n_sets`` sets at any width. ``n_sets``
    is two except on a sequence short enough not to need both. Two is also the
    throughput optimum -- for a two-stage pipeline, two buffers give a steady
    state period of ``max(render, encode)``, which is the floor whichever stage
    dominates, so a third buffer cannot help. (Which one dominates is not fixed:
    at the default png_compression the color PNG dominates; at level 0 the
    render does.)

    The futures live HERE rather than in one flat list on purpose. With a shared
    list, a slow color encode on set A plus three fast completions elsewhere
    leaves a low total count, and the loop would refill set A while its writer
    is still reading it -- a silent horizontal splice of two frames.
    """

    color: np.ndarray
    height: np.ndarray | None
    emission: np.ndarray | None
    futures: list[Future] = field(default_factory=list)

    def drain(self, step: int, total: int, message: str) -> Iterator[Progress]:
        """Wait for THIS set's encodes before the renderer refills it. Waiting
        per set rather than on a shared list is the whole double-buffer
        invariant -- see the class docstring."""
        yield from _drain(self.futures, step, total, message)


def _alloc_equirect_set(
    h: int, w: int, all_maps: bool, emission_on: bool, emission_dtype: Any
) -> _FrameSet:
    """Sequence frame buffers. Height is uint16 here (the sequence writes 16-bit
    gray PNGs), which halves it AND removes the 2.00x conversion temporary the
    float writer would otherwise build inside a worker thread. ``emission_dtype``
    must match what export_job used for frame 0, or the base map and the frames
    disagree on precision."""
    return _FrameSet(
        color=np.empty((h, w, 3), dtype=np.uint16),
        height=np.empty((h, w), dtype=np.uint16) if all_maps else None,
        emission=np.empty((h, w, 4), dtype=emission_dtype) if emission_on else None,
    )


def _alloc_cube_set(face_size: int, emission_on: bool, emission_dtype: Any) -> _FrameSet:
    """Cube face buffers. Height stays float32 -- the cube writes EXR."""
    return _FrameSet(
        color=np.empty((face_size, face_size, 3), dtype=np.uint16),
        height=np.empty((face_size, face_size), dtype=np.float32),
        emission=(
            np.empty((face_size, face_size, 4), dtype=emission_dtype)
            if emission_on else None
        ),
    )


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
    if video:
        # Fail fast BEFORE any dev/GL work, for the same reason as the cube guard
        # above: H.264 caps a coded dimension at 16384, so a wider sequence would
        # render every frame and only then die inside ffmpeg. Guarded HERE rather
        # than in the CLI because the GUI reaches this job directly
        # (app/main.py's _start_export), and build_ffmpeg_cmd's width/height are
        # documented as unused -- neither call site would otherwise check.
        seq_w = width or base_params.export.width
        if seq_w > H264_MAX_DIM:
            raise ValueError(
                f"cannot encode video for a {seq_w}px-wide sequence: H.264 caps a "
                f"coded dimension at {H264_MAX_DIM}. Export the frames without "
                f"video, or lower export.width."
            )
    if ramp_to is not None:
        # Fail fast BEFORE any GL/dev work: a RESTART-tier or seed diff can't ramp.
        from gasgiant.params.interp import validate_ramp

        validate_ramp(base_params, ramp_to)

    frames_dir = out_dir / "frames"
    written: list[Path] = []
    made: list[Any] = []
    pool = ThreadPoolExecutor(max_workers=3)
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
            # write_png16_gray clips internally; an outer np.clip here is
            # redundant (bitwise idempotent) and only adds a whole-map temporary.
            write_png16_gray(
                h0, read_exr_gray(out_dir / "height.exr"),
                params.export.png_compression,
            )
            if emission_on:
                e0 = frames_dir / "emission_0000.exr"
                written.append(e0)
                shutil.copyfile(out_dir / "emission.exr", e0)  # same format: copy
        step += 1
        yield Progress(step, total, "frame 0")

        tile_color = _tex(gpu, made, 4, "f4")
        tile_height = _tex(gpu, made, 1, "f4")
        tile_detail = _tex(gpu, made, 1, "f4", linear=True)
        tile_emission = _tex(gpu, made, 4, "f4") if emission_on else None

        # Double-buffered frame assembly (see _FrameSet). Allocated HERE, not at
        # the top of the try: export_job's own whole-map buffers are alive until
        # its generator completes above, and the frame-0 height conversion just
        # above builds a transient of its own. Hoisting these would stack all
        # three. np.empty charges commit immediately on Windows, so the ordering
        # is load-bearing for commit, not just resident set.
        # One set per frame this loop will actually render, capped at two:
        # frames == 1 renders none here (frame 0 is export_job's), frames == 2
        # renders exactly one. A full set is 12.00 GiB at 32768 --all-maps, so
        # allocating an unused one is not a rounding error.
        n_sets = min(max(frames - 1, 0), 2)
        emission_dtype = _emission_dtype(params)
        sets = [
            _alloc_equirect_set(h, w, all_maps, emission_on, emission_dtype)
            for _ in range(n_sets)
        ]

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
                # This set's previous encodes must finish before we overwrite it.
                # Placed after create_snapshot so extend_run + the snapshot clone
                # overlap the pending encode for free.
                fs = sets[fi % n_sets]
                yield from fs.drain(step, total, f"frame {fi} encoding")

                for ti, (x0, y0) in enumerate(tiles):
                    tw = min(TILE, w - x0)
                    th = min(TILE, h - y0)
                    derive_tile(
                        sim, snap, snap.params, x0, y0, w, h,
                        tile_color, tile_height, tile_detail, tile_emission,
                    )
                    _scatter_color_bgr(
                        fs.color, gpu.read_texture(tile_color)[:th, :tw, :3], x0, y0
                    )
                    if all_maps:
                        fs.height[y0 : y0 + th, x0 : x0 + tw] = _to_u16(
                            gpu.read_texture(tile_height)[:th, :tw, 0]
                        )
                    if emission_on:
                        fs.emission[y0 : y0 + th, x0 : x0 + tw] = gpu.read_texture(
                            tile_emission
                        )[:th, :tw]
                    step += 1
                    yield Progress(step, total, f"frame {fi} tile {ti + 1}/{len(tiles)}")
            finally:
                snap.release()

            # Frame arrays complete: enqueue the encodes off-thread. The buffers
            # go WITHOUT a copy -- this set is not refilled until its futures
            # complete. Track paths BEFORE submitting so a partial file is
            # always in the cleanup list.
            cpath = frames_dir / f"frame_{fi:04d}.png"
            written.append(cpath)
            fs.futures.append(pool.submit(
                write_png16_bgr_u16, cpath, fs.color, params.export.png_compression,
            ))
            if all_maps:
                hpath = frames_dir / f"height_{fi:04d}.png"
                written.append(hpath)
                fs.futures.append(pool.submit(
                    write_png16_gray_u16, hpath, fs.height,
                    params.export.png_compression,
                ))
            if emission_on:
                epath = frames_dir / f"emission_{fi:04d}.exr"
                written.append(epath)
                fs.futures.append(pool.submit(write_exr_rgba, epath, fs.emission))

        # Drain EVERY set before the manifest counts the files -- and before
        # encode_video_job below, which reads frame_%04d.png off disk and would
        # otherwise start on a frame still being written (short mp4, exit 0).
        for fs in sets:
            yield from fs.drain(step, total, "encoding")

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
        _release_all(made)
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

"""Iteration harness: develop a preset and render it as an EQUIRECT strip plus
orthographic GLOBE views, so a preset can be judged the way it will be seen.

The README generator writes equirect maps, which are the wrong thing to judge a
gas giant on: the poles are smeared across the full width, so polar treatment
looks catastrophic and band spacing looks wrong. A globe view is what the
Blender user gets.

    uv run python scripts/preview_globe.py --preset gas_giant_warm \
        --out out/pv/warm --sim-res 1024 --dev-steps 700

Writes <out>_eq.png and <out>_globe.png. Orientation is self-checked: pass
--probe to print the latitude of the darkest/brightest rows so a north/south
flip cannot slip through (gas_giant_warm's hero sits at latitude -24, so it must
appear in the LOWER half of the globe).

Not a gate, not a doc asset — a working tool. scripts/scratch/ is gitignored.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gasgiant.engine import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import resolve_preset


def apply_overrides(params, sets: list[str]) -> None:
    """Apply ``section.field=value`` overrides so a lever can be A/B'd WITHOUT
    rebuilding the preset JSON. Values are parsed as Python literals, falling
    back to the raw string (so enums like ``global`` work)."""
    import ast
    for spec in sets:
        path, _, raw = spec.partition("=")
        *parents, field = path.strip().split(".")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        target = params
        for part in parents:          # walk any depth, e.g. poles.north.strength
            target = getattr(target, part)
        setattr(target, field, value)
        print(f"  override {path} = {value!r}", flush=True)


def develop(preset: str, sim_res: int | None, dev_steps: int | None,
            seed: int | None, preview_width: int,
            sets: list[str] | None = None) -> np.ndarray:
    """Return the derived color map as float RGB (H, W, 3), already sRGB-encoded
    (derive writes display space -- the exporter clips 0..1 and labels it srgb)."""
    params = resolve_preset(preset)
    if sim_res is not None:
        params.sim.resolution = sim_res
    if dev_steps is not None:
        params.sim.dev_steps = dev_steps
    if seed is not None:
        params.seed = seed
    if sets:
        apply_overrides(params, sets)

    gpu = GpuContext.headless()
    sim = Simulation(params, gpu)
    t0 = time.time()
    sim.run_to_completion(chunk=64)
    tex, _ = sim.ensure_preview(preview_width)
    arr = np.asarray(gpu.read_texture(tex), dtype=np.float32)
    print(f"developed {params.sim.dev_steps} steps @ {params.sim.resolution} "
          f"in {time.time() - t0:.1f}s -> preview {arr.shape}", flush=True)
    return np.clip(arr[..., :3], 0.0, 1.0)


def _sample(eq: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear-free nearest sample of an equirect by latitude/longitude (rad).

    Row 0 is NORTH (verified by the convention-free pole probe recorded in
    docs/architecture.md), so latitude +pi/2 -> row 0.
    """
    h, w = eq.shape[:2]
    v = (0.5 - lat / np.pi)               # +pi/2 -> 0.0 (north, row 0)
    u = (lon / (2.0 * np.pi)) % 1.0
    r = np.clip((v * (h - 1)).astype(np.int32), 0, h - 1)
    c = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
    return eq[r, c]


def globe(eq: np.ndarray, size: int = 900, lat0: float = 8.0, lon0: float = 0.0,
          limb: float = 0.35, bg: float = 0.045) -> np.ndarray:
    """Orthographic globe centered on (lat0, lon0) degrees.

    Full-lit (no terminator) so the TEXTURE is what's being judged, with only a
    mild limb darkening -- a strong day/night terminator hides half the planet
    and flatters everything it leaves.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    x = (xx - (size - 1) / 2.0) / ((size - 1) / 2.0)
    y = -(yy - (size - 1) / 2.0) / ((size - 1) / 2.0)   # image y is down
    r2 = x * x + y * y
    disc = r2 <= 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    p0, l0 = np.radians(lat0), np.radians(lon0)
    # Camera basis at (lat0, lon0): n = view axis, e = east, u = north.
    n = np.array([np.cos(p0) * np.cos(l0), np.sin(p0), np.cos(p0) * np.sin(l0)])
    e = np.array([-np.sin(l0), 0.0, np.cos(l0)])
    u = np.cross(n, e)

    px = x * e[0] + y * u[0] + z * n[0]
    py = x * e[1] + y * u[1] + z * n[1]
    pz = x * e[2] + y * u[2] + z * n[2]

    lat = np.arcsin(np.clip(py, -1.0, 1.0))
    lon = np.arctan2(pz, px)
    rgb = _sample(eq, lat, lon)

    shade = (1.0 - limb * r2)[..., None]
    out = np.where(disc[..., None], rgb * shade, bg)
    return np.clip(out, 0.0, 1.0)


def save_png(path: Path, img: np.ndarray) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    out8 = (img * 255.0 + 0.5).astype(np.uint8)[..., ::-1]   # RGB -> BGR
    cv2.imwrite(str(path), out8, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"wrote {path}", flush=True)


def probe(eq: np.ndarray) -> None:
    """Print per-row mean luminance at a few latitudes. An orientation flip or a
    dead pole shows up here as numbers, not as a vibe."""
    h = eq.shape[0]
    lum = eq.mean(axis=2).mean(axis=1)
    for lat in (85, 60, 30, 0, -30, -60, -85):
        row = int(round((0.5 - lat / 180.0) * (h - 1)))
        print(f"  lat {lat:+4d} (row {row:4d}): mean luma {lum[row]:.4f}")
    dark = int(np.argmin(lum))
    print(f"  darkest row {dark} -> latitude {90.0 - 180.0 * dark / (h - 1):+.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--out", type=Path, required=True, help="path PREFIX")
    ap.add_argument("--sim-res", type=int, default=1024)
    ap.add_argument("--dev-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--preview-width", type=int, default=2048)
    ap.add_argument("--globe-size", type=int, default=900)
    ap.add_argument("--lat0", type=float, default=8.0)
    ap.add_argument("--lon0", type=float, default=0.0,
                    help="sub-camera longitude; sweep it to see other faces")
    ap.add_argument("--eq-width", type=int, default=1400)
    ap.add_argument("--views", type=int, default=1,
                    help="render N globes at evenly spaced longitudes into one sheet")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="section.field=value",
                    help="override a param without rebuilding the preset")
    args = ap.parse_args()

    eq = develop(args.preset, args.sim_res, args.dev_steps, args.seed,
                 args.preview_width, args.sets)
    if args.probe:
        probe(eq)

    import cv2
    eqs = cv2.resize(eq, (args.eq_width, args.eq_width // 2),
                     interpolation=cv2.INTER_AREA)
    save_png(Path(f"{args.out}_eq.png"), eqs)

    if args.views > 1:
        # One image showing the WHOLE planet: N globes at evenly spaced
        # longitudes. A single view hides half the storms and every seam.
        gap = 10
        s = args.globe_size
        sheet = np.full((s, s * args.views + gap * (args.views - 1), 3), 0.045)
        for i in range(args.views):
            lon = args.lon0 + 360.0 * i / args.views
            g = globe(eq, s, args.lat0, lon)
            x0 = i * (s + gap)
            sheet[:, x0:x0 + s] = g
        save_png(Path(f"{args.out}_globe.png"), sheet)
    else:
        save_png(Path(f"{args.out}_globe.png"),
                 globe(eq, args.globe_size, args.lat0, args.lon0))


if __name__ == "__main__":
    main()

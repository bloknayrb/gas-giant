"""Iteration harness: develop a preset and render it as an EQUIRECT strip plus
orthographic GLOBE views, so a preset can be judged the way it will be seen.

The README generator writes equirect maps, which are the wrong thing to judge a
gas giant on: the poles are smeared across the full width, so polar treatment
looks catastrophic and band spacing looks wrong. A globe view is what the
Blender user gets.

    uv run python scripts/preview_globe.py --preset gas_giant_warm \
        --out out/pv/warm --sim-res 1024 --dev-steps 700

Writes <out>_eq.png and <out>_globe.png. Orientation is checkable two ways, which
matters because this tool SHIPPED once with the globe vertically flipped:
  --selftest  renders synthetic north-white and east-white maps and asserts that
              north is up and east is right, pinning both axes and the column-0
              longitude origin (no GL, no preset) -- run it after touching globe();
  --probe     prints zonal-mean luminance at a ladder of latitudes plus the
              latitude of the darkest row, so a flip shows up as numbers.
Sanity anchor: gas_giant_warm's hero sits at latitude -24, so it must appear in
the LOWER half of the globe.

Not a gate and not a doc-asset generator -- a working tool.
"""
from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import resolve_preset


def apply_overrides(params, sets: list[str]) -> None:
    """Apply ``section.field=value`` overrides so a lever can be A/B'd WITHOUT
    rebuilding the preset JSON. Values are parsed as Python literals, falling
    back to the raw string (so enums like ``global`` work)."""
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

    Row 0 is NORTH -- established by a convention-free pole probe on the exported
    map during the 2026-07-24 orientation investigation, and re-checkable here at
    any time with --probe. So latitude +pi/2 maps to row 0.

    Column 0 is longitude -PI, not 0: the engine's equirect convention is
    ``lon = uv.x * 2*PI - PI`` (derive.comp:182, :285, :318). Without the +0.5
    shift, --lon0 0 centers on map longitude -180 and a --views sheet starts at
    the antimeridian, so sweeping --lon0 to find a feature at a known map
    longitude (or a pinned storms.hero_longitude) lands on its antipode.
    """
    h, w = eq.shape[:2]
    v = (0.5 - lat / np.pi)               # +pi/2 -> 0.0 (north, row 0)
    u = (lon / (2.0 * np.pi) + 0.5) % 1.0  # lon -PI -> column 0
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
    # u must be cross(e, n), NOT cross(n, e) -- the latter points SOUTH and renders
    # the planet upside down (at lat0=lon0=0: n=(1,0,0), e=(0,0,1), n x e =
    # (0,-1,0)). This tool shipped with that flip and it inverted every
    # hemisphere claim made from a globe view. `--selftest` pins it.
    n = np.array([np.cos(p0) * np.cos(l0), np.sin(p0), np.cos(p0) * np.sin(l0)])
    e = np.array([-np.sin(l0), 0.0, np.cos(l0)])
    u = np.cross(e, n)

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
    path.parent.mkdir(parents=True, exist_ok=True)
    out8 = (img * 255.0 + 0.5).astype(np.uint8)[..., ::-1]   # RGB -> BGR
    cv2.imwrite(str(path), out8, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"wrote {path}", flush=True)


def selftest() -> None:
    """Pin the globe's north/south AND east/west orientation with synthetic maps.

    Needs no GL and no sim. Worth having because the orientation of this tool is
    the one property everything judged through it depends on, it is invisible in
    any real render of a roughly symmetric planet, and it was in fact WRONG on
    first write (u = cross(n, e), which points south). This repo has already lost
    a multi-day investigation to the same class of confusion.

    The longitude half pins BOTH facts a globe view can get wrong about the
    east-west axis: handedness (east to the right) and the column-0 origin
    (-180, not 0). A quadrant map catches either independently -- a mirror puts
    the bright wedge on the wrong side, a missing +0.5 shift puts it off-disc.
    """
    eq = np.zeros((256, 512, 3), np.float32)
    eq[:128] = 1.0                      # northern half white; row 0 is north
    g = globe(eq, size=200, lat0=0.0, lon0=0.0, limb=0.0, bg=0.5)
    top = float(g[40:80, 90:110].mean())
    bottom = float(g[120:160, 90:110].mean())
    assert top > bottom, (
        f"globe() is vertically FLIPPED: a north-white equirect rendered "
        f"top={top:.3f} bottom={bottom:.3f}; north must be up"
    )
    print(f"selftest OK: north renders up (top {top:.3f} > bottom {bottom:.3f})")

    # Columns 256..383 are longitude 0..+90 under `lon = uv.x*2*PI - PI`, so
    # centering on lon0=0 must light the RIGHT (east) half of the disc.
    eq = np.zeros((256, 512, 3), np.float32)
    eq[:, 256:384] = 1.0
    g = globe(eq, size=200, lat0=0.0, lon0=0.0, limb=0.0, bg=0.5)
    left = float(g[90:110, 40:80].mean())
    right = float(g[90:110, 120:160].mean())
    assert right > left, (
        f"globe() longitude is wrong: an east-white quadrant (map columns "
        f"256..383 = lon 0..+90) rendered left={left:.3f} right={right:.3f}; "
        f"east of the center meridian must fall on the RIGHT of the disc"
    )
    print(f"selftest OK: east renders right (right {right:.3f} > left {left:.3f})")


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
    ap.add_argument("--preset", help="factory preset name or .json path")
    ap.add_argument("--selftest", action="store_true",
                    help="verify globe() orientation (no GL, no preset needed)")
    ap.add_argument("--out", type=Path, help="path PREFIX")
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

    if args.selftest:
        selftest()
        return
    if not args.preset or args.out is None:
        ap.error("--preset and --out are required unless --selftest is given")

    eq = develop(args.preset, sim_res=args.sim_res, dev_steps=args.dev_steps,
                 seed=args.seed, preview_width=args.preview_width, sets=args.sets)
    if args.probe:
        probe(eq)

    eqs = cv2.resize(eq, (args.eq_width, args.eq_width // 2),
                     interpolation=cv2.INTER_AREA)
    save_png(Path(f"{args.out}_eq.png"), eqs)

    # One image showing the WHOLE planet: --views N globes at evenly spaced
    # longitudes. A single view hides half the storms and every seam. N=1 is the
    # same path with no gaps, so there is no separate single-globe branch.
    views = max(1, args.views)
    gap = 10
    s = args.globe_size
    sheet = np.full((s, s * views + gap * (views - 1), 3), 0.045)
    for i in range(views):
        lon = args.lon0 + 360.0 * i / views
        sheet[:, i * (s + gap):i * (s + gap) + s] = globe(eq, s, args.lat0, lon)
    save_png(Path(f"{args.out}_globe.png"), sheet)


if __name__ == "__main__":
    main()

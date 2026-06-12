"""FYI montage (v1.5, no gate): jupiter_like hero_aspect 2.0 vs 1.0.

Renders the flagship preset with the new hero elongation OFF (1.0) and ON (2.0),
hero pinned to the real-GRS latitude so the crop is locatable, and emits a
dual-scale (native + matched) GRS crop montage. Lets the user sight the
flagship's identity change BEFORE the expensive P5 tuning loop.

Only hero_aspect (and hero_latitude, to locate the spot) vary; the v1.5 texture
knobs stay at preset defaults (P5 turns those on). rim_contrast left at preset.

Usage:
    uv run python scripts/fyi_hero_aspect.py --width 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import load_factory_preset

OUT = Path("out/audit/fyi_hero_aspect")
REF = Path("refs/PIA07782.jpg")
HERO_LAT = -22.5            # real GRS latitude (within cap for jupiter radius)
LON_HALF, LAT_HALF = 22.0, 13.0   # crop half-extents (deg) around the hero


def _crop_deg(img, lat0, lat1, lon0, lon1):
    h, w = img.shape[:2]
    y0 = int(np.clip(round((90.0 - lat0) / 180.0 * h), 0, h - 1))
    y1 = int(np.clip(round((90.0 - lat1) / 180.0 * h), y0 + 1, h))
    x0f = (lon0 + 180.0) / 360.0 * w
    x1f = (lon1 + 180.0) / 360.0 * w
    if x1f <= x0f:
        x1f += w
    x0 = int(np.floor(x0f)) % w
    span = max(1, int(round(x1f - x0f)))
    if x0 + span <= w:
        return img[y0:y1, x0:x0 + span]
    return np.roll(img, -x0, axis=1)[y0:y1, :span]


def _fit_width(img, width):
    if img.shape[1] == width:
        return img
    h = max(1, round(img.shape[0] * width / img.shape[1]))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def _save(path: Path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _pad_to(img, h, w):
    out = np.zeros((h, w, 3), dtype=img.dtype)
    out[: img.shape[0], : img.shape[1]] = img
    return out


def render(gpu, aspect: float, width: int):
    p = load_factory_preset("jupiter_like")
    p.storms.hero_latitude = HERO_LAT
    p.storms.hero_aspect = aspect
    sim = Simulation(p, gpu)
    color = np.clip(sim.render_maps(width)["color"][..., :3], 0.0, 1.0)
    hero = sim.vortices.heroes()[0]
    lon_deg = float(np.degrees(hero.lon))
    sim._release_sim()
    return color, lon_deg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=8192)
    args = ap.parse_args()

    gpu = GpuContext.headless()
    gpu.make_current()
    ref_w = cv2.imread(str(REF)).shape[1] if REF.exists() else 2048

    natives, matched = [], []
    for asp in (1.0, 2.0):
        full, lon = render(gpu, asp, args.width)
        box = (HERO_LAT + LAT_HALF, HERO_LAT - LAT_HALF, lon - LON_HALF, lon + LON_HALF)
        nat = _crop_deg(full, *box)
        mat = _crop_deg(_fit_width(full, ref_w), *box)
        _save(OUT / f"grs_native_aspect{asp:.0f}.png", nat)
        _save(OUT / f"grs_matched_aspect{asp:.0f}.png", mat)
        natives.append(nat)
        matched.append(mat)
        print(f"aspect {asp}: hero lon {lon:.1f} deg, native crop {nat.shape}, matched {mat.shape}")

    # Side-by-side montages (1.0 | 2.0), with a thin separator.
    for tag, pair in (("native", natives), ("matched", matched)):
        a, b = pair
        h = max(a.shape[0], b.shape[0])
        a, b = _pad_to(a, h, a.shape[1]), _pad_to(b, h, b.shape[1])
        sep = np.ones((h, 4, 3), dtype=a.dtype)
        _save(OUT / f"montage_{tag}_1p0_vs_2p0.png", np.concatenate([a, sep, b], axis=1))
    print(f"\nmontages -> {OUT}")


if __name__ == "__main__":
    main()

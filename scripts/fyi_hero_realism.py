"""FYI sweep: hero-realism levers on the gas_giant_warm flagship.

Sweeps ONE dotted param (e.g. storms.hero_mottle, storms.hero_tint_var,
detail.hero_collar_wrap) across a list of intensities, rebuilding the sim per
value, and emits a labeled GRS-crop montage so the lever intensity can be
chosen visually before baking it into the preset. A --combo mode sets all three
appearance levers at once (baseline vs combined) for the final gut check.

Each lever defaults to 0.0 (byte-identical), so the leftmost (0) crop is the
smooth baseline; the visual delta against it is the whole point.

Usage:
    uv run python scripts/fyi_hero_realism.py --param storms.hero_mottle \
        --values 0 0.3 0.6 1.0 --res 2048 --steps 700 --width 4096
    uv run python scripts/fyi_hero_realism.py --combo --mottle 0.8 --tint 0.5 \
        --collar 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import load_factory_preset

OUT = Path("out/audit/fyi_hero_realism")
PRESET = "gas_giant_warm"
HERO_LAT = -21.0  # tracks gas_giant_warm's belt-straddle placement (2026-07-15)
LON_HALF, LAT_HALF = 24.0, 15.0   # crop half-extents (deg) around the hero


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


def _save(path: Path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _label(img, text):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def _set_dotted(p, dotted, value):
    obj, attr = dotted.rsplit(".", 1)
    setattr(getattr(p, obj), attr, value)


def _render(gpu, width, res, steps, overrides: dict):
    p = load_factory_preset(PRESET)
    p.storms.hero_latitude = HERO_LAT
    p.sim.resolution = res
    p.sim.dev_steps = steps
    for dotted, value in overrides.items():
        _set_dotted(p, dotted, value)
    sim = Simulation(p, gpu)
    color = np.clip(sim.render_maps(width)["color"][..., :3], 0.0, 1.0)
    lon = float(np.degrees(sim.vortices.heroes()[0].lon))
    sim._release_sim()
    return color, lon


def _montage(crops, sep_w=6):
    h = max(c.shape[0] for c in crops)
    padded = []
    for c in crops:
        if c.shape[0] != h:
            c = np.pad(c, ((0, h - c.shape[0]), (0, 0), (0, 0)))
        padded.append(c)
    sep = np.ones((h, sep_w, 3), dtype=np.float32)
    parts = []
    for i, c in enumerate(padded):
        parts.append(c)
        if i < len(padded) - 1:
            parts.append(sep)
    return np.concatenate(parts, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", type=str, help="dotted param to sweep, e.g. storms.hero_mottle")
    ap.add_argument("--values", type=float, nargs="+", default=[0.0, 0.3, 0.6, 1.0])
    ap.add_argument("--combo", action="store_true", help="baseline vs all-three combined")
    ap.add_argument("--mottle", type=float, default=0.8)
    ap.add_argument("--tint", type=float, default=0.5)
    ap.add_argument("--collar", type=float, default=0.5)
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--width", type=int, default=4096)
    args = ap.parse_args()

    gpu = GpuContext.headless()
    gpu.make_current()

    crops = []
    if args.combo:
        combos = [
            ("baseline", {}),
            (f"m{args.mottle} t{args.tint} c{args.collar}", {
                "storms.hero_mottle": args.mottle,
                "storms.hero_tint_var": args.tint,
                "detail.hero_collar_wrap": args.collar,
            }),
        ]
        for label, ov in combos:
            full, lon = _render(gpu, args.width, args.res, args.steps, ov)
            crop = _crop_deg(full, HERO_LAT + LAT_HALF, HERO_LAT - LAT_HALF,
                             lon - LON_HALF, lon + LON_HALF)
            crops.append(_label(crop, label))
            print(f"combo {label}: hero lon {lon:.1f}, crop {crop.shape}")
        out = OUT / "montage_combo.png"
    else:
        for v in args.values:
            full, lon = _render(gpu, args.width, args.res, args.steps, {args.param: v})
            crop = _crop_deg(full, HERO_LAT + LAT_HALF, HERO_LAT - LAT_HALF,
                             lon - LON_HALF, lon + LON_HALF)
            crops.append(_label(crop, f"{args.param.split('.')[-1]}={v}"))
            print(f"{args.param}={v}: hero lon {lon:.1f}, crop {crop.shape}")
        out = OUT / f"montage_{args.param.replace('.', '_')}.png"

    _save(out, _montage(crops))
    print(f"\nmontage -> {out}")


if __name__ == "__main__":
    main()

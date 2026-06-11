"""Fit palette calibration data from a cylindrical reference map.

Reads a *cylindrical* true-color reference (PIA07782 — globe photographs
are not valid input; they would need limb-darkening removal) and emits, as
JSON:

- ``palette_rows``: gradient rows at anchor latitudes, each with stops
  fitted from the quartile decomposition (0.0 = belt/dark-quartile median,
  0.5 = overall median, 1.0 = zone/bright-quartile median of a +-window
  around the anchor). Latitudes are signed degrees, north positive.
- ``contrast_envelope``: (latitude_deg, p95-p5 luminance) samples — the
  latitude-contrast envelope target.
- ``latitude_table``: the full per-bin zone/belt/median colors, for
  per-band hue assignment (sample at each band's center latitude).

Fitting happens in display sRGB (pre-AgX); the Blender Cycles render
remains the saturation gate.

Output goes to stdout by default; ``--write PRESET`` merges the fitted
palette_rows into a preset file (requires preset format 2 / palette_rows —
Phase A; until then use --out to save the JSON).

Usage:
  uv run python scripts/calibrate_palette.py --reference refs/PIA07782.jpg
  uv run python scripts/calibrate_palette.py --anchors -60,-25,0,25,60 --out out/calib.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gasgiant.palette.reference import latitude_profile

_DEFAULT_ANCHORS = (-65.0, -40.0, -15.0, 10.0, 40.0, 65.0)
_DEFAULT_WINDOW_DEG = 9.0  # half-width of the latitude window sampled per anchor


def _load_srgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"error: cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _rgb(values: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in values]


def calibrate(
    img: np.ndarray,
    anchors: tuple[float, ...],
    bins: int,
    window_deg: float = _DEFAULT_WINDOW_DEG,
) -> dict:
    profile = latitude_profile(img, bins)

    rows = []
    for anchor in sorted(anchors):
        sel = np.abs(profile.lat_deg - anchor) <= window_deg
        if not sel.any():
            sel = np.argsort(np.abs(profile.lat_deg - anchor))[:3]
        rows.append(
            {
                "latitude": anchor,
                "stops": [
                    {"pos": 0.0, "color": _rgb(np.median(profile.belt_rgb[sel], axis=0))},
                    {"pos": 0.5, "color": _rgb(np.median(profile.median_rgb[sel], axis=0))},
                    {"pos": 1.0, "color": _rgb(np.median(profile.zone_rgb[sel], axis=0))},
                ],
            }
        )

    return {
        "palette_rows": rows,
        "contrast_envelope": [
            {"latitude": round(float(lat), 2), "contrast": round(float(c), 4)}
            for lat, c in zip(profile.lat_deg, profile.contrast, strict=True)
        ],
        "latitude_table": [
            {
                "latitude": round(float(profile.lat_deg[i]), 2),
                "zone": _rgb(profile.zone_rgb[i]),
                "belt": _rgb(profile.belt_rgb[i]),
                "median": _rgb(profile.median_rgb[i]),
            }
            for i in range(len(profile.lat_deg))
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", type=Path, default=Path("refs/PIA07782.jpg"))
    ap.add_argument(
        "--anchors",
        default=",".join(str(a) for a in _DEFAULT_ANCHORS),
        help="comma-separated anchor latitudes in signed degrees",
    )
    ap.add_argument("--bins", type=int, default=90)
    ap.add_argument(
        "--window", type=float, default=_DEFAULT_WINDOW_DEG,
        help="half-width (deg) of the latitude window sampled per anchor",
    )
    ap.add_argument("--out", type=Path, default=None, help="write JSON here instead of stdout")
    ap.add_argument(
        "--write", type=Path, default=None,
        help="merge fitted palette_rows into this preset file (preset format 2+)",
    )
    args = ap.parse_args()

    if not args.reference.exists():
        raise SystemExit(
            f"error: reference {args.reference} not found — run scripts/fetch_references.py first"
        )
    anchors = tuple(float(a) for a in args.anchors.split(","))
    doc = calibrate(_load_srgb(args.reference), anchors, args.bins, args.window)

    if args.write is not None:
        from gasgiant.params.presets import load_preset, save_preset

        params = load_preset(args.write)
        if not hasattr(params.appearance, "palette_rows"):
            raise SystemExit("error: --write requires preset format 2 (appearance.palette_rows)")
        params.appearance.palette_rows = doc["palette_rows"]  # validated by pydantic on save
        save_preset(params, args.write)
        print(f"wrote palette_rows into {args.write}")
        return 0

    text = json.dumps(doc, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

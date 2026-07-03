"""Build a review composite from two equirect PNGs: bilinear polar-ortho caps
side by side (off | on | 4x abs-diff) for a chosen hemisphere."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def polar_ortho(equirect: np.ndarray, pole_sign: float, size: int = 760,
                cap_deg: float = 38.0) -> np.ndarray:
    h, w = equirect.shape[:2]
    cap = np.deg2rad(cap_deg)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    nx = (xx - size / 2.0) / (size / 2.0)
    ny = (yy - size / 2.0) / (size / 2.0)
    rr = np.hypot(nx, ny)
    inside = rr <= 1.0
    colat = rr * cap
    lat = pole_sign * (np.pi / 2.0 - colat)
    lon = np.arctan2(ny, pole_sign * nx)
    fx = ((lon + np.pi) / (2.0 * np.pi)) * w
    fy = ((np.pi / 2.0 - lat) / np.pi) * h
    out = cv2.remap(equirect, fx.astype(np.float32), fy.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    out[~inside] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("out")
    ap.add_argument("--pole", choices=["n", "s"], default="s")
    args = ap.parse_args()
    sign = +1.0 if args.pole == "n" else -1.0
    off = cv2.imread(args.off)
    on = cv2.imread(args.on)
    co = polar_ortho(off, sign)
    cn = polar_ortho(on, sign)
    diff = np.clip(np.abs(cn.astype(np.float64) - co.astype(np.float64)) * 4.0, 0, 255).astype(np.uint8)
    gap = np.full((co.shape[0], 8, 3), 40, np.uint8)
    comp = np.hstack([co, gap, cn, gap, diff])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, comp)

    # DC guard (regression reviewer ask): the masked cap delta must be ~zero-
    # mean (bright crests balanced by dark troughs), else the lever is dumping
    # net albedo into the cap. Measured on the equirect polar rows poleward of
    # the 66 deg gate, where routeW>0.
    h, w = off.shape[:2]
    if args.pole == "n":
        band = slice(0, int(h * 24 / 180))
    else:
        band = slice(int(h * 156 / 180), h)
    sd = (on.astype(np.float64) - off.astype(np.float64))[band]
    lum = sd.mean(axis=2)  # per-pixel mean-channel delta
    frac_bright = float((lum > 2.0).mean())
    frac_dark = float((lum < -2.0).mean())
    print(f"wrote {args.out}  (OFF | ON | 4x-diff, {args.pole}-pole)", flush=True)
    print(f"  cap DC (signed mean delta, BGR): {sd.mean(axis=(0,1)).round(2)}  "
          f"|mean abs| {np.abs(sd).mean():.2f}", flush=True)
    # Reviewer gates: signed cap DC (mean luma) < ~8, frac-brightened < ~50%
    # (the latter distinguishes discrete wisps from a broad bright haze).
    dc = float(lum.mean())
    g1 = "PASS" if abs(dc) < 8.0 else "FAIL"
    g2 = "PASS" if frac_bright < 0.50 else "FAIL"
    print(f"  GATES: signedDC {dc:+.2f} [{g1} <8]   "
          f"frac_bright {frac_bright:.2f} frac_dark {frac_dark:.2f} [{g2} <0.50]", flush=True)


if __name__ == "__main__":
    main()

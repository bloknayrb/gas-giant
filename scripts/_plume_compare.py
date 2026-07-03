"""Throwaway plume review composite: OFF | ON | 4x abs-diff of the equirect,
cropped to the belt latitudes where convective white-plume trains land, plus a
visibility metric (how much brighter the ON belts got, and where).

Usage:
  python scripts/_plume_compare.py out/plume2/off_d700_equirect.png \
      out/plume2/on_d700_equirect.png out/plume2/cmp_d700.png
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("out")
    args = ap.parse_args()

    off = cv2.imread(args.off).astype(np.float64)
    on = cv2.imread(args.on).astype(np.float64)
    h = off.shape[0]
    # Belt band: |lat| in ~[11.5, 57] deg -> the dark-belt placement window.
    top = int(h * (90 - 57) / 180)
    bot = int(h * (90 + 57) / 180)
    band = slice(top, bot)

    diff = np.clip(np.abs(on - off) * 4.0, 0, 255).astype(np.uint8)
    gap = np.full((h, 8, 3), 40, np.uint8)
    comp = np.hstack([off.astype(np.uint8), gap, on.astype(np.uint8), gap, diff])
    cv2.imwrite(args.out, comp)

    d = (on - off)[band]
    lum = d.mean(axis=2)
    frac_bright = float((lum > 4.0).mean())
    peak = float(lum.max())
    print(f"belt mean dLum {lum.mean():+.2f}  peak +{peak:.1f}  "
          f"frac_brightened(>4) {frac_bright:.3f}  wrote {args.out}")


if __name__ == "__main__":
    main()

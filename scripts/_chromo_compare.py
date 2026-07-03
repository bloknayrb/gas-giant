"""Throwaway chromophore-aging review: OFF | ON | 4x-diff full equirect, plus a
tropical belt/zone crop (the region where chromophore variety reads), and a
chroma-stat readout (mean Oklab chroma off vs on, and correlation of chroma
change with the T2 freshness proxy via local luma as a sanity check)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("out")
    args = ap.parse_args()
    off = cv2.imread(args.off)
    on = cv2.imread(args.on)
    h, w = off.shape[:2]
    diff = np.clip(np.abs(on.astype(np.float64) - off.astype(np.float64)) * 4.0, 0, 255).astype(np.uint8)
    gap = np.full((h, 6, 3), 40, np.uint8)
    full = np.hstack([off, gap, on, gap, diff])
    # Tropical crop (|lat|<35): rows 0.30h..0.70h, a 0.45w longitude window.
    r0, r1 = int(0.30 * h), int(0.70 * h)
    c0, c1 = int(0.10 * w), int(0.55 * w)
    crop = np.hstack([off[r0:r1, c0:c1], gap[: r1 - r0], on[r0:r1, c0:c1]])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, full)
    cv2.imwrite(str(Path(args.out).with_name(Path(args.out).stem + "_crop.png")), crop)

    # Approx saturation (max-min over channels, 0-255) as a cheap chroma proxy.
    def sat(im):
        f = im.astype(np.float64)
        return f.max(axis=2) - f.min(axis=2)
    so, sn = sat(off), sat(on)
    band = slice(int(0.30 * h), int(0.70 * h))  # tropics
    mo, mn = so[band].mean(), sn[band].mean()
    sdo, sdn = so[band].std(), sn[band].std()
    mean_pct = 100.0 * (mn - mo) / max(mo, 1e-6)
    std_pct = 100.0 * (sdn - sdo) / max(sdo, 1e-6)
    # Reviewer gate: net mean-sat rise <= +12% (don't regress the muted Cassini
    # calibration); variety (std) should rise MORE than the mean (the on-target
    # signal dominates the side effect).
    g_mean = "PASS" if mean_pct <= 12.0 else "FAIL"
    g_ratio = "PASS" if std_pct >= mean_pct else "WEAK"
    print(f"wrote {args.out} (+_crop)", flush=True)
    print(f"  tropics mean sat OFF {mo:.1f} -> ON {mn:.1f} ({mean_pct:+.0f}%) [{g_mean} <=+12%]",
          flush=True)
    print(f"  tropics sat std  OFF {sdo:.1f} -> ON {sdn:.1f} ({std_pct:+.0f}% variety) "
          f"[{g_ratio}: variety>=mean]", flush=True)
    # Targeting gate (fidelity reviewer): warming (delta R - delta B) must
    # correlate NEGATIVELY with brightness -- pigment lands in DARK belts, not
    # bright zones. Per-row over the tropics.
    of = off.astype(np.float64); on_ = on.astype(np.float64)
    warm = (on_[..., 2] - of[..., 2]) - (on_[..., 0] - of[..., 0])  # BGR: R=idx2,B=idx0
    bright = of.mean(axis=2)
    rw = warm[band].mean(axis=1); rb = bright[band].mean(axis=1)
    corr = float(np.corrcoef(rw, rb)[0, 1])
    g_t = "PASS" if corr < 0.0 else "FAIL"
    print(f"  targeting corr(warming, brightness) = {corr:+.2f} [{g_t} <0 = belts not zones]",
          flush=True)


if __name__ == "__main__":
    main()

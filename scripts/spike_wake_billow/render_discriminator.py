"""Render terminal-state tracer bands of the unforced-decay discriminator runs,
alongside the metric coherence readout, for visual adjudication."""
from __future__ import annotations
import pathlib, sys
import numpy as np
import cv2
from scipy import ndimage

SPIKE = pathlib.Path(__file__).parent
sys.path.insert(0, str(SPIKE))
import config as C          # noqa: E402
import metrics as M         # noqa: E402

RUNS = SPIKE / "discriminator_runs"
FREE = ["deficit_A10_d0.15_gate1_free", "deficit_A10_d0.30_gate1_free",
        "deficit_A45_d0.15_gate1_free"]


def band_tile(tr_full, label):
    b = M.hp(M.extract_band(tr_full, C), C)
    rms0 = float(np.sqrt(np.mean(b ** 2)))
    lab, n, props, seg = M.find_billows(b, rms0, C)
    coh = M.coherence(b, C)
    lo, hi = np.percentile(b, 2), np.percentile(b, 98)
    g = np.clip((b - lo) / max(hi - lo, 1e-9), 0, 1)
    img = (np.dstack([g, g, g]) * 255).astype(np.uint8)
    if lab is not None and lab.max() > 0:
        edges = (lab > 0).astype(np.uint8)
        outline = edges & (~ndimage.binary_erosion(edges))
        img[outline.astype(bool)] = (0, 255, 0)
    bar = np.full((24, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, f"{label}  nbill={n} coh={coh:.2f}", (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    rows = []
    for name in FREE:
        d = np.load(RUNS / f"{name}.npz", allow_pickle=True)
        snaps = d["snap_tr"]; ssteps = d["snap_steps"]
        # mid (2/3) and terminal snapshot
        tiles = []
        for k in (len(ssteps) - 2, len(ssteps) - 1):
            tiles.append(band_tile(snaps[k], f"{name.replace('deficit_','').replace('_gate1','')} step{int(ssteps[k])}"))
        sep = np.full((tiles[0].shape[0], 8, 3), 80, np.uint8)
        rows.append(np.hstack([tiles[0], sep, tiles[1]]))
        rows.append(np.full((6, rows[-1].shape[1], 3), 60, np.uint8))
    panel = np.vstack(rows[:-1])
    out = SPIKE / "discriminator_terminal.png"
    cv2.imwrite(str(out), panel)
    print(f"-> {out}  ({panel.shape[1]}x{panel.shape[0]})")

    # also report what the REAL production wake reads in the SAME coherence metric
    try:
        band_c, rms0_c = M._status_quo_band(C)
        print(f"REAL production-render wake (metrics case c): coh={M.coherence(M.hp(band_c, C), C):.3f} "
              f"(reference-photo target 0.35-0.55) -- apples-to-apples check vs sim-tracer coh")
    except Exception as e:
        print(f"(status-quo coherence check skipped: {e})")


if __name__ == "__main__":
    main()

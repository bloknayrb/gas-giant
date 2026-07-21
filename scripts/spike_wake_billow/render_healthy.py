"""Render tracer-band snapshots (stable-window vs post-blowup) for visual
adjudication of the healthy-window verdict. Pure re-analysis of stored fields."""
from __future__ import annotations
import pathlib
import sys
import numpy as np
import cv2

SPIKE = pathlib.Path(__file__).parent
sys.path.insert(0, str(SPIKE))
import config as C          # noqa: E402
import metrics as M         # noqa: E402

RUNS = SPIKE / "gate1_runs"
CELLS = [
    ("A45_d0.15_deficit_s12", "deficit A45 d0.15 (top-peak cell)"),
    ("A10_d0.30_deficit_s12", "deficit A10 d0.30 (peak nb=8)"),
    ("A20_d0.15_deficit_s11", "deficit A20 d0.15 (frozen CF 0.79)"),
    ("A45_d0.30_deficit_shearoff_s12", "SHEAR-OFF control"),
]


def band_img(tr_full, label):
    b = M.hp(M.extract_band(tr_full, C), C)
    lo, hi = np.percentile(b, 2), np.percentile(b, 98)
    g = np.clip((b - lo) / max(hi - lo, 1e-9), 0, 1)
    img = (np.dstack([g, g, g]) * 255).astype(np.uint8)
    # segment reference-scale billows for the overlay (uses the frame's own rms0)
    rms0 = float(np.sqrt(np.mean(b ** 2)))
    lab, n, props, seg = M.find_billows(b, rms0, C)
    if lab is not None and lab.max() > 0:
        edges = (lab > 0).astype(np.uint8)
        outline = edges & (~np.asarray(
            __import__("scipy.ndimage", fromlist=["binary_erosion"]).binary_erosion(edges)))
        img[outline.astype(bool)] = (0, 255, 0)
    bar = np.full((22, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, f"{label}  nbill={n}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    rows = []
    for name, title in CELLS:
        d = np.load(RUNS / f"{name}.npz", allow_pickle=True)
        snaps = d["snap_tr"]; ssteps = d["snap_steps"]; ss = d["substep_count"]
        st = d["sample_steps"]; mw = d["max_w"]
        # pick a stable snapshot (earliest) and a late one
        i_stable = 0
        i_late = len(ssteps) - 1
        # is the "stable" snapshot actually stable?
        def subs_at(step):
            k = int(np.argmin(np.abs(st - step)))
            return int(ss[k]), float(mw[k])
        s0, m0 = subs_at(ssteps[i_stable])
        s1, m1 = subs_at(ssteps[i_late])
        a = band_img(snaps[i_stable], f"{title}: step {int(ssteps[i_stable])} subs={s0} mw={m0:.0f}")
        b = band_img(snaps[i_late], f"step {int(ssteps[i_late])} subs={s1} mw={m1:.0f}")
        sep = np.full((a.shape[0], 8, 3), 80, np.uint8)
        rows.append(np.hstack([a, sep, b]))
        rows.append(np.full((6, rows[-1].shape[1], 3), 60, np.uint8))
    panel = np.vstack(rows[:-1])
    out = SPIKE / "healthy_window_montage.png"
    cv2.imwrite(str(out), panel)
    print(f"-> {out}  ({panel.shape[1]}x{panel.shape[0]})")


if __name__ == "__main__":
    main()

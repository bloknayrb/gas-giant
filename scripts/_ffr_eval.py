"""Throwaway FFR review harness: render a preset with --set overrides (incl.
sim.dev_steps for varying playout) and emit (1) the full equirect color map and
(2) a polar-orthographic reprojection of each cap (clean top-down view to
compare against the Juno polar mosaics, e.g. refs/PIA21641.jpg).

Usage:
  python scripts/_ffr_eval.py out/ffr/jv_on_d500 --preset jupiter_vorticity --res 2048 \
      --set sim.dev_steps=500 --set detail.polar_filaments=1.3
Writes <stem>_equirect.png, <stem>_npole.png, <stem>_spole.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine import Simulation
from gasgiant.params.presets import load_factory_preset


def _apply(p, ov: str) -> None:
    path, val = ov.split("=")
    section, field = path.split(".")
    sub = getattr(p, section)
    cur = getattr(sub, field)
    cast = type(cur)(val) if not isinstance(cur, bool) else (val == "True")
    setattr(sub, field, cast)


def polar_ortho(equirect: np.ndarray, pole_sign: float, size: int = 720,
                cap_deg: float = 40.0) -> np.ndarray:
    """Orthographic top-down view of a polar cap, sampled from the equirect.
    pole_sign +1 = north (looking down at +90), -1 = south."""
    h, w = equirect.shape[:2]
    cap = np.deg2rad(cap_deg)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    # Normalized disc coords [-1,1]; radius -> colatitude in [0, cap].
    nx = (xx - size / 2.0) / (size / 2.0)
    ny = (yy - size / 2.0) / (size / 2.0)
    rr = np.hypot(nx, ny)
    inside = rr <= 1.0
    colat = rr * cap
    lat = pole_sign * (np.pi / 2.0 - colat)
    # azimuth -> longitude; flip handedness so north/south both read natural.
    lon = np.arctan2(ny, pole_sign * nx)
    u = (lon + np.pi) / (2.0 * np.pi)
    v = (np.pi / 2.0 - lat) / np.pi
    sx = np.clip((u * w).astype(np.int64), 0, w - 1)
    sy = np.clip((v * h).astype(np.int64), 0, h - 1)
    out = equirect[sy, sx]
    out[~inside] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--preset", default="jupiter_vorticity")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    p = load_factory_preset(args.preset)
    for ov in args.set:
        _apply(p, ov)

    sim = Simulation(p)
    color = np.clip(sim.render_maps(args.res)["color"][..., :3], 0.0, 1.0)
    u8 = (color * 255.0 + 0.5).astype(np.uint8)

    stem = Path(args.stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{stem}_equirect.png", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    for sign, tag in ((+1.0, "npole"), (-1.0, "spole")):
        cap = polar_ortho(u8, sign)
        cv2.imwrite(f"{stem}_{tag}.png", cv2.cvtColor(cap, cv2.COLOR_RGB2BGR))
    print(f"wrote {stem}_*.png  ({args.preset} res={args.res} set={args.set})", flush=True)


if __name__ == "__main__":
    main()

"""Compare a generated map (or preset) against a NASA reference image.

Renders a montage (ours over the reference at matched scale) and a
per-latitude profile strip — zone color, belt color, and the contrast
envelope from the quartile decomposition in gasgiant.palette.reference —
then prints the numeric profile distances. Drawn with OpenCV/numpy only.

Usage:
  uv run python scripts/compare_reference.py jupiter_like
  uv run python scripts/compare_reference.py out/jove/color.png --reference refs/PIA07782.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gasgiant.palette.reference import LatitudeProfile, latitude_profile, profile_distance

_STRIP_W = 72          # width of each color strip column, px
_CURVE_W = 220         # width of the contrast-curve panel, px
_MONTAGE_W = 1280


def _load_srgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"error: cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_target(spec: str, render_width: int) -> tuple[np.ndarray, str]:
    path = Path(spec)
    if path.is_dir():
        path = path / "color.png"
    if path.exists():
        return _load_srgb(path), path.stem
    # Not a file: treat as a preset name and render it.
    from gasgiant.engine import Simulation
    from gasgiant.params.presets import PresetError, resolve_preset

    try:
        params = resolve_preset(spec)
    except PresetError as exc:
        raise SystemExit(f"error: {exc}") from None
    sim = Simulation(params)
    color = sim.render_maps(render_width)["color"][..., :3]
    return np.clip(color, 0.0, 1.0), spec


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h = max(1, round(img.shape[0] * width / img.shape[1]))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _profile_panel(profile: LatitudeProfile, height: int, contrast_max: float) -> np.ndarray:
    """Zone strip | belt strip | contrast curve, one row block per latitude bin."""
    n = len(profile.lat_deg)
    rows = np.linspace(0, height, n + 1).astype(int)
    panel = np.full((height, 2 * _STRIP_W + _CURVE_W, 3), 1.0, dtype=np.float32)
    for i in range(n):
        panel[rows[i]:rows[i + 1], :_STRIP_W] = profile.zone_rgb[i]
        panel[rows[i]:rows[i + 1], _STRIP_W:2 * _STRIP_W] = profile.belt_rgb[i]
    # Contrast curve: x grows with p95-p5, drawn over a neutral background.
    panel[:, 2 * _STRIP_W:] = 0.94
    centers = ((rows[:-1] + rows[1:]) // 2)
    xs = 2 * _STRIP_W + (profile.contrast / contrast_max * (_CURVE_W - 12) + 6).astype(int)
    pts = np.stack([xs, centers], axis=1).reshape(-1, 1, 2)
    img8 = _to_u8(panel)
    cv2.polylines(img8, [pts], False, (40, 40, 200), 2, cv2.LINE_AA)
    return img8.astype(np.float32) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="preset name, mapset dir, or color image path")
    ap.add_argument("--reference", type=Path, default=Path("refs/PIA07782.jpg"))
    ap.add_argument("--res", type=int, default=2048, help="render width when target is a preset")
    ap.add_argument("--bins", type=int, default=90)
    ap.add_argument("--out", type=Path, default=Path("out/compare"))
    args = ap.parse_args()

    if not args.reference.exists():
        raise SystemExit(
            f"error: reference {args.reference} not found — run scripts/fetch_references.py first"
        )

    ours, name = _load_target(args.target, args.res)
    ref = _load_srgb(args.reference)

    p_ours = latitude_profile(ours, args.bins)
    p_ref = latitude_profile(ref, args.bins)
    dist = profile_distance(p_ours, p_ref)

    args.out.mkdir(parents=True, exist_ok=True)

    ours_fit = _label(_fit_width(ours, _MONTAGE_W), f"ours: {name}")
    ref_fit = _label(_fit_width(ref, _MONTAGE_W), f"reference: {args.reference.name}")
    montage = np.concatenate([ours_fit, np.ones((6, _MONTAGE_W, 3), np.float32), ref_fit])
    montage_path = args.out / f"{name}_montage.png"
    cv2.imwrite(str(montage_path), cv2.cvtColor(_to_u8(montage), cv2.COLOR_RGB2BGR))

    cmax = max(p_ours.contrast.max(), p_ref.contrast.max(), 1e-6)
    panel = np.concatenate(
        [
            _label(_profile_panel(p_ours, 720, cmax), "ours"),
            np.ones((720, 6, 3), np.float32),
            _label(_profile_panel(p_ref, 720, cmax), "ref"),
        ],
        axis=1,
    )
    profile_path = args.out / f"{name}_profiles.png"
    cv2.imwrite(str(profile_path), cv2.cvtColor(_to_u8(panel), cv2.COLOR_RGB2BGR))

    print(f"montage:  {montage_path}")
    print(f"profiles: {profile_path}  (zone | belt | contrast, +90N at top)")
    print("profile distance (mean abs, sRGB units / luminance):")
    for key, value in dist.items():
        print(f"  {key:9s} {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

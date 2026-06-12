"""T0 (v1.5) visual: is sim.resolution=4096 transformative for texture density?

Renders jupiter_like at the SAME export width with sim.resolution 2048 vs 4096
(v1.5 turbulence NOT yet added -- this isolates whether the grid resolution
alone densifies belt texture). Emits widest-belt crop pairs at native scale and
at matched (downsampled-to-reference) scale, plus a band-pass RMS texture-energy
number per crop. Feeds the Checkpoint-A "is it transformative" judgment.

The plan's claim: the density gap is injection-frequency-limited, NOT
grid-limited, so 4096 should NOT be transformative -> 2048 stays (and the T0
timing already shows 4096 cannot fit the 40 s gate regardless).

Usage:
    uv run python scripts/t0_visual_4096.py --width 8192
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import load_factory_preset

OUT = Path("out/audit/t0_4096")
REF = Path("refs/PIA07782.jpg")
# Tropical belt strip (NEB analog) -- a reliably-textured belt in jupiter_like.
BELT_LAT_C = 16.0
LAT_HALF = 5.0
LON0, LON1 = -60.0, 0.0


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] == width:
        return img
    h = max(1, round(img.shape[0] * width / img.shape[1]))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


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
    rolled = np.roll(img, -x0, axis=1)
    return rolled[y0:y1, :span]


def _bandpass_rms(rgb: np.ndarray) -> float:
    """DoG (sigma 1.5..4 px) band-pass RMS on luminance -- the TD-1 proxy."""
    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    lo = cv2.GaussianBlur(lum, (0, 0), 1.5)
    hi = cv2.GaussianBlur(lum, (0, 0), 4.0)
    band = lo - hi
    return float(np.sqrt(np.mean(band * band)))


def _save(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def render(gpu, resolution: int, width: int) -> np.ndarray:
    p = load_factory_preset("jupiter_like")
    p.sim.resolution = resolution
    sim = Simulation(p, gpu)
    color = np.clip(sim.render_maps(width)["color"][..., :3], 0.0, 1.0)
    sim._release_sim()
    return color


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=8192)
    args = ap.parse_args()

    gpu = GpuContext.headless()
    gpu.make_current()

    ref = cv2.cvtColor(cv2.imread(str(REF)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 \
        if REF.exists() else None
    ref_w = ref.shape[1] if ref is not None else 2048

    rows = []
    for res in (2048, 4096):
        full = render(gpu, res, args.width)
        # native scale (export width)
        native = _crop_deg(full, BELT_LAT_C + LAT_HALF, BELT_LAT_C - LAT_HALF, LON0, LON1)
        _save(OUT / f"belt_native_res{res}.png", native)
        # matched scale (downsample full to reference width, then crop)
        matched_full = _fit_width(full, ref_w)
        matched = _crop_deg(matched_full, BELT_LAT_C + LAT_HALF, BELT_LAT_C - LAT_HALF, LON0, LON1)
        _save(OUT / f"belt_matched_res{res}.png", matched)
        rows.append((res, native.shape, _bandpass_rms(native),
                     matched.shape, _bandpass_rms(matched)))

    if ref is not None:
        ref_crop = _crop_deg(ref, BELT_LAT_C + LAT_HALF, BELT_LAT_C - LAT_HALF, LON0, LON1)
        _save(OUT / "belt_matched_REFERENCE.png", ref_crop)
        ref_rms = _bandpass_rms(ref_crop)
    else:
        ref_rms = float("nan")

    print(f"\nwidest-belt (lat {BELT_LAT_C}+/-{LAT_HALF}) band-pass RMS (DoG 1.5-4px):")
    print(f"{'sim_res':>8} {'native_RMS':>12} {'matched_RMS':>12}")
    for res, _ns, nrms, _ms, mrms in rows:
        print(f"{res:>8} {nrms:>12.5f} {mrms:>12.5f}")
    print(f"{'REF':>8} {'':>12} {ref_rms:>12.5f}")
    print(f"\ncrops -> {OUT}")


if __name__ == "__main__":
    main()

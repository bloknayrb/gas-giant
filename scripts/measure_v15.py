"""v1.5 P5 measurement harness: TD-1 (belt texture density) + GRS-2 (collar
ripple) on a rendered preset/params.

TD-1: band-pass RMS (DoG sigma 1.5-4 px) on the L channel at width 640 of the
widest tropical-belt matched crop; reports ratio ours/ref and a JPEG-roundtrip
control. Necessary-not-sufficient (TD-2 judges are the gate).

GRS-2: collar ring ripple via measure_grs.ring_ripple_std on the matched-scale
L crop in the hero's own elliptical frame. The GATE is an ablation comparison
(hero_aspect=1, rim_contrast=1) run by the orchestrator against the v1.4 collar.

Usage:
    uv run python scripts/measure_v15.py --preset jupiter_like --width 8192
    uv run python scripts/measure_v15.py --preset jupiter_like --aspect 1.0 --rim 1.0  # ablation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_grs import ring_ripple_std  # noqa: E402

from gasgiant.engine.facade import Simulation  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402
from gasgiant.sim.vortices import KIND_HERO  # noqa: E402

REF = Path("refs/PIA07782.jpg")
JUP_CIRC_KM = 449_197.0


def _lum(rgb):
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def _bandpass_rms(rgb_or_l, lo=1.5, hi=4.0):
    lum = rgb_or_l if rgb_or_l.ndim == 2 else _lum(rgb_or_l)
    band = cv2.GaussianBlur(lum, (0, 0), lo) - cv2.GaussianBlur(lum, (0, 0), hi)
    return float(np.sqrt(np.mean(band * band)))


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


def _fit_width(img, width):
    if img.shape[1] == width:
        return img
    h = max(1, round(img.shape[0] * width / img.shape[1]))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def _widest_tropical_belt(sim):
    edges = np.degrees(sim.bands.edges.astype(np.float64))
    values = sim.bands.values
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[:-1] - edges[1:]
    is_belt = values < np.median(values)
    lowmid = np.abs(centers) < 35.0
    cand = np.where(is_belt & lowmid)[0]
    j = int(cand[np.argmax(widths[cand])])
    return centers[j], widths[j]


def measure(preset, width, aspect=None, rim=None, latitude=None, save_tag=None):
    gpu = GpuContext.headless()
    gpu.make_current()
    p = load_factory_preset(preset)
    if aspect is not None:
        p.storms.hero_aspect = aspect
    if rim is not None:
        p.storms.rim_contrast = rim
    if latitude is not None:
        p.storms.hero_latitude = latitude
    sim = Simulation(p, gpu)
    rgb = np.clip(sim.render_maps(width)["color"][..., :3], 0.0, 1.0)

    # TD-1: widest tropical belt, matched-to-reference scale, width-640 crop.
    bc, bw = _widest_tropical_belt(sim)
    ref = cv2.cvtColor(cv2.imread(str(REF)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ours_matched = _fit_width(rgb, ref.shape[1])
    box = (bc + bw * 0.45, bc - bw * 0.45, -60.0, 40.0)
    ours_belt = _fit_width(_crop_deg(ours_matched, *box), 640)
    ref_belt = _fit_width(_crop_deg(ref, *box), 640)
    td_ours = _bandpass_rms(ours_belt)
    td_ref = _bandpass_rms(ref_belt)
    # JPEG-roundtrip(q75) control on ours.
    enc = cv2.imencode(".jpg", cv2.cvtColor((ours_belt * 255).astype(np.uint8),
                       cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])[1]
    ours_jpg = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    td_ours_jpg = _bandpass_rms(ours_jpg)

    # GRS-2: hero collar ripple at matched scale, in the hero's elliptical frame.
    heroes = [v for v in sim.vortices.vortices if v.kind == KIND_HERO]
    grs_ripple = float("nan")
    if heroes:
        h = heroes[0]
        hlat, hlon = np.degrees(h.lat), np.degrees(h.lon)
        hr_deg = np.degrees(h.r_core)
        asp = h.aspect
        half = max(3.2 * hr_deg * asp, 3.2 * hr_deg) + 1.0
        hbox = (hlat + half, hlat - half, hlon - half, hlon + half)
        crop = _crop_deg(ours_matched, *hbox)
        L = _lum(crop)
        cy = L.shape[0] / 2.0
        cx = L.shape[1] / 2.0
        rc_px = (hr_deg / (2 * half)) * L.shape[0]
        grs_ripple = ring_ripple_std(L, cx, cy, rc_px, asp)
        if save_tag:
            outp = Path(f"out/audit/p5/{save_tag}_grs.png")
            outp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(outp), cv2.cvtColor((np.clip(crop, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR))

    if save_tag:
        outp = Path(f"out/audit/p5/{save_tag}_belt.png")
        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), cv2.cvtColor((np.clip(ours_belt, 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR))

    sim._release_sim()
    return {
        "belt_center": round(float(bc), 1),
        "TD1_ours": round(td_ours, 5),
        "TD1_ref": round(td_ref, 5),
        "TD1_ratio": round(td_ours / td_ref, 4),
        "TD1_ours_jpg": round(td_ours_jpg, 5),
        "GRS2_ripple": round(grs_ripple, 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="jupiter_like")
    ap.add_argument("--width", type=int, default=8192)
    ap.add_argument("--aspect", type=float, default=None)
    ap.add_argument("--rim", type=float, default=None)
    ap.add_argument("--latitude", type=float, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    r = measure(args.preset, args.width, args.aspect, args.rim, args.latitude, args.tag)
    for k, v in r.items():
        print(f"{k:>16}: {v}")


if __name__ == "__main__":
    main()

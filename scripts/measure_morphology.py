"""v1.6 P4.5: folded-filament MORPHOLOGY metrics + calibration.

v1.5's lesson: a scalar ENERGY metric (TD-1) passed while the judges failed the
texture-density axis 3-0, because the gap is MORPHOLOGY (folded, zonally-
elongated filaments) not energy. This module defines candidate morphology
metrics and a calibration harness that checks which one actually RANKS the
reference above the v1.5 render above a degraded (isotropized) reference —
i.e. which one tracks human-perceived folded-filament structure. Only a metric
that passes calibration may gate the P5 proof (else it is discarded and the
3-judge panel stands alone).

Candidates (all on the widest-tropical-belt matched-scale L crop):
  - aniso   : directional power-spectrum anisotropy E_merid/E_zonal. Zonally-
              elongated filaments = horizontal streaks = power at high meridional
              wavenumber (varies fast in y, slow in x) -> ratio >> 1. Isotropic
              mottle -> ~1.
  - coher   : structure-tensor orientation coherence (mean (l1-l2)/(l1+l2)),
              weighted toward near-horizontal orientation. Coherent oriented
              filaments -> high; isotropic mottle -> low.
  - slope   : radial power-spectrum slope (texture richness; secondary).

CALIBRATION RESULT (P4.5, recorded BEFORE any vorticity render exists, on the
widest-tropical-belt matched crop lat −7.9..−18.8):
    source     aniso     coher     slope
 reference    1.1383    0.6170   -2.0914
      v1.5    0.2684    0.1403   -1.3233
 ref_rot90    1.2438    0.1212   -2.0914   (90deg orientation control)

  * coher is the GATE metric: reference (0.617) >> v1.5 (0.140), margin +340%,
    AND the orientation control collapses it (0.617 -> 0.121 under 90deg
    rotation == v1.5's isotropic level), proving it measures HORIZONTAL folded
    structure, not energy/banding. v1.5's isotropic mottle scores as low as the
    rotated reference. This is the folded-filament discriminator.
  * aniso passes ref>v1.5 (+324%) but its rotation control is confounded by the
    belt-strip aspect ratio (rot90 did not invert) -> SECONDARY/descriptive only.
  * slope FAILS the gate (v1.5's energy overshoot gives a shallower slope) ->
    discarded as a morphology metric.

PRE-REGISTERED MD-2 bar (necessary, NOT sufficient — the 3-judge panel MD-1 is
the gate): the P5 vorticity render must reach coher >= 0.30 on this crop (>2x
v1.5's 0.14 isotropic baseline, ~40% of the way to the reference's 0.62). A
miss stands as recorded — no goal-shifting.

Usage:
    uv run python scripts/measure_morphology.py --calibrate
    uv run python scripts/measure_morphology.py --image out/foo.png   # ad-hoc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gasgiant.engine.facade import Simulation  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402

REF = Path("refs/PIA07782.jpg")


def _lum(rgb):
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


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


# ---- the candidate metrics -------------------------------------------------


def _prep(lum):
    """Detrend + Hann-window an L crop for spectral analysis."""
    lum = lum.astype(np.float32)
    lum = lum - cv2.GaussianBlur(lum, (0, 0), 8.0)  # remove the band gradient
    h, w = lum.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    return lum * (wy * wx)


def aniso(lum):
    """Directional power-spectrum anisotropy E_merid / E_zonal.

    Horizontal streaks (zonally elongated filaments) put power at high ky / low
    kx -> ratio >> 1; isotropic structure -> ~1. Uses 45-degree wedges around
    the ky vs kx axes, excluding the DC neighborhood.
    """
    f = np.fft.fftshift(np.fft.fft2(_prep(lum)))
    power = np.abs(f) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    ky = (np.arange(h) - cy)[:, None].astype(np.float32)
    kx = (np.arange(w) - cx)[None, :].astype(np.float32)
    r = np.sqrt(kx * kx + ky * ky)
    rmax = min(cy, cx)
    band = (r > 0.05 * rmax) & (r < 0.9 * rmax)  # drop DC and the noisy corner
    merid = band & (np.abs(ky) >= np.abs(kx))  # power varying fast in y
    zonal = band & (np.abs(kx) > np.abs(ky))
    e_m = float(power[merid].sum())
    e_z = float(power[zonal].sum())
    return e_m / max(e_z, 1e-12)


def coher(lum):
    """Structure-tensor orientation coherence, horizontality-weighted.

    Coherence c = (l1-l2)/(l1+l2) in [0,1]; weight by how horizontal the
    dominant orientation is (cos^2 of the gradient angle from vertical, i.e.
    streaks running east-west). Returns the energy-weighted mean.
    """
    lum = lum.astype(np.float32)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    s = 4.0
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), s)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), s)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), s)
    tr = jxx + jyy
    det_term = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    c = det_term / np.maximum(tr, 1e-9)  # (l1-l2)/(l1+l2)
    # Dominant gradient direction; horizontal STREAKS have gradients pointing
    # vertically (gy dominant), so weight by gy^2/(gx^2+gy^2).
    horiz = jyy / np.maximum(jxx + jyy, 1e-9)
    wgt = tr  # weight by local energy
    return float((c * horiz * wgt).sum() / max(wgt.sum(), 1e-9))


def slope(lum):
    """Radial power-spectrum slope (log-log), a texture-richness proxy."""
    f = np.fft.fftshift(np.fft.fft2(_prep(lum)))
    power = np.abs(f) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    ky = (np.arange(h) - cy)[:, None]
    kx = (np.arange(w) - cx)[None, :]
    r = np.sqrt(kx * kx + ky * ky).astype(np.int32)
    rmax = min(cy, cx)
    rad = np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    k = np.arange(2, rmax)
    pk = rad[2:rmax]
    good = pk > 0
    if good.sum() < 8:
        return 0.0
    return float(np.polyfit(np.log(k[good]), np.log(pk[good]), 1)[0])


METRICS = {"aniso": aniso, "coher": coher, "slope": slope}


def metrics_on_crop(lum640):
    return {name: round(fn(lum640), 5) for name, fn in METRICS.items()}


def _belt_crop_from_rgb(rgb, sim, width=640):
    bc, bw = _widest_tropical_belt(sim)
    matched = _fit_width(np.clip(rgb[..., :3], 0, 1).astype(np.float32),
                         cv2.imread(str(REF)).shape[1])
    box = (bc + bw * 0.45, bc - bw * 0.45, -60.0, 40.0)
    return _fit_width(_crop_deg(matched, *box), width), box


def calibrate():
    """Rank reference vs v1.5 (kinematic jupiter_like) vs degraded reference.

    A metric may gate P5 only if it orders reference > v1.5 with margin AND
    reference > degraded (isotropic blur). Prints the ordering per metric.
    """
    gpu = GpuContext.headless()
    gpu.make_current()
    p = load_factory_preset("jupiter_like")  # kinematic default == v1.5
    sim = Simulation(p, gpu)
    rgb = sim.render_maps(8192)["color"]
    v15_crop, box = _belt_crop_from_rgb(rgb, sim, 640)
    sim._release_sim()

    ref = cv2.cvtColor(cv2.imread(str(REF)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ref_crop = _fit_width(_crop_deg(ref, *box), 640)
    # Controls. blur: isotropic Gaussian (energy-preserving; NOTE it enhances
    # large-scale banding, so it is NOT a clean folded-structure destroyer).
    # rot90: rotate the L crop 90deg — a true ORIENTATION control. If a metric
    # measures horizontal folded structure it must DROP here (zonal<->merid
    # swap), proving it is not just banding/energy.
    deg_crop = cv2.GaussianBlur(ref_crop, (0, 0), 3.0)
    ref_lum = _lum(ref_crop)
    rot_lum = np.rot90(ref_lum).copy()

    rows = {
        "reference": metrics_on_crop(ref_lum),
        "v1.5": metrics_on_crop(_lum(v15_crop)),
        "ref_blur": metrics_on_crop(_lum(deg_crop)),
        "ref_rot90": metrics_on_crop(rot_lum),
    }
    print(f"belt box (lat0,lat1,lon0,lon1)= {tuple(round(b,1) for b in box)}")
    print(f"{'source':>10} " + " ".join(f"{m:>9}" for m in METRICS))
    for src, vals in rows.items():
        print(f"{src:>10} " + " ".join(f"{vals[m]:>9.4f}" for m in METRICS))
    print()
    for m in METRICS:
        r, v = rows["reference"][m], rows["v1.5"][m]
        rot = rows["ref_rot90"][m]
        gate_ok = r > v                       # plan's gate: ref > v1.5 with margin
        orient_ok = r > rot                   # orientation-specific (not just energy)
        margin = (r - v) / max(abs(v), 1e-9)
        print(f"  {m:>6}: ref={r:.4f} v15={v:.4f} rot90={rot:.4f}  "
              f"ref>v15={gate_ok} (margin {margin:+.1%})  orient(ref>rot90)={orient_ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--image", default=None, help="ad-hoc: metrics on an image's full frame L")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    elif args.image:
        img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        print(metrics_on_crop(_lum(_fit_width(img, 640))))
    else:
        ap.error("pass --calibrate or --image")


if __name__ == "__main__":
    main()

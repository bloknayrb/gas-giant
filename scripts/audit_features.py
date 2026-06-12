"""Feature-by-feature audit harness (the v1.4 realism pass).

Renders jupiter_like once at high resolution and emits labeled crop PAIRS
(ours vs reference) into out/audit/, at two scales:

- matched: both sides at the SAME deg/px as the global reference map
  (structure / shape / size / spacing / contrast judgements).
- native: ours at render resolution; the reference CLOSE-UP is resampled
  DOWN to our km/px at that latitude (texture-presence judgements --
  judging our render against detail it cannot physically encode would
  bias every texture verdict toward "ours lacks detail").

Crops with no reference coverage are labeled ADVISORY: they can establish
presence/absence against the formations.md claim, never MATCH.

Feature locations on our side come from the live Simulation objects
(vortex registry, band layout, lane list, wave latitudes) -- never from
hardcoded positions. Reference boxes are hand-pinned constants below,
located by inspection; each carries a comment naming what it shows.

Usage:
  uv run python scripts/audit_features.py
  uv run python scripts/audit_features.py --preset jupiter_like --width 8192
  uv run python scripts/audit_features.py --extract-template out/template.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

# Jupiter equatorial circumference, km (2*pi*71492).
_JUPITER_CIRC_KM = 449_197.0

_REF_DIR = Path("refs")
_OUT_DIR = Path("out/audit")

# The global cylindrical map: matched-scale ground truth.
_REF_MAP = "PIA07782.jpg"

# Close-up references with their approximate ground scale. Derivations:
# - PIA21775 (Juno PJ7 GRS): the GRS major axis spans ~1060 px of the 1280 px
#   frame; the 2017 GRS measured ~16,000 km long -> ~15 km/px (oblique view,
#   scale varies across the frame; good to ~25%).
# - PIA21641 (Juno south polar cluster): the circumpolar cyclones (~6,500 km
#   diameter, JIRAM-measured 5,600-7,000 km) span ~100 px -> ~65 km/px.
_REF_CLOSEUPS = {
    "PIA21775.jpg": {"km_per_px": 15.0, "shows": "GRS interior, collar, surroundings"},
    "PIA21641.jpg": {"km_per_px": 65.0, "shows": "south polar cyclone cluster"},
}

# Hand-pinned boxes on the 1280x640 reference map, (x0, y0, x1, y1) px.
# lat = 90 - y * 0.28125 ; the comments say what each box shows.
_MAP_BOXES = {
    # The GRS: salmon oval ~22 S with collar and surroundings.
    "grs": (385, 348, 535, 432),
    # Folded-filament turbulence west of the GRS (the classic wake region).
    "grs_wake": (120, 338, 392, 405),
    # NEB interior: the broad rusty belt, folded filaments and barges.
    "neb_interior": (540, 232, 940, 292),
    # NEB south edge: blue-gray festoon hooks crossing into the bright EZ.
    "ez_festoons": (240, 282, 700, 340),
    # Dark cigar-shaped barges embedded in the NEB.
    "barges": (600, 232, 860, 268),
    # White oval line in the southern temperate region.
    "white_ovals": (545, 418, 950, 458),
    # Small white ovals near ~40 S (string-of-pearls analog latitude).
    "pearls": (90, 448, 700, 482),
    # Scalloped wave curls along the NTB flank.
    "kh_curls": (300, 188, 700, 226),
    # SEB north edge: band-boundary meander at planetary wavenumbers.
    "meander": (200, 338, 800, 362),
    # EZ interior: bright zone texture (festoon-free stretch).
    "zone_interior": (760, 295, 1140, 338),
    # Dark compact 5-um hot-spot holes between festoon roots.
    "hotspots": (240, 278, 720, 302),
    # Full-height strips for overall band-layout / global-balance judgement.
    "strip_a": (100, 0, 260, 640),
    "strip_b": (500, 0, 660, 640),
    "strip_c": (900, 0, 1060, 640),
}

_MAP_DEG_PER_PX = 360.0 / 1280.0


# ---------------------------------------------------------------- image utils

def _load_srgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"error: cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _save(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] == width:
        return img
    h = max(1, round(img.shape[0] * width / img.shape[1]))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _crop_deg(img: np.ndarray, lat0: float, lat1: float,
              lon0: float, lon1: float) -> np.ndarray:
    """Crop an equirect image by degree box (lat0 > lat1, lon in -180..180;
    boxes crossing the date line are handled by rolling)."""
    h, w = img.shape[:2]
    y0 = int(np.clip(round((90.0 - lat0) / 180.0 * h), 0, h - 1))
    y1 = int(np.clip(round((90.0 - lat1) / 180.0 * h), y0 + 1, h))
    x0f = (lon0 + 180.0) / 360.0 * w
    x1f = (lon1 + 180.0) / 360.0 * w
    if x1f <= x0f:
        x1f += w
    x0 = int(np.floor(x0f)) % w  # boxes past the date line -> wrap, not an empty slice
    span = max(1, int(round(x1f - x0f)))
    if x0 + span <= w:
        return img[y0:y1, x0:x0 + span]
    rolled = np.roll(img, -x0, axis=1)
    return rolled[y0:y1, :span]


def _polar_azimuthal(img: np.ndarray, south: bool, extent_deg: float = 42.0,
                     size: int = 900) -> np.ndarray:
    """Reproject a polar cap of an equirect image to azimuthal equidistant
    (what the polar references show; the equirect cap is uselessly stretched)."""
    h, w = img.shape[:2]
    c = (size - 1) / 2.0
    j, i = np.meshgrid(np.arange(size), np.arange(size))
    dx = (j - c) / c * extent_deg
    dy = (i - c) / c * extent_deg
    rho = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    lat = (-90.0 + rho) if south else (90.0 - rho)
    lon = np.degrees(theta)
    map_x = ((lon + 180.0) / 360.0 * w).astype(np.float32)
    map_y = ((90.0 - lat) / 180.0 * h).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    out[rho > extent_deg] = 0.0
    return out


def _km_per_px(width: int, lat_deg: float) -> float:
    return _JUPITER_CIRC_KM * np.cos(np.radians(lat_deg)) / width


def _equalize_to(img: np.ndarray, img_km_px: float, target_km_px: float) -> np.ndarray:
    """Downsample the finer image to the coarser scale (never upsample)."""
    if img_km_px >= target_km_px:
        return img
    return _fit_width(img, max(8, round(img.shape[1] * img_km_px / target_km_px)))


# ---------------------------------------------------------------- the audit

def _box_px(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def _build_crops(sim, ours: np.ndarray, args) -> list[dict]:
    """Emit every crop pair + manifest entries. `ours` is the full render."""
    from gasgiant.sim.profiles import select_wave_latitudes
    from gasgiant.sim.vortices import (
        KIND_BARGE,
        KIND_DEBRIS,
        KIND_HERO,
        KIND_OVAL,
        KIND_PEARL,
    )

    ref_map = _load_srgb(_REF_DIR / _REF_MAP)
    ours_matched = _fit_width(ours, ref_map.shape[1])
    _save(_OUT_DIR / "ours_full.png", ours)
    _save(_OUT_DIR / "ours_matched.png", ours_matched)

    entries: list[dict] = []
    idx = 0

    def emit(feature: str, claim: str, scale: str,
             ours_crop: np.ndarray | None, ref_crop: np.ndarray | None,
             **extra) -> None:
        nonlocal idx
        entry = {"id": idx, "feature": feature, "claim": claim, "scale": scale}
        entry.update(extra)
        if ours_crop is not None and ours_crop.size:
            f = _OUT_DIR / f"{idx:02d}_{feature}_ours.png"
            _save(f, ours_crop)
            entry["ours_file"] = f.name
        if ref_crop is not None and ref_crop.size:
            f = _OUT_DIR / f"{idx:02d}_{feature}_ref.png"
            _save(f, ref_crop)
            entry["ref_file"] = f.name
        entries.append(entry)
        idx += 1

    def by_kind(kind: float):
        return sorted((v for v in sim.vortices.vortices if v.kind == kind),
                      key=lambda v: -v.r_core)

    def deg(x: float) -> float:
        return float(np.degrees(x))

    def matched_ours(lat0, lat1, lon0, lon1) -> np.ndarray:
        return _crop_deg(ours_matched, lat0, lat1, lon0, lon1)

    # -- band layout / global structure (matched) ---------------------------
    for name in ("strip_a", "strip_b", "strip_c"):
        lon0 = {"strip_a": -150.0, "strip_b": -30.0, "strip_c": 90.0}[name]
        emit(
            f"band_layout_{name}",
            "Band count / width distribution / EZ breadth / zone-belt "
            "alternation match the reference's overall structure.",
            "matched",
            matched_ours(90.0, -90.0, lon0, lon0 + 45.0),
            _box_px(ref_map, _MAP_BOXES[name]),
        )

    # -- widest belt / zone interiors (matched) ------------------------------
    edges = np.degrees(sim.bands.edges.astype(np.float64))
    values = sim.bands.values
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[:-1] - edges[1:]
    is_belt = values < np.median(values)
    lowmid = np.abs(centers) < 50.0

    def widest(mask) -> int:
        cand = np.where(mask & lowmid)[0]
        return int(cand[np.argmax(widths[cand])])

    jb = widest(is_belt)
    emit(
        "belt_interior",
        "Belts: darker brown/red-brown, visibly more turbulent than zones; "
        "wall-to-wall folded filament chaos.",
        "matched",
        matched_ours(centers[jb] + widths[jb] * 0.45, centers[jb] - widths[jb] * 0.45,
                     -60.0, 40.0),
        _box_px(ref_map, _MAP_BOXES["neb_interior"]),
    )
    jz = widest(~is_belt)
    emit(
        "zone_interior",
        "Zones: bright cream/white, smooth and hazy; quiet-interior "
        "granulation, not fuzz.",
        "matched",
        matched_ours(centers[jz] + widths[jz] * 0.45, centers[jz] - widths[jz] * 0.45,
                     -60.0, 40.0),
        _box_px(ref_map, _MAP_BOXES["zone_interior"]),
    )

    # -- thin dark lanes (matched; verdict pre-confirmed by user) ------------
    if sim.lanes:
        lane_lat = deg(sim.lanes[0][0])
        emit(
            "lanes",
            "Thin dark lane lines at jet cores. (Reference shows none this "
            "clean -- verdict pre-confirmed: not present in PIA07782.)",
            "matched",
            matched_ours(lane_lat + 4.0, lane_lat - 4.0, -60.0, 60.0),
            _box_px(ref_map, _MAP_BOXES["strip_b"]),
            lane_count=len(sim.lanes),
        )

    # -- GRS (matched + native) ----------------------------------------------
    heroes = by_kind(KIND_HERO)
    if heroes:
        h = heroes[0]
        hlat, hlon, hr = deg(h.lat), deg(h.lon), deg(h.r_core)
        emit(
            "grs",
            "GRS-class hero: salmon/red oval, bright collar, calm center, "
            "deflected surrounding flow; size relative to band width.",
            "matched",
            matched_ours(hlat + 2.6 * hr, hlat - 2.6 * hr,
                         hlon - 3.2 * hr, hlon + 3.2 * hr),
            _box_px(ref_map, _MAP_BOXES["grs"]),
            hero_lat=hlat, hero_lon=hlon, r_core_deg=hr,
        )
        wd = 1.0 if h.wake_dir >= 0 else -1.0
        lon0, lon1 = sorted((hlon + wd * 1.6 * hr, hlon + wd * 7.5 * hr))
        emit(
            "grs_wake",
            "Turbulent wake: chaotic folded bright filaments trailing "
            "downstream of the hero.",
            "matched",
            matched_ours(hlat + 2.2 * hr, hlat - 2.2 * hr, lon0, lon1),
            _box_px(ref_map, _MAP_BOXES["grs_wake"]),
        )
        # Native: GRS interior spiral vs the Juno close-up, ref downsampled
        # to our ground scale at the hero's latitude.
        ours_km = _km_per_px(ours.shape[1], hlat)
        closeup = _load_srgb(_REF_DIR / "PIA21775.jpg")
        ref_km = _REF_CLOSEUPS["PIA21775.jpg"]["km_per_px"]
        emit(
            "grs_spiral",
            "GRS internals: tightly wound thin spiral lanes peaking "
            "mid-radius around a calm bright center; winding follows the "
            "storm's rotation sense.",
            "native",
            _crop_deg(ours, hlat + 2.0 * hr, hlat - 2.0 * hr,
                      hlon - 2.4 * hr, hlon + 2.4 * hr),
            _equalize_to(closeup, ref_km, ours_km),
            ours_km_per_px=round(ours_km, 1), ref_km_per_px=ref_km,
        )

    # -- ovals / barges / pearls (matched) -----------------------------------
    ovals = [v for v in by_kind(KIND_OVAL) if v.r_core > 0.015][:3]
    for n, v in enumerate(ovals):
        vlat, vlon, vr = deg(v.lat), deg(v.lon), deg(v.r_core)
        emit(
            f"white_oval_{n}",
            "White ovals: compact bright anticyclones, often dark-rimmed, "
            "in same-latitude lines. (KIND_OVAL also covers the small-storm "
            "field and merge products -- these are the largest by r_core.)",
            "matched",
            matched_ours(vlat + 3.5 * vr, vlat - 3.5 * vr,
                         vlon - 5.0 * vr, vlon + 5.0 * vr),
            _box_px(ref_map, _MAP_BOXES["white_ovals"]),
        )
    barges = by_kind(KIND_BARGE)[:2]
    for n, v in enumerate(barges):
        vlat, vlon, vr = deg(v.lat), deg(v.lon), deg(v.r_core)
        emit(
            f"barge_{n}",
            "Brown barges: dark cigar-shaped cyclones stretched along the "
            "belt by the jets.",
            "matched",
            matched_ours(vlat + 4.0 * vr, vlat - 4.0 * vr,
                         vlon - 7.0 * vr, vlon + 7.0 * vr),
            _box_px(ref_map, _MAP_BOXES["barges"]),
        )
    pearls = by_kind(KIND_PEARL)
    if pearls:
        plat = deg(float(np.median([v.lat for v in pearls])))
        emit(
            "pearls",
            "String of pearls: 6-9 similar small white ovals evenly spaced "
            "around one latitude circle.",
            "matched",
            matched_ours(plat + 4.0, plat - 4.0, -180.0, -30.0),
            _box_px(ref_map, _MAP_BOXES["pearls"]),
            pearl_count=len(pearls), pearl_lat=plat,
        )

    # -- merger debris (matched if alive, else recorded absent) --------------
    debris = [v for v in sim.vortices.vortices if v.kind == KIND_DEBRIS and v.ttl > 0]
    if debris:
        v = debris[0]
        vlat, vlon = deg(v.lat), deg(v.lon)
        emit(
            "merge_debris",
            "Merger debris: transient bright turbulent collar folded into "
            "filaments at a recent merge site.",
            "advisory",
            matched_ours(vlat + 5.0, vlat - 5.0, vlon - 8.0, vlon + 8.0),
            None,
        )
    else:
        emit("merge_debris",
             "No live debris collar at run end (ttl expired) -- noted, "
             "feature exercised by tests not stills.", "advisory", None, None)

    # -- waves (matched) ------------------------------------------------------
    festoon_lat, _ribbon = select_wave_latitudes(sim.bands, sim.profiles)
    flat = deg(festoon_lat)
    emit(
        "festoons",
        "Festoons: blue-gray streamers hooking from the equatorial belt "
        "edge across the bright EZ, quasi-periodic.",
        "matched",
        matched_ours(flat + 7.0, flat - 9.0, -90.0, 30.0),
        _box_px(ref_map, _MAP_BOXES["ez_festoons"]),
        festoon_lat=flat,
    )
    emit(
        "hotspots",
        "5-um hot spots: compact cloud-free dark holes between festoons at "
        "the wave troughs.",
        "matched",
        matched_ours(flat + 4.0, flat - 4.0, -90.0, 30.0),
        _box_px(ref_map, _MAP_BOXES["hotspots"]),
    )
    # KH curls at the strongest interior jet flank.
    u = np.asarray(sim.profiles.u, dtype=np.float64)
    n_u = len(u)
    edge_lats = edges[1:-1]
    edge_idx = np.clip(((90.0 - edge_lats) / 180.0 * (n_u - 1)).astype(int), 0, n_u - 1)
    jk = int(np.argmax(np.abs(u[edge_idx])))
    klat = float(edge_lats[jk])
    emit(
        "kh_billows",
        "Kelvin-Helmholtz billows: scalloped wave curls along high-shear "
        "band boundaries.",
        "matched",
        matched_ours(klat + 3.0, klat - 3.0, -90.0, 0.0),
        _box_px(ref_map, _MAP_BOXES["kh_curls"]),
        kh_lat=klat,
    )
    emit(
        "meander",
        "Band-boundary meander: edges wander at planetary wavenumbers 5-20, "
        "never parallel circles.",
        "matched",
        matched_ours(klat + 3.0, klat - 3.0, 0.0, 150.0),
        _box_px(ref_map, _MAP_BOXES["meander"]),
    )

    # -- texture presence (native / advisory) --------------------------------
    belt_mid = centers[jb]
    zone_mid = centers[jz]
    for feature, lat_c, claim in (
        ("texture_belt_filaments", belt_mid,
         "Fine filaments: flow-stretched and folded streaks riding the "
         "jets; intermittent -- violent patches abut calm laminar runs."),
        ("texture_zone_cells", zone_mid,
         "Convective cell fields: closed-cell popcorn granulation in quiet "
         "zone interiors."),
        ("texture_striation", belt_mid,
         "Striation: fine along-flow thread texture inside belts."),
    ):
        emit(
            feature, claim, "advisory",
            _crop_deg(ours, lat_c + 6.0, lat_c - 6.0, 10.0, 34.0),
            None,
            note="ours-only at native scale; no reference at this ground "
                 "resolution covers a generic band interior -- presence/"
                 "absence vs the catalog claim, cannot MATCH.",
        )
    # Intermittency needs a long strip to show busy/calm alternation.
    emit(
        "texture_intermittency",
        "Texture density alternates along the belt: busy folded patches "
        "abut calm runs (flow-structured, not vignetting).",
        "advisory",
        _crop_deg(ours, belt_mid + 5.0, belt_mid - 5.0, -90.0, 0.0),
        None,
    )

    # -- poles (native-equalized) ---------------------------------------------
    ours_km_polar = _JUPITER_CIRC_KM / 2.0 / ours.shape[0]  # meridional km/px
    polar_ref = _load_srgb(_REF_DIR / "PIA21641.jpg")
    ref_km = _REF_CLOSEUPS["PIA21641.jpg"]["km_per_px"]
    south_ae = _polar_azimuthal(ours, south=True)
    emit(
        "polar_south",
        "South polar cyclone cluster: central cyclone ringed by circumpolar "
        "cyclones at polygon vertices, fluffy spiral arms; deep structural "
        "blue (reference is enhanced color -- flavor, not colorimetry).",
        "native",
        _equalize_to(south_ae, ours_km_polar, ref_km),
        polar_ref,
        ours_km_per_px=round(ours_km_polar, 1), ref_km_per_px=ref_km,
    )
    emit(
        "polar_north",
        "North polar cyclone cluster (8 ring cyclones; no matching-scale "
        "reference in the set -- judged against the south + catalog claim).",
        "advisory",
        _polar_azimuthal(ours, south=False),
        None,
    )

    return entries


def run_audit(args) -> None:
    from gasgiant.engine import Simulation
    from gasgiant.params.presets import resolve_preset

    params = resolve_preset(args.preset)
    sim = Simulation(params)
    print(f"rendering {args.preset} at {args.width} (seed {params.seed}) ...")
    color = np.clip(sim.render_maps(args.width)["color"][..., :3], 0.0, 1.0)

    entries = _build_crops(sim, color, args)

    ref_meta = {}
    for name in [_REF_MAP, *_REF_CLOSEUPS]:
        p = _REF_DIR / name
        img = _load_srgb(p)
        ref_meta[name] = {
            "sha1": _sha1(p),
            "size": [img.shape[1], img.shape[0]],
            **_REF_CLOSEUPS.get(name, {}),
        }
    manifest = {
        "preset": args.preset,
        "seed": params.seed,
        "render_width": args.width,
        "map_deg_per_px": _MAP_DEG_PER_PX,
        "judging_notes": [
            "Ours is a clean render; the reference map is a JPEG mosaic with "
            "seams and block artifacts. Before any 'ours too sharp/clean' "
            "verdict, sanity-check against a blur+JPEG roundtrip of our crop.",
            "Color verdicts only for features wider than ~1.1 deg (the "
            "reference map is 4:2:0 -- chroma is native at half width); "
            "smaller features' color is graded LIMIT-by-reference.",
            "Pre-template rule: grade feature color as offset from local "
            "band surround, not absolute (band layouts not yet aligned).",
        ],
        "references": ref_meta,
        "crops": entries,
    }
    out = _OUT_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"{len(entries)} crop entries -> {out}")


def run_extract(args) -> None:
    from gasgiant.palette.reference import band_template_arrays
    from gasgiant.sim.bands import BELT_VALUE, ZONE_VALUE

    img = _load_srgb(_REF_DIR / _REF_MAP)
    arrays = band_template_arrays(img, max_bands=args.max_bands)
    is_zone = arrays["is_zone"]
    lum = arrays["band_lum"]

    # Map band luminance into T0 color-index space with identity-guaranteed
    # ranges: zones occupy the upper half-range, belts the lower, so the
    # consumers' `values < median(values)` mask reproduces the extracted
    # classes exactly (and alternates, since classes alternate by
    # construction -- threshold crossings).
    values = np.empty(len(lum))
    heights = np.empty(len(lum))
    for mask, v_lo, v_hi, h_lo, h_hi in (
        (is_zone, 0.56, ZONE_VALUE, 0.55, 0.75),
        (~is_zone, BELT_VALUE, 0.52, 0.30, 0.50),
    ):
        sub = lum[mask]
        span = sub.max() - sub.min()
        norm = (sub - sub.min()) / span if span > 1e-9 else np.full(sub.shape, 0.5)
        values[mask] = v_lo + norm * (v_hi - v_lo)
        heights[mask] = h_lo + norm * (h_hi - h_lo)

    template = {
        "edges_deg": [round(float(e), 2) for e in arrays["edges_deg"]],
        "values": [round(float(v), 4) for v in values],
        "heights": [round(float(h), 4) for h in heights],
    }
    out = Path(args.extract_template)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, indent=2))
    kinds = "".join("Z" if z else "B" for z in is_zone)
    print(f"{len(values)} bands ({kinds}) -> {out}")
    for e0, e1, z, v in zip(template["edges_deg"][:-1], template["edges_deg"][1:],
                            is_zone, template["values"], strict=False):
        print(f"  {e0:7.2f} .. {e1:7.2f}  {'zone' if z else 'belt'}  value {v:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="jupiter_like")
    ap.add_argument("--width", type=int, default=8192)
    ap.add_argument("--extract-template", metavar="PATH",
                    help="extract a band template from the reference map "
                         "instead of running the audit render")
    ap.add_argument("--max-bands", type=int, default=18)
    args = ap.parse_args()
    if args.extract_template:
        run_extract(args)
    else:
        run_audit(args)


if __name__ == "__main__":
    main()

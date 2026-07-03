"""swirl_gate.py — the frozen acceptance gate for the "stable jets vs. the
storm-driven oversized swirl" milestone (see plan compressed-spinning-bentley).

Renders the canonical develop config and computes six metrics from the
absolute-vorticity field q (read back from the equirect omega state) plus the
color map.  Five are CO-GATED (all must hold to PASS); m2 is reported only.

  1. m1 swirl     <= M1_MAX    largest eddy-blob meridional extent, in band
                               widths.  Guards the "one oversized swirl eats a
                               band" failure.
  2. m2 meander   (reported)   jet-core meridional wander / band width.  Bad and
                               clean ranges OVERLAP across seeds, so no threshold
                               separates it — printed for forensics, NOT gated.
  3. m3 continuity >= M3_MIN   fraction of longitude where the jet-core vorticity
                               sign is unbroken.  Guards a torn transport barrier.
  4. m4 texture  in M4_RANGE   belt high-frequency energy / first-panel baseline.
                               Bidirectional — guards laminar WASHOUT.
  5. m5 hero      >= M5_MIN    hero color contrast (core vs annulus luminance).
                               Guards over-damping that erases the pinned hero.
  6. m6 medium   in M6_RANGE   medium-wavenumber eddy energy / first-panel
                               baseline.  The FLOOR is the only metric that fails
                               an over-flattened/sterile result (the others reward
                               laminarity or see only stamped speckle).

m4 and m6 are referenced to the FIRST swept config (expected drag=0), so the
sweep must include a no-drag panel as the texture/medium baseline.

Usage:
    python scripts/swirl_gate.py                      # default drag sweep
    python scripts/swirl_gate.py --drags 0,0.08,0.15 --width 1536
    python scripts/swirl_gate.py --calibrate          # print raw metrics only

This is a kept deliverable (not a throwaway): it is the crux gate every stage of
the milestone is judged against.  Thresholds are frozen in the M*_ constants
below after one calibration run that cleanly separates the visually swirl-free
panels (drag 0.08/0.15) from the swirl panel (drag 0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import resolve_preset

# --------------------------------------------------------------------------- #
# Frozen acceptance thresholds (calibrated 2026-06-25 across seeds 4201/7/99/   #
# 2718, W=1536).  Per-seed cross-calibration (bad = drag 0; clean = a passing   #
# drag) — values are min..max over the 4 seeds:                                 #
#   metric   bad (drag 0)    clean           gate           separates?          #
#   m1 swirl  1.63 .. 3.87    1.24 .. 1.94    <= 2.30        upper guard         #
#   m3 cont   0.50 .. 0.52    0.62 .. 0.79    >= 0.57        YES (no overlap)    #
#   m4 tex    1.00            1.25 .. 1.37    in [0.70,1.60] washout guard       #
#   m5 hero   0.43 .. 0.61    0.89 .. 1.11    in [0.70,1.60] YES (no overlap)    #
# m2 (meander) OVERLAPS bad/clean across seeds (bad 0.35..0.42, clean           #
# 0.26..0.37) — no threshold separates it, so it is REPORTED but NOT gated.     #
# The robust failure detectors are m3 + m5 (no overlap) backed by the m1 upper  #
# guard; every drag-0 panel fails at least one.  The FIRST swept config is the  #
# m4 texture baseline, so always sweep a reference (drag 0) panel first.        #
# --------------------------------------------------------------------------- #
M1_MAX = 2.30           # max eddy-blob latitude extent (band widths)
M3_MIN = 0.57           # band-continuity fraction (robust, no bad/clean overlap)
# Per-preset M3_MIN overrides — W9 re-baseline, 2026-07-03 (user decision on the
# 2026-07-02 comprehensive-review Top-10 #10 investigation, see
# docs/reviews/2026-07-03-gate-rebaseline-addendum.md): the 0.57 floor was
# calibrated on the gas_giant_warm develop config. jupiter_vorticity's narrower
# jets measured m3 = 0.51 at the very commit that froze the threshold (bbc1b9e,
# PR #8) and at every commit since (8201626 PR #9, 9c73b06 PR #10, 800f3dc
# master) — never drifted, never passed. PR #8 knowingly shipped it anyway
# ("guard-not-oracle", memory/preset-modernization.md); this dict encodes that
# accepted per-preset exception: 0.46 ≈ 10% below the stable measured 0.51.
# gas_giant_warm (and every preset not listed) keeps the calibrated 0.57.
M3_MIN_PER_PRESET = {"jupiter_vorticity": 0.46}
M4_RANGE = (0.70, 1.60)  # belt high-freq texture vs first-panel baseline
M5_MIN = 0.22           # hero color contrast (bold hero ~0.30+, lost hero ~0.16)
# m6: medium-wavenumber eddy energy vs first-panel baseline. TWO-SIDED — the
# FLOOR fails an over-flattened/sterile result (the failure all of m1/m3/m4/m5
# are blind to, because m3 rewards laminarity and m4 only sees stamped speckle);
# the ceiling guards injection-spam. Calibrated below once measured.
M6_RANGE = (0.45, 1.20)
# m2 is reported only (its bad/clean ranges overlap across seeds, so no threshold
# separates it) — see metric_meander / the printed table; it is never gated.

OUT = "C:/Users/blokn/Documents/Github/gas-giant/_diag"

# Vorticity magnitude percentile that defines a "strong" (coherent) region.
_STRONG_PCTILE = 85.0
# Half-window (in band widths) for the jet-core meander search.
_MEANDER_WIN = 0.6


# --------------------------------------------------------------------------- #
# Canonical develop config (matches scripts/seed_test.py / drag_test.py so the #
# gate scores the SAME field the _diag probes used).                          #
# --------------------------------------------------------------------------- #
def build_cfg(seed: int, drag: float, width: int, raw: bool = False,
              inject: float = 0.0, inject_mask: str = "belts", eddy_drag: float = 0.0,
              psi_drag: float = 0.0, preset: str = "gas_giant_warm",
              sweep_axis: str = "drag"):
    """raw=True is SHIP-CONFIG MODE: sweep the chosen drag axis on the UNMODIFIED
    `preset` (its own dev_steps / jets / storms / inject / L_d / hero), overriding
    ONLY that one drag param so the preset's real shipping dynamics are what gets
    gated.  `sweep_axis` selects which drag the swept value drives ("drag" =
    vort_drag, "eddy" = vort_eddy_drag, "psi" = vort_psi_drag); the swept value is
    set explicitly (incl. 0) so the baseline panel can force a baked drag OFF.

    raw=False applies the L_d develop config the _diag probes used (deformation_
    radius 0.18, inject 0, dev_steps 1256) on top of gas_giant_warm.  The two are
    genuinely different dynamics — see plan.  inject/inject_mask let the develop
    config carry broadband eddy injection; eddy_drag sets vort_eddy_drag (Gate 1)."""
    p = resolve_preset(preset).model_copy(update={"seed": seed})
    if raw:
        p.sim = p.sim.model_copy(update={"resolution": width, "dt_scale": 1.0})
        if sweep_axis == "psi":
            over = {"vort_psi_drag": psi_drag}
        elif sweep_axis == "eddy":
            over = {"vort_eddy_drag": eddy_drag}
        else:
            over = {"vort_drag": drag}
        p.solver = p.solver.model_copy(update=over)
        return p
    p.sim = p.sim.model_copy(update={"resolution": width, "dev_steps": 1256, "dt_scale": 1.0})
    solver_over = {
        "type": "vorticity", "poisson_iters": 48, "sor_omega": 1.7,
        "deformation_radius": 0.18, "vort_relax_tau": 600.0, "vort_hypervisc": 0.6,
        "coriolis_f0": 3.0, "vort_inject": inject, "vort_inject_scale": 2.5,
        "vort_inject_mask": inject_mask, "vort_drag": drag}
    if eddy_drag > 0.0:
        solver_over["vort_eddy_drag"] = eddy_drag
    if psi_drag > 0.0:
        solver_over["vort_psi_drag"] = psi_drag
    p.solver = p.solver.model_copy(update=solver_over)
    p.jets = p.jets.model_copy(update={
        "strength": 0.733, "equatorial_speed": 1.693, "equatorial_width": 0.194,
        "polar_decay": 0.648})
    p.waves = p.waves.model_copy(update={
        "festoon_strength": 2.0, "festoon_wavenumber": 20, "hotspot_depth": 1.0,
        "ribbon_strength": 2.0, "ribbon_wavenumber": 30})
    p.storms = p.storms.model_copy(update={
        "hero_count": 1, "hero_radius": 0.12, "hero_strength": 1.2, "hero_solid_core": 0.85,
        "hero_mottle": 0.35, "hero_tint_var": 0.35, "hero_aspect": 2.2, "rim_contrast": 1.3,
        "hero_latitude": -22.5, "oval_density": 3.0, "barge_density": 2.989, "pearls_count": 14,
        "wake_turbulence": 1.593, "small_density": 3.0, "stamp_contrast": 2.0,
        "merge_rate": 0.219, "merge_debris": 2.0})
    return p


# --------------------------------------------------------------------------- #
# Render: returns the color image, the relative-vorticity field, and the band/ #
# hero geometry the metrics need.                                              #
# --------------------------------------------------------------------------- #
class Field:
    def __init__(self, rgb, omega_rel, lat_row, hero_col, hero_row,
                 hero_rpix, jet_lats, band_width, belt_rows):
        self.rgb = rgb                  # (H, W, 3) uint8
        self.omega_rel = omega_rel      # (H, W) float32, q - f, hero-centered roll
        self.lat_row = lat_row          # (H,) latitude per row (rad)
        self.hero_col = hero_col        # int, hero column AFTER the centering roll
        self.hero_row = hero_row        # int
        self.hero_rpix = hero_rpix      # float, hero core radius in pixels
        self.jet_lats = jet_lats        # (K,) interior band-edge (jet core) lats
        self.band_width = band_width    # float, median band width (rad)
        self.belt_rows = belt_rows      # (H,) bool


def render(gpu, p, width) -> Field:
    sim = Simulation(p, gpu)
    try:
        rgb = (np.clip(sim.render_maps(width)["color"][:, :, :3], 0, 1) * 255).astype(np.uint8)
        q = sim.gpu.read_texture(sim.solver._omega_state.cur)[..., 0].astype(np.float32)
        H, W = q.shape

        lat_row = np.pi / 2.0 - (np.arange(H) + 0.5) / H * np.pi
        f0 = p.solver.coriolis_f0
        omega_rel = q - (f0 * np.sin(lat_row))[:, None]

        heroes = sim.vortices.heroes()
        hero = heroes[0]
        hero_col0 = int(((hero.lon / (2.0 * np.pi)) + 0.5) * W) % W
        hero_row = int(np.clip((np.pi / 2.0 - hero.lat) / np.pi * H, 0, H - 1))
        hero_rpix = hero.r_core / (np.pi / H)

        # Center the hero so the seam sits at its antipode (low-activity), making
        # connected-component areas robust to the ±180 deg wrap.
        roll = W // 2 - hero_col0
        omega_rel = np.roll(omega_rel, roll, axis=1)
        rgb = np.roll(rgb, roll, axis=1)
        hero_col = W // 2

        edges = sim.bands.edges.astype(np.float64)
        interior = edges[1:-1]
        jet_lats = interior[np.abs(interior) < np.radians(60.0)]  # equirect-owned band
        widths = np.abs(np.diff(edges))
        band_width = float(np.median(widths))

        belt = sim.profiles.belt_mask
        plat = sim.profiles.lat  # descending
        belt_row = np.interp(lat_row, plat[::-1], belt[::-1]) > 0.5

        return Field(rgb, omega_rel, lat_row, hero_col, hero_row, hero_rpix,
                     jet_lats, band_width, belt_row)
    finally:
        sim._release_sim()


# --------------------------------------------------------------------------- #
# Metric helpers                                                               #
# --------------------------------------------------------------------------- #
def _cos_weight(lat_row, W):
    return np.maximum(np.cos(lat_row), 1e-3)[:, None] * np.ones((1, W))


def _blobs(mask_signed, weight):
    """Connected components of a single-signed strong-vorticity mask.
    Returns (labels, area_w[label], stats) where stats is cv2's per-label table
    (CC_STAT_AREA / CC_STAT_HEIGHT etc.)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_signed.astype(np.uint8), connectivity=8)
    area_w = np.bincount(labels.ravel(), weights=weight.ravel(), minlength=n)
    return labels, area_w, stats


def metric_blobs(field: Field):
    """Returns (hero_latext_bw, max_latext_bw): the meridional extent of the hero
    eddy blob and of the LARGEST eddy blob, both in band-width units.

    Two design choices make this robust:

    * Blobs are detected in the EDDY field omega_rel - <omega_rel>_x (per-row
      zonal mean removed).  A clean zonal jet is a single-signed vorticity strip
      spanning ALL longitudes; counting it as a "blob" would fire on exactly the
      laminar bands we want.  Removing the zonal mean leaves only localized
      eddies (the swirl, hero, storms).
    * Size is measured as LATITUDE EXTENT in band widths, not pixel area, and
      never referenced to the (pathological) no-drag panel.  The oversized swirl
      spans several bands meridionally; the hero spans ~1; a near-laminar belt
      eddy spans ~0.  This is a clean absolute scale.

    The hero is found WITHOUT assuming its sign or trusting its exact center
    pixel (the solid-core hero can dip below threshold dead-center): dominant
    strong sign inside a window, then the blob with the largest footprint there.
    """
    H, W = field.omega_rel.shape
    weight = _cos_weight(field.lat_row, W)
    eddy = field.omega_rel - field.omega_rel.mean(axis=1, keepdims=True)
    strong = np.percentile(np.abs(eddy), _STRONG_PCTILE)
    pos = eddy > strong
    neg = eddy < -strong

    dlat = np.pi / H
    bw_px = max(field.band_width / dlat, 1.0)            # band width in rows
    min_px = 0.1 * np.pi * field.hero_rpix ** 2          # ignore specks

    # Hero coherence (replaces a connected-component extent, which fragments
    # under heavy injection): how strongly single-signed the hero footprint is,
    # measured LOCALLY in the eddy field within a disk around the known hero
    # centre, normalised by the strong-eddy scale.  A coherent hero -> high;
    # a shredded or over-damped hero -> ~0.  Local => robust to surrounding
    # injected eddies.
    # A vortex's vorticity is DIPOLAR (single-signed core + opposite-signed ring),
    # so coherence must be measured over the tight CORE only (a wider disk cancels
    # core against ring). Use a 0.7*r_core disk where a solid-body hero is single-
    # signed; coherence = mean signed eddy there / strong scale.  Local => robust
    # to surrounding injection.
    rr = int(max(0.9 * field.hero_rpix, 4))
    r0, r1 = max(field.hero_row - rr, 0), min(field.hero_row + rr + 1, H)
    c0, c1 = max(field.hero_col - rr, 0), min(field.hero_col + rr + 1, W)
    ly = np.arange(r0, r1)[:, None] - field.hero_row
    lx = np.arange(c0, c1)[None, :] - field.hero_col
    disk = (ly * ly + lx * lx) <= (0.7 * field.hero_rpix) ** 2
    vals = eddy[r0:r1, c0:c1][disk]
    if vals.size and strong > 0:
        dom = 1.0 if vals.mean() >= 0 else -1.0
        hero_coh = float((vals * dom).mean()) / strong
    else:
        hero_coh = 0.0

    max_latext = 0.0
    for mask in (pos, neg):
        _, _, st = _blobs(mask, weight)
        for lbl in range(1, st.shape[0]):
            if st[lbl, cv2.CC_STAT_AREA] < min_px:
                continue
            max_latext = max(max_latext, float(st[lbl, cv2.CC_STAT_HEIGHT]) / bw_px)
    return hero_coh, max_latext


def metric_meander(field: Field):
    """Metric 2: jet-core meridional wander / band width (max over cores)."""
    H, W = field.omega_rel.shape
    dlat = np.pi / H
    win = max(2, int(_MEANDER_WIN * field.band_width / dlat))
    amp = 0.0
    for jl in field.jet_lats:
        r0 = int(np.clip((np.pi / 2.0 - jl) / np.pi * H, win, H - win - 1))
        band = np.abs(field.omega_rel[r0 - win:r0 + win + 1, :])  # (2win+1, W)
        peak_row = np.argmax(band, axis=0)  # per-longitude row of max |omega|
        wander = float(np.std(peak_row)) * dlat
        amp = max(amp, wander / field.band_width)
    return amp


def metric_continuity(field: Field):
    """Metric 3: fraction of longitude where the jet-core vorticity sign is the
    dominant sign (min over cores)."""
    H, W = field.omega_rel.shape
    worst = 1.0
    for jl in field.jet_lats:
        r0 = int(np.clip((np.pi / 2.0 - jl) / np.pi * H, 0, H - 1))
        row = field.omega_rel[r0, :]
        s = np.sign(row)
        dom = 1.0 if s.sum() >= 0 else -1.0
        frac = float(np.mean(s == dom))
        worst = min(worst, frac)
    return worst


def metric_hero_color(field: Field):
    """Metric 5: hero readability measured in COLOR (the hero is a stamped bright
    oval, only loosely tied to the q field — so a q-based coherence cannot see it).
    Luminance contrast of the hero core disk vs its surrounding annulus, in units
    of the global luminance spread.  Present/bold hero -> high; washed-out or
    over-damped hero -> ~0."""
    H, W = field.omega_rel.shape
    lum = field.rgb.astype(np.float32).mean(axis=2)
    rp = field.hero_rpix
    rr = int(max(2.4 * rp, 8))
    r0, r1 = max(field.hero_row - rr, 0), min(field.hero_row + rr + 1, H)
    c0, c1 = max(field.hero_col - rr, 0), min(field.hero_col + rr + 1, W)
    ly = np.arange(r0, r1)[:, None] - field.hero_row
    lx = np.arange(c0, c1)[None, :] - field.hero_col
    d2 = ly * ly + lx * lx
    patch = lum[r0:r1, c0:c1]
    core = patch[d2 <= (1.0 * rp) ** 2]
    ann = patch[(d2 > (1.3 * rp) ** 2) & (d2 <= (2.2 * rp) ** 2)]
    if core.size == 0 or ann.size == 0:
        return 0.0
    return float(abs(core.mean() - ann.mean()) / (lum.std() + 1e-6))


def metric_medium(field: Field):
    """Metric 6 RAW: medium-wavenumber eddy energy — the structure a uniform
    eddy-drag flattens (band-edge waviness, mid-scale vortices, festoons). A
    difference-of-Gaussians band-pass of the eddy field omega_rel - <omega_rel>_x
    keeps ~0.5-3 band-width scales (rejects the gravest swirl as low-k and the
    fine stamped speckle as high-k), cos-weighted over the equirect-owned band.
    The ratio to the first (reference) panel is applied in main."""
    H, W = field.omega_rel.shape
    eddy = (field.omega_rel - field.omega_rel.mean(axis=1, keepdims=True)).astype(np.float32)
    bw_px = max(field.band_width / (np.pi / H), 2.0)
    med = (cv2.GaussianBlur(eddy, (0, 0), sigmaX=0.5 * bw_px, sigmaY=0.5 * bw_px)
           - cv2.GaussianBlur(eddy, (0, 0), sigmaX=3.0 * bw_px, sigmaY=3.0 * bw_px))
    w = _cos_weight(field.lat_row, W)
    band = (np.abs(field.lat_row) < np.radians(60.0))[:, None] & np.ones((1, W), bool)
    return float((w[band] * med[band] ** 2).sum() / (w[band].sum() + 1e-9))


def metric_texture(field: Field):
    """Metric 4 RAW: belt high-frequency luminance energy (ratio applied later)."""
    lum = field.rgb.astype(np.float32).mean(axis=2)
    blur = cv2.GaussianBlur(lum, (0, 0), sigmaX=max(field.rgb.shape[1] / 96.0, 1.0))
    hp = lum - blur
    belt = field.belt_rows
    if belt.sum() == 0:
        return float(np.mean(hp ** 2))
    return float(np.mean((hp[belt, :]) ** 2))


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def m3_min_for(preset: str) -> float:
    """Effective M3_MIN for a --preset argument (factory name or .json path)."""
    return M3_MIN_PER_PRESET.get(Path(preset).stem, M3_MIN)


def _passes(r, m3_min: float = M3_MIN) -> bool:
    """Co-gated acceptance: m1/m3/m4/m5/m6 must all hold (m2 is informational)."""
    return (r["m1"] <= M1_MAX and r["m3"] >= m3_min
            and M4_RANGE[0] <= r["m4"] <= M4_RANGE[1]
            and r["m5"] >= M5_MIN
            and M6_RANGE[0] <= r["m6"] <= M6_RANGE[1])


def label_img(img, lines):
    img = img.copy()
    y = 30
    for t in lines:
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        y += 30
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drags", default="0.0,0.08,0.15")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=4201)
    ap.add_argument("--calibrate", action="store_true",
                    help="print raw metrics only; do not apply pass/fail thresholds")
    ap.add_argument("--raw", action="store_true",
                    help="ship-config mode: sweep the drag axis on the UNMODIFIED preset "
                         "(its own solver/storms), not the develop config")
    ap.add_argument("--preset", default="gas_giant_warm",
                    help="preset to gate in ship-config (--raw) mode: a factory "
                         "preset name or a path to a .json preset file")
    ap.add_argument("--out", default=OUT,
                    help="directory for the swirl_gate.png montage (default: the "
                         "local _diag scratch dir)")
    ap.add_argument("--inject", type=float, default=0.0,
                    help="vort_inject on the develop config (test texture replenishment)")
    ap.add_argument("--inject-mask", default="belts")
    ap.add_argument("--eddy-drag", type=float, default=0.0,
                    help="vort_eddy_drag (Gate 1 eddy-only drag)")
    ap.add_argument("--sweep-eddy", action="store_true",
                    help="interpret --drags values as vort_eddy_drag (global drag forced 0)")
    ap.add_argument("--sweep-psi", action="store_true",
                    help="interpret --drags values as vort_psi_drag (scale-selective; global drag 0)")
    args = ap.parse_args()
    drags = [float(x) for x in args.drags.split(",")]
    m3_min = m3_min_for(args.preset)

    gpu = GpuContext.headless()
    rows = []
    tiles = []
    try:
        base_tex = None
        for drag in drags:
            g_drag = 0.0 if (args.sweep_eddy or args.sweep_psi) else drag
            e_drag = drag if args.sweep_eddy else args.eddy_drag
            p_drag = drag if args.sweep_psi else 0.0
            sweep_axis = "psi" if args.sweep_psi else ("eddy" if args.sweep_eddy else "drag")
            axis = f"{sweep_axis}_drag" if sweep_axis != "drag" else "drag"
            print(f"render {axis}={drag}", flush=True)
            field = render(gpu, build_cfg(args.seed, g_drag, args.width, raw=args.raw,
                                          inject=args.inject, inject_mask=args.inject_mask,
                                          eddy_drag=e_drag, psi_drag=p_drag,
                                          preset=args.preset, sweep_axis=sweep_axis), args.width)
            _, m1 = metric_blobs(field)        # max eddy-blob latitude extent
            m5 = metric_hero_color(field)      # hero readability (color contrast)
            m2 = metric_meander(field)
            m3 = metric_continuity(field)
            tex = metric_texture(field)
            med = metric_medium(field)
            if base_tex is None:
                base_tex, base_med = tex, med
            m4 = tex / base_tex if base_tex > 0 else float("inf")
            m6 = med / base_med if base_med > 0 else float("inf")

            rec = dict(drag=drag, m1=m1, m2=m2, m3=m3, m4=m4, m5=m5, m6=m6)
            rows.append(rec)

            verdict = "—" if args.calibrate else ("PASS" if _passes(rec, m3_min) else "FAIL")
            tiles.append(label_img(field.rgb[:, :, ::-1], [
                f"drag {drag}  {verdict}",
                f"m1 blob {m1:.2f}  m2 mean {m2:.2f}  m3 cont {m3:.2f}",
                f"m4 tex {m4:.2f}  m5 hero {m5:.2f}  m6 med {m6:.2f}",
            ]))
    finally:
        gpu.release()

    print("\n  drag |   m1   |  m2  |  m3  |  m4  |  m5  |  m6  | verdict  (m2 reported, not gated)")
    print("  -----+--------+------+------+------+------+------+--------")
    for r in rows:
        v = "—" if args.calibrate else ("PASS" if _passes(r, m3_min) else "FAIL")
        print(f"  {r['drag']:.3f}| {r['m1']:6.2f} | {r['m2']:.2f} | {r['m3']:.2f} "
              f"| {r['m4']:.2f} | {r['m5']:.2f} | {r['m6']:.2f} | {v}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "swirl_gate.png"), np.vstack(tiles))
    print(f"\nwrote {out_dir / 'swirl_gate.png'}")


if __name__ == "__main__":
    main()

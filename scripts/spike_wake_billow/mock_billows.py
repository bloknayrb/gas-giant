"""Gate 0b ACCEPTANCE MOCK (Task 3): a synthetic neutral-polarity billow-rope
chain, flow-warped along the baseline render's own wake curvature, composited
into gas_giant_warm's hero wake -- so the user can judge the *look* before we
invest in flow physics (Tasks 4-8). This script is a visual mock only; it is
independent of / parallel to the flow-physics Gate-1..5 pipeline that owns
config.py (spike-global-constraints.md), so it does not import config.py
(that file builds a full sim profile that is FROZEN for the Gate-1 cells;
touching it or importing side-effects from it here would be an unrelated
coupling). RC / LAT0 below are the same gas_giant_warm baked hero_radius /
hero_latitude values config.py also reads (P.storms.hero_radius = 0.108,
P.storms.hero_latitude = -24.0 deg -- see CLAUDE.md "BAKED into gas_giant_warm"
note and config.py:10-11), reproduced as plain literals here.

Reference scales (wake_reference_measurement.md, Round C, crop_ref_wide.png,
MIRRORED so the wake trails EAST there = our mirror of production's WEST):
  - along-wake fold wavelength: dominant 3.0 r_ns (harmonics 2.0/1.5/1.0)
  - transverse wavelength:      dominant 1.2 r_ns
  - orientation coherence:      0.43-0.47 (folded, not laminar)
  - polarity:                   NEUTRAL / interleaved (skew -0.12..+0.01)
  - extent:                     ~9 r_ns along-wake (RMS flat, no decay)
r_ns is the reference's core-radius anchor unit; in our render the equivalent
unit is RC = hero_radius (0.108 rad), so "rc" below means the same thing.

Geometry: rc-in-pixels conversion (px/rad along longitude, corrected by
cos(hero_latitude) for meridian convergence) matches the convention already
used elsewhere in this scratchpad for the identical measurement, see
braid_anchor_measure.py:63 (`ew_half_rad = ew_half_px * (2*np.pi/W) * np.cos(hero.lat)`).

argv: none. Reads ../wakeA/baseline/color.png and ../wake_ref.png (paths
relative to this file). Writes mock_gate0b.png / _crop.png / _panel.png here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, stats

HERE = Path(__file__).parent
SP = HERE.parent  # scratchpad root: wakeA/, wake_ref.png live here

BASELINE_PATH = SP / "wakeA" / "baseline" / "color.png"
REF_PATH = SP / "wake_ref.png"

# ---------------------------------------------------------------- inputs --
if not BASELINE_PATH.exists():
    print(f"[FATAL] baseline missing: {BASELINE_PATH}\n"
          f"        re-render via: uv run python \"{SP / 'wakeA_render.py'}\" "
          f"\"{BASELINE_PATH.parent}\"")
    sys.exit(1)
if not REF_PATH.exists():
    print(f"[FATAL] reference crop missing: {REF_PATH}")
    sys.exit(1)

baseline_u8 = cv2.imread(str(BASELINE_PATH))
assert baseline_u8 is not None, f"cv2 failed to read {BASELINE_PATH}"
H, W = baseline_u8.shape[:2]
baseline = baseline_u8.astype(np.float32) / 255.0  # BGR, 0..1

# ------------------------------------------------------- rc-in-pixels -----
# gas_giant_warm baked defaults (CLAUDE.md "BAKED into gas_giant_warm";
# also config.py RC/LAT0 for the sibling Gate-pipeline task -- same values,
# reproduced as literals here per this script's independence, see docstring).
RC = 0.108
LAT0 = np.radians(-24.0)

PX_PER_RAD_NS = H / np.pi                          # latitude (row) direction:
                                                     # uniform, no cos term
PX_PER_RAD_EW = W / (2.0 * np.pi * np.cos(LAT0))    # longitude (col)
                                                     # direction: cos(lat)
                                                     # correction for meridian
                                                     # convergence away from
                                                     # the equator, matching
                                                     # braid_anchor_measure.py:63
RC_PX_NS = RC * PX_PER_RAD_NS
RC_PX_EW = RC * PX_PER_RAD_EW


def rc_vec_to_px(dX_rc: np.ndarray, dY_rc: np.ndarray):
    """Physical (rc-space, locally isotropic) displacement -> pixel displacement."""
    return dX_rc * RC_PX_EW, dY_rc * RC_PX_NS


def px_vec_to_rc(dx_px: np.ndarray, dy_px: np.ndarray):
    """Pixel displacement -> physical (rc-space) displacement."""
    return dx_px / RC_PX_EW, dy_px / RC_PX_NS


# ------------------------------------------------------ hero pixel center -
def hero_center(img_f32: np.ndarray, H: int, W: int, lat_rad: float):
    """Auto-detect the hero core's pixel centroid: reddest connected blob in
    a window around the equirect (lat, lon=0) position. hero_longitude was
    pinned 0.0 in wakeA_render.py 'to center the crop', so lon=0 -> the
    image's center column."""
    hy_guess = (90.0 - np.degrees(lat_rad)) / 180.0 * H
    hx_guess = W / 2.0
    x0, x1 = int(hx_guess - 70), int(hx_guess + 70)
    y0, y1 = int(hy_guess - 40), int(hy_guess + 40)
    sub = img_f32[y0:y1, x0:x1]
    red = sub[..., 2] - sub[..., 0]  # BGR -> R - B
    peak, bg = np.percentile(red, 92), np.median(red)
    thr = bg + 0.32 * (peak - bg)
    mask = red > thr
    lab, n = ndimage.label(mask)
    if n == 0:
        return hx_guess, hy_guess
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    i = 1 + int(np.argmax(sizes))
    ys, xs = np.where(lab == i)
    return xs.mean() + x0, ys.mean() + y0


HX, HY = hero_center(baseline, H, W, LAT0)
RIM_X = HX - RC_PX_EW  # west edge of the storm footprint (one rc west of center)

print(f"[geometry] H={H} W={W} RC_PX_EW={RC_PX_EW:.3f} RC_PX_NS={RC_PX_NS:.3f} "
      f"hero_center=({HX:.1f},{HY:.1f}) rim_x={RIM_X:.1f}")

# ------------------------------------------------- chain extent (in rc) ---
START_OFFSET_RC = 2.0   # chain starts 2.0 rc west of the rim (spec: 1.5-3.0)
CHAIN_LEN_RC = 9.0      # chain runs 9.0 rc further west from there
TRACE_MARGIN_RC = 1.5   # extra trace length past both ends (smooth tangents)

S_CHAIN_START = START_OFFSET_RC
S_CHAIN_END = START_OFFSET_RC + CHAIN_LEN_RC
S_TRACE_END = S_CHAIN_END + TRACE_MARGIN_RC

# =====================================================================
# 1. Trace the wake's local curvature from the baseline render itself
#    (structure-tensor streamline from the rim, heading west) instead of
#    drawing a straight horizontal chain.
# =====================================================================
gray = cv2.cvtColor(baseline_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
sigma1 = max(1.0, RC_PX_NS * 0.35)
blur = cv2.GaussianBlur(gray, (0, 0), sigma1)
gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
Jxx, Jxy, Jyy = gx * gx, gx * gy, gy * gy
sigma2 = max(2.0, RC_PX_NS * 0.9)
Jxx = cv2.GaussianBlur(Jxx, (0, 0), sigma2)
Jxy = cv2.GaussianBlur(Jxy, (0, 0), sigma2)
Jyy = cv2.GaussianBlur(Jyy, (0, 0), sigma2)
# dominant-gradient orientation; the streak/band TANGENT is perpendicular to it
grad_angle = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
tangent_angle = grad_angle + np.pi / 2.0


def sample_tangent(x_px: float, y_px: float):
    xi = int(np.clip(round(x_px), 0, W - 1))
    yi = int(np.clip(round(y_px), 0, H - 1))
    th = tangent_angle[yi, xi]
    return np.cos(th), np.sin(th)


def trace_path(start_x, start_y, s_max_rc, step_px=1.5):
    xs, ys, ss = [start_x], [start_y], [0.0]
    pos = np.array([start_x, start_y], dtype=np.float64)
    prev_dir = np.array([-1.0, 0.0])
    s = 0.0
    while s < s_max_rc and pos[0] > 2 and pos[0] < W - 2 and 2 < pos[1] < H - 2:
        tx, ty = sample_tangent(pos[0], pos[1])
        cand = np.array([tx, ty])
        # structure-tensor tangent is only defined mod pi; pick the branch
        # continuous with the previous step (and initially heading west)
        if np.dot(cand, prev_dir) < 0:
            cand = -cand
        cand = cand / (np.linalg.norm(cand) + 1e-9)
        step_dir = 0.75 * prev_dir + 0.25 * cand
        step_dir = step_dir / (np.linalg.norm(step_dir) + 1e-9)
        d_px = step_dir * step_px
        d_rc = px_vec_to_rc(d_px[0], d_px[1])
        ds = float(np.hypot(*d_rc))
        pos = pos + d_px
        s += ds
        xs.append(pos[0]); ys.append(pos[1]); ss.append(s)
        prev_dir = step_dir
    return np.array(xs), np.array(ys), np.array(ss)


path_x, path_y, path_s = trace_path(RIM_X, HY, S_TRACE_END)
print(f"[trace] {len(path_s)} samples, s range [0, {path_s[-1]:.2f}] rc, "
      f"x range [{path_x.min():.0f}, {path_x.max():.0f}] "
      f"(monotonic west: {bool(np.all(np.diff(path_x) <= 0.05))})")

# tangent per sample (finite differences on the traced polyline), for the
# transverse-offset construction below
tan_x = np.gradient(path_x, path_s)
tan_y = np.gradient(path_y, path_s)
tan_norm = np.hypot(tan_x, tan_y) + 1e-9
tan_x, tan_y = tan_x / tan_norm, tan_y / tan_norm

# =====================================================================
# 2. Synthesize the neutral, folded, rolled billow-rope field in (s, v)
#    space: s = arc length along the traced wake (rc), v = transverse
#    offset from the traced centerline (rc). Both axes are isotropic rc
#    units, so 3.0/1.2 rc below are literal.
# =====================================================================
P_ALONG = 3.0   # dominant along-wake fold period (rc) -- reference target; carried by
                # SLOW size/depth modulation of a single billow stream (Fix round 2
                # defect #2 -- no positional triplet clusters, no dead zones)
P_TRANS = 1.2   # transverse roll/footprint scale (rc) -- reference target
SPACING0 = 1.0        # rc; nominal billow-to-billow spacing of the single stream
                       # (mean gap lands ~1.0 rc, inside the review's 1.0-1.5 window;
                       # 9 billows across the 9 rc chain)
SPACING_JITTER = 0.30 # +/-30% per-gap spacing jitter (raised from round 1's 20%: the
                       # extra positional smear is what keeps the ~1.0 rc "picket
                       # fence" line from beating the 3.0 rc modulation in the FFT --
                       # verified by parameter sweep, see report Fix round 2)
DEPTH_MOD = 0.80      # slow cos(2*pi*s/P_ALONG) depth-modulation depth: billows at fold
                       # peaks are up to 1.8x, at fold troughs 0.2x -- QUIET but never
                       # zero (review: "no zero-amplitude dead zones")
SIZE_MOD = 0.45       # same slow modulation applied to billow SIZE -- fold-peak billows
                       # are bigger as well as deeper, which moves real FFT energy (not
                       # just amplitude weighting) to the 3.0 rc fold period
SIZE_JITTER = 0.20       # +/-20% additional per-billow size jitter (defect #3, held)
DEPTH_JITTER = 0.30      # +/-30% additional per-billow roll-depth jitter (defect #3, held)
SIGMA_ALONG0 = 0.26   # rc; along-s Gaussian half-width of one billow's compact
                       # footprint (FWHM ~0.61 rc) -- under the ~1.0 rc nominal
                       # spacing, so individual billows stay discrete with visible
                       # gaps (defect #2 round 1: discrete cores, not a continuous rope)
SIGMA_TRANS0 = 0.52   # rc; transverse Gaussian half-width -> FWHM ~1.22 rc, matching
                       # the reference's "~1.2 rc transverse scale" per-billow footprint
CURL_GAIN = 2.0       # rolled-spiral coordinate-swirl angle at each billow's own center
                       # (Fix round 2 polish #3: raised 1.4 -> 2.0 AND the swirl decay
                       # radius widened x1.6 below, so the winding reaches the visible
                       # rim of each core instead of dying in the middle -- cores read
                       # as ROLLS with interior spiral striation, not soot spots)
CURL_DECAY_R = 1.6    # in per-billow normalized-radius units (was implicitly 1.0)
WARP_AMP = 0.09       # rc; low-freq domain warp amplitude (mechanical-tell jitter on
                       # top of the seeded per-billow jitter; reduced from 0.16 so it
                       # perturbs discrete cores without smearing them back into a rope)
MOD_AMPLITUDE = 0.37  # peak luminance modulation fraction (round 1: 0.41; trimmed in
                       # round 2 to rebalance the extreme tails after the dark-lobe
                       # soft clip -- clipping only the dark side left p99.5 well past
                       # |p0.5|, and the hp-RMS contrast had headroom above the 1.3
                       # floor to give some back; tuned by the self-checks below to
                       # keep the doc's ~1.3-1.6x wake/moat contrast ratio)
DARK_FLOOR = 0.80  # Fix round 2 defect #1 ("soot depth"): the NEGATIVE (dark)
                       # modulation lobe is soft-clipped (smooth tanh knee, no hard
                       # edge) so the COMPOSITED pixel never drops below
                       # DARK_FLOOR * (local background luminance) -- i.e. the
                       # darkest-core dip vs local background is capped at ~20%, the
                       # reference-roll value (median 175.9 -> p1 140.3 = 20%; ours
                       # measured 35% pre-fix). The clip is PER-PIXEL ADAPTIVE, not a
                       # fixed global cap: the reviewer's dip is measured against the
                       # LOCAL background, and the multiplicative composite stacks our
                       # added dip on top of the baseline's own dark texture (the
                       # western half of the traced chain runs along a baseline band
                       # that already dips 23-26% per-core with ZERO soot -- measured;
                       # two fixed global caps, 20% then 13%, both re-measured at
                       # 32-34% max per-core combined because of exactly this
                       # stacking). Adaptive form: where the baseline pixel is at its
                       # local background the full 20% dark budget is available; where
                       # the baseline is already dark the budget shrinks toward zero,
                       # so soot never deepens a pre-existing dark feature past the
                       # floor. Bright lobe untouched per the review instruction.

# LOCAL determinism: this script is a visual mock, independent of the
# Gate-1..5 physics pipeline's config.SEEDS registry (spike-global-constraints
# .md's determinism rule governs that pipeline's forcing/damping streams;
# config.py is frozen for those cells and not extended here). Same pattern,
# a script-local id not present in config.SEEDS ({11..17}).
rng = np.random.default_rng(np.random.SeedSequence([777, 1001]))
ph1, ph2, ph3, ph4 = rng.uniform(0, 2 * np.pi, size=4)

# Fix round 1 defect #3: per-billow mechanical-tell jitter (spacing/size/roll-depth),
# seeded and deterministic on the review-specified seed, kept as a SEPARATE stream
# from `rng` above (which only owns the low-freq domain-warp phases) so the two
# concerns don't couple.
jit_rng = np.random.default_rng(np.random.SeedSequence([777, 13]))

# ---- discrete billow-core placement (Fix round 2 defect #2: DE-CLUMPED) ----
# Round 1 used 3 positional clusters of 3 sub-billows; the re-review read that
# as three isolated soot clumps with dead zones between them. Round 2: a SINGLE
# jittered stream of ~9 billows (nominal spacing SPACING0=1.0 rc, +/-30% per
# gap), with the 3.0 rc dominant carried by a slow cos(2*pi*s/P_ALONG)
# modulation of billow DEPTH (x0.2..x1.8, never zero) and SIZE (x0.55..x1.45)
# instead of positional triplets. Verified by parameter sweep (see report Fix
# round 2): with +/-20% spacing jitter and depth-mod 0.65 the ~1.0 rc picket
# line still won the FFT (30.4 vs 23.1); +/-30% jitter + depth-mod 0.80 +
# size-mod 0.45 flips it decisively (2.80 rc at 38.5 vs 1.05 rc at 26.1) --
# the spacing smear suppresses the picket comb line while the size modulation
# moves real footprint energy (not just amplitude weighting) to the fold scale.
_centers = []
_s = S_CHAIN_START + SPACING0 * 0.5
while _s < S_CHAIN_END:
    _centers.append(_s)
    _s += SPACING0 * (1.0 + jit_rng.uniform(-SPACING_JITTER, SPACING_JITTER))
billow_centers = np.array(_centers)
N_BILLOW = len(billow_centers)

_mod_phase = jit_rng.uniform(0, 2 * np.pi)
_slow_depth = 1.0 + DEPTH_MOD * np.cos(2 * np.pi * billow_centers / P_ALONG + _mod_phase)
_slow_size = 1.0 + SIZE_MOD * np.cos(2 * np.pi * billow_centers / P_ALONG + _mod_phase)

billow_amp = _slow_depth * (1.0 + jit_rng.uniform(-DEPTH_JITTER, DEPTH_JITTER, N_BILLOW))
_size_jit = _slow_size * (1.0 + jit_rng.uniform(-SIZE_JITTER, SIZE_JITTER, N_BILLOW))
billow_sigma_along = SIGMA_ALONG0 * _size_jit
billow_sigma_trans = SIGMA_TRANS0 * _size_jit
billow_curl_sign = jit_rng.choice([-1.0, 1.0], size=N_BILLOW)  # per-billow roll handedness
billow_phase = jit_rng.uniform(0, 2 * np.pi, size=N_BILLOW)

_gaps = np.diff(billow_centers)
print(f"[billows] single stream: N_BILLOW={N_BILLOW} (target ~9), mean gap "
      f"{_gaps.mean():.2f} rc (target 1.0-1.5), gap range "
      f"[{_gaps.min():.2f},{_gaps.max():.2f}], min depth {billow_amp.min():.2f} "
      f"(no dead zones), centers span "
      f"[{billow_centers.min():.2f},{billow_centers.max():.2f}] rc")


def billow_field(s: np.ndarray, v: np.ndarray):
    """Neutral (zero-mean), discrete, jittered billow-rope field, in rc-space.

    Fix round 1 (defects #2/#3/#4): replaces the old continuous phase-modulated
    sine-ribbon with a sum of COMPACT, individually-windowed billow cores
    (Gaussian in both s and v, one per entry in `billow_centers`) so the field
    is exactly zero in the gaps between billows, not just quiet there. Each
    core still gets the same rolled/curled look as before via a local SWIRL
    coordinate warp around its own center (a smooth rotation angle that decays
    with a Gaussian envelope, not an added arctan2 phase term -- an earlier
    revision that summed CURL_GAIN*atan2(dv,du) over all centers had a phase
    discontinuity exactly on the v~0 centerline where each center's atan2
    branch cut sits, which corrupted the along-wake FFT self-check; see the
    original iteration note in task-3-report.md). The dominant along-wake
    period is now produced by the [big,medium,small] AMP_PATTERN repeating
    every N_SUB=3 billows (period = 3*SUB_SPACING0 = P_ALONG = 3.0 rc), not by
    an explicit FM term -- see the self-check below for the measured result.
    """
    warp_s = WARP_AMP * (np.sin(2 * np.pi * v / 4.0 + ph1)
                          + 0.5 * np.sin(2 * np.pi * s / 2.3 + ph2))
    warp_v = WARP_AMP * (np.sin(2 * np.pi * s / 2.7 + ph3)
                          + 0.5 * np.sin(2 * np.pi * v / 3.1 + ph4))
    sp, vp = s + warp_s, v + warp_v

    d_to_centers = np.abs(sp[..., None] - billow_centers[None, ...])
    k = np.argmin(d_to_centers, axis=-1)
    sk = billow_centers[k]
    amp_k = billow_amp[k]
    sal_k = billow_sigma_along[k]
    str_k = billow_sigma_trans[k]
    curl_k = billow_curl_sign[k]
    ph_k = billow_phase[k]

    du, dv = sp - sk, vp
    # compact per-billow support: zero (not just quiet) beyond a few sigma,
    # which is exactly what makes adjacent billows read as discrete with a
    # visible braid gap between them rather than one continuous rope.
    env = amp_k * np.exp(-(du / sal_k) ** 2) * np.exp(-(dv / str_k) ** 2)

    # swirl warp around the billow's own center, radius normalized by its own
    # (jittered) size so the roll always completes about one full turn
    # regardless of that billow's individual size jitter. Fix round 2 polish
    # #3: decay radius widened (CURL_DECAY_R=1.6 in normalized units) plus
    # CURL_GAIN raised to 2.0 so the winding persists out to the visible rim
    # of the core's Gaussian footprint -- interior stripes wrap into a spiral
    # (reads as a ROLL), instead of the swirl dying at the center of what
    # then reads as a plain dark spot.
    r = np.hypot(du / sal_k, dv / str_k)
    theta = curl_k * CURL_GAIN * np.exp(-(r / CURL_DECAY_R) ** 2)
    ct, st = np.cos(theta), np.sin(theta)
    du2, dv2 = du * ct - dv * st, du * st + dv * ct

    phase = 2 * np.pi * dv2 / P_TRANS + ph_k
    return np.sin(phase) * env


# --------------------------------------------------- rasterize into pixels
s_grid = np.arange(0.0, S_TRACE_END, 0.02)
v_grid = np.arange(-2.5, 2.5, 0.02)
Sg, Vg = np.meshgrid(s_grid, v_grid, indexing="ij")  # (Ns, Nv)

x_path_of_s = np.interp(Sg.ravel(), path_s, path_x)
y_path_of_s = np.interp(Sg.ravel(), path_s, path_y)
tx_of_s = np.interp(Sg.ravel(), path_s, tan_x)
ty_of_s = np.interp(Sg.ravel(), path_s, tan_y)
Txr, Tyr = px_vec_to_rc(tx_of_s, ty_of_s)
Tn = np.hypot(Txr, Tyr) + 1e-9
Txr, Tyr = Txr / Tn, Tyr / Tn
Nxr, Nyr = -Tyr, Txr  # rc-space unit normal

off_dX = Vg.ravel() * Nxr
off_dY = Vg.ravel() * Nyr
off_dx_px, off_dy_px = rc_vec_to_px(off_dX, off_dY)

px = x_path_of_s + off_dx_px
py = y_path_of_s + off_dy_px

field_vals = billow_field(Sg.ravel(), Vg.ravel())

mod_field = np.zeros((H, W), dtype=np.float64)
weight = np.zeros((H, W), dtype=np.float64)
xi = np.round(px).astype(np.int64)
yi = np.round(py).astype(np.int64)
valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
np.add.at(mod_field, (yi[valid], xi[valid]), field_vals[valid])
np.add.at(weight, (yi[valid], xi[valid]), 1.0)
nz = weight > 0
mod_field[nz] /= weight[nz]
mod_field = cv2.GaussianBlur(mod_field.astype(np.float32), (0, 0), 1.0)

# =====================================================================
# 3. Composite: multiplicative luminance modulation (preserves local hue
#    / palette, does not paste flat gray).
#    Fix round 2 defect #1 ("soot depth"): soft-clip the NEGATIVE lobe of
#    the modulation with a smooth tanh knee before compositing, capping the
#    darkest-core dip at ~MOD_AMPLITUDE*DARK_LIMIT = 20% of local
#    background. The BRIGHT lobe is untouched (review: "keep the bright
#    side as-is"). tanh is C-infinity and identity-sloped at 0, so small
#    negative values are essentially unchanged -- only the deep tail
#    compresses, which is exactly the extreme-tail asymmetry the reviewer
#    measured (deep-box skew -1.27, p1 dip 35% vs reference 20%).
# =====================================================================
# per-pixel dark budget: composited >= DARK_FLOOR * local background.
# local background = wide (1.5 rc) Gaussian blur of the baseline luminance;
# out/B = (base/B) * (1 + A*f) >= DARK_FLOOR  =>  f >= (DARK_FLOOR*B/base - 1)/A.
_LUM_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)
base_lum_f = (baseline @ _LUM_BGR).astype(np.float32)
local_bg = cv2.GaussianBlur(base_lum_f, (0, 0), 1.5 * RC_PX_NS)
_ratio = np.maximum(base_lum_f / np.maximum(local_bg, 1e-4), 1e-3)
f_min = (DARK_FLOOR / _ratio - 1.0) / MOD_AMPLITUDE  # <= 0 where base >= DARK_FLOOR*B
f_min = np.minimum(f_min, -1e-4)  # where baseline is ALREADY below the floor: no
                                   # added darkening (budget ~0), but never positive
                                   # (we don't force-brighten; bright lobes still may)
mod_soft = np.where(mod_field < 0.0,
                    f_min * np.tanh(mod_field / f_min),  # smooth knee, identity at 0
                    mod_field)
out = baseline * (1.0 + MOD_AMPLITUDE * mod_soft[..., None])
out = np.clip(out, 0.0, 1.0)
out_u8 = (out * 255.0 + 0.5).astype(np.uint8)
cv2.imwrite(str(HERE / "mock_gate0b.png"), out_u8)

# ----------------------------------------------------------------- crop --
pad_ew = 0.6 * RC_PX_EW
pad_ns = 0.6 * RC_PX_NS
cx0 = max(0, int(RIM_X - (S_TRACE_END) * RC_PX_EW - pad_ew))
cx1 = min(W, int(HX + 0.4 * RC_PX_EW + pad_ew))
cy0 = max(0, int(min(path_y.min(), HY) - 3.2 * RC_PX_NS))
cy1 = min(H, int(max(path_y.max(), HY) + 3.2 * RC_PX_NS))
crop = out_u8[cy0:cy1, cx0:cx1]
ZOOM = 3
crop_zoom = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM),
                        interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(HERE / "mock_gate0b_crop.png"), crop_zoom)
print(f"[crop] box x[{cx0},{cx1}] y[{cy0},{cy1}] -> {crop_zoom.shape[1]}x{crop_zoom.shape[0]} "
      f"({ZOOM}x)")


# ------------------------------------------------------------ panel image
def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (11 * len(text) + 10, 24), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                cv2.LINE_AA)
    return img


ref_u8 = cv2.imread(str(REF_PATH))
assert ref_u8 is not None
# FIX (Gate-0b review defect 1): wake_ref.png's storm is ALREADY on the right
# edge (measured max-redness column fraction 0.976, i.e. storm-right / wake
# trailing left) -- verified directly on this file, not assumed from the
# differently-oriented crop_ref_wide.png referenced in
# wake_reference_measurement.md. The previous cv2.flip(...,1) here mirrored
# the ALREADY-correct crop, dragging the storm to the far-left edge (frac
# ~0.01) -- exactly the review-caught bug (label said "flipped to WEST frame"
# but the flip made it wrong, not right; the label was never re-verified by
# pixel measurement). No flip needed: use the crop as-is so both panels agree
# storm-right / wake-left.
ref_frame = ref_u8

target_h = crop_zoom.shape[0]
ref_resized = cv2.resize(ref_frame,
                          (int(ref_frame.shape[1] * target_h / ref_frame.shape[0]), target_h))

left = label(ref_resized, "REFERENCE (PIA07782-class, storm-right frame)")
right = label(crop_zoom, "OUR MOCK (Gate 0b, wake box)")
gap = np.full((target_h, 8, 3), 40, np.uint8)
panel = np.hstack([left, gap, right])
cv2.imwrite(str(HERE / "mock_gate0b_panel.png"), panel)
print(f"[panel] {panel.shape[1]}x{panel.shape[0]} -> mock_gate0b_panel.png")

# =====================================================================
# 4. Self-check: measure OUR OWN along-wake / transverse wavelengths via
#    FFT on the synthesized (s, v) field (the clean, unprojected signal)
#    and report against the 3.0 rc / 1.2 rc targets.
# =====================================================================


def dominant_wavelength(sig_1d: np.ndarray, d: float, lo_rc=0.3, hi_rc=6.0):
    n = len(sig_1d)
    sig_1d = sig_1d - sig_1d.mean()
    spec = np.abs(np.fft.rfft(sig_1d * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=d)
    with np.errstate(divide="ignore"):
        wl = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9), np.inf)
    band = (wl >= lo_rc) & (wl <= hi_rc)
    if not np.any(band):
        return float("nan")
    i = np.argmax(spec[band])
    return float(wl[band][i])


def dominant_wavelength_envelope(sig2d, d: float, lo_rc=0.3, hi_rc=6.0):
    """Along-wake period from the RMS ENVELOPE across a band of parallel
    transverse (v) lines, then FFT of that 1-D envelope curve.

    Tried first (see task-3-report.md "Fix round 1"): averaging the raw
    per-line MAGNITUDE SPECTRA (review3_measure.py's own dom_wavelength_px
    approach) across the v-band. That measured 2.1 rc (30% off target) even
    with the swirl curl term set to ZERO, because each line's raw signal is
    sin(2*pi*v/P_TRANS + phase_k) * envelope -- the per-billow phase_k causes
    sign flips in v that differ billow-to-billow, and those flips inject
    their own along-s frequency content into the raw spectrum average,
    independent of the actual billow-to-billow spacing.
    Rectifying first (RMS across the v-band at each s, THEN one FFT) removes
    that sign-flip noise and isolates the coarse "how much roll activity is
    present here" envelope -- which is exactly what an along-wake fold
    wavelength should measure for a field of discrete, individually-phased
    rolls. This is also methodologically closer to how the reference
    measurement itself was taken (wake_reference_measurement.md's contrast
    numbers are 31-px high-pass LUMINANCE RMS, an unsigned activity measure,
    not a raw signed pixel FFT)."""
    env = np.sqrt((sig2d ** 2).mean(axis=0))
    return dominant_wavelength(env, d=d, lo_rc=lo_rc, hi_rc=hi_rc)


# transverse wavelength: cross-sections at several interior s values, v in
# the dense core of the band (avoid the tapered transverse edges)
interior_s = np.linspace(S_CHAIN_START + 1.0, S_CHAIN_END - 1.0, 7)
v_fine = np.arange(-1.3, 1.3, 0.01)
trans_wls = []
for sc in interior_s:
    sig = billow_field(np.full_like(v_fine, sc), v_fine)
    trans_wls.append(dominant_wavelength(sig, d=0.01, lo_rc=0.3, hi_rc=2.5))
trans_wl = float(np.nanmedian(trans_wls))

# along-wake wavelength: band-averaged across several transverse (v) lines
# spanning the billow footprint (fix round 1: a single v~0 centerline cut
# threads each discrete billow's internal swirl texture, which oscillates
# faster than -- and independent of -- the billow-to-billow envelope
# spacing we actually want to measure; averaging spectra across the band
# isolates the shared coarse period), avoiding the along-wake taper zones.
s_fine = np.arange(S_CHAIN_START + 0.3, S_CHAIN_END - 0.3, 0.01)
v_band = np.linspace(-0.9, 0.9, 13)
along_sig2d = np.stack([billow_field(s_fine, np.full_like(s_fine, vo)) for vo in v_band])
along_wl = dominant_wavelength_envelope(along_sig2d, d=0.01, lo_rc=1.0, hi_rc=6.0)

trans_err = abs(trans_wl - P_TRANS) / P_TRANS * 100
along_err = abs(along_wl - P_ALONG) / P_ALONG * 100

print(f"[self-check] transverse wavelength: measured {trans_wl:.3f} rc "
      f"(target {P_TRANS} rc, err {trans_err:.1f}%)")
print(f"[self-check] along-wake wavelength: measured {along_wl:.3f} rc "
      f"(target {P_ALONG} rc, err {along_err:.1f}%)")
print(f"[self-check] {'PASS' if trans_err <= 15 and along_err <= 15 else 'FAIL'} "
      f"(15% tolerance)")

# bonus diagnostic: structure-tensor coherence of the rasterized mod_field
# within the chain box (not a hard gate, design target ~0.45)
band_y0, band_y1 = max(0, int(HY - 2.2 * RC_PX_NS)), min(H, int(HY + 2.2 * RC_PX_NS))
band_x0, band_x1 = cx0, cx1
sub = mod_field[band_y0:band_y1, band_x0:band_x1].astype(np.float32)
sgx = cv2.Sobel(sub, cv2.CV_64F, 1, 0, ksize=3)
sgy = cv2.Sobel(sub, cv2.CV_64F, 0, 1, ksize=3)
jxx = cv2.GaussianBlur(sgx * sgx, (0, 0), 2.0)
jxy = cv2.GaussianBlur(sgx * sgy, (0, 0), 2.0)
jyy = cv2.GaussianBlur(sgy * sgy, (0, 0), 2.0)
tr = jxx + jyy
disc = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
coherence = np.where(tr > 1e-9, disc / (tr + 1e-9), 0.0)
print(f"[self-check] mean structure-tensor coherence in chain box: "
      f"{coherence.mean():.3f} (design target ~0.45)")

# =====================================================================
# 4b. COMPOSITED self-check (fix defect #4): the review's explicit ask is to
#     measure the along-wake dominant period "on the composited wake box",
#     i.e. AFTER flow-warp rasterization + boundary clipping + Gaussian
#     blur -- not on the idealized closed-form billow_field() above (which
#     can hide artifacts the raster introduces). mod_field is resampled
#     back onto a regular (s, v) grid with the SAME forward map used to
#     rasterize it (path position + rc-space normal offset -> px), then put
#     through the identical FFT method as section 4.
# =====================================================================


def sample_mod_field_sv(s_arr, v_arr):
    s_arr = np.asarray(s_arr, dtype=np.float64)
    v_arr = np.asarray(v_arr, dtype=np.float64)
    xp = np.interp(s_arr, path_s, path_x)
    yp = np.interp(s_arr, path_s, path_y)
    txv = np.interp(s_arr, path_s, tan_x)
    tyv = np.interp(s_arr, path_s, tan_y)
    txr, tyr = px_vec_to_rc(txv, tyv)
    tn = np.hypot(txr, tyr) + 1e-9
    txr, tyr = txr / tn, tyr / tn
    nxr, nyr = -tyr, txr
    dx_rc, dy_rc = v_arr * nxr, v_arr * nyr
    dx_px, dy_px = rc_vec_to_px(dx_rc, dy_rc)
    px_ = xp + dx_px
    py_ = yp + dy_px
    # map_coordinates expects (row, col) = (y, x)
    return ndimage.map_coordinates(mod_field, [py_, px_], order=1, mode="constant", cval=0.0)


trans_wls_c = []
for sc in interior_s:
    sig = sample_mod_field_sv(np.full_like(v_fine, sc), v_fine)
    trans_wls_c.append(dominant_wavelength(sig, d=0.01, lo_rc=0.3, hi_rc=2.5))
trans_wl_c = float(np.nanmedian(trans_wls_c))

along_sig2d_c = np.stack([sample_mod_field_sv(s_fine, np.full_like(s_fine, vo)) for vo in v_band])
along_wl_c = dominant_wavelength_envelope(along_sig2d_c, d=0.01, lo_rc=1.0, hi_rc=6.0)

trans_err_c = abs(trans_wl_c - P_TRANS) / P_TRANS * 100
along_err_c = abs(along_wl_c - P_ALONG) / P_ALONG * 100
print(f"[composited self-check] transverse wavelength: measured {trans_wl_c:.3f} rc "
      f"(target {P_TRANS} rc, err {trans_err_c:.1f}%)")
print(f"[composited self-check] along-wake wavelength: measured {along_wl_c:.3f} rc "
      f"(target {P_ALONG} rc, err {along_err_c:.1f}%)")
print(f"[composited self-check] {'PASS' if trans_err_c <= 15 and along_err_c <= 15 else 'FAIL'} "
      f"(15% tolerance, ON THE COMPOSITED raster per review defect #4)")

print(f"[billows] discrete billow count in chain s in [{S_CHAIN_START},{S_CHAIN_END}] rc: "
      f"{N_BILLOW} (reference qualitative target ~8-12)")

# ---------------------------------------------------- polarity / contrast --
_LUM_W = np.array([0.114, 0.587, 0.299])  # BGR -> luma weights
out_lum = out_u8[..., :3].astype(np.float64) @ _LUM_W
base_lum = baseline_u8[..., :3].astype(np.float64) @ _LUM_W
d_lum = out_lum - base_lum

wx_a, wx_b = int(RIM_X - S_CHAIN_END * RC_PX_EW), int(RIM_X - S_CHAIN_START * RC_PX_EW)
wx0, wx1 = min(wx_a, wx_b), max(wx_a, wx_b)
wy0, wy1 = int(HY - 2.2 * RC_PX_NS), int(HY + 2.2 * RC_PX_NS)
wbox = d_lum[wy0:wy1, wx0:wx1]
skew = float(stats.skew(wbox.ravel()))
frac_pos = float((wbox > 2).mean())
frac_neg = float((wbox < -2).mean())
print(f"[polarity] wake-box (mock-baseline) luminance skew {skew:.3f} (target ~0, "
      f"neutral) frac_bright>2:{frac_pos:.3f} frac_dark<-2:{frac_neg:.3f}")

# --- Fix round 2 reviewer quantities (soot depth / extreme-tail balance) ---
# darkest-core dip, PER-CORE against LOCAL background: for each billow center
# (projected to pixels through the traced path), core darkness = p1 of the
# luminance within 0.8 rc of the center, local background = median of the
# 1.0-2.2 rc annulus around it. Same construction as the reviewer's reference
# numbers (median 175.9 -> p1 140.3 = 20% dip is a roll-vs-its-surroundings
# measurement on the reference crop, not a whole-box percentile: the whole
# wake box here contains the baseline's OWN dark band structure, which
# already dips 31% with zero soot added -- measured on the raw baseline --
# so a whole-box p1 would be dominated by that pre-existing structure).
_bc_x = np.interp(billow_centers, path_s, path_x)
_bc_y = np.interp(billow_centers, path_s, path_y)
_yy, _xx = np.mgrid[0:H, 0:W]
core_dips = []
for _cx, _cy in zip(_bc_x, _bc_y):
    _r2 = (((_xx - _cx) / RC_PX_EW) ** 2 + ((_yy - _cy) / RC_PX_NS) ** 2)
    _core = out_lum[_r2 <= 0.8 ** 2]
    _ann = out_lum[(_r2 >= 1.0 ** 2) & (_r2 <= 2.2 ** 2)]
    _bg = float(np.median(_ann))
    _dip = (_bg - float(np.percentile(_core, 1))) / _bg * 100.0
    core_dips.append(_dip)
core_dips = np.array(core_dips)
print(f"[soot] per-core darkest dip vs local bg: max {core_dips.max():.1f}% "
      f"median {np.median(core_dips):.1f}% (target <= ~20%; reference rolls 20%)")
# deep-box skew: composited wake-box luminance skew (reviewer's -1.27 axis);
# extreme-tail balance: p0.5 vs p99.5 of the ADDED modulation (mock-baseline).
obox = out_lum[wy0:wy1, wx0:wx1]
deep_skew = float(stats.skew(obox.ravel()))
p_lo = float(np.percentile(wbox, 0.5))
p_hi = float(np.percentile(wbox, 99.5))
print(f"[soot] deep-box skew (composited wake-box luminance): {deep_skew:.3f} "
      f"(was -1.27 pre-fix; target much closer to 0)")
print(f"[soot] added-mod extreme tails: p0.5 {p_lo:.1f} vs p99.5 {p_hi:.1f} "
      f"(target roughly symmetric)")


def _hp_rms(lum2d, k=31):
    lp = ndimage.uniform_filter1d(lum2d, k, axis=1, mode="reflect")
    return float((lum2d - lp).std())


wake_hp = _hp_rms(out_lum[wy0:wy1, wx0:wx1])
base_hp = _hp_rms(base_lum[wy0:wy1, wx0:wx1])
print(f"[contrast] wake hp-RMS(31px) ratio mock/base: {wake_hp / base_hp:.3f} "
      f"(target 1.3-1.6x per wake_reference_measurement.md)")


# ------------------------------------------------------- frame verification
def _redness_col_frac(img_bgr):
    img_bgr = img_bgr.astype(np.float64)
    rb = img_bgr[..., 2] - img_bgr[..., 0]
    prof = rb.mean(axis=0)
    prof_s = np.convolve(prof, np.ones(40) / 40, mode="same")
    return float(np.argmax(prof_s)) / img_bgr.shape[1]


panel_half = panel.shape[1] // 2
ref_frac = _redness_col_frac(panel[:, :panel_half])
mock_frac = _redness_col_frac(panel[:, panel_half:])
print(f"[frame] panel max-redness column fraction: ref={ref_frac:.3f} "
      f"mock={mock_frac:.3f} (fix defect #1 target: BOTH > 0.8, storm-right)")

print("done: mock_gate0b.png, mock_gate0b_crop.png, mock_gate0b_panel.png")

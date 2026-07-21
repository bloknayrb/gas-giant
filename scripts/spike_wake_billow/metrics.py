"""FROZEN Gate-1 pass/fail criteria for the billow-rollup spike (Task 5).

Cardinal requirement (adversarial plan review): the metric MUST FAIL today's
production wake (3-5 large rolls) and MUST PASS a reference-true chain (dense
small billows). Every numeric threshold below is FROZEN by task-5-brief.md;
tune the *synthetic* self-test fields, NEVER the thresholds.

Pixel scale
-----------
Every length in the brief is written in units of RC/DX (production pixels).
`config.RC/config.DX ~= 70.4 px = 1 rc`. All band arrays fed to the scorers are
at THIS scale (1 px = DX rad); the status-quo PNG (case c) is resampled onto it
before scoring so the frozen area/wavelength bounds mean the same thing.

Coherence convention (PINNED)
-----------------------------
Structure-tensor coherence uses EXACTLY the reference measurement script's
formula (`..\\wake_measure.py`:178):
    c = sqrt((Jxx-Jyy)^2 + 4 Jxy^2) / (Jxx + Jyy + eps)
For a 2x2 tensor this is identically (l1-l2)/(l1+l2), the brief's sketch -- the
two agree, so nothing had to be overridden. The only pinned choice is the
tensor-smoothing sigma, which the brief freezes at 0.25*RC/DX px (wake_measure
used a bare sigma=4; the brief's sigma wins and is documented here).

API (consumed by Task 6's runner)
----------------------------------
  hp(f, cfg)                      -> high-pass a band array (mode='wrap')
  extract_band(field, cfg)        -> (NY,NX) full field -> band sub-array
  find_billows(hpb, rms0, cfg)    -> (labels, n, props) reference-scale billows
  wavelength(hpb, cfg)            -> WaveResult (lam, hypothesis, N*)
  coherence(hpb, cfg)             -> float band-averaged coherence
  score_band(trb, cfg, rms0, ..)  -> FrameScore (all PER-FRAME criteria)
  score_frame(tr_full, cfg, ..)   -> extract_band then score_band
  evaluate_run(samples, cfg, ..)  -> Gate1Result (the full frozen gate)
  gate1_verdict = evaluate_run    (alias per the brief's naming)
  selftest()                      -> runs the 4 mandatory cases + diagnostic png

`samples` for evaluate_run is a sequence of Sample(step, tr, w=None) (any object
exposing .step/.tr/.w, or a dict with those keys). tr/w are full (NY,NX) fields.
Nothing here touches src/gasgiant/** or git.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, stats

SPIKE = pathlib.Path(__file__).parent
sys.path.insert(0, str(SPIKE))
import config as _config  # noqa: E402  (spike-local single-source constants)

_EPS = 1e-12
_8CONN = np.ones((3, 3), dtype=bool)


# ============================================================================
# Geometry helpers (all derived from config -- the single source)
# ============================================================================
def _px_per_rc(cfg):
    return cfg.RC / cfg.DX


def _band_indices(cfg):
    """Row/col index arrays of the scoring band inside a full (NY,NX) field."""
    y = (np.arange(cfg.NY) + 0.5) * cfg.DX
    x = (np.arange(cfg.NX) + 0.5) * cfg.DX
    rows = np.where(np.abs(y - cfg.Y_SHEET) < cfg.BAND_HW)[0]
    cols = np.where((x >= cfg.BAND_X[0]) & (x < cfg.BAND_X[1]))[0]
    return rows, cols


def extract_band(field, cfg=_config):
    """Extract the scoring band (BAND_X x |y-Y_SHEET|<BAND_HW) from a full field."""
    rows, cols = _band_indices(cfg)
    return field[np.ix_(rows, cols)]


def band_dims(cfg=_config):
    """(band_ny, band_nx) at sim pixel scale -- resample target for the PNG."""
    band_nx = int(round((cfg.BAND_X[1] - cfg.BAND_X[0]) / cfg.DX))
    band_ny = int(round(2.0 * cfg.BAND_HW / cfg.DX))
    return band_ny, band_nx


# ============================================================================
# Core primitives
# ============================================================================
def hp(f, cfg=_config):
    """High-pass: f - gaussian_lowpass(f, sigma=0.5*RC/DX, mode='wrap').

    Computed on the BAND array (uniform treatment across full-field, synthetic
    and PNG inputs). mode='wrap' per the frozen spec; on a sub-window the x-edges
    wrap approximately -- negligible for the 35 px sigma vs the interior
    structures, and consistent for every case."""
    sigma = 0.5 * _px_per_rc(cfg)
    return f - ndimage.gaussian_filter(f, sigma, mode="wrap")


def find_billows(hpb, rms0, cfg=_config):
    """Segment reference-scale billows.

    threshold |hp| > 1.4*rms0 ; 8-connected components ; keep those with
    area in [(0.3*RC/DX)^2, (1.2*RC/DX)^2] px AND second-moment aspect < 4.
    The MAX area bound is the anti-status-quo criterion (rejects the big rolls).

    Returns (kept_label_image, n_billows, props, seg) where props is a list of
    dicts (area, aspect, centroid) for kept components and seg is a diagnostics
    dict counting why components were rejected. seg['n_oversize'] (rejected for
    area > a_max) is the direct evidence of the anti-status-quo big-roll reject."""
    ppc = _px_per_rc(cfg)
    a_min = (0.3 * ppc) ** 2
    a_max = (1.2 * ppc) ** 2
    mask = np.abs(hpb) > (1.4 * rms0)
    lab, n = ndimage.label(mask, structure=_8CONN)
    kept = np.zeros_like(lab)
    props = []
    seg = dict(n_raw=int(n), n_undersize=0, n_oversize=0, n_aspect_reject=0)
    if n:
        objs = ndimage.find_objects(lab)
        for i in range(1, n + 1):
            sl = objs[i - 1]
            sub = lab[sl] == i
            area = int(sub.sum())
            if area < a_min:
                seg["n_undersize"] += 1
                continue
            if area > a_max:
                seg["n_oversize"] += 1
                continue
            ys, xs = np.nonzero(sub)
            if area >= 2 and np.ptp(xs) > 0 and np.ptp(ys) > 0:
                cov = np.cov(np.vstack([xs.astype(float), ys.astype(float)]))
                ev = np.linalg.eigvalsh(cov)
                l2, l1 = float(ev[0]), float(ev[1])
                aspect = np.sqrt(max(l1, 0.0) / max(l2, _EPS))
            else:
                aspect = np.inf
            if aspect >= 4.0:
                seg["n_aspect_reject"] += 1
                continue
            kept[sl][sub] = len(props) + 1
            props.append(dict(
                area=area, aspect=float(aspect),
                cy=float(ys.mean() + (sl[0].start or 0)),
                cx=float(xs.mean() + (sl[1].start or 0)),
            ))
    return kept, len(props), props, seg


@dataclass
class WaveResult:
    lam_rad: float
    lam_rc: float
    hypothesis: str | None   # '1rc' | '3rc' | None
    n_star: int | None
    band_len_rc: float
    freqs: np.ndarray = field(default=None, repr=False)
    power: np.ndarray = field(default=None, repr=False)


def wavelength(hpb, cfg=_config):
    """Dominant along-x wavelength via per-row x-FFT, POWER averaged over band
    rows THEN peak (band-averaging first cancels transverse-alternating
    structure). Two-interval FROZEN bracket + wavelength-coupled count N*."""
    ny, nx = hpb.shape
    sig = hpb - hpb.mean(axis=1, keepdims=True)   # per-row DC removed
    win = np.hanning(nx)[None, :]
    spec = np.abs(np.fft.rfft(sig * win, axis=1)) ** 2
    power = spec.mean(axis=0)
    power[0] = 0.0
    freqs = np.fft.rfftfreq(nx, d=cfg.DX)          # cycles / rad
    band_len_rad = nx * cfg.DX
    with np.errstate(divide="ignore"):
        wl = np.where(freqs > 0, 1.0 / np.maximum(freqs, _EPS), np.inf)
    # resolvable: wavelength in [2*DX, band length]
    valid = (wl >= 2.0 * cfg.DX) & (wl <= band_len_rad)
    k = int(np.argmax(np.where(valid, power, -1.0)))
    lam_rad = float(wl[k])
    lam_rc = lam_rad / cfg.RC
    band_len_rc = band_len_rad / cfg.RC

    hyp, n_star = None, None
    if 0.9 <= lam_rc <= 1.5:
        hyp = "1rc"
        n_star = max(4, int(np.floor(band_len_rc / lam_rc)) - 1)
    elif 2.4 <= lam_rc <= 3.6:
        hyp = "3rc"
        n_star = max(2, int(np.floor(band_len_rc / lam_rc)) - 1)
    return WaveResult(lam_rad, lam_rc, hyp, n_star, band_len_rc, freqs, power)


# PINNED coherence convention = wake_measure.py's, in FULL (review fix round 1):
# formula (wake_measure.py:178) c = sqrt((Jxx-Jyy)^2 + 4 Jxy^2)/(Jxx+Jyy+eps)
# AND tensor smoothing sigma = 4 px (wake_measure.py:161 `def smooth(a, s=4)`).
# The brief's sketch said sigma = 0.25*RC/DX (17.6 px), but it also said the
# reference script's convention WINS where they differ -- they differ on sigma,
# so s=4 is pinned and the brief-sketch sigma is the documented deviation.
_COH_SIGMA_PX = 4.0


def coherence(hpb, cfg=_config):
    """Band-averaged structure-tensor coherence (PINNED wake_measure.py
    convention: its :178 formula AND its :161 smoothing sigma = 4 px)."""
    s = _COH_SIGMA_PX
    gy, gx = np.gradient(hpb)
    jxx = ndimage.gaussian_filter(gx * gx, s, mode="nearest")
    jyy = ndimage.gaussian_filter(gy * gy, s, mode="nearest")
    jxy = ndimage.gaussian_filter(gx * gy, s, mode="nearest")
    disc = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    c = disc / (jxx + jyy + _EPS)
    return float(c.mean())


def polarity_skew(hpb):
    return float(stats.skew(hpb.ravel()))


def band_rms_ratio(hpb, rms0):
    return float(np.sqrt(np.mean(hpb ** 2)) / max(rms0, _EPS))


def flank_leak(field, cfg=_config):
    """Sponge-leak diagnostic: hp-RMS in the FLANK_W band at the y-boundaries of
    the scoring x-window, relative to hp-RMS in the scoring band. Compared to
    config.FLANK_MAX by the gate. Operates on a full (NY,NX) field."""
    _, cols = _band_indices(cfg)
    y = (np.arange(cfg.NY) + 0.5) * cfg.DX
    band_rows = np.where(np.abs(y - cfg.Y_SHEET) < cfg.BAND_HW)[0]
    flank_rows = np.where(
        (np.abs(y - cfg.Y_SHEET) >= cfg.BAND_HW)
        & (np.abs(y - cfg.Y_SHEET) < cfg.BAND_HW + cfg.FLANK_W)
    )[0]
    band = hp(field[np.ix_(band_rows, cols)], cfg)
    flank = hp(field[np.ix_(flank_rows, cols)], cfg)
    br = np.sqrt(np.mean(band ** 2))
    return float(np.sqrt(np.mean(flank ** 2)) / max(br, _EPS))


# ============================================================================
# omega-core SECONDARY diagnostic (reported, NOT gating)
# ============================================================================
def omega_core_diagnostic(w_full, cfg=_config, sign_template=None):
    """Count coherent omega cores in the band. sigma0 is taken from the
    STRIP-noise band (STRIP_X columns, band rows) -- NOT the sheet-contaminated
    scoring band (review M3). Threshold |w| > 2*sigma0, 8-connected; sign_template
    gives the sheet-target omega sign per y-side (default: ambient sign, negative
    on both sides per config's uniform-sign bracket note). REPORTED, never gates."""
    y = (np.arange(cfg.NY) + 0.5) * cfg.DX
    x = (np.arange(cfg.NX) + 0.5) * cfg.DX
    band_rows = np.where(np.abs(y - cfg.Y_SHEET) < cfg.BAND_HW)[0]
    strip_cols = np.where((x >= cfg.STRIP_X[0]) & (x < cfg.STRIP_X[1]))[0]
    if strip_cols.size == 0:
        strip_cols = np.arange(0, 4)
    sigma0 = float(np.std(w_full[np.ix_(band_rows, strip_cols)]))
    _, cols = _band_indices(cfg)
    wb = w_full[np.ix_(band_rows, cols)]
    if sign_template is None:
        sign_template = -np.ones(wb.shape[0])   # ambient omega sign, both flanks
    sgn = np.sign(sign_template)[:, None]
    mask = (np.abs(wb) > 2.0 * max(sigma0, _EPS)) & (np.sign(wb) == sgn)
    _, n = ndimage.label(mask, structure=_8CONN)
    return dict(sigma0=sigma0, n_omega_cores=int(n))


# ============================================================================
# Per-frame scoring
# ============================================================================
@dataclass
class FrameScore:
    step: int
    n_billows: int
    wave: WaveResult
    skew: float
    coh: float
    rms_ratio: float
    # per-frame criteria (booleans)
    count_ok: bool
    wavelength_ok: bool
    polarity_ok: bool
    coherence_ok: bool
    band_rms_ok: bool
    per_frame_pass: bool
    labels: np.ndarray = field(default=None, repr=False)
    props: list = field(default=None, repr=False)
    seg: dict = field(default=None, repr=False)
    hpb: np.ndarray = field(default=None, repr=False)
    omega: dict = field(default=None, repr=False)

    def failing(self):
        names = [("count", self.count_ok), ("wavelength", self.wavelength_ok),
                 ("polarity", self.polarity_ok), ("coherence", self.coherence_ok),
                 ("band_rms", self.band_rms_ok)]
        return [n for n, ok in names if not ok]


def score_band(trb, cfg=_config, rms0=None, w_full=None, step=-1):
    """Score a single band-shaped tracer array against every PER-FRAME criterion.

    rms0 is the threshold-conditioning baseline (rms of hp(tr) over the band at
    t=0). If None, it is taken from this frame's own hp -- ONLY valid for a
    standalone frame where no t=0 exists; the run path always supplies the real
    t=0 rms0."""
    hpb = hp(trb, cfg)
    if rms0 is None:
        rms0 = float(np.sqrt(np.mean(hpb ** 2)))
    labels, n_bill, props, seg = find_billows(hpb, rms0, cfg)
    wave = wavelength(hpb, cfg)
    sk = polarity_skew(hpb)
    coh = coherence(hpb, cfg)
    ratio = band_rms_ratio(hpb, rms0)

    wavelength_ok = wave.hypothesis is not None
    count_ok = wavelength_ok and (wave.n_star is not None) and (n_bill >= wave.n_star)
    polarity_ok = abs(sk) <= 0.2
    coherence_ok = 0.35 <= coh <= 0.55
    band_rms_ok = ratio >= 1.4
    per_frame_pass = (count_ok and wavelength_ok and polarity_ok
                      and coherence_ok and band_rms_ok)
    omega = omega_core_diagnostic(w_full, cfg) if w_full is not None else None
    return FrameScore(step, n_bill, wave, sk, coh, ratio,
                      count_ok, wavelength_ok, polarity_ok, coherence_ok,
                      band_rms_ok, per_frame_pass, labels, props, seg, hpb, omega)


def score_frame(tr_full, cfg=_config, rms0=None, w_full=None, step=-1):
    """Extract the scoring band from a full (NY,NX) tracer field, then score it."""
    return score_band(extract_band(tr_full, cfg), cfg, rms0, w_full, step)


# ============================================================================
# Full-run frozen gate
# ============================================================================
def _as_sample(s):
    if isinstance(s, dict):
        return s.get("step"), s["tr"], s.get("w")
    return s.step, s.tr, getattr(s, "w", None)


@dataclass
class Gate1Result:
    gate1_pass: bool
    reasons: dict                      # full per-criterion pass/fail dict
    coherence_only_fail: bool          # True iff coherence is the ONLY failed criterion
    frames: list                       # FrameScore per sample
    n_star: int | None
    hypothesis: str | None
    lam_rc: float
    count_frac: float
    formation_step: int | None
    formation_bar: float
    window_start: float
    window_n: int
    stationarity: dict
    flank: float
    substep_count: int


def evaluate_run(samples, cfg=_config, *, sigma_expected, rms0=None,
                 substep_count=0, strip_to_band=None, band_fill=None,
                 flank=None, min_window_steps=2000):
    """The FROZEN Gate-1 verdict over a time series of sampled fields.

    Parameters
    ----------
    samples : sequence of Sample(step, tr, w) sampled every 50 steps FROM STEP 0.
    sigma_expected : expected KH growth rate (per production TIME unit), supplied
        per cell by the caller; sets the formation bar's development allowance.
    rms0 : threshold-conditioning baseline; if None, hp-RMS of samples[0] band.
    substep_count : cumulative CFL substeps over the run (gate requires == 0).
    strip_to_band, band_fill : advective clocks in STEPS; default from config.
    flank : flank-leak value (max over run); if None it is computed per-sample
        from the tracer fields via flank_leak() and the max taken.
    min_window_steps : the stationarity window must span at least this many steps.

    Formation bar (FROZEN): strip_to_band + max(DEV_STEPS, (2/sigma_expected)/DT)
    steps. 2/sigma_expected is a TIME; /DT converts it to steps.
    Stationarity window: post-band_fill portion (steps >= strip_to_band+band_fill).
    """
    steps = np.array([_as_sample(s)[0] for s in samples], dtype=float)
    order = np.argsort(steps)
    samples = [samples[i] for i in order]
    steps = steps[order]

    if strip_to_band is None or band_fill is None:
        _, s2b, bf = cfg.transit_report()
        strip_to_band = strip_to_band if strip_to_band is not None else s2b
        band_fill = band_fill if band_fill is not None else bf

    # rms0 from the t=0 frame
    if rms0 is None:
        _, tr0, _ = _as_sample(samples[0])
        rms0 = float(np.sqrt(np.mean(hp(extract_band(tr0, cfg), cfg) ** 2)))

    # score every frame
    frames = []
    flanks = []
    for st, tr, w in map(_as_sample, samples):
        frames.append(score_frame(tr, cfg, rms0=rms0, w_full=w, step=int(st)))
        if flank is None:
            flanks.append(flank_leak(tr, cfg))
    flank_val = flank if flank is not None else (max(flanks) if flanks else 0.0)

    # --- formation clock -----------------------------------------------------
    dev_allow = max(cfg.DEV_STEPS, (2.0 / max(sigma_expected, _EPS)) / cfg.DT)
    formation_bar = strip_to_band + dev_allow
    formation_step = None
    for fr in frames:
        if fr.count_ok:
            formation_step = fr.step
            break
    formation_ok = (formation_step is not None) and (formation_step <= formation_bar)

    # --- stationarity window (post band-fill) --------------------------------
    window_start = strip_to_band + band_fill
    in_win = steps >= window_start
    win_frames = [fr for fr, mk in zip(frames, in_win, strict=True) if mk]
    win_steps = steps[in_win]
    window_span = (win_steps[-1] - win_steps[0]) if win_steps.size >= 2 else 0.0
    window_ok_len = window_span >= min_window_steps and len(win_frames) >= 4

    # representative wavelength / N* from the window (median of bracketed frames)
    win_lams = [fr.wave.lam_rc for fr in win_frames if fr.wavelength_ok]
    if win_lams:
        lam_rc = float(np.median(win_lams))
    else:
        lam_rc = float(np.median([fr.wave.lam_rc for fr in win_frames])) if win_frames else np.nan
    hypothesis, n_star = None, None
    if 0.9 <= lam_rc <= 1.5:
        hypothesis = "1rc"
        n_star = max(4, int(np.floor(9.0 / lam_rc)) - 1)
    elif 2.4 <= lam_rc <= 3.6:
        hypothesis = "3rc"
        n_star = max(2, int(np.floor(9.0 / lam_rc)) - 1)

    counts = np.array([fr.n_billows for fr in win_frames], dtype=float)
    count_frac = (float((counts >= n_star).mean())
                  if (n_star is not None and len(win_frames)) else 0.0)
    count_ok = count_frac >= 0.80
    wavelength_ok = hypothesis is not None

    med_skew = float(np.median([abs(fr.skew) for fr in win_frames])) if win_frames else np.inf
    med_coh = float(np.median([fr.coh for fr in win_frames])) if win_frames else 0.0
    med_ratio = float(np.median([fr.rms_ratio for fr in win_frames])) if win_frames else 0.0
    polarity_ok = med_skew <= 0.2
    coherence_ok = 0.35 <= med_coh <= 0.55
    band_rms_ok = med_ratio >= 1.4

    # --- stationarity criteria ----------------------------------------------
    stat = dict(ok=False, half_diff=None, slope=None, slope_ci=None, lam_drift=None)
    if window_ok_len and len(win_frames) >= 4:
        h = len(win_frames) // 2
        c1 = counts[:h]
        c2 = counts[h:]
        half_diff = abs(np.median(c1) - np.median(c2))
        ts = stats.theilslopes(counts, win_steps)
        slope, lo, hi = ts[0], ts[2], ts[3]
        slope_zero = (lo <= 0.0 <= hi)
        lam1 = np.median([fr.wave.lam_rc for fr in win_frames[:h]])
        lam2 = np.median([fr.wave.lam_rc for fr in win_frames[h:]])
        lam_drift = abs(lam1 - lam2) / max(abs(lam1), _EPS)
        stat = dict(ok=bool(half_diff <= 1 and slope_zero and lam_drift <= 0.20),
                    half_diff=float(half_diff), slope=float(slope),
                    slope_ci=(float(lo), float(hi)), lam_drift=float(lam_drift))
    stationarity_ok = stat["ok"] and window_ok_len

    flank_ok = flank_val < cfg.FLANK_MAX
    cfl_ok = substep_count == 0

    reasons = dict(
        count=count_ok, wavelength=wavelength_ok, polarity=polarity_ok,
        coherence=coherence_ok, band_rms=band_rms_ok, stationarity=stationarity_ok,
        formation=formation_ok, flank=flank_ok, cfl_clean=cfl_ok,
    )
    gate1_pass = all(reasons.values())
    # Reviewer-requested adjudication flag: coherence adds no anti-status-quo
    # discrimination (the real baseline PASSES it under both conventions), so a
    # coherence-ONLY fail may be a false-FAIL of a cleaner-than-reference chain.
    # The frozen PASS definition is unchanged; Gate-6 branch analysis surfaces
    # this flag for USER adjudication instead of silently recording FAIL.
    coherence_only_fail = (not gate1_pass) and all(
        ok for name, ok in reasons.items() if name != "coherence") and not reasons["coherence"]

    return Gate1Result(
        gate1_pass=gate1_pass, reasons=reasons,
        coherence_only_fail=bool(coherence_only_fail), frames=frames, n_star=n_star,
        hypothesis=hypothesis, lam_rc=lam_rc, count_frac=count_frac,
        formation_step=formation_step, formation_bar=float(formation_bar),
        window_start=float(window_start), window_n=len(win_frames),
        stationarity=stat, flank=float(flank_val), substep_count=int(substep_count),
    )


# alias per the brief's naming
gate1_verdict = evaluate_run


# ============================================================================
# Synthetic band builders (self-test fixtures -- tune THESE, never thresholds)
# ============================================================================
def _corr_noise(ny, nx, scale_px, seed):
    """Unit-std spatially-correlated (Gaussian-smoothed) noise. Correlated (not
    white) noise perturbs LOCAL orientation over the coherence tensor window,
    pulling coherence down into the folded [0.35,0.55] band WITHOUT the sharp
    collapse white noise causes (white-noise coherence jumps to ~0.1-0.2 with
    no usable middle -- verified during tuning)."""
    nz = np.random.default_rng(np.random.SeedSequence([777, seed])).standard_normal((ny, nx))
    nz = ndimage.gaussian_filter(nz, scale_px)
    return nz / (nz.std() + _EPS)


# Fixture noise levels (fix round 1, re-tuned for the PINNED s=4 px coherence
# convention): each fixture = carrier + COARSE correlated noise (10-12 px --
# large-scale irregularity) + FINE correlated noise (3 px, amp 0.9 -- intra-rope
# texture at the coherence window's own scale, the physically-faithful analogue
# of the reference's "sub-0.4 r_ns fine texture within ropes" caveat). Under the
# pinned s=4 px window the coarse noise alone leaves the fields ~0.80-laminar;
# the 3 px fine texture is what lands them mid-[0.35,0.55]. Multi-seed validated
# (5 seed pairs each, all pass).
_FINE_SCALE_PX = 3.0
_FINE_AMP = 0.9


def _egg_carton_band(cfg, rms0=1.0):
    """CASE (a) fixture -- reference-scale rope chain: a 3.0 rc (along-x) x
    1.2 rc (transverse) folded egg-carton + coarse/fine correlated noise.
    Along-x carrier is a pure cos(2pi x/3rc) tone so the FFT peak lands cleanly
    in the 3-rc bracket despite the 0.5 rc high-pass halving the fundamental;
    the transverse cos(2pi y/1.2rc) breaks it into countable reference-scale
    billows. Locked (tuned; DO NOT change to fix a threshold): coh~0.46,
    lam=3.00 rc, ~30 billows, rmsR~2.6, skew~0."""
    ny, nx = band_dims(cfg)
    ppc = _px_per_rc(cfg)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x_rc = xx / ppc
    y_rc = (yy - ny / 2.0) / ppc
    fold = 0.35
    f = 4.0 * rms0 * (np.cos(2 * np.pi * x_rc / 3.0 + fold * np.sin(2 * np.pi * y_rc / 1.2))
                      * np.cos(2 * np.pi * y_rc / 1.2))
    f += 1.5 * rms0 * _corr_noise(ny, nx, 10.0, seed=101)
    f += _FINE_AMP * rms0 * _corr_noise(ny, nx, _FINE_SCALE_PX, seed=151)
    return f


def _blob_chain_band(cfg, rms0=1.0):
    """CASE (d) fixture -- a sparse 3.0-rc-spaced chain: a single-transverse-row
    symmetric cos(2pi x/3rc) carrier + coarse/fine correlated noise. The
    full-band carrier (vs 3 isolated blobs, which the correlated noise's low-k
    content beats and flips lam to the 9-rc band mode -- verified during tuning)
    makes the 3.0 rc FFT peak ROBUST across seeds; a single Gaussian transverse
    row makes it the SPARSE chain that tests the 2nd bracket's low count bar
    (N*=2). Multi-seed validated (5 seed pairs, all pass). Locked: coh~0.42,
    lam=3.00 rc, ~17 billows (>>N*=2), rmsR~2.1, skew~0."""
    ny, nx = band_dims(cfg)
    ppc = _px_per_rc(cfg)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    x_rc = xx / ppc
    y_rc = (yy - ny / 2.0) / ppc
    f = 5.0 * rms0 * (np.cos(2 * np.pi * x_rc / 3.0 + 0.35 * np.sin(2 * np.pi * y_rc / 1.2))
                      * np.exp(-(y_rc ** 2) / (2 * 0.9 ** 2)))
    f += 1.4 * rms0 * _corr_noise(ny, nx, 12.0, seed=123)
    f += _FINE_AMP * rms0 * _corr_noise(ny, nx, _FINE_SCALE_PX, seed=171)
    return f


def _status_quo_band(cfg):
    """CASE (c) fixture: extract the REAL production wake band from
    wakeA/baseline/color.png and resample it to sim pixel scale.

    Wake box convention (documented): the production wake trails WEST of the
    hero (post-chirality), matching mock_billows.py's westward chain. Box = the
    hero-centred band x in [3,12] rc WEST of the hero core, |y - hero| < 1.2 rc
    -- the same 3..12 rc x-extent and +-1.2 rc transverse half-width as the sim
    scoring band (config.BAND_X / BAND_HW), anchored on the render's detected
    hero instead of Y_SHEET. rms0 is the AMBIENT (moat/leading) hp-RMS so the
    wake's band-RMS ratio reproduces the reference's ~1.5x (band-RMS is NOT the
    intended killer; the max-size / wavelength criteria are).

    Returns (tr_band, rms0)."""
    import cv2
    png = SPIKE.parent / "wakeA" / "baseline" / "color.png"
    img = cv2.imread(str(png))
    assert img is not None, f"missing {png}"
    H, W = img.shape[:2]
    lum = img[..., :3].astype(np.float64) @ np.array([0.114, 0.587, 0.299])  # BGR
    lat0 = cfg.LAT0
    px_ns = H / np.pi
    px_ew = W / (2.0 * np.pi * np.cos(lat0))
    rc_ns = cfg.RC * px_ns
    rc_ew = cfg.RC * px_ew
    # hero detection (redness blob near equirect (lat0, lon=0) = image centre col)
    hy = (90.0 - np.degrees(lat0)) / 180.0 * H
    hx = W / 2.0
    x0, x1 = int(hx - 70), int(hx + 70)
    y0, y1 = int(hy - 40), int(hy + 40)
    sub = img[y0:y1, x0:x1].astype(np.float64)
    red = sub[..., 2] - sub[..., 0]
    thr = np.median(red) + 0.32 * (np.percentile(red, 92) - np.median(red))
    lab, n = ndimage.label(red > thr, structure=_8CONN)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        i = 1 + int(np.argmax(sizes))
        ys, xs = np.nonzero(lab == i)
        hx = xs.mean() + x0
        hy = ys.mean() + y0
    band_ny, band_nx = band_dims(cfg)

    def _resampled(x_lo, x_hi, y_c):
        yb0 = int(round(y_c - 1.2 * rc_ns))
        yb1 = int(round(y_c + 1.2 * rc_ns))
        box = lum[yb0:yb1, int(round(x_lo)):int(round(x_hi))]
        return cv2.resize(box, (band_nx, band_ny), interpolation=cv2.INTER_AREA)

    # wake band: 3..12 rc WEST of the hero core
    wake = _resampled(hx - 12.0 * rc_ew, hx - 3.0 * rc_ew, hy)
    # ambient (moat/leading) band: symmetric 3..12 rc EAST (upstream) for rms0
    lead = _resampled(hx + 3.0 * rc_ew, hx + 12.0 * rc_ew, hy)
    rms0 = float(np.sqrt(np.mean(hp(lead, cfg) ** 2)))
    return wake, rms0


# ============================================================================
# SELF-TEST (mandatory -- run before any Gate-1 run)
# ============================================================================
def selftest(make_png=True):
    cfg = _config
    ny, nx = band_dims(cfg)
    results = {}

    # ---- (a) reference-scale rope chain: 3.0 rc along-x, 1.2 rc transverse ----
    sa = score_band(_egg_carton_band(cfg, rms0=1.0), cfg, rms0=1.0, step=0)
    results["a"] = sa

    # ---- (b) white noise -> FAIL ----
    rng = np.random.default_rng(np.random.SeedSequence([777, 202]))
    band_b = rng.standard_normal((ny, nx))
    rms0_b = float(np.sqrt(np.mean(hp(band_b, cfg) ** 2)))   # its own level
    sb = score_band(band_b, cfg, rms0=rms0_b, step=0)
    results["b"] = sb

    # ---- (c) THE REAL STATUS QUO -> must FAIL ----
    band_c, rms0_c = _status_quo_band(cfg)
    sc = score_band(band_c, cfg, rms0=rms0_c, step=0)
    results["c"] = sc

    # ---- (d) 3-billow 3.0-rc chain -> PASS via 2nd interval, N* >= 2 ----
    sd = score_band(_blob_chain_band(cfg, rms0=1.0), cfg, rms0=1.0, step=0)
    results["d"] = sd

    # ---- verdict assertions -------------------------------------------------
    def _rep(tag, s, want):
        line = (f"({tag}) n_billows={s.n_billows} lam={s.wave.lam_rc:.2f}rc "
                f"hyp={s.wave.hypothesis} N*={s.wave.n_star} skew={s.skew:+.3f} "
                f"coh={s.coh:.3f} rmsR={s.rms_ratio:.2f} "
                f"oversize={s.seg['n_oversize']} raw={s.seg['n_raw']} "
                f"per_frame_pass={s.per_frame_pass} fail={s.failing()}")
        return line

    verdicts = {}
    # (a) per-frame criteria must all pass (single frame -> no stationarity/formation)
    verdicts["a"] = sa.per_frame_pass and sa.wave.hypothesis == "3rc"
    # (b) must FAIL
    verdicts["b"] = not sb.per_frame_pass
    # (c) status quo must FAIL
    verdicts["c"] = not sc.per_frame_pass
    # (d) must pass via 2nd interval with N* >= 2
    verdicts["d"] = (sd.per_frame_pass and sd.wave.hypothesis == "3rc"
                     and sd.wave.n_star is not None and sd.wave.n_star >= 2)

    print("=" * 74)
    print("SELF-TEST (Task 5 metrics.py)")
    print("=" * 74)
    for tag, s in results.items():
        print(_rep(tag, s, None))
    print("-" * 74)
    print(f"(a) reference rope chain PASSES per-frame : {verdicts['a']}")
    print(f"(b) white noise FAILS                     : {verdicts['b']}  "
          f"killed by {sb.failing()}")
    print(f"(c) REAL STATUS QUO FAILS                 : {verdicts['c']}  "
          f"killed by {sc.failing()}")
    print(f"(d) 3-billow 3rc chain PASSES (N*>=2)     : {verdicts['d']}")
    all_ok = all(verdicts.values())
    print("-" * 74)
    print("SELF-TEST:", "PASS" if all_ok else "FAIL", verdicts)

    if make_png:
        _selftest_png(results, verdicts)
    return results, verdicts, all_ok


def _selftest_png(results, verdicts):
    """Diagnostic panel (cv2 -- no matplotlib dependency): the four cases stacked,
    each showing the hp band (gray) with kept reference-scale billows outlined in
    green + a caption of the measured criteria and the verdict."""
    import cv2

    tags = ["a", "b", "c", "d"]
    titles = {
        "a": "(a) reference rope chain 3.0rc  [want PASS]",
        "b": "(b) white noise                 [want FAIL]",
        "c": "(c) REAL status quo baseline wake[want FAIL]",
        "d": "(d) 3-billow 3.0rc chain         [want PASS]",
    }
    tiles = []
    for tag in tags:
        s = results[tag]
        hpb = s.hpb
        lo, hi = np.percentile(hpb, 2), np.percentile(hpb, 98)
        g = np.clip((hpb - lo) / max(hi - lo, _EPS), 0, 1)
        img = (np.dstack([g, g, g]) * 255).astype(np.uint8)  # BGR gray
        # outline kept billows in green
        lab = s.labels
        if lab is not None and lab.max() > 0:
            edges = (lab > 0).astype(np.uint8)
            er = ndimage.binary_erosion(edges, iterations=1)
            outline = edges & (~er)
            img[outline.astype(bool)] = (0, 255, 0)
        verd = "PASS" if verdicts[tag] else "FAIL"
        cap = (f"{titles[tag]}  nb={s.n_billows} lam={s.wave.lam_rc:.2f} "
               f"hyp={s.wave.hypothesis} N*={s.wave.n_star} coh={s.coh:.2f} "
               f"skew={s.skew:+.2f} rmsR={s.rms_ratio:.2f} ovr={s.seg['n_oversize']}")
        cap2 = (f"per_frame={s.per_frame_pass} fail={s.failing()} ==> verdict {verd}")
        bar = np.full((46, img.shape[1], 3), 25, np.uint8)
        img = np.vstack([bar, img])
        cv2.putText(img, cap, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, cap2, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (120, 255, 120) if verd == "PASS" else (120, 120, 255), 1, cv2.LINE_AA)
        tiles.append(img)
        tiles.append(np.full((6, img.shape[1], 3), 60, np.uint8))
    panel = np.vstack(tiles[:-1])
    out = SPIKE / "metrics_selftest.png"
    cv2.imwrite(str(out), panel)
    print(f"diagnostic image -> {out}")


if __name__ == "__main__":
    selftest()

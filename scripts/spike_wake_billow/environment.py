"""Task 4 -- the EXPERIMENT ENVIRONMENT for the billow-rollup spike.

Spatially-developing "fed" configuration on the doubly-periodic pseudo-spectral
box (solver.py). Downstream (+x) is left-to-right:

    x=0 .............................................................. x=LX (wrap)
      | upstream | STRIP | ......... SCORING BAND ......... | ... | SPONGE |
                 (Dirichlet inflow)                                (relax->ambient)

  * STRIP (Dirichlet, 4 cells): every step w and tr are HARD-SET to
    ambient + sheet-target + fresh band-limited noise. This is the ONLY
    source of the vorticity sheet -- no relaxation feeding in the measurement
    domain (review F3).
  * MEAN-HOLD: a zonal-mean-only nudge of w back to the ambient profile,
    applied OUTSIDE the strip, per-step fraction MEANHOLD_RATE. Auto-DISABLED
    when the Task-7 nudge flag is on (that nudge already holds the mean -- both
    on = double-damping, review F5/M8).
  * SPONGE (last SPONGE_X before the x-wrap): cosine-ramped relaxation of
    (w - w_amb) and (tr - tr_amb) toward 0, peak per-step fraction SPONGE_RATE.
    Kills advected structure before it wraps around to re-enter upstream.

All constants come from config.py (the single source; frozen at Task-4 sign-off).
Every rate here is a PER-STEP FRACTION. Nothing here touches src/gasgiant/** or git.
"""
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

SPIKE = pathlib.Path(__file__).parent
sys.path.insert(0, str(SPIKE))
import config  # noqa: E402  (spike-local single-source constants)

NX, NY, DX = config.NX, config.NY, config.DX
LX, LY = config.LX, config.LY

# Cell-centered coordinates (match config.warm_profile_window / run_gate0a).
_Y = (np.arange(NY) + 0.5) * DX     # (NY,) rows
_X = (np.arange(NX) + 0.5) * DX     # (NX,) cols

# ---- Ambient fields (zonally uniform) --------------------------------------
# w_amb(y): the warm-profile vorticity (omega = -du/dy) broadcast over x.
_Y_PROF, _U_PROF, _OMEGA_PROF = config.warm_profile_window()
W_AMB_COL = _OMEGA_PROF.copy()                      # (NY,)
W_AMB_2D = np.broadcast_to(W_AMB_COL[:, None], (NY, NX))
# tr_amb(y) = tanh((y - Y_SHEET)/TR_STEP_W): a belt/zone step across the sheet.
TR_AMB_COL = np.tanh((_Y - config.Y_SHEET) / config.TR_STEP_W)   # (NY,)
TR_AMB_2D = np.broadcast_to(TR_AMB_COL[:, None], (NY, NX))

# ---- Region masks (x = cols, y = rows) -------------------------------------
STRIP_MASK = (_X >= config.STRIP_X[0]) & (_X < config.STRIP_X[1])    # 4 cols
NONSTRIP_COLS = np.where(~STRIP_MASK)[0]
SPONGE_MASK = _X >= (LX - config.SPONGE_X)                            # last cols
BAND_X_MASK = (_X >= config.BAND_X[0]) & (_X <= config.BAND_X[1])
BAND_Y_MASK = np.abs(_Y - config.Y_SHEET) < config.BAND_HW
FLANK_LO_MASK = _Y < config.FLANK_W                                   # near y=0
FLANK_HI_MASK = _Y > (LY - config.FLANK_W)                            # near y=LY

# Cosine ramp for the sponge: 0 at the upstream edge -> SPONGE_RATE at the wrap.
_sp_x = _X[SPONGE_MASK]
_sp_arg = np.pi * (_sp_x - (LX - config.SPONGE_X)) / config.SPONGE_X
SPONGE_FACTOR = config.SPONGE_RATE * 0.5 * (1.0 - np.cos(_sp_arg))    # (n_sponge,)

# y-flank EDDY sponges (Task-4 controller decision, config.py:FLANK_SPONGE_*):
# cosine-ramped 0 at the interior edge -> FLANK_SPONGE_RATE at each y wall,
# acting on eddy components only ((w - zonal_mean(w)) and (tr - tr_amb)) —
# domain-openness infrastructure: the doubly-periodic box retains beta-radiated
# waves and fed turbulence that an open domain sheds; the flanks are declared
# inert buffers, so sponging them enforces the declared geometry. The zonal
# MEAN stays governed by the mean-hold.
_fl_ramp = np.zeros(NY)
_lo = _Y < config.FLANK_SPONGE_W
_hi = _Y > (LY - config.FLANK_SPONGE_W)
_fl_ramp[_lo] = 0.5 * (1.0 - np.cos(
    np.pi * (config.FLANK_SPONGE_W - _Y[_lo]) / config.FLANK_SPONGE_W))
_fl_ramp[_hi] = 0.5 * (1.0 - np.cos(
    np.pi * (_Y[_hi] - (LY - config.FLANK_SPONGE_W)) / config.FLANK_SPONGE_W))
FLANK_SPONGE_FACTOR = config.FLANK_SPONGE_RATE * _fl_ramp             # (NY,)
FLANK_SPONGE_ROWS = np.where(_fl_ramp > 0.0)[0]                       # rows touched

# Band-limited noise mask (k < pi/(4 DX) in BOTH directions), rfft2 layout.
_KX = np.fft.rfftfreq(NX, d=DX) * 2.0 * np.pi        # (NX//2+1,) >= 0
_KY = np.fft.fftfreq(NY, d=DX) * 2.0 * np.pi         # (NY,) signed
_KCUT = np.pi / (4.0 * DX)
NOISE_MASK = (_KX[None, :] < _KCUT) & (np.abs(_KY[:, None]) < _KCUT)
_N_NOISE_MODES = int(NOISE_MASK.sum())


@dataclass
class EnvConfig:
    """Environment / sheet-target configuration. Minimal; Task 7 extends `nudge`."""
    A: float = 45.0                     # sheet-target vorticity amplitude
    delta: float = 0.30 * config.RC     # sheet half-thickness (s = (y-Y_SHEET)/delta)
    sheet_form: str = "deficit"         # 'deficit' (A*s*exp(-s^2)) | 'tanh' (-A*sech^2)
    nudge: bool = False                 # Task-7 production nudge; disables MEAN-HOLD
    # Strip-noise stream id (Task 6): the rng is SeedSequence([777, seed_id]).
    # None keeps the Task-4 default (config.SEEDS["seed_noise"]=13) so the smoke
    # __main__ is byte-identical; Gate-1 passes SEEDS["gate1"]/["gate1_replicate"].
    seed_id: int | None = None
    # y-envelope of the strip noise, sigma in units of delta; 0.0 = full-height.
    # CONTROLLER DECISION (smoke rev 1 -> rev 2): the rev-1 full-height strip
    # noise seeded the AMBIENT profile's own barotropic instability at every
    # latitude (Kuo criterion beta - u_yy changes sign 15x across the window;
    # mid-domain |du/dy| up to 31.6 -> eddy growth ~4e-3/step > sponge x-avg
    # damping 2.6e-3/step) -> domain-wide turbulence, flank_guard 1.1 >> 0.10.
    # The noise's job is to seed the SHEET instability, so it is confined to a
    # Gaussian envelope around Y_SHEET (hard-zeroed beyond 3 sigma; the flanks
    # get exactly zero direct seed). Spectral band-limit unchanged.
    noise_env_sigma: float = 2.0        # * delta; hard cutoff at 3*sigma


def _sech2(s):
    return 1.0 / np.cosh(s) ** 2


def w_sheet(cfg):
    """Sheet-target vorticity column w_sheet(y). Source: task-4-brief.md:9.

    'deficit' (primary, E2 zero-mean deficit jet): A*s*exp(-s^2). Zero net
      vorticity across the sheet -> a shear layer with no mean-circulation bias.
    'tanh' (variant, single-signed intensification): -A*sech^2(s). SIGN is
      NEGATIVE (controller decision): the ambient vorticity at the sheet line is
      ~ -26 (uniform-sign south shear flank, config.py:33-39); a negative
      sech^2 bump DEEPENS the existing shear line rather than opposing it.
    """
    s = (_Y - config.Y_SHEET) / cfg.delta
    if cfg.sheet_form == "deficit":
        return cfg.A * s * np.exp(-s * s)
    if cfg.sheet_form == "tanh":
        return -cfg.A * _sech2(s)             # single-signed, matches ambient sign
    raise ValueError(f"unknown sheet_form {cfg.sheet_form!r}")


def init_state(box, cfg):
    """Set box.w to the ambient vorticity profile and box.tr to the ambient
    tracer step. Both are zonally uniform (no sheet yet -- the strip injects it)."""
    box.w = W_AMB_2D.copy()
    box.tr = TR_AMB_2D.copy()


def make_forcing(cfg):
    """Build the per-step forcing closure forcing(box, step).

    Order per step: (1) STRIP hard-set, (2) MEAN-HOLD outside strip,
    (3) SPONGE. Applied ONCE per production step (solver.py:168-171); every
    rate is a per-step fraction.
    """
    # Deterministic strip-noise stream: one rng, advanced per step (F4). The
    # stream id comes from EnvConfig.seed_id (Task 6 threads SEEDS["gate1"]/
    # ["gate1_replicate"]); None preserves the Task-4 SEEDS["seed_noise"] default.
    _stream_id = config.SEEDS["seed_noise"] if cfg.seed_id is None else cfg.seed_id
    rng = np.random.default_rng(np.random.SeedSequence([777, int(_stream_id)]))
    noise_amp = config.SEED_AMP_FRAC * cfg.A
    w_sheet_col = w_sheet(cfg)                       # (NY,)
    w_strip_target = W_AMB_COL + w_sheet_col         # (NY,) ambient + sheet
    meanhold_on = not cfg.nudge

    # y-envelope for the strip noise (see EnvConfig.noise_env_sigma comment).
    if cfg.noise_env_sigma > 0.0:
        sig = cfg.noise_env_sigma * cfg.delta
        dy = _Y - config.Y_SHEET
        noise_env_col = np.exp(-0.5 * (dy / sig) ** 2)
        noise_env_col[np.abs(dy) > 3.0 * sig] = 0.0   # zero direct flank seed
    else:
        noise_env_col = np.ones(NY)                    # full-height (rev-1)

    # ---- log effective per-step damping rates (once, at build) -------------
    mean_damp = config.MEANHOLD_RATE if meanhold_on else 0.0
    print("[make_forcing] environment damping rates (per-step fractions):")
    print(f"  STRIP     : Dirichlet hard-set over {int(STRIP_MASK.sum())} cols "
          f"(x in {config.STRIP_X}); not a relaxation rate")
    print(f"  MEAN-HOLD : mean-damp={mean_damp:.5g} (=1/{config.TAU_STEPS:g}), "
          f"eddy-damp=0  [{'ON' if meanhold_on else 'DISABLED (nudge flag)'}]  "
          f"outside strip")
    print(f"  SPONGE    : peak rate={config.SPONGE_RATE:.5g} (=1/20) on mean AND "
          f"eddy, cosine-ramped over last {int(SPONGE_MASK.sum())} cols "
          f"(SPONGE_X={config.SPONGE_X:.4f})")
    print(f"  FLANK SPG : peak rate={config.FLANK_SPONGE_RATE:.5g} (=1/20) on "
          f"EDDY only (mean-damp=0), cosine-ramped over "
          f"{int((_Y < config.FLANK_SPONGE_W).sum())} rows at EACH y edge "
          f"(FLANK_SPONGE_W={config.FLANK_SPONGE_W:.4f})")
    print(f"  cfg: A={cfg.A} delta={cfg.delta:.5f} ({cfg.delta/config.RC:.3g} RC) "
          f"form={cfg.sheet_form} nudge={cfg.nudge}  noise_amp={noise_amp:.4g} "
          f"noise_env_sigma={cfg.noise_env_sigma}*delta (0=full-height)")

    def forcing(box, step):
        # (1) STRIP -- fresh band-limited noise, regenerated every step. Built
        #     directly in spectral space (random amplitudes on the allowed
        #     |kx|,|ky| < pi/(4 DX) modes, one inverse FFT) -- same construction
        #     as solver.py:_random_band_limited_w, ~2x cheaper than FFT'ing a
        #     full white field.
        w_hat = np.zeros(NOISE_MASK.shape, dtype=np.complex128)
        draw = rng.standard_normal((2, _N_NOISE_MODES))   # fixed count -> deterministic
        w_hat[NOISE_MASK] = draw[0] + 1j * draw[1]
        noise = np.fft.irfft2(w_hat, s=(NY, NX))
        noise *= noise_amp / (noise.std() + 1e-30)   # RMS at sheet = SEED_AMP_FRAC*A
        noise *= noise_env_col[:, None]              # y-confined to sheet vicinity
        box.w[:, STRIP_MASK] = (w_strip_target[:, None]
                                + noise[:, STRIP_MASK])
        box.tr[:, STRIP_MASK] = TR_AMB_COL[:, None]

        # (2) MEAN-HOLD -- zonal-mean-only nudge toward ambient, outside strip.
        if meanhold_on:
            corr = config.MEANHOLD_RATE * (box.w.mean(axis=1) - W_AMB_COL)  # (NY,)
            box.w[:, NONSTRIP_COLS] -= corr[:, None]

        # (3) SPONGE -- cosine-ramped relaxation of the perturbation to 0.
        box.w[:, SPONGE_MASK] -= SPONGE_FACTOR[None, :] * (
            box.w[:, SPONGE_MASK] - W_AMB_COL[:, None])
        box.tr[:, SPONGE_MASK] -= SPONGE_FACTOR[None, :] * (
            box.tr[:, SPONGE_MASK] - TR_AMB_COL[:, None])

        # (4) FLANK SPONGES -- eddy-only relaxation in the y-edge buffers
        #     (see FLANK_SPONGE_FACTOR comment; zonal mean untouched).
        r = FLANK_SPONGE_ROWS
        fac = FLANK_SPONGE_FACTOR[r][:, None]
        w_fl = box.w[r, :]
        box.w[r, :] = w_fl - fac * (w_fl - w_fl.mean(axis=1, keepdims=True))
        box.tr[r, :] -= fac * (box.tr[r, :] - TR_AMB_COL[r][:, None])

    return forcing


def flank_guard(box):
    """Max over the two y-boundary FLANK_W strips of the eddy-enstrophy fraction:
    rms(eddy) in a flank strip / rms(eddy) in the scoring band, where
    eddy = w - zonal_mean(w). A run is VOID if this exceeds config.FLANK_MAX
    during the scoring window (structure leaking to the y-boundaries)."""
    eddy = box.w - box.w.mean(axis=1, keepdims=True)
    e2 = eddy * eddy
    band_rms = np.sqrt(e2[np.ix_(BAND_Y_MASK, BAND_X_MASK)].mean())
    lo_rms = np.sqrt(e2[FLANK_LO_MASK, :].mean())
    hi_rms = np.sqrt(e2[FLANK_HI_MASK, :].mean())
    return max(lo_rms, hi_rms) / (band_rms + 1e-30)


# ===========================================================================
# Step 2 -- SMOKE RUN + constant freeze.
# ===========================================================================
def _tracer_front_x(box):
    """x-position of the CONTIGUOUS tracer-perturbation tongue advancing from
    the strip: the first column downstream of the strip (within the band's
    y-window) whose perturbation rms drops below threshold. Contiguity keeps
    remote instability-spawned perturbations from inflating the front."""
    tr_eddy = box.tr - TR_AMB_2D
    p_col = np.sqrt((tr_eddy[BAND_Y_MASK, :] ** 2).mean(axis=0))   # (NX,)
    start = int(np.where(STRIP_MASK)[0].max()) + 1
    below = np.where(p_col[start:] <= 0.02)[0]
    if below.size == 0:
        return _X[NX - 1]
    return _X[start + below[0]]


def _upstream_eddy_rms(box):
    """Eddy rms just UPSTREAM of the strip (x < STRIP_X[0]). Because the domain
    is periodic, this is where sponge-killed material re-enters -- it must stay
    ~ambient (near zero eddy)."""
    up = _X < config.STRIP_X[0]
    eddy = box.w - box.w.mean(axis=1, keepdims=True)
    return np.sqrt((eddy[:, up] ** 2).mean())


def _band_eddy_rms(box):
    eddy = box.w - box.w.mean(axis=1, keepdims=True)
    return np.sqrt((eddy[np.ix_(BAND_Y_MASK, BAND_X_MASK)] ** 2).mean())


def _make_figure_cv2(snaps, out):
    """Matplotlib-free fallback: stack w (RdBu-ish) and tr (gray) snapshot rows
    with region boundary lines, save via cv2 (spike tech stack)."""
    import cv2

    def _to_img(f, vmax, signed=True):
        if signed:
            g = np.clip(f / (2 * vmax) + 0.5, 0, 1)
        else:
            g = np.clip((f + 1) / 2, 0, 1)
        img = np.empty(f.shape + (3,), np.uint8)
        img[..., 2] = (255 * g).astype(np.uint8)          # R
        img[..., 1] = (255 * (1 - np.abs(2 * g - 1))).astype(np.uint8)
        img[..., 0] = (255 * (1 - g)).astype(np.uint8)    # B
        return img[::-1]                                   # origin lower

    cols = []
    for step, w, tr in snaps:
        wv = np.abs(w).max() + 1e-30
        panel = np.concatenate([_to_img(w, wv), _to_img(tr, 1.0)], axis=0)
        for xr in (config.STRIP_X[0], config.STRIP_X[1], config.BAND_X[0],
                   config.BAND_X[1], LX - config.SPONGE_X):
            c = int(xr / DX)
            panel[:, c] = (0, 255, 0)
        cv2.putText(panel, f"step {step}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cols.append(panel)
        cols.append(np.full((panel.shape[0], 4, 3), 255, np.uint8))
    cv2.imwrite(str(out), np.concatenate(cols[:-1], axis=1))
    print(f"figure (cv2 fallback) -> {out}")


def _make_figure(snaps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(18, 8))
    ext = [0, LX, 0, LY]
    for j, (step, w, tr) in enumerate(snaps):
        wv = np.abs(w).max()
        ax[0, j].imshow(w, origin="lower", cmap="RdBu_r", vmin=-wv, vmax=wv,
                        extent=ext, aspect="auto")
        ax[0, j].set_title(f"w  (step {step})")
        ax[1, j].imshow(tr, origin="lower", cmap="PuOr", vmin=-1, vmax=1,
                        extent=ext, aspect="auto")
        ax[1, j].set_title(f"tr (step {step})")
        for a in (ax[0, j], ax[1, j]):
            # strip (green), band (black), sponge start (red)
            a.axvline(config.STRIP_X[0], color="g", lw=1.2)
            a.axvline(config.STRIP_X[1], color="g", lw=1.2)
            a.axvline(config.BAND_X[0], color="k", lw=1.0, ls="--")
            a.axvline(config.BAND_X[1], color="k", lw=1.0, ls="--")
            a.axvline(LX - config.SPONGE_X, color="r", lw=1.2)
            a.axhline(config.Y_SHEET + config.BAND_HW, color="k", lw=0.8, ls=":")
            a.axhline(config.Y_SHEET - config.BAND_HW, color="k", lw=0.8, ls=":")
            a.axhline(config.FLANK_W, color="m", lw=0.8, ls=":")
            a.axhline(LY - config.FLANK_W, color="m", lw=0.8, ls=":")
            a.set_xlabel("x (rad, downstream ->)")
            a.set_ylabel("y (rad)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"figure -> {out}")


if __name__ == "__main__":
    import time

    from solver import Box  # noqa: E402

    u_sheet, strip_to_band, band_fill = config.transit_report()
    band_transit = band_fill                        # band_fill IS the band-crossing time
    n_steps = int(max(3 * config.DEV_STEPS, round(band_fill + 2.0 * band_transit)))
    # rev-3 extension (controller): Gate-1 stationarity needs fill + 2000+
    # (~3311+) steps of healthy field -- run 5500 so the tail is demonstrated.
    n_steps = max(n_steps, 5500)
    print("=" * 70)
    print("TASK 4 SMOKE RUN")
    print("=" * 70)
    print(f"u(sheet)={u_sheet:.4f}  strip->band={strip_to_band:.0f}  "
          f"band_fill={band_fill:.0f}  n_steps={n_steps}  "
          f"(3*DEV_STEPS={3*config.DEV_STEPS})")

    cfg = EnvConfig(A=45.0, delta=0.30 * config.RC, sheet_form="deficit")
    box = Box()                                     # screened, beta=config.BETA
    init_state(box, cfg)
    forcing = make_forcing(cfg)

    snap_steps = [int(round(strip_to_band)),
                  int(round(strip_to_band + band_fill)),
                  n_steps]
    snaps = []

    fronts, sample_steps = [], []
    cfls, flanks, up_rms, bd_rms = [], [], [], []
    sample_every = 20
    prog = open(SPIKE / "environment_smoke_progress.log", "w", buffering=1)

    t0 = time.time()
    for n in range(n_steps + 1):
        if n % sample_every == 0:
            sample_steps.append(n)
            fronts.append(_tracer_front_x(box))
            cfls.append(box.cfl())
            flanks.append(flank_guard(box))
            up_rms.append(_upstream_eddy_rms(box))
            bd_rms.append(_band_eddy_rms(box))
            prog.write(f"step {n:5d}/{n_steps}  front={fronts[-1]:.4f} "
                       f"cfl={cfls[-1]:.3f} flank={flanks[-1]:.3f} "
                       f"up_rms={up_rms[-1]:.3e} band_rms={bd_rms[-1]:.3e} "
                       f"substeps={box.substep_count} "
                       f"elapsed={time.time()-t0:.0f}s\n")
        if n in snap_steps:
            snaps.append((n, box.w.copy(), box.tr.copy()))
        if n < n_steps:
            box.step_spectral(forcing_fn=lambda b: forcing(b, n))
    wall = time.time() - t0
    prog.close()

    sample_steps = np.array(sample_steps)
    fronts = np.array(fronts)
    cfls = np.array(cfls)
    flanks = np.array(flanks)
    up_rms = np.array(up_rms)
    bd_rms = np.array(bd_rms)
    t_samples = sample_steps * config.DT

    # (a) front speed: fit x = u*t + c over the rising window (front between the
    #     strip exit and the band's downstream edge).
    rising = (fronts > config.STRIP_X[1] + 2 * DX) & (fronts < config.BAND_X[1])
    if rising.sum() >= 2:
        slope = np.polyfit(t_samples[rising], fronts[rising], 1)[0]
    else:
        slope = float("nan")

    # scoring windows: (i) from first band arrival; (ii) from band FILLED
    scoring = sample_steps >= strip_to_band
    flank_max_scoring = flanks[scoring].max() if scoring.any() else float("nan")
    filled = sample_steps >= (strip_to_band + band_fill)
    flank_max_filled = flanks[filled].max() if filled.any() else float("nan")

    print("-" * 70)
    print(f"(a) FRONT: fitted speed = {slope:.4f}  vs u(sheet)={u_sheet:.4f}  "
          f"(rel err {abs(slope-u_sheet)/u_sheet*100:.1f}%); "
          f"front reached x={fronts.max():.4f} (band end={config.BAND_X[1]:.4f}, "
          f"sponge start={LX-config.SPONGE_X:.4f})")
    print(f"(b) SPONGE: upstream eddy rms (last half) mean="
          f"{up_rms[len(up_rms)//2:].mean():.4e}  band eddy rms mean="
          f"{bd_rms[len(bd_rms)//2:].mean():.4e}  ratio="
          f"{up_rms[len(up_rms)//2:].mean()/(bd_rms[len(bd_rms)//2:].mean()+1e-30):.4e}")
    print(f"(c) FLANK_GUARD: max over scoring window (band-arrival) = "
          f"{flank_max_scoring:.4f}, (band-filled) = {flank_max_filled:.4f}  "
          f"(bar < {config.FLANK_MAX}); global max = {flanks.max():.4f}")
    print(f"(d) CFL: substep_count = {box.substep_count}  "
          f"cfl min/mean/max = {cfls.min():.4f}/{cfls.mean():.4f}/{cfls.max():.4f}")
    print(f"(e) WALL: {wall:.1f}s total, {wall/n_steps*1000:.2f}s per 1000 steps "
          f"({wall/n_steps*1000/1000:.4f}s/step)")

    # data FIRST (snapshots included so the figure can always be regenerated)
    np.savez_compressed(SPIKE / "environment_smoke.npz",
                        sample_steps=sample_steps, fronts=fronts, cfls=cfls,
                        flanks=flanks, up_rms=up_rms, bd_rms=bd_rms,
                        slope=slope, u_sheet=u_sheet, wall=wall,
                        substep_count=box.substep_count,
                        snap_steps=np.array([s for s, _, _ in snaps]),
                        snap_w=np.array([w for _, w, _ in snaps]),
                        snap_tr=np.array([t for _, _, t in snaps]))
    print("smoke data -> environment_smoke.npz")

    try:
        _make_figure(snaps, SPIKE / "environment_smoke.png")
    except Exception as exc:
        print(f"matplotlib figure failed ({exc!r}); using cv2 fallback")
        _make_figure_cv2(snaps, SPIKE / "environment_smoke.png")

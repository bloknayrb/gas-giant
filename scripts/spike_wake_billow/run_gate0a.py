"""Gate 0a -- known-answer STOP GATE for the billow spike pseudo-spectral solver.

Step 1: tanh-layer Kelvin-Helmholtz. Box(beta=0, screened=False).
  - Base omega_bar(y) = -(U0/delta) sech^2((y-LY/2)/delta), U0=1, delta=8*DX.
    (vorticity of u(y) = U0 tanh((y-LY/2)/delta)).
  - Seed box harmonic m=9 (k = 2*pi*9/LX, k*delta ~ 0.442) as a y-localized
    vorticity perturbation w' = A cos(k x) sech^2(s), s=(y-LY/2)/delta,
    A = 1e-4 * (U0/delta) (1e-4 relative to peak base vorticity).
  - Growth rate: track m=9 x-Fourier amplitude of omega; fit exp growth over
    the window where amplitude is in [1e-3, 1e-1] * saturation.
    PASS (FROZEN): sigma within 20% of sigma_theory = 0.1897*U0/delta.
  - Rollup spacing at FIRST saturation (argmax of m=9 amp): dominant x-mode.
    PASS (FROZEN): within 30% of lambda = 2*pi/k_m ~= 14.3*delta.

Step 2: screening check. Box(screened=True, beta=0). Static Gaussian blob
  (sigma=3*DX) at center, invert, azimuthally-averaged |u_theta|(r) vs C*K1(r/L_d).
  Fit C over r in [0.05, 0.2] rad. PASS (FROZEN): RMS relative deviation < 10%.

Nothing here touches src/gasgiant/** or git.
"""
import pathlib
import sys
import time

import numpy as np
from scipy.special import k1 as bessel_k1

SPIKE = pathlib.Path(__file__).parent
sys.path.insert(0, str(SPIKE))
import config  # noqa: E402
from solver import Box  # noqa: E402

U0 = 1.0
DELTA = 8.0 * config.DX
NX, NY, DX, DT = config.NX, config.NY, config.DX, config.DT
LX, LY = config.LX, config.LY
SIGMA_THEORY = 0.1897 * U0 / DELTA           # ~15.46 / time-unit
K_M = 2.0 * np.pi * 9.0 / LX                 # m=9 box harmonic
LAMBDA_M = 2.0 * np.pi / K_M                 # = LX/9 ~ 14.22*delta


# ---------------------------------------------------------------- geometry
_y = (np.arange(NY) + 0.5) * DX              # cell-centered y (rows)
_x = (np.arange(NX) + 0.5) * DX              # cell-centered x (cols)
_S = (_y - LY / 2.0) / DELTA                 # (NY,)  layer coordinate
_XX = _x[np.newaxis, :]                      # (1, NX)
_YY = _y[:, np.newaxis]                      # (NY, 1)


def sech2(s):
    return 1.0 / np.cosh(s) ** 2


def base_omega():
    """omega_bar(y) broadcast to (NY, NX). Net-vorticity return shear recorded."""
    ob = -(U0 / DELTA) * sech2(_S)           # (NY,)
    return np.broadcast_to(ob[:, np.newaxis], (NY, NX)).copy()


def seed_perturbation():
    """w' = A cos(k x) sech^2(s), A = 1e-4 * peak base vorticity (U0/delta)."""
    A = 1e-4 * (U0 / DELTA)
    return A * np.cos(K_M * _XX) * sech2(_S)[:, np.newaxis]


def mode9_amplitude(w):
    """L2 amplitude of the m=9 x-Fourier component of omega, summed over y.
    rfft along x; column index 9 is the m=9 harmonic."""
    wh = np.fft.rfft(w, axis=1)              # (NY, NX//2+1)
    col = wh[:, 9]
    return np.sqrt(np.sum(np.abs(col) ** 2)) / NX


def dominant_x_mode(w, band_rows):
    """Dominant nonzero x-wavenumber index in a y-band around the layer."""
    band = w[band_rows, :]
    wh = np.fft.rfft(band, axis=1)
    power = np.sum(np.abs(wh) ** 2, axis=0)  # (NX//2+1,)
    power[0] = 0.0                           # drop the x-mean
    m = int(np.argmax(power))
    return m, power


# ================================================================ STEP 1
def step1_kh(n_steps=1300, sample_every=3):
    print("=" * 70)
    print("STEP 1: tanh-layer Kelvin-Helmholtz")
    print("=" * 70)
    box = Box(beta=0.0, screened=False)

    # --- return-shear diagnostic (from base state alone, before stepping) ---
    ob = base_omega()
    mean_vort = ob.mean()
    box.w = ob.copy()
    u_base, _ = box.invert()
    u_zm = u_base.mean(axis=1)                # zonal-mean u(y)
    # far-field (>= 6 delta from layer) induced slope vs ideal-tanh (flat there)
    far = np.abs(_S) > 6.0
    slope_far = np.polyfit(_y[far], u_zm[far], 1)[0]
    peak_shear = U0 / DELTA
    ret_shear_meas = abs(slope_far) / peak_shear
    ret_shear_netvort = abs(mean_vort) / peak_shear
    print(f"  base peak |omega| = {np.abs(ob).max():.4f}  (U0/delta = {peak_shear:.4f})")
    print(f"  net mean vorticity = {mean_vort:.4e}  -> |<w>|/(U0/delta) = "
          f"{ret_shear_netvort*100:.2f}%")
    print(f"  measured far-field return shear |du/dy|/(U0/delta) = "
          f"{ret_shear_meas*100:.2f}%")

    # --- initialise: base + seeded perturbation ---
    box.w = ob + seed_perturbation()

    t0 = time.time()
    ts, amps = [], []
    best_amp, best_omega, best_step = -1.0, None, -1
    for n in range(n_steps + 1):
        if n % sample_every == 0:
            a = mode9_amplitude(box.w)
            ts.append(n * DT)
            amps.append(a)
            if a > best_amp:
                best_amp, best_omega, best_step = a, box.w.copy(), n
        if n < n_steps:
            box.step_spectral()
    wall = time.time() - t0
    ts = np.array(ts)
    amps = np.array(amps)

    # --- growth-rate fit: window amp in [1e-3, 1e-1]*saturation, rising phase ---
    sat = amps.max()
    t_sat = ts[np.argmax(amps)]
    if np.argmax(amps) >= len(amps) - 3:
        print("  WARNING: m=9 amplitude still rising at end of run -- "
              "saturation NOT captured; extend n_steps.")
    rising = ts <= t_sat
    win = rising & (amps >= 1e-3 * sat) & (amps <= 1e-1 * sat)
    n_win = int(win.sum())
    coef = np.polyfit(ts[win], np.log(amps[win]), 1)
    sigma_meas = coef[0]
    sig_err = abs(sigma_meas - SIGMA_THEORY) / SIGMA_THEORY

    # --- rollup spacing at first saturation ---
    band_rows = np.where(np.abs(_S) <= 2.0)[0]
    m_star, _power = dominant_x_mode(best_omega, band_rows)
    spacing = LX / m_star if m_star > 0 else np.inf
    spacing_delta = spacing / DELTA
    spacing_err = abs(spacing - LAMBDA_M) / LAMBDA_M

    print(f"  run: {n_steps} steps, {len(ts)} samples, substeps={box.substep_count}, "
          f"cfl(final)={box.cfl():.3f}, wall={wall:.1f}s")
    print(f"  m=9 saturation amp = {sat:.4e} at t={t_sat:.4f} (step {best_step})")
    print(f"  fit window: {n_win} points, amp in "
          f"[{1e-3*sat:.2e}, {1e-1*sat:.2e}]")
    print("-" * 70)
    print(f"  sigma_measured = {sigma_meas:.4f} / time-unit")
    print(f"  sigma_theory   = {SIGMA_THEORY:.4f} / time-unit")
    print(f"  rel error      = {sig_err*100:.2f}%   (BAR: <20%)  -> "
          f"{'PASS' if sig_err < 0.20 else 'FAIL'}")
    print("-" * 70)
    print(f"  dominant x-mode at saturation = m={m_star}")
    print(f"  spacing        = {spacing:.5f} rad = {spacing_delta:.2f}*delta")
    print(f"  lambda_theory  = {LAMBDA_M:.5f} rad = {LAMBDA_M/DELTA:.2f}*delta")
    print(f"  rel error      = {spacing_err*100:.2f}%   (BAR: <30%) -> "
          f"{'PASS' if spacing_err < 0.30 else 'FAIL'}")

    return dict(ts=ts, amps=amps, sat=sat, t_sat=t_sat, sigma_meas=sigma_meas,
                sig_err=sig_err, sig_pass=sig_err < 0.20, best_omega=best_omega,
                m_star=m_star, spacing_delta=spacing_delta, spacing_err=spacing_err,
                spacing_pass=spacing_err < 0.30, win=win, best_step=best_step,
                ret_shear_meas=ret_shear_meas, ret_shear_netvort=ret_shear_netvort,
                substeps=box.substep_count, wall=wall)


# ================================================================ STEP 2
def step2_screening():
    print("=" * 70)
    print("STEP 2: screened-Poisson (K1) inversion check")
    print("=" * 70)
    box = Box(beta=0.0, screened=True)
    sigma_blob = 3.0 * DX
    xc, yc = LX / 2.0, LY / 2.0
    r2 = (_XX - xc) ** 2 + (_YY - yc) ** 2
    box.w = np.exp(-r2 / (2.0 * sigma_blob ** 2))   # unit-amplitude Gaussian blob

    u, v = box.invert()
    rr = np.sqrt(r2)
    theta = np.arctan2(_YY - yc, _XX - xc)
    # tangential (counter-clockwise) component: -u sin + v cos
    u_theta = -u * np.sin(theta) + v * np.cos(theta)

    # azimuthal average of |u_theta| into radial bins
    rmin, rmax = 0.05, 0.20
    nbins = 24
    edges = np.linspace(rmin, rmax, nbins + 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    prof = np.empty(nbins)
    for i in range(nbins):
        sel = (rr.ravel() >= edges[i]) & (rr.ravel() < edges[i + 1])
        prof[i] = np.abs(u_theta).ravel()[sel].mean()

    model_shape = bessel_k1(rc / config.LD)
    C = np.sum(prof * model_shape) / np.sum(model_shape ** 2)  # least-squares amp
    model = C * model_shape
    rel = (prof - model) / model
    rms_rel = np.sqrt(np.mean(rel ** 2))
    rms_norm = np.sqrt(np.mean((prof - model) ** 2)) / np.sqrt(np.mean(prof ** 2))

    print(f"  L_d = {config.LD}, blob sigma = {sigma_blob:.4e} ({sigma_blob/DX:.1f} DX)")
    print(f"  fit range r in [{rmin}, {rmax}] rad ({nbins} bins), C = {C:.4e}")
    print("-" * 70)
    print(f"  RMS relative deviation = {rms_rel*100:.3f}%   (BAR: <10%) -> "
          f"{'PASS' if rms_rel < 0.10 else 'FAIL'}")
    print(f"  (normalized-RMS cross-check = {rms_norm*100:.3f}%)")

    return dict(rc=rc, prof=prof, model=model, C=C, rms_rel=rms_rel,
                rms_pass=rms_rel < 0.10, u_theta=u_theta, rr=rr)


# ================================================================ FIGURE
def make_figure(s1, s2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # panel 1: omega at first saturation
    om = s1["best_omega"]
    vmax = np.abs(om).max()
    ax[0].imshow(om, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 extent=[0, LX, 0, LY], aspect="auto")
    ax[0].set_title(f"omega at first saturation (step {s1['best_step']}, "
                    f"m*={s1['m_star']} billows)")
    ax[0].set_xlabel("x (rad)")
    ax[0].set_ylabel("y (rad)")

    # panel 2: growth curve vs theory slope
    ts, amps, win = s1["ts"], s1["amps"], s1["win"]
    ax[1].semilogy(ts, amps, "k.", ms=3, label="m=9 amplitude")
    ax[1].semilogy(ts[win], amps[win], "ro", ms=5, label="fit window")
    tw = ts[win]
    a0 = amps[win][0]
    ax[1].semilogy(tw, a0 * np.exp(SIGMA_THEORY * (tw - tw[0])), "b--",
                   label=f"theory slope {SIGMA_THEORY:.2f}")
    ax[1].semilogy(tw, a0 * np.exp(s1["sigma_meas"] * (tw - tw[0])), "g-",
                   label=f"fit slope {s1['sigma_meas']:.2f}")
    ax[1].set_title(f"m=9 growth  (sigma {s1['sigma_meas']:.2f} vs "
                    f"{SIGMA_THEORY:.2f}, err {s1['sig_err']*100:.1f}%)")
    ax[1].set_xlabel("time (production units)")
    ax[1].set_ylabel("m=9 amplitude")
    ax[1].legend(fontsize=8)

    # panel 3: u_theta(r) vs C*K1
    ax[2].plot(s2["rc"], s2["prof"], "ko", label="azimuthal |u_theta|")
    ax[2].plot(s2["rc"], s2["model"], "r-", label=f"C*K1(r/L_d), C={s2['C']:.3e}")
    ax[2].set_title(f"screened inversion  (RMS {s2['rms_rel']*100:.2f}%)")
    ax[2].set_xlabel("r (rad)")
    ax[2].set_ylabel("|u_theta|")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = SPIKE / "gate0a_panel.png"
    fig.savefig(out, dpi=110)
    print(f"\nfigure -> {out}")


DATA_NPZ = SPIKE / "gate0a_data.npz"


def _save_data(s1, s2):
    np.savez_compressed(
        DATA_NPZ,
        ts=s1["ts"], amps=s1["amps"], win=s1["win"], best_omega=s1["best_omega"],
        sigma_meas=s1["sigma_meas"], sig_err=s1["sig_err"], best_step=s1["best_step"],
        m_star=s1["m_star"], spacing_delta=s1["spacing_delta"],
        spacing_err=s1["spacing_err"],
        rc=s2["rc"], prof=s2["prof"], model=s2["model"], C=s2["C"],
        rms_rel=s2["rms_rel"])


def _load_data():
    d = np.load(DATA_NPZ)
    s1 = dict(ts=d["ts"], amps=d["amps"], win=d["win"], best_omega=d["best_omega"],
              sigma_meas=float(d["sigma_meas"]), sig_err=float(d["sig_err"]),
              best_step=int(d["best_step"]), m_star=int(d["m_star"]),
              spacing_delta=float(d["spacing_delta"]),
              spacing_err=float(d["spacing_err"]))
    s2 = dict(rc=d["rc"], prof=d["prof"], model=d["model"], C=float(d["C"]),
              rms_rel=float(d["rms_rel"]))
    return s1, s2


if __name__ == "__main__":
    if "--fig-only" in sys.argv:
        s1, s2 = _load_data()
        make_figure(s1, s2)
        sys.exit(0)

    t_start = time.time()
    s1 = step1_kh()
    print()
    s2 = step2_screening()
    _save_data(s1, s2)
    print(f"data cached -> {DATA_NPZ}")
    try:
        make_figure(s1, s2)
    except Exception as exc:  # figure is diagnostic; physics results already cached
        print(f"FIGURE FAILED ({exc!r}) -- rerun with --fig-only after fixing")

    print("\n" + "=" * 70)
    print("GATE 0a SUMMARY")
    print("=" * 70)
    print(f"  Step1 growth-rate  : sigma={s1['sigma_meas']:.3f} vs "
          f"{SIGMA_THEORY:.3f}  err={s1['sig_err']*100:.1f}%  "
          f"-> {'PASS' if s1['sig_pass'] else 'FAIL'}")
    print(f"  Step1 rollup spacing: {s1['spacing_delta']:.2f}*delta vs "
          f"{LAMBDA_M/DELTA:.2f}*delta  err={s1['spacing_err']*100:.1f}%  "
          f"-> {'PASS' if s1['spacing_pass'] else 'FAIL'}")
    print(f"  Step2 K1 inversion : RMS={s2['rms_rel']*100:.2f}%  "
          f"-> {'PASS' if s2['rms_pass'] else 'FAIL'}")
    print(f"  return shear (net-vort) = {s1['ret_shear_netvort']*100:.2f}%, "
          f"(far-field) = {s1['ret_shear_meas']*100:.2f}%")
    allpass = s1["sig_pass"] and s1["spacing_pass"] and s2["rms_pass"]
    print(f"  TOTAL wall = {time.time()-t_start:.1f}s")
    print("  GATE 0a:", "PASS" if allpass else "FAIL")

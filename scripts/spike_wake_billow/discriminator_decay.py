"""DISCRIMINATOR 1 -- unforced screened sheet-decay (cheap-first, user-approved).

Removes the Dirichlet enstrophy PUMP that caused every forced Gate-1 cell to
CFL-blow-up before reaching a steady state. A full-width zonally-uniform shear
sheet (the E2 deficit jet) + a y-confined broadband seed is set as the INITIAL
condition on top of the warm bracket ambient (screening L_d 0.18 + beta), then
evolved with NO forcing (forcing_fn=None) -- pure J(psi,w) advection + the
Nyquist-only nu8 filter. Doubly-periodic in x, so the sheet fills the domain
(a periodic shear layer = the cleanest KH-rollup setup, cf. Gate 0a).

Two questions this answers that the forced scan could not:
  (1) STABILITY: does the box stay numerically valid (no CFL blowup) once the
      enstrophy pump is gone? If YES -> the forced blowup was the harness pump,
      and we can finally read a terminal state. If NO -> the blowup is intrinsic
      to rollup at production cell size -> escalate to 2x-res+half-DT.
  (2) TERMINAL STRUCTURE: when it survives, does the sheet roll into a DENSE
      small-billow chain (reference: coh 0.35-0.55, band_rms>=1.4, count>=N*) or
      settle at SPARSE SMOOTH rolls (coh>>0.55, weak)? Dense -> dissipation
      hypothesis LIVE. Sparse-smooth-and-STABLE -> physics wall confirmed.

Variants per config: mode='free' (truly unforced) and mode='meanhold' (hold the
ZONAL MEAN profile only -- no eddy injection, just prevents the ambient jet from
decaying/drifting so the sheet eddies evolve against a steady background; NOT an
enstrophy pump). Both reported; 'free' is primary.

Reuses solver.Box, environment.{W_AMB_2D,TR_AMB_2D,TR_AMB_COL,w_sheet,EnvConfig},
metrics.score_frame. Nothing here touches src/gasgiant/** or git.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import numpy as np

SPIKE = pathlib.Path(__file__).parent
sys.path.insert(0, str(SPIKE))
import config as C          # noqa: E402
import environment as E     # noqa: E402
import metrics as M         # noqa: E402
from solver import Box      # noqa: E402

OUT = SPIKE / "discriminator_runs"
OUT.mkdir(exist_ok=True)

# stability ceiling: max_w above this * pre-rollup plateau => flag blowup onset.
MW_CEIL_FACTOR = 3.0


def _seed_field(A, delta, seed_id):
    """Band-limited noise, y-confined to the sheet vicinity (same construction and
    y-envelope as environment.make_forcing's strip noise, but as a ONE-TIME initial
    kick). Amplitude SEED_AMP_FRAC*A. Broadband so the flow SELECTS the wavelength."""
    rng = np.random.default_rng(np.random.SeedSequence([777, int(seed_id)]))
    w_hat = np.zeros(E.NOISE_MASK.shape, dtype=np.complex128)
    draw = rng.standard_normal((2, E._N_NOISE_MODES))
    w_hat[E.NOISE_MASK] = draw[0] + 1j * draw[1]
    noise = np.fft.irfft2(w_hat, s=(E.NY, E.NX))
    noise *= (C.SEED_AMP_FRAC * A) / (noise.std() + 1e-30)
    dy = E._Y - C.Y_SHEET
    env = np.exp(-0.5 * (dy / (2.0 * delta)) ** 2)
    env[np.abs(dy) > 3.0 * (2.0 * delta)] = 0.0
    return noise * env[:, None]


def run_one(A, delta_frac, form, seed_name, mode, n_steps=6000, sample_every=25):
    delta = delta_frac * C.RC
    seed_id = C.SEEDS[seed_name]
    cfg = E.EnvConfig(A=A, delta=delta, sheet_form=form, seed_id=seed_id)

    box = Box()                                   # screened, beta=config.BETA
    # INITIAL condition: ambient + FULL-WIDTH sheet + one-time seed
    w_sheet_col = E.w_sheet(cfg)                  # (NY,)
    box.w = (E.W_AMB_2D + w_sheet_col[:, None]) + _seed_field(A, delta, seed_id)
    box.tr = E.TR_AMB_2D.copy()

    meanhold = (mode == "meanhold")
    rms0 = None
    ts = dict(step=[], nb=[], lam=[], brk=[], coh=[], sk=[], rms=[], subs=[], mw=[])
    snaps = []
    snap_at = {0, n_steps // 3, 2 * n_steps // 3, n_steps}
    t0 = time.time()
    blown = -1
    for n in range(n_steps + 1):
        if n % sample_every == 0:
            fr = M.score_frame(box.tr, C, rms0=rms0, w_full=box.w, step=n)
            if rms0 is None:
                rms0 = float(np.sqrt(np.mean(M.hp(M.extract_band(box.tr, C), C) ** 2)))
                fr = M.score_frame(box.tr, C, rms0=rms0, w_full=box.w, step=n)
            mw = float(np.abs(box.w).max())
            ts["step"].append(n); ts["nb"].append(fr.n_billows)
            ts["lam"].append(fr.wave.lam_rc); ts["brk"].append(fr.wave.hypothesis or "")
            ts["coh"].append(fr.coh); ts["sk"].append(fr.skew)
            ts["rms"].append(fr.rms_ratio); ts["subs"].append(box.substep_count)
            ts["mw"].append(mw)
        if n in snap_at:
            snaps.append((n, box.tr.copy(), box.w.copy()))
        if n < n_steps:
            if meanhold:
                def _mh(b):
                    corr = C.MEANHOLD_RATE * (b.w.mean(axis=1) - E.W_AMB_COL)
                    b.w -= corr[:, None]
                box.step_spectral(forcing_fn=_mh)
            else:
                box.step_spectral(forcing_fn=None)
        # early stop if catastrophically blown (save compute)
        if box.substep_count > 0 and blown < 0:
            blown = n
        if box.substep_count > 50:
            break
    wall = time.time() - t0

    name = f"{form}_A{A}_d{delta_frac:.2f}_{seed_name}_{mode}"
    np.savez_compressed(
        OUT / f"{name}.npz",
        A=A, delta_frac=delta_frac, form=form, seed_name=seed_name, mode=mode,
        step=np.array(ts["step"]), nb=np.array(ts["nb"]), lam=np.array(ts["lam"]),
        brk=np.array(ts["brk"]), coh=np.array(ts["coh"]), sk=np.array(ts["sk"]),
        rms=np.array(ts["rms"]), subs=np.array(ts["subs"]), mw=np.array(ts["mw"]),
        snap_steps=np.array([s for s, _, _ in snaps]),
        snap_tr=np.array([t for _, t, _ in snaps]),
        snap_w=np.array([w for _, _, w in snaps]),
        blown_step=blown, wall=wall, n_steps_ran=ts["step"][-1] if ts["step"] else 0,
    )
    last = ts["step"][-1]
    print(f"[{name}] ran {last} steps, blown@{blown}, subs={ts['subs'][-1]}, "
          f"wall={wall:.0f}s -> {name}.npz")
    return name


CONFIGS = [
    # (A, delta_frac, form, seed_name)  -- the 2 longest-surviving forced cells
    (10, 0.30, "deficit", "gate1"),
    (10, 0.15, "deficit", "gate1"),
    (45, 0.15, "deficit", "gate1"),   # a higher-A cell: does it blow up unforced?
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=int, default=None, help="run only CONFIGS[i]")
    ap.add_argument("--mode", default="free", choices=["free", "meanhold", "both"])
    ap.add_argument("--nsteps", type=int, default=6000)
    args = ap.parse_args()
    cfgs = [CONFIGS[args.one]] if args.one is not None else CONFIGS
    modes = ["free", "meanhold"] if args.mode == "both" else [args.mode]
    for (A, df, form, seed) in cfgs:
        for mode in modes:
            run_one(A, df, form, seed, mode, n_steps=args.nsteps)


if __name__ == "__main__":
    main()

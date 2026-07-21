"""Task 6 -- the GATE-1 PHYSICS SCAN runner for the billow-rollup spike.

Drives the frozen environment (environment.py) + solver (solver.py) through the
frozen Gate-1 verdict (metrics.evaluate_run) over the frozen cell list, writing
one .npz + one progress.log line per run. NOTHING is committed to git; no file
under src/gasgiant/** is touched; every constant/criterion is IMPORTED from
config.py / metrics.py -- nothing is redefined here.

Cell list (FROZEN, from task-6-brief.md + the run parameters):
  main scan = A in {10,20,45,90} x delta in {0.15,0.30,0.50}*RC
              x sheet_form in {deficit,tanh}                     = 24 cells
            + shear-off CONTROL at (A=45, delta=0.30*RC, deficit) =  1 cell
            ------------------------------------------------------------------
            = 25 configs x 2 seeds (gate1, gate1_replicate)      = 50 runs.

Per cell:
  * sigma_expected = 0.095*A  -- KH growth clock for the formation bar.
      derivation: KH sigma ~ 0.19*U0/delta with U0 ~ A*delta/2 the sheet's
      velocity scale (deficit u' = +A*delta/2, tanh comparable at this
      precision) => sigma ~ 0.19*(A*delta/2)/delta = 0.095*A. Same for both
      forms; supplied to metrics.evaluate_run(sigma_expected=...).
  * CEILING flag: max|w_amb(y)+w_sheet(y)| > config.OMEGA_CEILING (60) marks the
      cell CONDITIONAL (computed analytically from the frozen profiles).
  * VOID (environment broke, NOT a physics FAIL): if environment.flank_guard >
      config.FLANK_MAX OR any CFL substep occurs DURING the scoring window
      (steps >= strip_to_band). The .npz keeps every field for offline
      re-adjudication; the break step + kind are recorded.

Shear-off CONTROL definition (documented deviation -- the frozen pseudo-spectral
solver has NO mean-flow term, so a literal uniform u=u(Y_SHEET) cannot be
injected): the ambient VORTICITY profile is replaced by ZERO (W_AMB_COL/2D := 0).
This removes the ambient STRAIN (du/dy) exactly, which is the control's purpose
(does the sheet roll up WITHOUT the ambient shear?). The sheet still transits
downstream under its OWN self-induced jet (deficit u'=+A*delta/2); the uniform
u=u(Y_SHEET) piece is a Galilean frame the mean-flow-free solver already sits in.
beta is retained (planetary vorticity gradient is not "ambient shear"). The
standard config.transit_report() advective clocks are used for every cell
(frozen metrics convention); the control transits slower, noted in the report.

Seeds: SEEDS["gate1"]=11 / ["gate1_replicate"]=12 feed the STRIP-NOISE stream id,
threaded through EnvConfig.seed_id (rng = SeedSequence([777, seed_id]); the same
stream construction environment.py already uses). NOTE the Task-4 smoke used
SEEDS["seed_noise"]=13, so a gate run's noise stream differs from the smoke's --
expected per the run parameters.

Usage (from repo root C:\\...\\gas-giant):
  uv run python "<SPIKE>\\run_gate1.py" --smoke 100   # fast plumbing check (~1 min)
  uv run python "<SPIKE>\\run_gate1.py" --pilot        # single cell, full 5500 (~50 min)
  uv run python "<SPIKE>\\run_gate1.py" --scan         # remaining runs, 8 workers, resumable
  uv run python "<SPIKE>\\run_gate1.py" --report       # summarize gate1_runs/*.npz
"""
# ---- single-thread the workers BEFORE numpy is imported (numpy FFT is
#      single-threaded; this stops 8 workers x BLAS from oversubscribing 16 cores).
import os  # noqa: E402
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

SPIKE = pathlib.Path(__file__).parent
sys.path.insert(0, str(SPIKE))
import config  # noqa: E402
import environment  # noqa: E402
import metrics  # noqa: E402
from solver import Box  # noqa: E402

# Captured ONCE at import (before any shear-off monkeypatch) so every task can
# reset the ambient to a known state regardless of worker reuse.
ORIG_W_AMB_COL = np.array(environment.W_AMB_COL, dtype=float)
ORIG_W_AMB_2D = np.array(environment.W_AMB_2D, dtype=float)

RUN_DIR = SPIKE / "gate1_runs"
LOG_DIR = RUN_DIR / "logs"
PROGRESS = RUN_DIR / "progress.log"

RUN_STEPS = 5500          # frozen run length
SAMPLE_EVERY = 50         # metrics sampled every 50 steps from step 0

A_VALS = [10, 20, 45, 90]
DELTA_FRACS = [0.15, 0.30, 0.50]
FORMS = ["deficit", "tanh"]
SEED_NAMES = ["gate1", "gate1_replicate"]

PILOT = dict(cell="A45_d0.30_deficit", A=45, frac=0.30, form="deficit",
             shear=False, seed="gate1")


# ============================================================================
# Cell list
# ============================================================================
def build_runs():
    """The frozen 50-run list (25 configs x 2 seeds)."""
    runs = []
    for A in A_VALS:
        for frac in DELTA_FRACS:
            for form in FORMS:
                cell = f"A{A}_d{frac:.2f}_{form}"
                for sname in SEED_NAMES:
                    runs.append(dict(cell=cell, A=A, frac=frac, form=form,
                                     shear=False, seed=sname))
    for sname in SEED_NAMES:      # shear-off control at (A=45, 0.30*RC, deficit)
        runs.append(dict(cell="A45_d0.30_deficit_shearoff", A=45, frac=0.30,
                         form="deficit", shear=True, seed=sname))
    return runs


def npz_path(spec):
    sid = config.SEEDS[spec["seed"]]
    return RUN_DIR / f"{spec['cell']}_s{sid}.npz"


# ============================================================================
# One run
# ============================================================================
def run_one(spec, run_steps=RUN_STEPS, log_stdout=True):
    """Execute one Gate-1 cell to completion; write the .npz; return
    (progress_line, summary_dict). Deterministic in `spec` alone (seed +
    ambient state are fully determined here, independent of scheduling)."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sid = config.SEEDS[spec["seed"]]
    cell, A, frac, form, shear = (spec["cell"], spec["A"], spec["frac"],
                                  spec["form"], spec["shear"])
    delta = frac * config.RC
    sigma_exp = 0.095 * A                                   # KH clock (see module docstring)
    out = RUN_DIR / f"{cell}_s{sid}.npz"

    logf = open(LOG_DIR / f"{cell}_s{sid}.log", "w", buffering=1)
    _old_stdout = sys.stdout
    if log_stdout:
        sys.stdout = logf
    try:
        print(f"[START] {cell} s{sid} ({spec['seed']}) A={A} delta={delta:.5f} "
              f"({frac}RC) form={form} shear_off={shear} pid={os.getpid()} "
              f"t={time.strftime('%H:%M:%S')}")

        # ---- reset ambient every task (robust to worker reuse) -------------
        if shear:
            environment.W_AMB_COL = np.zeros(config.NY)
            environment.W_AMB_2D = np.zeros((config.NY, config.NX))
        else:
            environment.W_AMB_COL = ORIG_W_AMB_COL.copy()
            environment.W_AMB_2D = ORIG_W_AMB_2D

        cfg = environment.EnvConfig(A=A, delta=delta, sheet_form=form, seed_id=sid)

        # ---- static CONDITIONAL ceiling (analytic) -------------------------
        w_amb_col = environment.W_AMB_COL
        w_sheet_col = environment.w_sheet(cfg)
        ceiling_val = float(np.abs(w_amb_col + w_sheet_col).max())
        conditional = ceiling_val > config.OMEGA_CEILING

        # ---- advective clocks (frozen metrics convention) ------------------
        u_sheet, s2b, bf = config.transit_report()
        strip_to_band = int(round(s2b))
        band_fill = int(round(bf))

        box = Box()                       # screened, beta=config.BETA
        environment.init_state(box, cfg)
        forcing = environment.make_forcing(cfg)

        # ---- snapshot targets: formation-bar, mid-run, end -----------------
        form_bar = strip_to_band + max(config.DEV_STEPS,
                                       (2.0 / max(sigma_exp, 1e-12)) / config.DT)
        snap_form = int(min(round(form_bar / SAMPLE_EVERY) * SAMPLE_EVERY, run_steps))
        snap_mid = int((run_steps // 2 // SAMPLE_EVERY) * SAMPLE_EVERY)
        snap_end = int(run_steps)
        snap_targets = sorted({snap_form, snap_mid, snap_end})

        samples = []                      # dicts (step, tr, w) float32 for evaluate_run
        steps_ts, flankg_ts, subs_ts, maxw_ts = [], [], [], []
        snaps = {}

        t0 = time.time()
        for n in range(run_steps + 1):
            if n % SAMPLE_EVERY == 0:
                tr32 = box.tr.astype(np.float32)
                w32 = box.w.astype(np.float32)
                samples.append(dict(step=n, tr=tr32, w=w32))
                steps_ts.append(n)
                flankg_ts.append(float(environment.flank_guard(box)))
                subs_ts.append(int(box.substep_count))
                maxw_ts.append(float(np.abs(box.w).max()))
                if n in snap_targets:
                    snaps[n] = (w32, tr32)
                if n % 500 == 0:
                    print(f"  step {n:5d}/{run_steps} flank={flankg_ts[-1]:.3f} "
                          f"subs={subs_ts[-1]} max|w|={maxw_ts[-1]:.1f} "
                          f"elapsed={time.time()-t0:.0f}s")
            if n < run_steps:
                box.step_spectral(forcing_fn=lambda b, k=n: forcing(b, k))
        wall = time.time() - t0

        # ---- FROZEN Gate-1 verdict (single scoring pass) -------------------
        res = metrics.evaluate_run(samples, sigma_expected=sigma_exp,
                                   substep_count=box.substep_count,
                                   strip_to_band=strip_to_band, band_fill=band_fill)

        # per-sample series pulled from the frozen scorer's frames
        n_bill_ts = [int(f.n_billows) for f in res.frames]
        lam_ts = [float(f.wave.lam_rc) for f in res.frames]
        bracket_ts = [(f.wave.hypothesis or "") for f in res.frames]
        coh_ts = [float(f.coh) for f in res.frames]
        skew_ts = [float(f.skew) for f in res.frames]
        rmsr_ts = [float(f.rms_ratio) for f in res.frames]

        steps_arr = np.array(steps_ts)
        flankg_arr = np.array(flankg_ts)
        subs_arr = np.array(subs_ts)

        # ---- VOID adjudication (environment broke in the scoring window) ---
        scoring = steps_arr >= strip_to_band
        void_flank = bool((flankg_arr[scoring] > config.FLANK_MAX).any())
        subs_at_start = int(subs_arr[scoring][0]) if scoring.any() else 0
        void_cfl = bool(int(subs_arr[-1]) > subs_at_start)
        break_steps = []
        if void_flank:
            fb = steps_arr[scoring][flankg_arr[scoring] > config.FLANK_MAX]
            break_steps.append(("flank", int(fb[0])))
        if void_cfl:
            inc = scoring & (subs_arr > subs_at_start)
            break_steps.append(("cfl", int(steps_arr[inc][0])))
        VOID = void_flank or void_cfl
        if break_steps:
            break_kind = "+".join(k for k, _ in sorted(break_steps, key=lambda z: z[1]))
            break_step = min(s for _, s in break_steps)
        else:
            break_kind, break_step = "", -1

        if VOID:
            verdict = "VOID"
        elif res.gate1_pass:
            verdict = "PASS"
        else:
            verdict = "FAIL"

        failing = [k for k, ok in res.reasons.items() if not ok]
        if verdict == "VOID":
            kill = f"{break_kind}@{break_step}"
        elif verdict == "FAIL":
            kill = ",".join(failing) if failing else "?"
        else:
            kill = "-"
        cond_tag = " CONDITIONAL" if conditional else ""

        formation_step = res.formation_step if res.formation_step is not None else -1
        n_star = res.n_star if res.n_star is not None else -1
        hypothesis = res.hypothesis or ""
        lam_rc = res.lam_rc if np.isfinite(res.lam_rc) else float("nan")

        # ---- write the .npz (everything, for offline re-adjudication) ------
        np.savez_compressed(
            out,
            # per-sample series
            sample_steps=steps_arr,
            n_billows=np.array(n_bill_ts),
            lam_star=np.array(lam_ts),
            bracket=np.array(bracket_ts),
            coherence=np.array(coh_ts),
            skew=np.array(skew_ts),
            band_rms_ratio=np.array(rmsr_ts),
            flank_guard=flankg_arr,
            substep_count=subs_arr,
            max_w=np.array(maxw_ts),
            # frozen verdict + adjudication fields
            verdict=np.array(verdict),
            gate1_pass=np.array(bool(res.gate1_pass)),
            reasons_json=np.array(json.dumps(res.reasons)),
            coherence_only_fail=np.array(bool(res.coherence_only_fail)),
            void=np.array(bool(VOID)),
            void_flank=np.array(bool(void_flank)),
            void_cfl=np.array(bool(void_cfl)),
            break_kind=np.array(break_kind),
            break_step=np.array(break_step),
            conditional=np.array(bool(conditional)),
            ceiling_val=np.array(ceiling_val),
            formation_step=np.array(formation_step),
            formation_bar=np.array(float(res.formation_bar)),
            stationarity_json=np.array(json.dumps(res.stationarity)),
            window_start=np.array(float(res.window_start)),
            window_n=np.array(int(res.window_n)),
            n_star=np.array(n_star),
            hypothesis=np.array(hypothesis),
            lam_rc=np.array(lam_rc),
            count_frac=np.array(float(res.count_frac)),
            flank_leak=np.array(float(res.flank)),
            substep_total=np.array(int(res.substep_count)),
            # clocks / cell metadata
            sigma_expected=np.array(sigma_exp),
            strip_to_band=np.array(strip_to_band),
            band_fill=np.array(band_fill),
            u_sheet=np.array(float(u_sheet)),
            cellname=np.array(cell),
            seed_name=np.array(spec["seed"]),
            seed_id=np.array(sid),
            A=np.array(A), delta=np.array(delta), delta_frac=np.array(frac),
            sheet_form=np.array(form), shear_off=np.array(bool(shear)),
            # field snapshots (float32)
            snap_steps=np.array(snap_targets),
            snap_w=np.array([snaps[s][0] for s in snap_targets]),
            snap_tr=np.array([snaps[s][1] for s in snap_targets]),
            # runtime metadata
            wall_sec=np.array(wall),
            steps_per_sec=np.array(run_steps / max(wall, 1e-9)),
        )

        line = (f"{cell:28s} s{sid} {verdict:5s}{cond_tag:12s} "
                f"kill={kill:22s} lam*={lam_rc:6.3f} count={res.count_frac:4.2f} "
                f"wall={wall/60:5.1f}min")
        print(f"[DONE] {line}  -> {out.name}")
        summary = dict(cell=cell, seed=sid, verdict=verdict, kill=kill,
                       conditional=conditional, lam_rc=lam_rc,
                       count_frac=float(res.count_frac), wall_min=wall / 60.0,
                       npz=out.name)
        return line, summary
    finally:
        sys.stdout = _old_stdout
        logf.close()


# ============================================================================
# Drivers
# ============================================================================
def _append_progress(line):
    with open(PROGRESS, "a", buffering=1) as f:
        f.write(line + "\n")


def _run_pool(args):
    """Pool entry: (spec,) -> (line, summary). Kept top-level for spawn-pickle."""
    return run_one(args)


def drive_pilot():
    print(f"[pilot] running {PILOT['cell']} s{config.SEEDS[PILOT['seed']]} "
          f"({PILOT['seed']}), full {RUN_STEPS} steps -- foreground of this "
          f"(detached) process. Expect ~50 min.", flush=True)
    line, summary = run_one(PILOT)
    _append_progress(line)
    print("[pilot] " + line, flush=True)
    print(f"[pilot] npz -> {npz_path(PILOT)}", flush=True)
    return summary


def drive_scan(workers):
    runs = build_runs()
    todo = [r for r in runs if not npz_path(r).exists()]
    done = len(runs) - len(todo)
    print(f"[scan] {len(runs)} total runs; {done} already have .npz (skipped, "
          f"resumable); {len(todo)} to run on {workers} workers.", flush=True)
    for r in todo:
        print(f"       queued: {r['cell']} s{config.SEEDS[r['seed']]}", flush=True)
    if not todo:
        print("[scan] nothing to do -- all .npz present.", flush=True)
        return
    t0 = time.time()
    n_ok = 0
    # maxtasksperchild=1: a fresh interpreter per run (clean ambient / no leak).
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for line, summary in pool.imap_unordered(_run_pool, todo):
            n_ok += 1
            _append_progress(line)
            print(f"[scan {n_ok}/{len(todo)}  +{(time.time()-t0)/60:.1f}min] "
                  + line, flush=True)
    print(f"[scan] complete: {n_ok} runs in {(time.time()-t0)/60:.1f} min.",
          flush=True)


def drive_report():
    runs = build_runs()
    print(f"{'cell':30s} {'seed':4s} {'verdict':9s} {'kill':22s} "
          f"{'lam*':>6s} {'cnt':>4s} {'wall':>6s}")
    print("-" * 92)
    seen = {}
    for r in sorted(runs, key=lambda z: (z["cell"], config.SEEDS[z["seed"]])):
        p = npz_path(r)
        if not p.exists():
            print(f"{r['cell']:30s} s{config.SEEDS[r['seed']]:<3d} (missing)")
            continue
        d = np.load(p, allow_pickle=False)
        v = str(d["verdict"])
        cond = " C" if bool(d["conditional"]) else ""
        kill = (f"{str(d['break_kind'])}@{int(d['break_step'])}" if v == "VOID"
                else ",".join(k for k, ok in json.loads(str(d["reasons_json"])).items()
                              if not ok) if v == "FAIL" else "-")
        print(f"{str(d['cellname']):30s} s{int(d['seed_id']):<3d} {v+cond:9s} "
              f"{kill:22s} {float(d['lam_rc']):6.3f} {float(d['count_frac']):4.2f} "
              f"{float(d['wall_sec'])/60:5.1f}m")
        seen.setdefault(str(d["cellname"]), []).append(v)
    print("-" * 92)
    # per-cell both-seed adjudication (pass=both / fail=both / split=MARGINAL)
    for cell, vs in seen.items():
        if len(vs) == 2:
            tag = ("PASS" if all(x == "PASS" for x in vs)
                   else "FAIL" if all(x == "FAIL" for x in vs)
                   else "VOID" if all(x == "VOID" for x in vs)
                   else "MARGINAL/SPLIT")
            print(f"  {cell:30s} seeds={vs} -> {tag}")


def main():
    ap = argparse.ArgumentParser(description="Gate-1 physics scan runner")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", type=int, metavar="N",
                   help="fast plumbing check: pilot cell for N steps")
    g.add_argument("--pilot", action="store_true", help="single pilot cell, full run")
    g.add_argument("--scan", action="store_true", help="all remaining runs (pool)")
    g.add_argument("--report", action="store_true", help="summarize gate1_runs/*.npz")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.smoke:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        spec = dict(PILOT, cell="_smoke")
        line, summary = run_one(spec, run_steps=args.smoke, log_stdout=False)
        print("[smoke] " + line)
        print("[smoke] verdict fields loaded OK; deleting scratch npz.")
        (RUN_DIR / f"_smoke_s{config.SEEDS[spec['seed']]}.npz").unlink(missing_ok=True)
    elif args.pilot:
        drive_pilot()
    elif args.scan:
        drive_scan(args.workers)
    elif args.report:
        drive_report()


if __name__ == "__main__":
    main()

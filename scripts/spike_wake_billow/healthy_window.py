"""Healthy-window re-adjudication of the Gate-1 scan (approved analysis).

The frozen gate's stationarity window (strip_to_band+band_fill .. end) includes
frames AFTER the box goes CFL-unstable (substep_count>0, max_w explodes), so the
stored count_frac/lam_rc are contaminated by numerical-blowup garbage. This
re-scores every cell on the STRICTLY-STABLE window only -- the leading run of
samples with substep_count==0 AND max_w below a sane multiple of its own plateau
-- and asks the ONE question the sink-free box can legitimately answer:

    does a sustained, discrete, reference-scale billow chain FORM while the
    solver is still numerically valid?

Reference-true per-frame criteria (metrics.py, frozen): n_billows >= n_star for
the frame's own bracket (1rc: N*>=4 dense; 3rc: N*>=2 sparse), bracket set,
|skew|<=0.2, coherence in [0.35,0.55], band_rms_ratio>=1.4.

Nothing here re-simulates or touches src/gasgiant/** or git; pure re-analysis of
the stored per-sample time series.
"""
from __future__ import annotations
import pathlib
import numpy as np

RUNS = pathlib.Path(__file__).parent / "gate1_runs"

# stable-window definition: the solver's own CFL guard (substep_count==0) is the
# principled instability boundary; add a max_w ceiling at 1.6x the pre-forcing
# plateau (samples 1..6 median, before any rollup) to catch the rising-max_w
# onset frames that precede the first substep by ~1 sample.
MW_PLATEAU_SLICE = slice(1, 7)
MW_CEIL_FACTOR = 1.6


def n_star_for_bracket(brk: str) -> int | None:
    if brk == "1rc":
        return max(4, int(np.floor(9.0 / 1.2)) - 1)  # dense transverse chain
    if brk == "3rc":
        return max(2, int(np.floor(9.0 / 3.0)) - 1)  # sparse along-wake chain
    return None


def frame_reference_true(nb, brk, coh, sk, rms) -> bool:
    ns = n_star_for_bracket(str(brk))
    if ns is None:
        return False
    return (nb >= ns and abs(sk) <= 0.2 and 0.35 <= coh <= 0.55 and rms >= 1.4)


def analyse(path: pathlib.Path) -> dict:
    d = np.load(path, allow_pickle=True)
    st = d["sample_steps"]; nb = d["n_billows"]; brk = d["bracket"]
    coh = d["coherence"]; sk = d["skew"]; rms = d["band_rms_ratio"]
    ss = d["substep_count"]; mw = d["max_w"]

    mw_plateau = float(np.median(mw[MW_PLATEAU_SLICE]))
    mw_ceil = MW_CEIL_FACTOR * mw_plateau
    # strictly-stable = leading contiguous run with substep==0 AND max_w<=ceil
    stable = np.ones(len(st), dtype=bool)
    bad = (ss > 0) | (mw > mw_ceil)
    first_bad = int(np.argmax(bad)) if bad.any() else len(st)
    stable[first_bad:] = False

    idx = np.where(stable)[0]
    last_stable_step = int(st[idx[-1]]) if idx.size else 0
    # per-frame reference-true flags within the stable window
    rt = np.array([frame_reference_true(int(nb[i]), brk[i], float(coh[i]),
                                        float(sk[i]), float(rms[i])) for i in idx])
    # longest run of consecutive reference-true stable frames (sustained rollup)
    best_run = cur = 0
    for v in rt:
        cur = cur + 1 if v else 0
        best_run = max(best_run, cur)
    # peak billow count in the stable window + its context
    if idx.size:
        j = idx[int(np.argmax(nb[idx]))]
        peak_nb = int(nb[j])
        peak_ctx = (int(st[j]), str(brk[j]), float(coh[j]), float(sk[j]),
                    float(rms[j]), int(ss[j]), float(mw[j]))
    else:
        peak_nb, peak_ctx = 0, (0, "", 0, 0, 0, 0, 0)

    # classification
    if best_run >= 3:
        cls = "FORMS"          # sustained reference-true chain while stable
    elif peak_nb <= 1:
        cls = "LAMINAR"        # never rolls up in the stable window
    else:
        cls = "MARGINAL"       # transient/onset structure, not sustained+clean

    return dict(
        name=path.stem, A=int(d["A"]), delta=float(d["delta_frac"]),
        form=str(d["sheet_form"]), seed=str(d["seed_name"]),
        shear_off=bool(d["shear_off"]),
        n_stable=int(idx.size), last_stable_step=last_stable_step,
        first_bad_step=int(st[first_bad]) if first_bad < len(st) else -1,
        mw_plateau=mw_plateau, peak_nb_stable=peak_nb, peak_ctx=peak_ctx,
        best_rt_run=int(best_run), cls=cls,
        # stored-frozen (contaminated) numbers for contrast
        frozen_count_frac=float(d["count_frac"]), frozen_lam_rc=float(d["lam_rc"]),
    )


def main():
    paths = sorted(RUNS.glob("*.npz"))
    rows = [analyse(p) for p in paths]
    rows.sort(key=lambda r: (r["shear_off"], r["A"], r["delta"], r["form"], r["seed"]))

    hdr = (f"{'cell':30s} {'cls':8s} {'nStbl':>5} {'lastStbl':>8} {'peakNb':>6} "
           f"{'rtRun':>5} {'peak@step/brk/coh/skew/rms/subs/mw':>40}  {'frozenCF':>8}")
    print(hdr); print("-" * len(hdr))
    tally = {}
    for r in rows:
        pc = r["peak_ctx"]
        peakstr = (f"{pc[0]}/{pc[1] or '--'}/{pc[2]:.2f}/{pc[3]:+.2f}/{pc[4]:.2f}"
                   f"/{pc[5]}/{pc[6]:.0f}")
        print(f"{r['name']:30s} {r['cls']:8s} {r['n_stable']:5d} "
              f"{r['last_stable_step']:8d} {r['peak_nb_stable']:6d} "
              f"{r['best_rt_run']:5d} {peakstr:>40}  {r['frozen_count_frac']:8.2f}")
        tally[r["cls"]] = tally.get(r["cls"], 0) + 1

    print("-" * len(hdr))
    print("CLASS TALLY:", tally)
    forms = [r for r in rows if r["cls"] == "FORMS"]
    print(f"\nFORMS (sustained reference-true chain in stable window): {len(forms)}")
    for r in forms:
        print(f"   {r['name']}  rt_run={r['best_rt_run']} peak_nb={r['peak_nb_stable']}")
    # laminar-max: highest peak_nb among stable windows regardless of class
    top = sorted(rows, key=lambda r: -r["peak_nb_stable"])[:8]
    print("\nHighest peak stable-window n_billows (any class):")
    for r in top:
        pc = r["peak_ctx"]
        print(f"   {r['name']:30s} peak_nb={r['peak_nb_stable']} @step{pc[0]} "
              f"brk={pc[1] or '--'} rms={pc[4]:.2f} skew={pc[3]:+.2f} "
              f"coh={pc[2]:.2f} subs={pc[5]} mw={pc[6]:.0f}  cls={r['cls']}")
    # sanity: shear-off controls must be LAMINAR
    print("\nShear-off controls:")
    for r in rows:
        if r["shear_off"]:
            print(f"   {r['name']:30s} cls={r['cls']} peak_nb={r['peak_nb_stable']} "
                  f"n_stable={r['n_stable']}")


if __name__ == "__main__":
    main()

# M3 coupling — q_target-bias & fast-nudge levers FALSIFIED (2026-06-16)

Follow-up to `2026-06-16-m3-coupling-verdict.md`. The additive coupling produced a
visual fidelity win at gain≈0.5 but failed the automated coherent-dominant gate
(`banded_coherent_fraction` ratio ~0.5 vs the >1.05 target) — design risk R1
confirmed. The named next lever was **q_target-bias injection** (fold the coherent
baroclinic source into the v1.6 nudge TARGET, before the relax, instead of
additively after). This document records its falsification **before any kernel was
written**, via three-lens adversarial review + a cheap empirical confirmation.

## Adversarial review (3 independent reviewers)

1. **Physics (the kill).** The "persistent attractor" justification is wrong. Both
   the additive path and the proposed target-bias path re-stamp the source into the
   field every step in the per-step kernel; neither has a memory/persistence term
   the other lacks. Folding `src` into `q_target` before the relax
   `q += (q_target − q)/τ` is **algebraically identical to additive injection with
   the gain rescaled by `confine/τ`** (τ = `vort_relax_tau` = 600 in
   `jupiter_vorticity`). So target-bias is the already-falsified additive injection
   at a different effective gain — and the gain axis was already swept and failed
   monotonically. The actual broadbander is the **nonlinear advective fold at the
   sheared vortex edges**, which is blind to where `src` enters the kernel.
   Confirmed corollary: the gate already runs `vort_inject=0` and `vort_drag=0`, so
   the broadband is NOT from v1.6's FBM source — it is intrinsic to the vorticity
   dynamics. (This also kills the secondary "reduce vort_inject" lever: it is
   already zero.)
2. **Implementation.** If built, needs specific guards (branch-guard the texture
   sample so it never executes at gain 0; single setter owning both gains; ValueError
   if a bias gain is set with a None field; patches unaffected via `#if DOMAIN==0`;
   5 tests). Moot given the physics kill, recorded for completeness.
3. **Metric integrity.** The hero metric is one-sided and gameable by the q_target
   attractor's OPPOSITE failure mode (an over-coherent single-mode ripple). Future
   coupling gates must add CO-gates: a **bidirectional** texture floor
   `0.7 ≤ highfreq(coupled)/highfreq(base) ≤ 1.4` (the lower bound forbids laminar
   collapse) and a **single-mode-share ≤ 0.60** cap, with all metric params frozen
   up front. These were adopted for the confirmation experiment below.

## Cheap empirical confirmation — the fast-nudge sweep

The one genuinely-untested knob the review left open: a **fast nudge** (small
`vort_relax_tau`) so the relax outraces the advective fold. Tested via
`scripts/sw_m3_taunudge.py` (RES=1024, gain=0.8, all four locked gates), sweeping
τ over a 300× span. Hero gate = `banded_coherent_fraction` ratio > 1.05.

| τ (steps) | hero ratio | texture ratio | 1-mode share | source m | verdict |
|---:|---:|---:|---:|---:|:--|
| 600 | 0.617 | 1.186 | 0.166 | 8 | fail |
| 150 | 0.419 | 1.336 | 0.143 | 8 | fail |
| 40  | 0.639 | 1.247 | 0.203 | 8 | fail |
| 10  | 0.922 | 1.149 | 0.163 | 8 | fail |
| 4   | 0.973 | 1.100 | 0.165 | 8 | fail |
| 2   | 0.991 | 1.034 | 0.166 | 8 | fail |

**Result: NO WINDOW.** The hero ratio rises monotonically as the nudge speeds up
but **asymptotes at ~0.99 — it never even reaches the baseline (1.0), let alone the
1.05 gate.** The texture floor held at every τ (no laminar collapse), so this is not
a hollow pass blocked by a CO-gate: even in the limit where the nudge fully
dominates the dynamics (τ=2 steps), the coupling's best is to *match* v1.6's
existing band coherence, never to exceed it.

## Conclusion

**Coherent-DOMINANT band structure is not achievable by vorticity injection into
v1.6, by any injection site (additive or target-bias) or nudge speed.** The thing
destroying coherence is the solver's own nonlinear advective fold, which is intrinsic
and injection-blind. The q_target-bias lever and the fast-nudge lever are both
FALSIFIED. The adversarial gate caught this before a kernel build, saving a full
subagent-driven-development cycle.

**What the coupling IS good for (unchanged):** a real VISUAL fidelity win at
gain≈0.5 — bands plus physically-grounded coherent mid-latitude storm roll-ups,
opposite of v1.6's latitude-agnostic FBM. If M3 ships, it ships as that visual
feature with the automated coherent gate recorded as a known-broadband diagnostic,
NOT as a passing coherent-dominant gate.

**Remaining unexplored directions (each a separate effort, NOT vorticity-injection
into v1.6):** render the baroclinic THICKNESS field directly as a height/albedo
channel (coherence lives in h2, not vorticity — see the laminar root-cause note);
or a genuinely different solver coupling (e.g. forcing the streamfunction, not the
vorticity). Both are large and speculative; neither is queued.

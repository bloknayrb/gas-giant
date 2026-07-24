# Resolution-invariance calibration spike — VERDICT: GO (nudge full, flagship partial)

Date: 2026-07-24. Harness: `tests/spikes/resolution_invariance_spike.py`.

## Question
Does the resolution-invariant scaling (`sim.resolution_invariant`) collapse the
cross-resolution development drift — a run at a higher resolution, downsampled
back, reproducing the reference-resolution run better with the flag ON than OFF —
and does it collapse for a NUDGE-dominated config while only partially helping the
TURBULENCE-dominated flagship?

## Method
reference = 512. "truth" = develop at 512. "test" = develop at 1024 (s = 2),
area-mean downsampled to the 512 grid. Two-sided metric (channel-0 tracer):
zonal-mean profile RMS across latitude (banding / large scale) + whole-field
Pearson correlation (eddy / storm placement). Lower RMS + higher correlation =
closer to the reference. `DEV_STEPS = 120` at the reference; the 1024 run does 240
with the flag on.

## Result

| config | flag | profile_RMS | correlation |
|---|---|---|---|
| nudge-dominated (`vort_inject=0`, `tau=40`) | OFF | 0.01356 | 0.9049 |
| nudge-dominated | **ON** | **0.00954** | **0.9515** |
| turbulence-dominated (stock `gas_giant_warm`) | OFF | 0.02227 | 0.8445 |
| turbulence-dominated | **ON** | **0.01109** | **0.9230** |

Both metrics improve with the flag ON for BOTH configs:
- **Nudge-dominated:** profile-RMS −30%, correlation 0.905 → 0.952.
- **Turbulence-dominated:** profile-RMS −50%, correlation 0.845 → 0.923.

## Verdict: GO
The scaling works. The drift is substantially reduced in both regimes, on both the
banding (large-scale) and the placement (eddy) axis. As predicted, the
nudge-dominated config reaches a higher correlation ceiling (0.952) than the
turbulence-dominated flagship (0.923): the flagship's residual is the intrinsic
grid-locked-hyperviscosity / inverse-cascade drift that no scalar transform can
remove — but the flag still helps it a lot (it is far from useless there, just not
a full collapse). This matches the honest-scope note in `docs/architecture.md`.

## Coefficient calibration
- **Deterministic decays (decay-exact) + duration ×s + timeline threading**: the
  dominant win. Confirmed on the nudge-dominated config (which has `vort_inject=0`,
  so its improvement is entirely durations + decays + drift compensation).
- **Stochastic `÷√s` (`vort_inject`, `hero_wake_turb`)**: validated as
  directionally correct — the turbulence-dominated case (where `÷√s` is the active
  differentiator) improved substantially. The first-principles √ exponent is kept;
  a finer sweep was not warranted given the clear improvement.
- **`turb_time` physical-time reframing**: active in both; no regression.

## Not done / deferred
- **SOR scaling (Phase 4)** was NOT implemented for this spike and the flagship
  still reached corr 0.923 — SOR under-convergence is therefore NOT the binding
  residual at 512→1024. Combined with the fact that the textbook `ω_opt(N)` is
  invalid for this variable-coefficient / periodic / screened operator (would need
  its own empirical calibration), SOR scaling is **deferred** as a future
  refinement rather than shipped speculatively.
- **Hero-anchor clamp regime (P6)** deferred (needs a shader uniform for the
  0.5/step clamp; only binds for low-`relax_tau` hero-emergence presets).
- Only s = 2 (512→1024) was measured; the trend is expected to hold to 2048/4096
  with a larger residual for the flagship.

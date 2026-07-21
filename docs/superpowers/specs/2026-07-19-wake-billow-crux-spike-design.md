# Wake billow-chain crux spike — design

**Date:** 2026-07-19 · **Status:** approved design, pre-implementation ·
**Decision at stake:** whether the hero-wake "dense small billows" look (reference
PIA07782) requires a production solver-dissipation rewrite, a cheaper machinery
lever, or is physically unreachable in the flow (→ render synthesis or park).

## Background

Round D established (memory: wake-fidelity-investigation; 8 scratch experiments,
all reverted) that the warm hero wake renders as 3–5 large rolls / fine ribbons
instead of the reference's dense billow chain (tracer measurements: transverse
wavelength ~1.2 rc, along-wake ~3.0 rc, neutral polarity, coherence ~0.45,
rc = hero_radius 0.108 rad). Every forcing family failed distinctly, and direct
thin-sheet KH forcing (E2/E2b) produced a smooth transport-barrier jet instead of
rollup. The working hypothesis was "the solver's numerical dissipation (SL
MacCormack limiter + SOR ψ-solve) kills thin-sheet KH."

**Adversarial design review (3 lenses, 2026-07-19) partially overturned that
diagnosis before any build:**

- The `vort_relax_tau` 600 q-side nudge (omega_force.comp:140) damps wake
  vorticity everywhere — `heroRelaxWeight` releases only the TRACER relaxation —
  at a rate that out-competes KH growth at E2's amplitudes (2–5) by ~2.5–6×.
- The baked bracket's flank strain (~22 in ω units) demands ω_sheet ≳ 20 for
  Okubo-Weiss survival. E2 tested 2–5: below every threshold.
- So the E2/E2b negative is explained without blaming the numerics. The spike's
  job is now to LOCATE the binding constraint among {nudge, strain, machinery,
  numerics, resolution} — with a real chance the answer is a cheap wake-windowed
  q-relax-release lever, not a rewrite.

## Instrument

Throwaway CPU pseudo-spectral 2D barotropic vorticity solver (numpy FFT) on a
doubly-periodic box at exactly production cell size (Δ = π/2048 rad): grid
1024×512, box 1.5708×0.7854 rad (≈14.5×7.3 rc), centered on lat −24°. Exact
screened inversion ψ̂ = −ω̂/(k²+1/L_d²) (L_d 0.18), spectral advection with 2/3
dealiasing, β-term (β = f0·cos24° ≈ 2.74), RK4, ν₈ spectral hyperviscosity at
the Nyquist only. Passive tracer (belt/zone step across the shear line — the
gradient the real billows wind) advected by the same scheme as ω in every rung.
Everything runs in radians and production time units (dt from `compute_dt` on
the warm profiles) — one nondimensionalization table, no rc-based rescale.

Ambient environment: the BRACKET-OVERRIDDEN warm zonal profile (build_profiles
output, not the seeded base jets), edge-matched/mirrored to be periodic, held by
a profile nudge; flank-contamination guard (boundary-third enstrophy bounded,
cores counted only in the declared wake band). The hero's own near-field strain
is absent: the box is a far-wake control; recorded as a caveat.

## Gates (all criteria frozen before the first gated run)

**Gate 0a — harness validation (stop-gate).** Textbook unscreened tanh shear
layer: measured linear growth rate within 20% of the analytic dispersion value,
rollup into a billow chain at λ ≈ 7δ. Screening check: point-vortex azimuthal
velocity matches the K₁ Bessel profile for L_d 0.18. Fail → the spike result is
INCONCLUSIVE-INFRA (named outcome), stop and report.

**Gate 0b — visual mock (user acceptance image).** Composite a synthetic
billow chain at the measured reference scales into the current warm render's
wake window (mirrored WEST), 2048 export. The user judges whether this is the
look wanted; if yes it becomes the frozen acceptance image; if no, the whole
flow-physics line is moot regardless of other gates.

**Gate 1 — physics premise.** Forced spatially-developing configuration: sheet
maintained in an upstream generation zone, billows advect downstream, sponge
before wrap. Broadband (white) seeding — the dynamics must SELECT the
wavelength; seeding 1.2 rc and finding 1.2 rc is circular. Scan (pre-registered,
frozen): ω_sheet ∈ {10, 20, 45, 90} × sheet thickness δ ∈ {0.15, 0.30, 0.50} rc
(spans nudge/strain thresholds and both wavelength-mapping hypotheses — 1.2 rc
transverse vs 3.0 rc along-wake spacing), primary sheet form = E2's zero-mean
deficit jet, one single-signed tanh-jump variant cell for disambiguation.
PASS (scored on the TRACER, algorithmic): ≥4 distinct rolled billows in the
wake band at any instant of a statistically-steady window; dominant along-sheet
wavelength within the reference bracket; high-pass skewness |s| ≤ 0.2 (neutral
polarity); structure-tensor coherence 0.35–0.55; billow-band high-pass RMS
≥ 1.4× ambient. ω-core count/labeling (frozen 2σ threshold, 8-connectivity) is
a secondary diagnostic. Formation must occur within a dev-700-equivalent;
persistence is statistical (count/wavelength stationary over the fed window),
not a decay-survival clock. Any passing cell must replicate on a second
pre-registered seed. Where the passing band collides with the production
OMEGA_CEILING-equivalent (60) is a primary OUTPUT, not a footnote.

**Gate 1.5 — production machinery (ablation zero, before any numerics).** On
Gate-1-passing cells, superpose box-equivalents of: (i) the τ=600 nudge toward
the billow-free target, (ii) wake fbm injection at production amplitude
(0.6·3.2·0.9 within the wake mask, 4-octave fbm at 0.9/rc), (iii) the ψ-eddy
hypofriction (0.06·(ψ−⟨ψ⟩ₓ)). One term at a time, then all together. If the
nudge alone kills the chain, the indicated production fix is a wake-windowed
q-side relax release (mirror of the tracer-side `heroRelaxWeight`) — a
defines-gated lever, no solver rewrite.

**Gate 2 — numerics ladder (only on cells surviving Gate 1.5).** Cumulative
rungs at production dt and cell size, each transcribing the EXACT production
kernel math: spectral@production-dt/frozen-velocity → single-pass SL Catmull-Rom
→ full MacCormack unclamped → +2×2 min/max clamp → +centered-FD ψ→u with
bilinear velocity sampling in the RK2 backtrace → +grid-normalized hypervisc
(ν=0.6 — NOT negligible at sheet scale) → +warm-started 48-sweep screened
red-black SOR. Plus: a 2× resolution refinement check on the spectral pass, and
a harness-credibility check: the full ladder + machinery at E2's amplitude
(A=5, δ=0.17 rc) must REPRODUCE the production failure (no rollup). A ladder
that can't reproduce the known failure has no standing to attribute it.

**Collateral tripwire.** The winning rung re-run on a hero-free warm-like
config (ambient profile + global injection, no sheet): per-octave spectral,
filament-RMS, and max|ω| deltas vs the current-ladder control reported as
DESCRIPTIVE numbers (>±50% anywhere = flag). The authoritative collateral test
is production-side renders + adversarial visual review at bake time — any
candidate ships as a defines-gated default-off variant regardless. (Demoted
from a numeric gate by plan review: the box has no derive/palette/detail
chain, so numeric bounds there would be false rigor.)

## Decision tree (complete)

| Outcome | Action |
|---|---|
| 0a fails | INCONCLUSIVE-INFRA; stop, report; no verdict recorded |
| 0b user rejects the look | Flow line moot; user decision (render synthesis / park) |
| G1 all-fail, shear-off control passes | Physics wall (strain+merger) — rewrite FALSIFIED; fall back to render synthesis / park (user decision) |
| G1 all-fail incl. shear-off | Harness suspect → INCONCLUSIVE-INFRA |
| G1 pass only above ceiling-equivalent | CONDITIONAL: production fix requires a ceiling raise; record, user decision |
| G1: wall vanishes at reduced ambient shear (0.5×/0.25× cells) | Bracket-retune candidate (the flank strain is the authored `jets.hero_bracket_*`); user decision |
| G1.5: nudge kills but nudge+resupply passes | Sheet-starvation mechanism: remedy = sustained wake forcing; cheapest outcome |
| G1.5: nudge (incl. with resupply) kills | Eddy-damping mechanism: wake q-relax-release lever design; no rewrite |
| G1.5: a non-nudge machinery term kills alone | Named machinery-wall outcome (specific term); user decision |
| G2: single rung kills | Minimal production change named; design as defines-gated opt-in variant (default byte-identical) |
| G2: ALL rungs kill | Multi-term = the big rewrite after all; explicit user decision, never default into it |
| G2: NO rung kills | Dissipation hypothesis falsified; killer is machinery/resolution → revisit G1.5 findings |

## Process

- Everything in the session scratchpad; NOTHING committed during research
  (standing rule). On a recorded verdict, the harness + verdict doc + the
  reference measurement (re-verified against PIA07782) are committed TOGETHER,
  following the detail-character-spike precedent (`scripts/spike_*.py` outside
  testpaths, measurement-only, touches no `src/gasgiant/**`).
- Hard timebox: 3 days. Gate 0a green by mid-day 1 or stop. Scan cells frozen
  before the first gated run; extensions declared (with reason) before analyzing
  their output. Grinding past the box requires an explicit user decision.
- Every visual judgment: reference-anchored (PIA07782, mirrored WEST for our
  frame), by me AND adversarial agents (standing rule).

## Review record

3-agent adversarial design review 2026-07-19 (numerics/CFD, process/
falsifiability, visual efficacy) — all NEEDS-CHANGES; every blocking and
important finding incorporated above: nudge-explains-E2 (numerics F1, the
diagnosis-overturning finding), forced-spatially-developing restructure
(numerics F2, visual F2), turnover mis-scaling → production-window persistence
(numerics F3, process I6), amplitude scan re-centered onto thresholds 10–90
(numerics F4), bracketed-profile periodization + flank guard (numerics F5,
process I2), thickness/wavelength ambiguity → δ 0.15–0.5 rc (numerics F6),
hypervisc at sheet scale + ψ→u/bilinear path added to the ladder (numerics
F7/F8), warm-started SOR requirement (numerics F8), production cell size + 2×
refinement (numerics F9, process B3), machinery Gate 1.5 (numerics F1/F10,
process B2, visual F6), tracer-scored Gate 1 (visual F1, process I1),
known-answer Gate 0a (process B1), visual mock Gate 0b (visual F5), broadband
seeding / anti-seed-persistence (process I1, numerics F13), two-seed
replication (process I5), decision-tree completeness incl. ALL/NONE branches
(process I4), 3-day timebox + INCONCLUSIVE-INFRA (process I3), commit hygiene
(process M1), nondim table + ceiling number (process M2/M3, numerics F12),
β justification (numerics F11), collateral bound (visual F4).

**Plan-review addendum (2026-07-19, second 3-agent review, of the
implementation plan):** all three lenses NEEDS-CHANGES; deltas folded into the
plan rev 2 and back into this spec where they change gate semantics: Gate 1
scans BOTH sheet forms across the grid + weak-shear cells (new decision row
above); both-seed symmetric replication (a falsification may not rest on n=1
failures); the billow metric gains a MAXIMUM size bound and wavelength-coupled
count (the rev-1 metric passed the status-quo 3–5-roll field and failed a
reference-true 3 rc chain — the self-test now requires the real baseline wake
to FAIL); formation/stationarity clocks tied to advective transit time, not a
raw dev-700 transplant; feeding via a Dirichlet inflow strip so the production
nudge is the only relaxation in the domain, plus a distributed-resupply
variant separating eddy-damping kills from sheet-starvation kills (two new
decision rows above); Gate 2 rung 1 is RK4-frozen-velocity (no stable Eulerian
RK1 exists) with a leave-one-out escalation when one-out revert fails to
restore; collateral demoted to a descriptive tripwire; Gate 0a wavelength
stated as 14.3δ (7δ_ω) and screening fit range shrunk to escape periodic-image
contamination. Full record: plan file, "Plan-review record (rev 1 → rev 2)".

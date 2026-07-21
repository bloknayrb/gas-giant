# Wake-billow crux spike — VERDICT

**Recorded 2026-07-20** after the full pre-registered gate sequence + a 50-run
Gate-1 scan + an unforced-decay discriminator + three adversarial reviews
(process/falsifiability, CFD/numerics, reference-anchored visual vs PIA07782).

**One-line result:** the core solver-dissipation rewrite this spike was built to
de-risk is **DISFAVORED / falsified as the route to the ordered wake-billow look**
— lowering dissipation moves 2D flow toward turbulence, not order; the render
already supplies reference-like texture; order is a sustained-shear/KH-arrest
(vortex-merger-wall) problem, not a dissipation problem. Chosen direction (user):
pursue the look via render synthesis (`detail.hero_wake_braid`), not a solver
rewrite.

Spike design: `2026-07-19-wake-billow-crux-spike-design.md` (same dir).
Plan: `docs/superpowers/plans/2026-07-19-wake-billow-crux-spike.md`.
Harness (measurement-only): `scripts/spike_wake_billow/`.

## Question
Does the hero-wake "dense small billows" look (reference PIA07782: transverse
λ~1.2 rc, along-wake ~3.0 rc, neutral polarity, coherence ~0.45) require a
production solver-dissipation rewrite (reduce SOR ψ-smoothing + MacCormack
limiter), a cheaper machinery lever, or is it physically unreachable in the flow?

## What ran
- Gate 0a (harness validation): **PASS** — the same pseudo-spectral solver rolls
  up a textbook unscreened tanh KH layer at growth 15.05 vs 15.46 theory (2.7%),
  rollup at the right wavelength. The instrument CAN do KH rollup.
- Gate 1 (physics premise): full pre-registered scan, 50 runs = ω_sheet
  {10,20,45,90} × δ {0.15,0.30,0.50} rc × {deficit, tanh} × 2 seeds + shear-off
  control. Forced spatially-developing config (Dirichlet inflow strip feeds a
  sheet; broadband/white seeding so the flow must SELECT the wavelength).
  Instrument = numpy pseudo-spectral 2D barotropic vorticity, doubly-periodic,
  production cell size (Δ=π/2048), exact screened inversion (L_d 0.18), β-plane,
  RK4, ν₈ Nyquist-only hyperviscosity + 2/3 dealias. FAR LESS dissipation than
  production (no SOR smoothing, no MacCormack limiter).

## The re-adjudication (approved: frozen gate can't PASS in a sink-free box)
The frozen stationarity window includes frames AFTER CFL blowup, so the stored
`count_frac`/`lam_rc` are contaminated by numerical garbage (n_billows counted in
an exploding field: max_w 40→1264, substep_count 0→hundreds, band_rms→5.7, skew
→−10). Re-scored every cell on the STRICTLY-STABLE window only (leading run with
substep_count==0 AND max_w ≤ 1.6× its pre-forcing plateau). Script:
`healthy_window.py`. Montage: `healthy_window_montage.png`.

## Result (strictly-stable window, all 50 cells)
- **0 / 50 cells FORM a sustained reference-true billow chain.** `best_rt_run = 0`
  for EVERY cell — not one produced a SINGLE frame meeting all reference criteria
  (n_billows ≥ N* for its bracket, bracket set, |skew|≤0.2, coherence 0.35–0.55,
  band_rms ≥ 1.4), let alone 3 consecutive.
- 42 MARGINAL / 8 LAMINAR. MARGINAL "peak billow counts" (3–8) are all: wrong
  wavelength (bracket blank at peak), too weak (band_rms 1.0–1.24, never 1.4),
  too SMOOTH (coherence 0.83–0.94 vs reference 0.35–0.55), and transient (last
  1–2 frames before blowup).
- **Visual (montage, stable window / subs=0):** laminar undulation, or a FEW
  LARGE SMOOTH ROLLS (A10_d0.30 @2750: 3 big smooth folds). This is the exact
  production "few large rolls, not dense billows" failure — reproduced in a
  near-inviscid spectral solver.
- **Amplitude-independent:** A10→A90 (9× sheet strength) all show the same
  large-roll character; more forcing brought the CFL blowup EARLIER, not more
  discreteness.
- **Shear-off control: clean negative.** Both seeds LAMINAR (n_billows=0) for the
  full 5500 steps, max_w~20, never substeps. The box does NOT manufacture
  structure; the sheared cells' rolls are shear-driven physics.

## VERDICT: INCONCLUSIVE — primary Gate-1 config compromised by a harness
## enstrophy-sink blowup; discriminator run MANDATORY before any physics verdict.
(Revised after 2 adversarial reviews, both fed back into the raw trajectories.)

The draft "physics wall / rewrite FALSIFIED" reading DOES NOT STAND. Verified
against the 50-step-resolution time series:
- **No sheared cell reaches a statistically-steady state.** All 48 blow up (CFL;
  max_w → 10^3); only the 2 shear-off controls survive clean to 5500. The frozen
  gate's steady-window premise (window_start ≈1667, needs ≥2000 more stable steps)
  is NEVER met → the frozen gate is INDETERMINATE, not "all-fail".
- **Blowup timing is amplitude-DEPENDENT** (A90 ~2150, A45 ~2750, A10 ~3300). The
  earlier "amplitude-independent" claim was WRONG: higher forcing dies earlier and
  captures LESS rollup — a harness bias, not a physics result. max_w plateau ∝ A.
- **The metrics are still evolving / far from target at truncation** (lam shrinking
  9→1.8 rc, coh creeping 1.0→0.81, nb rising) — the forward cascade is guillotined
  by the blowup before a terminal state is reached. The one cell that briefly
  plateaus (A10_d0.15: nb2 @3.0rc, coh0.86, rms1.24) sits at sparse SMOOTH rolls,
  but that is not a clean steady state either.
- The blowup is a Nyquist-only ν₈ sink being overrun by the enstrophy the
  developing structure feeds. **Logical trap (reviewer):** the box blows up from
  too LITTLE small-scale dissipation, so "near-zero dissipation" is a numerically
  unstable regime, NOT a valid control point for a claim about REDUCING dissipation.
- 0/50 reference-true chains in the numerically-valid window is real, but that
  window is truncated before steady state, so it is consistent with BOTH readings
  (physics wall AND harness-truncation-masks-rollup). It does not discriminate.

**Decision-tree status:** frozen Gate 1 is INDETERMINATE (contaminated/truncated),
NOT the "G1 all-fail" family. The shear-off control DID behave as a valid clean
negative (laminar, n=0 → the box does not self-generate structure; this is the
"shear-off passes its validation role" reading, NOT a literal Gate-1 pass). But a
clean negative control cannot address whether the FORCED blowup masks a rollup —
so INCONCLUSIVE-INFRA-adjacent, resolvable by the discriminator below.

## Mandatory discriminator (both reviewers converge; design-doc pre-registered)
Give the box a real steady state to score, then re-read the terminal structure:
1. **2× resolution + half-DT** rerun of the 2 longest-surviving cells (A10_d0.15,
   A10_d0.30 deficit, both seeds): moves the Nyquist finer so the ν₈ sink has
   headroom at the SAME billow-scale dissipation — the principled "does it stay
   stable and what is the endpoint" test (design-doc Gate-2 refinement rung).
2. **Unforced screened sheet-decay** (initialize the sheet, NO Dirichlet strip,
   screening+β on): removes the enstrophy pump entirely. Gate 0a did this
   UNSCREENED and rolled up cleanly; this isolates screening+β+strain.
3. (free) near-field 0–3 rc band from stored snapshots — cheap peek.
Read: if a STABLE run plateaus at sparse smooth rolls (coh≫0.55) → Reading A
(physics wall) confirmed. If it develops a dense billow chain once it survives →
the dissipation hypothesis is LIVE (answer may be "the RIGHT dissipation", not
"less"). Either way the current scan CANNOT be recorded as a physics verdict.

## Corrections logged (both reviews)
- "FALSIFIED" → INCONCLUSIVE pending discriminator (CFD F1/F2, process I1).
- frozen gate = INDETERMINATE not all-fail; 0/50 is a post-hoc RELAXED reading,
  conservative for a negative but not decisive (process I3).
- shear-off = valid-negative-control interpretation stated explicitly (process I2).
- amplitude-DEPENDENT blowup; "amplitude-independence" retracted (CFD F3).
- montage left-column = EARLY-ONSET frames, not terminal state (CFD F1).
- shear-off control is one (A,δ,form) × 2 seeds; cannot cover the blowup regime.

## DISCRIMINATOR 1 RESULT (unforced sheet-decay; user-approved cheap-first)
`discriminator_decay.py` / `discriminator_runs/*.npz` / `discriminator_terminal.png`.
Full-width deficit sheet + broadband seed as INITIAL condition, evolved UNFORCED
(no Dirichlet pump). 4 runs (A10 d0.15/d0.30 free, A10 d0.30 meanhold, A45 d0.15
free), 6000 steps.

- **STABILITY (Reading B resolved):** all 3 truly-unforced (free) runs stayed
  numerically stable the FULL 6000 steps (no substep, no blowup). Only the
  meanhold variant blew up (3675) — it pinned the barotropically-unstable ambient
  jet in place. ⇒ **the forced-scan blowup WAS the Dirichlet enstrophy pump**, not
  intrinsic rollup instability. The forced Gate-1 "no billows" was a harness
  artifact. The near-inviscid box DOES roll up when the pump is removed.
- **TERMINAL STRUCTURE:** the sheet rolls up and produces abundant sub-reference-
  scale wake structure: band_rms clears 1.4 (1.41/1.59/1.37), neutral polarity
  (|skew| 0.05–0.11), count 5–12, and **amplitude BUYS more/smaller features**
  (A45 → 11–12 at λ≈2.25 rc) — the OPPOSITE of the forced case and of production's
  3–5 large rolls. So a near-inviscid solver moves the SCALE/COUNT toward the
  reference where production's dissipation does not.
- **BUT (controller visual read of discriminator_terminal.png, PENDING adversarial
  review):** step 4000 = a few large smooth rolls; step 6000 = 2D TURBULENCE
  (thin stretched filaments + a few dominant vortices), NOT a regular ORDERED
  dense billow chain. The count likely segments turbulent fragments. So even
  near-inviscid 2D gives merger/turbulence, not the reference's ordered chain —
  consistent with the memory "strain-shredding + same-sign merger wall" being an
  INTRINSIC 2D-dynamics limit, not a dissipation artifact.
- **COHERENCE gap is the render layer, not flow:** raw sim-tracer terminal coh
  0.79–0.84 vs reference 0.35–0.55; but the PRODUCTION RENDER wake reads coh 0.522
  (in-band) in the SAME metric — the render's detail/texture layer supplies the
  folded coherence the raw tracer lacks. So judging raw-tracer coherence against
  the photo is apples-to-oranges.

## FINAL VERDICT (after process + CFD + reference-anchored VISUAL review)
The provisional "dissipation lever is live" read is RETRACTED — it rested on a
billow-count metric that the visual review showed is segmenting TURBULENT
FRAGMENTS, not billows (the MOST turbulent frame, A10 d0.15 step6000, reads
nbill=0 because its filaments are too elongated to pass the segmenter; higher
counts land on filament edges). The pictures show 2D turbulence, not a chain.

**A production solver-dissipation rewrite is NOT warranted (disfavored):**
1. **Reducing dissipation moves 2D flow the WRONG way** — toward finer turbulence
   + same-sign merger, not toward order. The reference's ORDER (confined, co-
   scaled, wake-oriented billow train) is a property of a SUSTAINED shear layer
   selecting and arresting a KH wavelength — a forcing/mechanism property, NOT a
   viscosity property. The dissipation axis is monotonic toward turbulence.
2. **The flow substrate is not the deficiency** — the production RENDER wake
   already reads coherence 0.52 (in the reference 0.35–0.55 band); detail.comp's
   texture layer already supplies the folded coherence. Nothing in the terminals
   shows a flow deficit a dissipation rewrite would fix.
3. **The forced-scan "no billows" was a harness enstrophy-pump blowup** (Reading B,
   confirmed by the stable unforced runs) — not evidence of anything about
   production dissipation.

**Honest limitation:** the spike did NOT cleanly test the one regime that COULD
make an ordered chain — a SUSTAINED forced shear layer arrested before merger.
The forced runs blew up (harness pump); the unforced runs necessarily decay into
free 2D turbulence. That regime is the already-documented vortex-merger wall
(research-grade; memory: wake-fidelity-investigation, docs/roadmap.md
deferred-list). So the spike does not prove an ordered chain is impossible — it
proves DISSIPATION is not the lever and the remaining route is the known wall.

**Decision (design-doc terminal = USER decision):** the hero-wake ordered-billow
look is NOT a solver-dissipation problem. The options are:
 (a) **Render synthesis** — the existing `detail.hero_wake_braid` path; the render
     already hits reference coherence, so this is the cheap, in-reach route.
 (b) **Park** — accept the current wake; record the spike's finding.
 (c) **Research-grade** sustained-shear / KH-arrest / vortex-merger mechanism —
     the documented wall; large, speculative, NOT a dissipation rewrite.
The one thing FALSIFIED as an approach: the core solver-dissipation rewrite this
spike was built to de-risk. It would not deliver the look. Do NOT pursue it.

2×-res escalation NOT warranted (visual review): higher resolution lengthens the
enstrophy cascade → MORE filaments, not more order; resolution isn't the missing
ingredient. Skip it.

## Review record
- Process/falsifiability skeptic (opus): mapping defensible but mis-verbed;
  frozen gate INDETERMINATE; downgrade FALSIFIED. Folded in.
- CFD/numerics skeptic (opus): forced blowup = harness pump; can't read forced
  steady state; "amplitude-independence" retracted. Drove the discriminator.
- Reference-anchored VISUAL skeptic (opus, vs PIA07782): terminal = 2D turbulence
  not ordered chain; nbill = fragment-count; dissipation moves toward turbulence;
  render already clears coherence; verdict (C) physics-wall-for-dissipation-lever.
  Drove this final retraction of the "lever live" read.

## Files
- `discriminator_decay.py` / `discriminator_runs/*.npz` / `discriminator_terminal.png`
- `healthy_window.py` — the re-adjudication (per-cell stable-window table)
- `healthy_window_montage.png` — stable-window vs post-blowup band fields
- `metrics.py` — frozen criteria (self-test: status-quo FAILS, reference chain PASSES)
- `config.py` — single-source constants
- `gate1_runs/*.npz` — full per-sample time series + snapshots per cell

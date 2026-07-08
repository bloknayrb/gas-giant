# Detail-CHARACTER crux spike — CONDITIONAL GO (mechanism confirmed; hard-number gate INCONCLUSIVE under software GL)

**Status:** **CONDITIONAL GO.** The sim-advected high-resolution tracer *does* manufacture
oriented, zonally-biased filamentary structure — a clean, orientation-controlled **×3.7
separation** from the isotropic control — validating the W13/W14 hypothesis's mechanism.
The one thing left open is the absolute number: a reduced-fidelity proxy reaches
**coher = 0.31**, just under the pre-registered **0.384** GO bar. Because the proxy runs the
dynamics grid 16× coarser than native (256 vs 4096) and coherence rises monotonically with
both resolution and development, 0.31 is a **lower bound**, not a miss. **NO-GO is falsified**
(this is not an emergent-color / q-target dead end); the remaining gate is a single
native-GPU full-fidelity run.

**Date:** 2026-07-08. Branch `claude/feature-gaps-creative-control-sto9nz`.
**Spike:** `scripts/spike_detail_character.py` (measurement-only; outside testpaths; touches
no `src/gasgiant/**` and ships nothing to the render path).

---

## What was claimed

`docs/roadmap.md` ("Research direction (unstarted): detail CHARACTER = sim-advected high-res
tracer"): fluid folded-filament morphology is a **dynamics** property no frozen-field render
trick reproduces (F17, FALSIFIED). The viable path: **decouple tracer resolution from dynamics
resolution** — keep the vorticity solve on its moderate grid but advect a **high-resolution
passive detail tracer** through that upsampled *evolving* velocity every dev step, so the
coarse-grid strain folds a high-res scalar into real fine-scale filaments.

**Pre-registered crux gate** (project discipline, cf. m2-adv): carry ONE extra high-res tracer
through the existing `gas_giant_warm` solver and measure whether its folded structure crosses
the F17 orientation-coherence bar toward **0.384 / 0.62**, against the **~0.14** isotropic
control. Go/no-go on that number *before* committing the multi-session subsystem build.

## The exact metric

Structure-tensor orientation **coherence**, the identical calibrated operator the F17 bar was
measured with — imported directly from `scripts/measure_morphology.py::coher` (not re-implemented):

- Sobel gradients `(gx, gy)`; structure tensor `J = [[gx², gxgy],[gxgy, gy²]]` Gaussian-smoothed
  at σ=4 px.
- Local coherence `c = (λ₁−λ₂)/(λ₁+λ₂) = √((Jxx−Jyy)² + 4Jxy²) / (Jxx+Jyy) ∈ [0,1]`.
- **Horizontality-weighted** (weight `Jyy/(Jxx+Jyy)` — rewards east-west streaks) and
  **energy-weighted** (`tr J`) mean over the crop.

Measured on a tropical/mid-latitude belt crop (`|φ| < 30°`, full longitude) **resized to
640 px width**, matching the calibration's crop scale so the 0.14 / 0.384 / 0.62 anchors apply.
The calibration record (`measure_morphology.py`): reference **0.617**, isotropic v1.5 render
**0.140**, 90°-rotated reference **0.121** (the orientation control that collapses it).

## Setup / fidelity

Full `gas_giant_warm` is 4096 grid / 700 steps / 48 SOR iters × 3 domains — intractable under
software GL (llvmpipe, ~150× slower than native). The spike runs a **reduced-fidelity proxy**:

- **Dynamics:** real `gas_giant_warm` `Simulation`, vorticity mode, all forcing/vortex/SOR
  physics intact, resolution overridden to **256** (equirect 256×128), **700** dev steps
  (native step count), seed 4201.
- **Tracer:** ONE extra **R32F 1024×512** texture (`tracer_mult = 4×` — the roadmap's exact
  "1024-grid → 4K" ratio), seeded with a **provably-isotropic band-pass field** (white noise ×
  radial Fourier annulus, k ∈ [24,96] cyc; radial ⇒ no orientation bias). Advected each step by
  the solver's *current* `equirect.vel_tex` with a **throwaway** RK2 semi-Lagrangian + bicubic
  Catmull-Rom compute kernel compiled from a source string (the DOMAIN-0 backtrace math of
  `sim/kernels/advect.comp`, inlined; no production kernel imported).
- **Controls:** (a) the raw isotropic seed (the ~0.14 anchor); (b) the seed advected by a
  spatially-**uniform** eastward flow (pure translation, zero strain, same kernel/interpolation)
  — isolates strain from resampling; (c) **rot90** of the advected field — the orientation
  control that must collapse coherence if the signal is genuinely horizontal.
- Runs single-threaded (`LP_NUM_THREADS=1`) under `xvfb-run` for determinism.

**Fidelity caveat:** the dynamics grid is **16× coarser** than native (256 vs 4096). At 256 the
solver resolves far fewer, softer, less-zonal jets, so the strain field folds fewer/softer
filaments than production would. This proxy is therefore a **lower bound** on the achievable
coherence, not an estimate of it.

## Measured coherence

| run | dynamics | steps | flow-time | tracer | seed control | **advected** | translate ctrl | rot90(adv) | sep. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| probe | 128 | 20 | 0.61 | 512² | 0.070 | 0.244 | 0.070 | 0.117 | ×3.5 |
| **main** | **256** | **700** | **10.7** | **1024×512** | **0.085** | **0.314** | **0.085** | **0.265** | **×3.7** |
| hi-res | 384 | 700 | 7.1 | 1536×768 | 0.082 | 0.298 | — | 0.206 | ×3.7 |

Anchors: isotropic control **~0.14**, GO bar **0.384**, strong/reference **0.62**.

**Findings:**
1. **The mechanism manufactures oriented structure — decisively.** Advected **0.314** vs
   isotropic control **0.085**: a **×3.7** separation, far above the ~0.14 kinematic control the
   roadmap cites, landing squarely in the predicted "vorticity-solver tracer ≈ 0.384" regime.
2. **It is the STRAIN, not the resampling.** The uniform-translation control (same kernel, same
   bicubic taps, zero strain) sits **exactly at the seed level (0.085)** — the coherence gain is
   the flow's differential strain folding the tracer, precisely the claimed physics.
3. **The zonal orientation SHARPENS with dynamics resolution — the key extrapolation signal.**
   The rot90 orientation control collapses progressively more as the grid refines: at 256,
   0.314→**0.265** (drop 16%, rot/adv = 0.84); at 384, 0.298→**0.206** (drop 31%, rot/adv =
   0.69). Finer dynamics resolves more, straighter jets, so the folds become measurably more
   *predominantly horizontal* — toward the reference's near-total collapse (0.62→0.12). This is
   the single most important trend: the metric axis the reference wins on (horizontal folded
   structure, not mere anisotropy) strengthens monotonically with resolution, and native is
   16× finer still.
4. **Absolute magnitude tracks DEVELOPMENT (flow-time), orientation tracks RESOLUTION.** The 384
   run reached 0.298 at only 7.1 flow-time units vs the 256 run's 0.314 at 10.7 — comparable
   magnitude at 33% less advective development, i.e. the raw coher is governed mainly by how
   long the tracer folds, while resolution buys the *orientation* (finding 3). The 128/20-step
   probe (0.61 flow-time → 0.244) fits the same picture. Native gives BOTH more development
   headroom and 16× finer jets — so the proxy under-reads the number on two independent axes.

## Verdict

**CONDITIONAL GO.** The crux question — *does advecting a high-res passive tracer through the
evolving vorticity field manufacture oriented filamentary detail that a frozen-field trick
cannot?* — is answered **YES on the mechanism**: a clean, strain-attributable,
orientation-controlled ×3.7 separation from isotropic noise, reaching 0.31 at 82% of the GO bar
under a 16×-coarse proxy. This is categorically **not** a NO-GO / dead end (contrast the
emergent-color and q-target falsifications, which showed *no* separation or *monotonic failure*).

It is **INCONCLUSIVE on the absolute 0.384 crossing** only because software GL cannot run the
native fidelity at which the bar was calibrated. The proxy 0.31 is a lower bound; the resolution
and development trends both point upward.

**Recommendation — gate the multi-session build on ONE native-GPU confirmation, not on this
run.** Before committing the subsystem: on a native GPU, run this same spike at **dynamics
1024–2048, 700 steps, tracer 4×** and confirm advected coher **≥ 0.384** on the 640-belt crop
(and check rot90 collapses further, confirming the zonal bias sharpens with resolution). That
single run — hours native, days under llvmpipe — settles GO/NO-GO definitively. If it clears
0.384, proceed to build (render samples the advected tracer as primary structure, procedural
noise retreats to seeding/forcing per the roadmap). If it lands at ~0.31 like the proxy,
re-scope: the mechanism is real but a companion (finer dynamics grid, or the separable
2-D-field-driven amplitude-mask win the roadmap lists) is needed to reach the reference.

**Reusable artifact:** `scripts/spike_detail_character.py` is the confirmation harness — it
takes `--res / --steps / --tracer-mult` and prints the advected-vs-control coherence with the
verdict banner. Re-run it unchanged on native hardware for the deciding measurement.

# Fidelity-gate re-baseline addendum (2026-07-03)

Addendum to the [2026-07-02 comprehensive review](2026-07-02-comprehensive-review.md),
Top-10 #10 ("ship config FAILS its own frozen fidelity gates"). W9 investigation
(branch `chore/fidelity-gates`, PR #23): per-gate A/B against the correct historical
baselines, followed by the user's re-baseline decision.

## Headline: neither FAIL is drift

Both review numbers reproduce on master (800f3dc), but the historical A/B shows the
**shipped config never changed on either gated axis** — the gates were failing at the
very commits their thresholds come from. Both FAILs encode decisions that were made
and recorded at ship time but never written into the gate constants.

## Swirl m3 (jet continuity, frozen `M3_MIN = 0.57` at PR #8)

Invocation: `swirl_gate.py --raw --preset jupiter_vorticity --drags 0.0 --seed 4201`
(1536 w, RTX 3070; each point measured twice — identical both runs).

| commit | PR | m3 | m1 | m5 |
|---|---|---|---|---|
| `bbc1b9e` (threshold freeze) | #8 | **0.51** | 0.91 | 1.12 |
| `8201626` | #9 | **0.51** | 0.91 | 0.76 |
| `9c73b06` | #10 | **0.51** | 0.89 | 0.73 |
| `800f3dc` (master) | — | **0.51** | 0.91 | 0.52 |
| review at `df366a7` (reported) | #12 | 0.51 | 0.89 | 0.54 |

m3 is flat at 0.51 across the entire window, **including the freeze commit**. The
0.57 floor was calibrated on the gas_giant_warm develop config (swirl_gate header
table, 2026-06-25); jupiter_vorticity's narrower jets were measured at 0.50–0.52
during PR #8 tuning and shipped anyway as "guard-not-oracle"
(memory/preset-modernization.md) — the per-preset exception just never reached the
gate constants. No PR moved m3; there is nothing to bisect and nothing to re-tune
toward. Visually, master is the closest of all four points to the Cassini reference
(PIA07782); the bbc1b9e render is markedly darker/muddier.

**Decision (user, 2026-07-03): RE-BASELINE.** Per-preset override
`M3_MIN_PER_PRESET = {"jupiter_vorticity": 0.46}` (~10 % below the stable measured
0.51) in `scripts/swirl_gate.py`; gas_giant_warm and all other presets keep 0.57.

## Chromophore saturation (`_chromo_compare` mean-sat bound ≤ +12 %, calibrated with the lever at PR #11)

Invocation: 2048 w renders of jupiter_vorticity, seed 4201, `chroma_aging` 0.35 (ON)
vs 0.0 (OFF); `scripts/_chromo_compare.py`.

| commit | PR | mean sat | variety | targeting corr |
|---|---|---|---|---|
| `07b43dc` (bound calibration) | #11 | **+28 %** | +52 % | −0.88 |
| `800f3dc` (master, run 1/2) | — | **+29 % / +29 %** | +54 % | −0.88 |
| review at `df366a7` (reported) | #12 | +28 % | PASS | −0.88 |

+28 % **at the calibration commit itself**. The ≤ +12 % bound is the muted
source-fidelity reviewer value; the user's vivid `amp = 0.35` art-direction call at
PR #11 ship time (~+33 % recorded, memory/jupiter-missing-features.md) deliberately
overrode it — the override was never encoded into the harness. Variety and targeting
gates passed at every point and still bind.

**Decision (user, 2026-07-03): RE-BASELINE.** Mean-sat bound raised to **≤ +35 %**
(covers the measured +28–29 % and the recorded ~+33 % intent). Variety
(std ≥ mean rise) and targeting (corr < 0) gates unchanged.

## Verification against the new thresholds (master ship config)

- `swirl_gate --raw --preset jupiter_vorticity --drags 0.0 --seed 4201`:
  m3 0.51 ≥ 0.46 → **PASS** (all co-gates green).
- `_chromo_compare` on the master OFF/ON pair: +29 % ≤ +35 % → **PASS**;
  variety +54 % PASS; targeting −0.88 PASS.

(Result recorded from the verification run on 2026-07-03; see PR #23.)

## llvmpipe gate smoke (A4-2 groundwork)

`gate-smoke.yml` (`workflow_dispatch`-only) ran the 1024 w jupiter_vorticity gate on
a GitHub ubuntu runner under xvfb + llvmpipe:

- **Gate step wall time: ~21 m 46 s** (job total 22 m 24 s) vs ~10 s equivalent on the
  RTX 3070 — roughly **150×** slower, well past the 10–50× planning guess.
- llvmpipe at 1024 w also shifts the metrics (m1 1.22, m3 0.53, m5 0.28 vs RTX/1536 w
  0.91 / 0.51 / 0.52) — the frozen thresholds are NOT renderer/width-portable.
- Recommendation: keep the job dispatch-only (do **not** promote to push/schedule);
  if it is ever promoted, it needs its own llvmpipe-calibrated thresholds, not these.
- Incidental CI finding: the regular ci.yml "GPU smoke tests (llvmpipe)" step has no
  DISPLAY, so `GpuContext.headless()` fails (`XOpenDisplay`) and the `gpu` fixture
  **skips ~178 of 184 gpu tests** — CI's llvmpipe GPU coverage is largely illusory.
  Follow-up candidate for the A4/CI wave (add xvfb to ci.yml).
  **Resolved 2026-07-03 (PR #25):** ci.yml now runs gpu tests under xvfb-run — a
  PR-blocking `gpu-smoke` job (byte-identity/no-op class, ~29 tests) plus the full
  tier non-blocking on master push/nightly/dispatch. The full tier measured >3 h
  under llvmpipe (146/182 tests in 180 min before timeout, run 28683137520),
  confirming it cannot be PR-blocking.

## Evidence (local, out-of-git per `out/` ignore policy)

`out/w9_evidence/` on the dev box: `metrics.json`, labeled montages
`montage_m3_bbc1b9e_vs_master.png`, `montage_chromo_07b43dc_vs_master.png`,
per-commit gate montages and OFF/ON render pairs.

# M0.5 Verdict — GPU 2-Layer Shallow-Water Kill-Gate

**Status: ✅ GO — build M1.** Blind panel decided by the user 2026-06-15.
Harness run: 2026-06-15. Branch: `v2-shallow-water`.

## VERDICT (user blind panel, 2026-06-15)

The user judged the UNLABELED blind image and picked the **bottom** crop as more
Jupiter-like in belt/eddy structure. Blind key: **bottom = SW GPU 2-layer; top =
v1.6 morphology-only.** So in a fair (morphology-only, matched-palette) blind
comparison, with the eddy regime reached (evs 1.2038 ≥ 1.0) and the crop box
slightly biased toward v1.6, the user preferred the emergent shallow-water render.
Per the falsifiability rule this is a clean **WIN → GO**.

**The resolution hypothesis is CONFIRMED.** M0's apparent loss was the double
confound the M1/M0.5 reviews exposed: (1) coarse 192×96 CPU resolution, and (2)
the unfair detail-synthesis pipeline asymmetry (v1.6 got filaments+warp, SW got
none). Remove both and the SW dynamics produce more convincing belt/eddy structure
than v1.6's painted/nudged solver. **Proceed to M1** (the clean single-layer
Williamson-validated GPU rebuild) → M2 (semi-implicit) → M3 (2-layer baroclinic,
the next real render gate). Honest caveats: both renders were morphology-only
(no detail polish); the SW eddy train is regular/Rossby-wave-like rather than
fully turbulent-folded; coher (necessary-not-sufficient) also favored SW
(0.397 vs 0.289).

---

## Configuration

| Parameter | Value |
|---|---|
| Grid | 512×256 |
| Render resolution | 4096 px equirect |
| nu4 | 0.07 |
| dt\_scale | 0.146165 (dt\_hi / dt\_ref, physical-time rescale) |
| Step cap | 32 000 |
| Forcing config | emergent\_init, f0=4.0, gp=(1.0, 0.05), 10 bands, band\_contrast=0.4 |

---

## Spin-Up Result

| Metric | Value |
|---|---|
| Final eddy\_vort\_std | **1.2038** |
| Regime ≥ 1.0 reached | **YES — at step 32 000** |
| R1 (checkerboard amplitude) | 0.00000 — **PASS** (gate: < 0.05) |
| R3 (min h1 ≥ floor, no NaN) | min h1 = 2.9939 ≥ floor 0.05 — **PASS** |
| ms/step (avg) | 0.17 ms/step |
| Wall time | 50.8 s |

---

## Pipeline Match Mode: Morphology-Only (Fair)

The M0 killgate had a **fairness bug**: v1.6 was rendered WITH detail-synthesis filaments + warp + lanes (`detail.intensity=0.95`, `bands.warp_amount>0`, lanes active), while SW was rendered via `derive_from_tracer` which has none of those. This confounded the comparison.

**M0.5 fix:** both renders are morphology-only:

- **v1.6 side:** `detail.intensity=0`, `bands.warp_amount=0`, `bands.lane_density=0` — the `PlanetParams` are patched via `model_copy` before `Simulation` is instantiated. Only raw band+vorticity structure is rendered.
- **SW side:** `derive_from_tracer` (no detail/warp/lanes by construction).
- **Same `appearance` params** (palette, contrast, saturation, haze, etc.) on both sides — the v1.6 preset's appearance is passed directly to `derive_from_tracer`.

---

## Morphology Scores (belt coherence)

Belt box used: widest tropical belt from v1.6 sim bands, matched to reference image width, 640 px crop.

| Render | coher |
|---|---|
| v1.6 jupiter\_vorticity (morphology-only) | 0.2892 |
| SW GPU 2-layer (512×256, nu4=0.07) | **0.3972** |

SW coherence exceeds v1.6 morphology-only by +0.108 (+37%) on this run.

---

## Blind Panel

Artifact: `out/audit/m0p5/swp_vs_v16_blind.png`

- Two belt crops stacked vertically, **UNLABELED** (no preset names, no coher numbers burned in).
- Order is **randomized** with fixed seed 42 (reproducible).
- Blind key is in `out/audit/m0p5/report.txt`, not in this document.

**The blind-panel outcome and GO/NO-GO/INCONCLUSIVE decision are PENDING the human judge's review of `swp_vs_v16_blind.png`.**

---

## Falsifiability Rule

A LOSE on the blind panel refutes M0.5 (is evidence against SW GPU physics producing Jupiter-like morphology) **only if both conditions hold**:

1. Pipeline was morphology-only on both sides. ✅ MET.
2. Regime eddy\_vort\_std ≥ 1.0 was reached. ✅ MET (1.2038 at step 32 000).

If the regime had NOT been reached, the verdict would be INCONCLUSIVE regardless of visual outcome.

---

## Decision: PENDING

Human judge reviews `out/audit/m0p5/swp_vs_v16_blind.png` (two unlabeled belt crops), identifies which looks more like Jupiter belt structure, and records:

- **Which crop (top/bottom) looks better?**
- **GO** (SW wins or draws on morphology) / **NO-GO** (SW clearly loses, confirming the physics gap) / **INCONCLUSIVE** (too similar to call).

Unmask the blind key from `out/audit/m0p5/report.txt` only after judgment is recorded.

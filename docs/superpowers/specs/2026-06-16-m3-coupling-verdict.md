# M3-coupling — Render Gate Verdict

**Date:** 2026-06-16. **Branch:** `v2-m3-baroclinic`.
**Status:** MECHANISM VALIDATED + VISUAL FIDELITY WIN (operating gain ≈ 0.5);
automated coherent-domination gate NOT passed (additive injection is broadband).

## What replaced the old gate
The T9 direct-render gate (top-layer eddy Rossby) is **closed** — falsified by two
adversarial passes (see `2026-06-16-m3-coupling-design.md`): the baroclinic field is
the wrong *direct* render driver. This gate measures the COUPLED render instead:
the validated 2-layer baroclinic instability supplies a physically-grounded vorticity
SOURCE at the active mid-latitudes; v1.6's barotropic turbulence + render make it
look natural.

## Metric note (important — the gate metric was corrected mid-finalization)
The first gate keyed on `latitude_concentration` (per-row eddy *variance* in the
active band vs outside). A gain sweep showed this is the **wrong** gate: coherent,
organized waves carry *less* row-variance than the broadband filament hash they
replace, so the metric paradoxically *drops* when the imprint becomes more organized.
It was replaced by the **hero** metric `banded_coherent_fraction` — the fraction of
active-band eddy energy carried by coherent low zonal wavenumbers (m=1..12) — which
directly matches the physical claim ("add coherent low-m structure at the active
bands"). `latitude_concentration` is retained only as a reported broadband diagnostic.

## Gate results (authoritative, RES=2048)

| gain | banded_coherent_fraction (base→coupled, ratio; gate >1.05) | latitude_concentration (diag) | texture ratio (gate 0.5–2.0) | source m |
|---|---|---|---|---|
| 0.5 | 0.413 → 0.217 (0.53) **FAIL** | 0.962 → 1.091 (1.13) | 1.99 **OK** | 8 OK |
| 1.2 | 0.413 → 0.190 (0.46) **FAIL** | 0.962 → 1.059 (1.10) | 2.07 (just over) | 8 OK |

Blind panel: `out/audit/m3/gate/gate_{baseline,source,coupled}.png`
(`gate_coupled.png` / `gate_coupled_g05.png` = gain 0.5, the operating point).

**Reading the table:** at production resolution the coupling is *strong*, not a weak
modulation — it roughly **doubles** the active-band texture energy and **raises**
broadband concentration (1.10–1.13×). What it does NOT do is raise the coherent
*fraction*: the baroclinic roll-ups shed broadband filament content at their sheared
edges, so absolute coherent (m≤12) power rises but total band energy rises faster →
the fraction falls. The additive source injects **broadband eddy energy**, not a
**coherent-low-m-dominant** imprint. The hero gate correctly fails; it was defined
from the claim and is not loosened.

## Visual verdict (the blind panel — the real arbiter, per design risk R4)
At **gain 0.5** the coupled disk keeps v1.6's clean banded structure, palette, and
combed filaments **and** gains a rich set of coherent vortex roll-ups + a festoon
chain through the baroclinically-active mid-latitudes (north strong, south weak —
matching the m=8 source). It reads as a more dynamically complete Jupiter (bands
*plus* storms) without losing the banded character. At gain 1.2 the turbulence
begins to overwhelm the elegant bands. **Operating point: gain ≈ 0.5.**

So by the blind-panel standard the coupling is a **fidelity win**; by the strict
automated coherent-domination standard it is **not yet** a pass. Both statements are
true and recorded honestly.

## Gain calibration
Swept gain ∈ {0.5, 0.8, 1.2, 1.5}. The coherent-fraction gate fails at every gain
(monotone broadband injection). **Chosen operating gain = 0.5**: it maximizes the
natural look (eddies enrich the bands rather than drowning them) while keeping the
texture ratio under the 2.0 ceiling. Higher gains add more eddies but progressively
disrupt the bands and push texture energy over budget.

## Cadence benchmark + residency decision
- Source cadence (RES=2048): baro ~20s + upload ~1.1s vs v1.6 ~1.1s over 22 updates.
- The residency rule reports `consider-residency`, BUT this is an artifact of the
  benchmark resolution: the CPU baroclinic advance is fixed (~3300 steps) while the
  GPU v1.6 work at dev_steps=700 is tiny at any resolution we time here (the heavy
  GPU cost is the final 16K *render*, not the dev steps). The per-cadence CPU advance
  + one small-texture upload is cheap in absolute terms (~1 ms upload).
- **Decision: stay with option-(a)** (CPU advance + texture re-upload). A GPU-resident
  baroclinic solver is NOT justified by this data; the source cadence is not the
  bottleneck. (The `SwpSolver` resident-GPU refactor remains unjustified, as predicted.)
- Cadence must keep the baroclinic solver in its **healthy pre-outcrop window**
  (outcrop ≈ step 12500): warmup 8000 + ~22×150 ≈ ends 11300. Advancing too fast
  (e.g. 400/update) runs past outcrop and freezes the source on a degenerate state —
  this caused the earlier de-concentration artifact and is now the gate-script default.

## Honest read
The coupling **mechanism is validated and strong**: a physically-grounded, evolving
baroclinic source visibly places vigorous eddies/roll-ups at the supercritical
mid-latitudes, and at gain ≈ 0.5 the result is a clear visual fidelity improvement
that preserves v1.6's natural texture. However, **additive injection after the nudge
produces broadband turbulence, not a coherent-low-m-dominant imprint**, so it does
not pass the automated coherent-fraction gate at any gain — this is precisely design
risk **R1** ("v1.6 re-equilibrates; even a per-step source may only modulate"), here
seen as: the source adds eddy energy but the nonlinear solver shreds it into
broadband filaments rather than holding it as organized low-m waves.

## Next lever (to convert broadband → coherent-dominant — own cycle)
The design named the mitigation: inject into the **nudge target** `q_target`
(`omega_force.comp` SUBPASS 0, before the `q += (q_target − q)/τ` relax) rather than
**additively after** the nudge. Biasing the target makes the baroclinic pattern a
persistent *attractor* the flow is continuously pulled toward (coherent, organized),
instead of an additive kick the solver immediately cascades to broadband. This is a
new kernel mechanism and warrants its own brainstorm → spec → adversarial review →
plan cycle (it changes the relax step; the current additive hook stays the
byte-identical default). Secondary lever: reduce v1.6's own FBM `vort_inject` so the
baroclinic source dominates the band's eddy budget.

## What shipped (this sub-project)
- Leak-free opt-in external-vorticity-source hook (byte-identical off).
- `baroclinic_source.py`: coherent geostrophic source from interface thickness +
  wavenumber coherence gate (rejects the C-grid checkerboard).
- `baroclinic_driver.py` + `baroclinic_coupling.py`: evolving source driver
  (warm-start, outcrop-safe) + interleave controller with measured cadence.
- `m3_metrics.py`: honest render-gate metrics (hero coherent fraction + texture +
  broadband diagnostic) and the `sw_m3_couple.py` gate/panel tool.
- All unit + GPU tests green; full diff reviewed (spec + quality) per task.

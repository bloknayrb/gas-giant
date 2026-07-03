# Gas Giant Studio — Comprehensive Review (Architecture + Functionality / Jupiter Fidelity)

**Date:** 2026-07-02
**Scope:** whole repo at HEAD `df366a7` (master) plus the unmerged PR #13 worktree (`f427a22`)
**Method:** two parallel review tracks — architecture (findings A1–A4) and usability/workflow (B1–B5) — plus a 45-row phenomenon coverage audit judged against NASA reference imagery by paired independent judges with tiebreaks. Every finding keeps its source id (A1-1, B5-2, F12, …) and traces to the JSON evidence bundle listed in Appendix A.

**Severity rubric** (used throughout): **Critical** = crash / data loss / user blocked / change-safety hazard; **High** = task completion blocked or major maintainability debt; **Medium** = friction with a workaround; **Low** = polish.

**Contents**

1. [Executive summary](#1-executive-summary)
2. [Architecture verdict and evidence](#2-architecture-verdict-and-evidence) (A1 dead code · A2 modularity · A3 agent legibility · A4 CI/hygiene)
3. [Usability verdict and evidence](#3-usability-verdict-and-evidence) (B1 first run · B2 tooltips · B3 docs/branch split · B4 workflow · B5 scenarios)
4. [Phenomenon coverage matrix and judgments](#4-phenomenon-coverage-matrix-and-judgments)
5. [Remediation roadmap](#5-remediation-roadmap)
6. [Appendix A — methodology and evidence inventory](#appendix-a--methodology-and-evidence-inventory)
7. [Appendix B — proposed CLAUDE.md](#appendix-b--proposed-claudemd)

---

## 1. Executive summary

**Architecture verdict:** fundamentally healthy. The layering is real and machine-enforced (import-linter in CI), there is a single small facade (`engine/facade.py::Simulation`, 373 lines) that GUI, CLI, export, and tests all consume, the `pfield` metadata system is a single source of truth that auto-generates GUI, invalidation tiers, randomization, and save/load schema, and the documentation culture (milestone verdict docs, inline falsification records, self-declared throwaway scripts) is well above typical. The debt it carries is removable rather than structural: ~6.3K lines of production-dead shallow-water milestone scaffolding with ~118 CI tests still guarding it, doc drift in `architecture.md`/`sliders.md`, and one genuinely dangerous configuration slip — CI never runs on pushes to master.

**Functionality verdict:** deep and largely capable. Of 45 audited phenomena, 18 are fully convincing (most against reference imagery), 22 are partial — and most partials are calibration-level (texture/contrast tuning, not missing levers) — 3 are code-audit-only, 1 was excluded, and exactly **one is a hard no** (Oval BA's two color states, blocked by hardwired per-storm tints). The high-leverage gaps are few and specific: two deterministic chirality defects (festoon rooting edge, GRS wake flank), the per-storm color-state gap, and a validator that makes a faded/whitened belt inexpressible.

### Merged Top-10 priority queue

Rank order is final (from the synthesis pass); severity/effort tags are the merged assessments.

| # | Finding | Severity | Effort | Why it's here |
|---|---------|----------|--------|---------------|
| 1 | CI never runs on master pushes — workflow push trigger targets `main` but the default branch is `master` (A4-1) | Critical | quick-win | Every "CI green" assumption since the branch rename has been PR-runs-only; one-line fix. |
| 2 | Merge PR #13 — the entire UX overhaul (help, search, playback, undo depth, export modal) exists but no user gets it (B3-2) | High | quick-win | 17 divergence rows (15 capabilities) differ between master and the branch; gates most UX remediation. |
| 3 | First-launch experience: measured 15.9 min (RTX 3070) of unexplained development before the default preset looks right (B1-1) | High | quick-win | Progress lives unlabeled in the 10%-height Performance pane; a new user cannot tell working from broken. |
| 4 | Deterministic placement chirality defects: festoons root on the wrong belt edge and the GRS wake reads on the wrong flank (F12, F06) | High | quick-win → project | Same class as the PR #9 antipode bug; confirmed (F12 by both judges; F06 by judge + code-geometry check); sign-blind nearest-edge selection in `sim/profiles.py`. |
| 5 | Per-storm color states are hardwired — blocks Oval BA reddening (the audit's only hard NO), Neptune GDS, and a paler modern GRS (A01, B5-1, A06, B5-5) | High | project | One lever design (per-kind/per-hero tint+brightness override, incl. dark values) unlocks three failed scenarios. |
| 6 | Dead solver scaffolding: ~6.3K lines + ~118 CI tests for solvers no production path reaches (A1-1..A1-4) | High (maintainability) | project | Evidence table is complete and evidence-table-backed (see §2.2); `shallow_water_ref` is half-live and must be pruned, not deleted. Gates god-file work. |
| 7 | Agent-legibility package: commit the drafted CLAUDE.md, document the no-GPU verification path, route agents to the falsified-approaches record (A3-1, A3-2, A3-8) | High | quick-win | Unit subset measured 7.4 min; no fast tier exists; draft CLAUDE.md is in Appendix B, ready to commit after review. |
| 8 | Faded/whitened belt is inexpressible: BandTemplate alternation validator hard-caps belt values (B5-2) | Medium | quick-win | Blocks the faded-SEB epoch scenario; also unlocks A11 ochre-EZ epoch recipes. |
| 9 | Docs-truth package: architecture.md drift, sliders.md missing 16 post-2026-06-25 params, 6–8 manual sync points for lever authors (A3-3..A3-7, B3-1, A2-1, B3-6) | Medium | project | PR #11 demonstrably dropped 2 sync points in the very PR that added the lever. |
| 10 | Fidelity-gate drift and WIP hygiene: ship config FAILS its own frozen swirl gate (m3 0.51 vs ≥0.57) and the chromo saturation bound (+28% vs ≤+12%); uncommitted WIP on master fails 6 GPU byte-identity gates; gates are manual ritual, not CI (A3-6, A4-2, A4-3) | Medium | project | Gate FAILs may be accepted drift from the source-fidelity pass — needs a human re-baseline decision first, then automation. |

**How to use this report.** Section 5 is the actionable artifact: work the Top-10 in order (items 1, 2, and 7 unblock or amplify the rest), then pull second-tier items opportunistically by theme. Sections 2–4 are the evidence — each finding id there backs a roadmap entry, and quotes are verbatim from the refuters and judges. Two findings were adversarially demoted (B2-1 High→Low, A3-2 High→Medium) and are labeled inline where they appear. Item 10 is the only entry that requires a decision from you before any work starts.

---

## 2. Architecture verdict and evidence

### 2.1 Strengths (real findings, stated plainly)

From the dead-code audit (A1):

- **Exemplary GPU-vs-CPU ground-truth discipline.** Every compute kernel header — live or dead — names the exact CPU function it ports (e.g. `kernels/sw_helmholtz_apply.comp:7` "Ports helmholtz_apply() from shallow_water_ref.py EXACTLY (radius a)"; `kernels/laplacian.glsl:10` locks the sign convention to `vorticity_ref.py`). This made the live-vs-dead audit mechanically verifiable.
- **Milestone verdict documents** (`docs/superpowers/specs/m0-…m3-verdict.md`) record graduation lineage and falsification decisions, so the dead-code family can be deleted without losing the reasoning — git history plus these docs fully replace the code as a record.
- **Strict namespacing quarantined the experiments.** `sw_`/`swp_` prefixes throughout; `sw_spike/__init__.py:1` self-declares "M0 throwaway… Graduates to shallow_water_ref in M1"; the production solver, CLI, export, and app import none of it. The layering held.
- **Production checkpointing is fully decoupled** from SwGpuSolver's private checkpoint format — removing the family carries zero save-file compatibility risk.
- **The one shallow-water thing that shipped, shipped through the tested path.** The baroclinic coupling (`engine/facade.py:127-142`, `engine/baroclinic_coupling.py`) reuses the validated CPU reference at a small source grid with explicit graceful degrade — not the falsified render path.
- **`vorticity_ref.py` shows the reference-module pattern working long-term:** 302 lines, production-imported, kernel-parity-cited, still actively maintained (last commit 2026-06-26).

From the modularity audit (A2):

- **Metadata-driven extension is the standout modularity win.** A single `pfield` in `params/model.py` automatically yields the GUI widget, the invalidation tier, randomization, and save/load schema. Of the `polar_filaments` sync points, everything metadata-enforced shipped correctly; only the two unenforced points (docs, dedicated test) were dropped.
- **`engine/facade.py` is a genuinely small single facade** and import-linter contracts keep the layering honest in CI, not just in docs.
- **The shader composition system punches above its weight:** SourceMap-based error mapping reports GLSL compile errors against the original included file/line; programs are cached by (package, name, defines); cross-package includes let render kernels reuse sim noise without duplication.
- **Default-off byte-identity is enforced by construction** for detail levers: the non-DETAIL_FX variant strips new-feature code at the GLSL preprocessor level, so neutral defaults cannot regress pixel output even if the new code is buggy — stronger than tolerance tests.
- **The codebase is self-aware about its silent-failure modes:** the B1 tripwire (`solver.py:353-368`) asserting baroclinic uniforms exist at build, the exact-float-compare gate contract comment, and `panels.py`'s `leaf_kind` pure function exist specifically so failures are loud or statically testable.
- **`params/model.py`'s long field descriptions are load-bearing documentation** — several record falsified-approach history (e.g. `vort_eddy_drag` explains why `vort_psi_drag` supersedes it), which is exactly where such notes survive.

From the agent-legibility audit (A3):

- **`architecture.md` is unusually accurate at the code level** for a doc of its depth — spot-checks of the solver API, sampler-unit contract, advance registry, TILE size, and variant caching all verified (the drift items below are the exceptions, not the rule).
- **The conftest `gpu` fixture is well designed for agents:** session-scoped headless context with a clean `pytest.skip` when GL 4.3 is unavailable — the no-GPU story needs documentation, not code changes.
- **Dual hash gates** (kinematic source-SHA1 pin + `p05_baseline_hash.py` float32 render-hash gate) give unusually strong, cheap regression tripwires, and the vorticity tolerance-floor policy is consistently documented in test docstrings with named constants.
- **CI mirrors the local commands one-to-one** (ruff, lint-imports, `pytest -m "not gpu"`, `pytest -m gpu` under llvmpipe), so agent instructions need no CI-vs-local translation.

From the repo-hygiene audit (A4):

- **CI quality is well above typical for a GPU project:** the GLSL 430 compute kernels are actually exercised in CI on Mesa llvmpipe software GL — not mocked.
- **`.gitignore` is thoughtful and project-aware** (out/, NASA refs/ with the fetch script named, session state, tooling caches) — the untracked-file noise is a discipline lapse, not missing infrastructure.
- **The throwaway scripts are exemplary self-documenting scratch:** nearly all open with the hypothesis under test and an explicit THROWAWAY marker, which made triage mechanical.
- **The uncommitted GPU test (`test_wake_entrance_ramps_smoothly`) is a model regression test:** docstring states the artifact, sampling strategy, and expected numeric signatures with and without the fix.

### 2.2 A1 — Dead solver scaffolding (live-vs-dead audit)

**A1-1 (Medium; merged into Top-10 #6 as High-maintainability).** The entire `sw_*`/`swp_*` GPU shallow-water family is production-dead; only `shallow_water_ref` (a subset) and `vorticity_ref` are live. Grep of `src/` shows **zero** imports of `sw_gpu`, `sw_gpu_probe`, `sw_spike`, or `sw_encode` from `engine/facade.py`, `sim/solver.py`, `cli/`, `export/`, or `app/`. The only production imports of the family: `engine/facade.py:133` + `sim/baroclinic_driver.py:15` (shallow_water_ref) and `sim/profiles.py:20` (vorticity_ref). CI runs every covering test (testpaths = tests/unit + tests/gpu, llvmpipe GPU step).

Full per-file evidence table (columns: imported-by; covering tests + in CI?; docs/CI refs; last commit; verdict):

| File (lines) | Imported by | Tests (in CI?) | Docs/CI refs | Last commit | Verdict |
|---|---|---|---|---|---|
| `sim/sw_gpu.py` (2022) | tests only (`test_sw_gpu`, `test_m2_gpu`, `test_m3_gpu`, `test_dual_path`); tracked scripts `sw_m1_williamson.py:31`, `sw_m2_validation.py:26`, `sw_m3_rendergate.py:59`. No src/ consumer. | ~53 tests (24+15+12+2), all in CI (gpu → llvmpipe step) | m1/m2 verdicts, m3 design doc; not named in ci.yml | 2026-06-16 `df983ae` | **DEAD-EXPERIMENTAL** (M1–M3 scaffold; M3 direct render falsified per `m3-verdict.md`) |
| `sim/sw_gpu_probe/solver.py` (1057) + `__init__.py` | `test_sw_gpu_probe.py`; scripts `swp_killgate.py:41`, `swp_spinup.py:39` | 14 tests, in CI (GPU) | `m0p5-verdict.md` | 2026-06-16 `df983ae` | **DEAD-EXPERIMENTAL** (M0.5 probe; validated swp_* kernels against sw_spike CPU — a dead solver tested against another dead solver) |
| `sim/sw_spike/` (grid 58, operators 145, solver 176, init 52, encode 31, `__init__` 1) | `test_sw_spike.py`, `test_sw_gpu_probe.py`, `test_m3_ref.py:6,37-38` (two parity cross-checks); 5 sw_spike/swp scripts | 21 tests + probe/m3_ref usage, all in CI | m0 plan + verdict | 2026-06-16 `df983ae` | **DEAD-EXPERIMENTAL** — self-declared: `__init__.py:1` = "M0 throwaway shallow-water spike (CPU). Graduates to shallow_water_ref in M1." |
| `sim/shallow_water_ref.py` (2254) | **PRODUCTION:** `sim/baroclinic_driver.py:15` (`baroclinic_test_state`, `step_2layer`, `eddy_interface_var`, `PositivityViolation`), `engine/facade.py:133`; consumed via `engine/baroclinic_coupling.py:14` — the shipped opt-in M3 coupling. Also: sw_gpu.py, sw_encode.py, 9 tracked scripts, tests/spikes | `test_shallow_water_ref` (13), `test_sw_conservation` (4), `test_m3_ref` (12), `test_m2_adv_ref` (13, covers FALSIFIED M2-adv), `test_m3_baroclinic`, `test_baroclinic_driver` (8), `test_baroclinic_source`, plus oracle role in the sw_gpu tests — all in CI | named "gold-standard CPU ground truth" in every milestone verdict | 2026-06-22 `56c629d` | **LIVE (2-layer baroclinic subset) + REFERENCE-GROUND-TRUTH** (Helmholtz semi-implicit machinery, Williamson-2, SLSI/PPM advection, single-layer stepping exist ONLY as sw_gpu test oracle or for falsified M2-adv) |
| `sim/vorticity_ref.py` (302) | **PRODUCTION:** `sim/profiles.py:20,128` (`jet_vorticity`) → profiles imported by facade, solver, vortices | `test_vorticity_ref` (19), `test_vorticity_p3a_unit`, `tests/gpu/test_eddy_drag.py:117` — all in CI | cited as parity source in live kernels (`laplacian.glsl`, `vortex_omega.glsl`, `zonal_mean.comp`) | 2026-06-26 `9d8aea5` | **LIVE — KEEP** (reference ground truth for the production v1.6 vorticity solver) |
| `sim/sw_encode.py` (64) | `test_sw_encode.py` (5, CI); scripts `sw_m3_rendergate.py:58`, `sw_m3_render_compare.py:35`. No src/ consumer despite its docstring claiming "production top-of-atmosphere encoder" | 5 tests, in CI | — | 2026-06-16 `df983ae` | **DEAD-EXPERIMENTAL** (encoder for the falsified M3 direct-render path) |
| kernels `sw_*` (19 files: bernoulli/continuity/continuity_conservative/divergence/forcing/grad/helmholtz_apply/helmholtz_residual/helmholtz_rhs/helmholtz_sor/momentum/montgomery/si_predictor/velocity_backsub/vorticity.comp + 4 glsl includes) | loaded EXCLUSIVELY by `sw_gpu.py:147-683` | covered only through sw_gpu tests | — | 2026-06-15..16 | **DEAD with sw_gpu** |
| kernels `swp_*` (7 files) | loaded EXCLUSIVELY by `sw_gpu_probe/solver.py:117-645` | via probe tests | — | 2026-06-15 | **DEAD with probe** |

Checkpoint compatibility: `SwGpuSolver` has its own private save/load checkpoint (exercised only in `test_m2_gpu.py:420,471`, `test_m3_gpu.py:332`); production `engine/checkpoint.py` never touches it — **deleting the family breaks no user save files**.

Recommendation (verbatim intent): **KEEP** `vorticity_ref.py`, `shallow_water_ref.py` (live via shipped coupling — see A1-2), `baroclinic_driver/source`. **ARCHIVE** (git mv to an archive/spikes package outside `src/gasgiant`, or delete — git history preserves): `sw_gpu.py` + the 19 `kernels/sw_*` files + `tests/unit/{test_sw_gpu,test_m2_gpu,test_m3_gpu,test_dual_path}.py` + the `sw_m*` scripts. **DELETE outright** (doubly superseded per own docstrings): `sw_gpu_probe/`, `sw_spike/`, `kernels/swp_*`, `sw_encode.py`, their tests, and the swp/spike scripts. Removes ~118 CI tests (~65 gpu-marked llvmpipe tests); edit `test_m3_ref.py` to drop its two sw_spike parity cross-checks; update the stale sw_gpu mention in `tests/unit/test_gl_context.py:5`. No import-linter change needed.

**A1-2 (Medium, interpretive).** Trap resolved: `shallow_water_ref.py` is NOT merely sw_gpu's test oracle — its 2-layer subset is the production baroclinic source — but roughly half its surface (Helmholtz SI solver, Williamson-2 states, the falsified SLSI/PPM advection family) becomes orphaned the moment `sw_gpu.py` is archived. Keep the file when archiving sw_gpu (zero behavior risk); prune the orphaned sections as a deliberate roadmap decision, not housekeeping — M2-AE (polar) is still a candidate milestone and the verdicts name this file the gold-standard ground truth. Do not re-attempt semi-Lagrangian (falsified 2026-06-16).

**A1-3 (Medium, interpretive).** CI executes ~65 GPU-diff tests (plus ~53 CPU tests) for solvers no production path reaches, on every push/PR under software-rendered llvmpipe — the heaviest tests in the suite (Helmholtz SOR sweeps, Williamson-2 spin-ups, 2-layer checkpoint round-trips) protect code with zero production consumers. Pure CI latency and a maintenance tax: any refactor of `gl/` or kernel common includes must keep 26 dead kernel files compiling. Move the test files with the code when archiving; keep `test_vorticity_ref.py`, `test_baroclinic_*.py`, `test_m3_baroclinic.py`, `tests/gpu/*` (those guard live code).

**A1-4 (Low).** Misleading "production" labels on dead modules would trap future maintainers: `sw_encode.py:3-4` claims it "Promotes … to the production α-aware 2-layer state" and its test file says "production top-of-atmosphere encoder", yet no src/ module imports it; the 2026-06-15 design doc promises `solver.type = "shallow_water"` but no such SolverType is reachable in production. Deleting per A1-1 resolves this automatically; keep the milestone verdict docs as the historical record either way.

### 2.3 A2 — Modularity, extension seams, and failure-mode consistency

**A2-1 (Medium).** Phenomenon-lever authors face **6–8 manual sync points**, and PR #11's `polar_filaments` demonstrably dropped 2 of them in the very PR that added the lever. The sync points: (1) `params/model.py` pfield; (2) `detail.comp` uniform decl + effect block; (3) `render/detail.py` hand-maintained `fx_on` predicate (now 8 ORed params); (4) the uniform `_set` call; (5) both preset build scripts; (6) regenerated preset JSONs; (7) a dedicated behavior test — grep `polar_filaments` in tests/ returns **zero** hits while sibling levers have tests; (8) a `docs/sliders.md` entry — absent. Everything metadata-enforced shipped correctly; the two unenforced points were silently skipped. The two riskiest failures are silent: a forgotten `fx_on` entry means the lever is inert with no error; a forgotten `_set` is swallowed by the KeyError-suppressing wrapper (`render/detail.py:23-27`). Recommendation: derive `fx_on` from pfield metadata (an `fx: true` flag) and add a cross-referencing test so the doc/test sync points fail loud; backfill the missing sliders.md entry and detail-fx test.

**A2-2 (Medium, interpretive).** Baroclinic graceful degrade is invisible to the GUI user: `facade.py:122-125,137-141` degrade paths emit `log.warning` only. The GUI has a Toasts system used at 8+ call sites, but grep `baro` in `src/gasgiant/app/` returns zero hits. A user who enables `solver.baroclinic` (RESTART-tier, 8000-step CPU warmup) and hits an outcrop gets a plain-v1.6 render with zero in-window signal — contradicting the codebase's otherwise loud policy (shader errors raise, validation errors toast). Secondary: the except clause catches bare `RuntimeError` while the docstring claims "a genuine unexpected error propagates loudly". Recommendation: expose `Simulation.baroclinic_status` ('off'|'active'|'degraded' + reason), toast on transition; narrow the catch.

**A2-3 (Medium).** scipy is declared test-only (`pyproject.toml:34-36`, comment "Test-only.") but is production-reachable via baroclinic: `sim/baroclinic_source.py:21` imports `scipy.ndimage` at module top, reached from `facade.py:116-121` whenever `baroclinic.enabled=True`. The except clause catches only RuntimeError, so on a plain install the resulting ImportError propagates out of `_build()` — a hard crash, precisely the "never crash construction" scenario the docstring promises to avoid. Preferred fix: replace the two scipy call sites with cv2 equivalents (opencv-python-headless is already a main dep). Minimum: catch ImportError alongside RuntimeError and fix the "Test-only" comment.

**A2-4 (Low, interpretive).** Lazy-import audit: 3 of 6 facade lazy imports are load-bearing (the baroclinic chain — scipy cold import measured ~3.95 s and scipy is optional), 3 are habit (`ExportSnapshot`, `hero_centers` — no cycle exists; `PolarRoute` is lazily imported from a module facade already imports at top level, zero effect). `solver.py:648/653` import `math`/`warnings` inside a loop body — pure noise. Hoist the habit cases; comment the load-bearing three.

**A2-5 (Low, interpretive).** `sim/solver.py` (1052 lines) mixes kinematic and vorticity paths; ~45% (≈470 lines) is vorticity-only with a clean extraction seam (`_OmegaState` + five `_omega_*` methods + `_copy_psi_to_work` → `sim/omega.py`; single dispatch branch at `_produce_psi:804-810`). A mechanical move — no kernel/uniform/dispatch-order change, byte-identity tests stay green. Worth doing opportunistically the next time the vorticity path is touched, not as standalone churn.

**A2-6 (Low, interpretive).** The silent-uniform-no-op hazard is pervasive (KeyError-suppressing `_set` in 5 modules) but the loud tripwire pattern the codebase itself invented (`solver.py:353-368`) is applied exactly once. The guarded-set idiom is genuinely required (GLSL prunes unused uniforms; variants legitimately lack some), so the fix is generalizing the compensating control: assert at build time that each DETAIL_FX param's uniform exists in the compiled fx variant, driven by the same metadata as A2-1.

**A2-7 (Low, interpretive).** `detail.comp` (506 lines) is at the edge of navigable: a 310-line `main()` of nine independently gated effect blocks. Verdict: the include graph and variant system are healthy (includes one level deep in production kernels; SourceMap error mapping works), and `detail.comp` does not urgently need decomposition — but it is one or two levers away from it, and the cut is mechanical (move the DETAIL_FX region into `detail_fx.glsl`). Separately, the include flattener (`gl/context.py:188-216`) lacks include-guards and cycle detection — a circular include would loop forever. Two-line fix: a seen-set.

**A2-8 (Low, interpretive — a deliberate non-finding).** `params/model.py` (1007 lines) should **not** be split: it is a cohesive declarative schema whose length is documentation (~60% is field description strings that double as GUI tooltips and docs source), three subsystems walk it generically, and no consumer imports a subset. The 1007-line count is a false positive for "god file". Revisit only past ~1500–2000 lines or if it gains behavior beyond validators.

### 2.4 A3 — Agent legibility and documentation truth

**A3-1 (High).** No documented agent-without-GPU verification path; the "unit" subset is not fast (**measured ~7.4 min**: `pytest -m "not gpu" -q` = 362 passed in 441 s; the GPU subset = 173 tests in 434 s on RTX 3070) and nothing documents the `-m` markers, skip behavior, or llvmpipe recipe (README says only `uv run pytest # unit + GPU tests (llvmpipe works)`). The mechanism is good — GPU tests auto-skip cleanly without GL 4.3, CI proves llvmpipe works — but none of it is written anywhere an agent would look, and heavy CPU shallow-water reference tests inside tests/unit mean there is no sub-minute smoke signal. Fix: ship the CLAUDE.md (Appendix B); consider a `slow` marker so `-m "not gpu and not slow"` becomes a genuine fast loop.

**A3-2 (originally High; refuter verdict PLAUSIBLE, adjusted **Medium**).** Falsified/do-not-reattempt knowledge is not discoverable from the repo root: README links six docs but never `docs/roadmap.md` (which holds the emergent-color FALSIFIED verdict and the Rhines research-grade gate); no doc links into `docs/superpowers/` where the M-milestone falsifications live; `roadmap.md:161` still references the dropped jupiter_baroclinic preset as existing. The refuter confirmed the discoverability gap but demoted severity: the jupiter_baroclinic drop IS recorded in-repo (`scripts/build_vorticity_presets.py:15-18,256` verbatim), the polar-discreteness gap is in `docs/realism.md:181-182` (only the "blocked on merger physics" rationale is memory-only), `grep -i falsified` surfaces the graveyard instantly, and the impact is wasted exploratory effort, not shipped defects. Fix: the CLAUDE.md "Falsified — do not re-attempt" section plus appending the two under-recorded items to roadmap.md.

**A3-3 (Medium).** `architecture.md:230-235` claims unqualified byte-determinism ("same seed → identical exports… (tested)") — stale. The shipped v1.6 vorticity SOR is NOT byte-deterministic (~1e-3 cross-instance / ~0.004 cross-session LSB noise), documented only in test docstrings (`_VORT_SOR_ATOL = 1e-3`, `GPU_NOISE_ATOL = 1e-2`). An agent reading only the doc would write a byte-exact assertion for a vorticity feature and get a flaky test. Amend the Determinism section: byte-exact holds for the kinematic path only.

**A3-4 (Medium).** `architecture.md` omits the shipped baroclinic coupling subsystem entirely — zero mentions of "baroclinic" despite three source modules, a params block in all five factory presets, and a validator. Its design record exists only in an unlinked superpowers spec.

**A3-5 (Medium).** Factual drift: `architecture.md:207` states `generation_version` = 3; `engine/checkpoint.py:36` says 5. Drop the literal or reference the constant by path.

**A3-6 (Medium, interpretive).** The current working tree fails 6 GPU byte-identity/no-op gates (`6 failed, 167 passed` on `pytest -m gpu`), consistent with the uncommitted in-flight work rather than a repo defect — but it demonstrates the ergonomics gap: nothing documents "establish a green/red baseline before verifying a change". Severity is about the missing guidance, not the red tests.

**A3-7 (Low).** `architecture.md:187-188` names two DETAIL_FX triggers; code has eight (`render/detail.py:76-83`). Change the parenthetical to "any DetailParams FX lever > 0" so it can't drift again.

**A3-8 (Low, interpretive).** `pfield()` and the shader include/variant conventions are discoverable only via code comments; no doc names them. Near-miss rather than a hole (the model.py docstring is complete) — covered by the CLAUDE.md Conventions section, which also captures the memory-only GLSL gotchas (`patch` reserved; uniforms before includes).

### 2.5 A4 — CI, git hygiene, and repo state

**A4-1 (High per-finding; Critical in the merged queue as a change-safety hazard).** `.github/workflows/ci.yml:3-6` — `on: push: branches: [main]` — but the default branch is `master` and no `main` branch exists anywhere. The push trigger is dead: the only CI that ever runs is the `pull_request` event. Post-merge commits on master (including squash merges and direct pushes — which demonstrably happen, see A4-3) get zero CI. One-line fix: `branches: [master]`. **Refuter status: fact-checked directly against ci.yml and `git branch -a` (non-interpretive; CONFIRMED).**

**A4-2 (Medium, interpretive).** The visual-fidelity gates the project's workflow revolves around (`swirl_gate.py --preset` ship-config mode, `compare_reference.py`, the three calibrated `_ffr/_chromo/_plume` compare scripts) run only when someone remembers — no workflow invokes them. CI does run both pytest tiers and import-linter. `swirl_gate`'s metric gates don't need the gitignored NASA refs, so at least those could run under llvmpipe on a schedule or `workflow_dispatch`.

**A4-3 (Medium, interpretive).** Work-in-progress sits uncommitted directly on master: a coherent, shippable tracer-channel preview feature (`view_transform.frag` +11/−2 `u_channel` 0–4; `viewport.py` +24 with `_CHANNEL_MAP` and AgX bypass; `facade.py` +4 `preview_tracers_texture`) plus a well-documented GPU regression test (`test_hero_wake.py` +45). Leaving it loose risks merge pain with PR #13's viewport changes and accidental inclusion in unrelated commits. Move to a feature branch, commit in two pieces, reconcile with the PR #13 worktree.

**A4-4 (Medium, interpretive).** 43 untracked `scripts/` files. Triage: **keep-and-track 4** (`_ffr_compare.py`, `_chromo_compare.py`, `_plume_compare.py` — the calibrated per-feature gates — plus `_ffr_eval.py`, the only polar-orthographic reprojection harness the FFR comparator depends on); **delete 36** self-marked THROWAWAY one-offs (conclusions already captured in memory/docs); **decide 3 borderline** (`palette_identities.py`, `vibe_gallery.py`, `seam_globe.py` — genuinely reusable inspection utilities). Caution: untracked files are unrecoverable from git once deleted — archive to a gitignored scratch dir if uncertain. Do NOT solve via `.gitignore` patterns on source-looking `.py` files. Adopt the convention: throwaway diagnostics go in a gitignored `scripts/scratch/` from day one.

**A4-5 (Low).** `_diag/` — 15 MB of regenerable PNG review artifacts at repo root, untracked and not gitignored. Add `_diag/` to `.gitignore` or point the scripts at the already-ignored `out/`.

**A4-6 (Low).** `.claude/tdd-guard/data/test.json` is a tracked machine-written state cache — perpetually dirty after every test run. `git rm --cached` it and ignore `.claude/tdd-guard/`.

**A4-7 (Low, interpretive).** Six stale local branches: three provably merged (ancestors of master: `feat/localized-eddy-injection`, `fix/baroclinic-shrink-eddy-scale`, `v1.6-vorticity`) — delete now, plus the one merged remote; three squash-merge-superseded (`feat/grs-realism` [gone], `feat/modernize-presets` [ahead 5/behind 5 — verify its 5 local-only commits landed in a master squash before deleting], `feat/storm-locality-deformation-radius`) — verify-then-delete. Keep `worktree-ui-ux-roadmap` (= the live PR #13) until the PR is resolved.

---

## 3. Usability verdict and evidence

Note on scope: the UX track was run against the PR #13 worktree (`.claude/worktrees/ui-ux-roadmap`, head `f427a22`) because that is where the GUI's current design lives; where master differs, the divergence itself is finding B3-2.

### 3.1 Strengths

- **Two-command install with the GPU requirement stated up front**, and `docs/sliders.md` is generated from the same pydantic model as the UI, so the reference genuinely tracks reality (its staleness is regeneration lag, not drift-prone hand text — B3-1).
- **Every discrete action gives toast feedback with useful detail** (preset name on load, seed on randomize, actionable strict-envelope error summaries); export failures never toast an empty message.
- **The edit loop is honest about cost:** tier badges P/V/R with hover legends, commit-on-release for heavy tiers with a visible "release to apply" hint, invalid typed values toast on the commit frame instead of being swallowed.
- **Dead-end prevention in the panel:** the hidden-advanced hint and the `bands.template` / `hero_latitude` escape banners stop the classic "slider does nothing and I don't know why" trap.
- **Export has a real confirm step**, live sliced progress with named phases (developing / tile i/n / encoding), a Cancel that cleans up partial output, and mid-export edits are coalesced and applied afterward rather than dropped.
- **Discoverability scaffolding is above average for an imgui tool:** "? Help (F1)" first in the Controls pane, complete shortcut list, `/` focuses search, search overrides the Advanced gate, section (?) blurbs on every header. Help-vs-wired-shortcuts is an exact match, both directions.
- **Undo architecture is unusually thorough for a v1 GUI:** UndoRecord carries preset identity AND the pristine baseline; gestures coalesce; preset loads, randomize, and reroll are all undoable; edits held during export flush as exactly one undo step.
- **Robust session handling:** autosave on exit, restore toast, pre-migration backup of old-format sessions, window geometry persistence.
- **EmissionParams is the tooltip literacy gold standard** (all 11 fields pass — "hot-spot chains blaze, barges glow, belts glimmer, zones stay dark"), newer levers consistently state visual read plus the safe zero, several sliders give calibrated anchors (haze "Jupiter (0) to Saturn (~0.6) axis"), and pfield description coverage is 100% — no empty tooltip on the auto-generated path.
- **The animation groundwork is real, not aspirational:** deterministic tick, `engine/snapshot.py` and `engine/checkpoint.py` self-describe as the future animation exporter's substrate, and the tolerant manifest policy means `frames[]` can ship without breaking old Blender importers.
- **Scenario-level strengths (B5):** Saturn's polar hexagon is a first-class working lever (saturn_pale ships it); `hero_latitude` pinning with the radius-coupled validator is exactly the right kind of goal-directed control; the storm_tints LUT is expressive enough that a Neptune Great DARK Spot is achievable today with zero code changes; the strict versioned preset envelope meant all four scenario recipes either validated completely or failed loudly.

### 3.2 B1 — First-run journey

The journey was traced end to end against the PR #13 worktree: (1) **Install** — README's two commands; risk points: plain `uv sync` → ImportError traceback, GL < 4.3 → traceback. (2) **First window** — docked Controls/Equirect/Sphere/Performance layout; a 4-second toast "started from gas_giant_warm"; then the dev run of 1256 steps on a 4096 grid at 2 steps/frame with a bare "N/1256" bar as the only indicator (the "recomputing…" spinner is never set at first launch; the "paused N/M" hint appears only when paused). (3) **Understand** — F1 help covers panel navigation, tiers, and shortcuts, but not the dev-run/playback model. (4) **Change** — POST edits commit live; heavy tiers commit on release with a jargon hint; RESTART edits re-run all 1256 steps with the cost stated only in a badge tooltip. (5) **Presets** — save dialog defaults to `~/.gasgiant/presets`; no overwrite-active save, no in-app delete; session restore degrades identity to "unsaved". (6) **Export** — a good confirm modal and sliced progress, but silent overwrite and a 4-second success toast. Findings below follow that path.

**B1-1 (High; Top-10 #3).** The first-launch multi-minute development run is never explained on screen. The startup preset (gas_giant_warm) develops 1256 steps on a 4096 grid at a default of 2 steps/frame; **measured cold-start time to first developed planet: 952 s ≈ 15.9 min on an RTX 3070**. During that entire period the ONLY indicator is an unlabeled progress bar reading "142/1256" inside a dock pane titled "Performance", sized to 10% of the left column. No text says "developing", that the visible image is not final, how long it takes, or that the Speed combo's "Max" (256 steps/frame) would finish far faster. The 4-second onboarding toast names the preset but sets no expectation. A new user watching a still-churning half-formed planet has no way to know whether the app is working, broken, or done. Fix: a labeled "Developing N/M" indicator in/over the viewport or Controls header, a Speed hint or faster default, and consider renaming the "Performance" pane since it hosts the transport controls.

**B1-2 (Medium).** Launch failures surface as raw tracebacks: plain `uv sync` (no gui extra) still installs the `gasgiant-studio` entry point, which dies with a bare ImportError for imgui_bundle; a GPU below OpenGL 4.3 propagates an unhandled exception from `init_gl`. Both are the first minute of the journey. Wrap the import; catch the GL-attach failure and name the 4.3 requirement.

**B1-3 (Medium).** Help (F1) never explains the development-run / playback model — the app's central concept (a dev run that replays after every RESTART edit, controlled from the Performance pane) is entirely absent from the designated learning surface. Also unexplained in-app: the `*` dirty marker and the "unsaved" label after session restore.

**B1-4 (Medium).** Export modal omits what will be written (`color.png`/`height.exr`/`mapset.json` never named); the exporter silently overwrites an existing map set (`out_dir.mkdir exist_ok=True`, unconditional writes); success feedback is a 4-second toast with no persistent "last exported to…" line, no open-folder affordance, no Blender next-step pointer.

**B1-5 (Low, interpretive).** Edit-loop status messages use tier jargon ("release to apply (restart/velocity)") and the export-hold notice draws in a different pane than the frozen sliders.

**B1-6 (Low, interpretive).** README sets no expectations for the GUI first-run and its preset list is stale (pre-confirmed; the fix exists in PR #13 — see B3-3).

**B1-7 (Low).** Enabling aurora gives zero preview feedback (pre-confirmed LIMIT — alpha channel, never composited in the RGB preview) and no in-UI disclosure; the emission section blurb even advertises aurora. See B4-3 for the fuller treatment.

**B1-8 (Low, interpretive).** Preset lifecycle friction: Ctrl+S never overwrites the active preset (always a dialog); session restore degrades identity to "unsaved"; user presets can't be deleted/renamed in-app.

### 3.3 B2 — Control literacy (tooltips)

**B2-1 (originally High; refuter verdict CONFIRMED-with-demotion, adjusted **Low**).** The factual core survived adversarial checking: tooltips are the pydantic descriptions verbatim, and the v1.6 solver-core block (`solver.type`, `poisson_iters`, `sor_omega`, `vort_relax_tau`, `vort_hypervisc`, `vort_inject_scale`, the baroclinic cadence trio) states pure mechanism with zero visual consequence, while EmissionParams and every post-v1.5 lever consistently lead with what appears on the picture — the "house style vs untouched v1.6 block" pattern is a correct diagnosis. The refuter demoted severity three independent ways: every allegedly failing field is Advanced-gated (the Basic-mode fail rate is ~5–10% by the reviewer's own sample); five of the claimed 9–11 fails are numerical plumbing where a jargon tooltip is arguably correct behavior (the baroclinic cadence fields are deliberately frozen — vivid tooltips would invite fiddling); and the solver knobs that actually ARE jupiter_vorticity's tuning surface (`vort_inject`, `vort_psi_drag`, `deformation_radius`, `baroclinic.gain`) already carry visual reads. Net: a real, cheap copy-editing gap on ~3–5 hidden Advanced fields plus front-loading two long descriptions. **Demoted High → Low; the systemic rewrite stays recommended as polish.**

**B2-2 (Medium, interpretive).** Worst-10 tooltip exemplar table (field → current → why it fails → better one-liner):

| # | Field (model.py line) | Current description | Why it fails | Suggested rewrite |
|---|---|---|---|---|
| 1 | `solver.type` (L719) | "Streamfunction solver: kinematic (analytic, v1.5) or vorticity (prognostic fluid, v1.6+)" | Names internal versions; says nothing about the look — and it's the single most consequential switch in the app | "How clouds move: kinematic = fast, painterly, bands stay put; vorticity = real fluid sim, storms interact and shed filaments (slower; needed for the hero/oval solid-core levers)" |
| 2 | `solver.poisson_iters` (L722) | "Fixed red-black SOR iterations per step (vorticity mode)" | Pure numerics, no direction cue | "Solver accuracy per step: too low leaves smeared/laggy swirls; higher is slower with diminishing returns (vorticity mode)" |
| 3 | `solver.sor_omega` (L724) | "SOR over-relaxation factor, must be in (1,2) exclusive (vorticity mode)" | Only a validity constraint; no effect stated | "Solver convergence speed; leave at 1.7 — it changes render time, not the picture, unless set so low the swirls lag" |
| 4 | `solver.vort_relax_tau` (L742) | "Vorticity nudging timescale toward jets+vortices (vorticity mode)" | "Nudging timescale" with no low-vs-high consequence | "How tightly the flow is leashed to the painted jets and storms: low = tidy and band-locked, high = free-running turbulence that can wander off the template" |
| 5 | `solver.vort_hypervisc` (L745) | "Scale-selective biharmonic hyperviscosity rate (vorticity mode)" | Three physics terms, zero picture | "Fine-scale smoothing: cleans up pixel-level crackle; too high blurs away the thinnest filaments" |
| 6 | `solver.coriolis_f0` (L747) | "Planetary vorticity magnitude f0 in f=f0*sin(lat); sets the Rhines/band scale (vorticity mode)" | Formula plus "Rhines"; direction unstated | "Planet-rotation strength: higher = more, narrower emergent bands and flatter storms; lower = fewer, fatter bands" |
| 7 | `solver.vort_inject_scale` (L754) | "Eddy-injection frequency as a multiple of bands.detail_freq (vorticity mode)" | Cross-reference to another field, no visual | "Size of the injected churn: higher = finer speckle the shear folds into thin filaments; lower = big blobs" |
| 8 | `solver.vort_drag` (L763) | "Linear (Rayleigh) drag fraction on relative vorticity per step; absorbs the 2D inverse-cascade energy that piles up at large scales (0 = off)" | "Inverse-cascade energy" is the load-bearing phrase and it is opaque | "Global brake on swirling: tames runaway planet-scale swirl but also weakens every storm; prefer vort_psi_drag, which targets only the oversized swirl" |
| 9 | `solver.baroclinic.warmup_steps` (L699; also the other two cadence fields) | "Baroclinic spin-up before coupling (fixed cadence). No rand. hi=20000 leaves headroom past the ~12500 outcrop…" | Written for the test author; "outcrop" is project-internal | "Internal pacing of the storm generator — leave at default; only affects how the extra mid-latitude storms mature" (or hide the trio; see B2-3) |
| 10 | `waves.festoon_wavenumber` (L527 — a non-Advanced section) | "Rossby wavenumber of the festoon/hot-spot train" | "Rossby wavenumber" is the exact fail archetype, in Basic | "How many festoon plumes fit around the equator (higher = more, smaller plumes)" |

Each rewrite keeps the physics term parenthesized so power users lose nothing.

**B2-3 (Low).** The three baroclinic cadence fields ship with empty `ui` group labels and render unlabeled in Advanced mode — the docstring intent ("fixed (non-UI)") and the behavior (rendered, unlabeled) disagree. Separately, the non-pfield leaves behind the custom palette widget (PaletteRow latitude, gradient stops, BandTemplate arrays) carry no descriptions and render with zero tooltips.

**B2-4 (Low, interpretive).** Unit/vocabulary inconsistency: five Basic-visible sliders specify latitude-like quantities in radians while everything else uses degrees, with no conversion hint; "GRS", "anticyclone", "cyclonic", "retrograde" appear unglossed. The Emission class docstring knows emission is preview-invisible but none of the 11 per-field tooltips carry the warning.

### 3.4 B3 — Documentation currency and the master/branch split

**B3-1 (Medium).** `docs/sliders.md` is stale: **16 params added after its 2026-06-25 generation are missing**, including all the recent hero/polar/aging levers (`hero_mottle`, `hero_tint_var`, `hero_rim_tint`, `hero_rim_warp`, `hero_wake_detail`, `hero_solid_core`, `oval_solid_core`, `hero_calm`, `hero_collar_wrap`, `zone_texture`, `polar_filaments`, `deformation_radius`, `vort_eddy_drag`, `vort_psi_drag`, `chroma_aging`, `polar_canvas_value`). Amplified by PR #13: its Help window points users at docs/sliders.md as "the full field reference". The generator auto-discovers fields, so this is regeneration lag — re-run `scripts/render_slider_examples.py` (resumable) and consider a CI check diffing the discovered field list against the doc.

**B3-2 (Medium; Top-10 #2).** Master-vs-worktree divergence: **user-visible GUI capabilities that exist only in the unmerged PR #13 worktree.** Full inventory (master = repo root, ships today; worktree = PR #13 head `f427a22`):

| # | Capability | Master | Worktree (PR #13) |
|---|-----------|--------|-------------------|
| 1 | Field search/filter box (name/label/description; overrides Advanced filter) | absent | yes (`/` shortcut) |
| 2 | Basic/Advanced split (81 `adv=True` fields hidden by default; hidden-advanced hint) | absent — all ~140 fields flat | yes (`A` toggle) |
| 3 | Undo/redo | single-slot undo of last discrete action; no redo; slider edits not undoable | multi-step history incl. committed slider edits, Ctrl+Z/Ctrl+Y |
| 4 | Preset dropdown | factory-only, no active-preset indicator | merged factory + `user/` entries, active label + dirty marker |
| 5 | User preset dir `~/.gasgiant/presets/` | concept absent | first-class (save dialog, dropdown, Ctrl+S) |
| 6 | Keyboard shortcuts (F1, /, A, R, Ctrl+Z/Y/S) | none | yes, text-input-safe, export-gated |
| 7 | Help window (F1 + button; tier legend, shortcut list) | absent | yes |
| 8 | Tier-cost badges per field + section (?) help markers | absent (tooltips only) | yes |
| 9 | Per-field right-click Lock (excluded from Randomize/Reroll) | absent | yes |
| 10 | Modified-from-default markers + per-field reset + per-section changed counts | absent | yes |
| 11 | Playback: Play/Pause, Step, Restart-dev, steps-per-frame speed | absent (sim always runs) | yes |
| 12 | `bands.template` banner + Clear; `hero_latitude` Unpin banner | absent | yes (authoring still JSON-only) |
| 13 | Sphere preview: sun azimuth/elevation sliders, double-click reset | fixed hardcoded sun | yes |
| 14 | Flat viewport channels | color, height | Color, Height (cloud-top), Emission |
| 15 | GUI startup default preset | jupiter_like | gas_giant_warm + "Reset to gas_giant_warm" button |
| 16 | Export-in-flight safety (disable edits/undo/randomize/load during export) | absent | yes |
| 17 | README factory-preset list | stale (3 of 5 listed) | corrected, marks default |

Supporting: the worktree also adds `engine/facade.py` rebuild support (+53 lines) and ~2,700 lines of new tests (shortcuts, undo/redo, panel state, preset identity, playback, session backup, commit loop). Every row ships with one action: merge (or explicitly close) PR #13.

**B3-3 (Low).** README factory-preset list stale on master (3 of 5 listed; pre-confirmed). The fix already exists in the PR #13 branch — resolve via the merge rather than a standalone patch (cherry-picking the README's "[default]" annotation without the GUI change would introduce a new inaccuracy).

**B3-4 (Low, interpretive).** `docs/presets.md` becomes silently incomplete the moment PR #13 merges: no mention of the user preset directory or merged dropdown (the branch changes no file under docs/). Add a "User presets" paragraph in or immediately after the merge.

**B3-5 (Low).** `docs/blender_addon.md` omits three import-panel options (Radius, Mesh Segments, Atmosphere "None"); the other 10 are documented with matching names/defaults.

**B3-6 (Low, interpretive).** The slider-doc generator structurally skips enum fields with no text fallback: `solver.vort_inject_mask` (a shipped GUI dropdown central to the gas_giant_warm look) has zero occurrences in sliders.md and will stay missing after regeneration. Extend the generator to emit text entries for StrEnum fields.

### 3.5 B4 — Workflow gaps (export, preview, undo)

**B4-1 (Medium, interpretive).** No animation/sequence export despite the live-evolving preview being the app's pitch: `frames[]` is reserved in the manifest schema but no writer populates it; `snapshot.py`/`checkpoint.py` self-describe as the animation exporter's stepping stones; the CLI has no frame-count or even `--dev-steps` flag. Today's only workaround is hand-editing `sim.dev_steps` per frame and re-developing from step 0 each run — hours for a short clip. All hard infrastructure (deterministic tick, snapshot, tolerant manifest) exists; the gap is the driver loop and CLI surface. Ship CLI-first: `gasgiant export --frames N --steps-per-frame K`; at minimum expose `--dev-steps` now.

**B4-2 (Medium).** The `bands.template` / `hero_latitude` escape hatches are one-way (clear/unpin only) — and the startup default preset ships with both engaged (as do jupiter_like and jupiter_vorticity). The common first-session state is "band sliders inert, hero latitude fixed", and the only GUI affordances are destructive: clearing the Cassini-calibrated band skeleton is one click; getting it back (beyond immediate Undo) requires hand-editing JSON, and the cleared value isn't preserved anywhere. Re-pinning or nudging the hero latitude is likewise JSON-only. Lowest-cost upgrades: an actual optional-float widget for hero_latitude (pin checkbox + validator-clamped slider) and a confirm (or source-preset note) on "Clear template".

**B4-3 (Medium).** Aurora is uneditable-by-eye: invisible in every in-app view (emission alpha, never composited — even in the branch's new Emission channel), with no in-UI disclosure on the five aurora sliders. Iterating aurora hue/radius/width requires a full export plus Blender import per tweak — a blind edit loop for five shipped parameters. Cheap fix: composite alpha × `aurora_color` in the Emission channel (the manifest already carries aurora_color for exactly this lift); interim, one disclosure line in the channel view and the `aurora_strength` tooltip.

**B4-4 (Medium).** `export.width` is editable in two places with different affordances AND different undo semantics: the auto-generated panel renders it as a Basic free slider (undoable), while the export modal renders it as a snapped 1K–16K combo committed outside undo history — which also silently clears the redo stack. A non-preset width set in the panel renders the modal combo blank. Give the field exactly one live editor and one undo policy.

**B4-5 (Low, interpretive).** Preset sharing trap, latent: `extra="forbid"` plus "format bumps only on breaking changes" means the first additive field shipped will make every older installation reject newer shared presets with an error indistinguishable from a typo — even though `app_version` (the disambiguating fact) is stored in every envelope and never consulted. Rated Low at v0.1.0 with a single released format; becomes Medium the day a field is added. Also: no "Import preset…" affordance (Load gives transient FILE identity only).

**B4-6 (Low).** Undo/redo scope is broad and well-engineered; the residual gaps are the modal-committed output settings (excluded from history by design AND destroying redo — "I changed PNG compression and lost my Redo" is unexplainable from the user's chair), and Help not disclosing the exclusions (export settings, Restart dev, locks) or the 64-entry cap.

### 3.6 B5 — Scenario expressiveness (art-direction stress test)

Four target-look scenarios were authored as complete preset JSONs (all validated through `PlanetParams.model_validate` and rendered — see Appendix A). The recipes are the B5 artifact; the stuck points they hit are the findings:

**B5-1 (High; Top-10 #5).** No per-hero color lever: hero tint (0.9) and brightness (0.05) are hardcoded in `vortices.py:430`, and `stamp_contrast` explicitly skips `KIND_HERO` (`vortices.py:493-497`). Both the Neptune dark-spot and paler-modern-GRS scenarios hinge on hero color, and neither has a direct lever — only resculpting the shared `storm_tints` gradient, which couples every other tinted feature (with jupiter_vorticity's stamp_contrast 2.4, barges land at LUT index 0.92, only 0.03 from the hero's 0.95, so hero paleness bleeds ~20% onto the reddest barges). Fix: `storms.hero_tint` (and optionally `hero_brightness`) replacing the constants — both goal looks become one-slider edits with zero palette coupling.

**B5-2 (High; Top-10 #8).** A fully faded/whitened belt is inexpressible: the BandTemplate alternation validator (`model.py:157-163` (worktree line numbers; master ≈135–139), values-below-median must strictly alternate) hard-caps every belt value strictly below the median. Computed: with jupiter_vorticity's other 11 template values fixed, the SEB ceiling is 0.55 vs original 0.52 — a ~6% fade, where the real event turns the belt zone-white. The workaround routes the fade through a palette row, which whitens the whole latitude ring, bleeds to neighbors, and lives on the wrong conceptual axis. Fix: per-band identity override or a first-class `bands.belt_fade` lever.

**B5-3 (Medium).** Great-White-Spot outbreaks cannot be placed or timed: candidates are restricted to the darkest half of belts at 0.20–1.0 rad (11.5–57°), belt and eruption epoch are seed-drawn. The 1990 equatorial-class GWS is unreachable (hardcoded 0.20 rad floor); recreating the 2010 event at ~35°N specifically is a seed lottery. Fix: `storms.outbreak_latitude` (nullable, validated like hero_latitude) + an `outbreak_phase` knob.

**B5-4 (Medium).** `faded_sector`'s target belt is not user-addressable: it picks the widest low/mid belt, and in the shipped Jupiter template the SEB wins by **0.01°** over the NEB — any template tweak silently moves the fade to the wrong hemisphere with no error. The scenario recipe defensively widens the SEB edge to keep the pick stable.

**B5-5 (Medium).** No lever for hero-adjacent bright companion clouds (Neptune GDS companion / Scooter class): companions attach only to oval hosts; nothing bright can be pinned next to the hero. A `storms.hero_companions` count would close the gap cheaply since both stamp mechanisms exist.

**B5-6 (Low, interpretive).** Vorticity-only storm levers silently no-op on kinematic presets: `hero_solid_core` does nothing on ice_giant, leaving the Neptune dark oval exposed to the known whirlpool-winding artifact, with no validation warning or GUI badge.

**B5-7 (Low, interpretive).** `stamp_contrast` conflates brightness and tint amplitude for all non-hero storms, migrating barges across the storm_tints LUT into the hero's color region — brightness and hue identity cannot be tuned independently. Fix: split into `stamp_contrast` (brightness) and `stamp_tint_contrast` (tint), new field defaulting to follow the old for byte-identity.

**B5-8 (Low, interpretive).** Outbreak look is mostly baked module constants (LIFETIME 300, BRIGHTNESS 1.9, TRAIN_N 6, …); `outbreak_strength` scales brightness and outflow in lockstep, so "brighter but dynamically gentler" is inexpressible.

---

## 4. Phenomenon coverage matrix and judgments

45 rows; capability verdicts are the paired-judge consensus. Summary: **18 yes, 22 partial, 1 no, 3 code-audit, 1 excluded.** Most partials are calibration-level (`candidate_poor = true` — texture/contrast tuning against the reference, not missing levers).

| Id | Phenomenon (planet) | Levers | Wired | Evidence class | Capability | Evidence quality | Chirality | Cand. poor | Notes |
|----|---------------------|--------|-------|----------------|------------|------------------|-----------|-----------|-------|
| F01 | Zones (bright bands) — Jup/Sat | 8 | yes | ref-image | yes | ref-matched | n/a | — | |
| F02 | Belts (dark bands) — Jup/Sat | 7 | yes | ref-image | yes | ref-matched | n/a | — | |
| F03 | Alternating zonal jets + superrotation — Jup/Sat | 5 | yes | ref-image | partial | ref-matched | n/a | yes | tiebroken; emergent Rhines jets DEFERRED research-grade — do not re-recommend |
| F04 | Band-boundary meander — Jup/Sat | 2 | yes | ref-image | yes | ref-matched | n/a | — | |
| F05 | GRS-class hero anticyclone — Jup | 12 | yes | ref-image | yes | ref-matched | yes | — | whirlpool root cause SOLVED via hero_solid_core (2026-06-25) |
| F06 | Turbulent wake of hero storms — Jup | 2 | yes | ref-image | partial | ref-matched | **no** | yes | wake trails due EAST, not WNW — see 4.1 |
| F07 | White ovals — Jup | 3 | yes | ref-image | partial | ref-matched | yes | yes | small ovals dissipate to eddies in the vorticity render |
| F08 | Brown barges — Jup | 2 | yes | ref-image | partial | ref-matched | n/a | yes | |
| F09 | Vortex street / string of pearls — Jup | 1 | yes | ref-image | partial | ref-matched | n/a | yes | |
| F10 | Merge debris collar — Jup | 2 | yes | prior-knowledge-advisory | partial | advisory-prior-knowledge | n/a | yes | tiebroken; pre-merge co-orbiting = recorded intentional deferral |
| F11 | Kelvin–Helmholtz billows — Jup/Sat | 2 | yes | ref-image | yes | ref-matched | n/a | — | |
| F12 | Festoons (blue-gray equatorial streamers) — Jup | 3 | yes | ref-image | partial | ref-matched | **no** | yes | roots on wrong belt edge — see 4.1 |
| F13 | 5-µm hot spots (visual holes) — Jup | 1 | yes | ref-image | partial | ref-matched | n/a | yes | mechanism wired; masked by filament churn at held-down festoon strength |
| F14 | Saturn ribbon wave | 2 | yes | prior-knowledge-advisory | partial | advisory-prior-knowledge | n/a | yes | |
| F15 | Mesoscale gravity-wave trains — Jup | 0 | no | code-audit-only | code-audit | — | — | — | no levers exist |
| F16 | Convective outbreaks (GWS class + NTB plumes) | 2 | yes | ref-image | yes | off-on-pair | n/a | yes | tiebroken |
| F17 | Folded filamentary regions (belt FFR) — Jup | 12 | yes | ref-image | yes | ref-matched | n/a | — | jet shear drives folding, not injection (recorded); resolution raise ruled out (0.9%) |
| F18 | Fine filaments / jet-riding streaks — Jup/Sat | 7 | yes | ref-image | yes | ref-matched | n/a | — | |
| F19 | Convective cell fields (popcorn granulation) | 1 | yes | ref-image | yes | single-seed | n/a | yes | tiebroken |
| F20 | Intermittent turbulence patches — Jup | 1 | yes | ref-image | partial | single-seed | n/a | yes | tiebroken |
| F21 | GRS internal spiral lanes + collar streamlines | 2 | yes | ref-image | partial | ref-matched | yes | yes | tiebroken |
| F22 | Storm-scale folded belt structure — Jup | 3 | yes | ref-image | yes | ref-matched | n/a | — | |
| F23 | Temperate lace mottle — Jup | 3 | yes | ref-image | partial | ref-matched | n/a | yes | |
| F24 | Reference band template (replica fidelity) — Jup | 1 | yes | code-audit-only | code-audit | — | — | — | |
| F25 | Thin dark lanes (Saturn-truthful) | 1 | yes | ref-image | yes | ref-matched | n/a | — | |
| F26 | Iso-luminance hue drift — Jup/Sat | 1 | yes | ref-image | partial | ref-matched | n/a | yes | emergent color FALSIFIED 2026-06-23 — do not re-recommend |
| F27 | Within-band color richness (pockets, aging) | 4 | yes | ref-image | yes | off-on-pair | n/a | — | frost root cause = bright-end chroma (recorded) |
| F28 | Jupiter polar cyclone clusters + FFR lace + teal cap | 13 | yes | ref-image | partial | ref-matched | n/a | — | discreteness = known deferred item; scored vs PIA21641 at map scale |
| F29 | Saturn polar hexagon | 3 | yes | ref-image | partial | ref-matched | n/a | yes | |
| F30 | Plain polar vortex — generic | 2 | yes | prior-knowledge-advisory | yes | single-seed | n/a | yes | tiebroken |
| F31 | Saturn-style global haze | 2 | yes | ref-image | yes | ref-matched | n/a | — | |
| F32 | Lightning storm glow (emission RGB) — Jup | 3 | yes | prior-knowledge-advisory | yes | single-seed | n/a | — | |
| F33 | Aurora ovals (emission alpha) — Jup/Sat | 5 | yes | prior-knowledge-advisory | yes | single-seed | n/a | — | |
| F34 | Vortex rims / collars | 4 | yes | ref-image | partial | ref-matched | n/a | yes | hero collar convincing; small-oval rings absent (tracks F07 limit) |
| A01 | Oval BA reddening (two color states) — Jup | 0 | no | ref-image | **no** | ref-matched | n/a | no | the audit's only hard no — see 4.2 |
| A04 | Saturn Great White Spot epoch | 2 | yes | ref-image | partial | ref-matched | n/a | yes | head reachable; planet-girdling tail absent |
| A05 | Belt–zone 5-µm emission contrast — Jup | 4 | yes | prior-knowledge-advisory | partial | single-seed | n/a | no | |
| A06 | Neptune Great Dark Spot + companions | 4 | yes | ref-image | partial | ref-matched | n/a | yes | dark spot reachable via reversed LUT; companions unplaceable (B5-5) |
| A07 | Neptune Scooter | 2 | no | ref-image | yes | ref-matched | n/a | — | nearest-lever path suffices for stills |
| A08 | Uranus polar hood | 5 | no | ref-image | partial | code-only | n/a | yes | |
| A09 | Polar haze caps (dark/tinted canvas) | 6 | yes | ref-image | yes | ref-matched | n/a | — | |
| A10 | Cyclone–anticyclone asymmetry (~9:1) | 4 | yes | prior-knowledge-advisory | partial | advisory-prior-knowledge | yes | yes | |
| A11 | EZ ochre-haze epoch (2018-19) — Jup | 4 | no | prior-knowledge-advisory | partial | advisory-prior-knowledge | n/a | no | expressible but undocumented — pair with B5-2 fix (Top-10 #8) |
| A12 | Limb darkening (render-side) | 0 | no | code-audit-only | code-audit | — | — | — | |
| X01 | White-oval convective towers | 0 | no | code-audit-only | not-judged | — | — | — | excluded (no levers, no batch) |

GUI-discoverability was assessed at the parameter level in the B-track (B2/B3 — the panel auto-generates every lever, gated only by the Advanced toggle) rather than per phenomenon.

### 4.1 Notable judgments — the two chirality fails

**F06 — GRS wake on the wrong flank.** Both morphology checks passed ("chaotic bright folded filaments and a coherent swirl street adjacent to the hero, same character class as the PIA07782 belt turbulence"), but the flank check failed with concrete mechanism evidence, quoted from the judge:

> "no — it trails due EAST: the audit crop places the hero at the crop's west edge with the wake field extending east; vortices.py wake_dir=+1 because the preset jet u at −22.5S is +0.106 (eastward; positive across −30..−15 deg, so hero_latitude nudges cannot flip it); psi.comp centers the wedge on the hero latitude (due east, not WNW)."

Judge's synthesis: "the wake wedge is stamped downstream of an eastward local jet, i.e. due east of the spot — the row's explicit FAIL condition — and this is systematic at the calibrated preset, not seed luck." Consensus was tiebroken (partial/yes/partial); `candidate_poor` because the calibration, not the lever, is at fault.

**F12 — Festoons root on the wrong belt edge.** The look itself is convincing ("thin blue-gray filaments with repeated curls against the cream EZ; correct darker/bluer offset"), but both judges independently confirmed the rooting defect:

> Judge 1: "bluish-pixel latitude profile of jv_on.png confines festoon tint to 0 to −8.5 deg; manifest festoon_lat −7.31; profiles.py select_wave_latitudes picks the template edge nearest 6.9 deg, which in the shipped Cassini template is −7.31 (SEB-N side) beating +5.91 — the train roots on the wrong belt edge."

> Judge 2: "the engine placed it at −7.31 deg; profiles.py picks the band edge nearest |7| deg sign-blind … which is the wrong belt edge per the chirality criterion (real festoons are NEB-south). … a template-driven placement defect, flippable by a band-template edit rather than a lever incapability."

The shared root cause — sign-blind nearest-edge selection preferring |−7.31| over |+5.91| for a +6.9° target — is the same defect class as the PR #9 antipode bug, hence Top-10 #4.

### 4.2 The single hard no — A01 Oval BA reddening

Both judges agreed with no tiebreak needed. The two HST references (opo0619a/b: GRS plus the smaller reddened Oval BA) were verified, and the render confirmed "our oval population is uniformly white — no second red oval exists or can be dialed in, since tint is per-kind constant and all heroes share one hero_latitude." Feature-check quote: "ovals carry fixed kind-constant tint (0.1) in vortices.py; warm-red T3 is hero-plumbed; hero_latitude is a single value so a second, smaller red hero at a different latitude is unreachable." `chroma_aging` tints by T2 freshness globally and cannot single out one oval. "Structural gap, not seed luck; the two-color-state epoch is unreachable." This is the same hardwired-tint root cause as B5-1 — one lever design (Top-10 #5) addresses both.

### 4.3 Scenario-render outcomes (B5 artifact)

All four scenario presets validated and rendered (wall-times in Appendix A). Each recipe is durably embedded in its render output — `out/scenario-*/mapset.json` → `preset.params` carries the full preset JSON — so the recipes survive even though the scratchpad copies will not persist. Their stuck points, condensed:

- **neptune_great_dark_spot** (from ice_giant): a dark storm tint IS achievable, but only by resculpting the shared storm_tints gradient (hero tint/brightness hardcoded; stamp_contrast skips KIND_HERO); no lever places bright companions adjacent to the hero; `hero_solid_core` is a silent no-op on the kinematic preset, so the dark oval risks the whirlpool-winding artifact; the reversed gradient also mildly darkens polar vortices — a coupling, not a choice.
- **saturn_hexagon_gws** (from saturn_pale): the hexagon half worked out of the box (first-class lever); the GWS half is a seed lottery — outbreak latitude not pinnable, equatorial GWS impossible (hardcoded 0.20 rad floor), eruption epoch seeded, plume brightness/train geometry are module constants.
- **jupiter_faded_seb** (from jupiter_vorticity): a fully whitened belt is inexpressible on the value axis (alternation validator caps the SEB at 0.55 vs original 0.52); `faded_sector`'s target won by 0.01° of width and had to be defensively stabilized; no per-band contrast trim or per-band dynamics lever, so the "faded" belt keeps churning like an active one; the whole recipe depends on the preset-only `bands.template`.
- **jupiter_modern_grs** (from jupiter_vorticity): no hero paleness scalar exists, so paleness had to be threaded into the shared LUT between the barge index (0.92) and hero index (0.95), with ~20% unavoidable bleed onto the reddest barges; hero brightness (0.05) likewise hardcoded — a milkier modern GRS core is not directly expressible.

### 4.4 Numeric-gate results

Run on the current tree, seed 4201 (details in Appendix A):

| Gate | Result | Detail |
|------|--------|--------|
| `swirl_gate --raw --preset jupiter_vorticity` (1536w) | **FAIL** | m1 0.89 PASS (≤2.30); **m3 jet continuity 0.51 FAIL (≥0.57)**; m5 0.54 PASS (≥0.22); m4/m6 = 1.0 by construction; overall FAIL on m3 |
| `_chromo_compare` | **FAIL** | **mean saturation +28% vs ≤+12% bound FAIL**; variety PASS; targeting corr −0.88 PASS |
| `_ffr_compare` | PASS | DC +1.23 < 8; frac_bright 0.48 < 0.50 — PASS/PASS |
| `_plume_compare` (efficacy) | measured | belt dLum +0.30, peak +149, frac 0.093 |

For orientation: `swirl_gate`'s m-metrics grade the vorticity-mode swirl morphology (m3 = jet continuity — how unbroken the zonal jets read through the eddy field; m1/m5 = swirl-size and medium-wavenumber texture bounds; m4/m6 pass by construction in `--raw` single-config mode). `_chromo_compare` bounds the `chroma_aging` lever's side effects (global mean-saturation shift must stay ≤ +12% while per-band variety and freshness-targeting improve); `_ffr_compare` checks the polar-filament lace for DC-glow and over-brightening against the PIA21641 polar reference; `_plume_compare` measures outbreak efficacy (belt luminance lift, peak brightness, affected fraction) rather than pass/fail.

**Caveat, stated deliberately:** the jupiter_vorticity ship config failing its own frozen swirl gate and the chromo saturation bound may be **accepted drift** from the source-fidelity calibration pass (#10) rather than regressions — the frozen thresholds predate that pass. This needs a human decision (re-baseline vs re-tune) before any automation; see Top-10 #10. Separately, the 6 failing GPU byte-identity tests trace to the uncommitted WIP on master (A3-6/A4-3), not a repo defect.

### 4.5 Protocol guardrails (honesty section)

- All fidelity renders were judged from a **single seed (4201)**; single-seed "partial" verdicts carry no engine-work recommendation per protocol — `candidate_poor` partials are recorded as calibration roadmap items, not engine gaps.
- Judging used **paired independent judges per batch with a third-judge tiebreak** (33 rows agreed, 8 tiebroken).
- Rows without reference imagery are capped at the **prior-knowledge-advisory evidence class** (matrix "Evidence class" column) and can never grade ref-matched: their quality is either advisory-prior-knowledge or, where a render was still judged, single-seed. Both grades are advisory-strength for engine conclusions.
- **No 3-seed mini-sweep was escalated:** every `capability=no` or chirality fail traces to a deterministic cause confirmed in code (validator cap, hardwired tint, sign-blind edge selection) — not seed luck — so a sweep would not have changed any verdict.
- Judges were fenced from re-recommending recorded falsifications/deferrals (emergent color, Rhines jets, polar-discreteness rework, pre-merge co-orbiting); the history column records the fences applied.

### 4.6 Cross-cutting themes

Four patterns recur across both tracks and are worth naming once, because they predict where the next defects will appear:

1. **Metadata-enforced sync points ship; convention-enforced ones drop.** The pfield system delivered the GUI widget, invalidation tier, and schema for `polar_filaments` for free; the two conventional points (doc entry, dedicated test) were silently skipped in the same PR (A2-1). The same split explains the `fx_on` hand-list risk (A2-1), the once-applied uniform tripwire (A2-6), and the sliders.md regeneration lag (B3-1). The durable fix in each case is the same: move the obligation into metadata or a cross-referencing test.

2. **Deterministic placement bugs are a class, not incidents.** The festoon rooting edge (F12, sign-blind nearest-edge selection), the wake flank (F06, jet-sign-driven wedge direction), the PR #9 antipode bug (already fixed), and `faded_sector`'s 0.01° widest-belt tiebreak (B5-4) are all the same shape: a geometric selection that ignores sign or hemisphere and happens to land wrong on the shipped template. Any new placement heuristic should get a signed-target test at review time.

3. **Shared color axes couple features that users think of as independent.** The storm_tints LUT serves heroes, barges, ovals, and polar vortices through one gradient with hardcoded per-kind indices (B5-1, B5-7, A01, A06) — every scenario that wanted to move one storm's color had to thread the needle between the others. The per-storm tint lever (Top-10 #5) is the systemic fix, not just an Oval BA feature.

4. **The loud-failure policy is real but has two blind spots.** Shader errors raise, preset errors toast, validation is strict — but the baroclinic degrade path logs silently to a file (A2-2), and the scipy import crash bypasses the degrade contract entirely (A2-3). Both are in the same subsystem, which shipped most recently; the policy needs to be a checklist item for new subsystems, not an emergent property.

---

## 5. Remediation roadmap

Effort tags: **quick-win** (< 1 session), **project** (1–3 sessions), **research** (open-ended). Verification names a concrete command/test/visual check for each item.

### Top-10, in final rank order

**1. Fix the CI push trigger** (A4-1) — Critical, quick-win.
Change `.github/workflows/ci.yml:5` from `branches: [main]` to `branches: [master]`.
*Dependencies:* none; do this first — items 6 and 10 assume post-merge CI exists.
*Verification:* push a trivial commit to master; confirm a CI run appears in the Actions tab.

**2. Merge PR #13** (B3-2, pre-confirmed) — High, quick-win.
The entire UX overhaul exists and is tested (~2,700 new test lines) but no user gets it.
*Dependencies:* **gates most UX items** (3 partially, and B1-3/B1-4/B1-5/B2-*/B3-3/B3-4/B4-* all describe branch code); reconcile with the uncommitted master viewport WIP (A4-3) first to avoid merge pain.
*Verification:* `gh pr merge 13`; launch the app from master; confirm the F1 help window opens.

**3. First-launch "Developing" feedback** (B1-1) — High, quick-win.
Label the progress bar with a verb where the user is looking (viewport overlay or Controls header), add a Speed hint or a faster default until first developed; consider a lower-res fast-preview default.
*Dependencies:* build on the PR #13 GUI (item 2).
*Verification:* smoke-run a cold start; confirm a labeled "Developing N/M" indicator is visible in the viewport/Controls pane plus a speed hint, without opening the Performance pane.

**4. Fix the placement chirality defects** (F12, F06) — High, quick-win → project.
Festoons: make `select_wave_latitudes` in `sim/profiles.py` sign-aware (it currently prefers |−7.31| over |+5.91| for a +6.9° target). Wake: place the wake wedge on the correct flank relative to the local jet (WNW of the hero), or expose the flank as a signed offset.
*Dependencies:* none; same defect class as the PR #9 antipode bug — check for other sign-blind selections while in there.
*Verification:* re-render jupiter_vorticity; the festoon blue-pixel latitude profile peaks at +5..+8° (NEB-S); wake filaments read WNW of the hero per the local jet.

**5. Per-storm color-state lever** (A01, B5-1, A06, B5-5) — High, project.
Replace the hardcoded hero tint (0.9) / brightness (0.05) with `storms.hero_tint` / `hero_brightness` pfields (default = current constants, byte-identical); consider a per-kind tint override so a second reddened oval (Oval BA) is expressible. Optionally add `storms.hero_companions` for the Neptune companion clouds (B5-5) while in the stamp code.
*Dependencies:* **needs no user decision on design intent beyond lever naming**, but follow the house rules: default-off/identical, forced-variant no-op test.
*Verification:* the new lever renders a dark anticyclone on ice_giant and a white-vs-red oval pair on jupiter_vorticity; A01's feature checks flip.

**6. Prune the dead solver scaffolding** (A1-1..A1-4) — High (maintainability), project.
Execute the per-file KEEP/ARCHIVE/DELETE lists in §2.2 exactly; keep `vorticity_ref` and the `shallow_water_ref` 2-layer subset; fix `test_m3_ref.py`'s two sw_spike cross-checks and the stale `test_gl_context.py` docstring.
*Dependencies:* **gates any future god-file/refactor work** (A2-5, A2-7) — do this before touching `sim/solver.py`; requires item 1 so the removal itself gets post-merge CI.
*Verification:* full suite green after removal; `uv run lint-imports` still passes; the llvmpipe CI step time drops measurably (~65 GPU tests removed).

**7. Agent-legibility package** (A3-1, A3-2, A3-8) — High, quick-win.
Commit the drafted CLAUDE.md (Appendix B) after user review; append the two under-recorded dead ends to `docs/roadmap.md` (jupiter_baroclinic dropped — pointer to build_vorticity_presets.py; polar discreteness blocked on merger physics); optionally add a `slow` pytest marker for a genuine fast loop.
*Dependencies:* none; amplifies every later item by making verification paths discoverable.
*Verification:* a fresh agent (or session) can find the test commands, layering rules, and the do-not-reattempt list from the repo root in one hop.

**8. Make a faded belt expressible** (B5-2) — Medium, quick-win.
Relax the BandTemplate alternation validator (explicit per-band identity override, e.g. optional `is_zone` list) or add a `bands.belt_fade` lever (belt selector + 0–1 whitening at derive time).
*Dependencies:* pairs with documenting epoch recipes — A11 (ochre EZ) was judged "expressible but undocumented", so write the two epoch recipes down when this lands. Consider folding in B5-4 (`faded_band_index` override) since it is the same addressability theme.
*Verification:* re-render the faded-SEB scenario; SEB luminance approaches zone level without the palette-row workaround.

**9. Docs-truth package** (A3-3, A3-4, A3-5, A3-7, B3-1, A2-1, B3-6) — Medium, project.
Fix the four architecture.md drift items (determinism qualification, baroclinic section, generation_version, DETAIL_FX list); regenerate sliders.md (`scripts/render_slider_examples.py`, resumable) and extend the generator for enum fields; derive `fx_on` from pfield metadata and add the doc/test cross-reference check; put the lever-author checklist in CLAUDE.md.
*Dependencies:* regenerate sliders.md AFTER item 2 merges (its Help window links to it) and after any lever additions from items 5/8.
*Verification:* re-run the generator and diff — zero missing numeric fields, `vort_inject_mask` present as text; a CI doc-drift check (param count vs sliders.md) goes green; architecture.md claims spot-verified.

**10. Fidelity-gate decision + WIP hygiene** (A3-6, A4-2, A4-3, gate runs) — Medium, project, **needs a user decision**.
First the human call: are the swirl m3 (0.51 vs ≥0.57) and chromo (+28% vs ≤+12%) FAILs accepted drift from the source-fidelity pass — re-baseline the thresholds — or regressions to re-tune? Then: branch/commit the uncommitted tracer-channel WIP (A4-3), which clears the 6 red byte-identity gates; then automate a cheap gate smoke (W=1024, 1 seed) in CI or a pre-release script (A4-2), and track the 4 keeper compare scripts (A4-4).
*Dependencies:* item 1 (CI must actually run on master); the WIP reconciliation interacts with item 2.
*Verification:* `swirl_gate --preset` and `_chromo_compare` PASS against the decided baselines; `pytest -m gpu` returns 0 failed on a clean tree; the gate smoke appears in CI or a documented checklist.

### Second tier (remaining Medium/Low, grouped by theme, one line each)

**Failure-mode surfacing and dependency correctness**
- A2-2 (Medium): surface baroclinic graceful degrade as a facade status + GUI toast; narrow the bare RuntimeError catch.
- A2-3 (Medium): remove the production scipy reach (cv2 replacement) or catch ImportError in `_init_baroclinic`; fix the "Test-only" comment.
- A2-6 (Low): generalize the B1 uniform tripwire to all DETAIL_FX uniforms via the A2-1 metadata.
- B5-6 (Low): warn when vorticity-only fields are non-default under `solver.type=kinematic`.

**Code shape (opportunistic, after Top-10 #6)**
- A2-5 (Low): extract the ~470 vorticity-only lines of `sim/solver.py` into `sim/omega.py` next time the path is touched.
- A2-7 (Low): add a seen-set to the include flattener (cycle/duplicate guard); split `detail.comp`'s DETAIL_FX region when the next lever lands.
- A2-4 (Low): hoist the three habit lazy imports; comment the three load-bearing ones.
- A2-8 (explicit non-action): leave `params/model.py` intact.

**Repo hygiene**
- A4-4 (Medium): track the 4 keeper scripts; delete/archive the 36 throwaways; decide the 3 borderline; adopt `scripts/scratch/` (gitignored).
- A4-5 (Low): gitignore or remove `_diag/`.
- A4-6 (Low): `git rm --cached .claude/tdd-guard/data/test.json`; ignore the directory.
- A4-7 (Low): delete the 3 provably merged branches + 1 remote now; verify-then-delete the 3 squash-merged.

**UX polish (post PR #13 merge)**
- B1-2 (Medium): friendly messages for the missing-gui-extra ImportError and GL<4.3 attach failure.
- B1-3 (Medium): add a "How the simulation develops" block to Help; explain the dirty marker and "unsaved".
- B1-4 (Medium): list output files in the export modal; warn on overwrite; persistent last-export path + open-folder.
- B4-2 (Medium): make hero_latitude an actual pin-checkbox widget; confirm on "Clear template".
- B4-3 (Medium): composite aurora (alpha × aurora_color) into the Emission preview channel; interim disclosure text.
- B4-4 (Medium): give `export.width` exactly one live editor and one undo policy.
- B4-1 (Medium): CLI-first sequence export (`--frames/--steps-per-frame`); at minimum expose `--dev-steps` now.
- B1-5, B1-6, B1-7, B1-8 (Low): pending-hint wording, README first-run sentence, aurora inline note, preset overwrite/delete affordances.
- B2-1 (Low, refuter-demoted from High), B2-2 (Medium): apply the ten tooltip rewrites; front-load the two long descriptions.
- B2-3, B2-4 (Low): label or hide the baroclinic cadence trio; tooltip the palette-widget leaves; add degree equivalents and gloss GRS/anticyclone.
- B3-4, B3-5, B3-6 (Low): presets.md user-preset paragraph; blender_addon.md three missing options; enum fallback in the slider-doc generator.
- B4-5, B4-6 (Low): app_version-aware preset rejection message + "Import preset…"; undo exclusions disclosed in Help (or fold output settings into history).

**Phenomenon calibration (single-seed partials — calibration roadmap, not engine work)**
- B5-3 (Medium): `outbreak_latitude` + phase knob; parameterize the 0.20 rad floor.
- B5-4 (Medium): `faded_band_index` override for the widest-belt heuristic (or fold into Top-10 #8).
- B5-5 (Medium): `hero_companions` (or fold into Top-10 #5).
- B5-7, B5-8 (Low): split stamp tint/brightness contrast; promote or document the outbreak constants.
- The 19 `candidate_poor` partials (of 22 partials total — only F28, A05, A11 are partial without the flag): F03, F06–F10, F12–F14, F20, F21, F23, F26, F29, F34, A04, A06, A08, A10 — revisit per-row after the Top-10 lever work, with a multi-seed pass before concluding anything about the engine.

### Decisions requested from the user

Five items in this roadmap cannot proceed on evidence alone; they need a call from you:

1. **Gate baselines (Top-10 #10):** are the swirl m3 and chromo FAILs accepted drift from the source-fidelity pass (re-baseline the thresholds) or regressions (re-tune the preset)?
2. **PR #13 (Top-10 #2):** merge, or explicitly close and cherry-pick — the review assumes merge.
3. **`shallow_water_ref` pruning depth (A1-2):** keep the orphaned Helmholtz/Williamson-2/SLSI sections as ground truth for a possible M2-AE milestone, or prune them with their tests once sw_gpu is archived.
4. **Borderline scripts (A4-4):** track or delete `palette_identities.py`, `vibe_gallery.py`, `seam_globe.py` — deletion is unrecoverable (untracked).
5. **Animation export priority (B4-1):** the CLI-first sequence exporter is scoped and cheap relative to its payoff, but it is new surface area — schedule it or leave `--dev-steps` as the interim.

---

## Appendix A — Methodology and evidence inventory

### Git state at review time

- HEAD: `df366a78200f349c8ab4f69208d93ff51b7fd57d` (master).
- Dirty files: `src/gasgiant/app/shaders/view_transform.frag`, `src/gasgiant/app/viewport.py`, `src/gasgiant/engine/facade.py`, `tests/gpu/test_hero_wake.py`, `.claude/tdd-guard/data/test.json`.
- **Dirty-files verdict: GUI-preview-only** (tracer-channel selector); the export/render path is identical to `df366a7`, so all fidelity renders reflect committed behavior.
- **Drift across the render window: none** — pre/post working-tree diff was byte-identical.
- PR #13 worktree: `f427a2269a7a4492eedf82e7224c8fe47b5be779`, in sync with origin.

### Render inventory (all on this machine's RTX 3070; total GPU wall time 1570 s)

| Job | Wall (s) | Output / result |
|-----|---------|-----------------|
| `compare_reference jupiter_vorticity --view agx` | 72 | `out/compare-jupiter_vorticity/` |
| `compare_reference gas_giant_warm --view agx` | 125 | `out/compare-gas_giant_warm/` |
| `audit_features --preset jupiter_vorticity` (8192w, 25 crop entries) | 119 | review-scratchpad audit renders |
| `swirl_gate --raw --preset jupiter_vorticity --drags 0.0 --seed 4201` (1536w) | 16 | m1 0.89 PASS ≤2.30 · **m3 0.51 FAIL ≥0.57** · m5 0.54 PASS ≥0.22 · m4/m6 = 1.0 by construction · overall FAIL on m3 |
| Lever driver: jupiter_vorticity 2048 ON + 3 OFF variants + judge views | 100 | ~25 s per render |
| Feature gates: `_ffr` / `_chromo` / `_plume` compares | 2 | ffr PASS/PASS (DC +1.23 < 8; frac_bright 0.48 < 0.50) · **chromo FAIL mean-sat +28% > +12%**, PASS variety, PASS targeting corr −0.88 · plume efficacy: belt dLum +0.30, peak +149, frac 0.093 |
| Scenario renders (B5): neptune_gds 4 s, saturn_hexagon_gws 5 s (kinematic); jupiter_faded_seb 90 s, jupiter_modern_grs 85 s (vorticity) | 184 | `out/scenario-*/` |
| B1 smoke cold-start: worktree GUI, gas_giant_warm 4096 grid / 1256 steps, 640 frames @ 2 steps/frame | **952** | **time to first developed planet ≈ 15.9 min** |

### Judge protocol

- 6 batches (global-whole-map, storms, temperate, equatorial, polar, emission-properties) × 2 independent judges each, plus tiebreak judges where the pair disagreed — 16 judging agents total.
- Consensus tally over the 41 judged rows: **33 agreed, 8 tiebroken** (tiebreak triples recorded per-row in the matrix).
- Evidence classes: ref-image (km/px-matched crop pairs against NASA refs: PIA07782, PIA21641, PIA21775, PIA12826, PIA00049, HST opo0619a/b), off-on lever pairs, single-seed renders, prior-knowledge-advisory (capped), and code-audit-only.
- Fidelity fences: judges were instructed not to re-recommend recorded falsifications/deferrals (matrix `history` column).
- Single master seed 4201 throughout; no multi-seed escalation was triggered (every hard verdict traced to a deterministic code cause).

### Agent inventory

3 explorers, 6 plan reviewers, architecture workflow 5 agents (A1–A4 + refuter), UX workflow 6 agents (B1–B5 + refuter), C0 matrix builder 1, judging 16, plus synthesis, critic, and writer agents.

### Evidence bundle

All source data lives in the session review scratchpad; this report is the durable artifact and every finding id above maps 1:1 to that bundle:

| File | Contents |
|------|----------|
| `findings/synthesis.json` | merged Top-10 priority queue, phenomenon summary, escalation decisions, demotions |
| `findings/arch.json` | architecture findings A1–A4 with strengths and refuter verdicts |
| `findings/ux.json` | usability findings B1–B5, journey trace, divergence table, scenario recipes (B5 artifact) |
| `findings/coverage_matrix.json` | the 45-row phenomenon matrix with consensus verdicts |
| `findings/judging.json` | full per-row judge reasoning and feature checks |
| `findings/claude_md_draft.md` | the CLAUDE.md draft reproduced verbatim in Appendix B |
| `appendix_inventory.json` | git-state snapshot, render wall-times, gate results |
| `c0_matrix.json`, `judge_manifest.json` | row metadata (levers, history) and judge crop assignments |
| `renders/`, `scenarios/`, `drivers/`, `refs-extra/` | rendered evidence, scenario presets, driver scripts, extra references |

---

## Appendix B — Proposed CLAUDE.md

Verbatim draft, **ready to commit after user review** (Top-10 #7). Fenced as a document; the inner ``` blocks are part of the file.

````markdown
# CLAUDE.md

Gas Giant Studio: GPU "sim-advected procedural" gas-giant texture generator. Python 3.13,
moderngl (GLSL 430 compute), pydantic params, imgui GUI, headless CLI, Blender addon.
`engine/facade.py::Simulation` is the single facade — GUI, CLI, and tests consume only it.

## Commands (all uv-based; verified against pyproject.toml)

```sh
uv sync --all-extras                 # deps incl. GUI extra (imgui-bundle)
uv run gasgiant-studio               # live-preview GUI (needs a display + GL 4.3)
uv run gasgiant export --preset gas_giant_warm --res 2048 --out out/x   # headless render
uv run gasgiant validate out/x       # seam/pole invariants on an exported map set
uv run ruff check .                  # lint (line-length 100; E701/E702 deliberately off)
uv run lint-imports                  # layer contracts — run after ANY new import
uv run pytest -m "not gpu" -q        # no-GPU subset (362 tests, ~7 min: CPU reference solvers dominate)
uv run pytest -m gpu -q              # GPU subset (173 tests, ~7 min on RTX 3070; llvmpipe in CI)
uv run python scripts/p05_baseline_hash.py --check   # float32 render-hash gate (machine-local baseline)
uv run python scripts/build_addon.py # -> dist/gasgiant_importer-*.zip
```

Factory presets (`src/gasgiant/presets/`): gas_giant_warm (flagship), jupiter_like,
jupiter_vorticity, saturn_pale, ice_giant. README's preset list is stale.

## Testing policy

- `pyproject.toml` testpaths = `tests/unit` + `tests/gpu` only. `tests/spikes/` is outside
  testpaths (run explicitly); `tests/blender/test_import.py` runs inside
  `blender --background --factory-startup --python tests/blender/test_import.py -- <mapset_dir>`.
- GPU tests are marked `pytestmark = pytest.mark.gpu` and use the session `gpu` fixture
  (`tests/conftest.py`), which **skips cleanly if no OpenGL 4.3 context exists** — a sandboxed
  agent without a GPU can still run the full command; gpu tests skip, unit tests run.
- CI (`.github/workflows/ci.yml`) runs everything on llvmpipe: `LIBGL_ALWAYS_SOFTWARE=1` plus
  apt `libegl1 libgl1-mesa-dri libosmesa6`. Replicate that on Linux for software-GL runs.
- **Byte-identity vs tolerance:** the kinematic path is byte-exact (source-hash pin in
  `tests/unit/test_kinematic_kernels_pinned.py`; render-hash gate in `scripts/p05_baseline_hash.py`).
  The vorticity path is NOT: its SOR Poisson solve carries ~1e-3 cross-instance / ~0.004
  cross-session LSB noise, so vorticity-touching GPU tests assert within documented floors
  (`GPU_NOISE_ATOL = 1e-2`, `_VORT_SOR_ATOL = 1e-3` — see tests/gpu/test_checkpoint.py).
  Never write a byte-exact assertion for vorticity-mode output; never "fix" a kinematic
  hash mismatch by adding tolerance — update the pin deliberately instead.
- **Establish a baseline before editing**: run the relevant subset (or the p05 --check) first.
  Byte-identity/no-op gates fail whenever tracked default output moves, including from
  someone else's uncommitted work — know what was red before you touched anything.
- New opt-in features must be default-off and byte-identical when off (preprocessor-gated
  out, not branch-guarded), with a forced-variant no-op test pinning the variant.

## Layering (import-linter-enforced; violations fail CI)

```
params | palette  ->  gl  ->  core  ->  sim  ->  render  ->  jobs  ->  export  ->  engine  ->  app | cli
```

Lower layers never import higher ones. GUI libraries (imgui_bundle, gasgiant.app) are
forbidden everywhere below `app`. `gl` is the ONLY moderngl touchpoint.

## Conventions

- **Params**: every tunable is `pfield()` (`src/gasgiant/params/model.py`) carrying
  `json_schema_extra` metadata — `tier` (POST = re-derive maps; VELOCITY = rebuild ψ, run
  continues; RESTART = dev run restarts), `rand` (seeded-randomize range), `log`, `ui`
  (panel group; GUI panels are auto-generated from this). Metadata is plain JSON — no
  callables, no GUI imports. Unknown preset keys are hard errors (strict models).
- **Shaders**: loaded via `GpuContext.compute(package, name, defines)` (`src/gasgiant/gl/context.py`);
  `#include "file.glsl"` expands with error line-mapping; cross-package form:
  `#include "gasgiant.sim.kernels:noise3d.glsl"`. `defines` are injected after `#version`
  and programs are cached per (package, name, defines). Optional features compile as
  preprocessor variants (derive.comp: EMISSION, CHROMA_FX; detail.comp: DETAIL_FX) so the
  default program text is the pre-feature kernel — byte-identical by construction.
- **GLSL gotchas**: `patch` is a reserved word; declare uniforms before the `#include` that
  uses them (includes are textual); every sampler uniform must be explicitly bound — an
  unassigned moderngl sampler silently reads unit 0.
- **Determinism**: one master seed; every stochastic subsystem uses a named SeedSequence
  substream (`params/seeds.py`, `subseed(seed, "name")`). Never share streams.
- **Solver modes**: `solver.type` = kinematic (v1.5 analytic ψ, default for legacy presets)
  or vorticity (v1.6 prognostic q, used by gas_giant_warm/jupiter_vorticity). Some levers are
  mode-specific: `psi.comp` params are feather-only in vorticity mode; `storms.hero_solid_core`
  is a no-op in kinematic mode. Opt-in baroclinic coupling (`engine/baroclinic_coupling.py`)
  requires vorticity mode; off = byte-identical.

## Docs map

- `docs/architecture.md` — solver, three-domain seamlessness, invalidation tiers, export,
  variants. Mostly accurate; known drift: generation_version literal, DETAIL_FX trigger list,
  no baroclinic section, determinism claim unqualified for vorticity (see Testing policy).
- `docs/formations.md` — the phenomenon catalog and which mechanism implements each.
- `docs/realism.md`, `docs/sliders.md`, `docs/presets.md`, `docs/blender_addon.md`.
- `docs/roadmap.md` — includes FALSIFIED verdicts; read before proposing sim-architecture work.
- `docs/superpowers/specs/*verdict*.md`, `*falsified*.md` — milestone (M0–M3) falsification
  records for the shallow-water line. `sim/sw_*`, `sim/*_ref.py`, `sim/sw_spike/` and the
  swp_* kernels are milestone scaffolding from that line, not the production solver
  (`sim/solver.py` is production).

## Falsified / dead ends — do NOT re-propose without flagging this history

- **Emergent color** (passive tracer + designed curl field): falsified 2026-06-23 —
  washes bands in both modes (`docs/roadmap.md`).
- **Rhines-mechanism emergent jets**: research-grade, explicitly gated off (`docs/roadmap.md`).
- **Semi-Lagrangian advection for the SW solver (M2-adv)** and **q-target-bias/fast-nudge
  coherence levers**: falsified (`docs/superpowers/specs/m2-adv-verdict.md`,
  `2026-06-16-m3-qtarget-falsified.md`).
- **jupiter_baroclinic preset**: dropped (comb reads mechanical); the coupling engine stays.
- **Polar cyclone discreteness**: blocked on vortex-merger physics.
- **Pre-merge co-orbiting vortices**: deferred to the animation release — it would break the
  merger gate's purity (`docs/formations.md`).

## Hygiene notes

- `scripts/` mixes tracked tooling (build_addon, p05_baseline_hash, swirl_gate, calibrate_*)
  with dozens of untracked one-off diagnostic scripts; `_diag/` is untracked scratch. Don't
  cite or extend the untracked ones.
- Any pass that binds an offscreen FBO must rebind the default framebuffer before returning
  (the imgui backend renders into whatever is bound).
````

*End of report.*

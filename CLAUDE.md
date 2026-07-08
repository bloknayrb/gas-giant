# CLAUDE.md

Gas Giant Studio: GPU "sim-advected procedural" gas-giant texture generator. Python 3.13,
moderngl (GLSL 430 compute), pydantic params, imgui GUI, headless CLI, Blender addon.
`engine/facade.py::Simulation` is the single facade — GUI, CLI, and tests consume only it.

## Commands (all uv-based; verified against pyproject.toml)

```sh
uv sync --all-extras                 # deps incl. GUI extra (imgui-bundle)
uv run gasgiant-studio               # live-preview GUI (needs a display + GL 4.3)
uv run gasgiant export --preset gas_giant_warm --res 2048 --out out/x   # headless render
uv run gasgiant export --preset saturn_pale --frames 60 --steps-per-frame 5 \
      --ramp-to gas_giant_warm --all-maps --video --fps 24 --out out/seq  # A->B animation seq
uv run gasgiant export --resume ckpt.npz --frames 60 --steps-per-frame 5 --out out/seq  # from checkpoint
uv run gasgiant export --recipe faded_seb --out out/x   # apply an epoch recipe overlay (docs/recipes.md)
uv run gasgiant checkpoint --preset gas_giant_warm --out ckpt.npz   # develop + save a resumable .npz
uv run gasgiant palette-fit --image jup.png --preset jupiter_like --out out/fit.json  # bake palette rows from a photo
uv run gasgiant sheet --preset gas_giant_warm --count 12 --res 256 --out sheet.png    # seed contact sheet
uv run gasgiant validate out/x       # seam/pole invariants on an exported map set
uv run ruff check .                  # lint (line-length 100; E701/E702 deliberately off)
uv run lint-imports                  # layer contracts — run after ANY new import
uv run pytest -m "not gpu and not slow" -q   # FAST loop — use this while iterating (~40 s)
uv run pytest -m "not gpu" -q        # full no-GPU tier (~380 tests, ~6 min: CPU reference solvers dominate)
uv run pytest -m gpu -q              # GPU tier (~7 min, needs GL 4.3; llvmpipe works)
uv run python scripts/p05_baseline_hash.py --check   # float32 render-hash gate (machine-local baseline)
uv run python scripts/build_addon.py # -> dist/gasgiant_importer-*.zip
```

Markers: `gpu` (needs a GL 4.3 context) and `slow` (heavy CPU reference-solver files).
Factory presets (`src/gasgiant/presets/`): gas_giant_warm (flagship; GUI startup default),
jupiter_like, jupiter_vorticity, saturn_pale, ice_giant. The GUI (post PR #13) has
searchable auto-generated panels, per-slider help, undo/redo, and playback controls.

## Testing policy

- `pyproject.toml` testpaths = `tests/unit` + `tests/gpu` only. `tests/spikes/` is outside
  testpaths (run explicitly); `tests/blender/test_import.py` runs inside
  `blender --background --factory-startup --python tests/blender/test_import.py -- <mapset_dir>`.
- GPU tests are marked `pytestmark = pytest.mark.gpu` and use the session `gpu` fixture
  (`tests/conftest.py`), which **skips cleanly if no OpenGL 4.3 context exists** — a sandboxed
  agent without a GPU can still run the full command; gpu tests skip, unit tests run.
- CI (`.github/workflows/ci.yml`) runs GL under xvfb-run + llvmpipe (`LIBGL_ALWAYS_SOFTWARE=1`
  plus apt `libegl1 libgl1-mesa-dri libosmesa6 xvfb`; glcontext's x11 backend needs a DISPLAY
  even for software GL — before 2026-07-03/PR #25 the runner had none, so ~178/185 gpu tests
  silently skipped while CI reported green). Per event: every PR runs the no-GPU tier plus a
  PR-blocking `gpu-smoke` job = the full byte-identity/no-op class
  (`pytest -m gpu -k "identical or noop or no_op"`, ~31 tests, ~24 min) under
  `LP_NUM_THREADS=1` — single-threaded llvmpipe is deterministic; the default thread pool
  gives run-to-run vorticity/SOR divergence of 0.06–0.53, far past the RTX-calibrated
  floors. The FULL gpu tier (~185 tests, >3 h under llvmpipe — ~150x slower than native
  GPU) runs as a non-blocking `gpu-full` job on push to master + nightly schedule +
  workflow_dispatch: threaded for speed with the 15 determinism-sensitive vorticity tests
  deselected, then those 15 re-run single-threaded in a second step. "CI is authoritative
  for byte-identity" = gpu-smoke on every PR. Replicate the xvfb+llvmpipe (+
  `LP_NUM_THREADS=1` for determinism) setup on Linux for software-GL runs.
- **Machine-local GPU flakiness (KNOWN, not your bug):** ~12 GPU byte-identity/no-op tests
  are flaky on the primary dev box (RTX 3070) — GL session-context LSB noise (~0.004) that
  appears after other GL work has run in the same session. They pass on a fresh boot and
  in CI under single-threaded llvmpipe (an earlier "they PASS in CI under llvmpipe" claim
  predated the 2026-07-03 CI fix — CI was silently skipping them; under THREADED llvmpipe
  the vorticity no-op tests fail with run-to-run divergence 0.06–0.53, hence
  `LP_NUM_THREADS=1` in ci.yml). The flaky set SHIFTS between reruns (10 vs 6 with different members
  observed) — no fixed blocklist is possible. Protocol: (a) establish a green/red baseline
  on the SAME machine/session BEFORE attributing any GPU test failure to your change;
  (b) CI is authoritative for byte-identity claims; (c) re-run a suspected-flaky GPU test
  once before investigating.
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
- **Export levers (all default-off / byte-identical when off)**: `export.projection` =
  equirect (default; manifest schema_version 1) or cube (6-face map, per-face `width/4`,
  manifest schema_version 2 with a per-map `faces` block — cube OMITS synthesized detail and
  the flow/rings maps, both equirect-space); `export.flow_map` adds `flow.exr` (east/north
  velocity, equirect only); `rings.enabled` adds `rings.exr` (radial strip, Blender-only,
  invisible in the GUI preview). derive.comp's preprocessor variants are now
  (EMISSION × CHROMA_FX × MASK) plus BAND_TINT and PROJECTION_CUBE; `mask.file` binds an
  imported paint mask whose band_fade/emission_gain/detail_gain art-direct POST output.

## Lever-author checklist (adding a new opt-in visual lever)

1. `pfield()` in the params model (tier + rand + ui metadata; default = no-op).
2. Shader uniform + preprocessor block (variant define, not a runtime branch).
3. Variant-selection predicate: for detail-FX levers, tag the pfield `fx=True` — the
   DETAIL_FX predicate, the build-time uniform tripwire, AND the cross-ref test all derive
   from that flag (`render/detail.py`; tests/unit/test_detail_fx_metadata.py). Other kernels
   need an explicit `fx_on`-style predicate so the variant only compiles when active.
4. `_set` call wiring the uniform in the render/sim pass.
5. Preset build scripts (`scripts/build_*_preset*.py`) updated if the lever ships in a preset.
6. Regenerated preset JSONs (build scripts have a load==save reproducibility diff-guard).
7. Dedicated behavior test + forced-variant no-op test (byte-identical when off).
8. `docs/sliders.md` entry: regenerate the text with
   `uv run python scripts/render_slider_examples.py --no-render` and render the new images
   (see the script docstring). CI fails on a stale sliders.md — the test job runs the
   generator's `--check` drift gate (text-only, no GL).

## Docs map

- `docs/architecture.md` — solver, three-domain seamlessness, invalidation tiers, export,
  variants, opt-in baroclinic coupling, mode-qualified determinism (kinematic byte-exact,
  vorticity within noise floors). Truth pass 2026-07-03 (W8) cleared the known drift list.
- `docs/formations.md` — the phenomenon catalog and which mechanism implements each.
- `docs/realism.md`, `docs/sliders.md`, `docs/presets.md`, `docs/blender_addon.md`.
- `docs/roadmap.md` — includes FALSIFIED verdicts and the dead-end record; read before
  proposing sim-architecture work.
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
- **jupiter_baroclinic preset**: dropped 2026-06-28 (comb reads mechanical); the coupling
  engine stays (`scripts/build_vorticity_presets.py` top-of-file note; `docs/roadmap.md`).
- **Polar cyclone discreteness**: deferred — blocked on vortex-merger physics
  (research-grade; `docs/realism.md`, `docs/roadmap.md`).
- **Pre-merge co-orbiting vortices**: deferred to the animation release — it would break the
  merger gate's purity (`docs/formations.md`).

## Hygiene notes

- `scripts/` mixes tracked tooling (build_addon, p05_baseline_hash, swirl_gate, calibrate_*)
  with dozens of untracked one-off diagnostic scripts; `_diag/` is untracked scratch. Don't
  cite or extend the untracked ones.
- Any pass that binds an offscreen FBO must rebind the default framebuffer before returning
  (the imgui backend renders into whatever is bound).

# Architecture

## Overview

Gas Giant Studio is a GPU "sim-advected procedural" generator: a physically
motivated velocity field (zonal jets + storm vortex registry + shear-driven
turbulence, all expressed through a streamfunction ψ so the flow is
divergence-free on the sphere) through which four cloud tracer fields are
advected by a semi-Lagrangian MacCormack solver. Map derivation turns the
tracer state into seamless equirectangular color/height maps at any
resolution; a thin Blender extension imports an exported map set as a
ready-to-render planet.

```
params/palette          pydantic parameter tree (validation = UI metadata =
                        invalidation tiers), gradient LUTs
   ↓
gl                      the ONLY moderngl touchpoint: context (windowed-attach
                        or headless), shader loader (#include + error
                        line-mapping), textures/FBOs/SSBOs
   ↓
core                    grid topology, texel-center conventions
   ↓
sim                     profiles (exact 1D jet/ψ/shear/stamp LUTs via numpy),
                        vortex registry, events, GLSL kernels, 3-domain solver
   ↓
render                  map derivation + detail synthesis kernels
   ↓
export / jobs           tiled snapshot exporter, writers, manifest contract
   ↓
engine                  the Simulation facade: tick/preview/export/snapshot +
                        invalidation dispatch
   ↓
app | cli               imgui GUI (hello_imgui runner)  |  headless CLI
```

The layer order is enforced by import-linter (`pyproject.toml`); GUI
libraries are forbidden below `app`. The GUI/CLI/tests all consume only the
`engine.Simulation` facade.

## The three domains

Seamlessness on a sphere is the central design constraint:

- **Equirect main grid** (2:1, periodic X, texel centers, no duplicated
  column). All stochastic fields sample 3D simplex noise at unit-sphere
  positions — inherently seamless in longitude and consistent across domains.
- **Two square azimuthal-equidistant polar patches** (poleward of ~56°,
  uniform resolution across the pole). The same GLSL kernels compile for all
  three domains via a `DOMAIN` define; patches backtrace in Cartesian patch
  space with the AE azimuthal metric ρ/sin ρ.
- **Per-step one-way nesting exchange:** equirect is authoritative below
  ~64°, patches above ~66°; each step both directions are resampled with
  feathered overwrites. Two independently-evolving chaotic solutions would
  decorrelate and ghost when feathered — per-step slaving is what makes the
  narrow derive-time composite feather (64–67°) invisible. Cross-domain RMS
  in the band is monitored (`Solver.exchange_band_rms`, ~1e-3 in practice).

## The solver

Per step: drift vortex centers with the zonal flow (CPU registry → SSBO),
rebuild ψ (turbulence decorrelates via a time uniform — between steps only),
derive the frozen velocity, then three MacCormack passes per domain:

- RK2 (midpoint) backtrace; velocity sampled with hardware bilinear (fresh
  each step — error does not accumulate in velocity).
- Tracers sampled with a 16-tap `texelFetch` Catmull-Rom with FP32 weights
  and explicit X wrap — hardware filtering's ~8-bit fractional weights would
  accumulate artifacts over hundreds of resampling steps.
- The MacCormack limiter clamps against the 2×2 stencil at the *backtraced*
  position (the samples actually used), falling back to plain SL on
  violation.
- **Relaxation forcing:** T0/T1 relax toward the analytic band stamp PLUS
  the live vortex/wave stamps (τ ≈ `turbulence.relax_tau` steps). This is the
  anti-washout mechanism — advection alone homogenizes any passive tracer;
  physically, condensation chemistry continually regenerates band coloration
  and anticyclones their bright tops. T2 is replenished with fresh evolving
  noise (≈ exponential-decay flow noise by construction).

There is no feedback from tracers to velocity, so the kinematic solver is
unconditionally stable: the only failure modes are visual (washout — guarded
by a variance test over 300 steps) and they are tested.

The advect kernel binds samplers at fixed units in `solver._correct`:
`u_src`/`u_vel`/`u_cur`/`u_back`/`u_profile_stamp` at 0–4, and (v1.5)
`u_profile_dyn` at **unit 5** for the belt mask (its `.a` channel) used by
`turbulence.belt_replenish`. moderngl trap: an *unassigned* sampler uniform
defaults to unit 0 (= the forward tracers in pass 2), so every sampler must be
explicitly bound even when its feature is guarded off — the belt-replenish
block is `if (u_belt_replenish > 0.0)` but `u_profile_dyn` is bound
unconditionally.

### v1.6 vorticity-streamfunction solver (opt-in)

`solver.type = vorticity` switches from the analytic kinematic ψ to a
prognostic vorticity-streamfunction fluid on all three domains. The tracer
advection pipeline (`advect.comp`) is unchanged; the new solver replaces how
ψ and velocity are rebuilt each step.

**Kinematic vs vorticity strategy.** The kinematic solver rebuilds ψ analytically
from the jet profile + vortex registry every step — the flow never evolves, it is
always the idealized prescribed field. The vorticity solver carries a live scalar
field q (absolute vorticity) that is advected by the flow, nudged toward a target
derived from the same jet profile and vortex registry, and then used to recover ψ
by inverting the Poisson equation ∇²ψ = +ω. The jet shear folds q into filaments
between nudge corrections, producing the belt morphology the kinematic path cannot.
One consequence: ψ = ∇⁻²ω is intrinsically ROUNDER than the vorticity that sources
it (Poisson low-pass — a hero ring held at aspect 2.2 induces boundary streamlines
of only ~1.3, and the dye rides streamlines), so `storms.hero_flow_aspect` (K)
pre-compensates by authoring the emergence ring/skirt on a K-widened EW metric
while every tracer stamp keeps the authored anatomy; the widened ring's net
circulation is held invariant by a CPU-computed spherical renorm
(`sim/flow_renorm.py`). Vorticity-mode only; inert at K = 1.

**Absolute vorticity formulation.** The prognostic variable is q = ω + f where
ω = ζ is the relative vorticity (sign convention ζ = +∇²ψ, consistent with the
existing ψ-velocity pairing) and f = f₀ · sin φ is the Coriolis term. Advecting q
instead of ω conserves the planetary vorticity contribution automatically and
sets the β/Rhines scale via `coriolis_f0`. After each MacCormack step the
Coriolis term is recovered as q − f when computing ω\_rel for the Poisson solve.

**Per-step vorticity pipeline (each domain independently):**
1. Three MacCormack passes (`omega_advect.comp`) advect q — same RK2 backtrace,
   Catmull-Rom interpolation, and 2×2 limiter as the tracer advect, but scalar R32F.
2. Nudging sub-pass (`omega_force.comp` SUBPASS 0): q\_target = ω\_jet + ω\_vortex + f.
   ω\_jet from a new `profile_omega` LUT (integrated from the jet-shear profile);
   ω\_vortex = analytic Gaussian-Laplacian from vortex SSBO. Rate = 1/τ\_ω per step.
   Also applies optional Rayleigh drag and a hard polar confinement gate to prevent
   eddy growth in the exchange band where the domain is non-authoritative.
3. Intermediate step: `omega_lap.comp` computes ∇²ω\_rel into a scratch texture.
4. Hyperviscosity sub-pass (`omega_force.comp` SUBPASS 1): grid-normalized biharmonic
   q += ν₄ · (−∇²(∇²ω\_rel)) · Δ⁴/64. Applies to ω\_rel only; Coriolis is excluded.
5. Recover ω\_rel = q − f (`omega_recover.comp`).
6. Red-black SOR Poisson solve (`poisson_sor.comp`): `poisson_iters` × (red + black)
   checkerboard sweeps update ψ in place; SOR factor `sor_omega` ∈ (1, 2).
7. Velocity derived from ψ by the unchanged `velocity.comp` (`−∂ψ/∂φ`, `(1/cosφ)∂ψ/∂λ`).

**Naive vs AE-metric Laplacian.** The equirect Poisson uses the standard 5-point
spherical-metric stencil (1/cos²φ longitude weights, tanφ split for the asymmetric
latitude weights). The polar patches use a 9-point AE-metric stencil in (s, t) patch
coordinates: coefficients c\_ss, c\_tt, c\_st (cross-derivative), and c\_g (radial)
derived from the azimuthal-equidistant metric ρ/sin ρ. The stencil is regular at the
pole (ρ → 0 is handled analytically via the limit); no coordinate singularity.

**Per-domain ω states.** Each of the three domains (`_OmegaState` in `solver.py`)
holds its own ping-pong R32F textures for q plus scratch textures for ∇²ω\_rel, ω\_rel,
the kinematic warm-start ψ, and the SOR working ψ. The equirect and polar-patch ω
states evolve independently per step; only the tracer exchange (and the shared jet/
vortex nudge target) couples them.

**Cross-domain ψ-coupling is unnecessary.** No boundary conditions are exchanged
between the equirect ψ and the patch ψ at the exchange band. The shared nudge target
(same `profile_omega` LUT and same vortex SSBO across all domains) plus the per-step
tracer exchange produce a seam-free full-sphere output — an adversarial polar judge
independently verified no visible discontinuity at the 64–67° band.

**Testing note.** An early Phase-A test suite passed while the vorticity solver was
effectively a no-op (uniforms not set). The gate was strengthened to require: (a)
param-responsiveness (changing `coriolis_f0` measurably moves the ω field), (b)
ω evolves over dev steps, (c) steady-state ω magnitude is bounded in the production
regime. Recorded as a test-design lesson.

### Opt-in baroclinic coupling (M3)

`solver.baroclinic.enabled` (`BaroclinicParams`) couples the v1.6 development
run to an evolving CPU baroclinic source. It requires `solver.type =
vorticity` — a params validator rejects the combination otherwise. A seeded,
cached `BaroclinicSourceDriver` (`sim/baroclinic_driver.py`) is warmed up at
construction (`warmup_steps`), advanced every `update_every` v1.6 steps, and
its coherent vorticity source is uploaded and overlaid onto the Poisson RHS
with `gain` (`Simulation.set_external_vorticity_source`). The overlay enters
ω\_rel in `omega_recover.comp` each step — NOT the persistent q state — so it
stays bounded and decoupled from `vort_relax_tau`. `Simulation.tick` clamps
step chunks so they never straddle a source-update boundary: preview (small
chunks) and export (large chunks) develop identically.

- **Off = byte-identical:** the default path never calls
  `set_external_vorticity_source`; the shader gate is an exact-zero compare
  (`u_external_gain != 0.0` in `omega_recover.comp`, always bound to a
  Python-side float), so disabled means a strict no-op, not a hoped-for
  branch elimination.
- **Graceful degrade:** on the documented failure signals only — warmup
  outcrop, mid-run outcrop / incoherent source, or a missing optional
  dependency — the facade drops to plain uncoupled v1.6 instead of crashing.
  `Simulation.baroclinic_status` reads `'off' | 'active' | 'degraded'` and
  `baroclinic_degraded_reason` carries the cause (the GUI toasts on the
  transition, so the degrade is never silent). Genuinely unexpected errors
  still propagate loudly.
- The `jupiter_baroclinic` preset was dropped 2026-06-28 (its comb read
  mechanical — see `docs/roadmap.md`); the coupling engine remains. The
  facade path above is what the GUI/CLI use;
  `engine/baroclinic_coupling.py::run_coupled` is the instrumented
  measurement harness (per-phase wall time, residency recommendation).

## Invalidation tiers

Every parameter declares its tier in field metadata; the engine diffs
validated parameter trees and dispatches the cheapest sufficient recompute:

| Tier | Examples | Action |
|---|---|---|
| POST | haze, palette, contrast, detail synthesis | re-derive maps (instant) |
| VELOCITY | jet strength, turbulence amplitude | rebuild profiles/ψ; run continues (+adaptation steps if finished) |
| RESTART | seed, band layout, storms, poles | rebuild everything, development run restarts (the GUI shows it evolving) |

### Hero jet environment (`jets.hero_bracket_*`)

A default-off, RESTART-tier lever set in `build_profiles`: with a pinned hero it
replaces the seeded band jets inside a feathered (C1 smoothstep) hero-centered
window with an authored two-sided bracket (flat pedestal + equatorward/poleward
gaussians), so the hero's local shear is artist-authored and seed-independent
instead of fighting the seeded jets. Applied after `u *= strength`/`polar_fade`
and before ψ/ω, so it is mode-agnostic (kinematic + vorticity). Byte-identical at
defaults (a structural `!= 0.0` guard skips the whole block — a CPU/numpy skip,
not a GLSL variant). A pure-function `seat_quality` metric (exposed via
`Simulation.seat_status`, rendered as a GUI meter) scores how well the *natural*
(bracket-off) jets already seat the hero — a diagnostic that never moves the
storm. All bracket geometry (offset/window/feather/gaussian-width) is expressed
in **units of the hero core radius** (`hero_radius`), so the bracket tracks storm
size automatically — the facade threads `hero_radius` into `build_profiles` at
every call site. The pfield defaults were calibrated against the warm hero
radius; the bare model-default `hero_radius` is 0.10. **Baked into
`gas_giant_warm`:** its hero sits at −24.0° (`hero_radius` 0.108) with an
authored two-sided anticyclonic bracket (`hero_bracket_north = −3.0`,
`hero_bracket_south = 3.0`, geometry in units of the hero core radius), which
replaced the earlier `local_jet` seat (now zeroed out of the preset).

## Export

`export/exporter.py` is a generator: finish development (yielding progress),
capture an **ExportSnapshot** (GPU-side clones of tracers/velocity/LUTs +
deep param copy — tiles can never disagree from mid-export edits), render
1024² tiles (the derive and detail kernels take an origin/full-size, and read
only sim-resolution snapshot textures + analytic noise, so tiles need no
apron), assemble per map directly in final dtype on the CPU, then encode in
a thread pool. The GUI runs one slice per frame with progress + cancel
(cancellation removes only the files we wrote); the CLI drains the same
generator. Maps: 16-bit color PNG + float32 height EXR, plus a float32
RGBA emission EXR (thermal hot-spot glow + lightning in RGB, aurora
intensity in alpha) when any `emission.*_strength` is nonzero. Measured:
16384×8192 with all maps and all FX variants on in ~31 s on an RTX 3070
(encode-bound).

**Projection + extra maps (all default-off, byte-identical when off).**
`export.projection = equirect` (default) writes the classic 2:1 set; `cube`
instead writes a 6-face cube map (`<map>_<face>.<ext>`, face size `width/4`
matched to the equator texel density) tiled exactly like the equirect path so
large faces still stream. Cube faces derive with the `PROJECTION_CUBE` variant
and — recorded LIMIT — OMIT synthesized detail (detail synthesis maps tile
pixels through an equirect lat/lon, so per-face it would tear at the cube
edges; the mask's `detail_gain` term, which reads the equirect tracer at the
correct direction, still applies) and omit the flow/rings maps (both use
equirect-space conventions). Two opt-in extra equirect maps ride alongside the
color/height set: `export.flow_map` writes `flow.exr` (the per-step velocity
resampled to the equirect grid as an east/north flow map — R eastward, G
northward — for motion vectors / advected compositor effects), and
`rings.enabled` writes `rings.exr` (a Saturn ring radial strip; a Blender-only
product feature, invisible in the GUI equirect preview — see
`docs/blender_addon.md`). Because both are separate files, enabling either
never touches the color/height/emission output (p05 render hash unaffected).

**Conditional manifest versioning.** The equirect path keeps
`schema_version 1` with each map's `file` key, so deployed importers never
warn. The cube path bumps to `schema_version 2` with `projection: "cube"` and
a per-map `faces` block; older importers that only build equirect geometry
reject it cleanly (`export/manifest.py`, `SCHEMA_VERSION` /
`SCHEMA_VERSION_CUBE`). Additive maps (flow, rings) stay on v1 — they are
extra keys a tolerant reader ignores, not a schema break.

**Imported-mask invariant.** When `mask.file` is set the derive pass binds the
imported paint mask (the `MASK` variant) whose `band_fade` / `emission_gain` /
`detail_gain` gains art-direct POST output; the mask is inert until at least
one gain is nonzero. The mask travels with each derive call (it is a
snapshot-resident texture read at the tile origin/full-size), so a tiled export
is bit-identical to a whole-frame derive — masks respect the same
tile-apron-free contract as the analytic passes.

**Program variants.** Optional shader features are preprocessor-gated and
cached per combination: derive.comp compiles per (EMISSION, CHROMA_FX, MASK)
plus the independent BAND_TINT (per-band RGB tint via
`appearance.band_tint_stops`/`_strength`), PROJECTION_CUBE (cube-face
sampling), and DETAIL_CHROMA (`appearance.detail_chroma` — L-preserving Oklab
material tint keyed to the synthesized detail excursion; oklab.glsl is shared
with CHROMA_FX through a single compound-guard include) variants; detail.comp
per (DETAIL_FX, SPREAD, HERO_EMERGENCE); the sim kernels per (HERO_EMERGENCE —
hero present and `storms.hero_emergence > 0`; FESTOON2 —
`waves.festoon_hero_strength > 0` and a facade-selected root edge near the
hero). Each new variant's
default is a no-op that preprocesses OUT (MASK only when a mask is bound and a
gain is nonzero; BAND_TINT only when `band_tint_strength > 0`; PROJECTION_CUBE
only on a cube export — guarded by the default-projection byte-identity test),
so a default equirect render is unchanged by construction. The DETAIL_FX variant
is selected whenever **any detail-FX lever is nonzero**; the lever set is
not hand-enumerated but derived from the `fx=True` pfield flag in
`params/model.py` (`render/detail.py::detail_fx_enabled`), which also
drives a build-time tripwire asserting every flagged lever's `u_<name>`
uniform exists in the compiled variant. A disabled feature preprocesses
OUT of the kernel text, so neutral-default output is identical by
construction rather than by hoping the compiler doesn't reschedule FP
around untaken branches; forced-variant no-op tests (epsilon parameter
values) pin each variant. The Color preview always derives with the
non-EMISSION variant — the viewport's Emission channel derives into its
own scratch textures (`engine/facade.py::ensure_preview_emission`), so
displaying emission never perturbs the Color path. The CHROMA_FX and
DETAIL_FX variants run in the GUI preview whenever their params are active
— they affect the displayed color.

**Detail synthesis** is advected-coordinate noise (positions backtraced
through the baked velocity for staggered pseudo-times, high-frequency sphere
noise sampled there → noise stretched and folded by the flow), masked by the
detail tracer and shear/speed, blended with Worley convective cells in quiet
zones. Poleward of 66–72° the backtrace routes through the polar patch
velocities (feather mixes noise values, never positions), so the caps carry
real texture instead of fading to neutral (v1.1).

**Hero wake braid** (`detail.hero_wake_braid`, default-off, POST tier, a
DETAIL_FX lever via the `fx=True` flag) inks the hero storm's turbulent wake at
render time as the reference GRS's chain of rolled billows (recumbent hairpin
folds) — brightening pale entrained tracer cores and darkening fold-boundary
rims — keyed to the sim's OWN advected tracer folds, not a synthetic strand
pattern (earlier synthetic-carrier revisions were rejected). The per-hero wake
frame (`wake_dir`, `wake_lat_off`) is threaded from the sim vortex registry
through the `engine/snapshot.py` hero-centers tuple — now 8 fields (`x, y, z,
r_core, spin, aspect, wake_dir, wake_lat_off`) — so the braid auto-tracks
wherever the flow puts the wake instead of assuming a fixed downstream
direction. Mode-agnostic (a POST render pass), but requires
`detail.intensity > 0`, a hero, and rides `storms.hero_wake_detail` (the
sim-side wake churn). Byte-identical when off; unbaked (no factory preset
enables it yet).

**Uniform detail coverage** (`detail.spread`, default-off, opt-in `SPREAD`
variant) applies the flow-folded detail-FX texture at EVEN density across
latitude instead of the band-gated LUT (belts textured, zones detail-starved).
`spread` is a single POST-tier level: `0` = today's band-gated look
(byte-identical, non-variant program); `>0` = every latitude gate
(belt/zone/mottle-window, the filament/cell weights) is interpolated toward the
uniform coverage level `u_spread` (pole-faded via `1 − routeW`), so there are no
detail-starved zones or stamped latitude bands. The backtrace sites still fold
the texture with the local flow, so it reads even *and* fluid, not flat noise.
Preprocessor-gated with verbatim `#else` arms ⇒ default output byte-identical
(p05 gate + a GPU-free no-defines projection hash guard both green). Character is
unchanged — this places *existing* flow-folded detail evenly; it does NOT advect
a tracer (distinct from the falsified frozen-field dye line and its parked
tracer-res redesign). *(An earlier strain-driven placement engine — an activity
pass computing local strain to concentrate texture on shear/rims — was built and
then dropped in calibration: strain-*selective* density read patchy; even
coverage won. See `docs/roadmap.md`.)*

## Checkpoints

`engine/checkpoint.py` saves a compressed .npz: the generating preset,
`generation_version` (the `GENERATION_VERSION` constant in
`engine/checkpoint.py` — cited by reference because it bumps whenever the
generation algorithms change), step counters (including the VELOCITY-edit
adaptation window), the three tracer textures, AND the vortex registry as
per-field float64 arrays + outbreak links — serialized, not replayed,
because live registry evolution (events, mergers) is not a pure function of
(seed, step) once mid-run edits enter. Loading rebuilds the sim from the
preset (bands/profiles/jets are seed-deterministic), overwrites tracers, and
swaps in the saved registry. A `generation_version` mismatch is refused
loudly: stale tracers would pair with differently-generated state. The live
step path advances the registry through one shared function
(`sim/advance.py: advance_registry` — events, drift, mergers) so any future
registry evolution lands in exactly one place.

Checkpoints are wired end to end: `gasgiant checkpoint` develops a run and
saves the .npz; `gasgiant export --resume <ckpt.npz>` reloads it (optionally
`--frames …` to export a sequence from the resumed state), and the GUI has
Save/Load state. A `generation_version` mismatch or corrupt file surfaces as a
clear CLI error rather than a traceback. GPU teardown goes through one path,
`Simulation.release()` — the public method the CLI, GUI, and `run_sheet` all
call to free textures/SSBOs/programs without leaking a context between runs.
This is the foundation for the animation exporter (restore → step k → export
per frame).

## GUI

imgui-bundle's native hello_imgui runner owns the window, the GL 4.3 core
context, and event handling (version-matched to imgui by construction — a
deliberate change from the original moderngl-window plan after its
integration broke against imgui-bundle 1.92). ModernGL attaches in
post_init; all sim GL work happens in the gui callbacks on the single GL
thread; any pass that binds an offscreen FBO rebinds the default framebuffer
before returning (the imgui backend renders into whatever is bound).
Parameter panels are auto-generated from pydantic field metadata.

## Determinism

One master seed; every stochastic subsystem draws from a named
`SeedSequence` substream (`params/seeds.py`), so changing storm parameters
never reshuffles the bands. No atomics or order-dependent GPU reductions.
How far "same seed → same bytes" goes is **solver-mode-dependent**:

- **Kinematic path: byte-exact** on the same machine/driver. Pinned by a
  kernel source-hash test (`tests/unit/test_kinematic_kernels_pinned.py`)
  and a float32 render-hash gate (`scripts/p05_baseline_hash.py --check`,
  machine-local baseline). A kinematic hash mismatch means the output
  actually moved — update the pin deliberately, never paper over it with a
  tolerance.
- **Vorticity path: deterministic within documented noise floors, NOT
  byte-exact.** Its red-black SOR Poisson solve carries ~1e-3
  cross-instance LSB noise (`_VORT_SOR_ATOL = 1e-3`,
  `tests/gpu/test_checkpoint.py`) and ~0.004 cross-session GL-context
  noise on real GPUs (asserted within `GPU_NOISE_ATOL = 1e-2` in the GPU
  tests; llvmpipe in CI is stable). Never write a byte-exact assertion
  against vorticity-mode output.

## Testing

- Unit (no GPU): params/presets/migrations, randomize determinism, palette,
  manifest contract round-trip (including the vendored Blender reader),
  seam validators on synthetic data.
- GPU (tiny resolutions; llvmpipe in CI, RTX locally): kernel smoke,
  washout/blow-up guards, zonal-transport pattern preservation, polar
  exchange RMS, formation signatures (festoons/outbreaks/ribbon/detail),
  tiled-export seams, snapshot isolation, cancellation safety.
- Blender: `tests/blender/test_import.py` runs inside
  `blender --background` (17 scene/material assertions), verified against
  Blender 5.1.2 plus a real headless Cycles render.

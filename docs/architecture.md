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

There is no feedback from tracers to velocity, so the solver is
unconditionally stable: the only failure modes are visual (washout — guarded
by a variance test over 300 steps) and they are tested.

## Invalidation tiers

Every parameter declares its tier in field metadata; the engine diffs
validated parameter trees and dispatches the cheapest sufficient recompute:

| Tier | Examples | Action |
|---|---|---|
| POST | haze, palette, contrast, detail synthesis | re-derive maps (instant) |
| VELOCITY | jet strength, turbulence amplitude | rebuild profiles/ψ; run continues (+adaptation steps if finished) |
| RESTART | seed, band layout, storms, poles | rebuild everything, development run restarts (the GUI shows it evolving) |

## Export

`export/exporter.py` is a generator: finish development (yielding progress),
capture an **ExportSnapshot** (GPU-side clones of tracers/velocity/LUTs +
deep param copy — tiles can never disagree from mid-export edits), render
1024² tiles (the derive and detail kernels take an origin/full-size, and read
only sim-resolution snapshot textures + analytic noise, so tiles need no
apron), assemble per map directly in final dtype on the CPU, then encode in
a thread pool. The GUI runs one slice per frame with progress + cancel
(cancellation removes only the files we wrote); the CLI drains the same
generator. Measured: 16384×8192 color+height in ~32 s on an RTX 3070
(encode-bound).

**Detail synthesis** is advected-coordinate noise (positions backtraced
through the baked velocity for staggered pseudo-times, high-frequency sphere
noise sampled there → noise stretched and folded by the flow), masked by the
detail tracer and shear/speed, blended with Worley convective cells in quiet
zones. Known limitation: the detail pass currently fades to neutral poleward
of ~56° (it backtraces through the equirect velocity only); routing polar
tiles through the patch velocities is future work.

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
never reshuffles the bands. No atomics or order-dependent GPU reductions:
same seed → identical exports on the same machine/driver (tested).

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

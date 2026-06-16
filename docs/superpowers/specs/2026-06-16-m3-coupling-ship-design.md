# M3 — Ship the baroclinic coupling into the render path (design)

**Date:** 2026-06-16
**Status:** design, adversarially reviewed (2 independent reviewers), ready for plan
**Predecessors:** `2026-06-16-m3-coupling-design.md` (built the coupling), `2026-06-16-m3-coupling-verdict.md` (visual win @ gain≈0.5, coherent gate fails), `2026-06-16-m3-qtarget-falsified.md` (no injection lever beats intrinsic advection — coherent-dominant is unreachable, so we ship the visual win).

## Goal

Make the validated baroclinic→v1.6 coupling reachable from the production render
path behind an opt-in param that is **byte-identical when off**, shipped as a
factory preset at the validated operating point (gain≈0.5). Today the coupling
exists end-to-end (`BaroclinicSourceDriver`, `run_coupled`, the
`set_external_vorticity_source` hook) but is only invoked from `scripts/sw_m3_couple.py`.

This is an INTEGRATION of validated components, not new physics. We do NOT chase
the coherent-dominant gate (proven unreachable); the coupling ships as the visual
fidelity feature it is — bands plus physically-grounded mid-latitude storm
roll-ups — with the automated coherent metric recorded as a known-broadband diagnostic.

## Architecture

One stepping path. The interleave (advance baroclinic driver → re-derive coherent
source → re-upload → step v1.6) folds into `Simulation.tick()`, so **both the live
preview and the export develop through the same coupled path** — no preview/export
mismatch. When no driver is present, `tick()` is unchanged.

### Components

1. **`BaroclinicParams(_Params)`** nested on `SolverParams` (`src/gasgiant/params/model.py`):
   - `enabled: bool = False` — `Tier.RESTART`, `rand=None`.
   - `gain: float = 0.5` — `Tier.RESTART`, `lo=0.0, hi=2.0`, `rand=None`, UI-exposed (the one legitimately taste-dependent knob; the validated operating point).
   - `warmup_steps: int = 8000`, `baro_steps_per_update: int = 150`, `update_every: int = 32` — fixed cadence proven to stay in the pre-outcrop window; non-UI (`ui=""`), `rand=None`. Present as params for testability/override, NOT slider tunables (YAGNI: not a research panel).
   - Default-factory so presets lacking the block load cleanly.
2. **Validator on `SolverParams`** (`mode="after"`, beside `_validate_sor_omega`): if
   `baroclinic.enabled` then `type` must be `SolverType.VORTICITY` (the external hook
   lives only in the vorticity kernel's equirect pass). The nested model cannot see
   `self.type`, so the validator MUST be on the parent.
3. **`Simulation` integration** (`src/gasgiant/engine/facade.py`):
   - `_build()`: when `p.solver.baroclinic.enabled`, construct/reuse the driver,
     seeded from `p.seed`; init `_baro_next_update = 0`. Else `_baro_driver = None`.
   - `tick()`: cadence-clamped interleave (below).
   - `_release_sim()`: drop the driver ref (external tex already released there).
4. **Factory preset** `src/gasgiant/presets/jupiter_baroclinic.json`: `jupiter_vorticity`
   base with `solver.baroclinic.enabled=true, gain=0.5`. THIS is the shipped artifact.

### Cadence-clamped tick (the preview==export fix)

```python
def tick(self, max_steps: int = 2) -> bool:
    remaining = self.steps_target - self.solver.step_index
    if remaining <= 0:
        return False
    if self._baro_driver is not None:
        if self.solver.step_index >= self._baro_next_update:
            self._update_baroclinic_source()         # advance + re-derive + upload (guarded)
            self._baro_next_update += self._baro_update_every
        # never let a chunk straddle the next source-update boundary
        to_boundary = self._baro_next_update - self.solver.step_index
        n = min(max_steps, remaining, to_boundary)
    else:
        n = min(max_steps, remaining)
    self.solver.step(n)
    self._tracers_changed = True
    return True
```

The source is updated at exactly `step_index ∈ {0, update_every, 2·update_every, …}`
and a step chunk never crosses a boundary, so `run_to_completion(64)` and repeated
`tick(2)` produce **bit-identical** developed state. (Cost: `run_to_completion`
issues more, smaller `solver.step` calls — accepted.)

### Determinism / seed contract

With `baroclinic.enabled=True`, `Simulation(params)` is deterministic in
`params.seed`: the driver seed is `params.seed` (matching the engine convention where
`generate_bands`/`build_profiles`/`generate_vortices`/`select_lanes` all take raw
`p.seed`; the baroclinic RNG draws are an independent stream so no aliasing). Same
`PlanetParams` → same coupled render on a given GPU/driver (the existing per-machine
float32 caveat from P0.5 still applies). Driver advance is by fixed integer step
counts → wall-clock-independent.

### Failure modes — degrade, never crash

- **Warmup outcrop** (`BaroclinicSourceDriver.__init__` raises `RuntimeError`): `_build`
  catches it, sets `_baro_driver = None`, logs a loud warning. The sim constructs and
  renders **byte-identically to the uncoupled run**. An enabled preset must never
  crash `Simulation(params)`.
- **Mid-run incoherence** (`current_source()` → `assert_coherent` raises, or driver
  outcrops mid-run): `_update_baroclinic_source` catches it, disables further updates
  (latch the driver off; retain the last good uploaded source or drop to gain 0), logs,
  and development continues. No exception escapes `tick()`.

### Driver caching

To avoid re-running the 8000-step warmup on unrelated `Tier.RESTART` edits, cache the
driver on the key `(equirect_w, equirect_h, warmup_steps, seed)`. `_build` reuses the
cached driver when the key is unchanged; rebuilds (re-warms) only when it changes.
Driver grid comes from `self.solver.equirect.size` (read after the solver is built).

### Tiled export (confirmed safe, no change)

Export does not re-step the solver: `render_maps` calls `run_to_completion()` then
`_derive` reads developed `tracers.cur`; `ExportSnapshot.capture` clones the developed
tracers. The external source is consumed only inside `solver.step()`. So the coupling
**bakes into the developed tracer field**; the snapshot captures it correctly and does
NOT need the driver live during tile derivation. The snapshot does not carry
`external_omega_tex`.

### Checkpoint policy

Old presets (no baroclinic block) load via the pydantic default — no migration. The
P0.5 hash baselines (`default`, `jupiter_like`) are kinematic and never enable
baroclinic → no rebaseline. The driver state is NOT serialized: a checkpoint of a
coupled run is **render-only** — resuming re-warms the driver from `params.seed` rather
than restoring mid-run driver state (acceptable for the still-image pipeline; stated,
not silently assumed).

## Testing

1. **Off-path byte-identity (load-bearing):** `baroclinic.enabled=False` → `render_maps`
   byte-identical to a baseline build; the new `SolverParams` field's mere presence does
   not perturb default output. Existing `test_off_path_byte_identical` + P0.5 stay green.
2. **Preview==export determinism:** develop one sim via `run_to_completion()` and an
   identical sim via repeated `tick(2)` → bit-identical developed tracers / render.
3. **Seed determinism:** same enabled params + same `seed` → identical render; different
   `seed` → different render (proves the seed is plumbed, not hardcoded).
4. **Validator:** `baroclinic.enabled=True` with kinematic solver → `ValidationError`.
5. **Old preset loads:** a preset JSON without the baroclinic block validates and runs.
6. **Graceful outcrop:** an enabled config forced to outcrop during warmup → sim
   constructs and renders byte-identically to the uncoupled run (no `RuntimeError`).
7. **Driver cache:** an unrelated `Tier.RESTART` edit does not re-run warmup when
   grid/seed/warmup_steps are unchanged.
8. **Factory preset smoke:** `load_factory_preset("jupiter_baroclinic")` builds and
   renders without error; driver is built; render differs from the uncoupled baseline.

GPU tests reuse the session-scoped `gpu` fixture, `pytestmark = pytest.mark.gpu`,
resolution ≥ 512.

## Scope (YAGNI)

IN: the param block, the validator, the seeded cadence-clamped `tick` integration,
graceful degradation, driver caching, the factory preset, the tests above.

OUT (explicit non-goals): GUI control panel / per-jet tuning; chasing the coherent
metric; live-preview perf optimization beyond the cadence clamp; a resident-GPU
baroclinic solver (option-(a) CPU advance + re-upload is sufficient per the measured
residency rule). The coherent gate stays a documented diagnostic, not a release gate.

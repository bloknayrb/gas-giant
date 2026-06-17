# Ship Baroclinic Coupling Into the Render Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the validated baroclinic→v1.6 coupling reachable from the production render path behind an opt-in param that is byte-identical when off, shipped as a factory preset at gain≈0.5.

**Architecture:** A nested `BaroclinicParams` on `SolverParams` (off by default). When enabled, `Simulation` owns a seeded `BaroclinicSourceDriver` (built with graceful outcrop-degradation + caching) and a cadence-clamped `tick()` interleaves the evolving source into the vorticity solver so live preview and export develop through one identical path.

**Tech Stack:** Python 3.13 + pydantic params; ModernGL vorticity solver; pytest with the session-scoped `gpu` fixture.

**Spec:** `docs/superpowers/specs/2026-06-16-m3-coupling-ship-design.md`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/gasgiant/params/model.py` | `BaroclinicParams` + nest on `SolverParams` + parent validator | Modify |
| `src/gasgiant/presets/jupiter_baroclinic.json` | Factory preset enabling the coupling (the shipped artifact) | Create |
| `src/gasgiant/engine/facade.py` | Seeded driver lifecycle (build/degrade/cache) + cadence-clamped `tick` | Modify |
| `tests/unit/test_baroclinic_params.py` | Validator + preset-load CPU tests | Create |
| `tests/gpu/test_m3_ship.py` | off byte-identity, preview==export, seed determinism, smoke | Create |

---

## Cross-cutting conventions

- TDD: failing test first, minimal code, green, commit.
- `_Params` base is `extra="forbid"` + `validate_assignment=True` — assignments re-validate, so the parent validator runs on `p.solver.type = ...` too.
- GPU tests: `pytestmark = pytest.mark.gpu`, reuse the `gpu` fixture, resolution ≥ 512, keep `dev_steps` and `warmup_steps` SMALL for speed.
- Driver and cadence are CPU-deterministic by integer step counts (no wall-clock).

---

### Task 1: BaroclinicParams + parent validator

**Files:**
- Modify: `src/gasgiant/params/model.py` (add `BaroclinicParams` after `SolverType` ~line 490; nest on `SolverParams` ~line 519; add validator beside `_validate_sor_omega` ~line 521)
- Test: `tests/unit/test_baroclinic_params.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_baroclinic_params.py
"""BaroclinicParams: defaults off, validator requires vorticity, presets load."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset


def test_baroclinic_defaults_off():
    p = PlanetParams()
    assert p.solver.baroclinic.enabled is False
    assert p.solver.baroclinic.gain == 0.5


def test_enabled_requires_vorticity():
    # kinematic (default) + enabled must be rejected
    with pytest.raises(ValidationError):
        PlanetParams.model_validate(
            {"solver": {"type": "kinematic", "baroclinic": {"enabled": True}}}
        )


def test_enabled_with_vorticity_ok():
    p = PlanetParams.model_validate(
        {"solver": {"type": "vorticity", "baroclinic": {"enabled": True}}}
    )
    assert p.solver.baroclinic.enabled is True


def test_existing_preset_without_block_loads():
    # jupiter_vorticity.json has no baroclinic block -> pydantic default, off
    p = load_factory_preset("jupiter_vorticity")
    assert p.solver.baroclinic.enabled is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_baroclinic_params.py -q`
Expected: FAIL (`AttributeError`/`ValidationError`: `baroclinic` field does not exist).

- [ ] **Step 3: Implement — add `BaroclinicParams` after the `SolverType` enum**

In `src/gasgiant/params/model.py`, immediately after the `class SolverType(StrEnum)` block (before `class SolverParams`):

```python
class BaroclinicParams(_Params):
    """Opt-in 2-layer baroclinic vorticity source coupled into the vorticity
    solver's equirect pass (M3). OFF by default => byte-identical to plain v1.6.
    Ships the validated visual operating point (gain=0.5: bands + physically-
    grounded mid-latitude storms). The cadence fields are fixed (non-UI): they
    keep the baroclinic CPU solver in its healthy pre-outcrop window."""

    enabled: bool = pfield(
        False, tier=Tier.RESTART, ui="Solver",
        description="Inject the evolving baroclinic vorticity source into the "
                    "vorticity solver (adds physically-grounded mid-latitude "
                    "storms; requires solver type=vorticity). Off = plain v1.6. "
                    "No rand: randomize() must never silently enable it.")
    gain: float = pfield(
        0.5, tier=Tier.RESTART, lo=0.0, hi=2.0, ui="Solver",
        description="Baroclinic source amplitude as a fraction of coriolis_f0 "
                    "(~3). 0.5 = validated operating point. No rand.")
    warmup_steps: int = pfield(
        8000, tier=Tier.RESTART, lo=500, hi=20000, ui="",
        description="Baroclinic spin-up before coupling (fixed cadence). No rand. "
                    "hi=20000 leaves headroom past the ~12500 outcrop so a forced "
                    "outcrop can be exercised by tests.")
    baro_steps_per_update: int = pfield(
        150, tier=Tier.RESTART, lo=10, hi=1000, ui="",
        description="Baroclinic steps per source refresh (fixed cadence). No rand.")
    update_every: int = pfield(
        32, tier=Tier.RESTART, lo=1, hi=512, ui="",
        description="v1.6 steps between source refreshes (fixed cadence). No rand.")
```

- [ ] **Step 4: Implement — nest the block on `SolverParams` and add the validator**

In `class SolverParams`, add the field after `vort_drag` (~line 519):

```python
    baroclinic: BaroclinicParams = Field(default_factory=BaroclinicParams)
```

And add a validator after the existing `_validate_sor_omega` (~line 527):

```python
    @model_validator(mode="after")
    def _validate_baroclinic(self) -> SolverParams:
        if self.baroclinic.enabled and self.type != SolverType.VORTICITY:
            raise ValueError(
                f"baroclinic.enabled requires solver type=vorticity "
                f"(got {self.type})"
            )
        return self
```

(`Field` and `model_validator` are already imported in this module.)

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/unit/test_baroclinic_params.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Regression — params suite still green**

Run: `python -m pytest tests/unit/test_model.py tests/unit/test_presets.py -q` (run whichever exist)
Expected: PASS (no existing param tests broken by the new default-off field).

- [ ] **Step 7: Commit**

```bash
git add src/gasgiant/params/model.py tests/unit/test_baroclinic_params.py
git commit -m "M3: BaroclinicParams (opt-in, off by default) + vorticity validator"
```

---

### Task 2: Factory preset `jupiter_baroclinic.json`

**Files:**
- Create: `src/gasgiant/presets/jupiter_baroclinic.json`
- Test: append to `tests/unit/test_baroclinic_params.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_jupiter_baroclinic_preset_enables_coupling():
    p = load_factory_preset("jupiter_baroclinic")
    assert p.solver.type == SolverType.VORTICITY
    assert p.solver.baroclinic.enabled is True
    assert p.solver.baroclinic.gain == 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_baroclinic_params.py::test_jupiter_baroclinic_preset_enables_coupling -q`
Expected: FAIL (`PresetError: unknown factory preset 'jupiter_baroclinic'`).

- [ ] **Step 3: Implement — create the preset**

Create `src/gasgiant/presets/jupiter_baroclinic.json` as a copy of `jupiter_vorticity.json` with two changes: (1) both `name` fields → `"jupiter_baroclinic"`, (2) add the `baroclinic` block inside `"solver"`. The solver block becomes:

```json
    "solver": {
      "type": "vorticity",
      "poisson_iters": 48,
      "sor_omega": 1.7,
      "vort_relax_tau": 600.0,
      "vort_hypervisc": 0.6,
      "coriolis_f0": 3.0,
      "vort_inject": 0.0,
      "vort_inject_scale": 0.5,
      "vort_drag": 0.0,
      "baroclinic": { "enabled": true, "gain": 0.5 }
    },
```

Copy ALL other blocks (`bands`, `jets`, `turbulence`, `storms`, `waves`, `poles`, `appearance`, `detail`, `emission`, `physical`, `export`) VERBATIM from `jupiter_vorticity.json`, and keep `"preset_format": 2`, `"app_version": "0.1.0"`. Set top-level `"name": "jupiter_baroclinic"` and `params.name: "jupiter_baroclinic"`, `params.seed: 4201`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_baroclinic_params.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gasgiant/presets/jupiter_baroclinic.json tests/unit/test_baroclinic_params.py
git commit -m "M3: jupiter_baroclinic factory preset (coupling on, gain 0.5)"
```

---

### Task 3: Simulation driver lifecycle (seeded build, graceful degrade, cache)

**Files:**
- Modify: `src/gasgiant/engine/facade.py` (`__init__` ~line 41-52; `_build` ~line 56-80; add `_init_baroclinic`)
- Test: `tests/gpu/test_m3_ship.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/gpu/test_m3_ship.py
"""Ship-path: driver lifecycle, off byte-identity, preview==export, seed determinism."""
from __future__ import annotations

import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu


def _baro_params(seed: int = 7, dev_steps: int = 40, warmup: int = 600,
                 enabled: bool = True) -> PlanetParams:
    p = load_factory_preset("jupiter_vorticity")
    p = p.model_copy(update={"seed": seed})
    p.sim.resolution = 512
    p.sim.dev_steps = dev_steps
    p.solver.type = SolverType.VORTICITY
    p.solver.baroclinic = p.solver.baroclinic.model_copy(update={
        "enabled": enabled, "gain": 0.5, "warmup_steps": warmup,
        "baro_steps_per_update": 60, "update_every": 16,
    })
    return p


def test_enabled_builds_driver(gpu):
    sim = Simulation(_baro_params(), gpu)
    try:
        assert sim._baro_driver is not None
    finally:
        sim._release_sim()


def test_disabled_has_no_driver(gpu):
    sim = Simulation(_baro_params(enabled=False), gpu)
    try:
        assert sim._baro_driver is None
    finally:
        sim._release_sim()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/gpu/test_m3_ship.py -q`
Expected: FAIL (`AttributeError: 'Simulation' object has no attribute '_baro_driver'`).

- [ ] **Step 3: Implement — `__init__` attrs**

In `Simulation.__init__` (`src/gasgiant/engine/facade.py`), before `self._build()` (line 52), add:

```python
        self._baro_driver = None
        self._baro_key: tuple | None = None
        self._baro_next_update = 0
        self._baro_update_every = 0
        self._baro_gain = 0.0
        self._baro_steps_per_update = 0
```

- [ ] **Step 4: Implement — `_init_baroclinic` and call it from `_build`**

In `_build`, after `self.solver.init_tracers()` (line 77), add `self._init_baroclinic()`. Then add the method (after `_release_sim`):

```python
    def _init_baroclinic(self) -> None:
        """Build/reuse the baroclinic source driver when enabled. Caches on
        (grid, warmup, seed) so unrelated RESTART edits don't re-warm. On warmup
        outcrop, degrade to uncoupled (driver=None) -- never crash construction."""
        bp = self.params.solver.baroclinic
        self._baro_next_update = 0
        self._baro_update_every = bp.update_every
        self._baro_gain = bp.gain
        self._baro_steps_per_update = bp.baro_steps_per_update
        if not bp.enabled:
            self._baro_driver = None
            self._baro_key = None
            return
        w, h = self.solver.equirect.size
        key = (w, h, bp.warmup_steps, self.params.seed)
        if self._baro_driver is not None and self._baro_key == key:
            return  # reuse cached driver (no re-warmup)
        from gasgiant.sim.baroclinic_driver import BaroclinicSourceDriver
        try:
            self._baro_driver = BaroclinicSourceDriver(
                grid_w=w, grid_h=h, warmup_steps=bp.warmup_steps,
                seed=self.params.seed)
            self._baro_key = key
        except RuntimeError as exc:
            log.warning("baroclinic coupling disabled: warmup outcropped (%s)", exc)
            self._baro_driver = None
            self._baro_key = None
```

(`log` is the module logger already defined at `facade.py:34`. Do NOT null `self._baro_driver` in `_release_sim` — the driver is CPU-only and the cache must survive a RESTART rebuild.)

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/gpu/test_m3_ship.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/gasgiant/engine/facade.py tests/gpu/test_m3_ship.py
git commit -m "M3: Simulation baroclinic driver lifecycle (seeded, cached, degrade-safe)"
```

---

### Task 4: Cadence-clamped tick + source update (preview==export)

**Files:**
- Modify: `src/gasgiant/engine/facade.py` (`tick` ~line 167-175; add `_update_baroclinic_source`)
- Test: append to `tests/gpu/test_m3_ship.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def _dev_render_bytes(p: PlanetParams, gpu) -> bytes:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(512)["color"].tobytes()
    finally:
        sim._release_sim()


def test_off_path_byte_identical(gpu):
    """baroclinic.enabled=False renders byte-identically to plain vorticity."""
    base = _dev_render_bytes(_baro_params(enabled=False), gpu)
    # a params object that never had the field touched, same seed/steps:
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = 40
    p.solver.type = SolverType.VORTICITY
    plain = _dev_render_bytes(p, gpu)
    assert base == plain, "default-off baroclinic must not change the render"


def test_preview_equals_export(gpu):
    """Developing via many small tick(2) calls must match one run_to_completion."""
    p = _baro_params(dev_steps=48)
    export = Simulation(p, gpu)
    try:
        export.run_to_completion(chunk=64)
        export_bytes = export.render_maps(512)["color"].tobytes()
    finally:
        export._release_sim()

    preview = Simulation(_baro_params(dev_steps=48), gpu)
    try:
        while preview.tick(2):
            pass
        preview_bytes = preview.render_maps(512)["color"].tobytes()
    finally:
        preview._release_sim()

    assert preview_bytes == export_bytes, "preview chunking must equal export"


def test_enabled_changes_render(gpu):
    on = _dev_render_bytes(_baro_params(enabled=True), gpu)
    off = _dev_render_bytes(_baro_params(enabled=False), gpu)
    assert on != off, "enabled coupling must change the render"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/gpu/test_m3_ship.py -q`
Expected: FAIL — `test_preview_equals_export` and `test_enabled_changes_render` fail (tick does not yet drive the source; on==off).

- [ ] **Step 3: Implement — `_update_baroclinic_source`**

Add to `Simulation` (after `_init_baroclinic`):

```python
    def _update_baroclinic_source(self) -> None:
        """Advance the baroclinic solver and re-upload the coherent source. On
        mid-run incoherence/outcrop, degrade to uncoupled and continue."""
        try:
            self._baro_driver.advance(self._baro_steps_per_update)
            src = self._baro_driver.current_source()
        except (ValueError, RuntimeError) as exc:
            log.warning("baroclinic source disabled mid-run: %s", exc)
            self.set_external_vorticity_source(None)
            self._baro_driver = None
            return
        self.set_external_vorticity_source(src, gain=self._baro_gain)
```

- [ ] **Step 4: Implement — cadence-clamped `tick`**

Replace the body of `tick` (`facade.py:167-175`) with:

```python
    def tick(self, max_steps: int = 2) -> bool:
        """Advance up to max_steps of the development run. Returns True if the
        sim stepped (callers re-derive the preview). When baroclinic coupling is
        active, the source is refreshed at fixed step_index boundaries and a step
        chunk never straddles a boundary -- so preview (small chunks) and export
        (large chunks) develop bit-identically."""
        remaining = self.steps_target - self.solver.step_index
        if remaining <= 0:
            return False
        if self._baro_driver is not None:
            if self.solver.step_index >= self._baro_next_update:
                self._update_baroclinic_source()
                self._baro_next_update += self._baro_update_every
            n = min(max_steps, remaining,
                    self._baro_next_update - self.solver.step_index)
        else:
            n = min(max_steps, remaining)
        self.solver.step(n)
        self._tracers_changed = True
        return True
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/gpu/test_m3_ship.py -q`
Expected: PASS (off byte-identical, preview==export, enabled changes render all green).

- [ ] **Step 6: Commit**

```bash
git add src/gasgiant/engine/facade.py tests/gpu/test_m3_ship.py
git commit -m "M3: cadence-clamped tick drives baroclinic source (preview==export)"
```

---

### Task 5: Seed determinism, graceful-outcrop, factory smoke

**Files:**
- Test: append to `tests/gpu/test_m3_ship.py`

- [ ] **Step 1: Write the failing/validating tests** (append)

```python
def test_seed_determinism(gpu):
    """Same enabled params + same seed -> identical; different seed -> different."""
    a = _dev_render_bytes(_baro_params(seed=11), gpu)
    a2 = _dev_render_bytes(_baro_params(seed=11), gpu)
    b = _dev_render_bytes(_baro_params(seed=12), gpu)
    assert a == a2, "same seed must reproduce the coupled render"
    assert a != b, "different seed must change the storm pattern (seed is plumbed)"


def test_graceful_warmup_outcrop(gpu):
    """A warmup past the ~12500 outcrop must degrade to uncoupled (driver=None),
    NOT crash construction, and render byte-identically to the uncoupled run."""
    p = _baro_params()
    p.solver.baroclinic = p.solver.baroclinic.model_copy(
        update={"warmup_steps": 15000})  # deterministically past outcrop (~12500)
    sim = Simulation(p, gpu)  # must NOT raise
    try:
        assert sim._baro_driver is None, "warmup outcrop must degrade to uncoupled"
        outcropped_render = sim.render_maps(512)["color"].tobytes()
    finally:
        sim._release_sim()

    base = _dev_render_bytes(_baro_params(enabled=False), gpu)
    assert outcropped_render == base, "degraded run must equal the uncoupled render"


def test_factory_preset_smoke(gpu):
    """jupiter_baroclinic builds, develops, renders; differs from uncoupled base."""
    p = load_factory_preset("jupiter_baroclinic")
    p.sim.resolution = 512
    p.sim.dev_steps = 40
    p.solver.baroclinic = p.solver.baroclinic.model_copy(
        update={"warmup_steps": 600, "baro_steps_per_update": 60, "update_every": 16})
    sim = Simulation(p, gpu)
    try:
        assert sim._baro_driver is not None
        coupled = sim.render_maps(512)["color"].tobytes()
    finally:
        sim._release_sim()

    p2 = p.model_copy()
    p2.solver.type = SolverType.VORTICITY
    p2.solver.baroclinic = p2.solver.baroclinic.model_copy(update={"enabled": False})
    base = _dev_render_bytes(p2, gpu)
    assert coupled != base
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/gpu/test_m3_ship.py -q`
Expected: PASS. If `test_graceful_warmup_outcrop` does NOT outcrop at 12000 (state still healthy), it still passes (driver non-None is fine); the assertion only requires no crash. If it DOES outcrop, `_init_baroclinic` catches the `RuntimeError` and sets driver None — also passing.

- [ ] **Step 3: Fix if red.** If construction raises instead of degrading, the `try/except RuntimeError` in `_init_baroclinic` (Task 3) is wrong — verify it wraps the `BaroclinicSourceDriver(...)` call. No new production code should be needed if Tasks 3-4 are correct.

- [ ] **Step 4: Full regression**

Run: `python -m pytest tests/gpu/test_m3_external_source.py tests/gpu/test_m3_ship.py tests/unit/test_baroclinic_params.py -q`
Expected: PASS (the original hook tests + all new ship tests).

- [ ] **Step 5: Commit**

```bash
git add tests/gpu/test_m3_ship.py
git commit -m "M3: ship-path validation (seed determinism, graceful outcrop, smoke)"
```

---

## Validation (end-to-end)

- Unit: `python -m pytest tests/unit/test_baroclinic_params.py -q`
- GPU: `python -m pytest tests/gpu/test_m3_ship.py tests/gpu/test_m3_external_source.py -q`
- Off-path discipline: the P0.5 hash baseline (`default`, `jupiter_like` — both kinematic) is unaffected; run its check if present: `python scripts/p05_baseline_hash.py --check`.
- Manual render (optional): `python -m pytest -k factory_preset_smoke` produces a coupled render; for a visual, run `scripts/sw_m3_couple.py 2048 0.5` (existing gate, unchanged).

## After all tasks

Final whole-implementation review (Opus), then `superpowers:finishing-a-development-branch`. Update memory `m3-render-laminar-rootcause.md` with the SHIPPED status.

## Notes / resolved risks

- **Off-path byte-identity:** default-off nested model compares equal in `diff_tiers`; driver never built; `external_gain` stays 0 (solver.py:406-407). Locked by `test_off_path_byte_identical`.
- **Preview==export:** cadence clamp guarantees the source-update schedule and step boundaries are chunk-size-independent. Locked by `test_preview_equals_export`.
- **No crash on outcrop:** `_init_baroclinic` catches the warmup `RuntimeError`; `_update_baroclinic_source` catches mid-run `ValueError`/`RuntimeError`. Both degrade to uncoupled.
- **Seed contract:** driver seeded from `params.seed` (matches the engine's per-subsystem convention). Locked by `test_seed_determinism`.
- **Cache:** driver reused across RESTART rebuilds unless `(grid, warmup, seed)` changes; `_release_sim` does not drop it.
- **Tiled export:** export reads developed tracers; coupling bakes in; snapshot needs no driver (confirmed in review).

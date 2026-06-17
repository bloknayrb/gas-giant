# Address PR #1 Review Findings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Resolve the two actionable findings from the PR #1 code review on branch `v2-m3-ship`: (1) an E402 lint error introduced by the tolerance constant, and (2) an under-tested determinism subtlety in cached-driver reuse.

**Architecture:** Both fixes are small and local. Finding 1 is a one-line move. Finding 2 makes the cached baroclinic driver deterministic on reuse by snapshotting its post-warmup state and restoring it whenever a reused driver begins a new development run — closing an interactive-path reproducibility gap without re-running the 8000-step warmup.

**Tech Stack:** Python 3.13 + pydantic; ModernGL vorticity solver; pytest (`gpu` fixture); ruff (pinned 0.15.16, run via `uv run ruff`).

**Branch:** work continues on `v2-m3-ship` (the open PR); commits push to update the PR.

---

## Review findings → disposition

| # | Finding | Disposition |
|---|---|---|
| 1 | **E402** in `tests/gpu/test_m3_ship.py:19-21` — `GPU_NOISE_ATOL` sits above the `gasgiant` imports | **Fix** (Task 1) |
| 2 | Cached-driver reuse after a partial preview tick advances the baroclinic stream → developed image depends on prior tick count (by-design, under-tested) | **Fix properly + test** (Task 2) |
| 3 | Reviewer couldn't run GPU tests (no GL context) — rests on CI/local | **No code change** — GPU suite was run locally (21 passed); note in PR reply |
| — | Note: ~581 pre-existing ruff errors on `master` (lint baseline) | **Out of scope** — separate repo-hygiene task, not this PR (see "Out of scope" below) |

### Design note for Finding 2 (why the proper fix, not a comment)

The reviewer rated this "not a blocker — a one-line comment or follow-up test." We take the stronger fix because the project's whole discipline is determinism (P0.5 byte-exact gates), and the wart lives exactly in the interactive tuning path: tweak a RESTART-tier slider after watching the preview develop, and the storms shift based on how long you watched — non-reproducible. The fix is ~10 lines and removes the wart entirely while keeping the cache's benefit (no re-warmup). The lighter alternative (document-only) is rejected: it leaves a real reproducibility hole in the path users actually interact with.

Export and fresh `Simulation(params)` are already deterministic (new driver, no prior ticks); the fix only affects the in-process reuse path (a `Tier.RESTART` edit on a live sim).

---

### Task 1: Fix the E402 lint error

**Files:**
- Modify: `tests/gpu/test_m3_ship.py` (move `GPU_NOISE_ATOL` from line 17 to below all imports)

- [ ] **Step 1: Confirm the failure**

Run: `uv run ruff check tests/gpu/test_m3_ship.py`
Expected: 3× `E402 Module level import not at top of file` at lines 19-21.

- [ ] **Step 2: Move the constant below the imports**

In `tests/gpu/test_m3_ship.py`, delete the `GPU_NOISE_ATOL` line from its current position (line 17, between `import pytest` and the `from gasgiant...` imports):

```python
GPU_NOISE_ATOL = 1e-2  # > measured ~0.004 session LSB noise, << ~0.69 coupling signal
```

and re-add it immediately AFTER the `pytestmark = pytest.mark.gpu` line, so the top of the file reads:

```python
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu

GPU_NOISE_ATOL = 1e-2  # > measured ~0.004 session LSB noise, << ~0.69 coupling signal
```

- [ ] **Step 3: Verify lint is clean**

Run: `uv run ruff check tests/gpu/test_m3_ship.py`
Expected: `All checks passed!` (no E402).

- [ ] **Step 4: Verify tests still import/collect**

Run: `py -3 -m pytest tests/gpu/test_m3_ship.py --collect-only -q`
Expected: 12 tests collected, no import error.

- [ ] **Step 5: Commit**

```bash
git add tests/gpu/test_m3_ship.py
git commit -m "M3: fix E402 — move GPU_NOISE_ATOL below imports (PR review finding 1)"
```

---

### Task 2: Deterministic cached-driver reuse

**Files:**
- Modify: `src/gasgiant/sim/baroclinic_driver.py` (snapshot post-warmup state in `__init__`; add `reset()`)
- Modify: `src/gasgiant/engine/facade.py` (`_init_baroclinic`: call `reset()` on cache hit)
- Test: `tests/unit/test_baroclinic_driver.py` (CPU reset unit test)
- Test: `tests/gpu/test_m3_ship.py` (GPU reuse-determinism test)

- [ ] **Step 1: Write the failing CPU unit test**

Append to `tests/unit/test_baroclinic_driver.py`:

```python
def test_reset_restores_warm_state():
    """reset() must return the driver to its post-warmup state so every dev run
    starts identically (deterministic cache reuse)."""
    import numpy as np
    from gasgiant.sim.baroclinic_driver import BaroclinicSourceDriver

    d = BaroclinicSourceDriver(grid_w=64, grid_h=32, warmup_steps=600, seed=0)
    s0 = d.current_source()
    d.advance(300)
    assert not np.allclose(s0, d.current_source()), "advance must change the source"
    d.reset()
    assert np.allclose(s0, d.current_source()), "reset must restore the post-warmup source"
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3 -m pytest tests/unit/test_baroclinic_driver.py::test_reset_restores_warm_state -q`
Expected: FAIL (`AttributeError: 'BaroclinicSourceDriver' object has no attribute 'reset'`).

- [ ] **Step 3: Implement snapshot + reset in the driver**

In `src/gasgiant/sim/baroclinic_driver.py`, add `import copy` at the top (after `from __future__ import annotations`):

```python
import copy
```

In `BaroclinicSourceDriver.__init__`, AFTER the existing warmup + outcrop check (i.e. after the `if self.outcropped: raise RuntimeError(...)` block), add:

```python
        # Post-warmup snapshot: a reused driver (cache hit on a RESTART rebuild)
        # restores this so every development run starts from the identical
        # baroclinic state -- deterministic regardless of prior preview ticks.
        self._warm_st = copy.deepcopy(self.st)
```

Add the `reset` method (after `advance`):

```python
    def reset(self) -> None:
        """Restore the post-warmup state. Called when a cached driver is reused
        for a new development run so the result is independent of how far a live
        preview was ticked before a RESTART-tier edit."""
        self.st = copy.deepcopy(self._warm_st)
        self.outcropped = False
```

- [ ] **Step 4: Run the CPU test to verify it passes**

Run: `py -3 -m pytest tests/unit/test_baroclinic_driver.py -q`
Expected: PASS (the new test + the existing driver tests).

- [ ] **Step 5: Wire `reset()` into the facade cache-hit path**

In `src/gasgiant/engine/facade.py`, in `_init_baroclinic`, the cache-hit branch currently reads:

```python
        if self._baro_driver is not None and self._baro_key == key:
            return  # reuse cached driver (no re-warmup)
```

Change it to reset the reused driver before returning:

```python
        if self._baro_driver is not None and self._baro_key == key:
            self._baro_driver.reset()  # deterministic: each dev run starts post-warmup
            return  # reuse cached driver (no re-warmup)
```

- [ ] **Step 6: Write the failing GPU determinism test**

Append to `tests/gpu/test_m3_ship.py`:

```python
def test_restart_reuse_independent_of_prior_ticks(gpu):
    """A reused (cached) driver must reset to post-warmup on a RESTART-tier edit,
    so the developed render is independent of how far a live preview was ticked
    before the edit. Both runs apply the SAME unrelated edit (vort_hypervisc),
    so only the baroclinic-stream position could differ -- and the reset removes
    that. Without the reset, the 10-pre-tick run advances the driver further and
    the renders diverge well above the noise floor."""
    def run(pre_ticks: int) -> np.ndarray:
        p = _baro_params(seed=5, dev_steps=48)
        sim = Simulation(p, gpu)
        try:
            for _ in range(pre_ticks):
                sim.tick(4)  # 10*4 = 40 steps crosses several update_every=16 boundaries
            new = p.model_copy(update={"solver": p.solver.model_copy(
                update={"vort_hypervisc": 0.65})})  # unrelated Tier.RESTART edit
            sim.update_params(new)
            return sim.render_maps(512)["color"].astype(np.float64)
        finally:
            sim._release_sim()

    a = run(0)
    b = run(10)
    maxdiff = np.abs(a - b).max()
    assert maxdiff <= GPU_NOISE_ATOL, f"restart reuse must be tick-count-independent (maxdiff={maxdiff})"
```

- [ ] **Step 7: Run the GPU test**

Run: `py -3 -m pytest tests/gpu/test_m3_ship.py::test_restart_reuse_independent_of_prior_ticks -q`
Expected: PASS. (Before Step 5's wiring it would FAIL with a maxdiff well above 0.01 — confirm the fix is load-bearing by temporarily reverting Step 5 if you want, then restore it.)

- [ ] **Step 8: Full M3 regression**

Run: `py -3 -m pytest tests/gpu/test_m3_ship.py tests/gpu/test_m3_external_source.py tests/unit/test_baroclinic_params.py tests/unit/test_baroclinic_driver.py -q`
Expected: PASS (all M3 ship + driver + hook tests).

- [ ] **Step 9: Commit**

```bash
git add src/gasgiant/sim/baroclinic_driver.py src/gasgiant/engine/facade.py tests/unit/test_baroclinic_driver.py tests/gpu/test_m3_ship.py
git commit -m "M3: deterministic cached-driver reuse (reset to post-warmup; PR review finding 2)"
```

---

## After both tasks

- [ ] Push to update the PR: `git push`
- [ ] Reply to the PR review noting: finding 1 fixed (E402); finding 2 fixed with the snapshot/reset (stronger than the suggested comment) + the new determinism test; finding 3 — the 21 M3 GPU tests were run locally (green) and run in CI on a GL-capable runner.
- [ ] (Optional) Final whole-implementation review of the two fix commits before merge.

## Out of scope (explicitly not in this plan)

- The ~581 pre-existing ruff errors on `master` (the lint baseline). The reviewer flagged this as a separate repo-hygiene task independent of this PR; reconciling it here would balloon the PR and mix concerns. Track it separately.
- Any change to the coupling physics, gain, or the coherent-metric verdict (settled; see the falsification + ship verdicts).

## Validation

- Lint: `uv run ruff check tests/gpu/test_m3_ship.py` → clean.
- Unit: `py -3 -m pytest tests/unit/test_baroclinic_driver.py tests/unit/test_baroclinic_params.py -q`.
- GPU: `py -3 -m pytest tests/gpu/test_m3_ship.py tests/gpu/test_m3_external_source.py -q`.
- P0.5 determinism unchanged (no production render path touched): `py -3 scripts/p05_baseline_hash.py --check` → 9/9 match.

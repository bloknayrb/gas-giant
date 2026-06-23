"""Ship-path: driver lifecycle, off-path no-op, preview==export, seed determinism.

Render comparisons here use a small tolerance (GPU_NOISE_ATOL), NOT byte-equality.
In a clean process the vorticity render is byte-deterministic (verified directly),
but across a long pytest session sharing ONE session-scoped GPU context, many
build/release cycles introduce ~0.004 LSB float nondeterminism in the render. That
noise is orders of magnitude below the coupling signal (on-vs-off maxdiff ~0.69,
~45% of pixels), so a tolerance of 0.01 cleanly separates the two and still catches
a cadence/clamp bug (which diverges far above 0.01). The dedicated BYTE-EXACT no-op
gate for the external hook lives in tests/gpu/test_m3_external_source.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu

GPU_NOISE_ATOL = 1e-2  # > measured ~0.004 session LSB noise, << ~0.69 coupling signal


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


def _dev_render(p: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(512)["color"].astype(np.float64)
    finally:
        sim._release_sim()


def test_off_path_matches_plain_vorticity(gpu):
    """baroclinic.enabled=False must develop the same render as plain vorticity
    (no-op within the GPU session noise floor; byte-exact gate is in
    test_m3_external_source.py)."""
    base = _dev_render(_baro_params(enabled=False), gpu)
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = 40
    p.solver.type = SolverType.VORTICITY
    plain = _dev_render(p, gpu)
    maxdiff = np.abs(base - plain).max()
    assert maxdiff <= GPU_NOISE_ATOL, f"default-off must not change render (maxdiff={maxdiff})"


def test_preview_equals_export(gpu):
    """Developing via many small tick(2) calls must match one run_to_completion.
    The cadence clamp makes the source-update schedule chunk-size-independent, so
    the two agree to the GPU noise floor; a clamp bug would diverge far above it."""
    p = _baro_params(dev_steps=48)
    export = Simulation(p, gpu)
    try:
        export.run_to_completion(chunk=64)
        export_img = export.render_maps(512)["color"].astype(np.float64)
    finally:
        export._release_sim()

    preview = Simulation(_baro_params(dev_steps=48), gpu)
    try:
        while preview.tick(2):
            pass
        preview_img = preview.render_maps(512)["color"].astype(np.float64)
    finally:
        preview._release_sim()

    maxdiff = np.abs(preview_img - export_img).max()
    assert maxdiff <= GPU_NOISE_ATOL, f"preview chunking must equal export (maxdiff={maxdiff})"


def test_enabled_changes_render(gpu):
    """Coupling on must produce a LARGE, structured change vs off (well above noise)."""
    on = _dev_render(_baro_params(enabled=True), gpu)
    off = _dev_render(_baro_params(enabled=False), gpu)
    maxdiff = np.abs(on - off).max()
    assert maxdiff > 0.05, f"enabled coupling must materially change the render (maxdiff={maxdiff})"


def test_seed_determinism(gpu):
    """Same enabled params + same seed -> same render (within noise); different
    seed -> materially different storm pattern (proves seed is plumbed)."""
    a = _dev_render(_baro_params(seed=11), gpu)
    a2 = _dev_render(_baro_params(seed=11), gpu)
    b = _dev_render(_baro_params(seed=12), gpu)
    assert np.abs(a - a2).max() <= GPU_NOISE_ATOL, "same seed must reproduce"
    assert np.abs(a - b).max() > 0.05, "different seed must change storm pattern"


def test_graceful_warmup_outcrop(gpu, monkeypatch):
    """A warmup past outcrop must degrade to uncoupled (driver=None), NOT crash
    construction, and render the same as the uncoupled run.

    The production config (gp2=0.075) is intentionally stable and never outcrops,
    so force the legacy unstable gp2=0.3 (outcrops ~12.3k) to exercise the
    graceful-degrade path with a warmup beyond it."""
    from gasgiant.sim import baroclinic_source as bsrc
    monkeypatch.setattr(bsrc, "GP2", 0.3)
    p = _baro_params()
    p.solver.baroclinic = p.solver.baroclinic.model_copy(update={"warmup_steps": 15000})
    sim = Simulation(p, gpu)  # must NOT raise
    try:
        assert sim._baro_driver is None, "warmup outcrop must degrade to uncoupled"
        outcropped = sim.render_maps(512)["color"].astype(np.float64)
    finally:
        sim._release_sim()
    base = _dev_render(_baro_params(enabled=False), gpu)
    assert np.abs(outcropped - base).max() <= GPU_NOISE_ATOL, "degraded == uncoupled"


def test_driver_cache_reused_on_unrelated_restart(gpu):
    """An unrelated RESTART-tier edit (same grid/warmup/seed) must REUSE the warm
    driver object, not rebuild it (no 8000-step re-warmup)."""
    sim = Simulation(_baro_params(), gpu)
    try:
        d0 = sim._baro_driver
        assert d0 is not None
        # change an unrelated RESTART-tier solver field (NOT seed/resolution/warmup)
        new = sim.params.model_copy(update={
            "solver": sim.params.solver.model_copy(update={"vort_hypervisc": 0.7})})
        sim.update_params(new)
        assert sim._baro_driver is d0, "unrelated RESTART must reuse cached driver"
    finally:
        sim._release_sim()


def test_driver_rebuilt_on_seed_change(gpu):
    """A seed change must REBUILD the driver (cache key includes seed)."""
    sim = Simulation(_baro_params(seed=11), gpu)
    try:
        d0 = sim._baro_driver
        new = sim.params.model_copy(update={"seed": 99})
        sim.update_params(new)
        assert sim._baro_driver is not d0, "seed change must rebuild driver"
    finally:
        sim._release_sim()


def _refresh_schedule(p, gpu, chunk):
    sim = Simulation(p, gpu)
    fired = []
    orig = sim._update_baroclinic_source
    def spy():
        fired.append(sim.solver.step_index)
        orig()
    sim._update_baroclinic_source = spy
    try:
        if chunk is None:
            while sim.tick(2):
                pass
        else:
            sim.run_to_completion(chunk=chunk)
    finally:
        sim._release_sim()
    return fired


def test_source_refresh_schedule_chunk_independent(gpu):
    """The set of step indices at which the source is refreshed must be identical
    for tick(2) (preview) and run_to_completion(64) (export) -- a deterministic
    guard against a single-step clamp phase bug that the render tolerance can miss."""
    preview = _refresh_schedule(_baro_params(dev_steps=48), gpu, None)
    export = _refresh_schedule(_baro_params(dev_steps=48), gpu, 64)
    assert preview == export, f"refresh schedule differs: {preview} vs {export}"
    assert preview == [0, 16, 32], f"unexpected schedule {preview}"


def test_factory_preset_smoke(gpu):
    """jupiter_baroclinic builds, develops, renders; differs from uncoupled base."""
    p = load_factory_preset("jupiter_baroclinic")
    p.sim.resolution = 512
    p.sim.dev_steps = 40
    p.solver.baroclinic = p.solver.baroclinic.model_copy(update={
        "warmup_steps": 600, "baro_steps_per_update": 60, "update_every": 16})
    sim = Simulation(p, gpu)
    try:
        assert sim._baro_driver is not None
        coupled = sim.render_maps(512)["color"].astype(np.float64)
    finally:
        sim._release_sim()
    p2 = p.model_copy()
    p2.solver.type = SolverType.VORTICITY
    p2.solver.baroclinic = p2.solver.baroclinic.model_copy(update={"enabled": False})
    base = _dev_render(p2, gpu)
    assert np.abs(coupled - base).max() > 0.05, "factory coupling must change render"


def test_restart_reuse_independent_of_prior_ticks(gpu):
    """A reused (cached) driver must reset to post-warmup on a RESTART-tier edit,
    so the developed render is independent of how far a live preview was ticked
    before the edit. Both runs apply the SAME unrelated edit (vort_hypervisc),
    so only the baroclinic-stream position could differ -- the reset removes it."""
    def run(pre_ticks: int) -> np.ndarray:
        p = _baro_params(seed=5, dev_steps=48)
        sim = Simulation(p, gpu)
        try:
            for _ in range(pre_ticks):
                sim.tick(4)  # crosses several update_every=16 boundaries
            new = p.model_copy(update={"solver": p.solver.model_copy(
                update={"vort_hypervisc": 0.65})})  # unrelated Tier.RESTART edit
            sim.update_params(new)
            return sim.render_maps(512)["color"].astype(np.float64)
        finally:
            sim._release_sim()

    a = run(0)
    b = run(10)
    maxdiff = np.abs(a - b).max()
    assert maxdiff <= GPU_NOISE_ATOL, f"restart reuse must be tick-independent (maxdiff={maxdiff})"


def test_mid_run_incoherence_degrades(gpu, monkeypatch):
    """If the source goes incoherent mid-run, tick must degrade to uncoupled and
    keep developing -- no exception escapes (the _update_baroclinic_source catch)."""
    from gasgiant.sim.baroclinic_source import IncoherentSourceError
    sim = Simulation(_baro_params(dev_steps=48), gpu)
    try:
        real = sim._baro_driver.current_source
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 2:  # fail the second refresh (at step_index 16)
                # the EXPECTED degrade signal the facade catches -- a plain
                # ValueError would now (post-refactor) propagate, not degrade.
                raise IncoherentSourceError("coherence gate (injected)")
            return real()

        monkeypatch.setattr(sim._baro_driver, "current_source", flaky)
        sim.run_to_completion(chunk=8)  # must NOT raise
        assert sim._baro_driver is None, "mid-run incoherence must disable coupling"
        assert sim.is_developed, "development must complete after degrading"
    finally:
        sim._release_sim()

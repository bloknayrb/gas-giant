"""Evolving baroclinic source driver + the residency decision rule."""
from __future__ import annotations

import numpy as np

from gasgiant.engine.baroclinic_coupling import CouplingStats, residency_recommendation
from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim.baroclinic_driver import BaroclinicSourceDriver


def test_driver_source_evolves():
    """Re-deriving the source after advancing the baroclinic solver gives a
    DIFFERENT field (the static->evolving upgrade is real)."""
    d = BaroclinicSourceDriver(grid_w=128, grid_h=64, warmup_steps=2500, seed=0)
    src_a = d.current_source()
    d.advance(1500)
    src_b = d.current_source()
    assert src_a.shape == (64, 128)
    assert float(np.abs(src_a - src_b).mean()) > 1e-3, "source must evolve in time"


def test_driver_holds_on_outcrop(monkeypatch):
    """Advancing far past lower-layer outcrop must not raise; the driver holds
    the last good state and still emits a finite, coherent source.

    The production config (gp2=0.075) is intentionally stable and does NOT outcrop
    (survives 40k+ steps), so force the legacy unstable gp2=0.3 to exercise the
    hold-on-outcrop path (it outcrops ~step 12.3k)."""
    monkeypatch.setattr(bsrc, "GP2", 0.3)
    d = BaroclinicSourceDriver(grid_w=64, grid_h=32, warmup_steps=500, seed=0)
    d.advance(20000)                       # well past the gp2=0.3 outcrop (~12.3k)
    assert d.outcropped is True
    src = d.current_source()
    assert np.all(np.isfinite(src))


def test_reset_restores_warm_state():
    """reset() must return the driver to its post-warmup state so every dev run
    starts identically (deterministic cache reuse)."""
    d = BaroclinicSourceDriver(grid_w=64, grid_h=32, warmup_steps=600, seed=0)
    s0 = d.current_source()
    d.advance(300)
    assert not np.allclose(s0, d.current_source()), "advance must change the source"
    d.reset()
    assert np.allclose(s0, d.current_source()), "reset must restore the post-warmup source"


def test_residency_rule():
    cheap = CouplingStats(v16_seconds=10.0, baro_seconds=1.0, upload_seconds=0.5)
    assert residency_recommendation(cheap) == "option-a-sufficient"
    pricey = CouplingStats(v16_seconds=10.0, baro_seconds=4.0, upload_seconds=1.0)
    assert residency_recommendation(pricey) == "consider-residency"

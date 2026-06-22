"""Evolving baroclinic source driver + the residency decision rule."""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.baroclinic_coupling import CouplingStats, residency_recommendation
from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim import shallow_water_ref as ref
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


def test_production_config_is_stable_and_coherent():
    """The SHIPPED config (gp2=0.075, M_ZONAL=14) must survive a real warmup
    WITHOUT outcropping and emit a coherent ~m14 source -- the claim the CPU
    sweep made, now enforced in CI. The monkeypatched outcrop tests only cover
    the legacy gp2=0.3, so without this the production path is unasserted.

    A 4000-step warmup is sufficient: gp2=0.075 survives 40k+ steps and the m=14
    mode is dominant well before 4000, so an early-outcrop or wrong-eddy-scale
    regression still fails here (~30s; no `slow` marker lane exists)."""
    d = BaroclinicSourceDriver(grid_w=64, grid_h=32, warmup_steps=4000, seed=0)
    assert d.outcropped is False, "production gp2=0.075 must not outcrop in warmup"
    src = d.current_source()                       # raises if the coherence gate fails
    assert np.all(np.isfinite(src))
    # Eddy scale: dominant zonal mode in the Jupiter-like band on BOTH the source
    # physics grid and the shipped resampled product.
    zeta = bsrc.geostrophic_vorticity_source(d.st, smooth_sigma=bsrc.SMOOTH_SIGMA)
    m_src, _ = bsrc.dominant_zonal_m(zeta)
    m_out, _ = bsrc.dominant_zonal_m(src)
    assert 10 <= m_src <= bsrc.M_GATE_MAX, f"source dominant m={m_src} out of band"
    assert 10 <= m_out <= bsrc.M_GATE_MAX, f"shipped dominant m={m_out} out of band"


def test_current_source_threads_smooth_sigma(monkeypatch):
    """current_source() must pass the production SMOOTH_SIGMA (1.26) into the
    geostrophic proxy, not silently fall back to the function default (2.5). Spy
    on the kwarg directly -- a value-diff would pass off normalization noise."""
    d = BaroclinicSourceDriver(grid_w=64, grid_h=32, warmup_steps=600, seed=0)
    seen = {}
    real = bsrc.geostrophic_vorticity_source

    def spy(st, **kw):
        seen["smooth_sigma"] = kw.get("smooth_sigma")
        return real(st, **kw)

    monkeypatch.setattr(bsrc, "geostrophic_vorticity_source", spy)
    d.current_source()
    assert seen["smooth_sigma"] == bsrc.SMOOTH_SIGMA


def test_advance_propagates_non_outcrop_error(monkeypatch):
    """advance() catches ONLY the positivity/outcrop signal; a genuine ValueError
    from the solver must PROPAGATE, not be mislabeled as a benign outcrop (which,
    with the stable gp2=0.075 config, should otherwise never happen)."""
    d = BaroclinicSourceDriver(grid_w=64, grid_h=32, warmup_steps=600, seed=0)

    def boom(_st):
        raise ValueError("not an outcrop -- a real bug")

    monkeypatch.setattr(ref, "step_2layer", boom)
    with pytest.raises(ValueError, match="real bug"):
        d.advance(1)
    assert d.outcropped is False, "a non-outcrop error must NOT latch outcropped"


def test_positivity_violation_is_valueerror_subclass():
    """PositivityViolation must subclass ValueError so the semi-implicit path's
    existing `except ValueError` catchers keep working unchanged."""
    assert issubclass(ref.PositivityViolation, ValueError)


def test_residency_rule():
    cheap = CouplingStats(v16_seconds=10.0, baro_seconds=1.0, upload_seconds=0.5)
    assert residency_recommendation(cheap) == "option-a-sufficient"
    pricey = CouplingStats(v16_seconds=10.0, baro_seconds=4.0, upload_seconds=1.0)
    assert residency_recommendation(pricey) == "consider-residency"

"""Evolving baroclinic source driver + the residency decision rule."""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.baroclinic_coupling import CouplingStats, residency_recommendation
from gasgiant.sim import baroclinic_source as bsrc
from gasgiant.sim import shallow_water_ref as ref
from gasgiant.sim.baroclinic_driver import (
    BaroclinicOutcropError,
    BaroclinicSourceDriver,
    BaroclinicWarmupError,
)

# ~150s of CPU reference-solver work; excluded from the fast loop (-m "not gpu and not slow").
pytestmark = pytest.mark.slow

#: (grid_w, grid_h, smooth_sigma) for current_source. These are derivation-time
#: inputs, NOT constructor ones -- the warmup runs on the fixed 192x96 source
#: grid and never sees either, so keying the facade's driver cache on them cost
#: a re-warmup for a bit-identical state (see BaroclinicSourceDriver).
DERIVE = (64, 32, bsrc.SMOOTH_SIGMA)


def test_driver_source_evolves():
    """Re-deriving the source after advancing the baroclinic solver gives a
    DIFFERENT field (the static->evolving upgrade is real)."""
    d = BaroclinicSourceDriver(warmup_steps=2500, seed=0)
    src_a = d.current_source(128, 64, bsrc.SMOOTH_SIGMA)
    d.advance(1500)
    src_b = d.current_source(128, 64, bsrc.SMOOTH_SIGMA)
    assert src_a.shape == (64, 128)
    assert float(np.abs(src_a - src_b).mean()) > 1e-3, "source must evolve in time"


def test_driver_reports_outcrop_and_holds_last_good_state(monkeypatch):
    """Advancing past lower-layer outcrop must RAISE, and still hold the last
    good state so the caller can degrade rather than crash.

    Previously advance() swallowed the outcrop and only latched a flag. That made
    the facade's mid-run handler unreachable: `baroclinic_status` kept reporting
    'active' while the source was frozen on a dead state -- silently reinstating
    the static stamp the driver exists to replace. The raise is the contract.

    The production eddy_scale (0.075) is intentionally stable and does NOT outcrop
    (survives 40k+ steps), so force the legacy unstable 0.3 to exercise the path
    (it outcrops ~step 12.3k)."""
    monkeypatch.setattr(bsrc, "GP2", 0.3)
    d = BaroclinicSourceDriver(warmup_steps=500, seed=0)
    with pytest.raises(BaroclinicOutcropError):
        d.advance(20000)               # well past the gp2=0.3 outcrop (~12.3k)
    assert d.outcropped is True
    src = d.current_source(*DERIVE)
    assert np.all(np.isfinite(src)), "the held state must still derive a usable source"


def test_advance_after_outcrop_keeps_raising(monkeypatch):
    """Once outcropped there is nothing left to advance, so every later call must
    keep reporting it rather than returning as if it had done the work."""
    monkeypatch.setattr(bsrc, "GP2", 0.3)
    d = BaroclinicSourceDriver(warmup_steps=500, seed=0)
    with pytest.raises(BaroclinicOutcropError):
        d.advance(20000)
    with pytest.raises(BaroclinicOutcropError):
        d.advance(1)


def test_warmup_outcrop_still_raises_the_warmup_error(monkeypatch):
    """A warmup outcrop must keep surfacing as BaroclinicWarmupError -- the
    facade catches that distinctly at construction, and app/main.py toasts it."""
    monkeypatch.setattr(bsrc, "GP2", 0.3)
    with pytest.raises(BaroclinicWarmupError):
        BaroclinicSourceDriver(warmup_steps=20000, seed=0)


def test_reset_restores_warm_state():
    """reset() must return the driver to its post-warmup state so every dev run
    starts identically (deterministic cache reuse)."""
    d = BaroclinicSourceDriver(warmup_steps=600, seed=0)
    s0 = d.current_source(*DERIVE)
    d.advance(300)
    assert not np.allclose(s0, d.current_source(*DERIVE)), "advance must change the source"
    d.reset()
    assert np.allclose(s0, d.current_source(*DERIVE)), "reset must restore the post-warmup source"


def test_production_config_is_stable_and_coherent():
    """The SHIPPED config (gp2=0.075, M_ZONAL=14) must survive a real warmup
    WITHOUT outcropping and emit a coherent ~m14 source -- the claim the CPU
    sweep made, now enforced in CI. The monkeypatched outcrop tests only cover
    the legacy gp2=0.3, so without this the production path is unasserted.

    A 4000-step warmup is sufficient: gp2=0.075 survives 40k+ steps and the m=14
    mode is dominant well before 4000, so an early-outcrop or wrong-eddy-scale
    regression still fails here (~30s; no `slow` marker lane exists)."""
    d = BaroclinicSourceDriver(warmup_steps=4000, seed=0)
    assert d.outcropped is False, "production gp2=0.075 must not outcrop in warmup"
    src = d.current_source(*DERIVE)                # raises if the coherence gate fails
    assert np.all(np.isfinite(src))
    # Eddy scale: dominant zonal mode in the Jupiter-like band on BOTH the source
    # physics grid and the shipped resampled product.
    zeta = bsrc.geostrophic_vorticity_source(d.st, smooth_sigma=bsrc.SMOOTH_SIGMA)
    m_src, _ = bsrc.dominant_zonal_m(zeta)
    m_out, _ = bsrc.dominant_zonal_m(src)
    assert 10 <= m_src <= bsrc.M_GATE_MAX, f"source dominant m={m_src} out of band"
    assert 10 <= m_out <= bsrc.M_GATE_MAX, f"shipped dominant m={m_out} out of band"


def test_current_source_forwards_its_smooth_sigma(monkeypatch):
    """The sigma the CALLER passes must reach the geostrophic proxy -- not the
    module constant, and not the proxy's own 2.5 default.

    Spy on the kwarg directly; a value-diff would pass off normalization noise.
    Uses a sigma equal to NEITHER fallback, so silently substituting either one
    fails. This is what keeps the "Storm edge softness" slider live now that a
    cached driver is reused across a change to it."""
    d = BaroclinicSourceDriver(warmup_steps=600, seed=0)
    probe = 3.75
    assert probe not in (bsrc.SMOOTH_SIGMA, 2.5), "probe must not match a fallback"
    seen = {}
    real = bsrc.geostrophic_vorticity_source

    def spy(st, **kw):
        seen["smooth_sigma"] = kw.get("smooth_sigma")
        return real(st, **kw)

    monkeypatch.setattr(bsrc, "geostrophic_vorticity_source", spy)
    d.current_source(64, 32, probe)
    assert seen["smooth_sigma"] == probe


def test_warm_state_ignores_the_derivation_inputs():
    """The measurement this whole split rests on: two drivers warmed identically
    hold bit-identical state, and the grid/sigma only change what is DERIVED off
    it. If this ever fails, the facade cache key is unsound and must take them
    back."""
    a = BaroclinicSourceDriver(warmup_steps=400, seed=0)
    b = BaroclinicSourceDriver(warmup_steps=400, seed=0)
    for f in ("h1", "h2", "u1", "v1", "u2", "v2"):
        assert np.array_equal(getattr(a.st, f), getattr(b.st, f)), f
    assert a.current_source(128, 64, 1.26).shape == (64, 128)
    assert a.current_source(64, 32, 1.26).shape == (32, 64)
    assert not np.allclose(a.current_source(64, 32, 1.26),
                           a.current_source(64, 32, 4.0)), "sigma must still bite"


def test_advance_propagates_non_outcrop_error(monkeypatch):
    """advance() catches ONLY the positivity/outcrop signal; a genuine ValueError
    from the solver must PROPAGATE, not be mislabeled as a benign outcrop (which,
    with the stable gp2=0.075 config, should otherwise never happen)."""
    d = BaroclinicSourceDriver(warmup_steps=600, seed=0)

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

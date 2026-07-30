"""A2-2: baroclinic graceful degrade must be VISIBLE and the catches NARROW.

The degrade paths in ``Simulation._init_baroclinic`` / ``_update_baroclinic_source``
used to emit ``log.warning`` only (invisible in the GUI) and caught bare
``RuntimeError`` (so a genuine unexpected error was silently swallowed as a
degrade, contradicting the docstring). These tests pin the new contract:

- ``Simulation.baroclinic_status`` is ``'off' | 'active' | 'degraded'``;
- ``Simulation.baroclinic_degraded_reason`` carries the human-readable cause;
- only the DOCUMENTED degrade signals (``BaroclinicWarmupError`` at build,
  ``PositivityViolation`` / ``IncoherentSourceError`` mid-run, plus ImportError
  belt-and-braces at build) degrade — anything else propagates loudly.

CPU-only: the Simulation is assembled via ``__new__`` with a stub solver (the
baroclinic wiring never touches GL), so no GPU context is needed. The GPU
end-to-end degrade renders stay in tests/gpu/test_m3_ship.py.
"""
from __future__ import annotations

import types

import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.sim import baroclinic_driver as bdrv
from gasgiant.sim.baroclinic_source import IncoherentSourceError
from gasgiant.sim.shallow_water_ref import PositivityViolation


def _params(enabled: bool = True) -> PlanetParams:
    p = PlanetParams()
    p.solver.type = SolverType.VORTICITY
    p.solver.baroclinic.enabled = enabled
    return p


def _stub_sim(params: PlanetParams) -> Simulation:
    """A Simulation shell with just the state _init_baroclinic /
    _update_baroclinic_source / the status properties touch."""
    sim = object.__new__(Simulation)
    sim.params = params
    sim.solver = types.SimpleNamespace(
        equirect=types.SimpleNamespace(size=(64, 32)),
        external_omega_tex=None,
        external_gain=0.0,
    )
    # set_external_vorticity_source uploads the source on the SUCCESS path only;
    # a tiny duck-typed texture factory keeps that path GL-free too.
    sim.gpu = types.SimpleNamespace(
        texture2d=lambda size, components, dtype, data=None, linear=False: types.SimpleNamespace(
            size=size, repeat_x=False, write=lambda raw: None, release=lambda: None
        )
    )
    sim._baro_driver = None
    sim._baro_key = None
    sim._baro_degraded_reason = None
    sim._baro_failed_key = None
    sim._baro_failed_reason = None
    return sim


class _StubDriver:
    def __init__(self, grid_w: int, grid_h: int, warmup_steps: int, seed: int,
                 **levers) -> None:
        # **levers absorbs the storm-band fields (latitude/width/eddy_scale/
        # zonal_count/smooth/phase_jitter/spectrum_width) so this stub does not
        # have to be re-edited every time one is added.
        self.levers = levers

    def reset(self) -> None:
        pass


# -- status surface -----------------------------------------------------------------


def test_status_off_when_disabled():
    sim = _stub_sim(_params(enabled=False))
    sim._init_baroclinic()
    assert sim.baroclinic_status == "off"
    assert sim.baroclinic_degraded_reason is None


def test_status_active_when_driver_builds(monkeypatch):
    monkeypatch.setattr(bdrv, "BaroclinicSourceDriver", _StubDriver)
    sim = _stub_sim(_params())
    sim._init_baroclinic()
    assert sim._baro_driver is not None
    assert sim.baroclinic_status == "active"
    assert sim.baroclinic_degraded_reason is None


# -- build-time degrade paths ---------------------------------------------------------


def test_warmup_outcrop_degrades_with_reason(monkeypatch):
    def boom(**kwargs):
        raise bdrv.BaroclinicWarmupError("warmup outcropped (injected)")

    monkeypatch.setattr(bdrv, "BaroclinicSourceDriver", boom)
    sim = _stub_sim(_params())
    sim._init_baroclinic()  # must NOT raise: documented graceful degrade
    assert sim._baro_driver is None
    assert sim.baroclinic_status == "degraded"
    assert "outcropped" in sim.baroclinic_degraded_reason


def test_import_error_degrades_with_reason(monkeypatch):
    """A2-3 belt-and-braces: a missing optional numerics dep at driver build time
    must degrade to uncoupled, not crash construction (the docstring's 'never
    crash construction' contract)."""
    def boom(**kwargs):
        raise ImportError("No module named 'scipy'")

    monkeypatch.setattr(bdrv, "BaroclinicSourceDriver", boom)
    sim = _stub_sim(_params())
    sim._init_baroclinic()  # must NOT raise
    assert sim._baro_driver is None
    assert sim.baroclinic_status == "degraded"
    assert "scipy" in sim.baroclinic_degraded_reason


def test_unexpected_build_error_propagates(monkeypatch):
    """The old bare ``except RuntimeError`` swallowed genuine bugs. A plain
    RuntimeError (NOT the documented BaroclinicWarmupError subclass) must now
    propagate loudly."""
    def boom(**kwargs):
        raise RuntimeError("genuine unexpected bug")

    monkeypatch.setattr(bdrv, "BaroclinicSourceDriver", boom)
    sim = _stub_sim(_params())
    with pytest.raises(RuntimeError, match="genuine unexpected bug"):
        sim._init_baroclinic()


def test_warmup_error_is_runtimeerror_subclass():
    """BaroclinicWarmupError must subclass RuntimeError so any existing
    ``except RuntimeError`` caller keeps working (the IncoherentSourceError /
    PositivityViolation compatibility pattern)."""
    assert issubclass(bdrv.BaroclinicWarmupError, RuntimeError)


# -- mid-run degrade paths ------------------------------------------------------------


def _mid_run_sim(source_exc: Exception | None) -> Simulation:
    sim = _stub_sim(_params())
    sim._baro_next_update = 0
    sim._baro_update_every = 16
    sim._baro_gain = 0.5
    sim._baro_steps_per_update = 1

    class Driver:
        def advance(self, n: int) -> None:
            pass

        def current_source(self):
            if source_exc is not None:
                raise source_exc
            import numpy as np
            return np.zeros((32, 64), dtype=np.float32)

    sim._baro_driver = Driver()
    return sim


def test_mid_run_incoherence_degrades_with_reason():
    sim = _mid_run_sim(IncoherentSourceError("coherence gate (injected)"))
    sim._update_baroclinic_source()  # must NOT raise: documented degrade
    assert sim._baro_driver is None
    assert sim.baroclinic_status == "degraded"
    assert "coherence gate" in sim.baroclinic_degraded_reason


def test_mid_run_outcrop_degrades_with_reason():
    sim = _mid_run_sim(PositivityViolation("lower-layer outcrop (injected)"))
    sim._update_baroclinic_source()  # must NOT raise: documented degrade
    assert sim._baro_driver is None
    assert sim.baroclinic_status == "degraded"
    assert "outcrop" in sim.baroclinic_degraded_reason


def test_mid_run_unexpected_runtime_error_propagates():
    """The mid-run catch listed bare RuntimeError; the docstring promised the
    opposite. A genuine RuntimeError must now propagate."""
    sim = _mid_run_sim(RuntimeError("genuine mid-run bug"))
    with pytest.raises(RuntimeError, match="genuine mid-run bug"):
        sim._update_baroclinic_source()
    assert sim._baro_driver is not None  # not silently degraded


def test_mid_run_plain_valueerror_propagates():
    sim = _mid_run_sim(ValueError("genuine value bug"))
    with pytest.raises(ValueError, match="genuine value bug"):
        sim._update_baroclinic_source()


def test_mid_run_success_keeps_active_status():
    sim = _mid_run_sim(None)
    sim._update_baroclinic_source()
    assert sim.baroclinic_status == "active"
    assert sim.baroclinic_degraded_reason is None


# -- the driver cache key -----------------------------------------------------
#
# Every storm-band lever changes the WARM STATE, so each must take part in the
# cache key. A lever missing from it is a lever the artist can move while the
# facade hands back the driver warmed at the OLD value -- the slider appears
# dead, and only on a full app restart does the change take effect.
#
# These reuse ONE Simulation and re-enter _init_baroclinic, which is the only
# arrangement that can detect the bug: a fresh Simulation per case always starts
# with _baro_driver = None and therefore always MISSES the cache, so it would
# pass whether or not the key is correct.


class _CountingDriver(_StubDriver):
    built: list[dict] = []

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        _CountingDriver.built.append(kw)


#: (params field, driver kwarg, test value). Three of the seven are renamed on
#: the way down -- the artist-facing name is plain English, the driver keeps the
#: physics name -- so the mapping is asserted rather than assumed.
_LEVERS = [
    ("latitude", "latitude", 30.0),
    ("width", "width", 18.0),
    ("eddy_scale", "gp2", 0.09),
    ("zonal_count", "m_zonal", 10),
    ("smooth", "smooth_sigma", 2.5),
    ("phase_jitter", "phase_jitter", 2.0),
    ("spectrum_width", "spectrum_width", 4),
]


@pytest.mark.parametrize("field,kwarg,value", _LEVERS)
def test_every_storm_band_lever_invalidates_the_cached_driver(
        monkeypatch, field, kwarg, value):
    monkeypatch.setattr(bdrv, "BaroclinicSourceDriver", _CountingDriver)
    _CountingDriver.built = []
    sim = _stub_sim(_params(enabled=True))

    sim._init_baroclinic()
    assert len(_CountingDriver.built) == 1, "first call must build"

    sim._init_baroclinic()
    assert len(_CountingDriver.built) == 1, "an unchanged config must reuse"

    setattr(sim.params.solver.baroclinic, field, value)
    sim._init_baroclinic()
    assert len(_CountingDriver.built) == 2, f"{field} must invalidate the cache"
    assert _CountingDriver.built[-1][kwarg] == value, f"{field} must reach the driver"


def test_lever_list_covers_every_storm_band_field():
    """Guards the guard: a new lever added to the params model without a row in
    ``_LEVERS`` would leave the cache-key test silently not covering it."""
    from gasgiant.params.model import BaroclinicParams
    covered = {f for f, _, _ in _LEVERS}
    cadence = {"enabled", "gain", "warmup_steps",
               "baro_steps_per_update", "update_every"}
    assert set(BaroclinicParams.model_fields) - cadence == covered


def test_a_failed_warmup_is_not_retried(monkeypatch):
    """A configuration that died in warmup must stay degraded, including across
    the disable/re-enable cycle an artist performs after seeing the failure.

    Re-running a warmup already known to die costs up to 20000 CPU steps (~2 min)
    and produces the same failure every time. The memo lives on its OWN key
    precisely because `_baro_key` is nulled by both the disable path and the
    mid-run degrade.
    """
    calls = []

    def boom(**kwargs):
        calls.append(kwargs)
        raise bdrv.BaroclinicWarmupError("warmup outcropped (injected)")

    monkeypatch.setattr(bdrv, "BaroclinicSourceDriver", boom)
    sim = _stub_sim(_params(enabled=True))

    sim._init_baroclinic()
    assert len(calls) == 1 and sim.baroclinic_status == "degraded"

    sim._init_baroclinic()
    assert len(calls) == 1, "an unchanged failing config must not be retried"
    assert sim.baroclinic_status == "degraded"
    assert "outcropped" in sim.baroclinic_degraded_reason

    # Disable then re-enable at the same settings: still must not retry.
    sim.params.solver.baroclinic.enabled = False
    sim._init_baroclinic()
    assert sim.baroclinic_status == "off"
    sim.params.solver.baroclinic.enabled = True
    sim._init_baroclinic()
    assert len(calls) == 1, "re-enabling the same failing config must not retry"

    # A DIFFERENT config is a different question and must be tried.
    sim.params.solver.baroclinic.latitude = 60.0
    sim._init_baroclinic()
    assert len(calls) == 2


def test_a_successful_build_clears_the_failure_memo(monkeypatch):
    """Otherwise the memo outlives the failure and a later degrade could report a
    reason from an unrelated earlier one."""
    monkeypatch.setattr(bdrv, "BaroclinicSourceDriver", _StubDriver)
    sim = _stub_sim(_params(enabled=True))
    sim._baro_failed_key = ("stale",)
    sim._baro_failed_reason = "stale reason"
    sim._init_baroclinic()
    assert sim.baroclinic_status == "active"
    assert sim._baro_failed_key is None
    assert sim._baro_failed_reason is None

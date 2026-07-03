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
    return sim


class _StubDriver:
    def __init__(self, grid_w: int, grid_h: int, warmup_steps: int, seed: int) -> None:
        pass

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

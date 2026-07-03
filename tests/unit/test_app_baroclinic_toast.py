"""A2-2 GUI half: the app must toast ONCE when the sim transitions to a
degraded baroclinic status (the engine-side degrade used to be log-file-only,
invisible in-window). Exercises ``StudioApp._check_baroclinic_status`` headless
(the same ``__new__``-shell pattern as test_commit_loop)."""
from __future__ import annotations

import types

import pytest

main = pytest.importorskip("gasgiant.app.main")
StudioApp = main.StudioApp


def _app(status: str = "off", reason: str | None = None):
    app = StudioApp.__new__(StudioApp)
    app.toasts = main.Toasts()
    app._baro_status_seen = "off"
    app.sim = types.SimpleNamespace(
        baroclinic_status=status, baroclinic_degraded_reason=reason
    )
    return app


def _error_toasts(app) -> list[str]:
    return [m for (m, is_err, _) in app.toasts._items if is_err]


def test_transition_to_degraded_toasts_once():
    app = _app(status="degraded", reason="warmup outcropped (test)")
    app._check_baroclinic_status()
    errors = _error_toasts(app)
    assert len(errors) == 1
    assert "baroclinic" in errors[0]
    assert "warmup outcropped (test)" in errors[0]
    # same status next frame: no toast spam
    app._check_baroclinic_status()
    assert len(_error_toasts(app)) == 1


def test_active_and_off_statuses_do_not_toast():
    for status in ("off", "active"):
        app = _app(status=status)
        app._check_baroclinic_status()
        app._check_baroclinic_status()
        assert app.toasts._items == []


def test_redegrade_after_recovery_toasts_again():
    """active -> degraded -> (rebuild) active -> degraded must toast on EACH
    transition into degraded, not only the first ever."""
    app = _app(status="active")
    app._check_baroclinic_status()
    app.sim.baroclinic_status = "degraded"
    app.sim.baroclinic_degraded_reason = "first"
    app._check_baroclinic_status()
    app.sim.baroclinic_status = "active"
    app._check_baroclinic_status()
    app.sim.baroclinic_status = "degraded"
    app.sim.baroclinic_degraded_reason = "second"
    app._check_baroclinic_status()
    errors = _error_toasts(app)
    assert len(errors) == 2


def test_no_sim_is_a_noop():
    app = StudioApp.__new__(StudioApp)
    app.toasts = main.Toasts()
    app._baro_status_seen = "off"
    app.sim = None
    app._check_baroclinic_status()  # must not raise
    assert app.toasts._items == []

"""B5-6: vorticity-only storm levers must WARN (not error) under kinematic.

``storms.hero_solid_core`` / ``storms.oval_solid_core`` only alter the
vorticity-mode stamp; on ``solver.type == "kinematic"`` they are exact no-ops
(e.g. ice_giant's Neptune dark oval stays exposed to the whirlpool-winding
artifact with no signal). A hard error would reject legitimate presets --
these are warnings, surfaced as a toast on preset load.

The model half (``PlanetParams.validation_warnings``) is here; the app half
(``StudioApp._toast_param_warnings`` on the two load paths) follows the
headless ``__new__``-shell pattern of test_commit_loop.
"""
from __future__ import annotations

import pytest

from gasgiant.params.model import PlanetParams, SolverType


def test_defaults_have_no_warnings():
    assert PlanetParams().validation_warnings() == []


def test_hero_solid_core_on_kinematic_warns_not_errors():
    p = PlanetParams()
    p.storms.hero_solid_core = 0.8  # must NOT raise (warn only)
    warnings = p.validation_warnings()
    assert len(warnings) == 1
    assert "hero_solid_core" in warnings[0]
    assert "kinematic" in warnings[0]


def test_oval_solid_core_on_kinematic_warns():
    p = PlanetParams()
    p.storms.oval_solid_core = 1.0
    warnings = p.validation_warnings()
    assert len(warnings) == 1
    assert "oval_solid_core" in warnings[0]


def test_both_levers_warn_individually():
    p = PlanetParams()
    p.storms.hero_solid_core = 0.5
    p.storms.oval_solid_core = 0.5
    assert len(p.validation_warnings()) == 2


def test_no_warning_under_vorticity_solver():
    p = PlanetParams()
    p.solver.type = SolverType.VORTICITY
    p.storms.hero_solid_core = 0.8
    p.storms.oval_solid_core = 1.0
    assert p.validation_warnings() == []


# -- app half: warnings toast on preset load ------------------------------------------


def _make_app():
    main = pytest.importorskip("gasgiant.app.main")
    app = main.StudioApp.__new__(main.StudioApp)
    app.toasts = main.Toasts()
    return app, main


def test_toast_param_warnings_surfaces_each_warning():
    app, _ = _make_app()
    p = PlanetParams()
    p.storms.hero_solid_core = 0.8
    app._toast_param_warnings(p)
    messages = [m for (m, _is_err, _) in app.toasts._items]
    assert len(messages) == 1
    assert "hero_solid_core" in messages[0]
    assert "warning" in messages[0].lower()


def test_toast_param_warnings_silent_when_clean():
    app, _ = _make_app()
    app._toast_param_warnings(PlanetParams())
    assert app.toasts._items == []


def test_load_preset_entry_toasts_warnings(monkeypatch, tmp_path):
    """The factory/user preset load path must surface the warnings."""
    from collections import deque

    app, main = _make_app()
    warned = PlanetParams()
    warned.storms.hero_solid_core = 0.8
    monkeypatch.setattr(main, "load_factory_preset", lambda name: warned)

    class FakeSim:
        def update_params(self, new):
            return set()

    app.params = PlanetParams()
    app.sim = FakeSim()
    app.viewport = None
    app._export = None
    app._recomputing = False
    app._live = app.params
    app._gesture_base = None
    app._undo_stack = deque(maxlen=64)
    app._redo_stack = deque(maxlen=64)
    app._active_preset = None
    app._pristine = None

    app._load_preset_entry("whatever", main.PresetSource.FACTORY)
    messages = [m for (m, _is_err, _) in app.toasts._items]
    assert any("hero_solid_core" in m for m in messages), messages

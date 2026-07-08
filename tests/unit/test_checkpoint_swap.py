"""T3: the checkpoint load-swap helper, driven headlessly.

``_apply_loaded_checkpoint`` performs steps 2-7 of the load-swap (release old
sim, adopt loaded sim + params, reset working copy, clear history, reset preset
identity + baroclinic latch, mark the preview dirty). ``_load_state`` adds step
1 (refuse while exporting) and the load-with-error-handling around it. Both are
driven against a FakeSim/FakeViewport (same headless pattern as
test_playback.py) so the swap contract is pinned without a real GL context.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import PresetSource

main = pytest.importorskip("gasgiant.app.main")
StudioApp = main.StudioApp


class FakeSim:
    def __init__(self, developed: bool = True) -> None:
        self.params = PlanetParams(seed=7)
        self.released = 0
        self._developed = developed
        self.steps_done = 42

    def release(self) -> None:
        self.released += 1

    @property
    def is_developed(self) -> bool:
        return self._developed


class FakeViewport:
    def __init__(self) -> None:
        self.stale = 0

    def mark_stale(self) -> None:
        self.stale += 1


def _make_app() -> StudioApp:
    app = StudioApp.__new__(StudioApp)
    app.sim = FakeSim()
    app.viewport = FakeViewport()
    app.gpu = object()
    app.toasts = main.Toasts()
    app._export = None
    app.params = app.sim.params
    app._live = app.sim.params
    app._gesture_base = None
    app._undo_stack = deque(maxlen=64)
    app._redo_stack = deque(maxlen=64)
    app._recomputing = True
    app._active_preset = ("gas_giant_warm", PresetSource.FACTORY)
    app._pristine = app.sim.params.model_copy(deep=True)
    app._baro_status_seen = "degraded"
    return app


def test_apply_loaded_checkpoint_runs_all_seven_swap_steps() -> None:
    app = _make_app()
    old_sim = app.sim
    # Seed some history that the swap must clear.
    app._undo_stack.append(object())
    app._redo_stack.append(object())
    # A pending draft that must be dropped (step 4 reset working copy).
    app._gesture_base = object()

    new_sim = FakeSim(developed=False)
    path = Path("/tmp/my_run.npz")
    app._apply_loaded_checkpoint(new_sim, path)

    # 2. old sim released (the real GPU-leak fix).
    assert old_sim.released == 1
    # 3. loaded sim adopted.
    assert app.sim is new_sim
    # 4. params adopted + working copy reset.
    assert app.params is new_sim.params
    assert app._live is app.params
    assert app._gesture_base is None
    # 5. undo/redo history cleared.
    assert len(app._undo_stack) == 0
    assert len(app._redo_stack) == 0
    # 6. transient checkpoint identity + baroclinic latch reset.
    assert app._active_preset == ("checkpoint/my_run", PresetSource.FILE)
    assert app._pristine == new_sim.params
    assert app._baro_status_seen == "off"
    # 7. preview marked dirty; recomputing reflects the still-undeveloped run.
    assert app.viewport.stale == 1
    assert app._recomputing is True


def test_apply_loaded_checkpoint_developed_run_is_not_recomputing() -> None:
    app = _make_app()
    new_sim = FakeSim(developed=True)
    app._apply_loaded_checkpoint(new_sim, Path("/tmp/done.npz"))
    assert app._recomputing is False


def test_load_state_refuses_while_exporting() -> None:
    """Step 1: a load must not swap the sim out from under a running export."""
    app = _make_app()
    old_sim = app.sim
    app._export = object()  # an export in flight
    app._load_state(Path("/tmp/whatever.npz"))
    # Untouched: no release, same sim, no identity change.
    assert old_sim.released == 0
    assert app.sim is old_sim
    assert app._active_preset == ("gas_giant_warm", PresetSource.FACTORY)


def test_load_state_load_failure_leaves_state_untouched(monkeypatch) -> None:
    """A load failure (bad version / corrupt file) toasts and leaves the app
    fully untouched -- no half-swap, old sim never released."""
    app = _make_app()
    old_sim = app.sim

    def _boom(path, gpu=None):  # noqa: ARG001
        raise ValueError("checkpoint generation_version 1 != 8; Re-create the checkpoint.")

    monkeypatch.setattr(main, "load_checkpoint", _boom)
    app._load_state(Path("/tmp/stale.npz"))

    assert old_sim.released == 0
    assert app.sim is old_sim
    assert app._active_preset == ("gas_giant_warm", PresetSource.FACTORY)

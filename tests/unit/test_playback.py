"""Phase 5 playback state machine: pause/step/speed over draw_equirect.

`draw_equirect` is a pure app-level presentation layer over the existing,
unmodified `Simulation.tick()` -- it only decides how often / with what
`max_steps` to call `tick()`. These tests drive `draw_equirect` directly
against a FakeSim/FakeViewport (same headless pattern as
test_commit_loop.py) so the (playing, single_step_requested, speed) decision
table is pinned without needing a real GL context or imgui frame.
"""

from __future__ import annotations

import pytest

from gasgiant.diagnostics import PerfCounter

main = pytest.importorskip("gasgiant.app.main")
StudioApp = main.StudioApp


class FakeSim:
    """Mirrors Simulation's stepping surface without touching the GPU. tick()
    behaves like the real facade: returns False (no-op) once the target is
    reached, otherwise advances and returns True."""

    def __init__(self, target: int = 100) -> None:
        self.step_index = 0
        self._target = target
        self.tick_calls: list[int] = []
        self.rebuild_calls = 0

    @property
    def steps_done(self) -> int:
        return self.step_index

    @property
    def steps_target(self) -> int:
        return self._target

    @property
    def is_developed(self) -> bool:
        return self.step_index >= self._target

    def tick(self, max_steps: int = 2) -> bool:
        self.tick_calls.append(max_steps)
        remaining = self._target - self.step_index
        if remaining <= 0:
            return False
        self.step_index += min(max_steps, remaining)
        return True

    def rebuild(self) -> None:
        self.rebuild_calls += 1
        self.step_index = 0


class FakeViewport:
    def __init__(self) -> None:
        self.stale = 0
        self.draw_calls = 0

    def mark_stale(self) -> None:
        self.stale += 1

    def draw(self, sim, width) -> None:  # noqa: ARG002 - matches EquirectViewport.draw
        self.draw_calls += 1


def _make_app(*, playing: bool, steps_per_frame: int, single_step: bool, target: int = 100) -> StudioApp:
    app = StudioApp.__new__(StudioApp)
    app.sim = FakeSim(target=target)
    app.viewport = FakeViewport()
    app.render_perf = PerfCounter()
    app._playing = playing
    app._steps_per_frame = steps_per_frame
    app._single_step_requested = single_step
    return app


def test_playing_ticks_at_the_configured_speed() -> None:
    app = _make_app(playing=True, steps_per_frame=4, single_step=False)
    app.draw_equirect()
    assert app.sim.tick_calls == [4]
    assert app.viewport.stale == 1


def test_paused_does_not_tick() -> None:
    app = _make_app(playing=False, steps_per_frame=4, single_step=False)
    app.draw_equirect()
    assert app.sim.tick_calls == []
    assert app.viewport.stale == 0


def test_single_step_while_paused_ticks_once_then_stops() -> None:
    app = _make_app(playing=False, steps_per_frame=6, single_step=True)
    app.draw_equirect()
    assert app.sim.tick_calls == [6], "one tick at the current speed"
    assert app._single_step_requested is False, "consumed"

    app.draw_equirect()  # next frame: paused, request already consumed
    assert app.sim.tick_calls == [6], "no further ticking without another Step press"


def test_single_step_while_playing_does_not_double_tick() -> None:
    app = _make_app(playing=True, steps_per_frame=2, single_step=True)
    app.draw_equirect()
    assert app.sim.tick_calls == [2], "exactly one tick call, not two, this frame"
    assert app._single_step_requested is False


def test_speed_selector_changes_tick_argument_only() -> None:
    """The speed selector must only change what max_steps is passed to
    tick() -- never tick()'s own internal stepping logic (unchanged here;
    FakeSim.tick faithfully mirrors the real signature/behavior)."""
    for speed in (1, 2, 4, 8, 16, main.MAX_STEPS_PER_FRAME):
        app = _make_app(playing=True, steps_per_frame=speed, single_step=False)
        app.draw_equirect()
        assert app.sim.tick_calls == [speed]


def test_max_speed_is_bounded() -> None:
    """"Max" must not be an unbounded/until-developed loop -- it's a large but
    finite steps-per-frame value, so a single frame can't stall for seconds
    on a big dev_steps target."""
    assert main.MAX_STEPS_PER_FRAME < 10_000
    labels = dict(main.SPEED_OPTIONS)
    assert labels[main.MAX_STEPS_PER_FRAME] == "Max"


def test_viewport_not_marked_stale_when_tick_is_a_noop() -> None:
    """Once the sim is fully developed, tick() returns False and the viewport
    must not be told to re-derive."""
    app = _make_app(playing=True, steps_per_frame=4, single_step=False, target=0)
    app.draw_equirect()
    assert app.sim.tick_calls == [4]
    assert app.viewport.stale == 0


def test_rebuild_resets_the_dev_run_state() -> None:
    """The engine hook the "Restart dev" button calls."""
    app = _make_app(playing=True, steps_per_frame=4, single_step=False)
    app.sim.step_index = 42
    app.sim.rebuild()
    assert app.sim.step_index == 0
    assert app.sim.rebuild_calls == 1

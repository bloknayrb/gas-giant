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
        # T4: draw_equirect skips its storm-tool block when no image rect was
        # captured, keeping this headless path a pure tick/blit body.
        self.image_rect_min = None
        self.image_rect_max = None

    def mark_stale(self) -> None:
        self.stale += 1

    def draw(self, sim, width, **kwargs) -> None:  # noqa: ARG002 - matches EquirectViewport.draw
        self.draw_calls += 1

    def draw_markers(self, cast, **kwargs) -> None:  # noqa: ARG002 - matches EquirectViewport
        pass


def _make_app(*, playing: bool, steps_per_frame: int, single_step: bool, target: int = 100) -> StudioApp:
    app = StudioApp.__new__(StudioApp)
    app.sim = FakeSim(target=target)
    app.viewport = FakeViewport()
    app.render_perf = PerfCounter()
    app._playing = playing
    app._steps_per_frame = steps_per_frame
    app._single_step_requested = single_step
    # T5 A/B compare state read by draw_equirect when calling viewport.draw.
    app._compare_mode = "off"
    app._snapshot_a = None
    app._flash_show_a = False
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


# -- #1: paused mid-development must not read as a hang -------------------------
#
# tick() no-ops once developed, so pause deliberately freezes the development
# animation. The bug was cosmetic: draw_perf kept animating the "recomputing..."
# spinner while paused and stalled, so a deliberate Pause looked like a hang.
# The label is now honest about the paused state.


def test_dev_progress_label_paused_is_static_and_actionable() -> None:
    label = main._dev_progress_label(done=10, target=100, playing=False, recomputing=True, spinner="/")
    assert label == "paused 10/100 (Play/Step to develop)"
    assert "recomputing" not in label


def test_dev_progress_label_playing_while_recomputing_animates() -> None:
    label = main._dev_progress_label(done=10, target=100, playing=True, recomputing=True, spinner="/")
    assert label == "/ recomputing... 10/100"


def test_dev_progress_label_playing_plain_progress() -> None:
    label = main._dev_progress_label(done=10, target=100, playing=True, recomputing=False, spinner="/")
    assert label == "developing 10/100"


# -- W10a (review B1-1): the dev run must be labeled with a verb, an ETA once
# measurable, and surfaced where the user is looking (viewport overlay), so a
# first launch never reads as a hang or a finished-but-ugly planet. The label
# deliberately promises no speed-combo speedup: measured throughput is flat
# across steps-per-frame (see DEFAULT_STEPS_PER_FRAME), so the honest signal
# is the ETA itself.


def test_dev_progress_label_playing_includes_eta_when_available() -> None:
    label = main._dev_progress_label(
        done=10, target=100, playing=True, recomputing=False, spinner="/", eta_seconds=330.0
    )
    assert label == "developing 10/100 (~6m left)"


def test_format_eta_rounds_up_and_switches_units() -> None:
    assert main._format_eta(45.0) == "~45s left"
    assert main._format_eta(59.4) == "~60s left"
    assert main._format_eta(61.0) == "~2m left"
    assert main._format_eta(330.0) == "~6m left"


def test_dev_rate_sampler_needs_two_seconds_of_samples() -> None:
    s = main.DevRateSampler()
    s.add(now=100.0, steps_done=0)
    s.add(now=101.0, steps_done=3)
    assert s.eta_seconds(now=101.0, remaining=90) is None, "under 2 s of span: no ETA yet"
    s.add(now=102.5, steps_done=8)
    eta = s.eta_seconds(now=102.5, remaining=92)
    assert eta is not None
    assert eta == pytest.approx(92 / (8 / 2.5))


def test_dev_rate_sampler_ignores_stalled_rate() -> None:
    s = main.DevRateSampler()
    s.add(now=100.0, steps_done=5)
    s.add(now=103.0, steps_done=5)  # paused/stalled: no progress
    assert s.eta_seconds(now=103.0, remaining=50) is None


def test_dev_rate_sampler_reset_forgets_history() -> None:
    s = main.DevRateSampler()
    s.add(now=100.0, steps_done=0)
    s.add(now=103.0, steps_done=30)
    s.reset()
    assert s.eta_seconds(now=103.0, remaining=50) is None


def test_dev_overlay_text_while_developing_and_playing() -> None:
    text = main._dev_overlay_text(done=238, target=1256, playing=True, eta_seconds=372.0)
    assert text == "Developing planet — 238/1256 (~7m left)"


def test_dev_overlay_text_without_eta_yet() -> None:
    text = main._dev_overlay_text(done=4, target=1256, playing=True, eta_seconds=None)
    assert text == "Developing planet — 4/1256"


def test_dev_overlay_text_hidden_when_done_or_paused() -> None:
    assert main._dev_overlay_text(done=1256, target=1256, playing=True, eta_seconds=None) is None
    assert main._dev_overlay_text(done=10, target=1256, playing=False, eta_seconds=None) is None


def test_default_speed_is_measured_choice_and_valid_option() -> None:
    """W10a measurement (2026-07-02, RTX 3070, gas_giant_warm): GUI dev-run
    throughput is FLAT across steps-per-frame (~3.0 steps/s at spf=2 AND
    spf=8; frame time just grows 653 ms -> 2379 ms), so raising the default
    buys no wall-time and only adds input latency. The default stays 2; the
    first-launch fix is the honest label + ETA + overlay, not a speed bump.
    Must stay a listed SPEED_OPTIONS value so the combo shows it."""
    assert main.DEFAULT_STEPS_PER_FRAME == 2
    assert main.DEFAULT_STEPS_PER_FRAME in [v for v, _ in main.SPEED_OPTIONS]

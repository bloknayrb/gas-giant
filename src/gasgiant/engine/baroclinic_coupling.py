"""M3 baroclinic -> v1.6 coupling controller.

Interleaves the v1.6 development run with an EVOLVING baroclinic source: every
`update_every` v1.6 steps, advance the baroclinic solver `baro_steps_per_update`
steps, re-derive the coherent source, and re-upload it. Records per-phase wall
time so the cadence/residency decision is measured, not assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from gasgiant.engine.facade import Simulation
from gasgiant.sim.baroclinic_driver import BaroclinicSourceDriver

# If (baro + upload) wall time exceeds this fraction of v1.6 wall time, the CPU
# source cadence is a real cost and a GPU-resident baroclinic solver is worth
# considering; otherwise option-(a) (CPU advance + texture re-upload) suffices.
RESIDENCY_FRACTION = 0.25


@dataclass
class CouplingStats:
    v16_steps: int = 0
    source_updates: int = 0
    baro_seconds: float = 0.0
    upload_seconds: float = 0.0
    v16_seconds: float = 0.0
    baro_outcropped: bool = False  # the baroclinic source froze (outcropped) mid-run


def residency_recommendation(stats: CouplingStats) -> str:
    """'consider-residency' if the CPU source cadence dominates, else
    'option-a-sufficient'."""
    overhead = stats.baro_seconds + stats.upload_seconds
    if stats.v16_seconds <= 0.0:
        return "option-a-sufficient"
    return ("consider-residency"
            if overhead / stats.v16_seconds > RESIDENCY_FRACTION
            else "option-a-sufficient")


def run_coupled(sim: Simulation, driver: BaroclinicSourceDriver, gain: float,
                update_every: int = 32, baro_steps_per_update: int = 400) -> CouplingStats:
    """Develop `sim` to completion while injecting the evolving baroclinic source.
    Returns timing/coverage stats."""
    stats = CouplingStats()
    while not sim.is_developed:
        t0 = time.perf_counter()
        driver.advance(baro_steps_per_update)
        t1 = time.perf_counter()
        sim.set_external_vorticity_source(driver.current_source(), gain=gain)
        t2 = time.perf_counter()

        remaining = sim.steps_target - sim.steps_done
        n = min(update_every, remaining)
        sim.tick(n)  # advances n v1.6 steps + flags the preview dirty
        t3 = time.perf_counter()

        stats.baro_seconds += t1 - t0
        stats.upload_seconds += t2 - t1
        stats.v16_seconds += t3 - t2
        stats.v16_steps += n
        stats.source_updates += 1
    stats.baro_outcropped = driver.outcropped
    return stats

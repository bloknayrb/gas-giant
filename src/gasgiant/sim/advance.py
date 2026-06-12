"""The single per-step registry advance shared by every live stepping path.

Events (outbreak spawn/decay) mutate the registry, then the whole population
drifts with the ambient zonal flow. Keeping this in one function means any
future registry evolution (mergers, debris aging) automatically applies to
every caller — the v1.1 checkpoint bug happened precisely because replay
re-implemented a subset of the live step's registry mutations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gasgiant.sim.vortices import resolve_mergers

if TYPE_CHECKING:
    from gasgiant.params.model import StormsParams
    from gasgiant.sim.events import EventSchedule
    from gasgiant.sim.profiles import LatProfiles
    from gasgiant.sim.vortices import VortexRegistry


def advance_registry(
    registry: VortexRegistry,
    profiles: LatProfiles,
    dt: float,
    step_index: int,
    events: EventSchedule | None = None,
    storms: StormsParams | None = None,
) -> list[tuple[float, float, float, float]]:
    """Advance the vortex registry by one step; returns the active outbreak
    outflow impulses as (lon, lat, radius, strength) for the velocity kernel."""
    impulses: list[tuple[float, float, float, float]] = []
    if events is not None:
        impulses = events.apply(step_index, registry)
    registry.drift(profiles, dt)
    if storms is not None and storms.merge_rate > 0.0:
        resolve_mergers(registry, profiles, storms)
    return impulses

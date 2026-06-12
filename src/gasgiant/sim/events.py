"""Discrete events during the development run.

Convective outbreaks (Saturn Great-White-Spot class): at a seeded step a
brilliant plume erupts at a zone latitude — a bright tracer stamp (a zero-
circulation registry entry whose brightness decays over its lifetime, picked
up automatically by the relaxation stamps) plus a brief divergent outflow
impulse in the velocity field. The ambient shear then draws the plume into
the planet-girdling turbulent collar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gasgiant.params.model import PlanetParams
from gasgiant.params.seeds import subseed
from gasgiant.sim.bands import BandLayout
from gasgiant.sim.vortices import Vortex, VortexRegistry

KIND_OUTBREAK = 6.0
LIFETIME = 160       # steps from eruption to fully sheared out
RAMP = 12            # outflow spin-up steps
RADIUS = 0.045       # radians
BRIGHTNESS = 0.5
OUTFLOW = 0.35       # peak outflow speed


@dataclass
class Outbreak:
    step: int
    lat: float
    lon: float
    vortex: Vortex | None = None


@dataclass
class EventSchedule:
    outbreaks: list[Outbreak] = field(default_factory=list)
    strength: float = 1.0

    @classmethod
    def generate(cls, params: PlanetParams, bands: BandLayout) -> EventSchedule:
        rng = subseed(params.seed, "events")
        sched = cls(strength=params.storms.outbreak_strength)
        count = params.storms.outbreak_count
        if count == 0 or params.sim.dev_steps < 50:
            return sched
        values = bands.values
        median = float(np.median(values))
        zone_centers = [
            0.5 * (bands.edges[j] + bands.edges[j + 1])
            for j in range(len(values))
            if values[j] >= median and 0.25 < abs(0.5 * (bands.edges[j] + bands.edges[j + 1])) < 1.0
        ]
        if not zone_centers:
            return sched
        for _ in range(count):
            step = int(rng.uniform(0.25, 0.65) * params.sim.dev_steps)
            lat = float(zone_centers[rng.integers(0, len(zone_centers))])
            lon = float(rng.uniform(-np.pi, np.pi))
            sched.outbreaks.append(Outbreak(step=step, lat=lat, lon=lon))
        return sched

    def apply(self, step: int, registry: VortexRegistry) -> list[tuple[float, float, float, float]]:
        """Spawn/age/retire outbreaks; returns active outflow impulses as
        (lon, lat, radius, strength) for the velocity kernel."""
        impulses: list[tuple[float, float, float, float]] = []
        for ob in self.outbreaks:
            age = step - ob.step
            if age < 0:
                continue
            if age > LIFETIME:
                if ob.vortex is not None:
                    # Identity-based removal: dataclass == is field equality, so
                    # list.remove could drop a different but field-equal vortex.
                    registry.vortices = [
                        v for v in registry.vortices if v is not ob.vortex
                    ]
                    ob.vortex = None
                continue
            if ob.vortex is None:
                ob.vortex = Vortex(
                    ob.lat, ob.lon, RADIUS, 0.0, KIND_OUTBREAK,
                    tint=0.0, brightness=BRIGHTNESS * self.strength,
                )
                registry.vortices.append(ob.vortex)
            decay = 1.0 - age / LIFETIME
            ob.vortex.brightness = BRIGHTNESS * self.strength * decay
            if len(impulses) < 2:
                ramp = min(age / RAMP, 1.0) * decay
                impulses.append(
                    (ob.vortex.lon, ob.vortex.lat, RADIUS * 1.5,
                     OUTFLOW * self.strength * ramp)
                )
        return impulses

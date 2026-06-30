"""Discrete events during the development run.

Convective white-plume outbreaks (Great-White-Spot / SEB-revival class): at a
seeded step a cluster of brilliant ammonia-white plumes erupts in a BELT and
shears out along it into a planet-girdling turbulent streak. Each eruption is a
TRAIN of zero-circulation registry entries (no psi, so the belt shear folds them
into filaments rather than a coherent vortex) bracketing the belt center, plus a
gentle divergent outflow impulse. Placing them in belts + stamping a ring (no
dome, see vortex_stamp.glsl) is what keeps them reading as convective churn
rather than a competing second GRS (the prior zone-placed, domed version did).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gasgiant.params.model import PlanetParams
from gasgiant.params.seeds import subseed
from gasgiant.sim.bands import BandLayout
from gasgiant.sim.vortices import MAX_VORTEX_LAT, Vortex, VortexRegistry

KIND_OUTBREAK = 6.0
LIFETIME = 300       # steps from eruption to fully sheared out (long streak)
RAMP = 16            # outflow spin-up steps
RADIUS = 0.048       # radians (plume scale, below the oval size range)
BRIGHTNESS = 1.9     # bright ammonia-white (must read at DISK scale, not the diff)
OUTFLOW = 0.18       # peak outflow speed (gentle -- not a vortex-scale kick)
TRAIN_N = 6          # plumes per eruption (a belt-girdling train, not one spot)
TRAIN_LAT_SPREAD = 0.035  # radians, kept tight so the chain stays in the belt core
TRAIN_LON_STEP = 0.06     # downstream longitude offset between successive knots


@dataclass
class Outbreak:
    step: int
    lat: float
    lon: float
    radius: float = RADIUS    # per-knot (head knots larger, tail knots smaller)
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
        # DARK belts only (review: a plume on a light zone/boundary is white-on-
        # white and vanishes). Take the darkest half of the belts so the bright-
        # on-dark convective-revival contrast is the rule, not the exception.
        belts = [
            (0.5 * (bands.edges[j] + bands.edges[j + 1]), float(values[j]))
            for j in range(len(values))
            if values[j] < median
            and 0.20 < abs(0.5 * (bands.edges[j] + bands.edges[j + 1])) < 1.0
        ]
        if not belts:
            return sched
        belts.sort(key=lambda cv: cv[1])              # darkest first
        dark_belts = belts[: max(1, (len(belts) + 1) // 2)]
        for _ in range(count):
            # Later window so the development snapshot catches plumes across
            # their life: freshly-bright ones plus older ones already sheared
            # into streaks (LIFETIME peaks brightness at mid-life).
            step0 = int(rng.uniform(0.55, 0.85) * params.sim.dev_steps)
            center = float(dark_belts[rng.integers(0, len(dark_belts))][0])
            base_lon = float(rng.uniform(-np.pi, np.pi))
            # A TRAIN strung DOWNSTREAM along the belt: a head knot plus a chain
            # offset in longitude with a tight latitude bracket and a size
            # falloff, so each eruption reads as a cluster (not one blob). The
            # belt's latitudinal shear then stretches the chain further.
            for k in range(TRAIN_N):
                frac = k / max(TRAIN_N - 1, 1) - 0.5      # -0.5 .. +0.5
                lat = float(np.clip(center + frac * TRAIN_LAT_SPREAD,
                                    -MAX_VORTEX_LAT, MAX_VORTEX_LAT))
                lon = float((base_lon + k * TRAIN_LON_STEP + rng.normal(0.0, 0.02)
                             + np.pi) % (2.0 * np.pi) - np.pi)
                radius = RADIUS * (1.0 - 0.45 * k / max(TRAIN_N - 1, 1))  # head big, tail small
                # Stagger so the train unfurls in sequence, not all at once.
                step = step0 + k * int(0.015 * params.sim.dev_steps) \
                    + int(rng.uniform(0.0, 0.04) * params.sim.dev_steps)
                sched.outbreaks.append(
                    Outbreak(step=step, lat=lat, lon=lon, radius=radius)
                )
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
                    ob.lat, ob.lon, ob.radius, 0.0, KIND_OUTBREAK,
                    tint=0.0, brightness=BRIGHTNESS * self.strength,
                )
                registry.vortices.append(ob.vortex)
            decay = 1.0 - age / LIFETIME
            ob.vortex.brightness = BRIGHTNESS * self.strength * decay
            if len(impulses) < 2:
                ramp = min(age / RAMP, 1.0) * decay
                impulses.append(
                    (ob.vortex.lon, ob.vortex.lat, ob.radius * 1.5,
                     OUTFLOW * self.strength * ramp)
                )
        return impulses

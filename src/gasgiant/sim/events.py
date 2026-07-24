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
from gasgiant.sim.profiles import LatProfiles
from gasgiant.sim.resolution_scaling import (
    effective_dev_steps,
    scale_duration,
    scale_factor,
)
from gasgiant.sim.vortices import (
    MAX_VORTEX_LAT,
    Vortex,
    VortexRegistry,
    drift_compensated_lon,
)

KIND_OUTBREAK = 6.0
# ---------------------------------------------------------------------------
# Outbreak look constants (review B5-8 promote-or-document pass, W6):
# these stay MODULE CONSTANTS deliberately. They were calibrated by paired
# adversarial visual review (see the LEAD_BRIGHT note below) and define what
# an outbreak IS -- a belt-girdling convective plume train -- rather than a
# per-planet look axis. The user-facing axes are the params: outbreak_count,
# outbreak_strength (brightness+outflow amplitude), outbreak_latitude /
# outbreak_phase (where/when, B5-3), and outbreak_lat_min (the auto-candidate
# floor, promoted from a hardcoded 0.20 in this pass). Recorded LIMIT:
# outbreak_strength scales BRIGHTNESS and OUTFLOW in lockstep, so "brighter
# but dynamically gentler" stays inexpressible until a real need shows up --
# splitting them doubles the knob count for a look nobody has asked for yet.
# ---------------------------------------------------------------------------
LIFETIME = 300       # steps from eruption to fully sheared out (long streak)
RAMP = 16            # outflow spin-up steps
RADIUS = 0.048       # radians (plume scale, below the oval size range)
BRIGHTNESS = 1.9     # bright ammonia-white (must read at DISK scale, not the diff)
OUTFLOW = 0.18       # peak outflow speed (gentle -- not a vortex-scale kick)
TRAIN_N = 6          # plumes per eruption (a belt-girdling train, not one spot)
TRAIN_LAT_SPREAD = 0.035  # radians, kept tight so the chain stays in the belt core
TRAIN_LON_STEP = 0.06     # downstream longitude offset between successive knots
LEAD_BRIGHT = 1.8   # lead-knot brightness boost. Two adversarial reviewers found
                    # x1.3 imperceptible in the direct render (only in the 4x diff);
                    # x1.8 makes the head read on the dark belt without doming.
LEAD_RADIUS = 1.2   # lead-knot size boost (the head reads as the eruption source;
                    # kept modest so it stays a turbulent patch, not a round storm)
# NOTE: the knots' cool cast is set in the stamp (vortex_stamp.glsl KIND_OUTBREAK
# dT3 push), NOT via Vortex.tint -- that branch `continue`s before the shared
# `dT3 += b.z` line, so per-vortex tint is inert for outbreaks. The push had to be
# cut there (0.15->0.07) because it scales with b.w: a brighter knot is bluer
# unless the coefficient drops, which reviewers flagged would show at this amplitude.


@dataclass
class Outbreak:
    step: int
    lat: float
    lon: float
    radius: float = RADIUS    # per-knot (head knots larger, tail knots smaller)
    bright_mul: float = 1.0   # per-knot brightness multiplier (lead knot boosted)
    vortex: Vortex | None = None


@dataclass
class EventSchedule:
    outbreaks: list[Outbreak] = field(default_factory=list)
    strength: float = 1.0
    # Resolution-invariant step scale (see VortexRegistry.step_scale). apply()
    # scales the outbreak LIFETIME/RAMP by it; s == 1 leaves them unchanged.
    step_scale: float = 1.0

    @classmethod
    def generate(
        cls,
        params: PlanetParams,
        bands: BandLayout,
        profiles: LatProfiles | None = None,
        dt: float | None = None,
    ) -> EventSchedule:
        rng = subseed(params.seed, "events")
        s = scale_factor(params)
        sched = cls(strength=params.storms.outbreak_strength, step_scale=s)
        count = params.storms.outbreak_count
        # Intent guard on RAW dev_steps: dev_steps * dt is the physical run length,
        # invariant across resolution, so "is there a real run" is a raw-steps test.
        if count == 0 or params.sim.dev_steps < 50:
            return sched
        # Schedule anchors use the EFFECTIVE (s-scaled) run length so eruptions
        # fire at the same physical time / phase at any resolution.
        eff = effective_dev_steps(params)
        values = bands.values
        # DARK belts only (review: a plume on a light zone/boundary is white-on-
        # white and vanishes). Take the darkest half of the belts so the bright-
        # on-dark convective-revival contrast is the rule, not the exception.
        # Belt identity from BandLayout.is_belt (frozen at layout build) AND
        # darkness ordering from bands.values, which is the PRE-fade view by
        # construction (bands.belt_fade only touches BandLayout.stamp_values):
        # a belt faded toward zone-white must STAY an outbreak candidate with
        # its pre-fade darkness rank — the SEB-revival story is outbreaks
        # erupting in the faded belt.
        pin = params.storms.outbreak_latitude  # degrees, None = seeded belt pick
        belts = [
            (0.5 * (bands.edges[j] + bands.edges[j + 1]), float(values[j]))
            for j in range(len(values))
            if bands.is_belt[j]
            and params.storms.outbreak_lat_min
            < abs(0.5 * (bands.edges[j] + bands.edges[j + 1]))
            < 1.0
        ]
        if not belts and pin is None:
            return sched
        belts.sort(key=lambda cv: cv[1])              # darkest first
        dark_belts = belts[: max(1, (len(belts) + 1) // 2)]
        for _ in range(count):
            # Later window so the development snapshot catches plumes across
            # their life: freshly-bright ones plus older ones already sheared
            # into streaks (LIFETIME peaks brightness at mid-life). The pins
            # (outbreak_phase / outbreak_latitude) CONSUME the same seeded
            # draws and then override, so toggling a pin never reshuffles the
            # other outbreak properties.
            draw_phase = float(rng.uniform(0.55, 0.85))
            phase = params.storms.outbreak_phase
            step0 = int((draw_phase if phase is None else phase) * eff)
            # The 0.0 fallback is only reachable when pinned (belts empty).
            center = (
                float(dark_belts[rng.integers(0, len(dark_belts))][0])
                if dark_belts else 0.0
            )
            if pin is not None:
                center = float(np.deg2rad(pin))
            # Consume the seeded longitude draw unconditionally (RNG-stream
            # position must not depend on the pin), then override for a pin.
            base_lon = float(rng.uniform(-np.pi, np.pi))
            if params.storms.outbreak_longitude is not None:
                # Best-effort drift compensation: the plume knots carry no
                # circulation (zero-strength registry entries), but the sim
                # velocity advects their stamps at ~the zonal rate for their
                # post-eruption life. Inverse-compensate that drift over the
                # REMAINING (dev_steps - step0) steps at the eruption latitude
                # so the head lands at the requested rendered longitude. Only
                # the head is precise -- the belt shear folds the tail into a
                # streak (a recorded caveat). With no profiles/dt (default
                # call), seed the target directly, no compensation.
                remaining = eff - step0
                base_lon = drift_compensated_lon(
                    profiles, center, params.storms.outbreak_longitude,
                    dt if profiles is not None else None, remaining,
                )
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
                # Lead knot (k==0) reads as the eruption source: boost it so the
                # train is visible at disk scale, not only in the 4x diff.
                bright_mul = LEAD_BRIGHT if k == 0 else 1.0
                if k == 0:
                    radius *= LEAD_RADIUS
                # Stagger so the train unfurls in sequence, not all at once.
                step = step0 + k * int(0.015 * eff) \
                    + int(rng.uniform(0.0, 0.04) * eff)
                sched.outbreaks.append(
                    Outbreak(step=step, lat=lat, lon=lon, radius=radius,
                             bright_mul=bright_mul)
                )
        return sched

    def apply(self, step: int, registry: VortexRegistry) -> list[tuple[float, float, float, float]]:
        """Spawn/age/retire outbreaks; returns active outflow impulses as
        (lon, lat, radius, strength) for the velocity kernel."""
        impulses: list[tuple[float, float, float, float]] = []
        # Resolution-invariant lifetimes: an outbreak lives the same physical
        # time (same fraction of the run) at any resolution. s == 1 => LIFETIME/RAMP.
        lifetime = scale_duration(LIFETIME, self.step_scale)
        ramp_steps = scale_duration(RAMP, self.step_scale)
        for ob in self.outbreaks:
            age = step - ob.step
            if age < 0:
                continue
            if age > lifetime:
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
                    tint=0.0, brightness=BRIGHTNESS * self.strength * ob.bright_mul,
                )
                registry.vortices.append(ob.vortex)
            decay = 1.0 - age / lifetime
            ob.vortex.brightness = BRIGHTNESS * self.strength * ob.bright_mul * decay
            if len(impulses) < 2:
                ramp = min(age / ramp_steps, 1.0) * decay
                impulses.append(
                    (ob.vortex.lon, ob.vortex.lat, ob.radius * 1.5,
                     OUTFLOW * self.strength * ramp)
                )
        return impulses

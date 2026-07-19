"""The pydantic parameter tree.

Every tunable field carries metadata in ``json_schema_extra``:

- ``tier``: invalidation tier — what the engine must redo when the field changes
  (POST: re-derive maps only; VELOCITY: rebuild the velocity field, sim continues;
  RESTART: re-initialize the development run from step 0).
- ``rand``: (lo, hi) range used by seeded randomization, or None if the field is
  never randomized.
- ``log``: randomize/UI slider on a log scale.
- ``ui``: display group label for auto-generated panels.
- ``adv``: Basic/Advanced curation. False (default) = visible in Basic mode --
  a new field "gets UI for free" without the author having to think about it.
  True = hidden unless Advanced is toggled on (or an active search matches
  it) -- power-user/fine-tuning knobs, opt-in byte-identical-off-by-default
  levers, and preset-only fields.

Metadata is plain JSON data only — no callables, no GUI imports — so the core
stays GUI-agnostic in fact, not just in name.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_BANDS = 40


class Tier(StrEnum):
    POST = "post"
    VELOCITY = "velocity"
    RESTART = "restart"


_UNSET = object()


def pfield(
    default: Any = _UNSET,
    *,
    tier: Tier,
    lo: float | None = None,
    hi: float | None = None,
    rand: tuple[float, float] | None = None,
    log: bool = False,
    ui: str = "",
    adv: bool = False,
    fx: bool = False,
    spread: bool = False,
    description: str = "",
    factory: Any = None,
) -> Any:
    """A pydantic ``Field`` carrying the panel/randomize metadata (tier, ui, log,
    adv, rand, fx). Pass ``factory`` (a zero-arg callable) instead of ``default``
    for a MUTABLE default so every model instance gets its own fresh value rather
    than sharing a module-level list -- pydantic already deep-copies a plain
    default per instance, but a ``default_factory`` is the explicit,
    no-shared-singleton form (and matches the ``Field(default_factory=...)`` idiom
    the rest of the tree uses for its nested models).

    ``fx=True`` marks a DetailParams lever that lives in the DETAIL_FX kernel
    variant: render/detail.py derives its variant-selection predicate AND its
    build-time uniform tripwire (u_<field-name> must exist in the compiled fx
    program) from this flag, so a new fx lever cannot silently miss either sync
    point (the A2-6/A2-1 hand-list hazard). ``spread=True`` is the analogous flag
    for the SPREAD (uniform-detail-coverage) variant. Both are stored only when
    True, like rand -- they never affect the randomize draw order."""
    extra: dict[str, Any] = {"tier": tier.value, "ui": ui, "log": log, "adv": adv}
    if rand is not None:
        extra["rand"] = list(rand)
    if fx:
        extra["fx"] = True
    if spread:
        extra["spread"] = True
    if factory is not None:
        return Field(
            default_factory=factory, ge=lo, le=hi, description=description, json_schema_extra=extra
        )
    return Field(
        default,
        ge=lo,
        le=hi,
        description=description,
        json_schema_extra=extra,
    )


class _Params(BaseModel):
    """Strict base: unknown keys are errors (hand-edited preset typos must not
    silently become defaults), assignments re-validate."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GradientStop(_Params):
    pos: float = Field(ge=0.0, le=1.0)
    color: tuple[float, float, float]


DEFAULT_PALETTE = [
    GradientStop(pos=0.0, color=(0.36, 0.27, 0.21)),  # deep belt brown
    GradientStop(pos=0.35, color=(0.62, 0.48, 0.36)),  # warm belt tan
    GradientStop(pos=0.65, color=(0.83, 0.74, 0.62)),  # pale zone cream
    GradientStop(pos=1.0, color=(0.93, 0.89, 0.81)),  # bright zone white
]

DEFAULT_STORM_TINTS = [
    GradientStop(pos=0.0, color=(0.55, 0.60, 0.66)),  # festoon blue-gray
    GradientStop(pos=0.5, color=(0.72, 0.58, 0.45)),  # neutral
    GradientStop(pos=1.0, color=(0.78, 0.42, 0.30)),  # GRS salmon red
]

# Neutral band-tint default: a flat mid-gray gradient. With band_tint_strength
# at its default 0 the tint is never sampled, but a NEUTRAL, non-empty default
# means turning the strength up does not snap the planet toward an arbitrary
# color -- the artist paints latitudes in from a blank slate.
DEFAULT_BAND_TINT = [
    GradientStop(pos=0.0, color=(0.5, 0.5, 0.5)),  # south pole
    GradientStop(pos=1.0, color=(0.5, 0.5, 0.5)),  # north pole
]


class PaletteRow(_Params):
    """One palette gradient anchored at a signed latitude (degrees, north
    positive). Rows are blended across latitude at derive time; a single row
    reproduces the latitude-independent v1 palette exactly."""

    latitude: float = Field(0.0, ge=-90.0, le=90.0)
    stops: list[GradientStop]


def default_palette_rows() -> list[PaletteRow]:
    return [PaletteRow(latitude=0.0, stops=list(DEFAULT_PALETTE))]


def palette_rows_from_fit(rows: list[dict]) -> list[PaletteRow]:
    """Convert the plain-dict rows returned by ``gasgiant.palette.fit.calibrate``
    into validated ``PaletteRow`` models.

    The ``palette`` layer cannot import ``params`` (it sits below it), so the
    fit function returns plain dicts/arrays; this bridge — called by the CLI,
    GUI, and the ``calibrate_palette`` script — performs the model conversion in
    the ``params`` layer where ``PaletteRow``/``GradientStop`` live."""
    return [
        PaletteRow(
            latitude=float(row["latitude"]),
            stops=[
                GradientStop(pos=float(s["pos"]), color=tuple(float(c) for c in s["color"]))
                for s in row["stops"]
            ],
        )
        for row in rows
    ]


class BandTemplate(_Params):
    """Explicit band skeleton: edge latitudes (degrees, strictly descending,
    +-90 endpoints, interior within +-76 -- the polar cap systems own the
    rest) with per-band color-index values and cloud-top heights.

    Values and heights are used VERBATIM: none of the seeded value seasoning
    (value_contrast scaling, value/hue jitter) applies on the template path,
    because every consumer re-derives zone/belt identity as
    ``values < median(values)`` from the final numbers -- jitter on both a
    value and the median can silently flip a band's identity. Verbatim
    values make identity deterministic; the validator requires the derived
    identity mask to strictly alternate zone/belt. (An odd-count template
    with belts in the majority can never satisfy the convention -- the
    median IS the top belt value then; merge or split to an even count.)"""

    edges_deg: list[float]
    values: list[float]
    heights: list[float]

    @model_validator(mode="after")
    def _validate(self) -> BandTemplate:
        e = self.edges_deg
        n = len(e) - 1
        if n < 2:
            raise ValueError("template needs at least 2 bands")
        if n > MAX_BANDS:
            raise ValueError(f"template has {n} bands; the cap is {MAX_BANDS}")
        if e[0] != 90.0 or e[-1] != -90.0:
            raise ValueError("edges_deg must start at +90 and end at -90")
        if any(b >= a for a, b in zip(e, e[1:], strict=False)):
            raise ValueError("edges_deg must be strictly descending")
        if any(abs(x) > 76.0 for x in e[1:-1]):
            raise ValueError(
                "interior edges must lie within +-76 deg (polar caps own the rest)"
            )
        if len(self.values) != n or len(self.heights) != n:
            raise ValueError("values/heights need len(edges_deg) - 1 entries")
        if any(not 0.0 <= v <= 1.0 for v in self.values + self.heights):
            raise ValueError("values/heights must lie in [0, 1]")
        med = statistics.median(self.values)
        mask = [v < med for v in self.values]
        if any(a == b for a, b in zip(mask, mask[1:], strict=False)):
            raise ValueError(
                "zone/belt identity (values < median) must strictly alternate; "
                "adjust values (odd belts-majority layouts cannot satisfy it)"
            )
        return self


class BandsParams(_Params):
    count: int = pfield(
        14, tier=Tier.RESTART, lo=2, hi=MAX_BANDS, rand=(6, 24), ui="Bands",
        description="Number of zones+belts pole to pole",
    )
    width_jitter: float = pfield(
        0.35, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.1, 0.6), ui="Bands",
        description="Randomness of band width distribution",
    )
    edge_softness: float = pfield(
        0.012, tier=Tier.RESTART, lo=0.001, hi=0.1, rand=(0.005, 0.03), log=True,
        adv=True, ui="Bands",
        description="Half-width of band-edge transitions, radians of latitude"
                    " (1 rad = 57.3 deg; default 0.012 rad is about 0.7 deg)",
    )
    value_contrast: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=2.0, rand=(0.6, 1.3), ui="Bands",
        description="Zone/belt brightness separation multiplier",
    )
    warp_amount: float = pfield(
        0.035, tier=Tier.RESTART, lo=0.0, hi=0.3, rand=(0.01, 0.09), ui="Bands",
        description="Band-boundary meander amplitude, radians of latitude"
                    " (1 rad = 57.3 deg; default 0.035 rad is about 2 deg)",
    )
    warp_freq: float = pfield(
        3.0, tier=Tier.RESTART, lo=0.5, hi=16.0, rand=(1.5, 6.0), log=True, adv=True, ui="Bands",
        description="Band-boundary meander spatial frequency",
    )
    detail_amount: float = pfield(
        0.10, tier=Tier.RESTART, lo=0.0, hi=0.5, rand=(0.04, 0.2), ui="Bands",
        description="Small-scale color-index noise amplitude",
    )
    hue_jitter: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=0.15, rand=(0.0, 0.08), adv=True, ui="Bands",
        description="Per-band color-index offset along the palette (NEB-orange vs "
                    "SEB-brown variation); seeded independently of the band layout",
    )
    variance_amount: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=0.3, rand=(0.02, 0.12), adv=True, ui="Bands",
        description="Within-band longitudinal color drift (real belts hold several "
                    "hues at once, varying slowly with longitude)",
    )
    faded_sector: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.0, 0.7), adv=True, ui="Bands",
        description="SEB-fade: one belt gets a pale desaturated sector spanning "
                    "~100 degrees of longitude",
    )
    belt_fade: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Bands",
        description="Whole-belt fade (the SEB-fade epoch): blends the target "
                    "band's stamped color toward the mean of its neighboring "
                    "bands, all the way around the planet -- at 1.0 a faded "
                    "belt reads as a pale ghost band at zone level. VISUAL "
                    "only (recorded LIMIT): the belt keeps belt-like churn/"
                    "dynamics and stays a storm host and outbreak candidate, "
                    "which is the real SEB-fade phenomenology (revival "
                    "outbreaks erupt IN the faded belt). Target band = "
                    "faded_band_index, or the widest low/mid belt when that "
                    "is unset. 0 = off (byte-identical)",
    )
    faded_band_index: int | None = pfield(
        None, tier=Tier.RESTART, lo=0, hi=MAX_BANDS - 1, adv=True, ui="Bands",
        description="Band targeted by belt_fade AND the faded_sector "
                    "longitude window (index 0 = northernmost band). None = "
                    "auto: the widest belt within ~52 deg of the equator -- "
                    "note the shipped Jupiter template's SEB wins that pick "
                    "by only 0.01 deg over the NEB, so set this explicitly "
                    "when the target matters. Pointing it at a ZONE is "
                    "allowed (the ochre-EZ recipe: the zone blends toward "
                    "its belt neighbors). Validated against the band count",
    )
    contrast_envelope: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.3, 0.8), adv=True, ui="Bands",
        description="Banding contrast collapse poleward of ~45 deg toward mottle "
                    "(the real latitude-contrast profile)",
    )
    lane_density: float = pfield(
        0.0, tier=Tier.VELOCITY, lo=0.0, hi=1.0, rand=(0.0, 0.8), adv=True, ui="Bands",
        description="Thin dark lane lines at jet cores, drawn analytically at "
                    "derive time (a 1-3 px line cannot survive the sim grid)",
    )
    edge_diversity: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.2, 0.8), adv=True, ui="Bands",
        description="Per-edge softness variation: some band edges diffuse, some "
                    "sharp (uniform edges are a procedural tell)",
    )
    width_tail: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.0, 0.7), adv=True, ui="Bands",
        description="Heavier-tailed band width distribution (real maps mix very "
                    "broad zones with thin strips)",
    )
    detail_freq: float = pfield(
        12.0, tier=Tier.RESTART, lo=2.0, hi=64.0, rand=(6.0, 24.0), log=True, adv=True, ui="Bands",
        description="Small-scale noise spatial frequency",
    )
    template: BandTemplate | None = pfield(
        None, tier=Tier.RESTART, adv=True, ui="Bands",
        description="Explicit band skeleton (edge latitudes + per-band values/"
                    "heights) replacing the seeded layout; preset-only -- value "
                    "seasoning (value_contrast, hue_jitter, width knobs) is "
                    "inert when set",
    )

    @model_validator(mode="after")
    def _validate_faded_band_index(self) -> BandsParams:
        if self.faded_band_index is not None:
            n = len(self.template.values) if self.template is not None else self.count
            if self.faded_band_index >= n:
                raise ValueError(
                    f"faded_band_index={self.faded_band_index} is out of range: the "
                    f"layout has {n} bands (indices 0..{n - 1})"
                )
        return self


class SimParams(_Params):
    resolution: int = pfield(
        2048, tier=Tier.RESTART, lo=512, hi=8192, ui="Simulation",
        description="Sim grid width (2:1 equirect); 2048 interactive, 4096+ for final quality",
    )
    dev_steps: int = pfield(
        500, tier=Tier.RESTART, lo=0, hi=3000, rand=(300, 800), ui="Simulation",
        description="Development steps: how long structures evolve before the snapshot",
    )
    dt_scale: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.2, hi=3.0, ui="Simulation",
        description="Time-step multiplier (peak jet displacement ~1.2 cells at 1.0)",
    )


class JetsParams(_Params):
    strength: float = pfield(
        1.0, tier=Tier.VELOCITY, lo=0.0, hi=3.0, rand=(0.6, 1.6), ui="Jets",
        description="Global zonal jet speed multiplier",
    )
    equatorial_speed: float = pfield(
        1.6, tier=Tier.VELOCITY, lo=-3.0, hi=4.0, rand=(0.5, 2.5), ui="Jets",
        description="Equatorial superrotation jet peak speed (negative ="
                    " retrograde, flowing against the planet's rotation)",
    )
    equatorial_width: float = pfield(
        0.12, tier=Tier.VELOCITY, lo=0.03, hi=0.4, rand=(0.07, 0.25), ui="Jets",
        description="Equatorial jet half-width, radians of latitude"
                    " (1 rad = 57.3 deg; default 0.12 rad is about 7 deg)",
    )
    polar_decay: float = pfield(
        0.5, tier=Tier.VELOCITY, lo=0.0, hi=1.0, rand=(0.3, 0.8), ui="Jets",
        description="How strongly jet amplitudes decay toward the poles",
    )
    local_jet_speed: float = pfield(
        0.0, tier=Tier.RESTART, lo=-3.0, hi=3.0, adv=True, ui="Jets",
        description="Extra local zonal jet, additive on top of the banded jet "
                    "profile (0 = off, byte-identical). Negative = retrograde. "
                    "Authors a westward SEBs-analog jet under an anticyclonic "
                    "hero storm; the amplitude is applied PRE jets.strength and "
                    "pre polar_fade (same convention as equatorial_speed), so "
                    "the effective peak speed is speed * jets.strength -- a "
                    "later jets.strength retune rescales it too. RESTART tier: "
                    "the live-edit VELOCITY path rebuilds the jet profile "
                    "without regenerating storms, which would flip the ambient "
                    "shear sign under stale storm rotations",
    )
    local_jet_latitude: float = pfield(
        -20.0, tier=Tier.RESTART, lo=-60.0, hi=60.0, adv=True, ui="Jets",
        description="Center latitude of the local zonal jet (degrees, north "
                    "positive). Only used while local_jet_speed is nonzero",
    )
    local_jet_width: float = pfield(
        0.05, tier=Tier.RESTART, lo=0.01, hi=0.3, adv=True, ui="Jets",
        description="Half-width of the local zonal jet, radians of latitude "
                    "(1 rad = 57.3 deg; default 0.05 rad is about 2.9 deg). "
                    "Only used while local_jet_speed is nonzero",
    )


class TurbulenceParams(_Params):
    intensity: float = pfield(
        1.0, tier=Tier.VELOCITY, lo=0.0, hi=3.0, rand=(0.5, 1.8), ui="Turbulence",
        description="Global turbulence (curl-noise) amplitude",
    )
    shear_coupling: float = pfield(
        1.0, tier=Tier.VELOCITY, lo=0.0, hi=3.0, rand=(0.5, 1.5), adv=True, ui="Turbulence",
        description="Extra turbulence where jet shear is strong",
    )
    belt_boost: float = pfield(
        1.6, tier=Tier.VELOCITY, lo=1.0, hi=4.0, rand=(1.2, 2.5), ui="Turbulence",
        description="Turbulence multiplier inside dark belts (cyclonic ="
                    " spinning with the local planetary rotation; the"
                    " storm-prone bands)",
    )
    scale: float = pfield(
        6.0, tier=Tier.VELOCITY, lo=1.0, hi=32.0, rand=(4.0, 12.0), log=True,
        adv=True, ui="Turbulence",
        description="Base spatial frequency of the turbulence noise",
    )
    evolution_rate: float = pfield(
        0.012, tier=Tier.VELOCITY, lo=0.0, hi=0.1, adv=True, ui="Turbulence",
        description="How fast the turbulence pattern decorrelates per step",
    )
    relax_tau: float = pfield(
        350.0, tier=Tier.RESTART, lo=50.0, hi=2000.0, log=True, adv=True, ui="Turbulence",
        description="Relaxation time (steps) pulling band color/height back toward the stamp",
    )
    replenish_rate: float = pfield(
        0.015, tier=Tier.RESTART, lo=0.0, hi=0.5, ui="Turbulence",
        description="Fresh detail-noise blended into the detail tracer per step. "
                    "High values (~0.3) keep quiescent zone bands detailed where the "
                    "zonal jets would otherwise smear the detail away to ~half the belts'",
    )
    kh_amplitude: float = pfield(
        0.35, tier=Tier.VELOCITY, lo=0.0, hi=2.0, rand=(0.1, 0.8), adv=True, ui="Turbulence",
        description="Kelvin-Helmholtz wave amplitude along high-shear band boundaries",
    )
    kh_wavenumber: int = pfield(
        24, tier=Tier.VELOCITY, lo=4, hi=80, rand=(14, 40), adv=True, ui="Turbulence",
        description="KH billow longitudinal wavenumber",
    )
    belt_replenish: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=0.08, adv=True, ui="Turbulence",
        description="Extra fine detail-noise replenished per step inside belts (emergent filaments)",  # noqa: E501
    )
    belt_replenish_scale: float = pfield(
        2.0, tier=Tier.RESTART, lo=1.0, hi=4.0, adv=True, ui="Turbulence",
        description="Belt replenishment frequency multiplier relative to the base detail frequency",
    )


def hero_latitude_cap(hero_radius: float) -> float:
    """Radius-coupled |latitude| limit for a placed storm stamp (degrees): the
    stamp must stay clear of the 63 deg storm-free exchange band. Single
    source of truth for the hero validator, the GUI's pin-slider bounds
    (B4-2), and the accent-oval validator/auto-placement, so no path can
    produce a latitude another path would reject."""
    return 63.0 - 206.3 * hero_radius


class CastKind(StrEnum):
    """The storm archetype a cast entry stamps. Each maps to a KIND_* constant
    plus a per-kind base strength law and sign convention in the generator
    (``sim/vortices.py::_add_cast``)."""
    HERO = "hero"     # GRS-class giant anticyclone (wake + solid-core capable)
    OVAL = "oval"     # white-oval anticyclone
    BARGE = "barge"   # brown-barge cyclone (co-rotates with the ambient shear)
    PEARL = "pearl"   # small bright string-of-pearls oval


class StormOverride(_Params):
    """One art-directed storm placed by hand (the 'cast list'): kind, rendered
    position, size, and an optional appearance override. Strict (unknown keys
    error, like every other params group). Cast entries are DETERMINISTIC (no
    RNG, no ``rand`` metadata) -- they are placed verbatim after the seeded
    populations and are exempt from the population cap and runtime mergers, so
    a director's storm survives the whole development run where the artist put
    it. ``lat_deg``/``lon_deg`` name the RENDERED (end-of-dev-run) position:
    the generator inverse-compensates the zonal drift (like the T1 pins) so the
    storm lands on target when the snapshot is taken."""

    kind: CastKind = pfield(
        CastKind.OVAL, tier=Tier.RESTART, ui="Cast",
        description="Which storm archetype to stamp: a GRS-class hero "
                    "anticyclone, a white oval, a brown-barge cyclone, or a "
                    "small pearl. The kind selects the base velocity law, the "
                    "rotation sign, and the default appearance",
    )
    lat_deg: float = pfield(
        0.0, tier=Tier.RESTART, lo=-68.0, hi=68.0, ui="Cast",
        description="Rendered latitude of the storm (degrees, north positive). "
                    "The effective range is radius-coupled (see the cast "
                    "validator) so the stamp stays clear of the 63 deg "
                    "storm-free exchange band, same rule as the hero pin",
    )
    lon_deg: float = pfield(
        0.0, tier=Tier.RESTART, lo=-180.0, hi=180.0, ui="Cast",
        description="Rendered longitude of the storm at the final snapshot "
                    "(degrees, -180..180). Drift-compensated at generation "
                    "(like the T1 longitude pins): the generator inverse-"
                    "compensates the eastward zonal drift over the development "
                    "run so the storm lands where you asked",
    )
    radius: float = pfield(
        0.03, tier=Tier.RESTART, lo=0.01, hi=0.15, ui="Cast",
        description="Core radius, radians of arc (1 rad = 57.3 deg; default "
                    "0.03 rad is about 1.7 deg). Larger radii tighten the "
                    "latitude cap (the stamp must stay clear of the exchange "
                    "band)",
    )
    strength_scale: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=3.0, ui="Cast",
        description="Multiplier on the per-kind base vorticity law: hero "
                    "0.045*hero_strength, oval 0.012*(radius/0.03), barge "
                    "-0.006, pearl 0.008. 1.0 = the kind's default strength; "
                    "0 = a color-only stamp with no circulation",
    )
    tint: float | None = pfield(
        None, tier=Tier.RESTART, lo=-1.0, hi=1.0, ui="Cast",
        description="Storm tint (T3): positive = warm/red end of the "
                    "storm_tints gradient, negative = cool. None = the kind "
                    "default (hero: hero_tint; oval 0.1; barge 0.35; pearl "
                    "0.05). Applied verbatim (bypasses stamp_contrast)",
    )
    brightness: float | None = pfield(
        None, tier=Tier.RESTART, lo=-0.5, hi=0.5, ui="Cast",
        description="Storm brightness (T0); negative = a dark storm. None = "
                    "the kind default (hero: hero_brightness; oval 0.22; barge "
                    "-0.28; pearl 0.25). Applied verbatim (bypasses "
                    "stamp_contrast)",
    )
    aspect: float = pfield(
        1.0, tier=Tier.RESTART, lo=1.0, hi=3.0, ui="Cast",
        description="lon:lat elongation of the stamp (1.0 = round). Stretches "
                    "the iso-contours along longitude, like hero_aspect",
    )


class WakeDir(StrEnum):
    """Hero wake trailing direction."""
    AUTO = "auto"   # follow the strongest nearby jet under hero_emergence;
                    # legacy authored westward otherwise (review F06)
    EAST = "east"   # force east-trailing
    WEST = "west"   # force west-trailing


class StormsParams(_Params):
    """Field declaration order matches the panel's Hero / Ovals / Accents /
    Barges / Pearls / Outbreaks / Small storms / Mergers sub-groups (contiguous runs
    of the same ``ui`` sub-label) so ``_draw_model`` emits one
    ``separator_text`` per group boundary, not one per field. ``rim_contrast``
    and ``wake_turbulence`` are hero-perimeter/wake effects -> Hero.
    ``stamp_contrast`` touches ovals/barges/pearls/small storms but not the
    hero stamp -> grouped with Small storms, the most general/catch-all of
    those four population types.

    NOTE: declaration order is ALSO the canonical ``randomize()`` draw order --
    the walk in randomize.py draws one RNG value per ``rand`` field in field
    order, so reordering a ``rand``-bearing field here changes the randomized
    output for every field after it. ``test_randomize_output_is_pinned`` guards
    this; if you reorder, either keep ``rand`` fields in place or re-baseline
    that golden deliberately."""

    # -- Hero -----------------------------------------------------------
    hero_count: int = pfield(
        1, tier=Tier.RESTART, lo=0, hi=3, rand=(0, 2), ui="Hero",
        description="Giant anticyclones of Great Red Spot (GRS) class — the"
                    " planet-dominating bright/red oval storms (co-rotates with"
                    " the local ambient shear vorticity of the zone it sits in,"
                    " which is what lets it persist against differential shear"
                    " instead of getting torn apart)",
    )
    hero_radius: float = pfield(
        0.10, tier=Tier.RESTART, lo=0.03, hi=0.25, rand=(0.06, 0.16), ui="Hero",
        description="Hero vortex core radius, radians of arc (1 rad = 57.3"
                    " deg; default 0.10 rad is about 5.7 deg — GRS-scale)",
    )
    hero_strength: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.2, hi=3.0, rand=(0.7, 1.6), ui="Hero",
        description="GRS-class hero storm vorticity amplitude",
    )
    hero_latitude: float | None = pfield(
        None, tier=Tier.RESTART, lo=-55.0, hi=55.0, adv=True, ui="Hero",
        description="Pin the hero storm to this latitude (degrees; the 'pin' "
                    "checkbox toggles it). Unpinned (None) = seeded tropical-zone "
                    "placement. The effective range is further limited by "
                    "hero_radius (see validator) so the stamp stays clear of the "
                    "63 deg exchange band",
    )
    hero_longitude: float | None = pfield(
        None, tier=Tier.RESTART, lo=-180.0, hi=180.0, adv=True, ui="Hero",
        description="Pin the hero storm's RENDERED longitude (degrees, "
                    "-180..180; the 'pin' checkbox toggles it). Unpinned (None) "
                    "= seeded placement. The value is the end-of-run longitude, "
                    "not the seed: the generator inverse-compensates the storm's "
                    "eastward zonal drift over the whole development run so the "
                    "spot lands where you asked when the snapshot is taken. A "
                    "hero that merges with or absorbs another storm deviates "
                    "(a recorded caveat)",
    )
    rim_contrast: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=2.5, adv=True, ui="Hero",
        description="Scales the hero storm's dark perimeter ring + bright collar "
                    "(the Red Spot Hollow) amplitude; 1.0 = default, >1 deepens "
                    "the rim contrast, 0 removes the ring/collar",
    )
    hero_aspect: float = pfield(
        1.0, tier=Tier.RESTART, lo=1.0, hi=3.0, adv=True, ui="Hero",
        description="Hero storm lon:lat elongation (real GRS ~2:1); 1.0 = round. "
                    "Stretches the stamp, perimeter ring, collar, spiral lanes "
                    "and detail mask along longitude. Wake across-width and "
                    "merge capture stay isotropic (recorded LIMITs)",
    )
    hero_mottle: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Hero",
        description="Turbulent interior churn inside hero storms: a flow-scale "
                    "fbm breaks up the smooth Gaussian core so the spot reads as "
                    "churning cloud, not an airbrushed blob. Windowed to the "
                    "interior so the perimeter ring/collar stay clean; stamped "
                    "into the relaxation target so the solver folds it into "
                    "filaments. 0 = smooth v1 core (byte-identical)",
    )
    hero_tint_var: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Hero",
        description="Interior color variation inside hero storms: a flow-scale "
                    "fbm modulates the warm-red tint tracer (T3) toward "
                    "salmon/white in the troughs, so the spot reads festooned "
                    "rather than flat red. 0 = uniform v1 tint (byte-identical)",
    )
    hero_rim_tint: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Hero",
        description="Dark reddish collar (the GRS 'Red Spot Hollow' rim): the "
                    "perimeter currently only darkens; this reddens (raises the "
                    "warm-red tint) and darkens the perimeter annulus so the "
                    "oval reads as a discrete vortex with a dark-red rim rather "
                    "than a soft stain on the band. 0 = no rim tint "
                    "(byte-identical)",
    )
    hero_rim_warp: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Hero",
        description="Lumpy-oval boundary: warps the hero's dark perimeter ring + "
                    "bright collar with a low-azimuthal-wavenumber (few-lobe) "
                    "per-hero perturbation, so the spot edge reads as a naturally "
                    "irregular oval instead of a flawless azimuthally-symmetric "
                    "ring (the 'over-regular' look). Scale-invariant lobes (not "
                    "pixel-frequency noise) so it holds up at full-disk and "
                    "close-up; rim and collar warp independently. 0 = perfect "
                    "oval (byte-identical, the fbm is never evaluated)",
    )
    hero_wake_detail: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Hero",
        description="Wake filament structure: the downstream wake is stamped as a "
                    "smooth wedge into the relaxation target, so it reads as a "
                    "blob even though the wake velocity is turbulent. This frays "
                    "the wedge envelope and carves its interior with an "
                    "anisotropic, intermittent, flow-aligned fbm so the wake reads "
                    "as ragged folded filaments. Scale-invariant (rc-normalized); "
                    "the velocity wake supplies the along-flow folding. 0 = smooth "
                    "wedge (byte-identical, the fbm is never evaluated)",
    )
    hero_wake_dir: WakeDir = pfield(
        WakeDir.AUTO, tier=Tier.RESTART, adv=True, ui="Hero",
        description="Which way the hero's wake trails. auto = follow the "
                    "strongest jet near the wake lane when hero_emergence is "
                    "on (the wake is real fluid machinery there — folds advect "
                    "with the flow), legacy authored westward otherwise. "
                    "east/west force the direction; forcing AGAINST the local "
                    "jet reads weaker, because the flow drains the folds out "
                    "of the wake window. Flips the moat's torn-open arc too "
                    "(it is keyed to the wake side).",
    )
    hero_solid_core: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Hero",
        description="Solid-body hero rotation (vorticity mode): blends the hero's "
                    "vorticity from the Gaussian profile (center-peaked -> "
                    "differential rotation -> the interior winds into a "
                    "center-draining whirlpool) toward a near-uniform vorticity "
                    "patch (rigid solid-body interior rotation -> a coherent "
                    "GRS-like oval with spiral arms only OUTSIDE it). 0 = Gaussian "
                    "(byte-identical); 1 = full patch. Pairs with a larger "
                    "hero_radius and lower hero_strength.",
    )
    hero_emergence: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Hero",
        description="GRS-realism pack for hero storms (Juno/Voyager-anchored). "
                    "Morphs the hero from a soft stamped whirlpool to the real "
                    "storm architecture: (1) the vorticity becomes an ANNULAR "
                    "RING — the ~430 km/h winds live at the periphery while the "
                    "interior is stagnant, so the quiescent core HOLDS its fill "
                    "instead of winding into a dark-eye pinwheel — wrapped in a "
                    "partial opposite-signed shield skirt so the ring's net "
                    "circulation cannot wind the far neighborhood into a "
                    "pinwheel; (2) the tint/brightness stamp becomes a FILLED "
                    "PLATEAU (the GRS is a near-uniform red oval, not a "
                    "Gaussian stain); (3) the prognostic core is ANCHORED to "
                    "the registry position so the red fill lands on the visible "
                    "vortex; (4) tracer relaxation fades in the ring band — the "
                    "ring's shear folds a ragged, filament-shedding boundary "
                    "that exchanges material with the jets — and BOOSTS in the "
                    "outer annulus so the bands re-assert parallel within ~2 "
                    "spot radii; (5) the render detail layer goes QUIET over "
                    "the spot (the real interior is smooth tonal fields with "
                    "faint wisps, not loud churn). Vorticity-mode levers (1)(3) "
                    "need solver.type=vorticity; the rest act in both modes. "
                    "Hero-local (nothing beyond ~3.6 hero radii is touched; the "
                    "visible oval edge sits AT hero_radius). 0 = legacy stamped "
                    "hero (byte-identical, every path is compiled out)",
    )
    wake_turbulence: float = pfield(
        1.8, tier=Tier.RESTART, lo=0.0, hi=5.0, rand=(1.0, 3.0), adv=True, ui="Hero",
        description="Turbulence boost in the wake wedge downstream of hero storms",
    )
    hero_tint: float = pfield(
        0.9, tier=Tier.RESTART, lo=-1.0, hi=1.0, adv=True, ui="Hero",
        description="Hero storm tint (T3) stamped at generation: positive pulls "
                    "toward the warm/red end of the storm_tints gradient, negative "
                    "toward the cool end. 0.9 = the previously hardwired GRS red "
                    "(byte-identical default). Capped at 1.0: the storm-tint LUT "
                    "lookup clamps at the sampler edge (derive.comp indexes it at "
                    "(T3+1)/2 clamped to [0,1]), so values past 1.0 saturate and "
                    "buy nothing. Exempt from stamp_contrast (KIND_HERO exclusion)",
    )
    hero_brightness: float = pfield(
        0.05, tier=Tier.RESTART, lo=-0.5, hi=0.5, adv=True, ui="Hero",
        description="Hero storm brightness (T0) stamped at generation. 0.05 = the "
                    "previously hardwired GRS value (byte-identical default). "
                    "NEGATIVE = dark storm — the Neptune Great-Dark-Spot one-slider "
                    "(barges use -0.28, polar vortices -0.22, so dark stamps are a "
                    "supported axis). Exempt from stamp_contrast (KIND_HERO "
                    "exclusion)",
    )
    hero_companions: int = pfield(
        0, tier=Tier.RESTART, lo=0, hi=3, adv=True, ui="Hero",
        description="Bright companion clouds pinned beside each hero storm "
                    "(Neptune GDS companion / Scooter class): KIND_PEARL stamps "
                    "offset a few core radii from the hero on its wake-free flank, "
                    "seeded on their own substream after the population cap. "
                    "0 = off (byte-identical)",
    )
    companion_aspect: float = pfield(
        1.0, tier=Tier.RESTART, lo=1.0, hi=5.0, adv=True, ui="Hero",
        description="East-west elongation (lon:lat) of the bright companion "
                    "clouds; 1.0 = round. Stretches each KIND_PEARL companion "
                    "into a wispy cirrus streak beside the hero (real Neptune's "
                    "GDS companion clouds are sheared streaks, not round dots), "
                    "via the same generic aspect path as hero_aspect. "
                    "1.0 = round (byte-identical)",
    )
    companion_brightness: float = pfield(
        0.32, tier=Tier.RESTART, lo=0.0, hi=0.8, adv=True, ui="Hero",
        description="T0 brightness of the hero companion clouds. 0.32 = the "
                    "pre-lever constant (byte-identical). Reference flank "
                    "clouds are among the brightest pixels in the GRS "
                    "neighborhood — on a pale-moat placement the default "
                    "reads as a faint smudge",
    )
    hero_shape: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=1.5, adv=True, ui="Hero",
        description="Low-order deformation of the hero's outline away from a "
                    "perfect ellipse: equatorward flattening (the belt presses "
                    "the rim flat) plus seeded lobes so aspect and curvature "
                    "drift around the arc. 0 = exact analytic oval, 1 = the "
                    "calibrated GRS egg (the ships-at-1.0 exception to the "
                    "default=off lever convention: the deformation is part of "
                    "the emergence pack's calibration; the OFF state is 0). "
                    "Rides the emergence variant — inert at hero_emergence 0. "
                    "Past ~1.4 the ragged-release band drifts onto the bright "
                    "annulus",
    )
    hero_shape_seed: int = pfield(
        0, tier=Tier.RESTART, lo=0, hi=99999, adv=True, ui="Hero",
        description="Re-rolls the hero's seeded shape lobes on their own "
                    "substream of the master seed — changing it never "
                    "perturbs any other seeded draw",
    )
    hero_taper: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.5, adv=True, ui="Hero",
        description="Upstream-end wedge taper: the reference GRS's boundary "
                    "converges toward a point on the side the flow arrives "
                    "from (measured 20-40% of local radius), while the wake "
                    "end stays blunt. Deterministic (no seed), follows "
                    "hero_wake_dir, deepest at ~35 deg off the upstream tip "
                    "in the aspect-squashed frame (physically closer to the "
                    "tip on an elongated hero — ~14 deg at aspect 2.9); "
                    "the tip, the flanks and the whole downstream half are "
                    "untouched. Inert at hero_emergence 0",
    )
    hero_flow_aspect: float = pfield(
        1.0, tier=Tier.RESTART, lo=1.0, hi=2.5, adv=True, ui="Hero",
        description="Flow-field elongation multiplier over hero_aspect: the "
                    "streamfunction the vorticity ring induces is intrinsically "
                    "rounder than the ring (Poisson low-pass), so the developed "
                    "storm reads rounder than authored; >1 widens only the "
                    "FLOW's east-west footprint. Calibration verdict: raising "
                    "this stretches the pale ENVELOPE while the interior "
                    "erasure machinery (still sized to the anatomy) dilutes "
                    "the red core — for a more elongated STORM raise "
                    "hero_aspect itself. Vorticity mode only; inert in "
                    "kinematic mode and at hero_emergence 0 / hero_solid_core "
                    "0",
    )

    # -- Ovals ------------------------------------------------------------
    oval_density: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=4.0, rand=(0.4, 1.8), ui="Ovals",
        description="White-oval anticyclone population multiplier",
    )
    oval_solid_core: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Ovals",
        description="Solid-body rotation for LARGE white ovals (vorticity mode): "
                    "the same anti-whirlpool patch as hero_solid_core, applied to "
                    "ovals with core radius >= 0.035 rad. A Gaussian oval is "
                    "center-peaked -> differential rotation -> at long dev_steps it "
                    "winds the tracer into a mini-bullseye; this blends its "
                    "vorticity toward a near-uniform disk (rigid interior rotation) "
                    "so it stays a coherent spot. 0 = Gaussian (byte-identical); "
                    "1 = full patch. Ovals/small storms below the radius threshold "
                    "are unaffected. Pairs with hero_solid_core to de-bullseye the "
                    "whole field without lowering dev_steps or oval_density.",
    )

    # -- Accents ----------------------------------------------------------
    # Explicitly colored ovals (the Oval BA / second-red-spot unlock, review
    # A01): a shared scalar group — count 0-2, one latitude, one appearance.
    accent_count: int = pfield(
        0, tier=Tier.RESTART, lo=0, hi=2, adv=True, ui="Accents",
        description="Accent ovals: KIND_OVAL storms with EXPLICIT color (the "
                    "Oval BA 'second red spot' unlock — a red oval beside the "
                    "white population). Seeded on their own substream after the "
                    "population cap, so the base storm field is untouched; "
                    "count=2 places a pair at offset longitudes with identical "
                    "appearance. 0 = off (byte-identical)",
    )
    accent_latitude: float | None = pfield(
        None, tier=Tier.RESTART, lo=-55.0, hi=55.0, adv=True, ui="Accents",
        description="Pin accent ovals to this latitude (degrees). None = seeded "
                    "zone placement. Like hero_latitude, the effective range is "
                    "radius-coupled (see validator) so the stamp stays clear of "
                    "the 63 deg storm-free exchange band",
    )
    accent_longitude: float | None = pfield(
        None, tier=Tier.RESTART, lo=-180.0, hi=180.0, adv=True, ui="Accents",
        description="Pin the accent ovals' RENDERED longitude (degrees, "
                    "-180..180). Unpinned (None) = seeded Poisson-disc "
                    "placement. The value is the end-of-run longitude of the "
                    "FIRST accent: the generator inverse-compensates the shared "
                    "zonal drift so it lands where you asked, and a count=2 pair "
                    "is offset a fixed step (0.6 rad) downstream of it. Accents "
                    "that get caught in a merger deviate (a recorded caveat)",
    )
    accent_tint: float = pfield(
        0.9, tier=Tier.RESTART, lo=-1.0, hi=1.0, adv=True, ui="Accents",
        description="Accent oval tint (T3): positive = warm/red end of the "
                    "storm_tints gradient (Oval BA red), negative = cool. Applied "
                    "verbatim — accents bypass stamp_contrast/stamp_tint_contrast",
    )
    accent_brightness: float = pfield(
        0.12, tier=Tier.RESTART, lo=-0.5, hi=0.5, adv=True, ui="Accents",
        description="Accent oval brightness (T0); negative = dark oval. Applied "
                    "verbatim — accents bypass stamp_contrast",
    )
    accent_radius: float = pfield(
        0.05, tier=Tier.RESTART, lo=0.02, hi=0.12, adv=True, ui="Accents",
        description="Accent oval core radius (radians of arc; 1 rad = 57.3 deg, "
                    "so default 0.05 ~ 2.9 deg). Default 0.05 sits "
                    "above the 0.035 solid-body threshold (OVAL_SOLID_MIN_R in "
                    "vortex_omega.glsl), so oval_solid_core>0 keeps accents "
                    "coherent in vorticity mode; below 0.035 they stay Gaussian "
                    "and can wind into eddies over a long dev run (F07)",
    )
    accent_aspect: float = pfield(
        1.0, tier=Tier.RESTART, lo=1.0, hi=5.0, adv=True, ui="Accents",
        description="Accent oval east-west elongation (lon:lat); 1.0 = round. "
                    "Stretches the bright accent stamp into a wispy cirrus streak "
                    "(Neptune bright-cloud / Scooter class) via the same generic "
                    "aspect path as hero_aspect. 1.0 = round (byte-identical)",
    )

    # -- Barges -------------------------------------------------------------
    barge_density: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=3.0, rand=(0.3, 1.5), ui="Barges",
        description="Brown-barge cyclone population multiplier (belts)",
    )

    # -- Pearls -------------------------------------------------------------
    pearls_count: int = pfield(
        7, tier=Tier.RESTART, lo=0, hi=14, rand=(0, 9), ui="Pearls",
        description="String-of-pearls ovals on one seeded latitude (0 = off)",
    )

    # -- Outbreaks ------------------------------------------------------
    outbreak_count: int = pfield(
        0, tier=Tier.RESTART, lo=0, hi=3, rand=(0, 2), adv=True, ui="Outbreaks",
        description="Convective outbreaks (Great-White-Spot events) during the development run",
    )
    outbreak_strength: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.2, hi=3.0, adv=True, ui="Outbreaks",
        description="Convective outbreak vorticity amplitude",
    )
    outbreak_latitude: float | None = pfield(
        None, tier=Tier.RESTART, lo=-55.0, hi=55.0, adv=True, ui="Outbreaks",
        description="Pin convective outbreaks to this latitude (degrees; the "
                    "'pin' checkbox toggles it) -- the 2010 Saturn Great White "
                    "Spot erupted at ~35 N, the 1990 event on the equator. "
                    "None = seeded placement in a dark belt. A pin bypasses "
                    "the belt-candidate selection entirely (including the "
                    "outbreak_lat_min floor), so equatorial eruptions work",
    )
    outbreak_longitude: float | None = pfield(
        None, tier=Tier.RESTART, lo=-180.0, hi=180.0, adv=True, ui="Outbreaks",
        description="Pin the outbreak train's RENDERED longitude (degrees, "
                    "-180..180; the 'pin' checkbox toggles it). Unpinned (None) "
                    "= seeded placement. The value is where the eruption head "
                    "sits at the final snapshot: since the plume knots carry no "
                    "circulation, the sim velocity advects them at roughly the "
                    "zonal rate, so the generator inverse-compensates that drift "
                    "over the post-eruption life (best-effort -- the belt shear "
                    "folds the tail into a streak, so only the head lands "
                    "precisely)",
    )
    outbreak_phase: float | None = pfield(
        None, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Outbreaks",
        description="Pin WHEN outbreaks erupt: eruption start as a fraction "
                    "of the development run (0 = at init, 1 = at the final "
                    "snapshot). None = seeded 0.55..0.85 draw per eruption, "
                    "which catches plumes across their life. ~0.6 shows a "
                    "fresh mid-eruption train at the snapshot; early values "
                    "leave only the sheared-out streak",
    )
    outbreak_lat_min: float = pfield(
        0.20, tier=Tier.RESTART, lo=0.0, hi=1.0, adv=True, ui="Outbreaks",
        description="Minimum |latitude| for AUTO outbreak-belt selection, "
                    "radians of latitude (1 rad = 57.3 deg; default 0.20 rad "
                    "is about 11.5 deg). The floor keeps seeded eruptions off "
                    "the equatorial zone where white-on-white plumes vanish; "
                    "lower it to admit equatorial belts to the candidate "
                    "pool, or use outbreak_latitude to pin exactly",
    )

    # -- Small storms ---------------------------------------------------
    small_density: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=4.0, rand=(0.4, 1.8), adv=True, ui="Small storms",
        description="Small-storm field: sub-oval white spots and dark spots scattered "
                    "in loose latitude rows (0 = off, the pre-v1.1 look)",
    )
    stamp_contrast: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=3.0, rand=(0.8, 1.3), adv=True, ui="Small storms",
        description="Tracer-stamp contrast of ovals/barges/pearls/small storms (1 = v1)",
    )
    stamp_tint_contrast: float | None = pfield(
        None, tier=Tier.RESTART, lo=0.0, hi=3.0, adv=True, ui="Small storms",
        description="Tint amplitude of ovals/barges/pearls/small storms, split "
                    "from the brightness amplitude (review B5-7): stamp_contrast "
                    "scales brightness, this scales tint. None = follow "
                    "stamp_contrast (byte-identical legacy coupling). Like "
                    "stamp_contrast it EXCLUDES the hero (use hero_tint) and "
                    "does not touch accents (explicit color)",
    )

    # -- Mergers ------------------------------------------------------------
    merge_rate: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.0, 0.8), adv=True, ui="Mergers",
        description="Anticyclone merger aggressiveness: converging same-sign "
                    "ovals coalesce when their gap falls under ~1.5*rate*(r1+r2), "
                    "and generation seeds convergent pairs so mergers actually "
                    "occur during the dev run (0 = off, the v1.1 behavior)",
    )
    merge_debris: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=2.0, adv=True, ui="Mergers",
        description="Brightness of the transient turbulent collar a fresh "
                    "merger leaves behind (inert while merge_rate is 0)",
    )

    # -- Cast (art-directed storms) ----------------------------------------
    cast: list[StormOverride] = pfield(
        factory=list, tier=Tier.RESTART, adv=True, ui="Cast",
        description="Cast list: storms placed by hand (kind + rendered "
                    "position + size + optional color). Each entry is stamped "
                    "verbatim after the seeded populations, exempt from the "
                    "population cap and runtime mergers, so a director's storm "
                    "survives the whole run where it was placed. Empty (the "
                    "default) = no cast, byte-identical to the seeded-only "
                    "field. Capped at 16 entries",
    )

    @model_validator(mode="after")
    def _validate_cast(self) -> StormsParams:
        if len(self.cast) > 16:
            raise ValueError(
                f"storms.cast has {len(self.cast)} entries; the cap is 16"
            )
        for i, entry in enumerate(self.cast):
            cap = hero_latitude_cap(entry.radius)
            if abs(entry.lat_deg) > cap:
                raise ValueError(
                    f"storms.cast[{i}].lat_deg={entry.lat_deg} exceeds the "
                    f"radius-coupled limit +-{cap:.1f} deg "
                    f"(radius={entry.radius}); the stamp would smear into the "
                    f"63 deg storm-free exchange band. Lower lat_deg or radius."
                )
        return self

    @model_validator(mode="after")
    def _validate_hero_latitude(self) -> StormsParams:
        if self.hero_latitude is not None:
            cap = hero_latitude_cap(self.hero_radius)
            if abs(self.hero_latitude) > cap:
                raise ValueError(
                    f"hero_latitude={self.hero_latitude} exceeds the radius-coupled "
                    f"limit +-{cap:.1f} deg (hero_radius={self.hero_radius}); the stamp "
                    f"would smear into the 63 deg storm-free exchange band. Lower "
                    f"hero_latitude or hero_radius."
                )
        return self

    @model_validator(mode="after")
    def _validate_accent_latitude(self) -> StormsParams:
        if self.accent_latitude is not None:
            cap = hero_latitude_cap(self.accent_radius)
            if abs(self.accent_latitude) > cap:
                raise ValueError(
                    f"accent_latitude={self.accent_latitude} exceeds the radius-coupled "
                    f"limit +-{cap:.1f} deg (accent_radius={self.accent_radius}); the "
                    f"stamp would smear into the 63 deg storm-free exchange band. "
                    f"Lower accent_latitude or accent_radius."
                )
        return self


class WavesParams(_Params):
    festoon_strength: float = pfield(
        0.8, tier=Tier.RESTART, lo=0.0, hi=3.0, rand=(0.0, 1.4), ui="Waves",
        description="Festoon plumes + hot spots on the equatorial belt edge (0 = off)",
    )
    festoon_wavenumber: int = pfield(
        12, tier=Tier.RESTART, lo=4, hi=24, rand=(8, 16), ui="Waves",
        description="How many festoon plumes fit around the equator "
                    "(higher = more, smaller plumes; the Rossby wavenumber of "
                    "the train)",
    )
    hotspot_depth: float = pfield(
        0.6, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.2, 0.9), ui="Waves",
        description="Depth of the cloud-free hot spots at the wave troughs",
    )
    ribbon_strength: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=3.0, rand=(0.0, 1.0), ui="Waves",
        description="Saturn-style ribbon wave on one mid-latitude jet (0 = off)",
    )
    ribbon_wavenumber: int = pfield(
        12, tier=Tier.RESTART, lo=4, hi=30, ui="Waves",
        description="Wavenumber of the Saturn-style ribbon wave",
    )
    festoon_hero_strength: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=3.0, adv=True, ui="Waves",
        description="Second festoon train rooted on the band edge nearest the "
                    "hero storm (plumes only, no hot spots): streamers weaving "
                    "through the hero's wake lane, tails brushing the collar. "
                    "0 = off; a silent no-op without a hero, without a band "
                    "edge within 0.15 rad of it, or when that edge IS the "
                    "primary festoon's root (one edge is never double-"
                    "trained)",
    )
    festoon_hero_wavenumber: int = pfield(
        11, tier=Tier.RESTART, lo=4, hi=24, adv=True, ui="Waves",
        description="Wavenumber of the hero-adjacent festoon train (the "
                    "default deliberately differs from festoon_wavenumber — "
                    "twin wavenumbers read as a mechanical comb)",
    )


class DetailParams(_Params):
    intensity: float = pfield(
        0.55, tier=Tier.POST, lo=0.0, hi=2.0, rand=(0.3, 0.9), ui="Detail",
        description="Export/preview detail synthesis amplitude",
    )
    flow_phases: int = pfield(
        3, tier=Tier.POST, lo=1, hi=4, ui="Detail",
        description="Staggered advected-noise phases (more = richer filaments)",
    )
    flow_stretch: float = pfield(
        1.0, tier=Tier.POST, lo=0.1, hi=4.0, rand=(0.6, 1.6), ui="Detail",
        description="How far detail noise is advected along the flow",
    )
    frequency: float = pfield(
        48.0, tier=Tier.POST, lo=8.0, hi=256.0, log=True, ui="Detail",
        description="Base spatial frequency of the detail noise",
    )
    cellular_amount: float = pfield(
        0.6, tier=Tier.POST, lo=0.0, hi=2.0, rand=(0.3, 1.0), ui="Detail",
        description="Convective cell (closed-cell/popcorn) texture in quiet zones",
    )
    striation_amount: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.5, rand=(0.2, 0.8), ui="Detail",
        description="Ropey flow-parallel striations inside belts (intra-band "
                    "thread texture; 0 = the pre-v1.1 look)",
    )
    striation_frequency: float = pfield(
        96.0, tier=Tier.POST, lo=16.0, hi=512.0, log=True, ui="Detail",
        description="Base spatial frequency of the striation noise",
    )
    polar_stipple: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=2.0, rand=(0.3, 1.0), ui="Detail",
        description="Bright granular storm speckle (popcorn) poleward of ~55 deg "
                    "(the band-to-mottle transition character)",
    )
    intermittency: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, ui="Detail", fx=True,
        description="Longitudinal patchiness of the filament/striation texture: "
                    "violent folded patches abutting calm laminar runs (the real "
                    "mosaic's chaos is intermittent, not uniform). No rand: a "
                    "draw here would reshuffle every later randomize draw",
    )
    hero_calm: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Detail",
        description="Calm the band-aligned grain inside hero storms: the "
                    "detail filament streak + striation are flow/band-aligned "
                    "and are amplified near heroes, so they cross the GRS as "
                    "straight 'wood-grain' that ignores the vortex rotation. "
                    "This attenuates those two terms inside the hero (weighted "
                    "by the hero mask) so the vortex-aligned spiral lanes and "
                    "the sim-side hero_mottle churn carry the interior instead. "
                    "0 = full band grain (byte-identical)",
    )
    hero_spiral: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.5, ui="Detail", fx=True,
        description="Tightly wound internal spiral lanes inside hero storms "
                    "(the Juno-close-up GRS look) plus collar streamlines; "
                    "winds in the hero's actual rotation sense. Stationary in "
                    "the hero frame — fine for stills",
    )
    hero_collar_wrap: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, ui="Detail", fx=True,
        description="Tightly-pitched wound-lane filaments wrapping the hero "
                    "collar (the GRS 'hollow' look in stills): a log-spiral on "
                    "the rim window, wound in the storm's rotation sense. "
                    "Independent of hero_spiral (interior lanes); stationary in "
                    "the hero frame. 0 = off",
    )
    zone_texture: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=2.5, adv=True, ui="Detail", fx=True,
        description="Flow-folded luminance structure inside ZONES (the calm "
                    "lanes between belts, gated by 1 - belt_mask). Belt "
                    "interiors get belt_texture and shear-gated filaments; "
                    "zones get neither and read as detail-starved smooth bands "
                    "cutting across the disk. This gives zones their own "
                    "flow-structured fold (calmer than belts, not flat). "
                    "0 = starved zones (byte-identical)",
    )
    belt_texture: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=2.5, ui="Detail", fx=True,
        description="Storm-scale folded luminance structure inside belts "
                    "(0.5-3 deg, flow-backtraced so patches fold with the "
                    "flow) + a belt floor for the fine filaments; the v1.4 "
                    "audit's dominant texture gap on broad-band layouts",
    )
    belt_texture_fine: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=2.5, ui="Detail", fx=True,
        description="Finer sub-grid belt fold octave: a second flow-aligned "
                    "backtrace hop folds mid-frequency noise below the sim "
                    "grid scale, densifying belt texture at matched scale",
    )
    mottle: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.5, ui="Detail", fx=True,
        description="Temperate lace mottle (35-60 deg): granular bright "
                    "rings, dark dots, and lacy folds where banding gives "
                    "way -- the reference's mid-latitude storm-flecked "
                    "character",
    )
    polar_filaments: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=2.0, adv=True, ui="Detail", fx=True,
        description="Polar folded-filamentary region (the Juno cap look): "
                    "dense, multi-scale, flow-folded RIDGED filaments tangling "
                    "between the circumpolar cyclones poleward of ~65 deg. "
                    "Backtraced through the polar patch velocity so the lace "
                    "winds with the cap vortices; only active when the polar "
                    "route is on (cyclone-cluster/plain poles). 0 = off "
                    "(byte-identical)",
    )
    spread: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Detail",
        spread=True,
        description="Uniform detail coverage across latitude: 0 = band-gated "
                    "(belts textured, zones calmer, the default look, "
                    "byte-identical), >0 = the flow-folded detail-FX texture "
                    "(belt/zone/mottle folds + filaments) applied at EVEN density "
                    "everywhere at this level, so there are no detail-starved "
                    "zones or stamped latitude bands. Still flow-folded (not flat "
                    "noise). Pole-faded. ~0.36 is a balanced value",
    )
    cirrus_fibers: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=2.0, adv=True, ui="Detail", fx=True,
        description="Render-time combed-fiber synthesis over the ELONGATED "
                    "bright-cloud stamps (companion/accent storms with "
                    "aspect > 1, the Neptune methane-cirrus class): carves "
                    "dark inter-strand lanes + gentle bright ridges into each "
                    "streak, flow-oriented and flow-warped. Stamping fibers "
                    "into the tracer was falsified (they smear over the dev "
                    "run; docs/roadmap.md) — this synthesizes them "
                    "post-advection. Requires detail.intensity > 0. No rand: "
                    "a draw here would reshuffle every later randomize draw. "
                    "0 = off (byte-identical)",
    )
    cirrus_fiber_freq: float = pfield(
        6.0, tier=Tier.POST, lo=2.0, hi=24.0, log=True, ui="Detail",
        description="Strand density of the cirrus fibers: strands across "
                    "each bright-cloud streak half-width. Amplitude is "
                    "attenuated when strands approach the output pixel size "
                    "(spacing ~ cloud_radius/freq radians), so high values "
                    "need high export resolution. Inert unless "
                    "cirrus_fibers > 0",
    )
    streak_mute: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Detail", fx=True,
        description="Suppress the WHOLE filament-streak accumulator (the "
                    "ungated base flow-streak + its intermittency gate + the "
                    "belt_texture filament floor; the SPREAD streak too, if "
                    "spread > 0). The base streak has a speed/shear floor and "
                    "no zero lever of its own, so smooth laminar planets that "
                    "enable detail.intensity only for cirrus_fibers would gain "
                    "planet-wide flow-grain without this. No rand (draw-order "
                    "safe). 0 = full streak (byte-identical)",
    )


class SolverType(StrEnum):
    KINEMATIC = "kinematic"   # v1.5 analytic-streamfunction (default)
    VORTICITY = "vorticity"   # v1.6 prognostic vorticity-streamfunction (opt-in)


class InjectMask(StrEnum):
    """Spatial localization of eddy-vorticity injection."""
    GLOBAL = "global"   # churn everywhere (legacy uniform behavior)
    BELTS = "belts"     # cyclonic dark bands only; anticyclonic zones stay smooth
    SHEAR = "shear"     # jet-shear flanks only; filaments where shear is high


# Shader code (u_inject_mask int) for each InjectMask. Kept beside the enum so
# the GLSL contract can't drift from the Python names.
INJECT_MASK_CODE: dict[InjectMask, int] = {
    InjectMask.GLOBAL: 0,
    InjectMask.BELTS: 1,
    InjectMask.SHEAR: 2,
}


class BaroclinicParams(_Params):
    """Opt-in 2-layer baroclinic vorticity source coupled into the vorticity
    solver's equirect pass (M3). OFF by default => byte-identical to plain v1.6.
    Default gain=2.0 (bounded mid-latitude belt-texture enrichment; the final
    aesthetic gain is the user's full-res call). The cadence fields are a
    fixed trio (Advanced, "Fixed cadence" sub-label, no rand): they keep the
    baroclinic CPU solver in its healthy pre-outcrop window."""

    enabled: bool = pfield(
        False, tier=Tier.RESTART, adv=True, ui="Solver",
        description="Inject the evolving baroclinic vorticity source into the "
                    "vorticity solver (adds physically-grounded mid-latitude "
                    "storms; requires solver type=vorticity). Off = plain v1.6. "
                    "No rand: randomize() must never silently enable it.")
    gain: float = pfield(
        2.0, tier=Tier.RESTART, lo=0.0, hi=8.0, adv=True, ui="Solver",
        description="Baroclinic source amplitude as a fraction of coriolis_f0 "
                    "(~3). The source is injected into the Poisson RHS (NOT the "
                    "vorticity state), so it is bounded (no accumulation) and "
                    "coherent (never folded by advection -- it is read fresh from "
                    "the source each step and never enters the advected q state), "
                    "enriching mid-latitude belt texture. ~2 = subtle; high gain "
                    "over-boils. No rand.")
    # The cadence trio renders under a "Fixed cadence" sub-label (B2-3: they
    # previously shipped ui="" and drew unlabeled in Advanced mode).
    warmup_steps: int = pfield(
        8000, tier=Tier.RESTART, lo=500, hi=20000, adv=True, ui="Fixed cadence",
        description="Internal pacing of the baroclinic storm generator — leave "
                    "at default; only affects how the extra mid-latitude storms "
                    "mature (spin-up steps before coupling; fixed cadence, no "
                    "rand; hi=20000 leaves headroom past the ~12500 lower-layer "
                    "blow-up so tests can force it)")
    baro_steps_per_update: int = pfield(
        150, tier=Tier.RESTART, lo=10, hi=1000, adv=True, ui="Fixed cadence",
        description="Internal pacing of the baroclinic storm generator — leave "
                    "at default (baroclinic steps per source refresh; fixed "
                    "cadence, no rand)")
    update_every: int = pfield(
        32, tier=Tier.RESTART, lo=1, hi=512, adv=True, ui="Fixed cadence",
        description="Internal pacing of the baroclinic storm generator — leave "
                    "at default (main-solver steps between source refreshes; "
                    "fixed cadence, no rand)")


# Practical floor for a screened-Poisson deformation radius: below ~a few grid
# cells (dphi at typical resolutions ~0.005 rad) 1/L_d^2 swamps the Laplacian and
# the solve degenerates. 0.0 (off) is always allowed; the (0, floor) band is not.
_DEFORMATION_RADIUS_FLOOR = 0.05


class SolverParams(_Params):
    type: SolverType = pfield(SolverType.KINEMATIC, tier=Tier.RESTART, adv=True, ui="Solver",
        description="How clouds move: kinematic = fast and painterly, bands stay "
                    "where they are painted (analytic streamfunction, v1.5); "
                    "vorticity = a real fluid sim — storms interact and shed "
                    "filaments, slower, and required by the solid-core storm "
                    "levers (prognostic vorticity, v1.6+)")
    poisson_iters: int = pfield(48, tier=Tier.RESTART, lo=8, hi=512, adv=True, ui="Solver",
        description="Solver accuracy per step: too low leaves smeared, laggy "
                    "swirls; higher is slower with diminishing returns "
                    "(fixed red-black SOR iterations; vorticity mode)")
    sor_omega: float = pfield(1.7, tier=Tier.RESTART, lo=1.0, hi=2.0, adv=True, ui="Solver",
        description="Solver convergence speed — leave at 1.7: it changes solve "
                    "time, not the picture, unless set so low the swirls lag "
                    "(SOR over-relaxation factor, must be in (1,2) exclusive; "
                    "vorticity mode)")
    deformation_radius: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=3.14, adv=True, ui="Solver",
        description="Storm locality: how far each vortex's swirl reaches. "
                    "Smaller = more local — a dominant hero stirs its own band "
                    "without destabilizing the rest of the map; 0 = off "
                    "(infinite reach, plain 2D, byte-identical). Values in the "
                    "(0, 0.05) rad band are rejected (degenerate solve). "
                    "(Physics: Rossby deformation radius L_d in RADIANS, "
                    "1 rad = 57.3 deg; vorticity mode. Screens the inversion to "
                    "(nabla^2 - 1/L_d^2)psi = omega — equivalent-barotropic / "
                    "1.5-layer reduced gravity — so induced velocity decays "
                    "~exp(-r/L_d) beyond L_d instead of the 2D ~1/r tail; real "
                    "Jupiter has L_d << the GRS. With screening on, the advected "
                    "q is equivalent-barotropic QGPV, so vortex/inject/relax "
                    "strengths tuned for the plain 2D path read weaker and more "
                    "localized -- expect to re-tune. No rand.)")
    vort_relax_tau: float = pfield(
        120.0, tier=Tier.RESTART, lo=20.0, hi=2000.0, log=True, adv=True, ui="Solver",
        description="How tightly the flow is leashed to the painted jets and "
                    "storms: low = tidy and band-locked, high = free-running "
                    "turbulence that can wander off the template (nudging "
                    "timescale in steps; vorticity mode)")
    vort_hypervisc: float = pfield(1.0, tier=Tier.RESTART, lo=0.0, hi=10.0, adv=True, ui="Solver",
        description="Fine-scale smoothing: cleans up pixel-level crackle; too "
                    "high blurs away the thinnest filaments (scale-selective "
                    "biharmonic hyperviscosity; vorticity mode)")
    coriolis_f0: float = pfield(2.0, tier=Tier.RESTART, lo=0.0, hi=20.0, adv=True, ui="Solver",
        description="Planet-rotation strength: higher = more, narrower bands "
                    "and flatter storms; lower = fewer, fatter bands (f0 in "
                    "f = f0*sin(lat), sets the Rhines/band scale; vorticity mode)")
    vort_inject: float = pfield(0.0, tier=Tier.RESTART, lo=0.0, hi=5.0, adv=True, ui="Solver",
        description="Broadband eddy-vorticity injection amplitude per step; the "
                    "jet shear folds it into filaments (the emergent-turbulence "
                    "source; 0 = off, smooth jets stay zonal). Vorticity mode.")
    vort_inject_scale: float = pfield(0.5, tier=Tier.RESTART, lo=0.1, hi=4.0, adv=True, ui="Solver",
        description="Size of the injected churn: higher = finer speckle that "
                    "the shear folds into thin filaments; lower = big blobs "
                    "(injection frequency as a multiple of bands.detail_freq; "
                    "vorticity mode)")
    vort_inject_mask: InjectMask = pfield(
        InjectMask.GLOBAL, tier=Tier.RESTART, adv=True, ui="Solver",
        description="Spatial localization of eddy injection: global = churn "
                    "everywhere; belts = cyclonic dark bands only (anticyclonic "
                    "zones stay smooth); shear = jet-shear flanks only (filaments "
                    "where shear is high). Vorticity mode.")
    vort_drag: float = pfield(0.0, tier=Tier.RESTART, lo=0.0, hi=0.3, adv=True, ui="Solver",
        description="Global brake on swirling: tames runaway planet-scale swirl "
                    "but also weakens every storm — prefer vort_psi_drag, which "
                    "targets only the oversized swirl (linear Rayleigh drag "
                    "fraction on relative vorticity per step, absorbing the 2D "
                    "inverse-cascade pileup at large scales; 0 = off; vorticity "
                    "mode)")
    vort_eddy_drag: float = pfield(0.0, tier=Tier.RESTART, lo=0.0, hi=0.3, adv=True, ui="Solver",
        description="Linear drag fraction on the EDDY vorticity q - <q>_x (the "
                    "deviation from the per-latitude zonal mean) per step. Leaves "
                    "the zonal-mean jets intact, but is FLAT in wavenumber, so it "
                    "damps medium eddies (festoons, band-edge waves) as hard as the "
                    "gravest-mode swirl -> over-flattens the field. Prefer "
                    "vort_psi_drag (scale-selective). Equirect only. 0 = off "
                    "(byte-identical). Vorticity mode.")
    vort_psi_drag: float = pfield(0.0, tier=Tier.RESTART, lo=0.0, hi=20.0, adv=True, ui="Solver",
        description="Removes oversized planet-scale swirl while PRESERVING "
                    "festoons, band-edge waves, and mid-size vortices — the "
                    "scale-selective brake to reach for before vort_drag or "
                    "vort_eddy_drag. 0 = off (byte-identical). (Physics: "
                    "large-scale hypofriction — a vorticity sink proportional to "
                    "the EDDY STREAMFUNCTION psi - <psi>_x; because psi ~ "
                    "omega/(k^2 + 1/L_d^2), the effective drag rate ~1/(k^2+"
                    "1/L_d^2) hits the gravest-mode inverse-cascade swirl far "
                    "harder than medium eddies, unlike the flat-in-k "
                    "vort_eddy_drag. Reuses the screened-Poisson psi the solver "
                    "already computes (one step stale); coefficient runs "
                    "numerically larger than vort_eddy_drag since psi << omega. "
                    "Equirect only. Vorticity mode.)")
    baroclinic: BaroclinicParams = Field(default_factory=BaroclinicParams)

    @model_validator(mode="after")
    def _validate_sor_omega(self) -> SolverParams:
        if not (1.0 < self.sor_omega < 2.0):
            raise ValueError(
                f"sor_omega={self.sor_omega} must be strictly in (1.0, 2.0) exclusive"
            )
        return self

    @model_validator(mode="after")
    def _validate_baroclinic(self) -> SolverParams:
        if self.baroclinic.enabled and self.type != SolverType.VORTICITY:
            raise ValueError(
                f"baroclinic.enabled requires solver type=vorticity "
                f"(got {self.type})"
            )
        return self

    @model_validator(mode="after")
    def _validate_deformation_radius(self) -> SolverParams:
        # 0 = off. A finite L_d below a few grid cells makes 1/L_d^2 swamp the
        # Laplacian metric -> degenerate (frozen, ~dead-velocity) solve. Reject
        # the degenerate band so it can't be set by accident; the floor is well
        # below any useful screening length.
        if 0.0 < self.deformation_radius < _DEFORMATION_RADIUS_FLOOR:
            raise ValueError(
                f"deformation_radius={self.deformation_radius} is in the "
                f"degenerate band (0, {_DEFORMATION_RADIUS_FLOOR}); use 0.0 to "
                f"disable screening or a value >= {_DEFORMATION_RADIUS_FLOOR} rad "
                f"(a few grid cells) for a well-resolved screened solve."
            )
        return self


class PoleStyle(StrEnum):
    CYCLONE_CLUSTER = "cyclone_cluster"  # Jupiter: central cyclone + polygon ring
    POLYGON_JET = "polygon_jet"          # Saturn: hexagonal (k-gonal) jet
    PLAIN_VORTEX = "plain_vortex"        # single tight polar vortex
    CALM = "calm"


class PoleParams(_Params):
    style: PoleStyle = pfield(
        PoleStyle.CYCLONE_CLUSTER, tier=Tier.RESTART, ui="Poles",
        description="Polar feature style",
    )
    cyclone_count: int = pfield(
        6, tier=Tier.RESTART, lo=3, hi=9, rand=(5, 8), ui="Poles",
        description="Ring cyclones around the central one (cyclone_cluster style)",
    )
    polygon_sides: int = pfield(
        6, tier=Tier.RESTART, lo=3, hi=9, ui="Poles",
        description="Polygon wavenumber of the polar jet (polygon_jet style)",
    )
    strength: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=3.0, rand=(0.6, 1.5), ui="Poles",
        description="Polar feature vorticity amplitude (central cyclone / polygon jet)",
    )
    field_density: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=2.0, rand=(0.4, 1.4), ui="Poles",
        description="Background small-cyclone field filling the cap poleward of "
                    "70 deg (PIA21641's dense cyclone hierarchy; 0 = off)",
    )


class PolesParams(_Params):
    north: PoleParams = Field(default_factory=PoleParams)
    south: PoleParams = Field(
        default_factory=lambda: PoleParams(style=PoleStyle.PLAIN_VORTEX)
    )


class AppearanceParams(_Params):
    palette_rows: list[PaletteRow] = pfield(
        factory=default_palette_rows, tier=Tier.POST, ui="Appearance",
        description="Latitude-anchored color gradients indexed by the color tracer "
                    "(belt dark -> zone bright), blended across signed latitude",
    )
    storm_tints: list[GradientStop] = pfield(
        factory=lambda: [s.model_copy(deep=True) for s in DEFAULT_STORM_TINTS],
        tier=Tier.POST, adv=True, ui="Appearance",
        description="Secondary tint axis for storms/festoons/hot spots",
    )
    band_tint_stops: list[GradientStop] = pfield(
        factory=lambda: [s.model_copy(deep=True) for s in DEFAULT_BAND_TINT],
        tier=Tier.POST, adv=True, ui="Appearance",
        description="Per-latitude RGB tint laid over the whole planet as a final "
                    "art-direction override: pick a color at each latitude (pos 0 = "
                    "south pole, 1 = north pole) and it recolors that band directly, "
                    "applied after every other grade so it wins. Neutral gray = no "
                    "visible shift. Only acts when band_tint_strength > 0",
    )
    band_tint_strength: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Appearance",
        description="How strongly the per-latitude band_tint_stops override the "
                    "planet color (0 = off, byte-identical; 1 = the tint fully "
                    "replaces the graded color). Blended in after the post chain and "
                    "chroma FX so the tint is not re-graded by contrast/saturation",
    )
    haze_amount: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, rand=(0.0, 0.7), ui="Appearance",
        description="Global haze: the Jupiter (0) to Saturn (~0.6) axis",
    )
    haze_color: tuple[float, float, float] = pfield(
        (0.85, 0.78, 0.62), tier=Tier.POST, ui="Appearance",
        description="Tint of the global haze blend (see haze_amount)",
    )
    contrast: float = pfield(
        1.0, tier=Tier.POST, lo=0.2, hi=2.0, rand=(0.8, 1.2), ui="Appearance",
        description="Color contrast multiplier about mid-gray",
    )
    saturation: float = pfield(
        1.0, tier=Tier.POST, lo=0.0, hi=2.0, rand=(0.7, 1.2), ui="Appearance",
        description="sRGB saturation multiplier (luma-preserving mix toward gray); "
                    "prefer chroma_scale for perceptual (Oklab) saturation",
    )
    gamma: float = pfield(
        1.0, tier=Tier.POST, lo=0.4, hi=2.5, ui="Appearance",
        description="Final tone-curve gamma on the color map",
    )
    chroma_scale: float = pfield(
        1.0, tier=Tier.POST, lo=0.0, hi=2.0, adv=True, ui="Appearance",
        description="Oklab chroma multiplier on the final color (1 = off) — "
                    "perceptual saturation, recommended over 'saturation' "
                    "(an sRGB luma mix). No rand: adding a draw would "
                    "reshuffle every later randomize draw",
    )
    chroma_variance: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=0.5, adv=True, ui="Appearance",
        description="Longitudinal within-band chroma drift: bands hold pockets "
                    "of more/less saturated material varying slowly with "
                    "longitude (the reference's saturated-pocket texture)",
    )
    hue_variance: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=0.35, adv=True, ui="Appearance",
        description="Iso-luminance Oklab hue drift (radians of max rotation; "
                    "1 rad = 57.3 deg): "
                    "differently-hued material at the same lightness, which a "
                    "luminance-keyed palette gradient cannot express -- the "
                    "hue-diversity lever the realism metrics name",
    )
    detail_chroma: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Appearance",
        description="Two-material tint for synthesized detail: bright detail "
                    "excursions shade toward a cool pale-cloud material, dark "
                    "excursions (weaker) toward warm belt material -- the "
                    "reference's interleaved cool/warm texture read, which a "
                    "luminance-only detail multiply cannot express. "
                    "L-preserving (Oklab a/b push), palette-independent. "
                    "Needs detail.intensity > 0 (the Detail panel); inert "
                    "without it. 0 = off (byte-identical)",
    )
    chroma_aging: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=0.6, adv=True, ui="Appearance",
        description="Chromophore aging: ties color saturation to the dynamical "
                    "freshness tracer (T2). Aged/stagnant air holds more "
                    "reddish-brown chromophore (more saturated); fresh upwelling "
                    "air is whiter (less saturated). Chroma-only -- the latitude "
                    "palette's HUE is untouched, so the band browns/creams just "
                    "deepen where air is old and pale where it is fresh, tying "
                    "color to the flow instead of latitude alone. 0 = off "
                    "(byte-identical)",
    )
    polar_tint_color: tuple[float, float, float] = pfield(
        (0.42, 0.50, 0.58), tier=Tier.POST, adv=True, ui="Appearance",
        description="Polar cap tint (Juno blue-gray); applied where cloud tops "
                    "are LOW -- the blue is structural, bright clouds stay pearly",
    )
    polar_tint_strength: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, rand=(0.2, 0.7), adv=True, ui="Appearance",
        description="Polar tint blend strength (0 = off, the pre-v1.1 look)",
    )
    polar_tint_start_lat: float = pfield(
        55.0, tier=Tier.POST, lo=30.0, hi=80.0, adv=True, ui="Appearance",
        description="Latitude (deg) where the polar tint begins",
    )
    polar_canvas_value: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, rand=(0.0, 0.5), adv=True, ui="Appearance",
        description="Deepens the polar cap canvas toward a dark blue-teal floor so "
                    "the folded-filament lace and cyclones pop; 0 = off. Applied "
                    "after the lace and keyed on low local luminance, so it darkens "
                    "the dark inter-wisp floor while bright crests stay bright "
                    "(raises contrast, does not flatten)",
    )


class EmissionParams(_Params):
    """Emission map components (night-side glow for Blender emission
    shading). All-zero strengths disable the map: no emission.exr is
    written and manifest consumers tolerate its absence. Values are linear
    radiance multipliers (the EXR is float32 HDR; AgX rolls strong emitters
    off into bloomable hotspots). Preview via the viewport's Emission channel
    (aurora composited as alpha x aurora_color); the Color preview never
    composites emission. No rand on the strengths: emission is invisible in
    the default Color view, so seeded randomization silently enabling it
    would surprise at export time."""

    thermal_strength: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=2.0, adv=True, ui="Emission",
        description="5-micron thermal glow through cloud gaps (gated on the "
                    "cloud-top DEPRESSION vs the band stamp: hot-spot chains "
                    "blaze, barges glow, belts glimmer, zones stay dark). "
                    "Preview: Emission channel, not Color",
    )
    thermal_color: tuple[float, float, float] = pfield(
        (1.0, 0.36, 0.08), tier=Tier.POST, adv=True, ui="Emission",
        description="Ember red-orange; linear radiance hue. Preview: Emission "
                    "channel, not Color",
    )
    thermal_threshold: float = pfield(
        0.18, tier=Tier.POST, lo=0.05, hi=0.5, adv=True, ui="Emission",
        description="Cloud-gap anomaly where the HDR hot-spot term begins "
                    "(higher = only the deepest holes blaze). Preview: Emission "
                    "channel, not Color",
    )
    thermal_hdr: float = pfield(
        16.0, tier=Tier.POST, lo=1.0, hi=40.0, adv=True, ui="Emission",
        description="Radiance of the deepest hot spots relative to the faint "
                    "belt glow (real 5-micron maps span ~50:1). Preview: Emission "
                    "channel, not Color",
    )
    lightning_strength: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=2.0, adv=True, ui="Emission",
        description="Frozen lightning-flash clusters in cyclonic belts and "
                    "at high latitudes (the Juno look: light pools under the "
                    "deck plus sparse HDR cores). Preview: Emission channel, not "
                    "Color",
    )
    lightning_color: tuple[float, float, float] = pfield(
        (0.72, 0.82, 1.0), tier=Tier.POST, adv=True, ui="Emission",
        description="Lightning flash hue; linear radiance. Preview: Emission "
                    "channel, not Color",
    )
    lightning_density: float = pfield(
        0.5, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Emission",
        description="Lightning-flash cluster population density. Preview: "
                    "Emission channel, not Color",
    )
    aurora_strength: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=2.0, adv=True, ui="Emission",
        description="Auroral ovals around the (offset) magnetic poles; "
                    "written to emission.exr's ALPHA channel so the importer "
                    "can lift it onto a shell. Preview via the viewport's "
                    "Emission channel (composited as alpha x aurora_color); "
                    "not visible in the Color preview",
    )
    aurora_color: tuple[float, float, float] = pfield(
        (0.85, 0.35, 0.60), tier=Tier.POST, adv=True, ui="Emission",
        description="H/H2 emission is pink-magenta (Earth's oxygen green "
                    "is impossible in a hydrogen atmosphere). Composited into "
                    "the Emission channel preview; not in Color",
    )
    aurora_radius: float = pfield(
        14.0, tier=Tier.POST, lo=5.0, hi=25.0, adv=True, ui="Emission",
        description="Oval angular radius from the magnetic pole, degrees. "
                    "Preview: Emission channel, not Color",
    )
    aurora_width: float = pfield(
        2.5, tier=Tier.POST, lo=0.5, hi=8.0, adv=True, ui="Emission",
        description="Auroral oval ring thickness, degrees. Preview: Emission "
                    "channel, not Color",
    )
    aurora_pole_offset: float = pfield(
        8.0, tier=Tier.POST, lo=0.0, hi=20.0, adv=True, ui="Emission",
        description="Magnetic-pole tilt from the rotation pole, degrees "
                    "(longitude seeded); Saturn's axis is aligned: use 0. "
                    "Preview: Emission channel, not Color",
    )

    @property
    def enabled(self) -> bool:
        return (
            self.thermal_strength > 0.0
            or self.lightning_strength > 0.0
            or self.aurora_strength > 0.0
        )


class MaskParams(_Params):
    """Imported paint mask: a single-channel grayscale equirect (2:1) PNG sidecar
    that drives three POST-tier art-direction targets. Every gain defaults 0.0,
    which is an EXACT no-op (each target uses a ``mix(1.0, mask, gain)``-style
    factor/weight), so a planet with no mask -- or a mask with all-zero gains --
    renders byte-identically to no mask at all. The mask travels per-derive and
    is a preprocessor variant of derive.comp (MASK), never a runtime branch."""

    file: str | None = pfield(
        None, tier=Tier.POST, adv=True, ui="Mask",
        description="Path to a grayscale equirect (2:1) PNG mask that paints WHERE "
                    "the three Mask targets act (white = full effect, black = none). "
                    "Use forward slashes. None = no mask (all Mask targets inert). "
                    "The path is resolved relative to a loaded preset's folder and "
                    "re-saved next to a preset you save, so a preset stays portable; "
                    "a missing file at load warns and disables the mask (never crashes)",
    )
    band_fade: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Mask",
        description="Fade the busy features (storm tint, polar tint, detail, lanes) "
                    "back toward the plain band color where the mask is painted -- a "
                    "way to calm chosen regions to clean bands. Weight is "
                    "mask * this gain; 0 = off (byte-identical)",
    )
    emission_gain: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Mask",
        description="Modulate the night-side emission map (thermal/lightning glow + "
                    "aurora) by the mask, dimming the glow where the mask is dark. "
                    "Factor is mix(1, mask, this gain); 0 = off (byte-identical). "
                    "Only visible on the Emission map, not Color",
    )
    detail_gain: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Mask",
        description="Modulate color luminance/detail by the mask, settling painted-"
                    "dark regions while painted-bright regions stay untouched. Factor "
                    "is mix(1, mask, this gain); 0 = off (byte-identical)",
    )


class PhysicalParams(_Params):
    """Real-world scale hints passed through to the Blender importer."""

    radius_km: float = pfield(
        69911.0, tier=Tier.POST, lo=1000.0, hi=200000.0, adv=True, ui="Physical",
        description="Planet equatorial radius in kilometers, passed through to the "
                    "Blender importer for scale",
    )
    height_scale: float = pfield(
        0.004, tier=Tier.POST, lo=0.0, hi=0.05, adv=True, ui="Physical",
        description="Cloud-deck relief as a fraction of planet radius (full height-map range)",
    )
    height_midlevel: float = pfield(
        0.5, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Physical",
        description="Height-map value mapped to the mid cloud deck (Blender importer "
                    "reference level)",
    )
    ring_inner_km: float = pfield(
        74500.0, tier=Tier.POST, lo=1000.0, hi=1000000.0, adv=True, ui="Physical",
        description="Inner radius of the ring system in kilometers, measured from the "
                    "planet center (default = Saturn's C-ring inner edge). Only meaningful "
                    "when rings are enabled; passed through to the Blender importer, which "
                    "builds an annulus from ring_inner_km..ring_outer_km",
    )
    ring_outer_km: float = pfield(
        136780.0, tier=Tier.POST, lo=1000.0, hi=1000000.0, adv=True, ui="Physical",
        description="Outer radius of the ring system in kilometers (default = Saturn's "
                    "A-ring outer edge). Only meaningful when rings are enabled",
    )


class RingsParams(_Params):
    """Saturn-style ring system. DEFAULT OFF and a Blender-only product feature:
    rings are exported as a separate ``rings.exr`` radial strip and rebuilt as an
    annulus by the importer -- they are NOT part of the equirect map set and are
    invisible in the GUI preview. Because rings are a separate map, enabling them
    never touches the color/height/emission output (p05 render hash is unaffected).

    The radial optical-depth profile is a BOUNDED, hardcoded table modelling the
    real Saturn C/B/Cassini-division/A structure (see export/rings.py); the knobs
    here scale coverage/brightness/tint and add seeded fine grain. The radial
    EXTENT is set by physical.ring_inner_km / ring_outer_km."""

    enabled: bool = pfield(
        False, tier=Tier.POST, adv=True, ui="Rings",
        description="Export a ring texture strip (rings.exr) and, in Blender, build a "
                    "Saturn-style annulus from it. Blender-only -- invisible in the GUI "
                    "equirect preview. Off by default: the default export file-set "
                    "(color + height) is unchanged. No rand",
    )
    opacity: float = pfield(
        1.0, tier=Tier.POST, lo=0.0, hi=2.0, adv=True, ui="Rings",
        description="Multiplier on the ring alpha (coverage) derived from the optical-"
                    "depth table. 1.0 = physically-derived Beer-Lambert coverage",
    )
    brightness: float = pfield(
        1.0, tier=Tier.POST, lo=0.0, hi=2.0, adv=True, ui="Rings",
        description="Multiplier on the ice reflectance (ring RGB brightness)",
    )
    tint_color: tuple[float, float, float] = pfield(
        (0.86, 0.83, 0.78), tier=Tier.POST, adv=True, ui="Rings",
        description="Slightly warm ice tint (linear RGB) applied to the ring particles",
    )
    fine_grain: float = pfield(
        0.15, tier=Tier.POST, lo=0.0, hi=1.0, adv=True, ui="Rings",
        description="Amount of seeded fine-grain ringlet variation added on top of the "
                    "bounded optical-depth table (0 = the smooth table only). Uses the "
                    "master seed's 'rings' substream, so it is deterministic",
    )


class ProjectionKind(StrEnum):
    """Output projection for the exported map set."""
    EQUIRECT = "equirect"   # 2:1 equirectangular (default; the only legacy form)
    CUBE = "cube"           # 6-face cube map for game-engine / real-time use


class ExportParams(_Params):
    width: int = pfield(
        2048, tier=Tier.POST, lo=512, hi=16384, ui="Export",
        description="Equirect map width in pixels; height is width/2",
    )
    projection: ProjectionKind = pfield(
        ProjectionKind.EQUIRECT, tier=Tier.POST, ui="Export",
        description="Output projection. 'equirect' writes the classic 2:1 "
                    "equirectangular color/height(/emission) set (the default -- "
                    "unchanged file-set and manifest). 'cube' instead writes a "
                    "6-face cube map (px,nx,py,ny,pz,nz per map) sized width/4 per "
                    "face, for game engines / real-time renderers that texture a "
                    "sky-cube or cube-mapped sphere. Cube export bumps the manifest "
                    "schema to v2 (projection='cube', per-map 'faces' block); older "
                    "importers that only build equirect geometry reject it cleanly. "
                    "No rand.",
    )
    png_compression: int = pfield(
        2, tier=Tier.POST, lo=0, hi=9, ui="Export",
        description="PNG deflate level (low = much faster at 16K)",
    )
    flow_map: bool = pfield(
        False, tier=Tier.POST, ui="Export",
        description="Also export flow.exr: the sim's per-step velocity field "
                    "resampled to the equirect grid as an (east, north) flow "
                    "map (R = eastward, G = northward; B=0, A=1), so Blender / a "
                    "compositor can drive motion vectors or advected effects. "
                    "Off by default -- the default export file-set (color + "
                    "height) is unchanged. No rand.",
    )


class PlanetParams(_Params):
    seed: int = pfield(
        0, tier=Tier.RESTART, lo=0, hi=2**31 - 1, ui="Global",
        description="Master RNG seed; the development run is deterministic from this",
    )
    name: str = pfield(
        "unnamed", tier=Tier.POST, ui="Global",
        description="Display/preset name",
    )
    sim: SimParams = Field(default_factory=SimParams)
    solver: SolverParams = Field(default_factory=SolverParams)
    bands: BandsParams = Field(default_factory=BandsParams)
    jets: JetsParams = Field(default_factory=JetsParams)
    turbulence: TurbulenceParams = Field(default_factory=TurbulenceParams)
    storms: StormsParams = Field(default_factory=StormsParams)
    waves: WavesParams = Field(default_factory=WavesParams)
    poles: PolesParams = Field(default_factory=PolesParams)
    appearance: AppearanceParams = Field(default_factory=AppearanceParams)
    detail: DetailParams = Field(default_factory=DetailParams)
    mask: MaskParams = Field(default_factory=MaskParams)
    emission: EmissionParams = Field(default_factory=EmissionParams)
    physical: PhysicalParams = Field(default_factory=PhysicalParams)
    rings: RingsParams = Field(default_factory=RingsParams)
    export: ExportParams = Field(default_factory=ExportParams)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> PlanetParams:
        return cls.model_validate_json(text)

    def validation_warnings(self) -> list[str]:
        """Cross-field WARNINGS -- configurations that are valid but silently
        inert, so they must never be validation errors (legitimate presets may
        carry them; a kinematic preset derived from a vorticity one keeps its
        storm levers). The GUI surfaces these as toasts on preset load (B5-6).

        Current checks: the vorticity-only solid-core storm levers are exact
        no-ops under ``solver.type == "kinematic"`` (the stamp branch that
        consumes them never runs), e.g. a Neptune dark oval preset silently
        stays exposed to the whirlpool-winding artifact."""
        warnings: list[str] = []
        if self.solver.type == SolverType.KINEMATIC:
            for field_name in ("hero_solid_core", "oval_solid_core"):
                value = getattr(self.storms, field_name)
                if value != 0.0:
                    warnings.append(
                        f"storms.{field_name}={value:g} has no effect with the "
                        f"kinematic solver (vorticity-only lever); set "
                        f"solver.type=vorticity or reset it to 0"
                    )
        return warnings


@dataclass(frozen=True)
class FieldMeta:
    """Typed view over a ``pfield``'s ``json_schema_extra`` metadata, so panel
    code reads ``meta.adv``/``meta.tier`` instead of an untyped ``.get("adv")``
    on a bare dict. Values are the JSON-plain forms ``pfield`` stores (``tier`` as
    the ``Tier`` value, ``rand`` as a list); defaults match a non-``pfield`` leaf
    (no tier, not advanced), so the fields that drive visibility/badges behave
    exactly as the old ``extra.get(...)`` reads did."""

    tier: Any = None
    ui: str = ""
    log: bool = False
    adv: bool = False
    rand: list[Any] | None = None

    @classmethod
    def of(cls, info: Any) -> FieldMeta:
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        return cls(
            tier=extra.get("tier"),
            ui=extra.get("ui", ""),
            log=bool(extra.get("log", False)),
            adv=bool(extra.get("adv", False)),
            rand=extra.get("rand"),
        )


def field_meta(model: type[BaseModel], field_name: str) -> FieldMeta:
    """The typed ``FieldMeta`` for a field (an all-default ``FieldMeta`` if the
    field carries no ``pfield`` metadata)."""
    return FieldMeta.of(model.model_fields[field_name])

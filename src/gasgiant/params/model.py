"""The pydantic parameter tree.

Every tunable field carries metadata in ``json_schema_extra``:

- ``tier``: invalidation tier — what the engine must redo when the field changes
  (POST: re-derive maps only; VELOCITY: rebuild the velocity field, sim continues;
  RESTART: re-initialize the development run from step 0).
- ``rand``: (lo, hi) range used by seeded randomization, or None if the field is
  never randomized.
- ``log``: randomize/UI slider on a log scale.
- ``ui``: display group label for auto-generated panels.

Metadata is plain JSON data only — no callables, no GUI imports — so the core
stays GUI-agnostic in fact, not just in name.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MAX_BANDS = 40


class Tier(StrEnum):
    POST = "post"
    VELOCITY = "velocity"
    RESTART = "restart"


def pfield(
    default: Any,
    *,
    tier: Tier,
    lo: float | None = None,
    hi: float | None = None,
    rand: tuple[float, float] | None = None,
    log: bool = False,
    ui: str = "",
    description: str = "",
) -> Any:
    extra: dict[str, Any] = {"tier": tier.value, "ui": ui, "log": log}
    if rand is not None:
        extra["rand"] = list(rand)
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


class PaletteRow(_Params):
    """One palette gradient anchored at a signed latitude (degrees, north
    positive). Rows are blended across latitude at derive time; a single row
    reproduces the latitude-independent v1 palette exactly."""

    latitude: float = Field(0.0, ge=-90.0, le=90.0)
    stops: list[GradientStop]


def default_palette_rows() -> list[PaletteRow]:
    return [PaletteRow(latitude=0.0, stops=list(DEFAULT_PALETTE))]


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
        0.012, tier=Tier.RESTART, lo=0.001, hi=0.1, rand=(0.005, 0.03), log=True, ui="Bands",
        description="Half-width of band-edge transitions, radians of latitude",
    )
    value_contrast: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=2.0, rand=(0.6, 1.3), ui="Bands",
        description="Zone/belt brightness separation multiplier",
    )
    warp_amount: float = pfield(
        0.035, tier=Tier.RESTART, lo=0.0, hi=0.3, rand=(0.01, 0.09), ui="Bands",
        description="Band-boundary meander amplitude, radians of latitude",
    )
    warp_freq: float = pfield(
        3.0, tier=Tier.RESTART, lo=0.5, hi=16.0, rand=(1.5, 6.0), log=True, ui="Bands",
        description="Band-boundary meander spatial frequency",
    )
    detail_amount: float = pfield(
        0.10, tier=Tier.RESTART, lo=0.0, hi=0.5, rand=(0.04, 0.2), ui="Bands",
        description="Small-scale color-index noise amplitude",
    )
    hue_jitter: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=0.15, rand=(0.0, 0.08), ui="Bands",
        description="Per-band color-index offset along the palette (NEB-orange vs "
                    "SEB-brown variation); seeded independently of the band layout",
    )
    variance_amount: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=0.3, rand=(0.02, 0.12), ui="Bands",
        description="Within-band longitudinal color drift (real belts hold several "
                    "hues at once, varying slowly with longitude)",
    )
    faded_sector: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.0, 0.7), ui="Bands",
        description="SEB-fade: one belt gets a pale desaturated sector spanning "
                    "~100 degrees of longitude",
    )
    contrast_envelope: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.3, 0.8), ui="Bands",
        description="Banding contrast collapse poleward of ~45 deg toward mottle "
                    "(the real latitude-contrast profile)",
    )
    lane_density: float = pfield(
        0.0, tier=Tier.VELOCITY, lo=0.0, hi=1.0, rand=(0.0, 0.8), ui="Bands",
        description="Thin dark lane lines at jet cores, drawn analytically at "
                    "derive time (a 1-3 px line cannot survive the sim grid)",
    )
    edge_diversity: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.2, 0.8), ui="Bands",
        description="Per-edge softness variation: some band edges diffuse, some "
                    "sharp (uniform edges are a procedural tell)",
    )
    width_tail: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.0, 0.7), ui="Bands",
        description="Heavier-tailed band width distribution (real maps mix very "
                    "broad zones with thin strips)",
    )
    detail_freq: float = pfield(
        12.0, tier=Tier.RESTART, lo=2.0, hi=64.0, rand=(6.0, 24.0), log=True, ui="Bands",
        description="Small-scale noise spatial frequency",
    )


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
        description="Equatorial superrotation jet peak speed (negative = retrograde)",
    )
    equatorial_width: float = pfield(
        0.12, tier=Tier.VELOCITY, lo=0.03, hi=0.4, rand=(0.07, 0.25), ui="Jets",
        description="Equatorial jet half-width, radians of latitude",
    )
    polar_decay: float = pfield(
        0.5, tier=Tier.VELOCITY, lo=0.0, hi=1.0, rand=(0.3, 0.8), ui="Jets",
        description="How strongly jet amplitudes decay toward the poles",
    )


class TurbulenceParams(_Params):
    intensity: float = pfield(
        1.0, tier=Tier.VELOCITY, lo=0.0, hi=3.0, rand=(0.5, 1.8), ui="Turbulence",
        description="Global turbulence (curl-noise) amplitude",
    )
    shear_coupling: float = pfield(
        1.0, tier=Tier.VELOCITY, lo=0.0, hi=3.0, rand=(0.5, 1.5), ui="Turbulence",
        description="Extra turbulence where jet shear is strong",
    )
    belt_boost: float = pfield(
        1.6, tier=Tier.VELOCITY, lo=1.0, hi=4.0, rand=(1.2, 2.5), ui="Turbulence",
        description="Turbulence multiplier inside dark belts (cyclonic bands)",
    )
    scale: float = pfield(
        6.0, tier=Tier.VELOCITY, lo=1.0, hi=32.0, rand=(4.0, 12.0), log=True, ui="Turbulence",
        description="Base spatial frequency of the turbulence noise",
    )
    evolution_rate: float = pfield(
        0.012, tier=Tier.VELOCITY, lo=0.0, hi=0.1, ui="Turbulence",
        description="How fast the turbulence pattern decorrelates per step",
    )
    relax_tau: float = pfield(
        350.0, tier=Tier.RESTART, lo=50.0, hi=2000.0, log=True, ui="Turbulence",
        description="Relaxation time (steps) pulling band color/height back toward the stamp",
    )
    replenish_rate: float = pfield(
        0.015, tier=Tier.RESTART, lo=0.0, hi=0.1, ui="Turbulence",
        description="Fresh detail-noise blended into the detail tracer per step",
    )
    kh_amplitude: float = pfield(
        0.35, tier=Tier.VELOCITY, lo=0.0, hi=2.0, rand=(0.1, 0.8), ui="Turbulence",
        description="Kelvin-Helmholtz wave amplitude along high-shear band boundaries",
    )
    kh_wavenumber: int = pfield(
        24, tier=Tier.VELOCITY, lo=4, hi=80, rand=(14, 40), ui="Turbulence",
        description="KH billow longitudinal wavenumber",
    )


class StormsParams(_Params):
    hero_count: int = pfield(
        1, tier=Tier.RESTART, lo=0, hi=3, rand=(0, 2), ui="Storms",
        description="GRS-class giant anticyclones",
    )
    hero_radius: float = pfield(
        0.10, tier=Tier.RESTART, lo=0.03, hi=0.25, rand=(0.06, 0.16), ui="Storms",
        description="Hero vortex core radius, radians of arc",
    )
    hero_strength: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.2, hi=3.0, rand=(0.7, 1.6), ui="Storms",
    )
    oval_density: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=3.0, rand=(0.4, 1.8), ui="Storms",
        description="White-oval anticyclone population multiplier",
    )
    barge_density: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=3.0, rand=(0.3, 1.5), ui="Storms",
        description="Brown-barge cyclone population multiplier (belts)",
    )
    pearls_count: int = pfield(
        7, tier=Tier.RESTART, lo=0, hi=14, rand=(0, 9), ui="Storms",
        description="String-of-pearls ovals on one seeded latitude (0 = off)",
    )
    wake_turbulence: float = pfield(
        1.8, tier=Tier.RESTART, lo=0.0, hi=5.0, rand=(1.0, 3.0), ui="Storms",
        description="Turbulence boost in the wake wedge downstream of hero storms",
    )
    outbreak_count: int = pfield(
        0, tier=Tier.RESTART, lo=0, hi=3, rand=(0, 2), ui="Storms",
        description="Convective outbreaks (Great-White-Spot events) during the development run",
    )
    outbreak_strength: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.2, hi=3.0, ui="Storms",
    )
    small_density: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=3.0, rand=(0.4, 1.8), ui="Storms",
        description="Small-storm field: sub-oval white spots and dark spots scattered "
                    "in loose latitude rows (0 = off, the pre-v1.1 look)",
    )
    stamp_contrast: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=2.0, rand=(0.8, 1.3), ui="Storms",
        description="Tracer-stamp contrast of ovals/barges/pearls/small storms (1 = v1)",
    )
    merge_rate: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.0, 0.8), ui="Storms",
        description="Anticyclone merger aggressiveness: converging same-sign "
                    "ovals coalesce when their gap falls under ~1.5*rate*(r1+r2), "
                    "and generation seeds convergent pairs so mergers actually "
                    "occur during the dev run (0 = off, the v1.1 behavior)",
    )
    merge_debris: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=2.0, ui="Storms",
        description="Brightness of the transient turbulent collar a fresh "
                    "merger leaves behind (inert while merge_rate is 0)",
    )


class WavesParams(_Params):
    festoon_strength: float = pfield(
        0.8, tier=Tier.RESTART, lo=0.0, hi=2.0, rand=(0.0, 1.4), ui="Waves",
        description="Festoon plumes + hot spots on the equatorial belt edge (0 = off)",
    )
    festoon_wavenumber: int = pfield(
        12, tier=Tier.RESTART, lo=4, hi=24, rand=(8, 16), ui="Waves",
        description="Rossby wavenumber of the festoon/hot-spot train",
    )
    hotspot_depth: float = pfield(
        0.6, tier=Tier.RESTART, lo=0.0, hi=1.0, rand=(0.2, 0.9), ui="Waves",
        description="Depth of the cloud-free hot spots at the wave troughs",
    )
    ribbon_strength: float = pfield(
        0.0, tier=Tier.RESTART, lo=0.0, hi=2.0, rand=(0.0, 1.0), ui="Waves",
        description="Saturn-style ribbon wave on one mid-latitude jet (0 = off)",
    )
    ribbon_wavenumber: int = pfield(
        12, tier=Tier.RESTART, lo=4, hi=30, ui="Waves",
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
        default_palette_rows(), tier=Tier.POST, ui="Appearance",
        description="Latitude-anchored color gradients indexed by the color tracer "
                    "(belt dark -> zone bright), blended across signed latitude",
    )
    storm_tints: list[GradientStop] = pfield(
        DEFAULT_STORM_TINTS, tier=Tier.POST, ui="Appearance",
        description="Secondary tint axis for storms/festoons/hot spots",
    )
    haze_amount: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, rand=(0.0, 0.7), ui="Appearance",
        description="Global haze: the Jupiter (0) to Saturn (~0.6) axis",
    )
    haze_color: tuple[float, float, float] = pfield(
        (0.85, 0.78, 0.62), tier=Tier.POST, ui="Appearance",
    )
    contrast: float = pfield(
        1.0, tier=Tier.POST, lo=0.2, hi=2.0, rand=(0.8, 1.2), ui="Appearance",
    )
    saturation: float = pfield(
        1.0, tier=Tier.POST, lo=0.0, hi=2.0, rand=(0.7, 1.2), ui="Appearance",
    )
    gamma: float = pfield(
        1.0, tier=Tier.POST, lo=0.4, hi=2.5, ui="Appearance",
        description="Final tone-curve gamma on the color map",
    )
    polar_tint_color: tuple[float, float, float] = pfield(
        (0.42, 0.50, 0.58), tier=Tier.POST, ui="Appearance",
        description="Polar cap tint (Juno blue-gray); applied where cloud tops "
                    "are LOW -- the blue is structural, bright clouds stay pearly",
    )
    polar_tint_strength: float = pfield(
        0.0, tier=Tier.POST, lo=0.0, hi=1.0, rand=(0.2, 0.7), ui="Appearance",
        description="Polar tint blend strength (0 = off, the pre-v1.1 look)",
    )
    polar_tint_start_lat: float = pfield(
        55.0, tier=Tier.POST, lo=30.0, hi=80.0, ui="Appearance",
        description="Latitude (deg) where the polar tint begins",
    )


class PhysicalParams(_Params):
    """Real-world scale hints passed through to the Blender importer."""

    radius_km: float = pfield(69911.0, tier=Tier.POST, lo=1000.0, hi=200000.0, ui="Physical")
    height_scale: float = pfield(
        0.004, tier=Tier.POST, lo=0.0, hi=0.05, ui="Physical",
        description="Cloud-deck relief as a fraction of planet radius (full height-map range)",
    )
    height_midlevel: float = pfield(0.5, tier=Tier.POST, lo=0.0, hi=1.0, ui="Physical")


class ExportParams(_Params):
    width: int = pfield(
        2048, tier=Tier.POST, lo=512, hi=16384, ui="Export",
        description="Equirect map width in pixels; height is width/2",
    )
    png_compression: int = pfield(
        2, tier=Tier.POST, lo=0, hi=9, ui="Export",
        description="PNG deflate level (low = much faster at 16K)",
    )


class PlanetParams(_Params):
    seed: int = pfield(0, tier=Tier.RESTART, lo=0, hi=2**31 - 1, ui="Global")
    name: str = pfield("unnamed", tier=Tier.POST, ui="Global")
    sim: SimParams = Field(default_factory=SimParams)
    bands: BandsParams = Field(default_factory=BandsParams)
    jets: JetsParams = Field(default_factory=JetsParams)
    turbulence: TurbulenceParams = Field(default_factory=TurbulenceParams)
    storms: StormsParams = Field(default_factory=StormsParams)
    waves: WavesParams = Field(default_factory=WavesParams)
    poles: PolesParams = Field(default_factory=PolesParams)
    appearance: AppearanceParams = Field(default_factory=AppearanceParams)
    detail: DetailParams = Field(default_factory=DetailParams)
    physical: PhysicalParams = Field(default_factory=PhysicalParams)
    export: ExportParams = Field(default_factory=ExportParams)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> PlanetParams:
        return cls.model_validate_json(text)


def field_meta(model: type[BaseModel], field_name: str) -> dict[str, Any]:
    """The json_schema_extra metadata dict for a field ({} if none)."""
    info = model.model_fields[field_name]
    extra = info.json_schema_extra
    return dict(extra) if isinstance(extra, dict) else {}

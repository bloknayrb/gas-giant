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
    detail_freq: float = pfield(
        12.0, tier=Tier.RESTART, lo=2.0, hi=64.0, rand=(6.0, 24.0), log=True, ui="Bands",
        description="Small-scale noise spatial frequency",
    )


class AppearanceParams(_Params):
    palette: list[GradientStop] = pfield(
        DEFAULT_PALETTE, tier=Tier.POST, ui="Appearance",
        description="Color gradient indexed by the color tracer (belt dark -> zone bright)",
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
    bands: BandsParams = Field(default_factory=BandsParams)
    appearance: AppearanceParams = Field(default_factory=AppearanceParams)
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

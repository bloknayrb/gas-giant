from __future__ import annotations

import pytest
from pydantic import ValidationError

from gasgiant.params.model import BandsParams, PlanetParams, Tier, field_meta


def test_defaults_validate():
    p = PlanetParams()
    assert p.bands.count >= 2
    assert p.export.width == 2048


def test_json_round_trip():
    p = PlanetParams(seed=123)
    p.bands.count = 9
    p.appearance.haze_amount = 0.4
    q = PlanetParams.from_json(p.to_json())
    assert q == p


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        PlanetParams.model_validate({"seed": 1, "tubrulence": 0.5})


def test_nested_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        PlanetParams.model_validate({"bands": {"cuont": 9}})


def test_bounds_enforced():
    with pytest.raises(ValidationError):
        BandsParams(count=1)
    with pytest.raises(ValidationError):
        BandsParams(count=999)


def test_every_tunable_field_declares_a_tier():
    from pydantic import BaseModel

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, info in model.model_fields.items():
            ann = info.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                walk(ann, f"{prefix}{name}.")
                continue
            extra = info.json_schema_extra
            if name in ("pos", "color"):  # GradientStop components
                continue
            assert isinstance(extra, dict) and extra.get("tier") in {t.value for t in Tier}, (
                f"field {prefix}{name} missing tier metadata"
            )

    walk(PlanetParams, "")


def test_field_meta_helper():
    meta = field_meta(BandsParams, "count")
    assert meta["tier"] == Tier.RESTART.value
    assert meta["rand"] == [6, 24]


def test_every_pfield_has_a_description():
    """Every pfield leaf needs a description: panels.py surfaces it as the
    slider tooltip. The pfield-leaf set is derived programmatically -- a leaf
    is a pfield if its json_schema_extra carries 'tier' -- so non-pfield
    leaves (GradientStop.pos/.color, PaletteRow.latitude, BandTemplate's
    edges_deg/values/heights, all declared with plain Field()) are excluded
    without a hand-maintained skip list."""
    from pydantic import BaseModel

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, info in model.model_fields.items():
            path = f"{prefix}{name}"
            extra = info.json_schema_extra
            if isinstance(extra, dict) and "tier" in extra:
                assert info.description, f"pfield {path} has no description"
            ann = info.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                walk(ann, f"{path}.")

    walk(PlanetParams, "")

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

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
    assert meta.tier == Tier.RESTART.value
    assert meta.rand == [6, 24]


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


# -- Phase 4: Basic/Advanced curation guard tests (consolidated list #6, #10) --


def _pfield_leaves(model: type[BaseModel] = PlanetParams, prefix: str = ""):
    """Every pfield leaf as (dotted path, top-level section name, extra
    dict), derived the same programmatic way as test_every_pfield_has_a_description
    ('tier' in json_schema_extra) -- never a hand-maintained skip list."""
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            yield from _pfield_leaves(ann, f"{path}.")
            continue
        extra = info.json_schema_extra
        if isinstance(extra, dict) and "tier" in extra:
            top = path.split(".", 1)[0]
            yield path, top, extra


def test_adv_is_a_bool_on_every_pfield_leaf():
    for path, _top, extra in _pfield_leaves():
        assert isinstance(extra.get("adv"), bool), f"{path}: adv is not a bool ({extra.get('adv')!r})"


def test_overall_basic_set_is_non_empty():
    """Guard test #6 (H1), part a: at least one leaf across the whole tree
    stays adv=False (visible without ever toggling Advanced) -- would fail
    if curation accidentally marked everything advanced."""
    basic = [path for path, _top, extra in _pfield_leaves() if not extra.get("adv")]
    assert basic, "no Basic-visible (adv=False) leaves anywhere in PlanetParams"


def test_every_non_allowlisted_section_has_a_basic_leaf():
    """Guard test #6 (H1), part b: every top-level section except the
    reviewed fully-advanced allowlist (solver, emission, physical -- and
    baroclinic, nested under solver, so it's covered by the same 'solver'
    top-level key and listing it again would be redundant-but-harmless) must
    keep at least one adv=False leaf, so a newcomer never lands on a section
    that's silently empty in Basic mode. This is a real assertion: it would
    fail today if e.g. every Storms field were marked adv=True."""
    from collections import defaultdict

    allowlist = {"solver", "emission", "physical"}
    basic_count: dict[str, int] = defaultdict(int)
    for _path, top, extra in _pfield_leaves():
        if not extra.get("adv"):
            basic_count[top] += 1

    all_sections = {top for _path, top, _extra in _pfield_leaves()}
    for section in all_sections - allowlist:
        assert basic_count.get(section, 0) > 0, (
            f"section {section!r} (not in the fully-advanced allowlist) has zero "
            f"Basic-visible leaves"
        )
    # And the allowlisted sections are indeed fully advanced (sanity check that
    # the allowlist itself isn't stale/wrong).
    for section in allowlist:
        assert basic_count.get(section, 0) == 0, (
            f"section {section!r} is in the fully-advanced allowlist but has "
            f"{basic_count.get(section, 0)} Basic-visible leaves"
        )


def test_adv_does_not_perturb_field_meta_or_serialization():
    """Adding 'adv' to the json_schema_extra dict must stay plain-JSON and
    must not affect field_meta()'s existing keys or preset (de)serialization
    -- presets store VALUES (PlanetParams.model_dump()), never
    json_schema_extra metadata, so this is mostly a sanity check that the
    reasoning holds."""
    import json

    meta = field_meta(BandsParams, "count")
    assert meta.tier == Tier.RESTART.value
    assert meta.rand == [6, 24]
    assert meta.adv is False  # count is a Basic headline knob

    p = PlanetParams(seed=99)
    p.bands.template = None  # no-op assignment; keep default
    dumped = json.loads(p.to_json())
    assert "adv" not in dumped["bands"], "adv is schema metadata, not a param value"
    assert PlanetParams.from_json(p.to_json()) == p

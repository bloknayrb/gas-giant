"""Field-driven detail: the FIELD_DRIVE variant selector is METADATA (pfield
``field_drive=True``), mirroring the DETAIL_FX ``fx=True`` machinery. Only
``field_drive`` selects the variant; ``field_scale``/``field_vort_influence``
are plain sample-time tunables and must NOT be selectors (design M5)."""
from __future__ import annotations

from gasgiant.params.model import DetailParams
from gasgiant.render import detail as detail_mod

EXPECTED_FIELD_DRIVE_LEVERS = {"field_drive"}


def test_field_drive_metadata_matches_selector():
    assert set(detail_mod._FIELD_DRIVE_PARAMS) == EXPECTED_FIELD_DRIVE_LEVERS


def test_field_drive_flag_lives_on_exactly_field_drive():
    flagged = {
        name
        for name, info in DetailParams.model_fields.items()
        if isinstance(info.json_schema_extra, dict)
        and info.json_schema_extra.get("field_drive")
    }
    assert flagged == EXPECTED_FIELD_DRIVE_LEVERS


def test_predicate_off_by_default_and_only_field_drive_selects():
    assert detail_mod.field_drive_enabled(DetailParams()) is False
    assert detail_mod.field_drive_enabled(DetailParams(field_drive=1e-6)) is True
    # field_scale / field_vort_influence alone must NOT select the variant
    assert detail_mod.field_drive_enabled(DetailParams(field_scale=4.0)) is False
    assert detail_mod.field_drive_enabled(
        DetailParams(field_vort_influence=1.0)
    ) is False


def test_new_levers_are_post_tier_and_no_rand():
    for name in ("field_drive", "field_scale", "field_vort_influence"):
        extra = DetailParams.model_fields[name].json_schema_extra
        assert extra["tier"] == "post", name
        assert "rand" not in extra, f"{name} rand draw would reorder randomize"


def test_new_levers_are_not_fx_flagged():
    """The FIELD_DRIVE levers are independent of DETAIL_FX; flagging one fx would
    wrongly force the DETAIL_FX variant and expand EXPECTED_FX_LEVERS."""
    for name in ("field_drive", "field_scale", "field_vort_influence"):
        extra = DetailParams.model_fields[name].json_schema_extra
        assert "fx" not in extra, f"{name} must not be fx-flagged"

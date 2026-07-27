from __future__ import annotations

import pytest
from pydantic import ValidationError

from gasgiant.params.model import (
    BandsParams,
    BandTemplate,
    GradientStop,
    PaletteRow,
    PlanetParams,
    PoleParams,
    StormOverride,
    Tier,
    field_meta,
    iter_leaves,
    iter_pfields,
)


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


#: Models that are DATA RECORDS, not param sections -- a gradient stop, a
#: palette row, an authored band template. Their leaves are values a preset
#: carries, not knobs the GUI draws, so they legitimately declare no tier.
#: Excluded by MODEL rather than by field name: the previous name-based skip
#: ("pos", "color") would also have silenced a real ``pos`` added anywhere else
#: in the tree, and it was only small because the walk it guarded never reached
#: PaletteRow or BandTemplate in the first place.
_DATA_RECORD_MODELS = (GradientStop, PaletteRow, BandTemplate)


def test_every_tunable_field_declares_a_tier():
    """The one guard asserting the COMPLEMENT: that a leaf which SHOULD be a
    pfield actually is one.

    It cannot be built on ``iter_pfields``, which filters on ``tier`` and so is
    definitionally blind to a leaf missing it. Before this used ``iter_leaves``
    it hand-rolled an annotation-only walk that reached 204 of 236 leaves --
    every ``storms.cast.*`` lever among the missing -- so a ``StormOverride``
    field declared with a plain ``Field()`` was invisible to every guard in the
    repo while ``panels._draw_flat_model_fields`` drew it anyway, with no tier
    badge and an empty tooltip."""
    tiers = {t.value for t in Tier}
    for leaf in iter_leaves():
        if leaf.model in _DATA_RECORD_MODELS:
            continue
        extra = leaf.info.json_schema_extra
        assert isinstance(extra, dict) and extra.get("tier") in tiers, (
            f"field {leaf.path} missing tier metadata"
        )


def test_the_tier_guard_still_reaches_the_awkward_corners():
    """Pins the reach of the walk above, so a traversal regression shows up as
    a red test rather than as quietly checking fewer fields."""
    paths = {leaf.path for leaf in iter_leaves() if leaf.model not in _DATA_RECORD_MODELS}
    assert "storms.cast.radius" in paths, "list[StormOverride] not descended"
    assert "poles.south.strength" in paths, "the second PoleParams instance was collapsed"
    assert "solver.baroclinic.gain" in paths, "doubly-nested section not reached"


def test_data_record_models_really_declare_no_tier():
    """Keeps the exclusion honest: if one of these gains a pfield, the skip
    above would silently stop checking it."""
    for leaf in iter_leaves():
        if leaf.model in _DATA_RECORD_MODELS:
            extra = leaf.info.json_schema_extra
            assert not (isinstance(extra, dict) and "tier" in extra), (
                f"{leaf.path} is now a pfield; drop {leaf.model.__name__} from "
                "_DATA_RECORD_MODELS so the tier guard covers it"
            )


def test_field_meta_helper():
    meta = field_meta(BandsParams, "count")
    assert meta.tier == Tier.RESTART.value
    assert meta.rand == [6, 24]


def test_every_pfield_has_a_description():
    """Every pfield leaf needs a description: panels.py surfaces it as the
    slider tooltip. ``iter_pfields`` supplies the leaf set (see its docstring
    for why the walk is structural rather than a skip list, and for the two
    traversal subtleties that silently under-count if you re-roll it)."""
    for leaf in iter_pfields():
        assert leaf.description, f"pfield {leaf.path} has no description"


# -- Phase 4: Basic/Advanced curation guard tests (consolidated list #6, #10) --


def _pfield_leaves():
    """``(dotted path, top-level section, extra dict)`` for every pfield leaf.

    Kept as a thin adapter over ``iter_pfields`` rather than its own walk: this
    file previously carried a second, subtly different traversal, and the two
    disagreed by 22 leaves -- every ``storms.cast.*`` lever was invisible to the
    three curation guards below, with nothing red to say so."""
    for leaf in iter_pfields():
        yield leaf.path, leaf.path.split(".", 1)[0], leaf.info.json_schema_extra


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
    reviewed fully-advanced allowlist (solver, emission, physical, mask, rings
    -- and baroclinic, nested under solver, so it's covered by the same 'solver'
    top-level key and listing it again would be redundant-but-harmless) must
    keep at least one adv=False leaf, so a newcomer never lands on a section
    that's silently empty in Basic mode. This is a real assertion: it would
    fail today if e.g. every Storms field were marked adv=True.

    'mask' is fully advanced like emission: an opt-in art-direction feature that
    is a no-op (byte-identical) until a power user supplies a paint mask. 'rings'
    is likewise fully advanced: a Blender-only, default-off product feature
    invisible in the GUI preview."""
    from collections import defaultdict

    allowlist = {"solver", "emission", "physical", "mask", "rings"}
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


# -- the traversal invariants iter_pfields' docstring calls load-bearing --------
#
# Both were measured revertible with the ENTIRE fast tier green: dropping the
# get_args descent takes 226 pfield leaves to 204, and deduping on
# (model, name) takes it to 221. Every consumer degrades identically and
# silently -- it simply checks fewer leaves -- so the walk needs its own
# assertions rather than relying on a downstream test to notice.


def test_the_walk_descends_into_a_list_of_models():
    """storms.cast is list[StormOverride], which fails issubclass(ann, BaseModel).
    An annotation-only walk misses all 22 cast levers -- the exact bug that left
    the Basic/Advanced curation guards blind to them."""
    paths = {leaf.path for leaf in iter_pfields()}
    assert {f"storms.cast.{n}" for n in StormOverride.model_fields} <= paths


def test_the_walk_keeps_both_instances_of_a_reused_submodel():
    """poles.north and poles.south are two PoleParams instances. Keying on
    (model, name) rather than on the dotted path silently collapses 5 leaves --
    and is why 226 leaves carry only 221 distinct descriptions."""
    paths = {leaf.path for leaf in iter_pfields()}
    for pole in ("north", "south"):
        assert {f"poles.{pole}.{n}" for n in PoleParams.model_fields} <= paths


def test_a_nesting_field_that_is_itself_a_tunable_is_still_a_leaf():
    """storms.cast and bands.template both nest AND carry pfield metadata.
    Yielding only non-nesting fields drops them -- which is how two hand-written
    wave tables lost storms.cast, the field driving the whole cast editor."""
    paths = {leaf.path for leaf in iter_pfields()}
    assert {"storms.cast", "bands.template"} <= paths


def test_pfields_are_the_tier_carrying_subset_of_leaves():
    """Ties the two walks together, so they cannot drift apart the way the
    hand-rolled copies did."""
    leaves = list(iter_leaves())
    pfields = {leaf.path for leaf in iter_pfields()}
    expected = {
        leaf.path for leaf in leaves
        if isinstance(leaf.info.json_schema_extra, dict)
        and "tier" in leaf.info.json_schema_extra
    }
    assert pfields == expected
    assert pfields < {leaf.path for leaf in leaves}, "sanity: some leaves are not pfields"

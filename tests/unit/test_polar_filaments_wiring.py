"""A2-1 backfill: dedicated test for ``detail.polar_filaments`` (PR #11).

The lever shipped with ZERO test hits (grep polar_filaments in tests/ was
empty) while sibling levers (mottle, belt_texture, zone_texture) have tests --
exactly the convention-enforced sync point the review flagged. CPU-side
coverage of the pfield/fx wiring and kernel uniform presence; the
compiled-variant uniform presence is enforced for ALL fx levers by the A2-6
build tripwire (gpu-marked in tests/gpu/test_detail_fx.py).
"""
from __future__ import annotations

from gasgiant.gl.context import _load_flattened
from gasgiant.params.model import DetailParams, PlanetParams, Tier
from gasgiant.render.detail import detail_fx_enabled


def test_polar_filaments_pfield_metadata():
    info = DetailParams.model_fields["polar_filaments"]
    extra = info.json_schema_extra
    assert extra["tier"] == Tier.POST.value
    assert extra.get("fx") is True, "polar_filaments must be a DETAIL_FX lever"
    assert "rand" not in extra, "adding a rand draw would reorder seeded randomize"
    assert DetailParams().polar_filaments == 0.0, "default-off is the byte-identity contract"


def test_polar_filaments_selects_fx_variant():
    assert detail_fx_enabled(DetailParams()) is False
    assert detail_fx_enabled(DetailParams(polar_filaments=0.5)) is True


def test_polar_filaments_uniform_declared_and_consumed_in_kernel():
    """The flattened detail.comp must both DECLARE u_polar_filaments and consume
    it in an effect block: a declared-but-unused uniform is exactly the silent
    no-op the KeyError-suppressing _set would hide (the driver prunes it and the
    lever goes inert with no error)."""
    source, _ = _load_flattened("gasgiant.render.kernels", "detail.comp", {"DETAIL_FX": "1"})
    assert "uniform float u_polar_filaments" in source
    uses = source.count("u_polar_filaments") - 1  # minus the declaration
    assert uses >= 1, "u_polar_filaments declared but never consumed in the kernel"


def test_polar_filaments_is_post_tier():
    """POST tier: a detail-lever edit re-derives maps only (mirrors
    test_new_detail_knobs_are_post_tier for the earlier levers)."""
    from gasgiant.engine.invalidation import diff_tiers

    a = PlanetParams()
    b = a.model_copy(deep=True)
    b.detail.polar_filaments = 0.8
    assert diff_tiers(a, b) == {Tier.POST}

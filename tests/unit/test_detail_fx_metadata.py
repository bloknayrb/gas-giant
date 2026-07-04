"""A2-6: the DETAIL_FX lever list is METADATA, not a hand-list, and the
silent-uniform-no-op hazard has a loud build-time tripwire.

Previously render/detail.py hand-maintained an 8-way OR (`fx_on`) plus per-lever
`_set` calls whose KeyError suppression means a forgotten/renamed uniform is
silently inert. The compensating control the codebase already invented (the B1
tripwire in sim/solver.py) was applied exactly once. This generalizes it:

- each DETAIL_FX lever pfield carries ``fx=True`` metadata;
- the fx-variant selection predicate is derived from that metadata;
- at fx-program build time every fx param's ``u_<name>`` uniform must exist in
  the compiled DETAIL_FX variant, or the build raises (loud, at startup).

CPU side here; the end-to-end compiled-variant tripwire test (a deliberately
dropped uniform must fail the build) is GPU-marked in tests/gpu/test_detail_fx.py.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from gasgiant.params.model import DetailParams
from gasgiant.render import detail as detail_mod

# The 8 levers the hand-written fx_on predicate ORed (render/detail.py) -- the
# review's A2-6 finding. If a NEW fx lever lands, adding fx=True to its pfield
# updates predicate + tripwire + this list's failure message in one place.
EXPECTED_FX_LEVERS = {
    "intermittency",
    "hero_spiral",
    "hero_collar_wrap",
    "belt_texture",
    "belt_texture_fine",
    "zone_texture",
    "mottle",
    "polar_filaments",
}


def test_fx_metadata_matches_the_detail_fx_levers():
    assert set(detail_mod._FX_PARAMS) == EXPECTED_FX_LEVERS


def test_fx_flag_lives_on_exactly_those_pfields():
    flagged = {
        name
        for name, info in DetailParams.model_fields.items()
        if isinstance(info.json_schema_extra, dict) and info.json_schema_extra.get("fx")
    }
    assert flagged == EXPECTED_FX_LEVERS


def test_fx_predicate_derived_from_metadata():
    assert detail_mod.detail_fx_enabled(DetailParams()) is False
    for name in EXPECTED_FX_LEVERS:
        solo = DetailParams(**{name: 1e-6})
        assert detail_mod.detail_fx_enabled(solo) is True, name
    # a non-fx lever alone must NOT select the fx variant (byte-identity path)
    assert detail_mod.detail_fx_enabled(DetailParams(striation_amount=0.6)) is False
    assert detail_mod.detail_fx_enabled(DetailParams(hero_calm=1.0)) is False


def test_tripwire_passes_when_all_uniforms_present():
    prog = {f"u_{name}": object() for name in EXPECTED_FX_LEVERS}
    detail_mod._assert_fx_uniforms(prog)  # must not raise


def test_tripwire_raises_naming_the_missing_uniform():
    prog = {f"u_{name}": object() for name in EXPECTED_FX_LEVERS}
    del prog["u_polar_filaments"]
    with pytest.raises(RuntimeError, match="u_polar_filaments"):
        detail_mod._assert_fx_uniforms(prog)


def _fx_on_block() -> ast.If:
    """The ``if fx_on:`` dispatch block inside DetailSynth.synthesize."""
    src = textwrap.dedent(inspect.getsource(detail_mod.DetailSynth.synthesize))
    fn = ast.parse(src).body[0]
    for node in ast.walk(fn):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "fx_on"):
            return node
    raise AssertionError("no `if fx_on:` block found in DetailSynth.synthesize")


def test_fx_levers_cross_reference_the_synthesize_dispatch_block():
    """W8 cross-ref gate for the LAST hand-maintained fx sync point: the
    per-lever ``_set(prog, "u_<name>", params.<name>)`` calls in
    DetailSynth.synthesize. The predicate and the uniform tripwire are already
    DERIVED from the ``fx=True`` pfield metadata, but the Python-side upload
    block is hand-written -- a new fx lever whose uniform exists in the shader
    would pass the build tripwire yet stay at 0.0 forever if its ``_set`` call
    is missing. Both directions:

    - every fx-flagged pfield is read (``params.<name>``) and uploaded
      (``"u_<name>"``) inside the ``if fx_on:`` block;
    - every ``params.<field>`` the block consumes is fx-flagged (a non-fx
      lever wired here would silently require the fx variant to work).
    """
    block = _fx_on_block()
    fx = set(detail_mod._FX_PARAMS)
    params_reads = {
        node.attr
        for node in ast.walk(block)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "params"
    }
    uniform_names = {
        node.value
        for node in ast.walk(block)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("u_")
    }
    assert params_reads == fx, (
        f"fx pfield metadata vs synthesize `if fx_on:` block out of sync: "
        f"flagged-but-not-wired={sorted(fx - params_reads)}, "
        f"wired-but-not-flagged={sorted(params_reads - fx)}"
    )
    not_uploaded = {name for name in fx if f"u_{name}" not in uniform_names}
    assert not not_uploaded, (
        f"fx lever(s) read but never uploaded to their u_<name> uniform "
        f"inside the fx_on block: {sorted(not_uploaded)}"
    )


def test_fx_levers_have_no_rand_metadata():
    """The fx flag is pure metadata on EXISTING fields: none of them carries a
    ``rand`` range, so the seeded-randomize draw order is untouched (the pinned
    golden in test_randomize.py is the cross-check)."""
    for name in EXPECTED_FX_LEVERS:
        extra = DetailParams.model_fields[name].json_schema_extra
        assert "rand" not in extra, f"{name} gained a rand draw -- reorders randomize"

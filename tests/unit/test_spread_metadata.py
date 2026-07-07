"""Uniform detail coverage (`detail.spread`): the SPREAD variant selector is
METADATA (pfield ``spread=True``), mirroring the DETAIL_FX ``fx=True`` machinery.
``spread`` is the only selector -- the single knob (coverage level)."""
from __future__ import annotations

import ast
import inspect
import textwrap

from gasgiant.params.model import DetailParams
from gasgiant.render import detail as detail_mod

EXPECTED_SPREAD_LEVERS = {"spread"}


def test_spread_metadata_matches_selector():
    assert set(detail_mod._SPREAD_PARAMS) == EXPECTED_SPREAD_LEVERS


def test_spread_flag_lives_on_exactly_spread():
    flagged = {
        name
        for name, info in DetailParams.model_fields.items()
        if isinstance(info.json_schema_extra, dict)
        and info.json_schema_extra.get("spread")
    }
    assert flagged == EXPECTED_SPREAD_LEVERS


def test_predicate_off_by_default_and_spread_selects():
    assert detail_mod.spread_enabled(DetailParams()) is False
    assert detail_mod.spread_enabled(DetailParams(spread=1e-6)) is True


def test_spread_is_post_tier_no_rand_not_fx():
    extra = DetailParams.model_fields["spread"].json_schema_extra
    assert extra["tier"] == "post"
    assert "rand" not in extra, "a rand draw would reorder randomize"
    assert "fx" not in extra, "spread must not be fx-flagged"


def _spread_on_block() -> ast.If:
    """The ``if spread_on:`` dispatch block inside DetailSynth.synthesize."""
    src = textwrap.dedent(inspect.getsource(detail_mod.DetailSynth.synthesize))
    fn = ast.parse(src).body[0]
    for node in ast.walk(fn):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "spread_on"):
            return node
    raise AssertionError("no `if spread_on:` block in DetailSynth.synthesize")


def test_spread_lever_read_and_uploaded_in_dispatch_block():
    """Cross-ref (mirrors the DETAIL_FX gate): ``spread`` must be read
    (``params.spread``) and uploaded (``"u_spread"``) inside the ``if spread_on:``
    block, or the variant would compile but stay at 0.0."""
    block = _spread_on_block()
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
    assert "spread" in params_reads, "spread not read in spread_on block"
    assert "u_spread" in uniform_names, "u_spread not uploaded in spread_on block"

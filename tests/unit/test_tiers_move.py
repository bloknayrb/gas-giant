"""T8: the tier-walk moved to params.tiers (params layer) with a re-export from
engine.invalidation. Both import paths must resolve to the SAME object so every
existing ``from gasgiant.engine.invalidation import diff_tiers`` keeps working,
and the params-layer copy must be usable without importing the engine layer."""

from __future__ import annotations

from gasgiant.engine.invalidation import diff_tiers as diff_tiers_engine
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.params.tiers import diff_tiers as diff_tiers_params


def test_diff_tiers_is_the_same_object_from_both_paths():
    assert diff_tiers_params is diff_tiers_engine


def test_both_paths_report_the_same_tiers():
    a = PlanetParams(seed=0)
    b = PlanetParams(seed=0)
    b.appearance.contrast = 1.5  # POST
    assert diff_tiers_params(a, b) == {Tier.POST}
    assert diff_tiers_engine(a, b) == {Tier.POST}

    c = PlanetParams(seed=0)
    c.bands.count = 20  # RESTART
    assert diff_tiers_params(a, c) == diff_tiers_engine(a, c) == {Tier.RESTART}


def test_params_tiers_does_not_import_the_engine_layer():
    # The whole point of the move: a params-layer module cannot depend on engine.
    import ast

    import gasgiant.params.tiers as tiers_mod

    with open(tiers_mod.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not any(m.startswith("gasgiant.engine") for m in modules), modules

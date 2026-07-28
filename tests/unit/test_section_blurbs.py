"""Every params section gets a one-line blurb behind the (?) beside its header.

The blurb is the first text an artist reads while hunting for the right panel,
and it is the ONE string that is always visible without hovering a single
slider. A section with no blurb simply draws no help marker -- there is nothing
red, nothing missing on screen, just a section that never explains itself.
``mask`` and ``rings`` were in that state.

``importorskip`` because ``panels`` imports imgui_bundle at module load; the
contents are plain data, but reaching them needs the GUI extra installed.
"""

from __future__ import annotations

import pytest

from gasgiant.params.model import PlanetParams

panels = pytest.importorskip("gasgiant.app.panels")


def _top_level_sections() -> set[str]:
    """The section names ``draw_params_panel`` will actually draw a header for:
    the nested-model fields of ``PlanetParams``. Derived, never listed -- a
    hand-written list is how ``mask`` and ``rings`` went missing in the first
    place."""
    from pydantic import BaseModel

    return {
        name for name, info in PlanetParams.model_fields.items()
        if isinstance(info.annotation, type) and issubclass(info.annotation, BaseModel)
    }


def test_section_blurbs_cover_every_section():
    missing = sorted(_top_level_sections() - set(panels._SECTION_BLURBS))
    assert not missing, f"sections drawn with no help marker: {missing}"


def test_no_blurb_describes_a_section_that_does_not_exist():
    """The other direction: a renamed section leaves a blurb pointing at
    nothing, which reads as coverage while covering nothing."""
    stale = sorted(set(panels._SECTION_BLURBS) - _top_level_sections())
    assert not stale, f"blurbs for non-existent sections: {stale}"


@pytest.mark.parametrize("section", sorted(panels._SECTION_BLURBS))
def test_a_blurb_is_a_sentence_not_a_label(section):
    blurb = panels._SECTION_BLURBS[section]
    assert blurb.strip() == blurb
    assert blurb[0].isupper(), f"{section}: blurbs open with a capital"
    assert blurb.rstrip().endswith("."), f"{section}: blurbs are sentences"
    assert 30 <= len(blurb) <= 400, f"{section}: {len(blurb)} chars"


def test_the_tier_glyph_legend_covers_every_tier():
    """The P/V/R badges beside each slider are meaningless without their
    hover legend, and a new Tier would ship with a badge nobody can decode."""
    from gasgiant.params.model import Tier

    assert {t.value for t in Tier} == set(panels._TIER_GLYPHS)
    for tier, (glyph, _color, legend) in panels._TIER_GLYPHS.items():
        assert len(glyph) == 1, f"{tier}: the badge is a single letter"
        assert legend.strip(), f"{tier}: badge with no legend"

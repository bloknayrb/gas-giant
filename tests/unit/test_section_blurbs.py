"""Every params section gets a one-line blurb behind the (?) beside its header.

The blurb is the first explanation an artist reaches while hunting for the right
panel: one hover on the section header, versus hunting slider by slider. (The
blurb is itself a tooltip -- ``_draw_help_marker`` draws a ``(?)`` and hangs the
text off it, so only the glyph is always on screen.) A section with no blurb
simply draws no marker -- nothing red, nothing visibly missing, just a section
that never explains itself. ``mask`` and ``rings`` were in that state.

``importorskip`` because ``panels`` imports imgui_bundle at module load; the
contents are plain data, but reaching them needs the GUI extra installed.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from gasgiant.params.model import PlanetParams, Tier

panels = pytest.importorskip("gasgiant.app.panels")


def _top_level_sections() -> set[str]:
    """The sections that get a BLURB: the top-level nested models of
    ``PlanetParams``.

    Not the sections that get a header -- ``_draw_model`` draws one for every
    nested ``BaseModel`` at any depth, so ``solver.baroclinic``,
    ``poles.north`` and ``poles.south`` have headers too. It is the
    ``if top_level:`` gate that attaches a blurb, and this predicate mirrors
    that gate, not the header call.

    Derived, never listed -- a hand-written list is how ``mask`` and ``rings``
    went missing in the first place."""
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
    assert {t.value for t in Tier} == set(panels._TIER_GLYPHS)
    for tier, (glyph, _color, legend) in panels._TIER_GLYPHS.items():
        assert len(glyph) == 1, f"{tier}: the badge is a single letter"
        assert legend.strip(), f"{tier}: badge with no legend"

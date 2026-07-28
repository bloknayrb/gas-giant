"""What the copy audit may NOT drop.

The rubric next door (``test_description_rubric.py``) is a shape check: it can
tell that a headline is short and jargon-free without noticing that the rewrite
threw away the one word carrying the design. This module pins the words.

Two separate hazards, hence two halves:

* **Findability.** A description is the search haystack, and search bypasses the
  Basic/Advanced gate -- so for the 141 Advanced-only levers it is very often the
  ONLY way an artist reaches the field at all. Deleting a term to make a
  sentence read better silently un-finds its lever. Rule 11 of the rubric
  explicitly permits demoting engineering provenance to a ``#`` comment, which
  removes text from the haystack; these terms are the ones it may not touch.
* **Load-bearing tokens.** A handful of descriptions carry a qualifier that
  separates the SHIPPED design from one this project's own record marks
  falsified. Losing it turns the tooltip into confident misinformation, which is
  worse than the jargon the audit is removing.

Every assertion lowercases both sides, exactly as ``panels._haystack`` does. An
earlier draft compared against the raw description and would have been red on
day one for four of the six search terms.
"""

from __future__ import annotations

import pytest

from gasgiant.params.model import iter_pfields

_DESCRIPTIONS = {leaf.path: leaf.description for leaf in iter_pfields()}


def _haystack(path: str) -> str:
    """The description as the GUI search actually sees it: lowercased."""
    return _DESCRIPTIONS[path].lower()


#: ``(term, the field it must keep reaching)``. Each is a term a user who knows
#: the physics would type, for a lever whose new artist-facing headline
#: deliberately no longer contains it -- which is precisely why the term has to
#: survive in the tail.
SEARCHABLE_BY_PHYSICS = [
    ("hypofriction", "solver.vort_psi_drag"),
    ("rossby", "solver.deformation_radius"),
    ("rhines", "solver.coriolis_f0"),
    ("solid-body", "storms.hero_solid_core"),
    ("gaussian", "storms.hero_mottle"),
    ("grs", "storms.hero_radius"),
]


@pytest.mark.parametrize(("term", "path"), SEARCHABLE_BY_PHYSICS)
def test_the_physics_term_still_reaches_its_lever(term, path):
    assert term in _haystack(path), (
        f"{path} is no longer findable by {term!r}. Search is the only route to an "
        f"Advanced-only lever; move the term into the trailing parenthetical rather "
        f"than deleting it."
    )


#: ``(path, token, why it is load-bearing)``. Deliberately spelled out one by
#: one: a rewrite that trips one of these needs a human to read the reason, not
#: a count to go down.
LOAD_BEARING = [
    (
        "storms.hero_emergence",
        "partial",
        "the shield is PARTIAL. A full opposite-signed shield rolls up into a "
        "companion cyclone -- the project record marks that design falsified, so "
        "dropping the word documents the thing that does not work.",
    ),
    (
        "storms.hero_emergence",
        "3.6 hero radii",
        "the locality contract, and the only place it is written down. Without it "
        "nothing tells an artist the lever is hero-local rather than global.",
    ),
    (
        "storms.hero_emergence",
        "juno",
        "reference-source provenance. Rule 11 actively invites deleting this as "
        "engineering trivia; this project's calibration discipline says otherwise.",
    ),
    (
        "storms.companion_brightness",
        "0.32",
        "test_hero_frame_helpers pins the DEFAULT at 0.32 and names this "
        "description as where the claim lives -- but asserting the default cannot "
        "notice the claim being deleted from the copy.",
    ),
    (
        "bands.belt_fade",
        "revival outbreaks",
        "test_band_fade's module docstring says the recorded LIMIT is 'mirrored in "
        "the field description'. The fade is VISUAL only: a faded belt keeps its "
        "churn and stays an outbreak host, which is the real phenomenology.",
    ),
]


@pytest.mark.parametrize(("path", "token", "why"), LOAD_BEARING, ids=[
    f"{p.rsplit('.', 1)[-1]}-{t.replace(' ', '_')}" for p, t, _ in LOAD_BEARING
])
def test_a_load_bearing_token_survives_the_rewrite(path, token, why):
    assert token in _haystack(path), f"{path} lost {token!r}: {why}"


def test_hero_emergence_keeps_its_mechanism_numbering():
    """The mode-scoping sentence scopes mechanisms BY NUMBER ("(1)(3) need
    solver.type=vorticity"). So "preserve the mechanism scoping" is satisfiable
    while renumbering the list -- which silently re-scopes two of them to the
    wrong solver mode. Pin the numbering itself, not just the sentence."""
    text = _DESCRIPTIONS["storms.hero_emergence"]
    for n in range(1, 6):
        assert f"({n})" in text, f"mechanism ({n}) lost its number"
    assert "solver.type=vorticity" in text, "the mode-scoping sentence is gone"

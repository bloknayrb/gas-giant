"""What the copy audit may NOT drop.

The rubric next door (``test_description_rubric.py``) is a shape check: it can
tell that a headline is short and jargon-free without noticing that the rewrite
threw away the one word carrying the design. This module pins the words.

Two separate hazards, hence two halves:

* **Findability.** A description is the search haystack, and search bypasses the
  Basic/Advanced gate -- so for the 155 ``adv=True`` levers it is very often the
  ONLY way an artist reaches the field at all. Deleting a term to make a
  sentence read better silently un-finds its lever.
* **Load-bearing tokens.** A handful of descriptions carry a qualifier that
  separates the SHIPPED design from one this project's own record marks
  falsified. Losing it turns the tooltip into confident misinformation, which is
  worse than the jargon the audit is removing.

Note what this module does NOT claim to have caught. The audit's other two
drift catches -- four Turbulence fields losing the word "turbulence", and a
handful of fields losing a term still reachable through their own name -- came
from ``scripts/audit_descriptions.py``, which diffs tokens against a base
revision. This module can only pin what someone thought to list here; the
script sees everything that changed. They are complementary, and the script is
the one that finds the unknown unknowns.

The two halves need DIFFERENT predicates, which an earlier draft got wrong by
using one for both:

* Findability asks "can the user still reach this field?", so it must go
  through the real search -- ``panels._leaf_visible``, the same predicate the
  leaf draw and the section pre-pass gate on. Its haystack is
  ``name + derived_label + field_label + description``, so a term surviving in
  the FIELD NAME still counts. Checking the description alone both overstates
  the requirement and, more importantly, never touches the search path: a
  regression in ``_haystack`` would un-find every lever while the test stayed
  green.
* Load-bearing tokens ask "does the copy still make the claim?", which is
  specifically about the description. A token that survives only because it
  happens to appear in the field name has NOT kept the claim.

So the first half imports panels (hence ``importorskip``) and the second stays
dependency-free, following ``test_field_labels.py``'s local-import pattern so
the GUI-free assertions still run without the GUI extra.
"""

from __future__ import annotations

import re

import pytest

from gasgiant.params.model import iter_pfields

_LEAVES = {leaf.path: leaf for leaf in iter_pfields()}


def _findable_by(path: str, term: str) -> bool:
    """Whether the GUI's search actually reaches ``path`` when the user types
    ``term`` -- the real predicate, not a re-rolled mirror of it."""
    panels = pytest.importorskip("gasgiant.app.panels")
    leaf = _LEAVES[path]
    return panels._leaf_visible(
        leaf.name, leaf.info, {}, panels.PanelState(search=term, show_advanced=False)
    )


def _copy_of(path: str) -> str:
    """The description alone, lowercased -- what the LOAD_BEARING half judges."""
    return _LEAVES[path].description.lower()


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
    assert _findable_by(path, term), (
        f"{path} is no longer findable by {term!r}. Search is the only route to an "
        f"Advanced-only lever; move the term into the trailing parenthetical rather "
        f"than deleting it."
    )


def test_search_really_bypasses_the_advanced_gate():
    """The premise the whole module rests on: a non-matching search must HIDE
    the field. Otherwise the assertions above pass for the wrong reason, and
    keep passing after the term is deleted.

    FIVE of the six are ``adv=True``; ``storms.hero_radius`` is a Basic lever.
    An earlier version of this docstring claimed all six were, and the guard
    filtered on ``.meta.adv`` to match -- which silently excluded hero_radius,
    i.e. the ``"grs"`` case, the single term the audit script singles out as
    "what an artist actually types into search". The entry that most needed the
    premise checked was the one entry the check skipped.

    No filter now: with a non-empty query ``_leaf_visible`` decides purely on
    the haystack match for Basic fields too, so the assertion is meaningful for
    every entry regardless of tier.
    """
    assert any(_LEAVES[p].meta.adv for _t, p in SEARCHABLE_BY_PHYSICS), (
        "sanity: these are meant to be mostly Advanced-only levers"
    )
    for _term, path in SEARCHABLE_BY_PHYSICS:
        assert not _findable_by(path, "zzz-no-such-term"), (
            f"{path}: a non-matching search still shows the field, so the "
            f"findability assertions prove nothing"
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
        "reference-source provenance. A tightening pass reads this as engineering "
        "trivia and deletes it; this project's calibration discipline says the "
        "opposite -- which reference a lever was tuned against is the claim.",
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
    # Word-boundary, not plain `in`. A bare substring test passes when the token
    # is EXTENDED rather than kept: rewriting "0.32" to "0.325" satisfied it
    # (because "0.32" is inside "0.325") while the sibling default pin in
    # test_hero_frame_helpers stayed green too, since the DEFAULT was untouched.
    # Both halves of the advertised coupling reported success over a description
    # that now makes a byte-identity claim about a number that is not the
    # default. Same hole for "3.6 hero radii" -> "3.65 hero radii".
    assert re.search(rf"(?<![\w.]){re.escape(token)}(?![\w.])", _copy_of(path)), (
        f"{path} lost {token!r} (or extended it into a different value): {why}"
    )


def test_the_load_bearing_check_rejects_an_extended_token():
    """Pins the boundary itself. Without it this module's central assertion
    silently accepts the one edit that changes a pinned constant's meaning."""
    pattern = rf"(?<![\w.]){re.escape('0.32')}(?![\w.])"
    assert re.search(pattern, "companion sits at 0.32 (byte-identical)")
    assert not re.search(pattern, "companion sits at 0.325 (byte-identical)")
    assert not re.search(pattern, "companion sits at 10.32 (byte-identical)")


#: ``(path, regex the copy must keep matching)``. A lever that only does
#: something in one solver mode, one projection, or one import path MUST say so:
#: the rubric in ``params/model.py`` calls a dropped activation clause "a lever
#: that silently does nothing", and it is the failure this project's own
#: solver-mode notes warn about by name.
#:
#: Two independent review passes reached this gap from opposite directions --
#: the rubric next door has no shape rule that can express it, and the token
#: diff in ``scripts/audit_descriptions.py`` was suppressing the very words that
#: carry it ("only", "not", "without") as connective tissue. Both are fixed, and
#: this pins the outcome directly. Every entry holds today, so the set costs
#: nothing until someone tightens one of these sentences.
MODE_QUALIFIED = [
    ("storms.hero_solid_core", r"vorticity mode"),
    ("storms.hero_flow_aspect", r"[Vv]orticity mode"),
    ("solver.baroclinic.enabled", r"solver type=vorticity"),
    ("solver.vort_inject", r"[Vv]orticity mode"),
    ("solver.vort_inject_mask", r"[Vv]orticity mode"),
    ("solver.vort_psi_drag", r"[Ee]quirect only"),
    ("solver.vort_eddy_drag", r"[Ee]quirect only"),
]


@pytest.mark.parametrize(("path", "pattern"), MODE_QUALIFIED,
                         ids=[p for p, _ in MODE_QUALIFIED])
def test_a_mode_scoped_lever_still_says_so(path, pattern):
    assert re.search(pattern, _LEAVES[path].description), (
        f"{path} lost its activation clause (expected /{pattern}/). Without it "
        f"the tooltip describes a lever that does nothing in the artist's "
        f"current mode -- move the clause, never delete it."
    )


def test_hero_emergence_keeps_its_mechanism_scoping():
    """The mode-scoping sentence scopes mechanisms BY NUMBER, so "preserve the
    scoping" is satisfiable while renumbering the list -- which silently
    re-scopes two mechanisms to the wrong solver mode.

    The pairing is what has to be pinned, NOT the numbers individually. An
    earlier version asserted only that (1)..(5) each appeared somewhere and
    that "solver.type=vorticity" appeared; mutating the sentence to "(2)(4)"
    left it green while falsely declaring mechanisms (2) and (4)
    vorticity-only and (1) and (3) mode-agnostic, because (1) and (3) still
    occur as list markers further up.
    """
    text = _LEAVES["storms.hero_emergence"].description
    assert "levers (1)(3) need solver.type=vorticity" in text, (
        "the mode-scoping sentence changed. Mechanisms (1) and (3) are the "
        "omega-path ones; re-scoping or renumbering them makes the tooltip "
        "claim the wrong solver mode."
    )
    for n in range(1, 6):
        assert f"({n})" in text, f"mechanism ({n}) lost its number"

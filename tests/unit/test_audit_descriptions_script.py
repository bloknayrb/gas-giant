"""``scripts/audit_descriptions.py``'s comparison logic.

The script is a manual review aid rather than a gate, which is exactly why its
core wanted pinning: it is the tool a reviewer trusts to say "nothing important
was dropped", and a false NEGATIVE there is invisible. It shipped with three
measured defects, all of which this module now covers:

* ``~3.6`` read as a drop against ``3.6``, because ``_TOKEN`` admits ``~`` but
  the strip did not remove it -- a false positive landing on one of the exact
  load-bearing tokens ``test_description_findability`` protects.
* ``kh`` read as a drop from ``kh_wavenumber``, where the search plainly still
  reaches the field through its name.
* A field present at the base and GONE now was skipped entirely, because
  ``main`` iterated the new corpus only. Rename a field and every token it
  carried vanished while ``--fail-on-drop`` still exited 0.

``sys.path`` rather than an import hook: ``scripts/`` is not a package, and the
repo has no other convention for reaching it from tests.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

# A PLAIN import, deliberately. `importorskip` is for genuinely optional deps
# (imgui_bundle, scipy) -- this script is a tracked file in the same checkout,
# so its absence is a real failure, not a reason to stand down. Under
# importorskip, renaming the script silently turned all of these into
# `1 skipped` with exit 0, and CI's `test` job runs without `-rs`, so the only
# trace would be a skip count nobody diffs. This repo has already been burned
# by exactly that (~178 GPU tests skipping while CI reported green).
import audit_descriptions as audit  # noqa: E402  first-party; must not be skippable


def _stub_corpora(monkeypatch, old, new) -> None:
    """Run ``main`` against a hand-built base/live pair.

    Both sides are stubbed together, always: leaving either real would diff a
    two-field fixture against the 226-field model and bury the case under the
    whole corpus. ``_descriptions_at`` takes the revision argument and ignores
    it, so the ``--base`` these tests pass is never resolved by git.
    """
    monkeypatch.setattr(audit, "_descriptions_at", lambda rev: old)
    monkeypatch.setattr(audit, "_current_descriptions", lambda: new)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Storm locality", {"storm", "locality"}),
        # load-bearing forms survive whole -- splitting them would hide their loss
        ("prefer vort_psi_drag", {"prefer", "vort_psi_drag"}),
        ("0.32 = the pre-lever constant", {"0.32", "pre-lever", "constant"}),
        ("solid-body rotation", {"solid-body", "rotation"}),
        # the tilde is decoration on a number, not part of it
        ("~3.6 hero radii", {"3.6", "hero", "radii"}),
        # stopwords carry no signal when dropped
        ("the and of to in", set()),
        # single characters count -- the rubric's mandatory gloss is a bare `0`,
        # and a two-char floor made its removal structurally unreportable
        ("0 = off", {"0", "off"}),
        # a leading sign is part of the number: -0.28 and 0.28 are opposite
        # instructions to an artist ("barges use -0.28" = darken)
        ("barges use -0.28", {"barges", "use", "-0.28"}),
        # ...but a dash used as punctuation is not a token
        ("churn -- the fine kind", {"churn", "fine", "kind"}),
        # negations and scope qualifiers survive tokenization; they invert or
        # unscope a claim, so they are never connective tissue
        ("only in vorticity mode", {"only", "vorticity", "mode"}),
    ],
)
def test_tokens(text, expected):
    assert audit._tokens(text) == expected


def test_a_tilde_prefixed_number_is_not_a_false_drop():
    """``~3.6 hero radii`` -> ``3.6 hero radii`` changes decoration, not the
    claim. Before the strip included ``~`` this reported a drop on the exact
    token pinned as load-bearing for hero_emergence's locality contract."""
    assert audit._lost("nothing beyond ~3.6 hero radii", "nothing beyond 3.6 hero radii") == []


def test_an_inflection_is_not_a_drop():
    """``panels._haystack`` matches by substring, so "festoon" still reaches a
    description containing "festoons". Reporting those trains the reader to
    skim the DROPPED line, which defeats the tool."""
    assert audit._lost("festoon plumes", "festoons and plumes") == []


def test_a_token_still_carried_by_the_field_name_is_not_a_drop():
    """The real haystack is name + captions + description, so "kh" survives in
    ``kh_wavenumber`` even after the description stops saying it."""
    assert audit._lost("KH billow wavenumber", "How many billows fit", "kh_wavenumber") == []
    # ...but with no field name to fall back on, it IS unreachable.
    assert "kh" in audit._lost("KH billow wavenumber", "How many billows fit")


def test_a_token_still_carried_by_the_authored_caption_is_not_a_drop():
    """The haystack carries the authored ``label`` too. Omitting it would
    report "turbulence" lost from ``relax_tau``, captioned "Turbulence leash",
    while GUI search still reaches it -- the same false-positive class as the
    field-name case above, and the docstring promised to cover it."""
    lost = audit._lost(
        "Turbulence relaxation time", "How hard the bands are pulled back",
        "relax_tau", "Turbulence leash",
    )
    assert "turbulence" not in lost, "the authored caption still carries it"
    # "relaxation" and "time" ARE genuinely gone here -- asserting an empty list
    # would pass for the wrong reason and stop testing the caption at all.
    assert set(lost) == {"relaxation", "time"}


def test_section_scoping_also_scopes_the_gone_report(monkeypatch, capsys):
    """``--section`` filtered the changed loop but not the GONE block, so
    scoping to one section still failed on a rename in another -- reporting a
    field the caller explicitly asked not to hear about.

    ``AppearanceParams`` must keep a SURVIVING field here. Scoping a removal
    works off the class's live dotted path, so a fixture that leaves the class
    with no live fields is not "out of scope" at all -- it is the wholly-gone
    case the test below covers, and the two would contradict each other.
    """
    _stub_corpora(
        monkeypatch,
        old={("AppearanceParams", "vanished"): "some old copy"},
        new={
            ("StormsParams", "kept"): (("storms.kept",), "unchanged", "Kept"),
            ("AppearanceParams", "survivor"): (("appearance.survivor",), "still here", "S"),
        },
    )
    code = audit.main(["--base", "HEAD", "--section", "storms.", "--fail-on-drop"])
    out = capsys.readouterr().out
    assert "vanished" not in out, "an out-of-scope removal was reported anyway"
    assert code == 0, "--fail-on-drop fired for a field outside the section"


def test_a_wholly_deleted_class_is_reported_even_under_section(monkeypatch, capsys):
    """The companion to the test above, and the case it did NOT cover.

    ``in_scope_classes`` is built from the classes still present in the NEW
    corpus, so a class deleted or renamed WHOLESALE has no live dotted path,
    matches no section, and was filtered out of the GONE block entirely --
    every token it carried gone in silence while ``--fail-on-drop`` exited 0.
    That is the vacuous pass this block exists to close, reintroduced on the
    scoped path. A class that cannot be placed must be reported, not dropped.
    """
    _stub_corpora(
        monkeypatch,
        old={("DeletedParams", "orphan"): "Rossby locality of the hero"},
        new={("StormsParams", "kept"): (("storms.kept",), "unchanged", "Kept")},
    )
    code = audit.main(["--base", "HEAD", "--section", "storms.", "--fail-on-drop"])
    out = capsys.readouterr().out
    assert "DeletedParams.orphan" in out, "a class that vanished entirely was not reported"
    assert code == 1, "--fail-on-drop must fail when a whole described class vanished"


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # Each of these was measured returning [] before the fix. The first
        # three invert the claim outright; the fourth unscopes it from one
        # solver mode to all of them.
        ("Renders without emission", "Renders with emission", "without"),
        ("No effect in kinematic mode", "Effect in kinematic mode", "no"),
        ("Affects only the hero storm", "Affects the hero storm", "only"),
        ("Applies in vorticity mode only", "Applies in vorticity mode", "only"),
    ],
)
def test_a_dropped_negation_or_scope_qualifier_is_reported(old, new, expected):
    """The rubric this audit added demands mode-specific and conditional
    clauses be carried VERBATIM -- "a dropped activation clause is a lever that
    silently does nothing". The tool meant to catch that was suppressing every
    one of those words as connective tissue.

    ``'The jet does not brighten its own latitude'`` -> ``'The jet brightens
    ...'`` was the worst case: it reported ``DROPPED: does``, which reads as
    textbook filler, and waved through a complete inversion."""
    assert expected in audit._lost(old, new)


def test_a_sign_flip_on_a_number_is_reported():
    """``_tokens`` stripped ``-``, so ``-0.28`` and ``0.28`` were one token and
    flipping an artist instruction from darken to brighten was unreportable.
    The shape rubric cannot see it either -- the shape does not change."""
    assert "-0.28" in audit._lost("barges use -0.28", "barges use 0.28")


def test_an_inflection_still_is_not_a_drop_after_the_boundary_change():
    """Guard against over-correcting: moving from whole-blob substring to
    token-boundary matching must not resurrect the inflection noise that made
    the DROPPED line unreadable. ``per step`` -> ``per-step`` is a hyphenation
    change, not a lost word."""
    assert audit._lost("decorrelates per step", "per-step decorrelation") == ["decorrelates"]


def test_the_ast_walk_agrees_with_the_live_corpus():
    """The one contract binding the two halves, and it had no test at all.

    Every ``main`` test above monkeypatches BOTH ``_descriptions_at`` and
    ``_current_descriptions`` away, so the AST walk and the ``iter_pfields``
    adapter were never executed by the suite -- and they must select the same
    population or the diff is nonsense in either direction:

    * A field the walk cannot see on the OLD side is silently read as "brand
      new" (``old_text is None`` -> ``continue``): no line, no exit code, its
      whole history invisible. f-strings, ``"a" + "b"`` and a positional
      description all land here.
    * A field the walk picks up that ``iter_pfields`` does not (a plain
      ``Field()`` in GradientStop/PaletteRow/BandTemplate gaining a
      ``description=``) can never appear on the live side, so the GONE block
      would fire forever -- a permanently stuck exit 1.

    Comparing at HEAD is the cheap way to pin both directions at once.
    """
    assert set(audit._descriptions_at("HEAD")) == set(audit._current_descriptions())


def test_an_unmatched_section_is_an_error_not_a_clean_bill(capsys):
    """``--section storm.`` (for ``storms.``) printed "no tokens dropped" and
    exited 0 -- a confident all-clear over an audit that never examined a single
    field. A typo must not be indistinguishable from a pass."""
    code = audit.main(["--base", "HEAD", "--section", "storm.", "--fail-on-drop"])
    assert code == 2, "an unmatched --section must not report success"
    assert "matched no field" in capsys.readouterr().err


def test_both_pole_paths_are_scopable():
    """``poles.north`` and ``poles.south`` share one ``PoleParams`` declaration.
    Keying on (class, field) and storing a single path let the last one win, so
    ``--section poles.north.`` -- a correctly spelled, legitimate prefix --
    matched nothing and exited 0. Every path a declaration serves is kept."""
    paths = {p for paths, _d, _c in audit._current_descriptions().values() for p in paths}
    assert any(p.startswith("poles.north.") for p in paths)
    assert any(p.startswith("poles.south.") for p in paths)


def test_a_bad_base_revision_exits_2_not_1(capsys):
    """Exit 1 means "a token was dropped". A base revision git cannot resolve
    used to escape as an uncaught CalledProcessError -- which also exits 1, with
    git's own explanation captured into the exception and never printed. A
    scripted gate could not tell "copy regressed" from "could not run"."""
    code = audit.main(["--base", "no-such-rev-xyz", "--fail-on-drop"])
    assert code == 2, "infrastructure failure must not share the finding code"
    err = capsys.readouterr().err
    assert "no-such-rev-xyz" in err
    assert "invalid object name" in err, "git's own message must reach the user"


def test_a_genuinely_deleted_term_is_reported():
    """The property everything else exists to preserve: a real removal must not
    be filtered away by any of the suppressions above."""
    lost = audit._lost("Rossby deformation radius", "How far the swirl reaches", "deformation_radius")
    assert "rossby" in lost


def test_main_reports_a_field_that_vanished_since_the_base(monkeypatch, capsys):
    """``main`` iterated the NEW corpus only, so a field present at the base and
    absent now was skipped in silence -- a renamed field took every token it
    carried with it while ``--fail-on-drop`` still exited 0. That is the
    vacuous-pass mode of the audit's primary drift mitigation."""
    _stub_corpora(
        monkeypatch,
        old={("StormsParams", "gone_field"): "Rossby locality of the hero"},
        new={("StormsParams", "kept"): (("storms.kept",), "unchanged copy", "Kept")},
    )
    code = audit.main(["--base", "HEAD", "--fail-on-drop"])
    out = capsys.readouterr().out
    assert "gone_field" in out, "a field that disappeared was not reported"
    assert "GONE since the base" in out
    assert code == 1, "--fail-on-drop must fail when a described field vanished"

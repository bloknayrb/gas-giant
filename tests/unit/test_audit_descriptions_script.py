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

audit = pytest.importorskip("audit_descriptions")


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
    monkeypatch.setattr(
        audit, "_descriptions_at",
        lambda rev: {("AppearanceParams", "vanished"): "some old copy"},
    )
    monkeypatch.setattr(
        audit, "_current_descriptions",
        lambda: {
            ("StormsParams", "kept"): ("storms.kept", "unchanged", "Kept"),
            ("AppearanceParams", "survivor"): ("appearance.survivor", "still here", "S"),
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
    monkeypatch.setattr(
        audit, "_descriptions_at",
        lambda rev: {("DeletedParams", "orphan"): "Rossby locality of the hero"},
    )
    monkeypatch.setattr(
        audit, "_current_descriptions",
        lambda: {("StormsParams", "kept"): ("storms.kept", "unchanged", "Kept")},
    )
    code = audit.main(["--base", "HEAD", "--section", "storms.", "--fail-on-drop"])
    out = capsys.readouterr().out
    assert "DeletedParams.orphan" in out, "a class that vanished entirely was not reported"
    assert code == 1, "--fail-on-drop must fail when a whole described class vanished"


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
    monkeypatch.setattr(
        audit, "_descriptions_at",
        lambda rev: {("StormsParams", "gone_field"): "Rossby locality of the hero"},
    )
    monkeypatch.setattr(
        audit, "_current_descriptions",
        lambda: {("StormsParams", "kept"): ("storms.kept", "unchanged copy", "Kept")},
    )
    code = audit.main(["--base", "HEAD", "--fail-on-drop"])
    out = capsys.readouterr().out
    assert "gone_field" in out, "a field that disappeared was not reported"
    assert "GONE since the base" in out
    assert code == 1, "--fail-on-drop must fail when a described field vanished"

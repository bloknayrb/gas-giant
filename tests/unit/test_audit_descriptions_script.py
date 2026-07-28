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
        lambda: {("StormsParams", "kept"): ("storms.kept", "unchanged copy")},
    )
    code = audit.main(["--base", "HEAD", "--fail-on-drop"])
    out = capsys.readouterr().out
    assert "gone_field" in out, "a field that disappeared was not reported"
    assert "GONE since the base" in out
    assert code == 1, "--fail-on-drop must fail when a described field vanished"

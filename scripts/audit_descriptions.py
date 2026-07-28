"""Token-diff the artist-facing descriptions against a base revision.

The copy audit went over all 226 of them and rewrote 107. Its dominant risk
was never a bad sentence -- it is a rewrite that quietly drops a word the
design depends on. That has already happened twice during PLANNING alone: a
draft rewrite of ``storms.hero_emergence`` lost "partial", the one word
separating the shipped shield from a design this project's record marks
falsified, and lost "GRS", which is what an artist actually types into search.

``tests/unit/test_description_rubric.py`` cannot see either loss: it measures
the SHAPE of a headline, and both rewrites were shapely. So this script diffs
the TOKENS instead, and reports every word a wave removed.

It is a review aid, not a gate -- it prints, and only exits non-zero when asked
to. Read the dropped tokens and confirm each one is either connective tissue or
a deliberate, stated removal::

    uv run python scripts/audit_descriptions.py --base master
    uv run python scripts/audit_descriptions.py --base master --section storms.
    uv run python scripts/audit_descriptions.py --base HEAD~1 --fail-on-drop

The base revision is read through ``git show``, and the OLD model is parsed
without importing it -- ``ast`` walks the historical source for
``description=`` literals. Importing two versions of ``gasgiant.params.model``
into one interpreter is not possible, and checking the old revision out to a
temp tree to import it would be a much heavier way to get the same strings.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

_MODEL = "src/gasgiant/params/model.py"

#: Words whose loss is never interesting: articles, prepositions, and the
#: connective tissue a tightening pass exists to remove. Kept deliberately
#: SMALL -- every word not listed here gets reported, because the failure mode
#: is a filter that quietly swallows the one token that mattered.
#: Kept as prose rather than a 60-element list literal, which is far harder to
#: scan for "is X in here?" -- the only question ever asked of it.
_STOPWORD_TEXT = """
a an the and or but of to in on at by for from with without into onto over under
is are was were be been being it its this that these those as if then than so
you your they them their there here when while during after before each every
any all both no not only also more most less least very much many few
"""

_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

#: One-char tokens count: the rubric's mandatory "0 = off" gloss hangs on a
#: bare `0`, and a ``{2,}`` floor made its removal unreportable. Widening was
#: measured free -- the DROPPED report over all 107 rewritten fields is
#: byte-identical either way; only the (advisory) `added:` lines gain a "0".
_TOKEN = re.compile(r"[A-Za-z0-9_.~+-]+")


def _tokens(text: str) -> set[str]:
    """Comparable tokens: lowercased, stopwords dropped, punctuation-only gone.

    Numbers and hyphenated/underscored forms are kept whole (``0.32``,
    ``3.6``, ``solid-body``, ``vort_psi_drag``) -- those are exactly the
    load-bearing tokens, and splitting them would hide their loss.
    """
    return {t for t in (m.group().strip(".-~+").lower() for m in _TOKEN.finditer(text))
            if t and t not in _STOPWORDS}


def _lost(old_text: str, new_text: str, field_name: str = "", caption: str = "") -> list[str]:
    """Tokens the rewrite made UNREACHABLE, judged the way search actually works.

    ``panels._haystack`` is ``name + derived_label + field_label + description``,
    lowercased, matched by plain substring. So a token is only lost if it is
    absent from ALL of that -- not merely absent from the new description's
    token set. Two corrections that came out of real noise on this audit:

    * Set subtraction alone reports every inflection as a casualty. On the pilot
      wave it called 15 tokens lost where the substring rule suppresses 4 of
      them -- "per", "step", "festoon" -> "festoons", "billow" -> "billows".
    * Ignoring the field name reported ``kh`` lost from ``kh_wavenumber``, where
      search plainly still reaches it. The authored ``label`` counts too, and
      omitting it would report "turbulence" lost from ``relax_tau`` (captioned
      "Turbulence leash") for the same bad reason.

    Noise at that rate trains the reader to skim, which is the one thing this
    script must not do.
    """
    haystack = (
        f"{field_name} {field_name.replace('_', ' ')} {caption} {new_text}"
    ).lower()
    return sorted(t for t in _tokens(old_text) - _tokens(new_text) if t not in haystack)


def _descriptions_at(rev: str) -> dict[tuple[str, str], str]:
    """``{(class_name, field_name): description}`` parsed from the model at ``rev``.

    Keyed by the OWNING CLASS, not by field name alone. ``StormsParams`` and
    ``StormOverride`` declare three same-named fields with different copy, so a
    name-only key silently keeps whichever the walk hit last -- and since this
    walk and ``iter_pfields`` visit them in different orders, that reported
    three phantom "changed" descriptions on an unmodified tree. Nor is a dotted
    path an option here: reconstructing one would mean re-implementing the
    nesting walk against historical source. The declaring class is exact, needs
    no reconstruction, and both sides can produce it.
    """
    src = subprocess.run(
        ["git", "show", f"{rev}:{_MODEL}"],
        capture_output=True, text=True, check=True, encoding="utf-8",
        cwd=Path(__file__).resolve().parents[1],  # not the caller's CWD
    ).stdout
    out: dict[tuple[str, str], str] = {}
    for cls in ast.walk(ast.parse(src)):
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            # `name: type = pfield(..., description="...")`
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            for kw in node.value.keywords:
                if (kw.arg == "description" and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)):
                    out[(cls.name, node.target.id)] = kw.value.value
    return out


def _current_descriptions() -> dict[tuple[str, str], tuple[str, str, str]]:
    """``{(class_name, field_name): (dotted_path, description, caption)}`` from the
    live model. The caption comes along because it is part of the real search
    haystack, so ``_lost`` needs it to avoid false positives.

    ``poles.north`` and ``poles.south`` are two ``PoleParams`` instances, so
    five keys here map to two paths each and the later one wins. That is
    correct rather than lossy: both paths read the SAME declaration, so one
    report per declaration is exactly one report per edit.
    """
    from gasgiant.params.model import iter_pfields

    return {(leaf.model.__name__, leaf.name): (leaf.path, leaf.description, leaf.caption)
            for leaf in iter_pfields()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="master", help="revision to diff against (default: master)")
    ap.add_argument("--section", help="restrict to one dotted-path prefix, e.g. 'storms.'")
    ap.add_argument("--fail-on-drop", action="store_true",
                    help="exit 1 if any description dropped a token (for a scripted gate)")
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    if not (repo / _MODEL).exists():  # a clear failure beats an empty diff
        print(f"error: {_MODEL} not found under {repo}", file=sys.stderr)
        return 2

    old = _descriptions_at(args.base)
    new = _current_descriptions()

    dropped_any = False
    changed = 0
    for key, (path, new_text, caption) in sorted(new.items(), key=lambda kv: kv[1][0]):
        if args.section and not path.startswith(args.section):
            continue
        old_text = old.get(key)
        if old_text is None or old_text == new_text:
            continue
        changed += 1
        lost = _lost(old_text, new_text, key[1], caption)
        gained = sorted(_tokens(new_text) - _tokens(old_text))
        delta = len(new_text) - len(old_text)
        print(f"\n{path}  ({len(old_text)} -> {len(new_text)} chars, {delta:+d})")
        if lost:
            dropped_any = True
            print(f"  DROPPED: {', '.join(lost)}")
        if gained:
            print(f"  added:   {', '.join(gained)}")
        if not lost and not gained:
            print("  reordered only")

    # Fields that EXISTED at the base and are gone now. Iterating `new` alone
    # skips them silently, so a renamed field (or a renamed owning class) took
    # every token it carried with it while --fail-on-drop still exited 0 --
    # a vacuous pass in the audit's primary drift mitigation.
    live_classes = {cls for (cls, _f) in new}
    in_scope_classes = {cls for (cls, _f), (path, _d, _c) in new.items()
                        if not args.section or path.startswith(args.section)}
    # A class that still exists is scoped by its live path. One that is GONE
    # ENTIRELY has no path to scope by, so --section cannot place it -- report
    # it either way. Dropping it instead is how the scoped path re-acquired
    # the very vacuous pass this block was added to close.
    removed = sorted(
        k for k in old
        if k not in new
        and (not args.section or k[0] in in_scope_classes or k[0] not in live_classes)
    )
    if removed:
        dropped_any = True
        print("\nGONE since the base (renamed, moved class, or deleted):")
        for cls, field in removed:
            print(f"  {cls}.{field}  -- every token in its description is unreachable")

    scope = f"section {args.section!r}" if args.section else "all fields"
    print(f"\n{changed} description(s) changed vs {args.base} ({scope})")
    if not dropped_any:
        print("no tokens dropped")
        return 0
    print("review every DROPPED token: connective tissue, or a stated removal?")
    return 1 if args.fail_on_drop else 0


if __name__ == "__main__":
    raise SystemExit(main())

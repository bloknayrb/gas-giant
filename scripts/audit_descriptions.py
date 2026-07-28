"""Token-diff a wave of the description copy audit against its base revision.

The audit rewrites 226 artist-facing descriptions in ten waves. Its dominant
risk is not a bad sentence -- it is a rewrite that quietly drops a word the
design depends on. That has already happened twice during PLANNING alone: a
draft rewrite of ``storms.hero_emergence`` lost "partial", the one word
separating the shipped shield from a design this project's record marks
falsified, and lost "GRS", which is what an artist actually types into search.

``tests/unit/test_description_rubric.py`` cannot see either loss: it measures
the SHAPE of a headline, and both rewrites were shapely. So this script diffs
the TOKENS instead, and reports every word a wave removed.

It is a review aid, not a gate -- it prints, and only exits non-zero when asked
to. Run it per wave, read the dropped tokens, and confirm each one is either
connective tissue or a deliberate, stated removal::

    uv run python scripts/audit_descriptions.py --base master
    uv run python scripts/audit_descriptions.py --base master --wave 1
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

_TOKEN = re.compile(r"[A-Za-z0-9_.~+-]{2,}")


def _tokens(text: str) -> set[str]:
    """Comparable tokens: lowercased, stopwords dropped, punctuation-only gone.

    Numbers and hyphenated/underscored forms are kept whole (``0.32``,
    ``3.6``, ``solid-body``, ``vort_psi_drag``) -- those are exactly the
    load-bearing tokens, and splitting them would hide their loss.
    """
    return {t for t in (m.group().strip(".-").lower() for m in _TOKEN.finditer(text))
            if t and t not in _STOPWORDS}


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


def _current_descriptions() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class_name, field_name): (dotted_path, description)}`` from the live model.

    ``poles.north`` and ``poles.south`` are two ``PoleParams`` instances, so
    five keys here map to two paths each and the later one wins. That is
    correct rather than lossy: both paths read the SAME declaration, so one
    report per declaration is exactly one report per edit.
    """
    from gasgiant.params.model import iter_pfields

    return {(leaf.model.__name__, leaf.name): (leaf.path, leaf.description)
            for leaf in iter_pfields()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="master", help="revision to diff against (default: master)")
    ap.add_argument("--wave", type=int, help="restrict to one wave of the audit's table")
    ap.add_argument("--fail-on-drop", action="store_true",
                    help="exit 1 if any wave dropped a token (for a scripted gate)")
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    if not (repo / _MODEL).exists():  # a clear failure beats an empty diff
        print(f"error: {_MODEL} not found under {repo}", file=sys.stderr)
        return 2

    old = _descriptions_at(args.base)
    new = _current_descriptions()

    scope: set[str] | None = None
    if args.wave is not None:
        sys.path.insert(0, str(repo / "tests" / "unit"))
        from test_description_rubric import WAVES  # noqa: PLC0415 -- optional, test-only

        if args.wave not in WAVES:
            print(f"error: no wave {args.wave} (have {sorted(WAVES)})", file=sys.stderr)
            return 2
        scope = WAVES[args.wave]

    dropped_any = False
    changed = 0
    for key, (path, new_text) in sorted(new.items(), key=lambda kv: kv[1][0]):
        if scope is not None and path not in scope:
            continue
        old_text = old.get(key)
        if old_text is None or old_text == new_text:
            continue
        changed += 1
        lost = sorted(_tokens(old_text) - _tokens(new_text))
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

    label = f"wave {args.wave}" if args.wave is not None else "all fields"
    print(f"\n{changed} description(s) changed vs {args.base} ({label})")
    if not dropped_any:
        print("no tokens dropped")
        return 0
    print("review every DROPPED token: connective tissue, or a stated removal?")
    return 1 if args.fail_on_drop else 0


if __name__ == "__main__":
    raise SystemExit(main())

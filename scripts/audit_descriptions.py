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


class _CannotRun(RuntimeError):
    """The audit could not be performed at all -- distinct from "it ran and
    found a drop". Kept separate so ``main`` can reserve exit 1 exclusively for
    findings and exit 2 for "do not read anything into this run"."""


#: Words whose loss is never interesting: articles, prepositions, and the
#: connective tissue a tightening pass exists to remove. Kept deliberately
#: SMALL -- every word not listed here gets reported, because the failure mode
#: is a filter that quietly swallows the one token that mattered.
#: Kept as prose rather than a 60-element list literal, which is far harder to
#: scan for "is X in here?" -- the only question ever asked of it.
#:
#: NEGATIONS AND SCOPE QUALIFIERS ARE DELIBERATELY ABSENT (see _ALWAYS_REPORT).
_STOPWORD_TEXT = """
a an the and or but of to in on at by for from with into onto over under
is are was were be been being it its this that these those as if then than so
you your they them their there here when while during after before
also very much
"""

#: Never suppressed: dropping one of these INVERTS or UNSCOPES the claim rather
#: than tightening it, so "connective tissue" is exactly the wrong reading.
#: Measured before this list existed:
#:     'Applies in vorticity mode only' -> '...mode'      LOST = []
#:     'Renders without emission'       -> 'with...'      LOST = []
#:     'The jet does not brighten...'   -> 'brightens'    LOST = ['does']
#: That last one is worse than silence: the reader is shown a textbook piece of
#: connective tissue and waves through a complete inversion. This codebase is
#: dense with exactly these claims -- hero_solid_core is a kinematic NO-OP, the
#: seat meter is diagnostic ONLY and NEVER moves the storm -- and the rubric in
#: params/model.py demands they be carried VERBATIM. Cost of un-suppressing
#: them, measured over all 107 rewritten fields: one advisory token.
#:
#: Prose rather than a list literal for the same reason as _STOPWORD_TEXT --
#: "is X in here?" is the only question ever asked of it. (A bare .split() on
#: an inline literal also trips ruff SIM905.)
_ALWAYS_REPORT_TEXT = "no not never without only all every each any both most least more less"

_ALWAYS_REPORT = frozenset(_ALWAYS_REPORT_TEXT.split())

_STOPWORDS = frozenset(_STOPWORD_TEXT.split()) - _ALWAYS_REPORT

#: One-char tokens count: the rubric's mandatory "0 = off" gloss hangs on a bare
#: `0`, which a ``{2,}`` floor could never report. Widening alone did NOT
#: deliver that -- the old whole-blob substring haystack still swallowed the `0`
#: whenever any surviving number contained the digit ("10"). It takes the
#: token-boundary matching in ``_lost`` as well; the two go together.
_TOKEN = re.compile(r"[A-Za-z0-9_.~+-]+")


def _tokens(text: str) -> set[str]:
    """Comparable tokens: lowercased, stopwords dropped, punctuation-only gone.

    Numbers and hyphenated/underscored forms are kept whole (``0.32``,
    ``3.6``, ``solid-body``, ``vort_psi_drag``) -- those are exactly the
    load-bearing tokens, and splitting them would hide their loss.

    A LEADING SIGN IS PART OF THE NUMBER. Stripping ``-`` made ``-0.28`` and
    ``0.28`` the same token, so flipping ``storms.hero_brightness``'s "barges
    use -0.28" from darken to brighten was unreportable -- an inverted artist
    instruction, invisible to the shape rubric too because the shape is
    unchanged. A trailing hyphen is still decoration and goes.
    """
    out: set[str] = set()
    for m in _TOKEN.finditer(text):
        t = m.group().strip(".~+").rstrip("-")
        if t.startswith("-") and not t[1:2].isdigit():
            t = t.lstrip("-")          # a dash artefact, not a signed number
        t = t.lower()
        if t and t not in _STOPWORDS:
            out.add(t)
    return out


def _lost(old_text: str, new_text: str, field_name: str = "", caption: str = "") -> list[str]:
    """Tokens the rewrite made UNREACHABLE, judged the way search actually works.

    ``panels._haystack`` is ``name + derived_label + field_label + description``.
    A token survives if it still reaches the artist through ANY of that -- not
    merely if it is still in the new description's token set. Corrections that
    came out of real noise on this audit:

    * Set subtraction alone reports every inflection as a casualty. On the pilot
      wave it called 15 tokens lost where this rule suppresses 6 -- "per",
      "step", "festoon" -> "festoons", "billow" -> "billows", plus "kh" and
      "depth", which survive in the field NAMES kh_wavenumber/hotspot_depth.
    * Ignoring the field name reported ``kh`` lost from ``kh_wavenumber``, where
      search plainly still reaches it. The authored ``label`` counts too, and
      omitting it would report "turbulence" lost from ``relax_tau`` (captioned
      "Turbulence leash") for the same bad reason.

    Matching is at a TOKEN BOUNDARY, not against the joined blob. Substring-on-
    the-blob suppressed anything that happened to appear inside an unrelated
    word -- most sharply the bare ``0``, swallowed by any surviving "10". The
    rule is: a token survives if some surviving token STARTS with it (the
    inflection case) or equals one of its hyphen/underscore parts (``step``
    surviving into ``per-step``). Known limit: ``3.6`` -> ``3.65`` still reads
    as survival, because 3.6 does prefix 3.65. Changed NUMBERS are not what
    this tool can see; dropped words are.

    Noise at that rate trains the reader to skim, which is the one thing this
    script must not do.
    """
    hay = _tokens(f"{field_name} {field_name.replace('_', ' ')} {caption} {new_text}")
    hay |= {part for h in hay for part in re.split(r"[-_]", h) if part}
    return sorted(t for t in _tokens(old_text) - _tokens(new_text)
                  if not any(h.startswith(t) for h in hay))


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

    Gated on ``pfield(`` specifically, to match what the live side selects.
    ``iter_pfields`` counts a leaf only if its ``json_schema_extra`` carries a
    tier, which drops ``GradientStop``/``PaletteRow``/``BandTemplate`` (plain
    ``Field()``). An ungated walk would pick those up the moment one gained a
    ``description=``, and since they can never appear on the live side the GONE
    block would then fire forever -- a permanently stuck red light on the
    audit's primary drift mitigation.
    """
    try:
        src = subprocess.run(
            ["git", "show", f"{rev}:{_MODEL}"],
            capture_output=True, text=True, check=True, encoding="utf-8",
            cwd=Path(__file__).resolve().parents[1],  # not the caller's CWD
        ).stdout
    except FileNotFoundError as exc:                       # git absent
        raise _CannotRun("git is not on PATH; the base revision is read via `git show`") from exc
    except subprocess.CalledProcessError as exc:
        # check=True + capture_output means git's own message lands in
        # exc.stderr and str(exc) drops it -- the one sentence that says what
        # to do. Worse, the traceback exits 1, which is what --fail-on-drop
        # returns for a real finding, so a scripted gate cannot tell "copy
        # regressed" from "could not run".
        raise _CannotRun(
            f"cannot read {_MODEL} at {rev!r}\n"
            f"  git: {(exc.stderr or '').strip()}\n"
            f"  (a shallow PR checkout has no local 'master' -- try "
            f"--base origin/master, or `git fetch origin master:master`)"
        ) from exc
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
            func = node.value.func
            if not (isinstance(func, ast.Name) and func.id == "pfield"):
                continue
            for kw in node.value.keywords:
                if (kw.arg == "description" and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)):
                    out[(cls.name, node.target.id)] = kw.value.value
    return out


def _current_descriptions() -> dict[tuple[str, str], tuple[tuple[str, ...], str, str]]:
    """``{(class_name, field_name): (dotted_paths, description, caption)}`` from
    the live model. The caption comes along because it is part of the real
    search haystack, so ``_lost`` needs it to avoid false positives.

    ``poles.north`` and ``poles.south`` are two ``PoleParams`` instances, so
    five keys here carry TWO paths each. Reporting stays one line per
    declaration -- both paths read the same string, so one report per edit is
    right -- but every path is kept, because ``--section`` matches on them.
    Keeping only the last silently broke ``--section poles.north.``: a
    correctly-spelled prefix that matched nothing, printed a confident
    "no tokens dropped", and exited 0 even under --fail-on-drop.
    """
    from gasgiant.params.model import iter_pfields

    out: dict[tuple[str, str], tuple[tuple[str, ...], str, str]] = {}
    for leaf in iter_pfields():
        key = (leaf.model.__name__, leaf.name)
        seen = out[key][0] if key in out else ()
        out[key] = (seen + (leaf.path,), leaf.description, leaf.caption)
    return out


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

    try:
        old = _descriptions_at(args.base)
    except _CannotRun as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2                       # never 1 -- that means "a token dropped"
    new = _current_descriptions()

    def _in_scope(paths: tuple[str, ...]) -> bool:
        return not args.section or any(p.startswith(args.section) for p in paths)

    # A --section that matches nothing is a typo ('storm.' for 'storms.'), and
    # printing "no tokens dropped / exit 0" for it is a confident all-clear over
    # an audit that never ran.
    if args.section and not any(_in_scope(paths) for paths, _d, _c in new.values()):
        known = sorted({p.split(".")[0] + "." for paths, _d, _c in new.values() for p in paths})
        print(f"error: --section {args.section!r} matched no field; "
              f"known prefixes: {', '.join(known)}", file=sys.stderr)
        return 2

    dropped_any = False
    changed = 0
    for key, (paths, new_text, caption) in sorted(new.items(), key=lambda kv: kv[1][0]):
        if not _in_scope(paths):
            continue
        old_text = old.get(key)
        if old_text is None or old_text == new_text:
            continue
        changed += 1
        lost = _lost(old_text, new_text, key[1], caption)
        gained = sorted(_tokens(new_text) - _tokens(old_text))
        delta = len(new_text) - len(old_text)
        path = " / ".join(paths)
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
    in_scope_classes = {cls for (cls, _f), (paths, _d, _c) in new.items()
                        if _in_scope(paths)}
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

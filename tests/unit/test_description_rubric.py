"""The description copy audit's machinery, pinned before any copy is rewritten.

Nothing in this file rewrites a description. It exists because the audit plan
carried its key numbers in PROSE, and every one of them that got re-measured
turned out to need a correction:

- "226 pfield leaves / 221 unique description strings" is right, but only if the
  walk keys on the DOTTED PATH. A ``(model, name)`` key silently collapses 5
  leaves, because ``poles.north`` and ``poles.south`` are two instances of one
  ``PoleParams`` declaration. That is also the whole reason 226 != 221 -- the
  plan attributed the 5 duplicate strings to ``StormsParams``/``StormOverride``,
  which measurably share only 3 field names, none with equal descriptions.
- The blocklist had to drop three terms that punish compliance with other rules
  (see ``_EXCLUDED_FROM_BLOCKLIST``).
- Rule G could not key on ``default == 0`` alone: 0 is a legitimate VALUE for a
  seed and for a latitude, and ``detail.spread`` documents "0 = band-gated",
  which is more accurate than the flat "0 = off" the plan wanted to mandate.

So the rule here is: a number the audit depends on is either derived from the
model or asserted as a pinned set. It is never left in a docstring.

``KNOWN_VIOLATIONS`` is the baseline of copy that does not yet satisfy the
rubric. Each wave of the audit deletes entries from it. Set EQUALITY is
deliberate -- a plain subset check cannot tell "wave 5 fixed this" from "wave 5
never touched this", and the audit's main risk is a wave silently under-scoping.

Run this module as a script to regenerate the ``KNOWN_VIOLATIONS`` literal.
"""

from __future__ import annotations

import re

import pytest

from gasgiant.params.model import ParamLeaf, StormOverride, StormsParams, iter_pfields

#: Every pfield leaf, keyed by dotted path. ``iter_pfields`` is the params
#: layer's own walk (its docstring carries the two traversal subtleties that
#: silently under-count if re-rolled, and ``tests/unit/test_params.py`` pins
#: them); this module deliberately does not carry a third copy.
CORPUS: list[ParamLeaf] = list(iter_pfields())

CORPUS_LEAVES = 226
CORPUS_UNIQUE_DESCRIPTIONS = 221


def test_corpus_size_is_pinned():
    """The two numbers count different things and must never be conflated: 226
    is dotted paths, 221 is distinct description STRINGS."""
    assert len(CORPUS) == CORPUS_LEAVES
    assert len({leaf.description for leaf in CORPUS}) == CORPUS_UNIQUE_DESCRIPTIONS


def test_dotted_paths_are_unique():
    """The property that makes the path a safe dict key everywhere below.
    ``iter_pfields`` guarantees it by construction; asserted here because this
    module's baseline would silently lose entries if it ever stopped holding."""
    paths = [leaf.path for leaf in CORPUS]
    assert len(paths) == len(set(paths))


def test_the_duplicate_strings_are_exactly_the_pole_pair():
    """Documents WHY 226 != 221, so a future drift in either number has a
    named cause to check rather than a mystery to re-derive.

    ``poles.north`` and ``poles.south`` are two ``PoleParams`` instances, so
    each of its 5 descriptions is reachable at two paths. Editing one edits
    both -- which is correct, and is why the pole fields sit in a single wave.
    """
    by_desc: dict[str, list[str]] = {}
    for leaf in CORPUS:
        by_desc.setdefault(leaf.description, []).append(leaf.path)
    shared = {tuple(sorted(paths)) for paths in by_desc.values() if len(paths) > 1}
    assert shared == {
        ("poles.north.cyclone_count", "poles.south.cyclone_count"),
        ("poles.north.field_density", "poles.south.field_density"),
        ("poles.north.polygon_sides", "poles.south.polygon_sides"),
        ("poles.north.strength", "poles.south.strength"),
        ("poles.north.style", "poles.south.style"),
    }


def test_storms_and_cast_share_no_description():
    """The plan asserted 5 identical strings across ``StormsParams`` and
    ``StormOverride`` and built a cross-wave sequencing constraint on it. They
    share 3 field NAMES and zero descriptions, so no such constraint exists --
    waves 9 and 10 are independent."""
    shared = set(StormsParams.model_fields) & set(StormOverride.model_fields)
    assert shared == {"companion_aspect", "companion_brightness", "rim_contrast"}
    for name in sorted(shared):
        assert (
            StormsParams.model_fields[name].description
            != StormOverride.model_fields[name].description
        ), f"{name}: section leaf and cast row now share a string; waves 9/10 must be one commit"


# -- S1: the headline ----------------------------------------------------------

_ABBREVIATIONS = ("e.g", "i.e", "cf", "vs", "approx", "etc", "deg", "ca")


def split_headline(text: str) -> tuple[str, bool]:
    """``(headline, is_proper_prefix)``.

    S1 ends at the first ``.``/``;``/``:`` that is at paren-depth 0 AND followed
    by whitespace or end-of-string. The trailing-whitespace condition is what
    makes the rule survive the corpus's real punctuation -- ``1.7``, ``57.3``,
    ``0.035``, ``v1.5``, ``16:9`` and ``lon:lat`` all keep their delimiter
    glued to a following non-space, so none of them split. That condition
    replaces (and strictly dominates) an earlier "not between two digits" rule,
    which mis-fired on the ``lon:lat`` idiom and reduced two descriptions to the
    single word "lon".

    Paren-depth matters because rule 8 puts the physics in a trailing
    parenthetical, which is full of sentence punctuation that is not a headline
    boundary. Unbalanced parens degrade to "no split" rather than raising.

    ``is_proper_prefix`` is False when no delimiter exists, i.e. the whole
    description IS the headline. That is a rubric violation (rule A), not an
    error, and it is by far the largest class in the baseline -- see
    ``REMAINING_BY_RULE`` for the count.
    """
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch in ".;:" and depth == 0 and (i + 1 == len(text) or text[i + 1].isspace()):
            head = text[:i]
            if any(head.lower().endswith(a) for a in _ABBREVIATIONS):
                continue  # "e.g. foo" is not a sentence boundary
            return head.strip(), True
    return text.strip(), False


@pytest.mark.parametrize(
    ("text", "expected", "proper"),
    [
        ("Storm locality. Rossby physics after.", "Storm locality", True),
        ("No delimiter at all", "No delimiter at all", False),
        ("Default 0.035 rad is about 2 deg", "Default 0.035 rad is about 2 deg", False),
        ("Aspect 16:9 stays glued", "Aspect 16:9 stays glued", False),
        ("lon:lat elongation of the stamp", "lon:lat elongation of the stamp", False),
        ("Solver v1.5 analytic path", "Solver v1.5 analytic path", False),
        ("Head (a; b; c) tail", "Head (a; b; c) tail", False),
        ("Head (a (b; c) d) tail; real", "Head (a (b; c) d) tail", True),
        ("Head (unbalanced; still fine", "Head (unbalanced; still fine", False),
        ("Trailing stop.", "Trailing stop", True),
        ("Colon head: the rest", "Colon head", True),
        ("Applies to e.g. belts; then this", "Applies to e.g. belts", True),
    ],
)
def test_split_headline(text, expected, proper):
    assert split_headline(text) == (expected, proper)


# -- rule E: the blocklist -----------------------------------------------------

_PARENS = re.compile(r"\([^()]*\)")
_IDENTIFIER = re.compile(r"[a-z]+_[a-z_]+")


def scrub(text: str) -> str:
    """Remove the spans rule E must not judge.

    Parentheticals go because rule 8 REQUIRES the physics to live in one -- a
    matcher that did not strip them would flag the four fields already doing
    exactly what the rubric asks. snake_case identifiers go because rule 7
    requires cross-references to stay literal (``prefer vort_psi_drag``); the
    pattern is deliberately narrow, matching 7 genuine identifiers corpus-wide
    and zero prose. Stripping every ``model_fields`` key instead -- the earlier
    proposal -- would delete ordinary words like ``color`` (36 descriptions) and
    ``latitude`` (28), and would blind the matcher to its own targets.
    """
    prev = None
    while prev != text:  # nested parens, innermost-out
        prev = text
        text = _PARENS.sub(" ", text)
    return _IDENTIFIER.sub(" ", text)


#: Terms an artist should not have to meet in a HEADLINE. They stay legal in the
#: trailing parenthetical -- rule 8 is what keeps the power-user information.
BLOCKLIST = (
    # solver / fluid dynamics
    "vorticity", "vortical", "qgpv", "enstrophy", "hypofriction",
    "hyperviscosity", "hypervisc", "viscosity", "dissipation", "barotropic",
    "geostrophic", "coriolis", "rossby", "rhines", "beta-plane", "poisson",
    "laplacian", "sor", "red-black", "rayleigh", "advection", "advective",
    "advected", "streamfunction", "solenoidal", "eddy", "wavenumber",
    "harmonic", "spectral", "prognostic", "kinematic", "nudging", "e-folding",
    # rendering / engine
    "texel", "lut", "oklab", "srgb", "equirect", "byte-identical",
    "preprocessor", "ssbo", "dispatch",
    # symbols better spelled out in a headline
    "tau", "omega", "psi",
)

#: Excluded from the blocklist, each because another pinned rule requires it.
_EXCLUDED_FROM_BLOCKLIST = {
    # rule 6 mandates "radians of ... (1 rad = 57.3 deg)", and
    # test_control_literacy pins the gloss. Banning it punishes compliance.
    "radians",
    # the name of the panel section the field sits under, and a model field
    # name besides -- an artist reads it as a heading, not as jargon.
    "baroclinic",
    # prose collateral: detail.spread's "Uniform detail coverage" is ordinary
    # English and has nothing to do with a GLSL uniform.
    "uniform",
}

#: Per-field exemptions from rule E, spelled out one by one so every hole is
#: visible. A name-derived allowlist was rejected: it would have handed
#: vort_eddy_drag an exemption for "eddy" -- repealing the rule exactly where it
#: is aimed.
#:
#: Empty, and correctly so. The plan carried one entry,
#: ``turbulence.shear_coupling: {"shear"}``, which was DEAD: "shear" is not on
#: the blocklist, so the exemption granted nothing. It read as evidence the
#: mechanism worked while exempting nothing at all.
PER_FIELD_ALLOW: dict[str, set[str]] = {}


def test_per_field_allowances_name_real_blocklist_terms():
    """An exemption for a term that is not banned is not a narrow exemption --
    it is dead config that looks like a considered decision."""
    for path, terms in PER_FIELD_ALLOW.items():
        assert terms <= set(BLOCKLIST), (
            f"{path} is exempted from {sorted(terms - set(BLOCKLIST))}, which rule E "
            f"never flags; drop the entry or add the term to BLOCKLIST"
        )


def test_blocklist_and_exclusions_are_disjoint():
    assert not set(BLOCKLIST) & _EXCLUDED_FROM_BLOCKLIST


def test_blocklist_is_lowercase_and_deduplicated():
    assert len(set(BLOCKLIST)) == len(BLOCKLIST)
    assert all(t == t.lower() for t in BLOCKLIST)


def blocklist_hits(path: str, headline: str) -> list[str]:
    text = scrub(headline).lower()
    allowed = PER_FIELD_ALLOW.get(path, set())
    return [
        t for t in BLOCKLIST
        if t not in allowed and re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", text)
    ]


def test_scrub_protects_rule_8_and_rule_7():
    assert blocklist_hits("x", "Global brake on swirling (Rayleigh drag)") == []
    assert blocklist_hits("x", "Prefer vort_psi_drag instead") == []
    assert blocklist_hits("x", "Broadband eddy-vorticity injection") == ["vorticity", "eddy"]


def test_word_boundaries_do_not_catch_prose():
    """A bare-substring matcher would flag all of these."""
    for benign in ("Plateau of the curve", "Ellipsis rendering", "Sorted by latitude",
                   "Psychological contrast", "A tautology"):
        assert blocklist_hits("x", benign) == [], benign


# -- rule G: what does zero do? ------------------------------------------------

_ZERO_GLOSS = re.compile(r"\b0(\.0)? *=")

#: Fields whose default is 0 but for which 0 is a VALUE, not a disabled state:
#: two RNG seeds and the two cast coordinates (equator / prime meridian).
ZERO_GLOSS_EXEMPT = {"seed", "storms.hero_shape_seed", "storms.cast.lat_deg", "storms.cast.lon_deg"}


def _is_numeric_zero(default: object) -> bool:
    return isinstance(default, (int, float)) and not isinstance(default, bool) and default == 0


def test_rule_g_exemptions_are_all_really_zero_defaulted():
    """Keeps the exemption set honest: an entry that stopped defaulting to 0 is
    dead weight hiding a real field."""
    zero = {leaf.path for leaf in CORPUS if _is_numeric_zero(leaf.info.default)}
    assert zero >= ZERO_GLOSS_EXEMPT, f"stale exemptions: {ZERO_GLOSS_EXEMPT - zero}"


# -- the rules -----------------------------------------------------------------

RULES = {
    "A": "S1 is a proper prefix (a depth-0 sentence delimiter exists)",
    "B": "S1 opens with a capitalised word, never with '('",
    "C": "S1 is at most 200 characters",
    "D": "S1 is at least 15 characters",
    "E": "S1 carries no blocklist term (parentheticals and identifiers stripped)",
    "F": "the whole description is at most 600 characters, or is listed",
    "G": "a zero-defaulted lever says what 0 does",
}

#: Descriptions permitted past 600 chars. Every one is a showcase lever whose
#: mechanism genuinely needs the room; hero_emergence in particular carries
#: load-bearing tokens (the word "partial", "GRS", "~3.6 hero radii",
#: "Juno/Voyager-anchored", and the (1)-(5) mechanism numbering that a later
#: sentence scopes BY NUMBER) that a shortening pass has already been caught
#: dropping once.
LONG_EXCEPTIONS = {
    "storms.hero_emergence", "solver.deformation_radius", "solver.vort_psi_drag",
    "storms.cast.emergence", "storms.hero_flow_aspect", "storms.hero_shape",
}


def violations(leaf: ParamLeaf) -> list[str]:
    headline, proper = split_headline(leaf.description)
    out = []
    if not proper:
        out.append("A")
    first = headline.split()[0] if headline.split() else ""
    # A leading token containing an uppercase letter is a proper noun (sRGB),
    # not sloppy copy; only an all-lowercase opener or a '(' opener is flagged.
    if headline.startswith("(") or (first and first == first.lower() and first[0].isalpha()):
        out.append("B")
    if len(headline) > 200:
        out.append("C")
    if len(headline) < 15:
        out.append("D")
    if blocklist_hits(leaf.path, headline):
        out.append("E")
    if len(leaf.description) > 600 and leaf.path not in LONG_EXCEPTIONS:
        out.append("F")
    if (
        _is_numeric_zero(leaf.info.default)
        and leaf.path not in ZERO_GLOSS_EXEMPT
        and not _ZERO_GLOSS.search(leaf.description)
    ):
        out.append("G")
    return out


def measure() -> set[tuple[str, str]]:
    return {(leaf.path, rule) for leaf in CORPUS for rule in violations(leaf)}


def test_every_reported_rule_is_documented():
    assert {rule for _, rule in measure()} <= set(RULES)


# -- the baseline --------------------------------------------------------------

KNOWN_VIOLATIONS: set[tuple[str, str]] = {
    ('storms.accent_count', 'D'),
    ('storms.barge_density', 'A'),
    ('storms.cast', 'D'),
    ('storms.cast.aspect', 'B'),
    ('storms.cast.companion_aspect', 'B'),
    ('storms.cast.strength_scale', 'E'),
    ('storms.hero_count', 'A'),
    ('storms.hero_count', 'C'),
    ('storms.hero_radius', 'A'),
    ('storms.hero_shape_seed', 'A'),
    ('storms.hero_strength', 'A'),
    ('storms.hero_strength', 'E'),
    ('storms.hero_taper', 'G'),
    ('storms.merge_debris', 'A'),
    ('storms.outbreak_count', 'A'),
    ('storms.outbreak_count', 'G'),
    ('storms.outbreak_strength', 'A'),
    ('storms.outbreak_strength', 'E'),
    ('storms.oval_density', 'A'),
    ('storms.pearls_count', 'A'),
    ('storms.stamp_contrast', 'A'),
    ('storms.wake_turbulence', 'A'),
}


#: The debt still outstanding, per rule. UPDATE THIS EVERY WAVE -- it is the one
#: line in a wave's diff that states what the wave actually bought, as a number
#: rather than as a 100-element set difference.
#:
#: At Wave 0 it was ``{"A": 82, "B": 2, "C": 1, "D": 5, "E": 21, "G": 22}``
#: over 111 fields. Rule A dominated at 82 of 226: for better than a third of
#: the corpus the description was ONE sentence, so "the headline" was not a
#: prefix of anything -- the single most consequential measurement behind the
#: rubric, since rules D and E would otherwise mean different things on
#: different fields for ten waves.
REMAINING_BY_RULE = {"A": 12, "B": 2, "C": 1, "D": 2, "E": 3, "G": 2}
REMAINING_FIELDS = 18


def test_the_remaining_debt_is_pinned():
    from collections import Counter

    assert dict(Counter(rule for _, rule in measure())) == REMAINING_BY_RULE
    assert len({path for path, _ in measure()}) == REMAINING_FIELDS


def test_known_violations_matches_the_corpus():
    """Set equality, with the two directions reported separately -- a single
    symmetric-difference message reads as an opaque blob the moment one wave
    fixes a field and regresses another on a different rule."""
    actual = measure()
    newly = sorted(actual - KNOWN_VIOLATIONS)
    stale = sorted(KNOWN_VIOLATIONS - actual)
    assert not newly, (
        "newly violating (copy regressed, or a new pfield landed without an audit pass):\n  "
        + "\n  ".join(f"{p}  rule {r}: {RULES[r]}" for p, r in newly)
    )
    assert not stale, (
        "listed but now clean -- delete these from KNOWN_VIOLATIONS in the same "
        "commit that fixed them:\n  " + "\n  ".join(f"{p}  rule {r}" for p, r in stale)
    )


# -- the B2 acceptance criterion ----------------------------------------------

#: Fields whose description content ``tests/unit/test_control_literacy.py``
#: pins. The audit may not silently undo that pass.
B2_PINNED_FIELDS = frozenset({
    "solver.type", "solver.poisson_iters", "solver.sor_omega",
    "solver.vort_relax_tau", "solver.vort_hypervisc", "solver.coriolis_f0",
    "solver.vort_inject_scale", "solver.vort_drag", "solver.deformation_radius",
    "solver.vort_psi_drag", "waves.festoon_wavenumber",
    "solver.baroclinic.warmup_steps", "solver.baroclinic.baro_steps_per_update",
    "solver.baroclinic.update_every",
})


def test_b2_pinned_fields_are_clean_under_rule_e():
    """The rubric's acceptance criterion. The B2 pass already rewrote these to
    lead with the visual read and keep the physics parenthesized, so if rule E
    flags one, the RUBRIC is wrong rather than the copy -- that is what forced
    the three ``_EXCLUDED_FROM_BLOCKLIST`` terms out.

    Note this is specifically rule E. B2 says nothing about headline LENGTH, so
    a B2 field may legitimately appear in KNOWN_VIOLATIONS under another rule.
    """
    flagged = sorted(p for p, r in measure() if r == "E" and p in B2_PINNED_FIELDS)
    assert not flagged, f"rule E contradicts the B2 pass on: {flagged}"


def test_b2_pinned_fields_all_exist():
    paths = {leaf.path for leaf in CORPUS}
    assert paths >= B2_PINNED_FIELDS, f"stale B2 pins: {sorted(B2_PINNED_FIELDS - paths)}"


# -- the wave table ------------------------------------------------------------


def _wave(*prefixes: str, exclude: str = "", exact: tuple[str, ...] = ()) -> set[str]:
    return {
        leaf.path for leaf in CORPUS
        if (leaf.path.startswith(prefixes) and not (exclude and leaf.path.startswith(exclude)))
        or leaf.path in exact
    }


WAVES: dict[int, set[str]] = {
    1: _wave("turbulence.", "waves."),                                    # pilot
    2: _wave("export.", "physical.", "rings.", "sim.", "poles.", exact=("name", "seed")),
    3: _wave("solver."),
    4: _wave("emission.", "mask."),
    5: _wave("bands."),
    6: _wave("jets."),
    7: _wave("appearance."),
    8: _wave("detail."),
    9: _wave("storms.cast."),
    10: _wave("storms.", exclude="storms.cast."),
}


def test_the_wave_table_is_a_partition():
    """Under-scoping is the failure ``KNOWN_VIOLATIONS`` structurally cannot
    catch: a field left out of every wave simply stays listed and green
    forever. Both of the plan's hand-written wave counts were wrong this way --
    they dropped ``bands.template`` and ``storms.cast``, the latter being the
    field that drives the whole cast editor."""
    union: set[str] = set()
    total = 0
    for paths in WAVES.values():
        union |= paths
        total += len(paths)
    assert total == len(union), "waves overlap"
    assert union == {leaf.path for leaf in CORPUS}, (
        f"unassigned: {sorted({leaf.path for leaf in CORPUS} - union)}"
    )


def test_wave_sizes_are_pinned():
    """Sizes are load-bearing: the audit's stated mitigation is small waves, and
    the pilot's sign-off only generalises if later waves stay comparable."""
    assert {k: len(v) for k, v in WAVES.items()} == {
        1: 18, 2: 31, 3: 18, 4: 16, 5: 18, 6: 15, 7: 18, 8: 22, 9: 22, 10: 48
    }


if __name__ == "__main__":  # regenerate the KNOWN_VIOLATIONS literal
    print("KNOWN_VIOLATIONS: set[tuple[str, str]] = {")
    for pair in sorted(measure()):
        print(f"    {pair!r},")
    print("}")

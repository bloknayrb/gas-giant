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

``KNOWN_VIOLATIONS`` was the baseline of copy that did not yet satisfy the
rubric; each wave of the audit deleted entries from it, and it is now EMPTY.
Set EQUALITY is deliberate -- a plain subset check cannot tell "wave 5 fixed
this" from "wave 5 never touched this", and the audit's main risk was a wave
silently under-scoping. Empty, the same assertion becomes the standing gate: a
new pfield whose description misses any rule fails here.

What this file CANNOT check is whether a rewrite kept the words that carry the
design. That is ``test_description_findability.py``, and it is the one that
actually caught things during the audit -- a rubric-clean rewrite dropped
"0.32", another dropped the word "turbulence" from four fields in the
Turbulence section. Shape and substance need separate guards.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from gasgiant.params.model import ParamLeaf, StormOverride, StormsParams, iter_pfields

#: Every pfield leaf, keyed by dotted path. ``iter_pfields`` is the params
#: layer's own walk (its docstring carries the two traversal subtleties that
#: silently under-count if re-rolled, and ``tests/unit/test_params.py`` pins
#: them); this module deliberately does not carry a third copy.
CORPUS: list[ParamLeaf] = list(iter_pfields())

CORPUS_LEAVES = 234
CORPUS_UNIQUE_DESCRIPTIONS = 229


def test_corpus_size_is_pinned():
    """The two numbers count different things and must never be conflated:
    CORPUS_LEAVES counts dotted paths, CORPUS_UNIQUE_DESCRIPTIONS counts
    distinct description STRINGS. Adding one pfield bumps both."""
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
    """A section leaf and its per-storm override are DIFFERENT controls: the
    global sets the population default, the cast row overrides one storm. They
    share three field names, and copy that reads identically for both hides
    that distinction from the artist -- which is the whole reason the cast
    editor is confusing to newcomers.

    Deliberately not pinned to a fixed set of shared names: a fourth shared
    name is a legitimate model change, and it should not break a COPY test.
    """
    shared = set(StormsParams.model_fields) & set(StormOverride.model_fields)
    assert shared, "sanity: the two models still share field names"
    for name in sorted(shared):
        assert (
            StormsParams.model_fields[name].description
            != StormOverride.model_fields[name].description
        ), (
            f"{name}: the global default and the per-storm override now read "
            f"identically, so nothing tells the artist which one they are editing"
        )


# -- S1: the headline ----------------------------------------------------------

#: Words after which a period does NOT end the sentence. Measured against the
#: corpus: only ``deg.`` ever actually occurred, and in both places ("poleward
#: of ~55 deg. Higher = ..." and "at 35-60 deg: granular ...") the punctuation
#: IS the headline boundary -- so listing "deg" here was itself the defect, on
#: top of the suffix match that let it swallow any word ending in those letters.
#: "ca" went with it: nothing uses circa, and it silently captured "silica",
#: "mica", "replica". The rest are kept as genuine sentence-internal forms.
_ABBREVIATIONS = ("e.g", "i.e", "cf", "vs", "approx", "etc")


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

    Paren-depth matters because the rubric puts the physics in a trailing
    parenthetical (model.py rule 4), which is full of sentence punctuation that
    is not a headline boundary. Unbalanced parens degrade to "no split" rather
    than raising.

    ``is_proper_prefix`` is False when no delimiter exists, i.e. the whole
    description IS the headline. That is a rubric violation (rule A), not an
    error -- it was the largest class in the audit's opening baseline by a wide
    margin (see ``KNOWN_VIOLATIONS``), which is why this function returns the
    flag rather than raising on it.
    """
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch in ".;:" and depth == 0 and (i + 1 == len(text) or text[i + 1].isspace()):
            head = text[:i]
            if _ends_with_abbreviation(head):
                continue  # "e.g. foo" is not a sentence boundary
            return head.strip(), True
    return text.strip(), False


def _ends_with_abbreviation(head: str) -> bool:
    """Whether ``head``'s LAST WORD is one of ``_ABBREVIATIONS``.

    A bare ``head.lower().endswith(a)`` matched any word merely ending in those
    letters, and the list contains ``ca`` and ``deg``. Measured consequence on
    the live corpus: ``detail.polar_stipple`` and ``detail.mottle`` both end
    their intended headline in "deg", so the splitter ran past the real
    delimiter and rules C/D/E were applied to a span twice the right length.
    ``silica.``/``mica.``/``replica.`` would do the same, and a single-sentence
    description ending that way would trip rule A spuriously.
    """
    words = head.lower().split()
    return bool(words) and words[-1] in _ABBREVIATIONS


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
        # an ordinary word that merely ENDS in an abbreviation must still split:
        # "ca" and "deg" are in the list, and a suffix test swallowed the real
        # delimiter on two live fields (see _ends_with_abbreviation)
        ("Bright silica. Higher = more", "Bright silica", True),
        ("Speckle poleward of ~55 deg. Higher = more", "Speckle poleward of ~55 deg", True),
        # ...while the genuine abbreviation still does not split
        ("Measured approx. 3 radii across; then this", "Measured approx. 3 radii across", True),
    ],
)
def test_split_headline(text, expected, proper):
    assert split_headline(text) == (expected, proper)


# -- rule E: the blocklist -----------------------------------------------------

_PARENS = re.compile(r"\([^()]*\)")
_IDENTIFIER = re.compile(r"[a-z]+_[a-z_]+")


def scrub(text: str) -> str:
    """Remove the spans rule E must not judge.

    Parentheticals go because the rubric REQUIRES the physics to live in one
    (model.py rule 4) -- a matcher that did not strip them would flag the four
    fields already doing exactly what the rubric asks. snake_case identifiers go
    because cross-references must stay literal so search finds them (rule 4
    again): ``prefer vort_psi_drag`` is not jargon.

    The pattern is deliberately narrow. Stripping every ``model_fields`` key
    instead -- the earlier proposal -- would delete ordinary English words that
    happen to be field names and blind the matcher to its own targets; the
    counts are measured in ``test_scrub_narrowness_is_measured`` rather than
    asserted here, because this module's own rule (see the module docstring) is
    that a number the audit depends on is derived or pinned, never left in
    prose. The three that used to sit in this docstring had all rotted.
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
    # the rubric mandates "radians of ... (1 rad = 57.3 deg)" (model.py rule 4)
    # and test_control_literacy pins the gloss. Banning it punishes compliance.
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


def test_scrub_protects_the_trailing_parenthetical_and_cross_refs():
    assert blocklist_hits("x", "Global brake on swirling (Rayleigh drag)") == []
    assert blocklist_hits("x", "Prefer vort_psi_drag instead") == []
    assert blocklist_hits("x", "Broadband eddy-vorticity injection") == ["vorticity", "eddy"]


def test_word_boundaries_do_not_catch_prose():
    """A bare-substring matcher would flag all of these."""
    for benign in ("Plateau of the curve", "Ellipsis rendering", "Sorted by latitude",
                   "Psychological contrast", "A tautology"):
        assert blocklist_hits("x", benign) == [], benign


def test_scrub_narrowness_is_measured():
    """The claim that ``[a-z]+_[a-z_]+`` strips identifiers and NOT prose, held
    as an assertion rather than as a number in a docstring.

    ``scrub`` removing too much would blind rule E to its own targets; removing
    too little would flag a compliant cross-reference. The bound is what
    matters, not the exact count -- the counts that used to be written into
    ``scrub``'s docstring (13 identifiers, latitude in 31, color in 25) were all
    wrong at both HEAD and master, in the one module whose stated rule forbids
    exactly that.
    """
    hits = [m for leaf in CORPUS
            for m in _IDENTIFIER.findall(split_headline(leaf.description)[0])]
    assert hits, "scrub strips nothing from any headline -- the pattern is dead"
    # Every match must be a real identifier, not prose that happens to fit.
    # Prefix-matching because the pattern carries no digits, so `coriolis_f0`
    # can only ever match as `coriolis_f`.
    names = {leaf.name for leaf in CORPUS}
    #: snake_case names that are NOT pfields but are legitimately cited in copy.
    #: Listed rather than pattern-matched so a new one is a visible decision.
    NON_PFIELD_IDENTIFIERS = {"belt_mask"}
    stray = {h for h in hits
             if h not in NON_PFIELD_IDENTIFIERS
             and not any(n == h or n.startswith(h) for n in names)}
    assert not stray, f"scrub is deleting prose, not identifiers: {sorted(stray)}"
    # and the bare English field names must NOT be strippable by it
    for word in ("latitude", "color", "radius", "strength"):
        assert not _IDENTIFIER.search(word), f"{word} would be stripped as an identifier"


# -- rule G: what does zero do? ------------------------------------------------

#: The literal "0 = ..." gloss. The lookbehind matters: a bare ``\b0`` also
#: matches inside a decimal, so ``"1.0 = round"`` would have satisfied rule G
#: without the field ever saying what ZERO does.
_ZERO_GLOSS = re.compile(r"(?<![0-9.])0(\.0)? *=")

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
    words = headline.split()
    first = words[0] if words else ""
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


# -- the rule engine itself ----------------------------------------------------
#
# Everything above tests the HELPERS; everything below the corpus tests the
# OUTCOME. `violations` sat between them untested, and because the corpus is now
# clean no rule is ever observed firing through it. Measured: neutering C
# (`> 200` -> `> 20000`), D (`< 15` -> `< 0`) and F (`> 600` -> `> 60000`)
# together left the whole file green at 27 passed. Five of the seven rules could
# be deleted outright with CI green, while the module docstring promises "a new
# pfield whose description misses ANY rule fails here". A synthetic table is the
# only way to observe a rule fire once the corpus stops violating it.


def _leaf(description: str, default: object = 1.0) -> ParamLeaf:
    """A ParamLeaf carrying just what `violations` reads: path, description,
    and info.default. Constructed positionally-agnostically so a field added to
    ParamLeaf does not silently break the table."""
    return SimpleNamespace(
        path="synthetic.field", name="field", description=description,
        info=SimpleNamespace(default=default), model=None, caption="Field", meta=None,
    )


_LONG_TAIL = " and then some more prose to push it out past the limit"

RULE_CASES = [
    # (label, description, default, expected rules)
    ("clean baseline",        "Storm size on the map. Higher = bigger; 0 = off.", 1.0, []),
    ("A: no delimiter",       "A headline with no sentence delimiter anywhere",  1.0, ["A"]),
    ("B: lowercase opener",   "lowercase opener here. Then a tail.",             1.0, ["B"]),
    ("B: paren opener",       "(a parenthetical opener). Then a tail.",          1.0, ["B"]),
    ("C: headline over 200",  "S" + "o long" * 45 + ". Tail.",                   1.0, ["C"]),
    ("D: headline under 15",  "Storm. Tail here.",                               1.0, ["D"]),
    ("E: blocklist term",     "Sets the vorticity of the flow. Tail here.",      1.0, ["E"]),
    ("F: body over 600",      "Storm size on the map. Higher = bigger."
                              + _LONG_TAIL * 12,                                 1.0, ["F"]),
    ("G: zero default, no gloss",
                              "Storm size on the map. Higher = bigger.",         0.0, ["G"]),
    ("G satisfied",           "Storm size on the map. Higher = bigger; 0 = off.", 0.0, []),
    # A decimal must not satisfy rule G -- `\b0` alone matches inside "1.0 ="
    ("G: decimal is not a zero gloss",
                              "Storm size on the map. 1.0 = round, higher = wider.", 0.0, ["G"]),
    # rules compound rather than short-circuit
    ("A+B+D together",        "tiny",                                            1.0, ["A", "B", "D"]),
]


@pytest.mark.parametrize(
    ("label", "description", "default", "expected"),
    RULE_CASES, ids=[c[0] for c in RULE_CASES],
)
def test_violations_fires_each_rule(label, description, default, expected):
    assert violations(_leaf(description, default)) == expected


@pytest.mark.parametrize(
    ("length", "rule"),
    [(14, "D"), (15, None), (200, None), (201, "C")],
)
def test_rule_c_and_d_boundaries(length, rule):
    """Off-by-one on a threshold is the classic way a length rule stops meaning
    what its docstring says, and it is invisible against a clean corpus."""
    headline = "S" + "x" * (length - 1)
    assert len(headline) == length
    got = violations(_leaf(f"{headline}. Tail here."))
    assert (rule in got) if rule else (got == [])


def test_rule_f_boundary_and_exception_list():
    body = "Storm size on the map. Higher = bigger."
    at_limit = body + "x" * (600 - len(body))
    assert violations(_leaf(at_limit)) == []
    assert violations(_leaf(at_limit + "x")) == ["F"]
    # LONG_EXCEPTIONS must actually exempt, or the list is decoration
    over = _leaf(at_limit + "x")
    over.path = next(iter(LONG_EXCEPTIONS))
    assert violations(over) == []


def test_every_reported_rule_is_documented():
    assert {rule for _, rule in measure()} <= set(RULES)


# -- the baseline --------------------------------------------------------------

#: EMPTY, and it must STAY empty. The audit cleared the corpus, so this is no
#: longer a shrinking baseline but a plain "every description satisfies every
#: rule" gate. When it fails, fix the COPY -- re-growing this set to make the
#: test green puts the debt back and silences the gate for every field listed.
#:
#: For scale, the audit opened at 133 pairs over 111 of 226 fields:
#: ``{"A": 82, "B": 2, "C": 1, "D": 5, "E": 21, "G": 22}``. Rule A dominated --
#: for better than a third of the corpus the description was ONE sentence, so
#: "the headline" was not a prefix of anything. That was the single most
#: consequential measurement behind the rubric: without it, rules D and E would
#: have meant different things on different fields for ten waves.
KNOWN_VIOLATIONS: set[tuple[str, str]] = set()


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


def test_long_exceptions_are_all_really_long():
    """Keeps the last hand-maintained exception set honest, matching the guards
    ``ZERO_GLOSS_EXEMPT`` and ``B2_PINNED_FIELDS`` already carry. An entry that
    has since been trimmed under 600 is dead weight that would silently excuse
    the field if it grew again."""
    by_path = {leaf.path: leaf for leaf in CORPUS}
    stale = sorted(
        path for path in LONG_EXCEPTIONS
        if path not in by_path or len(by_path[path].description) <= 600
    )
    assert not stale, f"no longer over 600 chars (or gone); drop from LONG_EXCEPTIONS: {stale}"

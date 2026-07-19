# Hero-bracket size-relative geometry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or
> superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Express the `jets.hero_bracket_*` geometry as multiples of the hero core radius
(`hero_radius`) so the bracket tracks storm size automatically.

**Architecture:** Six pfields change from absolute deg/rad to dimensionless × r_core;
`build_profiles` gains a `hero_r_core` (radians) kwarg and multiplies every bracket
geometry quantity by it; the facade threads `hero_radius` at its three `build_profiles`
call sites. CPU/numpy only — no GLSL, byte-identical when the bracket is off.

**Tech Stack:** Python 3.13, pydantic pfields, numpy, pytest.

## Global Constraints

- Byte-identical when the bracket is off (strength 0 skips the whole block; the structural
  guard is unchanged). p05 must stay 9/9 (default-program tripwire — no GLSL here).
- No factory preset bakes a bracket → no migration, no checkpoint break, no preset regen.
- Determinism (seed-independent two-sided shear) preserved: geometry is a pure function of
  hero lat + radius + params, no seed.
- Tier stays RESTART. Param NAMES unchanged (unit/default/bounds/description change only).
- `ruff check .`, `lint-imports`, fast tier (`-m "not gpu and not slow"`), and the
  sliders `--check` drift gate must pass at the end of every task that touches their inputs.

---

### Task 1: Size-relative geometry (params + build_profiles + tests)

**Files:**
- Modify: `src/gasgiant/params/model.py:400-434` (6 pfields)
- Modify: `src/gasgiant/sim/profiles.py:96` (signature) and `:129-145` (override block)
- Test: `tests/unit/test_hero_bracket.py`

**Interfaces:**
- Produces: `build_profiles(seed, bands, bands_params, jets, hero_lat_deg=None,
  hero_r_core=None)` — `hero_r_core` is the hero core radius in RADIANS (= `hero_radius`);
  read only inside the strength-guarded override block. New pfield units: offset/window/
  feather/width all × r_core.

- [ ] **Step 1: Update the three affected tests for the new units.** In
  `test_hero_bracket.py`:

  (a) `test_hero_bracket_defaults_are_off` — new default values (all are ×r_core
  multiples now; the old values were 3.0°/−3.0°/4.0°/5.0°/0.05rad):

```python
def test_hero_bracket_defaults_are_off():
    j = PlanetParams(seed=1).jets
    assert j.hero_bracket_north == 0.0
    assert j.hero_bracket_south == 0.0
    assert j.hero_bracket_north_offset == 1.0     # x hero core radius (jet at storm edge)
    assert j.hero_bracket_south_offset == -1.0
    assert j.hero_bracket_window == 1.0
    assert j.hero_bracket_feather == 1.4
    assert j.hero_bracket_north_width == 0.8
    assert j.hero_bracket_south_width == 0.8
```

  (b) `test_hero_bracket_fields_are_restart_tier` — its sample values (window 5.0,
  feather 6.0, widths 0.06) are now OUT OF BOUNDS (new hi 4.0 / lo 0.1) and would raise
  `ValidationError` on assignment (`validate_assignment=True`). Move them in-range; the
  RESTART-tier assertion itself is unchanged:

```python
    for field, val in (
        ("hero_bracket_north", -1.0), ("hero_bracket_south", 0.6),
        ("hero_bracket_north_offset", 2.0), ("hero_bracket_south_offset", -2.0),
        ("hero_bracket_window", 2.0), ("hero_bracket_feather", 2.0),
        ("hero_bracket_north_width", 0.15), ("hero_bracket_south_width", 0.15),
    ):
```

  (c) `test_bracket_off_is_byte_identical_even_with_pinned_hero` — its off-default
  `hero_bracket_window = 9.0` is now out of bounds (new hi 4.0); lower it to a legal
  off-default AND pass `hero_r_core` with an off-default radius on the OFF call, proving
  radius is irrelevant when the bracket is off:

```python
    variant.jets.hero_bracket_window = 3.5   # off-default, in-range; must not matter
    variant.jets.hero_bracket_feather = 2.0
    off = build_profiles(seed, bands, p.bands, variant.jets,
                         hero_lat_deg=-22.0, hero_r_core=0.11)  # off-default radius, unread
```

- [ ] **Step 2: Run the affected tests — expect FAIL** (defaults still 3.0/-3.0/…;
  `hero_r_core` kwarg does not exist yet).

Run: `uv run pytest tests/unit/test_hero_bracket.py -q`
Expected: FAIL (assert 3.0 == 1.0) and a TypeError on the unknown `hero_r_core` kwarg.

- [ ] **Step 3: Change the 6 pfields** in `model.py` (units, defaults, bounds, description).
  Replace lines 400-434:

```python
    hero_bracket_north_offset: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=4.0, adv=True, ui="Hero Bracket",
        description="Equatorward-flank jet center offset, in units of the hero CORE "
                    "RADIUS (jet center latitude = hero_latitude + this * hero_radius). "
                    "1.0 puts the jet at the storm's edge; scales with hero_radius so the "
                    "bracket keeps straddling the storm. KNOWN LIMITATION: lo=0 assumes a "
                    "SOUTHERN hero (equatorward = +offset); a northern hero would need a "
                    "negative offset (hemisphere-agnostic offsets deferred)",
    )
    hero_bracket_south_offset: float = pfield(
        -1.0, tier=Tier.RESTART, lo=-4.0, hi=0.0, adv=True, ui="Hero Bracket",
        description="Poleward-flank jet center offset, in units of the hero core radius "
                    "(jet center latitude = hero_latitude + this * hero_radius)",
    )
    hero_bracket_window: float = pfield(
        1.0, tier=Tier.RESTART, lo=0.0, hi=4.0, adv=True, ui="Hero Bracket",
        description="Full-override half-width, in units of the hero core radius: seeded "
                    "jets are fully replaced within this many core radii of the hero",
    )
    hero_bracket_feather: float = pfield(
        1.4, tier=Tier.RESTART, lo=0.15, hi=4.0, adv=True, ui="Hero Bracket",
        description="Smoothstep feather beyond the full window, in units of the hero core "
                    "radius; a C1 (zero-derivative) taper so the carved jet adds no "
                    "vorticity spike at the window edge",
    )
    hero_bracket_north_width: float = pfield(
        0.8, tier=Tier.RESTART, lo=0.1, hi=2.0, adv=True, ui="Hero Bracket",
        description="Equatorward-flank jet gaussian half-width, in units of the hero core "
                    "radius",
    )
    hero_bracket_south_width: float = pfield(
        0.8, tier=Tier.RESTART, lo=0.1, hi=2.0, adv=True, ui="Hero Bracket",
        description="Poleward-flank jet gaussian half-width, in units of the hero core "
                    "radius",
    )
```

- [ ] **Step 4: Add `hero_r_core` to `build_profiles` and scale the geometry.** In
  `profiles.py`, change the signature (line ~96) to add `hero_r_core: float = 0.0`, and
  replace the override block INCLUDING the stale unit comment just above it (lines
  **128**-145 — line 128 currently says "…within `window` deg…", which is now wrong) so
  every geometry quantity is × r:

```python
        # C1 window: 1 within `window` CORE RADII of the hero, smoothstep to 0 by
        # window+feather core radii. Zero derivative at both ends -> no du/dphi jump.
        # All bracket geometry (offset/window/feather/width) is a multiple of the hero
        # core radius r = hero_r_core, so the bracket tracks storm size.
        r = hero_r_core
        if r <= 0.0:
            raise ValueError(
                "build_profiles: hero_r_core (radians, = hero_radius) must be > 0 when the "
                "hero bracket is active (hero_lat_deg set + non-zero bracket strength); "
                "the facade always passes it -- a 0 here is a caller bug (forgotten kwarg)"
            )
        full = jets.hero_bracket_window * r
        outer = (jets.hero_bracket_window + jets.hero_bracket_feather) * r
        x = np.clip((np.abs(lat - hero) - full) / max(outer - full, 1e-9), 0.0, 1.0)
        w = 1.0 - (x * x * (3.0 - 2.0 * x))
        pedestal = float(np.interp(hero, lat[::-1], u[::-1]))
        north_c = hero + jets.hero_bracket_north_offset * r
        south_c = hero + jets.hero_bracket_south_offset * r
        bracket = jets.strength * (
            jets.hero_bracket_north
            * np.exp(-(((lat - north_c) / (jets.hero_bracket_north_width * r)) ** 2))
            + jets.hero_bracket_south
            * np.exp(-(((lat - south_c) / (jets.hero_bracket_south_width * r)) ** 2))
        )
        u = u * (1.0 - w) + (pedestal + bracket) * w
```

Note: `hero` above is already `np.deg2rad(hero_lat_deg)` from the existing line just above
the comment (unchanged). The `r <= 0.0` guard is inside the `hero_lat_deg is not None and
strength != 0` block, so it fires only for a caller that activated the bracket while
leaving `hero_r_core` at its 0.0 default — a loud caller bug (the facade always passes the
radius; Task 2), turning a silent divide-by-zero into a clear error. It is a LIVE guard,
not dead: the 0.0 default makes the bad state reachable for a test that forgets the kwarg.

- [ ] **Step 5: Update the strength-ON test callers to pass `hero_r_core`.** In
  `test_hero_bracket.py`, the calls at the two-sided-shear test and the continuity test
  (currently `build_profiles(..., hero_lat_deg=-22.0)`) gain `hero_r_core=<radius>`. Use
  the warm hero radius so the geometry matches the calibrated regime:

```python
    HERO_R = 0.062  # warm hero_radius (radians); the bracket geometry scales by this
    ...
    on = build_profiles(seed, bands, p.bands, v.jets, hero_lat_deg=-22.0, hero_r_core=HERO_R)
```

The continuity test's near-window mask (`np.abs(lat_deg - (-22.0)) < 12.0`) still covers
the feathered window (window 1.0 + feather 1.4 = 2.4 core radii * 3.55 deg ≈ 8.5 deg <
12); keep it. The two-sided-shear test samples rims at ±3 deg — with offsets ±1.0 * 3.55
deg ≈ ±3.55 deg the jets straddle those rims, so `u_eq < u_pol` still holds; keep the
assertion.

- [ ] **Step 6: Add the defining scaling test** to `test_hero_bracket.py`:

```python
def test_bracket_geometry_scales_with_hero_radius():
    """Doubling hero_r_core doubles the effective jet-center offset in latitude:
    the imposed equatorward jet sits at hero + offset*r_core, so its distance
    from the hero doubles with r_core. This is the property the size-relative
    change exists to guarantee. We locate the imposed jet as the argmin of the
    ON-minus-OFF difference (NOT argmin(u)): the difference isolates the bracket's
    contribution, so a strong seed-dependent natural jet elsewhere in the window
    cannot win the argmin (that confound fails ~2.5% of seeds on a raw argmin(u))."""
    HERO = -22.0

    def north_jet_offset_deg(seed, r_core):
        p, bands = _rich(seed)
        off = build_profiles(seed, bands, p.bands, p.jets)  # bracket off (baseline)
        v = PlanetParams(seed=seed)
        v.jets.hero_bracket_north = -1.0
        v.jets.hero_bracket_south = 0.0            # isolate the equatorward jet
        v.jets.hero_bracket_north_offset = 1.0
        on = build_profiles(seed, bands, p.bands, v.jets,
                            hero_lat_deg=HERO, hero_r_core=r_core)
        lat_deg = np.rad2deg(on.lat)
        near = np.abs(lat_deg - HERO) < 20.0
        diff = on.u - off.u                        # the imposed bracket only
        idx = np.argmin(np.where(near, diff, np.inf))   # most-westward imposed change
        return abs(float(lat_deg[idx]) - HERO)

    for seed in (3, 17, 42):                       # robust across seeds, not seed-lucky
        d1 = north_jet_offset_deg(seed, 0.05)
        d2 = north_jet_offset_deg(seed, 0.10)
        assert d2 == pytest.approx(2.0 * d1, rel=0.1), (seed, d1, d2)
```

Add `import pytest` at the top if absent.

- [ ] **Step 6b: Add the guard test** (pins the live `r <= 0.0` guard):

```python
def test_active_bracket_without_radius_raises():
    """Activating the bracket while leaving hero_r_core at its 0.0 default is a
    caller bug (forgotten kwarg); build_profiles raises rather than silently
    dividing by zero. The facade always passes the radius, so this never fires
    in production -- it protects test authors and future callers."""
    seed = 1
    p, bands = _rich(seed)
    v = PlanetParams(seed=seed)
    v.jets.hero_bracket_north = -1.0
    with pytest.raises(ValueError, match="hero_r_core"):
        build_profiles(seed, bands, p.bands, v.jets, hero_lat_deg=-22.0)  # no hero_r_core
```

- [ ] **Step 7: Run the bracket suite — expect PASS.**

Run: `uv run pytest tests/unit/test_hero_bracket.py -q`
Expected: PASS (all, incl. the new scaling test and byte-identity-off with off-default radius).

- [ ] **Step 8: Commit.**

```bash
git add src/gasgiant/params/model.py src/gasgiant/sim/profiles.py tests/unit/test_hero_bracket.py
git commit -m "hero_bracket geometry is size-relative (x hero core radius)"
```

---

### Task 2: Facade threading

**Files:**
- Modify: `src/gasgiant/engine/facade.py` (`_hero_r_core` helper near `_hero_lat_deg` ~52;
  call sites ~99, ~263, ~440)
- Test: `tests/unit/test_facade_seat.py` (or a small new `test_facade_hero_r_core.py`)

**Interfaces:**
- Consumes: `build_profiles(..., hero_r_core=...)` from Task 1.
- Produces: `_hero_r_core(params) -> float` returning `params.storms.hero_radius`
  unconditionally (NOT None-able — the radius always exists; build_profiles reads it only
  inside the strength-guarded block, so passing it at a bracket-off site is harmless).

- [ ] **Step 1: Write the helper test** (pure function, no GL). In a new
  `tests/unit/test_facade_hero_r_core.py`:

```python
from gasgiant.engine.facade import _hero_r_core
from gasgiant.params.model import PlanetParams


def test_hero_r_core_returns_the_hero_radius():
    p = PlanetParams(seed=1)
    p.storms.hero_radius = 0.077
    assert _hero_r_core(p) == 0.077

def test_hero_r_core_is_the_radius_regardless_of_pin():
    """Not None-able: it returns the radius even with no pinned hero, because
    build_profiles ignores it when the override block is skipped."""
    p = PlanetParams(seed=1)          # hero_latitude None by default
    assert _hero_r_core(p) == p.storms.hero_radius
```

- [ ] **Step 2: Run — expect FAIL** (`_hero_r_core` does not exist).

Run: `uv run pytest tests/unit/test_facade_hero_r_core.py -q`
Expected: ImportError / FAIL.

- [ ] **Step 3: Add `_hero_r_core`** beside `_hero_lat_deg` in `facade.py`:

```python
def _hero_r_core(params: PlanetParams) -> float:
    """The hero core radius (radians = hero_radius) threaded into build_profiles
    for the size-relative bracket geometry (r_core = the storm's MINOR-axis
    half-extent; aspect-independent, matching the seat meter's r_core at
    facade.py ~278). Not None-able: build_profiles reads it only inside the
    strength-guarded override block, so passing it at a bracket-off / no-hero
    site is harmless."""
    return params.storms.hero_radius
```

- [ ] **Step 4: Thread it at all three `build_profiles` call sites** (~99, ~263, ~440).
  Each already passes `hero_lat_deg=_hero_lat_deg(...)`; add `hero_r_core=_hero_r_core(...)`
  on the same call — UNCONDITIONALLY, including the bracket-off seat cache at ~263 (which
  passes `hero_lat_deg=None`; the radius is simply unread there). Passing it everywhere
  means no site can forget it and trip the guard. Also add a one-line comment at the seat
  meter's `r_core_deg = rad2deg(p.storms.hero_radius)` (~facade.py:278) noting it is the
  SAME r_core (minor axis, aspect-independent) the bracket geometry now uses, so a future
  editor keeps the two reads in lockstep if the reference axis is ever changed.

- [ ] **Step 5: Run the facade helper test + the existing facade-seat suite — expect PASS.**

Run: `uv run pytest tests/unit/test_facade_hero_r_core.py tests/unit/test_facade_seat.py -q`
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/gasgiant/engine/facade.py tests/unit/test_facade_hero_r_core.py
git commit -m "facade: thread hero core radius into build_profiles for the size-relative bracket"
```

---

### Task 3: Docs + drift gates

**Files:**
- Modify: `docs/sliders.md` (regenerated), `docs/architecture.md`, `CLAUDE.md`
- (No test file; the sliders `--check` gate is the guard.)

- [ ] **Step 1: Regenerate the sliders text** for the six changed descriptions.

Run: `uv run python scripts/render_slider_examples.py --no-render`

- [ ] **Step 2: Update the hero_bracket notes** in `docs/architecture.md` (the export/lever
  or hero-jet-environment section) and `CLAUDE.md`'s hero_bracket bullet to state the
  geometry is in units of the hero core radius (tracks `hero_radius`). State the defaults
  are calibrated to the WARM hero radius (0.062 rad); the bare model default `hero_radius`
  is 0.10, so a default-`PlanetParams` bracket is larger in degrees — harmless (no preset
  bakes a bracket). Do NOT claim the new defaults byte-preserve the old absolute geometry.

- [ ] **Step 3: Run the full fast + lint + drift gates.**

Run: `uv run pytest -m "not gpu and not slow" -q && uv run ruff check . && uv run lint-imports && uv run python scripts/render_slider_examples.py --no-render --check`
Expected: all pass; sliders `--check` reports no drift.

- [ ] **Step 4: Confirm p05 unchanged (default-program tripwire).**

Run: `uv run python scripts/p05_baseline_hash.py --check`
Expected: 9/9 unchanged (no GLSL touched).

- [ ] **Step 5: Commit.**

```bash
git add docs/sliders.md docs/architecture.md CLAUDE.md
git commit -m "docs: hero_bracket geometry is size-relative (x hero core radius)"
```

---

## Cross-task note (inter-task guard gap)

Task 1 makes `hero_r_core` *required* when a bracket is active (the `r <= 0.0` guard), but
the facade isn't threaded until Task 2. Between the two commits, an active-bracket path
through `Simulation` would raise. This is safe: no factory preset or fast-tier test
activates a bracket via the facade, so the fast tier stays green across the gap, and the
branch squash-merges as one unit. Land Task 1 and Task 2 back-to-back; do not ship Task 1
alone.

## Self-review

- **Spec coverage:** units change (T1), threading (T2), byte-identity-off with off-default
  radius (T1 test 1c), scaling property across 3 seeds via ON−OFF diff (T1 Step 6), guard
  (T1 Step 6b), determinism (unchanged — no seed added), docs + p05 tripwire (T3). All spec
  sections mapped.
- **Placeholder scan:** none — every step has concrete code/commands.
- **Type consistency:** `hero_r_core` is `float` (radians, default 0.0) everywhere;
  `_hero_r_core` returns `params.storms.hero_radius` (radians) consistent with
  `build_profiles`'s use (`hero` is radians; all products `offset*r`, `window*r`, `width*r`
  are radians). Offsets/window/feather/widths are dimensionless multipliers throughout.
- **Reviewer findings folded:** test-bounds ValidationError blocker (T1 Step 1b/1c);
  None-ability removed → `float=0.0` + live guard (T1 Step 4, Step 6b; T2 helper);
  seed-fragile scaling test → ON−OFF diff over 3 seeds (T1 Step 6); stale "deg" comment at
  profiles.py:128 (T1 Step 4); default-radius doc correction 0.062→warm, model default 0.10
  (spec + T3 Step 2); r_core cross-reference comment (T2 Step 4); dangling `variant.storms`
  line dropped (T1 Step 1).

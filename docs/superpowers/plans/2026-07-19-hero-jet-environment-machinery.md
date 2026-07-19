# Hero Jet-Environment Machinery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the default-off, byte-identical *machinery* for artist-authored hero jet environments — a `jets.hero_bracket_*` carve-and-impose jet override plus a diagnostic seat meter — without migrating away from `local_jet`, re-baking `gas_giant_warm`, or moving any render hash.

**Architecture:** Two pure functions in `sim/profiles.py` (the carve-and-impose override folded into `build_profiles`, and a `seat_quality`/`seat_scan` metric), exposed to the GUI through `engine/facade.py` (a `seat_quality`/`seat_status` readout, precedent: `baroclinic_status`), rendered by a colored readout in `app/panels.py` (precedent: `_draw_hero_latitude_escape`). Every new lever defaults to a true no-op; the override is skipped by a structural `!= 0.0` guard so default output is byte-identical.

**Tech Stack:** Python 3.13, numpy, pydantic (`pfield` metadata model), imgui-bundle (GUI), pytest. No GLSL/GPU changes — this is a CPU/numpy profile lever.

## Global Constraints

- **Byte-identity holds at defaults only.** `hero_bracket_north == 0.0 and hero_bracket_south == 0.0` (the defaults) MUST skip the entire override block (structural guard, mirroring `if jets.local_jet_speed != 0.0:` at `profiles.py:106`). Proven by an `np.array_equal` capture of every `LatProfiles` field (pattern: `tests/unit/test_local_jet.py:19-40`).
- **NO migration / NO re-bake in this plan.** Do NOT remove `local_jet_*`. Do NOT edit any `presets/*.json`. Do NOT edit `scripts/build_*_preset.py`. Do NOT change the p05 baseline. `gas_giant_warm` keeps `local_jet -0.9 @ -20°`. The warm migration + bracket bake is a DEFERRED user visual-calibration checkpoint, out of scope here.
- **p05 MUST stay 9/9 unchanged** — nothing in this plan moves the default/jupiter_like kinematic render output. But `p05_baseline_hash.py --check` constructs a GPU `Simulation` + `render_maps` (needs GL 4.3 + the machine-local baseline), so it runs ONLY on a GPU box and ONLY where the render path could move: after **Task 2** (the `build_profiles` change — even though the override is guarded off and jupiter_like's pinned `hero_latitude -22.5` still yields a zero, skipped bracket) and the **Task 6** final sweep. Params-only (Task 1) and pure-metric (Task 3) tasks do NOT run p05 (no render path, and it errors off-GPU). Any p05 movement is a real defect to fix, never re-baseline.
- **Tier = RESTART, no `rand`** on every `hero_bracket_*` field (a VELOCITY rebuild must not flip ambient shear under a stale storm rotation; geometry/offset levers must never be seeded-randomized — deliberate omission).
- **Naming discipline:** name rim variables by ROLE (`equatorward_rim`, `poleward_rim`), never by absolute compass sign. For a southern hero at −22°, the equatorward rim is at −19° (LESS negative = `L + DLAT`), the poleward rim at −25° (`L − DLAT`).
- **Layering (import-linter enforced):** metric + override live in `sim/profiles.py`; the GUI reads them ONLY through `engine/facade.py`. `app/` must never `import gasgiant.sim.profiles` directly. Run `uv run lint-imports` after any new import.
- **`build_profiles` signature is append-only backward-compatible:** the new `hero_lat_deg: float | None = None` keyword MUST default to `None` so every existing caller (tests, scripts, the checkpoint path) is byte-identical without edits.
- Ships against merged master (chirality PR #45, `ec11f6b`). Branch: `feat/hero-bracket`.

---

## File Structure

- `src/gasgiant/params/model.py` — add 8 `hero_bracket_*` pfields to `JetsParams` (after the `local_jet_*` block, ~line 371). Responsibility: declarative lever metadata.
- `src/gasgiant/sim/profiles.py` — (a) add `hero_lat_deg` param + the carve-and-impose override block to `build_profiles`; (b) add `seat_quality`, `seat_scan`, `seat_band` pure functions. Responsibility: the profile math + the diagnostic metric.
- `src/gasgiant/engine/facade.py` — thread `hero_lat_deg` into both `build_profiles` call sites (`:84` RESTART, `:367` VELOCITY); add `seat_quality(lat_deg=None)` and `seat_status()` methods. Responsibility: expose sim math to the app across the layer boundary.
- `src/gasgiant/app/panels.py` — a colored seat-meter readout injected beside `hero_latitude` (pattern: `_draw_hero_latitude_escape`, `:320`/`:436`). Responsibility: GUI presentation.
- `tests/unit/test_hero_bracket.py` — CPU no-op, deterministic-across-seeds bracket shear, continuity guard, RESTART tier, params defaults.
- `tests/unit/test_seat_metric.py` — seat metric monotonicity / known-good vs known-bad.
- `tests/unit/test_facade_seat.py` — facade `seat_quality`/`seat_status` behavior (bracket-off profile; no-pinned-hero → None).
- `docs/sliders.md` — regenerated (new pfields).
- `docs/architecture.md`, `CLAUDE.md` — one-line notes on the mode-agnostic override + its deferred bake.

---

## Task 1: `jets.hero_bracket_*` params (default-off)

**Files:**
- Modify: `src/gasgiant/params/model.py` (in `JetsParams`, immediately after the `local_jet_width` pfield that closes the `local_jet_*` block, ~line 382 — the named anchor is authoritative, not the numeral)
- Test: `tests/unit/test_hero_bracket.py`

**Interfaces:**
- Produces: `JetsParams.hero_bracket_north`, `.hero_bracket_south`, `.hero_bracket_north_offset`, `.hero_bracket_south_offset`, `.hero_bracket_window`, `.hero_bracket_feather`, `.hero_bracket_north_width`, `.hero_bracket_south_width` — all `float`, defaults `(0.0, 0.0, +3.0, −3.0, 4.0, 5.0, 0.05, 0.05)`, tier RESTART, no `rand`.

- [ ] **Step 1: Write the failing test** — create `tests/unit/test_hero_bracket.py`:

```python
"""jets.hero_bracket_*: the carve-and-impose hero jet override (build_profiles).
Default-off, structurally guarded no-op; RESTART tier (a VELOCITY rebuild would
flip ambient shear under a stale storm rotation). Machinery only — no preset
bakes ship these; the warm migration is a deferred visual checkpoint."""
from __future__ import annotations

import dataclasses

import numpy as np

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles


def test_hero_bracket_defaults_are_off():
    j = PlanetParams(seed=1).jets
    assert j.hero_bracket_north == 0.0
    assert j.hero_bracket_south == 0.0
    assert j.hero_bracket_north_offset == 3.0
    assert j.hero_bracket_south_offset == -3.0
    assert j.hero_bracket_window == 4.0
    assert j.hero_bracket_feather == 5.0
    assert j.hero_bracket_north_width == 0.05
    assert j.hero_bracket_south_width == 0.05


def test_hero_bracket_fields_are_restart_tier():
    for field, val in (
        ("hero_bracket_north", -1.0), ("hero_bracket_south", 0.6),
        ("hero_bracket_north_offset", 2.0), ("hero_bracket_south_offset", -2.0),
        ("hero_bracket_window", 5.0), ("hero_bracket_feather", 6.0),
        ("hero_bracket_north_width", 0.06), ("hero_bracket_south_width", 0.06),
    ):
        old = PlanetParams(seed=1)
        new = PlanetParams(seed=1)
        setattr(new.jets, field, val)
        assert diff_tiers(old, new) == {Tier.RESTART}, field


def test_hero_bracket_fields_have_no_rand():
    """pfield stores `rand` only when non-None, and pydantic v2 merges
    json_schema_extra into the property top-level (no nested key), so a stray
    rand would appear as a top-level "rand" on the property. Its absence is the
    no-seeded-randomize contract (geometry/offset levers must not be
    randomized)."""
    from gasgiant.params.model import JetsParams
    schema = JetsParams.model_json_schema()["properties"]
    for field in ("hero_bracket_north", "hero_bracket_south",
                  "hero_bracket_north_offset", "hero_bracket_south_offset",
                  "hero_bracket_window", "hero_bracket_feather",
                  "hero_bracket_north_width", "hero_bracket_south_width"):
        assert schema[field].get("rand") is None, field
```

> Verify the `rand` surface against an existing no-rand field (e.g. an
> already-`rand`-less pfield) vs a rand-carrying one (`strength`) by printing
> `JetsParams.model_json_schema()["properties"]["strength"]` once; adjust the
> lookup key if `pfield` nests metadata differently than assumed. The intent is
> invariant: no `rand` metadata on any `hero_bracket_*` field.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_hero_bracket.py -q`
Expected: FAIL — `AttributeError: 'JetsParams' object has no attribute 'hero_bracket_north'`.

- [ ] **Step 3: Add the pfields** — in `src/gasgiant/params/model.py`, after the `local_jet_width` pfield (which closes the `local_jet_*` block, ~line 382, still inside `class JetsParams`), insert:

```python
    hero_bracket_north: float = pfield(
        0.0, tier=Tier.RESTART, lo=-3.0, hi=3.0, adv=True, ui="Hero Bracket",
        description="Carve-and-impose hero jet override: equatorward-flank jet "
                    "strength (negative = westward, the anticyclone-seating sign). "
                    "0 = off, byte-identical. With hero_bracket_south, replaces the "
                    "seeded band jets inside a feathered hero-centered window with an "
                    "authored two-sided bracket; needs a pinned hero. 'north'/'south' "
                    "name the flanks for the SOUTHERN-hemisphere GRS hero (the only "
                    "one that ships): north = equatorward, south = poleward. Machinery "
                    "lever -- not baked into any factory preset yet",
    )
    hero_bracket_south: float = pfield(
        0.0, tier=Tier.RESTART, lo=-3.0, hi=3.0, adv=True, ui="Hero Bracket",
        description="Carve-and-impose hero jet override: poleward-flank jet strength "
                    "(positive = eastward, the anticyclone-seating sign). 0 = off, "
                    "byte-identical",
    )
    hero_bracket_north_offset: float = pfield(
        3.0, tier=Tier.RESTART, lo=0.0, hi=12.0, adv=True, ui="Hero Bracket",
        description="Degrees equatorward of the hero for the equatorward-flank jet "
                    "center (jet center latitude = hero_latitude + this)",
    )
    hero_bracket_south_offset: float = pfield(
        -3.0, tier=Tier.RESTART, lo=-12.0, hi=0.0, adv=True, ui="Hero Bracket",
        description="Degrees poleward of the hero for the poleward-flank jet center "
                    "(jet center latitude = hero_latitude + this)",
    )
    hero_bracket_window: float = pfield(
        4.0, tier=Tier.RESTART, lo=0.0, hi=15.0, adv=True, ui="Hero Bracket",
        description="Full-override half-width (deg): seeded jets are fully replaced "
                    "within this many degrees of the hero",
    )
    hero_bracket_feather: float = pfield(
        5.0, tier=Tier.RESTART, lo=0.5, hi=15.0, adv=True, ui="Hero Bracket",
        description="Smoothstep feather (deg) beyond the full window; a C1 (zero-"
                    "derivative) taper so the carved jet adds no vorticity spike at "
                    "the window edge",
    )
    hero_bracket_north_width: float = pfield(
        0.05, tier=Tier.RESTART, lo=0.01, hi=0.3, adv=True, ui="Hero Bracket",
        description="Equatorward-flank jet gaussian half-width, radians (1 rad = "
                    "57.3 deg)",
    )
    hero_bracket_south_width: float = pfield(
        0.05, tier=Tier.RESTART, lo=0.01, hi=0.3, adv=True, ui="Hero Bracket",
        description="Poleward-flank jet gaussian half-width, radians (1 rad = "
                    "57.3 deg)",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_hero_bracket.py -q`
Expected: PASS (3 tests). If `test_hero_bracket_fields_have_no_rand` fails on schema shape, inspect one existing no-rand field's `model_json_schema()` output and adjust the `extra` lookup to match how `pfield` serializes metadata (the assertion intent: no `rand` key present).

- [ ] **Step 5: Confirm existing presets still load (missing keys default)**

Run: `uv run pytest -m "not gpu and not slow" -q -k "preset or params"`
Expected: PASS — strict models (`extra="forbid"`) accept MISSING keys (they default); only UNKNOWN keys hard-error, and no preset gained a key. (No p05 here — Task 1 touches no render path; p05 is a GPU-box gate deferred to Task 2 per Global Constraints.)

- [ ] **Step 6: Commit**

```bash
git add src/gasgiant/params/model.py tests/unit/test_hero_bracket.py
git commit -m "jets.hero_bracket_*: carve-and-impose override params (RESTART, default-off)"
```

---

## Task 2: carve-and-impose override in `build_profiles`

**Files:**
- Modify: `src/gasgiant/sim/profiles.py` (`build_profiles` signature ~line 77; insert override block after `u *= polar_fade(lat)` at line 111, before the `psi` integration at line 116)
- Test: `tests/unit/test_hero_bracket.py` (extend)

**Interfaces:**
- Consumes: `JetsParams.hero_bracket_*` (Task 1).
- Produces: `build_profiles(seed, bands, bands_params, jets, hero_lat_deg=None) -> LatProfiles` — new trailing keyword `hero_lat_deg`. When `hero_lat_deg is None` OR `north == 0.0 and south == 0.0`, output is byte-identical to the pre-change function.

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/test_hero_bracket.py`:

```python
def _rich(seed, **jet_over):
    p = PlanetParams(seed=seed)
    for k, v in jet_over.items():
        setattr(p.jets, k, v)
    bands = generate_bands(seed, p.bands)
    return p, bands


def test_bracket_off_is_byte_identical_even_with_pinned_hero():
    """north==south==0 skips the whole override, so a pinned hero_lat_deg and
    off-default geometry must not perturb ANY LatProfiles field (structural
    guard, matching the local_jet no-op contract)."""
    seed = 42
    p, bands = _rich(seed)
    base = build_profiles(seed, bands, p.bands, p.jets)  # no hero_lat_deg

    variant = PlanetParams(seed=seed)
    variant.jets.hero_bracket_north = 0.0   # off
    variant.jets.hero_bracket_south = 0.0   # off
    variant.jets.hero_bracket_window = 9.0  # off-default; must not matter
    variant.jets.hero_bracket_feather = 2.0
    off = build_profiles(seed, bands, p.bands, variant.jets, hero_lat_deg=-22.0)

    for field in dataclasses.fields(base):
        a, b = getattr(base, field.name), getattr(off, field.name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), field.name
        else:
            assert a == b, field.name


def test_bracket_noops_without_pinned_hero():
    """A nonzero bracket with hero_lat_deg=None (no pinned hero) is skipped."""
    seed = 7
    p, bands = _rich(seed)
    base = build_profiles(seed, bands, p.bands, p.jets)
    v = PlanetParams(seed=seed)
    v.jets.hero_bracket_north = -1.0
    v.jets.hero_bracket_south = 0.6
    got = build_profiles(seed, bands, p.bands, v.jets, hero_lat_deg=None)
    assert np.array_equal(base.u, got.u)


def test_bracket_seats_two_sided_shear_deterministically_across_seeds():
    """The bracket erases the seeded jets inside the window and imposes a flat
    pedestal + authored gaussians, so the ON profile's TWO-SIDED shear
    u(equatorward_rim) - u(poleward_rim) == bracket(-19) - bracket(-25): the
    seed-dependent pedestal cancels EXACTLY, leaving a seed-independent shear.
    (NOTE: the per-rim on-minus-off INCREMENT is NOT seed-independent -- it
    carries u_base(hero) - u_base(rim), the natural background shear, swing ~0.4.
    Assert the shear, not the increment; and use the pedestal-independent
    ordering equatorward-more-westward-than-poleward for the sign, since the
    absolute u_eq<0 / u_pol>0 ride on the seed-dependent pedestal.)"""
    def u_at(prof, ld):
        return float(np.interp(np.deg2rad(ld), prof.lat[::-1], prof.u[::-1]))
    shears = []
    for seed in (4201, 1234, 555):
        p, bands = _rich(seed)
        v = PlanetParams(seed=seed)
        v.jets.hero_bracket_north = -1.0
        v.jets.hero_bracket_south = 0.6
        on = build_profiles(seed, bands, p.bands, v.jets, hero_lat_deg=-22.0)
        u_eq = u_at(on, -19.0)   # equatorward rim (role, not compass)
        u_pol = u_at(on, -25.0)  # poleward rim
        assert u_eq < u_pol, (
            f"seed {seed}: bracket did not seat anticyclonic shear "
            f"(equatorward {u_eq} not more westward than poleward {u_pol})")
        shears.append(u_eq - u_pol)
    swing = max(shears) - min(shears)
    assert swing < 1e-6, f"bracket two-sided shear not seed-independent: swing {swing}"


def test_bracket_window_has_no_vorticity_spike():
    """The C1 smoothstep window keeps du/dphi continuous, so omega_jet has no
    isolated spike at the feather edge. Assert the largest single-sample jump in
    du/dphi anywhere in the hero neighborhood is within a small multiple of the
    median jump there (a linear feather would produce a delta-function spike)."""
    seed = 99
    p, bands = _rich(seed)
    v = PlanetParams(seed=seed)
    v.jets.hero_bracket_north = -1.0
    v.jets.hero_bracket_south = 0.6
    on = build_profiles(seed, bands, p.bands, v.jets, hero_lat_deg=-22.0)
    lat_deg = np.rad2deg(on.lat)
    near = np.abs(lat_deg - (-22.0)) < 12.0  # window+feather+margin
    du = np.gradient(on.u, on.lat)
    d2 = np.abs(np.diff(du[near]))
    # A C1 smoothstep gives max/median curvature-jump ~4x; a broken C0 LINEAR
    # feather (w' jumps to 0 at the edges -> a du step) gives ~13x. 8x sits
    # between, so this PASSES the smoothstep and FAILS a linear regression.
    # (Reviewer-measured on the 2048-pt grid; re-confirm if the grid changes.)
    ratio = d2.max() / np.median(d2[d2 > 0])
    assert ratio < 8.0, f"vorticity spike at window edge: curvature ratio {ratio}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_hero_bracket.py -q -k "bracket_off or noops_without or seats_two_sided or vorticity_spike"`
Expected: FAIL — `build_profiles() got an unexpected keyword argument 'hero_lat_deg'`.

- [ ] **Step 3: Add the `hero_lat_deg` param + override block** — edit `src/gasgiant/sim/profiles.py`.

3a. Change the signature (line 77-79):

```python
def build_profiles(
    seed: int, bands: BandLayout, bands_params: BandsParams, jets: JetsParams,
    hero_lat_deg: float | None = None,
) -> LatProfiles:
```

3b. Insert the override block AFTER `u *= polar_fade(lat)` (line 111) and BEFORE the `# psi(phi)` comment (line 113):

```python
    # Carve-and-impose hero jet override (jets.hero_bracket_*). Structural guard
    # (mirrors the local_jet != 0.0 skip): default north==south==0 -> the whole
    # block is skipped, byte-identical. Requires a pinned hero (hero_lat_deg).
    # Applied AFTER strength+polar_fade so the pedestal samples the same u that
    # is blended, and BEFORE psi/shear/omega so every derived field sees the
    # carved u. The bracket carries jets.strength (baked into the amplitudes) so
    # a later strength retune rescales it consistently; it is intentionally NOT
    # polar_faded (documented LIMIT for a high-latitude hero).
    if hero_lat_deg is not None and (
        jets.hero_bracket_north != 0.0 or jets.hero_bracket_south != 0.0
    ):
        hero = np.deg2rad(hero_lat_deg)
        # C1 window: 1 within `window` deg of the hero, smoothstep to 0 by
        # window+feather deg. Zero derivative at both ends -> no du/dphi jump.
        full = np.deg2rad(jets.hero_bracket_window)
        zero = np.deg2rad(jets.hero_bracket_window + jets.hero_bracket_feather)
        d = np.abs(lat - hero)
        x = np.clip((d - full) / max(zero - full, 1e-9), 0.0, 1.0)
        w = 1.0 - (x * x * (3.0 - 2.0 * x))            # 1 near hero, 0 outside
        # Flat pedestal = the base u at the hero (keeps the bracket zero-crossing
        # on the hero; a sloped ramp would reintroduce seed-dependent shear).
        pedestal = float(np.interp(hero, lat[::-1], u[::-1]))
        north_c = np.deg2rad(hero_lat_deg + jets.hero_bracket_north_offset)
        south_c = np.deg2rad(hero_lat_deg + jets.hero_bracket_south_offset)
        bracket = jets.strength * (
            jets.hero_bracket_north
            * np.exp(-(((lat - north_c) / jets.hero_bracket_north_width) ** 2))
            + jets.hero_bracket_south
            * np.exp(-(((lat - south_c) / jets.hero_bracket_south_width) ** 2))
        )
        u = u * (1.0 - w) + (pedestal + bracket) * w
```

- [ ] **Step 4: Thread the new keyword is optional — verify existing callers untouched**

Run: `uv run pytest tests/unit/test_local_jet.py -q`
Expected: PASS (the `local_jet` no-op still holds; `hero_lat_deg` defaults `None`).

- [ ] **Step 5: Run the Task-2 tests to verify they pass**

Run: `uv run pytest tests/unit/test_hero_bracket.py -q`
Expected: PASS (7 tests). If `test_bracket_window_has_no_vorticity_spike` is too tight/loose, tune the `40.0` factor against a printed `d2.max()/median` — the intent is "no delta spike," not a precise bound.

- [ ] **Step 6: Byte-identity + lint gates**

Run: `uv run pytest -m "not gpu and not slow" -q` (expect the full suite green, count = prior + new tests)
Run: `uv run python scripts/p05_baseline_hash.py --check` (expect 9/9)
Run: `uv run ruff check .` and `uv run lint-imports` (expect clean)

- [ ] **Step 7: Commit**

```bash
git add src/gasgiant/sim/profiles.py tests/unit/test_hero_bracket.py
git commit -m "profiles: carve-and-impose hero jet override in build_profiles (default-off byte-identical)"
```

---

## Task 3: seat metric pure functions

**Files:**
- Modify: `src/gasgiant/sim/profiles.py` (add `seat_quality`, `seat_scan`, `seat_band` after `build_profiles`, ~line 152)
- Test: `tests/unit/test_seat_metric.py`

**Interfaces:**
- Consumes: `LatProfiles` (its `.lat` descending, `.u`).
- Produces:
  - `seat_quality(profiles: LatProfiles, lat_deg: float, r_core_deg: float, spin_sign: float = 1.0) -> float`
  - `seat_scan(profiles: LatProfiles, lats_deg, r_core_deg: float, spin_sign: float = 1.0) -> list[tuple[float, float]]`
  - `seat_band(quality: float) -> str` returning `"green"` / `"amber"` / `"red"`.

- [ ] **Step 1: Write the failing test** — create `tests/unit/test_seat_metric.py`:

```python
"""seat_quality: a diagnostic proxy for how well the NATURAL jets give a hero a
two-sided anticyclonic bearing at a candidate latitude. Pure function of the
bracket-off profile; higher = better bearing. Used by the GUI seat meter."""
from __future__ import annotations

import numpy as np

from gasgiant.params.model import PlanetParams
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles, seat_band, seat_quality, seat_scan


def _warm_like_profile(seed=4201):
    p = PlanetParams(seed=seed)
    bands = generate_bands(seed, p.bands)
    return build_profiles(seed, bands, p.bands, p.jets)


def test_seat_quality_best_seat_is_not_the_iconic_latitude():
    """Design premise: on warm the natural best bearing is NOT at the iconic
    hero latitude (-22). The scan's argmax must be a DISTINCT latitude, not -22
    itself -- a `max(scan) >= q(-22)` check would be tautological because the
    scan grid contains -22. (This is exactly why the bracket override exists.)"""
    prof = _warm_like_profile()
    q22 = seat_quality(prof, -22.0, 3.0)
    # scan grid deliberately EXCLUDES -22 so the comparison is non-vacuous
    scan = seat_scan(prof, [l for l in np.arange(-14.0, -44.0, -1.0)
                            if abs(l - (-22.0)) > 0.5], 3.0)
    best_lat, best_q = max(scan, key=lambda t: t[1])
    assert best_q > q22, f"no natural seat beats -22 (q22={q22}, best={best_q})"
    assert abs(best_lat - (-22.0)) > 3.0, f"best seat {best_lat} too close to -22"


def test_seat_quality_sign_flips_with_spin():
    """A seat that is good for an anticyclone (spin +1) is bad for a cyclone
    (spin -1) at the same latitude: quality changes sign of its two_sided term."""
    prof = _warm_like_profile()
    lat = -19.0
    qa = seat_quality(prof, lat, 3.0, spin_sign=1.0)
    qc = seat_quality(prof, lat, 3.0, spin_sign=-1.0)
    assert qa != qc


def test_seat_band_thresholds():
    assert seat_band(0.3) == "green"
    assert seat_band(0.05) == "amber"
    assert seat_band(-0.2) == "red"


def test_seat_scan_returns_lat_quality_pairs():
    prof = _warm_like_profile()
    scan = seat_scan(prof, [-20.0, -30.0, -40.0], 3.0)
    assert len(scan) == 3
    assert all(len(t) == 2 for t in scan)
    assert scan[0][0] == -20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_seat_metric.py -q`
Expected: FAIL — `ImportError: cannot import name 'seat_quality'`.

- [ ] **Step 3: Add the metric functions** — in `src/gasgiant/sim/profiles.py`, after `build_profiles` (before `MAX_LANES`, ~line 152):

```python
# Green/amber/red thresholds for the natural-bearing seat meter. Coarse bands
# (the reading is a pre-development proxy; the developed velocity-zero sits
# ~1.8 deg poleward), calibrated so warm's iconic -22 hero reads amber/red
# (natural bearing poor -> enable the bracket) and its best natural seat ~-40
# reads green.
_SEAT_GREEN = 0.15
_SEAT_AMBER = 0.0


def seat_quality(
    profiles: LatProfiles, lat_deg: float, r_core_deg: float, spin_sign: float = 1.0
) -> float:
    """Natural two-sided bearing quality at `lat_deg` for a storm of half-extent
    `r_core_deg`, from the (bracket-off) profile. spin_sign +1 = anticyclone
    (wants westward equatorward rim + eastward poleward rim), -1 = cyclone.
    quality = min(-spin*u_equatorward, spin*u_poleward) - 0.5*|u_center|;
    two_sided is magnitude-based (not sign-only), so a correct-sign-but-weak
    bearing scores low. Reported as a coarse pre-development proxy.

    (Deliberate simplification vs spec 3.1: the explicit r_core-relative
    'moat-orientation weight' is DROPPED. two_sided is already magnitude-based,
    which addresses the spec's core 'not sign-only' concern; r_core_deg still
    sets WHERE the rims are sampled. The coarse green/amber/red band absorbs the
    calibration the weight would have carried. Restore the weight only if
    calibration shows a strong-sign/weak-moat false-green.)"""
    lat_asc = profiles.lat[::-1]
    u_asc = profiles.u[::-1]

    def u_at(ld: float) -> float:
        return float(np.interp(np.deg2rad(ld), lat_asc, u_asc))

    equatorward_rim = u_at(lat_deg + r_core_deg)   # less-negative side for a SH hero
    poleward_rim = u_at(lat_deg - r_core_deg)
    center = abs(u_at(lat_deg))
    two_sided = min(-spin_sign * equatorward_rim, spin_sign * poleward_rim)
    return two_sided - 0.5 * center


def seat_scan(
    profiles: LatProfiles, lats_deg, r_core_deg: float, spin_sign: float = 1.0
) -> list[tuple[float, float]]:
    """(lat_deg, quality) over a latitude sweep — the GUI's 'find a good seat'
    readout. Diagnostic only: never moves the storm."""
    return [
        (float(ld), seat_quality(profiles, float(ld), r_core_deg, spin_sign))
        for ld in lats_deg
    ]


def seat_band(quality: float) -> str:
    """Coarse green/amber/red classification of a seat_quality value."""
    if quality >= _SEAT_GREEN:
        return "green"
    if quality >= _SEAT_AMBER:
        return "amber"
    return "red"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_seat_metric.py -q`
Expected: PASS (4 tests). If `test_seat_quality_rewards_a_two_sided_bracket`'s `best >= q22` is flaky on a seed, switch `_warm_like_profile` to a fixed seed known to expose the −22-poor / −40-good pattern (4201), and assert on the scan shape rather than exact latitude.

- [ ] **Step 5: Lint + commit**

Run: `uv run ruff check src/gasgiant/sim/profiles.py tests/unit/test_seat_metric.py`
```bash
git add src/gasgiant/sim/profiles.py tests/unit/test_seat_metric.py
git commit -m "profiles: seat_quality/seat_scan/seat_band diagnostic metric (bracket-off bearing)"
```

---

## Task 4: facade wiring — thread `hero_lat_deg` + expose the meter

**Files:**
- Modify: `src/gasgiant/engine/facade.py` (both `build_profiles` call sites: `:84`, `:367`; import `seat_quality`/`seat_band`; add `seat_quality`/`seat_status` methods near `baroclinic_status`, ~line 221)
- Test: `tests/unit/test_facade_seat.py`

**Interfaces:**
- Consumes: `build_profiles(..., hero_lat_deg=)` (Task 2), `seat_quality`/`seat_band` (Task 3).
- Produces:
  - `Simulation.seat_quality(lat_deg: float | None = None) -> float | None` (None when no pinned hero)
  - `Simulation.seat_status() -> str | None` (a human string like `"seat: amber (poor bearing — enable hero_bracket)"`, None when no pinned hero)

- [ ] **Step 1: Write the failing test** — create `tests/unit/test_facade_seat.py`:

```python
"""Facade seat-meter exposure: the GUI reads the natural-bearing quality through
Simulation, computed on the BRACKET-OFF profile (so it reports the natural
bearing even when the override is on).

WHITE-BOX construction (deliberate): Simulation.__init__ has NO CPU-only path --
it calls GpuContext.headless() (needs GL 4.3) and _build() (allocates GPU LUTs),
and there is zero Simulation() usage in tests/unit. But seat_quality/seat_status
read ONLY self.params and self.bands, both CPU-constructible. So we bypass
__init__ with object.__new__ and inject exactly those two attributes, keeping
these behavioral tests in the no-GPU tier. A future edit that makes the seat
methods touch self.solver/self.gpu will surface here as an AttributeError --
that is the intended tripwire."""
from __future__ import annotations

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.sim.bands import generate_bands


def _seat_sim(params: PlanetParams) -> Simulation:
    sim = object.__new__(Simulation)          # bypass GPU __init__
    sim.params = params
    sim.bands = generate_bands(params.seed, params.bands)
    sim._seat_profile = None                  # the lazy bracket-off cache slot
    return sim


def test_seat_quality_none_without_pinned_hero():
    p = PlanetParams(seed=4201)
    p.storms.hero_latitude = None
    sim = _seat_sim(p)
    assert sim.seat_quality() is None
    assert sim.seat_status() is None


def test_seat_quality_uses_bracket_off_profile():
    """With the bracket ON, seat_quality still reports the NATURAL bearing (it
    builds the profile with hero_lat_deg=None, which skips the override), so
    turning the bracket on does not change the meter reading."""
    p = PlanetParams(seed=4201)
    p.storms.hero_latitude = -22.0
    off = _seat_sim(p).seat_quality()
    p2 = PlanetParams(seed=4201)
    p2.storms.hero_latitude = -22.0
    p2.jets.hero_bracket_north = -1.0
    p2.jets.hero_bracket_south = 0.6
    on = _seat_sim(p2).seat_quality()
    assert on == off


def test_seat_quality_reads_draft_latitude_override():
    """The GUI passes the live (draft) hero latitude; seat_quality(lat_deg=...)
    samples there without rebuilding, so a scan is cheap."""
    p = PlanetParams(seed=4201)
    p.storms.hero_latitude = -22.0
    sim = _seat_sim(p)
    assert sim.seat_quality(-40.0) != sim.seat_quality(-22.0)


def test_seat_status_is_a_banded_string():
    p = PlanetParams(seed=4201)
    p.storms.hero_latitude = -22.0
    s = _seat_sim(p).seat_status()
    assert any(band in s for band in ("green", "amber", "red"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_facade_seat.py -q`
Expected: FAIL — `AttributeError: 'Simulation' object has no attribute 'seat_quality'` (or a construction error resolved per the note).

- [ ] **Step 3: Thread `hero_lat_deg` at both call sites** — in `src/gasgiant/engine/facade.py`:

3a. Line 84 (`_build`, RESTART):
```python
        self.profiles = build_profiles(
            p.seed, self.bands, p.bands, p.jets,
            hero_lat_deg=(p.storms.hero_latitude if p.storms.hero_count > 0 else None),
        )
        self._seat_profile = None  # invalidate the bracket-off meter cache
```
3b. Line 367 (`update_params`, VELOCITY):
```python
            self.profiles = build_profiles(
                new_params.seed, self.bands, new_params.bands, new_params.jets,
                hero_lat_deg=(new_params.storms.hero_latitude
                              if new_params.storms.hero_count > 0 else None),
            )
            self._seat_profile = None  # invalidate the bracket-off meter cache
```

(The `_seat_profile` attribute is created by these assignments; `_bracket_off_profile`
reads it via `getattr(self, "_seat_profile", None)` so a not-yet-built instance is safe.)

- [ ] **Step 4: Add the import + methods** — extend the `from gasgiant.sim.profiles import (...)` block (line 28) with `seat_band,` and `seat_quality,` (keep alphabetical if the block is sorted; note `seat_quality` will shadow nothing — the facade method has the same name but is an attribute). To avoid the name clash, import them aliased:

```python
from gasgiant.sim.profiles import (
    build_profiles,
    seat_band as _seat_band,
    seat_quality as _seat_quality,
    select_hero_festoon_latitude,
    select_lanes,
    select_wave_latitudes,
)
```

Add methods after `baroclinic_degraded_reason` (~line 232). `np` is already imported (facade.py:19):

```python
    def _bracket_off_profile(self):
        """The natural (bracket-off) profile for the seat meter, cached. Built
        with hero_lat_deg=None so the override is SKIPPED regardless of the
        bracket params -- no model_copy/zeroing needed. Cache is invalidated
        wherever self.profiles is rebuilt (see _build / the VELOCITY branch),
        so a hero_latitude drag (RESTART -- not applied mid-drag) reuses it and
        the per-frame meter does no profile work."""
        if getattr(self, "_seat_profile", None) is None:
            p = self.params
            self._seat_profile = build_profiles(
                p.seed, self.bands, p.bands, p.jets, hero_lat_deg=None
            )
        return self._seat_profile

    def seat_quality(self, lat_deg: float | None = None) -> float | None:
        """Natural two-sided bearing quality at the hero latitude (or `lat_deg`
        -- the GUI passes the live draft latitude), on the BRACKET-OFF profile
        so the reading is the natural bearing even when the override is engaged.
        None when no hero is pinned."""
        p = self.params
        if lat_deg is None:
            lat_deg = p.storms.hero_latitude
        if lat_deg is None or p.storms.hero_count <= 0:
            return None
        r_core_deg = float(np.rad2deg(p.storms.hero_radius))
        # Anticyclone by default (the GRS case); a cyclonic hero would flip this
        # (deferred -- no cyclonic-hero preset ships).
        return _seat_quality(self._bracket_off_profile(), float(lat_deg),
                             r_core_deg, spin_sign=1.0)

    def seat_status(self, lat_deg: float | None = None) -> str | None:
        """One-line banded readout for the GUI seat meter (None if no pinned
        hero). Pre-development proxy -- the developed bearing sits ~1.8 deg
        poleward of this profile-level reading."""
        q = self.seat_quality(lat_deg)
        if q is None:
            return None
        band = _seat_band(q)
        hint = {
            "green": "natural bearing OK here",
            "amber": "natural bearing weak -- consider hero_bracket",
            "red": "natural bearing poor -- enable hero_bracket",
        }[band]
        return f"seat: {band} ({hint})"
```

Invalidate the cache wherever `self.profiles` is rebuilt so a committed edit refreshes it: after the `build_profiles` assignment at BOTH `_build` (line ~84) and the VELOCITY branch (line ~367), add `self._seat_profile = None`. (During a `hero_latitude` drag nothing is applied — RESTART commits on release — so the cache legitimately persists and the meter re-samples the draft latitude cheaply.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_facade_seat.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Full gate sweep**

Run: `uv run pytest -m "not gpu and not slow" -q` (green)
Run: `uv run lint-imports` (the facade→sim import is legal; app must not import sim directly — unchanged)
Run: `uv run python scripts/p05_baseline_hash.py --check` (9/9)

- [ ] **Step 7: Commit**

```bash
git add src/gasgiant/engine/facade.py tests/unit/test_facade_seat.py
git commit -m "facade: thread hero_lat_deg into build_profiles + expose seat_quality/seat_status meter"
```

---

## Task 5: GUI seat meter

**Files:**
- Modify: `src/gasgiant/app/panels.py` (a colored readout beside `hero_latitude`, pattern: `_draw_hero_latitude_escape` at `:320`, invoked at `:436`)
- Test: covered at the facade level (Task 4); pixel/interaction intentionally untested (stated in spec §6.8).

**Interfaces:**
- Consumes: `Simulation.seat_status(lat_deg=None)` (Task 4).
- **Threading reality (verified in review):** the panel path is dict-only —
  `draw_params_panel(params, state)` (panels.py:128) works on `params.model_dump()`;
  `_draw_hero_latitude_escape(doc[name])` (`:436`) gets the storms SUB-DICT, NOT a
  `Simulation`. So the meter's `sim` must be threaded explicitly through three
  signatures down to the storms field-render site.

- [ ] **Step 1: Read the injection point**

Open `panels.py:128` (`draw_params_panel`), `:392` (`_draw_model` + its recursive
self-call ~`:444`), `:320`/`:436` (`_draw_hero_latitude_escape` and its call site in
the field loop). Open `main.py:1517` (the `draw_params_panel(self._live, self.panel_state)`
call inside `StudioApp.draw_controls`) and confirm `self.sim` is in scope there (it is;
it can be `None` before GL init). Confirm the storms sub-dict at the call site exposes
the live draft `hero_latitude` (it is `doc[name]["hero_latitude"]`).

- [ ] **Step 2: Thread `sim` to the storms render site + draw the meter** — in `panels.py`:

2a. Add the optional `sim` parameter (default `None`) to the two signatures and the
recursive call, so existing callers/tests are unaffected:
- `def draw_params_panel(params, state, sim=None):` (`:128`) → pass `sim=sim` into its
  `_draw_model(...)` call.
- `def _draw_model(model, doc, baseline, state, ..., sim=None):` (`:392`) → forward
  `sim=sim` in its recursive self-call (`:444`).

2b. Add the meter helper + colors (module scope):

```python
_SEAT_COLORS = {
    "green": (0.4, 0.8, 0.4, 1.0),
    "amber": (0.9, 0.7, 0.2, 1.0),
    "red": (0.9, 0.4, 0.4, 1.0),
}


def _draw_seat_meter(sim, storms_doc) -> None:
    """Live natural-bearing readout under hero_latitude. Diagnostic ONLY -- it
    never moves the storm (auto-snap was rejected in design review). Reads the
    DRAFT hero_latitude so it updates live during a drag; the underlying
    bracket-off profile is cached on the facade (no per-frame profile rebuild).
    Pre-development proxy (~1.8 deg poleward when developed); hidden when no
    hero is pinned or before GL init (sim is None)."""
    if sim is None:
        return
    draft_lat = storms_doc.get("hero_latitude")
    status = sim.seat_status(lat_deg=draft_lat)
    if status is None:
        return
    band = status.split()[1]  # "seat: <band> (...)"
    color = _SEAT_COLORS.get(band, (0.7, 0.7, 0.7, 1.0))
    imgui.text_colored(imgui.ImVec4(*color), status)
```

2c. Call `_draw_seat_meter(sim, doc[name])` right after the existing
`_draw_hero_latitude_escape(doc[name])` invocation in the field loop (`:436`), inside
the `if name == "hero_latitude"` branch (or wherever that escape widget is drawn), so
the readout sits directly under the `hero_latitude` slider.

2d. In `main.py:1517`, pass the sim: `draw_params_panel(self._live, self.panel_state, sim=self.sim)`.

Note (intentional): the meter reads the DRAFT latitude but the cached bracket-off
PROFILE is from the last committed params -- correct, because only `hero_latitude`
(the sample position) moves during a drag; the profile (bands/jets) is unchanged until
a RESTART commit. The meter is therefore live in position without any per-frame rebuild.

- [ ] **Step 3: Smoke-test the import path (no display needed)**

Run: `uv run python -c "import gasgiant.app.panels as pnl; print('import ok', hasattr(pnl, '_draw_seat_meter'))"`
Expected: `import ok True` (module imports without a GL context).

- [ ] **Step 4: Lint + layering**

Run: `uv run ruff check src/gasgiant/app/panels.py`
Run: `uv run lint-imports` (the app reads the meter via the facade `Simulation`, NOT by importing `sim.profiles` — confirm no new `from gasgiant.sim` import crept into `panels.py`)

- [ ] **Step 5: Commit**

```bash
git add src/gasgiant/app/panels.py src/gasgiant/app/main.py
git commit -m "app: live seat meter readout beside hero_latitude (diagnostic, no snap)"
```

---

## Task 6: docs + final gate sweep

**Files:**
- Modify: `docs/sliders.md` (regenerated), `docs/architecture.md`, `CLAUDE.md`

- [ ] **Step 1: Regenerate the slider docs (text only)**

Run: `uv run python scripts/render_slider_examples.py --no-render`
Then verify: `uv run python scripts/render_slider_examples.py --check`
Expected: `docs\sliders.md is up to date`. (Slider IMAGES for the new levers are a calibration-time artifact; text regen satisfies the blocking CI drift gate. If `--check` demands images, note it and render per the script docstring — but the default-off levers need no baked preset, so text-only is expected to pass.)

- [ ] **Step 2: Add the architecture + CLAUDE notes**

In `docs/architecture.md`, in the export/levers or solver-modes lever list, add one line:
```
- `jets.hero_bracket_*` — carve-and-impose hero jet override (CPU profile lever in
  build_profiles; replaces the seeded band jets in a feathered hero-centered window
  with an authored two-sided bracket). Default-off byte-identical; mode-agnostic
  (kinematic + vorticity, since it shapes u before psi/omega). Machinery only — no
  factory preset bakes it yet (the warm migration is a deferred visual checkpoint).
  A `seat_quality` diagnostic (facade `seat_status`) scores the natural bearing.
```

In `CLAUDE.md`, beside the `local_jet` / solver-modes bullet, add:
```
- `jets.hero_bracket_*` (carve-and-impose hero jet override) needs a pinned hero and
  is default-off byte-identical (structural `!= 0.0` guard over the whole block).
  Unlike a GLSL variant this is a CPU/numpy skip. Not baked into any preset yet.
```

- [ ] **Step 3: Final full gate sweep**

Run each; all must pass:
```bash
uv run pytest -m "not gpu and not slow" -q
uv run ruff check .
uv run lint-imports
uv run python scripts/render_slider_examples.py --check
uv run python scripts/p05_baseline_hash.py --check   # 9/9 unchanged
```

- [ ] **Step 4: Commit**

```bash
git add docs/sliders.md docs/architecture.md CLAUDE.md
git commit -m "docs: hero_bracket override + seat meter (sliders regen, architecture, CLAUDE)"
```

---

## Self-Review notes (author)

- **Spec coverage:** §3.1 metric → Task 3; §3.2 meter → Tasks 4–5; §3.3 override (ordering post-strength/pre-omega, C1 smoothstep, flat pedestal, full-block guard) → Task 2; §3.4 params (RESTART, no rand, pinned-hero gate) → Tasks 1, 4; §4 byte-identity-at-defaults → Task 2 no-op test + p05 gate every task; §6 testing items 1–4,7,8 → Tasks 2–4 (items 5 migration & 6 re-bake are OUT OF SCOPE by the "hold the bake" decision and intentionally omitted). §7 calibration items (bald-stripe, moat acceptance, developed-shift, feather-seam) are the deferred visual checkpoint — NOT tasks here.
- **Out of scope (deferred to the user's visual checkpoint), do NOT do:** remove `local_jet_*`; edit any `presets/*.json` or `build_*_preset.py`; re-capture p05; break/bump `GENERATION_VERSION`; replace `test_local_jet.py`.
- **Type consistency:** `build_profiles(..., hero_lat_deg=None)`, `seat_quality(profiles, lat_deg, r_core_deg, spin_sign=1.0)`, `seat_scan(...)-> list[tuple[float,float]]`, `seat_band(q)->str`, facade `seat_quality`/`seat_status` — names consistent across Tasks 2–5.
- **PR:** single branch `feat/hero-bracket`, one PR ("hero jet-environment machinery — default-off"); all byte-identical, cohesive, independently reviewable. Optionally split the GUI (Task 5) into a follow-up if review prefers, but one PR is fine since there is zero byte-identity risk across the set.

---

## Plan-review record

3-lens adversarial plan review 2026-07-19 (sim/numeric correctness, process/gates/tests,
facade/GUI/layering feasibility) — all findings incorporated BEFORE execution. The core
design (byte-identity guard, ordering post-strength/pre-omega, C1 smoothstep, flat
pedestal, layering, scope "hold the bake", strict-model missing-key behavior, p05 logic)
was **verified correct** by all three; every fix landed in the test design / GUI plumbing:

- **Sim + Process (BLOCKER, converged):** the determinism test asserted the per-rim
  on-minus-off increment (empirically swing ~0.4 → would FAIL, risking a silent
  tolerance-loosen). The seed-independent quantity is the ON two-sided shear
  `u(−19)−u(−25)` (pedestal cancels, swing ~0). Test rewritten to assert the shear
  (`< 1e-6`) + a pedestal-independent shear-ordering sign check.
- **Facade/GUI + Process (BLOCKER, converged):** Task-4 test used a nonexistent
  `Simulation(p, gpu=None, build_gpu=False)`; there is no CPU-only build path. Replaced
  with a documented white-box `object.__new__(Simulation)` + `params`/`bands` injection
  that keeps the behavioral tests in the no-GPU unit tier (the seat methods read only
  those two attributes).
- **Sim (IMPORTANT):** continuity test 40× threshold didn't discriminate C1 from a C0
  linear feather (both pass). Tightened to 8× (smoothstep ~4×, linear ~13×).
- **Facade/GUI + Process (IMPORTANT):** panels receive a dict, not a `Simulation` — the
  "mirror _draw_hero_latitude_escape" premise was wrong. Threading now explicitly routes
  `sim` through `draw_params_panel`→`_draw_model`(recursive)→`main.py:1517`, guards
  `sim is None`, and passes the DRAFT latitude to `seat_status(lat_deg=…)` for a live
  readout. The bracket-off profile is cached on the facade (invalidated where
  `self.profiles` rebuilds) so the per-frame meter does no profile work.
- **Process (IMPORTANT):** p05 is a GPU-box gate; running it on params-only (Task 1) and
  pure-metric (Task 3) tasks is impossible off-GPU and pointless. Scoped p05 to Tasks 2,
  4, 6 (render-path-affecting + final).
- **Sim/Process (MINOR):** tautological seat test (scan included −22) → assert argmax at a
  distinct latitude; moat-orientation weight dropped → documented as an intentional
  coarse-proxy simplification; compass param names → SH-only assumption stated in the
  field description; no-rand schema lookup simplified; stale line-anchor numerals fixed.

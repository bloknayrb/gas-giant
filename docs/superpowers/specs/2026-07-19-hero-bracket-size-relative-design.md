# Hero-bracket size-relative geometry — design

## Goal

Make the `jets.hero_bracket_*` geometry track the hero storm's size automatically,
so the two-sided jet bracket keeps straddling the storm when `hero_radius` changes.
Today the geometry is in absolute degrees/radians and must be re-tuned by hand every
time the storm is resized — a trap discovered during the −24 GRS bake calibration
(the ±2° offset that seated a 3.6° storm left a 6.2° storm's boundary mushy; it had to
be re-opened to ±6° by hand).

## Background (the coupling)

The bracket seats a storm well only when its two jets straddle the storm at an offset
≈ the storm's core radius `r_core` (CPU seat-quality scans confirm the sweet spot moves
with `r_core`). The carve window must likewise be at least as large as the storm, or it
fails to replace the seeded jets under the storm. Both relationships are *physical* and
proportional to size, so the parameters should encode them as ratios rather than leaving
a manual coupling for the next person who touches `hero_radius`.

## Design

All four bracket geometry knobs change from absolute units to **dimensionless multiples
of the hero core radius** `r_core = hero_radius` (radians). Grow the storm → the whole
bracket grows with it, uniformly.

| Param | Old unit / default | New unit | New default | New bounds |
|---|---|---|---|---|
| `hero_bracket_north_offset` | deg, 3.0 | × r_core | 1.0 | 0.0 … 4.0 |
| `hero_bracket_south_offset` | deg, −3.0 | × r_core | −1.0 | −4.0 … 0.0 |
| `hero_bracket_window` | deg, 4.0 | × r_core | 1.0 | 0.0 … 4.0 |
| `hero_bracket_feather` | deg, 5.0 | × r_core | 1.4 | 0.15 … 4.0 |
| `hero_bracket_north_width` | rad, 0.05 | × r_core | 0.8 | 0.1 … 2.0 |
| `hero_bracket_south_width` | rad, 0.05 | × r_core | 0.8 | 0.1 … 2.0 |

New defaults are calibrated to the **warm-preset hero radius** (0.062 rad = 3.55°), where
the old absolutes map to clean multiples: old 3.0° / 3.55° ≈ 0.85 → **1.0** (jet at the
storm edge, matching the −24 GRS calibration finding that the seating sweet spot sits at
≈1.0·r_core); old 4.0° → 1.0; old 5.0° → 1.4; old 0.05 rad = 2.86° → 0.8. (Note: the model
*default* `hero_radius` is 0.10 rad = 5.7°, NOT 0.062 — so on a bare `PlanetParams` an
enabled bracket's degree extent is ~1.6× the old absolutes. This is harmless: no preset
bakes a bracket, and the defaults are anchored to the warm regime where the bracket is
actually authored.) Param **names are unchanged**; only unit, default, bounds, and
description change.

**Reference axis = `r_core`, not `r_core·aspect`.** The bracket straddles in *latitude*
(the storm's minor axis; `hero_aspect` elongates it in *longitude*), so the latitudinal
half-extent is exactly `r_core`. Aspect correctly does not enter the scaling.

### Threading

`build_profiles(seed, bands, bands_params, jets, hero_lat_deg=None)` gains a
`hero_r_core: float = 0.0` kwarg (the hero core radius in **radians** = `hero_radius`).
It is NOT None-able: `hero_radius` is a pfield that always has a value, so the facade
passes `params.storms.hero_radius` **unconditionally** at all three call sites (harmless at
the bracket-off / no-hero sites, where the override block is skipped and the value is
unread). The override block runs only when `hero_lat_deg is not None` and a bracket
strength is non-zero; the FIRST thing inside it is a live guard `if hero_r_core <= 0.0:
raise ValueError(...)` — reachable only if a caller activates a bracket while leaving
`hero_r_core` at its 0.0 default (a caller bug, e.g. a test forgetting the kwarg; the guard
turns a silent division-by-zero into a loud error). Every geometry quantity is then
multiplied by `hero_r_core`:

```
r = hero_r_core                                   # radians (= hero_radius)
full  = jets.hero_bracket_window  * r
outer = (jets.hero_bracket_window + jets.hero_bracket_feather) * r
north_c = hero + jets.hero_bracket_north_offset * r     # hero already in radians
south_c = hero + jets.hero_bracket_south_offset * r
width_n = jets.hero_bracket_north_width * r
width_s = jets.hero_bracket_south_width * r
```

The facade computes it beside the existing `_hero_lat_deg` helper (a `_hero_r_core`
returning `params.storms.hero_radius` when a hero is pinned, else None) and passes it at
both `build_profiles` call sites (`_build`, the VELOCITY branch). `hero_r_core` is only
read inside the strength-guarded block, so a caller that passes `hero_lat_deg` without
`hero_r_core` (only reachable with a non-zero bracket) is a caller error the plan's tests
pin, not a silent fallback.

## Safety properties (all preserved)

- **Byte-identical when off.** Bracket strength 0 (north == south == 0) still skips the
  entire override block structurally — pure CPU/numpy, no GLSL variant. p05 unaffected.
- **No preset migration.** No factory preset bakes a bracket yet (grep of `presets/*.json`
  for `hero_bracket` is empty), so nothing depends on the old absolute semantics — this is
  a pure machinery refinement with zero shipped-look change. The in-flight −24 GRS bake
  calibration holds its bracket values in the OLD degree units (scratch render-script args,
  never saved to a preset), so it must be RE-DERIVED in the new units, never reloaded. The
  new bounds only partially catch a stale old-unit reload (old feather 5.0 > new hi 4.0 and
  old width 0.05 < new lo 0.1 hard-fail; old window 4.0 / offset 3.0 stay in range and would
  silently remap) — hence: re-derive, don't reload.
- **Determinism preserved.** Geometry stays a pure function of hero lat + radius + params,
  no seed; the seed-independent two-sided-shear property is unchanged.
- **Tier unchanged** (RESTART). The seat meter is untouched — it reads the natural
  (bracket-off) profile and already scales its own sampling by `r_core`.

## Testing

Update `tests/unit/test_hero_bracket.py` for the new units (defaults, RESTART tier,
no-rand, byte-identity-off with a pinned hero, C1 continuity, seed-independent shear —
each now passing `hero_r_core`), and add the defining behavior test:

- **`test_bracket_geometry_scales_with_hero_radius`**: with a fixed offset, DOUBLING
  `hero_r_core` doubles the effective jet-center offset in degrees (measure the latitude
  of the imposed jet extremum, or the effective full/outer window edges). This is the
  property the whole change exists to guarantee.
- Byte-identity-off must still hold with an *off-default* `hero_radius` (the guard skips
  before any geometry math, so radius must not matter when the bracket is off).

Docs: `docs/sliders.md` text regen (`--no-render`) for the changed descriptions;
one line in `docs/architecture.md`'s export/lever notes and `CLAUDE.md`'s hero_bracket
note recording the size-relative units. No GLSL, so p05 is a default-program tripwire
only (must stay 9/9).

## Out of scope

- The −24 GRS bake itself (still the user's held visual checkpoint).
- Hemisphere-agnostic offsets (the `north_offset ≥ 0` SH-only limitation is unchanged).
- Any change to the seat metric, the carve/impose mechanism, or the pedestal.

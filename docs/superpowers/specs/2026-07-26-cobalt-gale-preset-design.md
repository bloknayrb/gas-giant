# `cobalt_gale` — a scattering-blue hot Jupiter (design)

Date: 2026-07-26. Status: approved, in build.
Branch: `feat/cobalt-gale-preset`, based on `fix/ember-dwarf-tooling-followups` (PR #58)
rather than master — that branch carries the `preview_globe` longitude-origin fix and the
`probe_lut_usage` override-precedence fix, and authoring a preset against either of those
bugs would corrupt the judgement this whole exercise depends on.

## What it is

The library's eighth factory preset and its second non-solar-system object: a tidally
locked hot Jupiter in the manner of HD 189733b.

The premise is a **mechanism contrast with `neptune`**. Neptune is blue by *absorption* —
methane eats the red end. This planet is blue by **Rayleigh scattering** off high-altitude
MgSiO₃ (silicate) haze, which goes as λ⁻⁴ and therefore throws back blue. HD 189733b is
the exoplanet this was actually measured on (Evans et al. 2013, HST secondary-eclipse
photometry).

That difference is not academic; it inverts the tonal structure. Neptune is a *bright* blue
ball with smooth zones. Here the haze is the only reflector, so:

- **bright = thick silicate cloud** (blue-white haze crest)
- **dark = clearer air**, seen down into a hot, near-black, alkali-absorbing depth

Emission stays at zero. At roughly 1200 K this object glows in the infrared, not the
visible, so every bright pixel is reflected light. This is also the deliberate separation
from `ember_dwarf`, whose grammar is the opposite ("bright = a hole in the deck, with
hotter gas showing through"). Two presets sharing that inverted reading and differing only
in hue would be one preset with a hue slider.

## Realism latitude (explicit, so it is not "corrected" later)

The user's standing direction for this preset: **"I want cool and plausible, absolute
realism isn't required."** Contrast, chroma and jet speed may be pushed past what a
photometric or GCM-faithful model would justify when the globe looks better for it. The
science above fixes the *premise and the direction* of every choice — blue from scattering,
bright means cloud, no visible-light glow — not the numbers. A future reviewer should not
file a deliberate exaggeration here as a defect.

## Known expressive limit, accepted

A tidally locked planet's signature features are **longitudinal**: the day/night contrast
and the eastward-displaced hot spot. `appearance.palette_rows` anchor on *latitude* only,
and the renderer is full-lit by design (a terminator hides half the planet and flatters
what it leaves). So the color and the jet are in scope; a substellar hot spot is not.
Accepted at design time rather than discovered in iteration.

## Composition: the jet is the subject

Chosen over a hero-storm composition because a jet-led planet is what keeps a hot Jupiter
from reading as Jupiter-in-blue. Tidally locked giants develop **equatorial
superrotation** — one broad prograde equatorial jet dominating, rather than Jupiter's many
alternating bands — and the engine has a first-class lever for it.

- `jets.equatorial_speed` ≈ 3.0–3.4, `jets.equatorial_width` ≈ 0.30–0.36 rad (17–21°).
  Deliberately broad; `ember_dwarf` runs 2.1 / 0.16 rad for comparison.
- `jets.polar_decay` ≈ 0.9 to quiet the mid-latitudes.
- **An AUTHORED 9-band template**, which reverses this document's original "seeded bands,
  low count" plan. Reason, measured: bands are laid out in *sin-latitude* space with
  jittered widths (`sim/bands.py`), so at `count` 10 the near-equator bands are only ~11°
  wide and their edges fall wherever the seed puts them. One landed on the equator and split
  the jet into two bright lanes with a dim one between (rendered luma +5 0.577, 0 0.436, −5
  0.343, −10 0.607). The equator was *already* dominant over the mid-latitudes by then
  (0.58–0.61 vs 0.38) — it just was not a *jet*, and no seed roulette fixes that reliably.
  The original argument against the template path was that the shipped template **is
  Jupiter's layout**; that argues against inheriting warm's, not against authoring one.
  Authoring it also makes the band values — which *are* the LUT index — numbers in the build
  script rather than emergent accidents.
  Constraint worth recording: identity is derived as `values < median(values)` and must
  strictly alternate, so a band *centred* on the equator requires `count ≡ 1 (mod 4)`.
  9 works; 11 would put a **belt** (dark lane) on the equator.
- `solver.vort_inject_mask = "shear"` so eddies form on the jet flanks as chevrons.
  Unmasked injection is a known failure: `gas_giant_warm` needed exactly this masking to
  stop broadband churn from washing out its band structure.
- `solver.vort_psi_drag` for swirl scale — the validated 2-D scale-control lever.
- `storms.hero_count` 0. First factory preset whose identity is a jet, not a storm.

Vorticity solver, consistent with `gas_giant_warm` / `neptune` / `ember_dwarf`.

## Palette

Knees below are placeholders. **The measured LUT histogram sets them**, not this table —
see Method. Seven rows on a **mirrored** signed-latitude ladder (0, ±32, ±56, ±78).

| LUT index | Reading | Approx color |
|---|---|---|
| 0.00–0.25 | clear air, deep absorption | near-black indigo `(0.03, 0.035, 0.09)` |
| 0.25–0.50 | thin haze | dark cobalt `(0.08, 0.13, 0.30)` |
| 0.50–0.70 | haze deck | cobalt / steel `(0.20, 0.30, 0.55)` |
| 0.70–0.90 | thick cloud | pale steel-lavender `(0.55, 0.62, 0.80)` |
| 0.90–1.00 | cloud crest | blue-white `(0.85, 0.88, 0.96)` |

Poleward rows drift toward a colder, darker slate — less haze production away from the
equatorial upwelling. This buys the jet its contrast from the palette's latitude structure
instead of from a global contrast lever, which would flatten everything else to get it.

## Guardrails carried from `ember_dwarf`

These are inherited-lever traps already paid for once; see the
`preset-authoring-measure-the-lut` note.

1. **`appearance.chroma_aging` = 0.0 — the measured blue-killer.** `gas_giant_warm` bakes
   0.35. It ties saturation to the freshness tracer and deposits reddish-brown chromophore
   in aged air. Measured on-disc mean RGB across 0.35 → 0: `(0.308, 0.001, 0.061)` →
   `(0.210, 0.004, 0.183)` — it *collapses blue*, tripling it when removed. On a preset
   whose entire premise is a blue, this is the most dangerous single value to inherit.
2. **`appearance.detail_chroma` = 0.0.** It drags dark decks red; this deck is dark *and*
   blue. (It keys on the synthesized detail field, so it is weaker than it looks — but the
   strong side cools bright excursions, and here those are the haze crests.)
3. **Palette rows mirrored.** Rows blend on *signed* latitude and **clamp** outside the
   outermost anchor, so a one-sided ladder paints an entire hemisphere with the row at 0.0.
4. **`bands.value_contrast` is inert on the band-template path and drastic on the seeded
   path.** Pick a path and state which, rather than inheriting a number that does nothing
   in its source and everything here. *Resolved:* the template path was chosen (above), so
   `value_contrast`, `count`, `width_jitter`, `width_tail` and `hue_jitter` are all inert and
   are pinned to inert values rather than left carrying warm's numbers — nothing should read
   as a tuning decision that cannot have an effect. `faded_sector` **is** live on the
   template path (its geometry is picked in `_finish_layout`, which both paths run) and is
   pinned to 0: it is Jupiter's SEB-fade epoch feature and would be an accidental import.

Prefer `chroma_scale` (Oklab) over `saturation` (sRGB mix).

## Method

1. Build script `scripts/build_cobalt_gale_preset.py` with the load==save reproducibility
   diff-guard, per existing build-script convention.
2. **Dynamics first, with `detail.intensity = 0`** (standing instruction: calibrate sim
   behavior detail-off).
3. **Measure before authoring any color.** `scripts/probe_lut_usage.py` swaps in a linear
   grayscale ramp so rendered luminance *is* the palette index. Authoring blind is what
   put `ember_dwarf`'s colors where the planet never went — twice, in both directions
   (knee too low → uniform lava; too high → flat ball with one spot).
4. Author the ramp to the measured histogram, then **re-probe at the shipped resolution**:
   the distribution shifts materially with sim-res (≈1.55× for `ember_dwarf`'s ember
   fraction between 1024 and 4096). Quote the exact invocation with any histogram.
5. **Judge on globes**, via `scripts/preview_globe.py`, including multi-longitude sheets.
   Equirect smears the poles across the full width, so polar treatment looks catastrophic
   and band spacing looks wrong. The user judges each round.
6. Ship: README entry + image, `render_readme_examples.py` PRESETS list, gates
   (`p05 --check` 9/9, `pytest -m "not gpu and not slow"`, `ruff`, `lint-imports`, the
   palette-anchoring test), PR.

`sim.resolution` 4096 is the ship value, matching the other modern presets.

**Outcome (measured):** it does NOT need a `SIM_RES_OVERRIDES` entry, unlike `ember_dwarf`.
Fraction of the disc above LUT index 0.80, detail on: 7.8% at 1024 → 8.5% at 4096, ≈1.09×,
against `ember_dwarf`'s 13.0% → 20.2% (≈1.55×). Structural reason: `ember_dwarf`'s premise
is the fraction of the deck torn open by *convective excursions*, which is
resolution-dependent; this preset's composition is an authored band template with verbatim
values, which is resolution-independent by construction. Confirmed visually — the reduced
grid renders the same planet with coarser filaments, not a different one.

## Risks

**A dark planet can read muddy.** The documented failure mode in both directions, and the
reason the LUT probe exists. Mitigation is measurement, not taste.

**A jet-led composition may read flat** without a focal feature. Fallback, only if it does:
one mid-latitude anticyclone. Try without first — that is the point of the choice.

**Render cost** if 4096 is pinned: another 16×-pixel README render, as with `ember_dwarf`.

## Gate

`/code-review high` is the required pre-merge gate and is **user-invoked only** — the skill
is `disable-model-invocation`. Open the PR, verify CI ran on the current head SHA (and that
`gpu-smoke` is *present* in the check list, since an absent required check reads identically
to a passing one), then stop and wait for the findings.

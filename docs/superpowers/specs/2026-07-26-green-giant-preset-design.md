# `green_giant` — an enriched sulfur-chromophore giant (design)

Date: 2026-07-26. Status: approved, built, in review.
Branch: `feat/green-giant-preset`, based on `master`.

## What it is

The library's ninth factory preset. Unlike `ember_dwarf` (a brown dwarf) and
`cobalt_gale` (a hot Jupiter), this one **is still a Jupiter-class planet** — the brief was
*"kind of like Jupiter but not: a different atmospheric composition that causes different
colors and different weather"*. So the separation has to be carried by chemistry, not by
moving the object to a different corner of the HR diagram.

The premise is an **enriched (high-metallicity), sulfur-rich giant**, and one fact drives
both halves of the brief:

- **Color.** H₂S outstrips NH₃ in the visible atmosphere. That is real, not invented:
  Uranus's cloud deck *is* H₂S (Irwin et al. 2018). UV photolysis polymerises it into
  S₈ / polysulfide chromophores, which are yellow, and residual methane absorbs the red
  end. Yellow pigment under a red-eating atmosphere reads olive → chartreuse.
- **Weather, from the same fact.** A metal-enriched atmosphere has a higher mean molecular
  weight, hence a smaller scale height (`H = kT/μg`) and a smaller deformation radius
  (`L_d = NH/f`). Finer, busier structure than Jupiter at the **same** rotation rate.

It fills the library's only remaining hue gap — nothing was green — and adds a fifth kind
of composition.

## Realism latitude (explicit, so it is not "corrected" later)

The user's standing direction: **"cool and plausible, absolute realism isn't required."**
The science fixes the *premise and the direction* of every choice — green from sulfur
chromophores, bright = fresh cloud, no visible thermal glow — not the numbers.

Three caveats stated up front rather than papered over:

1. CH₄'s visible absorption is weak shortward of ~700 nm, so in practice it *desaturates*
   yellow toward khaki rather than rotating it to green. The palette has to do real work;
   see the hue guard.
2. Sulfur chemistry is the classical candidate for Jupiter's **browns** (NH₄SH photolysis
   products are red-brown). "Sulfur → yellow" picks the friendliest of several products.
   Defensible — S₈ genuinely is yellow, cf. Io — but not the only reading.
3. H₂S is normally cold-trapped below the photolysis region on a cold giant, so a
   Jupiter-temperature planet would not get the chromophore at all. The escape hatch is
   this preset's own enrichment premise: H₂S/NH₃ > 1 leaves free H₂S above the NH₄SH
   cloud. That is precisely the Uranus argument already cited, and it is load-bearing
   rather than decorative.

## Composition: an ALTERNATION

> **Glassy chartreuse zones studded with pale cream ovals, alternating with churned olive
> belts streaked with dark cigars.**

The eye lands on two materially different surfaces side by side — one laminar and light,
one boiling and dark. That is the fifth composition in the library (`gas_giant_warm` a hero
GRS, `cobalt_gale` a jet, `ember_dwarf` convective tears, `saturn_pale` nothing in
particular).

**An earlier draft had this exactly backwards** and it is worth recording, because the
error was invisible from the parameter names. It described pale ovals in the *churning
belts*. The seeder does the opposite (`sim/vortices.py`):

- `oval_density` → **white ovals, anticyclones, ZONES only**, `brightness +0.22`
- `barge_density` → **brown barges, cyclones, BELTS**, `brightness −0.28`
- `small_density` → speckle in both, but **signed**: dark in belts, bright in zones

No lever moves the bright ovals into belts; only an explicit `storms.cast` entry places a
vortex at an arbitrary latitude. The honest anatomy is Jupiter's own, and it makes the
better picture: each half of the planet gets furniture of its own polarity, which is what
makes the alternation read at all.

## The lever: `vort_inject_mask = belts`, and the trap in it

The first factory preset to use `belts` — five presets declare `shear`, three `global`, and
neptune's `shear` is inert at `vort_inject = 0.0`, so only four had a live mask.

**A mask is a MULTIPLIER, and `belts` is the WIDE one.** Measured on this preset's own
16-band layout:

| mask | mean | latitudes > 0.5 |
|---|---|---|
| `shear_norm` | 0.110 | 3.9% |
| `belt_mask` | 0.475 | 47.6% |

So `belts` spreads churn over ~12× the area `shear` does and sits much nearer `global` —
which the project record says dissolves the banding at warm's amplitude. Inheriting warm's
`vort_inject = 1.8` would have applied roughly 5× the integrated forcing warm was
calibrated at. Any intuition that `belts` "concentrates" churn relative to `shear` is
backwards by an order of magnitude.

Mean-mask matching gives 0.36 (`1.8 × 0.096 ≈ 0.36 × 0.475`). That is where to *start*, not
where to stop: the same integral spread over 12× the area still reads much busier, and the
shipped value is **0.30**.

Second trap, **and it turned out not to be one** — recorded because the reasoning was
plausible and wrong. The mask samples an **unwarped** latitude profile
(`omega_force.comp`) while visible band edges ride `bands.warp_amount`, which suggests a
large meander leaves churn stopping on razor-straight lines across meandering bands. The
preset initially shipped `warp_amount` 0.01 on that basis. Two things refute it: the
concern is **not new to `belts`** (`detail.comp` samples the same unwarped `u_profile_dyn`
for `zone_texture` and the lightning gate, so `gas_giant_warm` already ships
0.04-with-unwarped without the defect), and measured, 0.01 against 0.04 gives a mean
absolute on-disc difference of **0.007** — at this injection amplitude the advection smears
the mask boundary long before it can read as a straight edge. Reverted to the
library-standard 0.04, since `docs/formations.md` calls straight band edges "the single
biggest 'procedural planet' tell" and trading a documented realism lever for an unmeasured
worry is the wrong direction.

**The mask was proved before being committed to** — a detail-off A/B at sim-res 1024,
`belts @0.30` against `shear @1.8`. `belts` won decisively: crisp lane alternation with
legible vortices rolled up inside the dark belts, where `shear` smeared the southern
hemisphere into a continuous marbled soup. Documented fallback to `shear` was not needed.

## Field

The premise motivates the **deformation radius and nothing else**. In particular it does
not motivate touching `coriolis_f0`: this planet rotates at Jupiter's rate, and warm's 3.0
*is* this library's Jupiter value. An earlier draft lowered it to 2.8 "to be Jupiter-rate",
which was both redundant and backwards — lowering f0 makes structures *larger*, against the
premise it was cited for — and then asserted the value lay "strictly between" 3.0 and 5.5,
an assert that fails on its own number.

| preset | f0 | L_d | f0/L_d | vs warm |
|---|---|---|---|---|
| `gas_giant_warm` | 3.0 | 0.18 | 16.7 | — |
| **`green_giant`** | **3.0** | **0.13** | **23.1** | **1.4×** |
| `ember_dwarf` | 5.5 | 0.11 | 50.0 | 3.0× |
| `cobalt_gale` | 2.0 | 0.26 | 7.7 | 0.46× |

## Bands: seeded, and the count must be EVEN

Seeded, **reversing `cobalt_gale` deliberately**. That preset authored a template because
*one* band was the subject and a width lottery could split it. Here no single band carries
the composition and organic jittered widths are an asset. Dropping the template also clears
the 12-band Jovian skeleton shared by warm / jupiter_like / jupiter_vorticity / neptune.

**The inert-lever trap runs the other way from `cobalt_gale`'s.** On the template path
`count`, `width_jitter`, `width_tail`, `value_contrast` and `hue_jitter` are inert; setting
`template = None` makes all five live at once, inheriting warm's 22 / 0.4 / 0.35 / 1.7 /
0.04. Every one is set deliberately.

**`count` must be even.** `sim/bands.py` picks `zone_first` by coin flip and derives
identity as `values < median(values)`. At an **odd** count with belts in the majority the
median *is* the top belt value, so that belt is silently reclassified as a zone. Measured
over 400 seeds: count 14 → 0 failures, **count 15 → 204**, count 16 → 0, count 17 → 208.
The model has an alternation validator, but it runs **only on the template path**. The
build script therefore runs the real layout and asserts alternation directly.

**`value_contrast` 1.15**, which is `ember_dwarf`'s measured value used for its measured
reason. An earlier draft set 1.9 on the argument that "the belts grammar needs a strong
zone/belt step" — which is simply false: belt identity is `BandLayout.is_belt`, frozen at
layout build, and the mask reads that frozen identity. `value_contrast` does not strengthen
the mask by one bit. At 1.9 the layout also collapses to a binary square wave (belts
0.03–0.14, zones 0.95–1.00, empty gap between) and pins one band at exactly 1.000 — here a
~33° *polar* zone, the widest possible instance of ember_dwarf's razor-edged-ring defect.

## Palette: measured, and green by assertion

Baseline (`probe_lut_usage.py --preset green_giant`, sim-res 1024, detail off):
p01 0.170, p50 0.615, p99 0.911, mean 0.565.

| .0–.1 | .1–.2 | .2–.3 | .3–.4 | .4–.5 | .5–.6 | .6–.7 | .7–.8 | .8–.9 | .9–1 |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 2.4 | 11.3 | 14.4 | 10.0 | 10.2 | 19.2 | 21.5 | 9.7 | 1.4 |

The distribution is **bimodal** — a belt lobe at 0.25–0.45 and a zone lobe at 0.60–0.80
with a trough between. That is the composition showing up in the histogram, and the trough
is where the palette's step belongs; it is placed at p50.

The first authored ramp was placed by eye and wasted its bottom third — knees at 0.00 and
0.30 carrying near-black when only 2.4% of the disc is ever below 0.2, which rendered the
belts nearly **black** rather than olive. Symmetrically it reserved a cream crest above
0.88 that 1.4% of the disc reaches. Both ends are now pulled to where the planet lives:
deepest knee at p01, crest at p99.

**The hue is a requirement, not a preference.** The draft's "olive" placeholders measured at
hue **60–72°** — which is yellow; green starts at 90° — with a max-channel-minus-second
separation of 0.005–0.06, an order of magnitude below every other preset (neptune 0.19–0.44,
ember_dwarf 0.39–0.54). Its brightest stop was `(0.90, 0.90, 0.66)`: R = G, i.e. pure yellow
with **zero green content**. It would have shipped as khaki ~25° of hue from `saturn_pale`
and would not have filled the gap it exists to fill. The build script now asserts
**G − R ≥ 0.08** across the upper half of the mid-latitude ramps.

Polar rows are clamped (no stop above 0.40): poleward of ~60–64° `omega_force.comp` zeroes
relative vorticity, and bands lay out in sin-latitude so the outermost are ~33° wide — a
large unforced cap that paints whatever the outer bands hold.

**The clamp is applied at two ceilings, because the unforced cap is wider than the ±78
rows cover.** At seed 3216 the southernmost band spans **−56.95°..−90° with value 0.772**
— a 33°-wide *bright zone* sitting almost entirely inside the unforced region — and
between ~57° and ~78° `bake_rows` is dominated by `SUBPOLAR`, not `POLAR`. The shipped
render is fine (zonal-mean luma 0.29 at −60° falling to 0.19 at −85°, against 0.67 at the
equator, so there is no bright ring), but nothing *asserted* that, so a future brightening
of `SUBPOLAR` could repaint the cap silently. The build script now also caps the ≥55° rows
at a looser 0.66 — enough to exclude a haze-crest-grade value while leaving those rows
their legitimate mid-tones. This is a regression guard, not a bug fix.

## `chroma_aging` is disqualifying — analytically, not by taste

`appearance.chroma_aging` (warm bakes 0.35) is pinned to 0.0, and this is **not** a
measurement deferred to an A/B. Its `pfield` description claims "chroma-only — the latitude
palette's HUE is untouched". **That description is stale.** `render/kernels/derive.comp`
applies a *fixed additive push* along +a/+b in Oklab —
`lab.yz += u_chroma_aging * chromo * 0.28 * normalize(vec2(0.6, 0.8))` — and the shader's
own comment says why ("hence the directional hue push"). It is gated on **dark** material,
i.e. exactly this deck. Computed on these stops at 0.35, the two dark rows rotate 45–58° of
hue and **clip the green channel to zero**, rendering red/orange.

`detail_chroma` is 0.0 for the ordinary reason (it drags dark decks red; this deck is dark
*and* green).

`emission.*` is all 0.0. The `lightning_strength` premise was actually *correct* —
`derive.comp` gates lightning on the same `belt_mask` channel, so it would land in the
churning belts — but lightning writes only to `out_emission`, which the Color preview never
composites, making it invisible to every step of the authoring loop. Shipping an
art-direction value nobody in this workflow can see is not a decision.

## Resolution: pinned, and for a different reason than `ember_dwarf`

Fraction of the disc above LUT index 0.80, detail on:

| preset | 1024 | 4096 | ratio | pinned |
|---|---|---|---|---|
| `green_giant` | 16.6% | 21.6% | ~1.30× | **yes** |
| `ember_dwarf` | 13.0% | 20.2% | ~1.55× | yes |
| `cobalt_gale` | 7.8% | 8.5% | ~1.09× | no |

The ratio lands *between* the two precedents and does not settle it. **The render does.** At
1024 this is a chaotic marbled ball; at 4096 the lanes go crisp and zonal and the pale zone
ovals become discrete and countable — and those countable ovals are half the stated
composition, so a 1024 asset would misrepresent the preset. Structurally this is the mirror
of `cobalt_gale`: its subject is a seeded **vortex population** whose members must be
individually resolved to read as members, where cobalt_gale's is a verbatim band template,
resolution-independent by construction.

**Measured caveat worth keeping:** the round-3 calming (`vort_inject` 0.36→0.30,
`turbulence.intensity` 1.4→1.15, `belt_boost` 2.2→1.8) was judged at 1024. Re-rendered at
4096, hot-vs-calm gives a mean absolute on-disc difference of **0.008** and high-frequency
energy within **1.8%** (0.0390 vs 0.0383). The grid organises the field far more than those
three levers do. The calmer values are kept because they are clearly better at 1024 — what
GUI iteration shows — and indistinguishable at 4096.

## Differentiation

Scalar fields differing from `gas_giant_warm` (200 fields, palette arrays excluded):

| preset | differing |
|---|---|
| `saturn_pale` | 150 |
| **`green_giant`** | **86** |
| `ember_dwarf` | 86 |
| `cobalt_gale` | 86 |
| `neptune` | 71 |
| `jupiter_vorticity` (acknowledged near-duplicate) | 29 |

Exactly level with the two sibling presets. `build_ember_dwarf_preset.py` states the rule —
*"a different object is a whole-preset job, not a recolor"* — and this clears it by
measurement rather than by assertion.

## Known risks and residue

- **No focal feature.** The subject is an alternation, which is weaker than a hero or a jet.
  Contingency if it ever reads flat: one accent oval via `storms.cast`, the only mechanism
  that places a vortex at an arbitrary latitude. Not needed so far.
- **Khaki drift.** Guarded by the G − R assert; `chroma_scale`, `haze_color` and both
  chroma levers are all held off the yellow-brown axis.
- **`tests/gpu/test_vort_inject_mask.py::test_mask_is_noop_when_inject_zero` is RED on the
  primary dev box, and it is PRE-EXISTING** — proven by running it on stashed-clean master
  (0.1424 there vs 0.1417 on this branch). The mask has exactly one consumer, inside
  `if (u_vort_inject > 0.0)`, so it cannot act at inject 0; this is SOR noise amplified
  through 60 steps of chaotic advection, i.e. the documented vorticity-path flakiness. CI's
  single-threaded `gpu-smoke` is authoritative and the test matches its `noop` filter.

## Gate

`/code-review high` is the required pre-merge gate and is **user-invoked only**. Open the
PR, verify CI ran on the current head SHA and that `gpu-smoke` is *present* in the check
list (an absent required check reads identically to a passing one), then wait for findings.

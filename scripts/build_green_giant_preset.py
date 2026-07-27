"""Build the `green_giant` factory preset — an enriched, sulfur-chromophore giant.

The library's ninth factory preset. Unlike `ember_dwarf` (a brown dwarf) and
`cobalt_gale` (a hot Jupiter), this one IS a Jupiter-class planet — the brief was
"kind of like Jupiter but not: a different atmospheric composition that causes
different colors and different weather". So the separation has to be carried by
chemistry, not by moving the object to a different corner of the HR diagram.

The premise is an ENRICHED (high-metallicity) SULFUR-RICH giant, and one fact
drives both halves of the brief:

  * **Color.** H2S outstrips NH3 in the visible atmosphere. That is real, not
    invented: Uranus's cloud deck IS H2S (Irwin et al. 2018). UV photolysis
    polymerises it into S8 / polysulfide chromophores, which are yellow, and
    residual methane absorbs the red end. Yellow pigment under a red-eating
    atmosphere reads olive -> chartreuse.
  * **Weather, from the SAME fact.** A metal-enriched atmosphere has a higher
    mean molecular weight, hence a smaller scale height (H = kT/mu*g) and a
    smaller deformation radius (L_d = NH/f). Finer, busier structure than
    Jupiter at the SAME rotation rate.

Three caveats stated up front, so a reviewer sees them acknowledged rather than
papered over:

  1. CH4's visible absorption is weak shortward of ~700 nm, so in practice it
     DESATURATES yellow toward khaki rather than rotating it to green. The
     palette has to do real work; see the hue guard in build().
  2. Sulfur chemistry is the classical candidate for Jupiter's BROWNS (NH4SH
     photolysis products are red-brown). "Sulfur -> yellow" picks the friendliest
     of several products. Defensible -- S8 genuinely is yellow, cf. Io -- but not
     the only reading.
  3. H2S is normally cold-trapped below the photolysis region on a cold giant, so
     a Jupiter-temperature planet would not get the chromophore at all. The
     escape hatch is this preset's own enrichment premise: H2S/NH3 > 1 leaves
     free H2S above the NH4SH cloud. That is precisely the Uranus argument
     already cited, and it is load-bearing rather than decorative.

Realism latitude, from the user and recorded so it is not "corrected" later:
"cool and plausible, absolute realism isn't required". The science fixes the
PREMISE and the DIRECTION of every choice -- green from sulfur chromophores,
bright = fresh cloud, no visible thermal glow -- not the numbers.

THE COMPOSITION: an ALTERNATION, not a hero and not a jet.

    Glassy chartreuse zones studded with pale cream ovals, alternating with
    churned olive belts streaked with dark cigars.

The eye lands on two materially different surfaces sitting next to each other --
one laminar and light, one boiling and dark. That is the fifth composition in the
library (warm = a hero GRS, cobalt_gale = a jet, ember_dwarf = convective tears,
saturn_pale = nothing in particular).

The lever that carries it is `solver.vort_inject_mask = BELTS`: eddy churn
confined to the cyclonic dark lanes, zones left laminar in vorticity. NO factory
preset used it before this one -- five declare SHEAR, three declare GLOBAL, and
neptune's SHEAR is inert at vort_inject 0.0, so only four presets even have a
live mask. See the SOLVER block for why that required retuning the AMPLITUDE too,
which is the non-obvious half.

Run: uv run python scripts/build_green_giant_preset.py
"""
from __future__ import annotations

from pathlib import Path

from gasgiant.params.model import GradientStop, InjectMask, PaletteRow
from gasgiant.params.presets import load_factory_preset, load_preset, save_preset
from gasgiant.sim.bands import generate_bands

PRESETS_DIR = Path("src/gasgiant/presets")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Four latitude PROFILES (equator/midlat/subpolar/polar), mirrored into seven
# palette rows at build time -- see the comment on `rows` in build() for why the
# mirroring is mandatory rather than tidy.
#
# Unlike `cobalt_gale`, THE KNEE LADDER IS NOT THE COMPOSITION here. That preset
# needed early knees at the equator so its jet would become the subject; this
# preset's subject is the zone/belt alternation, which is the same at every
# latitude. So the equatorial and mid-latitude rows share one shape, and what the
# ladder has to deliver instead is a STEEP STEP near mid-ramp, so that band values
# straddling the median land on visibly different materials rather than on two
# points of one smooth gradient.
#
# The stop POSITIONS are measured, not guessed, and are COUPLED to the field
# tuning below AND to sim resolution -- RE-MEASURE if either moves, quoting the
# exact invocation (the tool defaults to sim-res 1024 with detail OFF, a
# materially different distribution):
#
#     uv run python scripts/probe_lut_usage.py --preset green_giant \
#         --sim-res 4096 --keep-detail
#
# THE HUE IS A REQUIREMENT, NOT A PREFERENCE, and it is the one thing the first
# draft of this preset got wrong. Authored "olive" placeholders measured at hue
# 60-72 degrees -- which is YELLOW; green starts at 90 -- with a max-channel minus
# second-channel separation of 0.005-0.06, an order of magnitude below every other
# preset in the library (neptune 0.19-0.44, ember_dwarf 0.39-0.54 at identity).
# The draft's brightest stop was (0.90, 0.90, 0.66): R == G, i.e. pure yellow with
# ZERO green content. It would have shipped as khaki roughly 25 degrees of hue from
# `saturn_pale` and would not have filled the gap it exists to fill.
#
# Hence: G - R >= 0.08 across the identity range, enforced in build().
# THE KNEE POSITIONS BELOW ARE THE MEASUREMENT, not a guess. Baseline to diff a
# re-measurement against (decile shares of the disc, in %):
#
#   uv run python scripts/probe_lut_usage.py --preset green_giant
#     sim-res 1024, detail OFF: p01 0.173  p50 0.614  p99 0.910  mean 0.565
#        .0-.1  .1-.2  .2-.3  .3-.4  .4-.5  .5-.6  .6-.7  .7-.8  .8-.9  .9-1
#         0.0    2.2   11.1   14.5   10.1   10.4   19.4   21.3    9.6    1.3
#
# The distribution is BIMODAL, which is the composition showing up in the
# histogram: a belt lobe around 0.25-0.45 and a zone lobe around 0.60-0.80, with a
# trough between them. That trough is where the palette's step belongs, and p50
# (0.614) is where it is put.
#
# The first authored ramp was placed by eye and wasted its bottom third: knees at
# 0.00 and 0.30 carrying near-black, when only 2.2% of the disc is ever below 0.2.
# The visible consequence was belts that rendered nearly BLACK rather than olive.
# Symmetrically it reserved a cream crest above 0.88 that 1.3% of the disc reaches.
# Both ends are now pulled in to where the planet actually lives: the deepest knee
# sits at p01 and the crest at p99.
MIDLAT = [  # the deck proper -- the ladder the composition actually runs on
    (0.00, (0.030, 0.050, 0.028)),   # floor: anchors the ramp, ~0% of the disc
    (0.17, (0.075, 0.120, 0.055)),   # p01 -- the deepest value actually reached
    (0.35, (0.150, 0.240, 0.100)),   # THE BELT LOBE (14.5%): readable olive
    (0.50, (0.230, 0.350, 0.150)),
    (0.62, (0.360, 0.545, 0.240)),   # THE STEP, at p50 -- the histogram's trough
    (0.75, (0.560, 0.715, 0.335)),   # THE ZONE LOBE (21.3%)
    (0.90, (0.780, 0.900, 0.560)),   # p99 -- fresh cloud crest, sulfur cream
    (1.00, (0.850, 0.950, 0.680)),   # rare highlight
]
EQUATOR = [  # same shape, slightly richer -- more upwelling, more fresh cloud
    (0.00, (0.034, 0.055, 0.030)),
    (0.17, (0.082, 0.132, 0.060)),
    (0.35, (0.165, 0.262, 0.110)),
    (0.50, (0.250, 0.380, 0.163)),
    (0.62, (0.390, 0.585, 0.258)),
    (0.75, (0.600, 0.755, 0.358)),
    (0.90, (0.810, 0.925, 0.595)),
    (1.00, (0.875, 0.965, 0.710)),
]
SUBPOLAR = [  # cooling: less photolysis, so the bright end never reaches cream
    (0.00, (0.026, 0.042, 0.025)),
    (0.17, (0.060, 0.092, 0.046)),
    (0.35, (0.110, 0.165, 0.078)),
    (0.50, (0.160, 0.230, 0.108)),
    (0.62, (0.235, 0.325, 0.150)),
    (0.75, (0.330, 0.435, 0.205)),
    (0.90, (0.455, 0.565, 0.290)),
    (1.00, (0.530, 0.640, 0.350)),
]
# The polar rows exist to CAP THE CAPS, and this is a borrowed fix rather than a
# guessed one. omega_force.comp's polar confine ramp zeroes relative vorticity
# poleward of 60-64 deg, and bands lay out in SIN-latitude so the outermost bands
# are ~33 deg wide -- so whatever value the outer bands happen to hold gets painted
# nearly flat across a large, unforced cap. On `ember_dwarf` that produced blazing
# gold laminar streaks over the south pole, and it survived several dynamics-side
# fixes AND a strengthened polar tint, because a tint is a HUE push and the defect
# is LUMINANCE. Clamping the bright end of the polar rows is the only fix
# independent of the band lottery and of sim resolution.
POLAR = [
    (0.00, (0.022, 0.034, 0.022)),
    (0.17, (0.046, 0.068, 0.038)),
    (0.35, (0.080, 0.115, 0.060)),
    (0.50, (0.112, 0.155, 0.082)),
    (0.62, (0.150, 0.202, 0.108)),
    (0.75, (0.195, 0.256, 0.138)),
    (0.90, (0.250, 0.322, 0.176)),
    (1.00, (0.288, 0.368, 0.202)),
]

# Storm LUT. The ovals here are fresh-cloud CONCENTRATIONS -- upwelling piles up
# unaged, unstained cloud and reads pale -- so the ramp runs from a stained dark
# core to a cream crest, the same direction as the palette rows and the same
# polarity as `cobalt_gale`. (Contrast `ember_dwarf`, where a storm is a TEAR in
# the deck and the bright end is exposed hot gas.)
STORM_TINTS = [
    (0.00, (0.075, 0.105, 0.055)),
    (0.55, (0.330, 0.470, 0.215)),
    (1.00, (0.840, 0.915, 0.640)),
]

SIM = {
    "dev_steps": 800,
    # 4096, matching the other modern presets -- AND this preset DOES need a
    # SIM_RES_OVERRIDES entry in render_readme_examples.py, which is a measured
    # decision rather than a copied one. Fraction of the disc above LUT index 0.80,
    # detail on:
    #
    #     green_giant  16.6% (1024) -> 21.6% (4096)  ~1.30x   <- pinned
    #     ember_dwarf  13.0% (1024) -> 20.2% (4096)  ~1.55x   <- pinned
    #     cobalt_gale   7.8% (1024) ->  8.5% (4096)  ~1.09x   <- not pinned
    #
    # The ratio alone lands between the two precedents and does not settle it; the
    # RENDER does. At 1024 this preset is a chaotic marbled ball; at 4096 the lanes
    # go crisp and zonal and the pale ovals in the zones become discrete and
    # countable. That is a different planet, not the same planet with coarser
    # filaments, and the countable ovals ARE half the stated composition -- so a
    # 1024 README asset would misrepresent the preset.
    #
    # Structural reason, and it is the mirror of cobalt_gale's: this preset's
    # subject is a seeded VORTEX POPULATION whose members have to be individually
    # resolved to read as members, where cobalt_gale's is an authored band template
    # with verbatim values -- resolution-independent by construction.
    "resolution": 4096,
}

# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------
# The premise -- a heavier atmosphere, hence a smaller scale height -- motivates
# the DEFORMATION RADIUS and nothing else. In particular it does NOT motivate
# touching coriolis_f0: this planet rotates at Jupiter's rate, and warm's 3.0 IS
# this library's Jupiter value. An earlier draft lowered f0 to 2.8 "to be
# Jupiter-rate", which was both redundant and backwards -- lowering f0 makes
# structures LARGER, against the very premise it was cited for.
#
#   preset          f0    L_d    f0/L_d   vs warm
#   gas_giant_warm  3.0   0.18    16.7    --
#   green_giant     3.0   0.13    23.1    1.4x
#   ember_dwarf     5.5   0.11    50.0    3.0x
#   cobalt_gale     2.0   0.26     7.7    0.46x
#
# THE INJECTION AMPLITUDE IS RETUNED WITH THE MASK, and this is the non-obvious
# half of adopting BELTS. A mask is a MULTIPLIER on vort_inject, and the two masks
# have wildly different coverage. Measured on this preset's OWN band layout (run
# build_profiles on the shipped params and take |mask| over the latitude LUT):
#
#     shear_norm   mean 0.110   3.9% of latitudes above 0.5
#     belt_mask    mean 0.475  47.6% of latitudes above 0.5
#
# So BELTS is ~4.3x the mean amplitude over ~12x the area -- much nearer GLOBAL
# (1.0) than SHEAR. It does NOT "concentrate" churn relative to shear; it spreads
# it, and any intuition to the contrary is backwards by an order of magnitude. The
# project record is explicit that global injection at 1.8 dissolves the whole
# planet, so inheriting warm's 1.8 here would have applied ~5x the integrated
# forcing warm was calibrated at.
#
# 0.36 is MEAN-MASK MATCHED to warm's integrated shear forcing:
#     warm         1.8  * mean(shear_norm) = 1.8  * 0.096 = 0.173
#     green_giant  0.36 * mean(belt_mask)  = 0.36 * 0.475 = 0.171
# (warm's own shear_norm mean is 0.096 on ITS layout, not this one's 0.110 -- the
# match is against warm's number, since warm is what was calibrated.) An earlier
# 0.55 was arrived at from an approximate mask mean and measured 1.52x warm.
SOLVER = {
    "coriolis_f0": 3.0,
    "deformation_radius": 0.13,
    # Round 2 took this from the mean-mask-matched 0.36 down to 0.30. Matching
    # warm's INTEGRATED forcing is the right place to start but not where to stop:
    # warm spends its forcing on 3.9% of latitudes (shear flanks) and this preset
    # spreads the same total over 47.6% (belt interiors), so the same integral
    # buys a much more uniformly churned planet. Measured visually -- at 0.36 the
    # churn was winning almost everywhere south of the equator and the zones only
    # read as the quiet half in the northern hemisphere, which spends the whole
    # point of choosing BELTS over SHEAR without collecting it.
    #
    # MEASURED CAVEAT, worth keeping so nobody re-litigates this: that judgement
    # was made at sim-res 1024, and AT THE SHIPPED 4096 IT BARELY MATTERS. Rendering
    # 0.36/1.4/2.2 against 0.30/1.15/1.8 at 4096 gives a mean absolute on-disc
    # difference of 0.008 and high-frequency energy within 1.8% (0.0390 vs 0.0383).
    # The grid organises the field far more than these three levers do. The calmer
    # values are kept because they are clearly better at 1024 -- which is what GUI
    # iteration and any reduced-grid render show -- and indistinguishable at 4096.
    "vort_inject": 0.30,
    "vort_inject_scale": 2.5,
    "vort_inject_mask": InjectMask.BELTS,
    "vort_hypervisc": 0.5,
    # Modest: a small L_d already limits the inverse-cascade pileup that ember
    # (0.45 at high injection) and cobalt (0.45 at L_d 0.26) have to fight.
    "vort_psi_drag": 0.20,
}

# Seeded bands, NOT Jupiter's baked template -- and this deliberately REVERSES
# `cobalt_gale`, so the reasoning is worth keeping. That preset authored a template
# because ONE band was the subject and a width lottery could split it (it did: an
# edge landed on the equator and cut the jet into two lanes). Here no single band
# carries the composition and organic jittered widths are an ASSET. Dropping the
# template also clears the 12-band Jovian skeleton that gas_giant_warm,
# jupiter_like, jupiter_vorticity and neptune all share.
#
# THE TRAP RUNS THE OTHER WAY FROM COBALT_GALE'S. On the template path `count`,
# `width_jitter`, `width_tail`, `value_contrast` and `hue_jitter` are all INERT
# (bands.py short-circuits before reading any of them). Setting template=None makes
# all five LIVE at once, inheriting warm's 22 / 0.4 / 0.35 / 1.7 / 0.04. Every one
# is therefore set deliberately below.
#
# COUNT MUST BE EVEN. sim/bands.py picks `zone_first` by coin flip and derives
# identity as `values < median(values)`. At an ODD count with belts in the
# majority, the median IS the top belt value, so `values < median` is False for it
# and one belt is silently reclassified as a ZONE. Measured over 400 seeds:
# count 14 -> 0 failures, count 15 -> 204, count 16 -> 0, count 17 -> 208. The
# model has a validator enforcing alternation, but ONLY on the template path
# (params/model.py); the seeded path has none, which is why build() runs the
# layout and asserts alternation directly.
#
# The knock-on of a misclassification would compound badly here: the stray band
# would drop out of the BELTS inject mask, get seeded with white ovals instead of
# barges, and change fade-sector candidacy.
#
# value_contrast 1.15 is ember_dwarf's measured value and is used for its measured
# reason. warm ships 1.7 but ships a TEMPLATE, so that 1.7 is inert in its source.
# At 1.9 (an earlier draft of this preset) the layout collapses to a binary square
# wave -- belts 0.03-0.14, zones 0.95-1.00 with an empty gap between -- and pins one
# band at exactly 1.000, which at this band geometry is a ~33 deg POLAR zone: the
# widest possible instance of ember_dwarf's "razor-edged gold ring". It would also
# defeat the palette above, whose whole design is a step at mid-ramp that band
# values are supposed to straddle.
#
# NOTE the draft's stated reason for 1.9 -- "the belts grammar needs a strong
# zone/belt step" -- was simply FALSE. Belt identity is BandLayout.is_belt, frozen
# at layout build; the belt mask reads that frozen identity (sim/profiles.py).
# value_contrast does not strengthen the mask by one bit.
BANDS = {
    "template": None,
    "count": 16,
    "value_contrast": 1.15,
    "width_jitter": 0.32,
    "width_tail": 0.30,
    "edge_softness": 0.012,
    "detail_amount": 0.16,
    "contrast_envelope": 0.50,
    "edge_diversity": 0.55,
    "variance_amount": 0.24,
    # warp_amount stays at the library-standard 0.04, and this REVERSES an earlier
    # cut to 0.01 that was reasoned rather than measured.
    #
    # The reasoning was: the BELTS inject mask samples an UNWARPED latitude profile
    # (omega_force.comp) while visible band edges ride warp_amount, so a large
    # meander should leave churn stopping on razor-straight lines across meandering
    # bands. Two things refute it. (a) The concern is NOT new to BELTS --
    # detail.comp samples the same unwarped u_profile_dyn for zone_texture and the
    # lightning gate, so gas_giant_warm already ships 0.04-with-unwarped and shows
    # no such defect. (b) Measured: rendering 0.01 against 0.04 gives a mean
    # absolute on-disc difference of 0.007, i.e. nothing -- at this injection
    # amplitude the advection smears the mask boundary well before it can read as
    # a straight edge.
    #
    # Meanwhile 0.01 has a real cost: it is the smallest in the library (every
    # other preset ships 0.04) and docs/formations.md calls straight band edges
    # "the single biggest 'procedural planet' tell". Trading a documented realism
    # lever for an unmeasured alignment worry is the wrong direction.
    "warp_amount": 0.04,
    # Inheritance pins (warm bakes both; neither belongs here):
    #   faded_sector 0.55 is Jupiter's SEB-FADE epoch feature -- a paled ~100-deg
    #     longitude sector on one belt. Importing a named Jovian epoch event onto a
    #     different planet would be accidental rather than authored.
    #   hue_jitter perturbs per-band values on exactly the band-lottery axis the
    #     value_contrast note above exists to defend against.
    "faded_sector": 0.0,
    "hue_jitter": 0.0,
}

# Jupiter-rate rotation, so the jet profile stays close to warm's in character.
# equatorial_width 0.14 is narrower than warm's and far narrower than
# cobalt_gale's 0.33: the equator is NOT the subject here and must not become one.
# polar_decay 0.6 keeps mid-latitude jets alive all the way out, which the
# alternation composition needs -- a high polar_decay (cobalt_gale's 0.9) would
# quiet exactly the mid-to-high latitudes where the zone/belt pairs have to keep
# reading.
JETS = {
    "strength": 1.0,
    "equatorial_speed": 2.4,
    "equatorial_width": 0.14,
    "polar_decay": 0.6,
    # No authored local jet or hero bracket: warm's are seated for a GRS at
    # -24 deg, and this preset has no hero at all.
    "local_jet_speed": 0.0,
    "hero_bracket_north": 0.0,
    "hero_bracket_south": 0.0,
}

# belt_boost 2.2 is ON-PREMISE rather than inherited: the belts are the convecting,
# churning half of this planet, so the tracer forcing should favour them the same
# way the vorticity injection does. replenish_rate 0.5 is deliberately ABOVE warm's
# 0.4 and is the specific guard against the "frosted glass" failure -- quiescent
# zones reading as smooth color patches. This preset makes zones laminar in
# VORTICITY on purpose, so it leans harder than usual on tracer replenishment plus
# detail.zone_texture to keep them glassy rather than blank.
TURBULENCE = {
    # 1.15 / 1.8, down from a round-1 1.4 / 2.2. Same reasoning as the vort_inject
    # cut above: the BELTS composition only pays off if the zones actually read as
    # the quiet half, and at round-1 values the churn was dominant across most of
    # the southern hemisphere. replenish_rate is deliberately NOT cut with them --
    # see below.
    "intensity": 1.15,
    "shear_coupling": 0.75,
    "scale": 6.0,
    "belt_boost": 1.8,
    "replenish_rate": 0.5,
    "kh_amplitude": 0.8,
    "kh_wavenumber": 20,
}

# NO HERO. The composition is the alternation, expressed in one field.
#
# hero_count 0 makes every hero_* lever inert, which is exactly why they are pinned
# rather than left inherited: warm bakes a full Great Red Spot (hero_strength 1.9,
# radius 0.108, solid core, emergence, rim warp, wake, companions) that sits inert
# in the emitted JSON but becomes a real GRS the moment anybody raises hero_count to
# debug something. Same set the sibling ember_dwarf and cobalt_gale scripts pin.
#
# The two populations that ARE the subject, and which half of the planet each lands
# on -- verified in sim/vortices.py rather than assumed, because an earlier draft of
# this preset had them backwards:
#
#   oval_density  -> WHITE OVALS, anticyclones in ZONES, brightness +0.22.
#   barge_density -> BROWN BARGES, weak cyclones in BELTS, brightness -0.28.
#   small_density -> speckle in both, but SIGNED: dark in belts, bright in zones.
#
# So the pale ovals live in the smooth zones and the dark cigars in the churned
# belts. That is Jupiter's own anatomy, and it is what makes the alternation read:
# each half of the planet gets furniture of its own polarity.
#
# barge_density is therefore raised, not zeroed -- unusually for a non-Jovian
# preset, dark cigars in churning belts are exactly on-premise here.
#
# NOTE there is no oval SIZE lever: radius is hardcoded (vortices.py) and biased
# small. Legibility comes from stamp_contrast and the palette, not from size.
STORMS = {
    "hero_count": 0,
    "hero_latitude": None,
    "hero_longitude": None,
    # Amplitude-side hero levers, pinned for the reason stated above.
    "hero_strength": 1.0,
    "hero_radius": 0.1,
    "wake_turbulence": 1.8,
    "companion_brightness": 0.32,
    "hero_companions": 0,
    "hero_solid_core": 0.0,
    "hero_emergence": 0.0,
    "hero_rim_tint": 0.0,
    "hero_rim_warp": 0.0,
    "hero_wake_detail": 0.0,
    "hero_mottle": 0.0,
    "hero_tint_var": 0.0,
    "hero_taper": 0.0,
    "hero_flow_aspect": 1.0,
    "hero_aspect": 1.0,
    # The two subject populations.
    "oval_density": 3.5,
    "oval_solid_core": 1.0,
    "barge_density": 3.0,
    "small_density": 1.5,
    # pearls_count is Jupiter's "string of pearls" -- a NAMED Jovian feature, so it
    # goes to 0 on the same argument as faded_sector. accent/outbreak would each be
    # a focal feature, and the composition is deliberately focal-feature-free.
    "pearls_count": 0,
    "accent_count": 0,
    "accent_latitude": None,
    "outbreak_count": 0,
    # Mergers are generic 2-D vortex physics rather than a Jovian import, and on a
    # swarm planet they are thematically right: the population should visibly
    # coarsen over a long development run.
    "merge_rate": 0.25,
    "stamp_contrast": 2.6,
}

# Festoons are a wave train on a band edge -- Jupiter's are the dark scalloped
# clouds along the NEB's southern boundary. On a planet whose subject IS the
# zone/belt boundary they earn their keep, so the lever is kept rather than zeroed;
# festoon_hero_strength goes to 0 with the hero (warm bakes 1.6, a hero lever that
# would otherwise sit live at hero_count 0).
WAVES = {
    "festoon_strength": 1.2,
    "festoon_wavenumber": 18,
    "festoon_hero_strength": 0.0,
    "ribbon_strength": 0.0,
}

# zone_texture is LOAD-BEARING on this preset, not decorative. It paints
# flow-folded luminance structure inside zones, gated by (1 - belt_mask) -- i.e.
# exactly the half of the planet the BELTS inject mask leaves laminar. It is what
# makes the zones read as GLASSY rather than BLANK, which is the difference between
# the intended look and the "frosted glass" defect. belt_texture is raised to match
# the churn on the other side.
DETAIL = {
    "intensity": 0.95,
    "spread": 0.40,
    "zone_texture": 0.95,
    "belt_texture": 1.3,
    "belt_texture_fine": 1.2,
    "mottle": 1.0,
    "cellular_amount": 0.8,
    "intermittency": 0.55,
    "polar_filaments": 1.0,
    "polar_stipple": 0.6,
    # All four hero-keyed detail levers pinned off with the hero.
    "hero_calm": 0.0,
    "hero_spiral": 0.0,
    "hero_collar_wrap": 0.0,
    "hero_wake_braid": 0.0,
}

APPEARANCE = {
    # 0.95 / 1.05, down from a round-1 1.05 / 1.25. At the round-1 values the
    # planet read as saturated INK rather than as pigment in an atmosphere -- vivid,
    # but synthetic next to the other eight presets. The cut is small and it is
    # bounded on the other side: chroma_scale is the main thing buying back the
    # saturation methane's weak visible absorption takes away, so dropping it
    # further would land on exactly the khaki the palette was authored to escape.
    # The G - R >= 0.08 guard in build() is the backstop.
    "contrast": 0.95,
    # chroma_scale is the Oklab (perceptual) multiplier and the recommended lever;
    # `saturation` is an sRGB luma mix, left at 1.0.
    "chroma_scale": 1.05,
    "saturation": 1.0,
    # Haze is a GLOBAL WASH, so its color fights the premise directly. A pale
    # sulfur-yellow haze (thematically tempting, and what an earlier draft proposed)
    # would pull the entire planet back toward the khaki this preset exists to
    # avoid. Kept light and hue-neutral-to-green.
    "haze_amount": 0.05,
    "haze_color": (0.72, 0.78, 0.62),
    "chroma_variance": 0.30,
    "hue_variance": 0.16,
    # BOTH inherited chroma levers off, and the chroma_aging case is ANALYTIC
    # rather than a matter of taste.
    #
    # chroma_aging (warm bakes 0.35): its pfield description says "chroma-only --
    # the latitude palette's HUE is untouched". THAT DESCRIPTION IS STALE. The
    # shader (render/kernels/derive.comp) applies a FIXED ADDITIVE PUSH along +a/+b
    # in Oklab -- `lab.yz += u_chroma_aging * chromo * 0.28 * normalize(vec2(0.6,
    # 0.8))` -- and its own comment says why ("hence the directional hue push").
    # It is gated on DARK material, i.e. exactly this deck. Computed on the stops
    # above at 0.35, the two dark rows rotate 45-58 degrees of hue and CLIP THE
    # GREEN CHANNEL TO ZERO, rendering red/orange. On a preset whose premise and
    # whose name are both green, that is disqualifying, and it is predictable
    # analytically rather than something to discover in an A/B.
    #
    # detail_chroma (warm bakes 0.6) drags dark decks red; this deck is dark AND
    # green.
    "chroma_aging": 0.0,
    "detail_chroma": 0.0,
    # Polar treatment. warm's inherited slate-blue tint fights an olive world, so
    # the tint is re-keyed to a cold green-slate. Read this as cosmetic hardening
    # only: the real defense against a flat bright cap is the clamped POLAR rows
    # above. On ember_dwarf, strengthening the tint made the cap LOOK fixed at
    # sim-res 1024 while both real causes were still live, and that false
    # confirmation cost two rounds.
    "polar_tint_strength": 0.65,
    "polar_tint_start_lat": 52.0,
    "polar_tint_color": (0.24, 0.30, 0.22),
    "polar_canvas_value": 0.6,
}

POLES = {"strength": 1.1, "field_density": 1.0}

# ZERO emission across the board, pinned against warm's inherited thermal 0.35 /
# lightning 0.15.
#
# thermal: at Jupiter temperatures the object's thermal peak is far into the
# infrared, so in the visible it is seen ENTIRELY in reflected light.
#
# lightning is the interesting one, because the PREMISE for keeping it was
# actually correct: derive.comp gates lightning on the same belt_mask channel the
# inject mask uses, so it would land in the churning belts, which is exactly right
# for this planet. It is pinned off anyway for a workflow reason -- lightning
# writes only to out_emission, which the Color preview never composites, so it is
# INVISIBLE to every step of the iteration loop used to author this preset
# (preview_globe returns the color map; probe_lut_usage neutralizes appearance).
# Shipping an art-direction value that nobody in this workflow can see is not a
# decision. A later Blender-facing pass could revisit it deliberately.
EMISSION = {
    "thermal_strength": 0.0,
    "lightning_strength": 0.0,
    "aurora_strength": 0.0,
}


def _stops(
    spec: list[tuple[float, tuple[float, float, float]]],
) -> list[GradientStop]:
    return [GradientStop(pos=p, color=list(c)) for p, c in spec]


def build() -> None:
    p = load_factory_preset("gas_giant_warm")
    # Captured BEFORE p.bands is replaced, so the "cleared warm's template" assert
    # needs no second load from disk.
    warm_had_template = p.bands.template is not None
    # A distinct seed, so this preset's band/storm/turbulence draws are its own
    # rather than warm's 4201 replayed through a different tuning. warm,
    # jupiter_vorticity, neptune and ember_dwarf all replay 4201, which already
    # makes four presets sharing the same substreams. 3216 is a nod to sulfur
    # (S-32, Z-16).
    p.seed = 3216
    p.sim = p.sim.model_copy(update=SIM)
    p.solver = p.solver.model_copy(update=SOLVER)
    p.bands = p.bands.model_copy(update=BANDS)
    p.jets = p.jets.model_copy(update=JETS)
    p.turbulence = p.turbulence.model_copy(update=TURBULENCE)
    p.storms = p.storms.model_copy(update=STORMS)
    p.waves = p.waves.model_copy(update=WAVES)
    p.detail = p.detail.model_copy(update=DETAIL)
    p.poles = p.poles.model_copy(update={
        "north": p.poles.north.model_copy(update=POLES),
        "south": p.poles.south.model_copy(update=POLES),
    })
    p.emission = p.emission.model_copy(update=EMISSION)
    # Rows MUST be mirrored across the equator. bake_rows blends on SIGNED latitude
    # (-90..+90) and CLAMPS outside the outermost anchor, so a northern-only ladder
    # (0/32/56/78) silently paints the ENTIRE southern hemisphere with the row at
    # 0.0 -- here the brightest, crest-capable EQUATOR row, right over the south
    # pole. On ember_dwarf that produced a bright orange south cap that survived
    # every dynamics-side fix and was misread for rounds as a band-value lottery.
    # `latitude` is signed, not an absolute value.
    rows = [
        PaletteRow(latitude=-78.0, stops=_stops(POLAR)),
        PaletteRow(latitude=-56.0, stops=_stops(SUBPOLAR)),
        PaletteRow(latitude=-32.0, stops=_stops(MIDLAT)),
        PaletteRow(latitude=0.0, stops=_stops(EQUATOR)),
        PaletteRow(latitude=32.0, stops=_stops(MIDLAT)),
        PaletteRow(latitude=56.0, stops=_stops(SUBPOLAR)),
        PaletteRow(latitude=78.0, stops=_stops(POLAR)),
    ]
    p.appearance = p.appearance.model_copy(
        update={**APPEARANCE, "palette_rows": rows, "storm_tints": _stops(STORM_TINTS)}
    )

    # save_preset only writes the ENVELOPE name; without this the inner params.name
    # stays "gas_giant_warm" (a wart neptune.json also carries).
    p.name = "green_giant"

    out = PRESETS_DIR / "green_giant.json"
    save_preset(p, out, name="green_giant")
    # save_preset does NOT re-validate; load_preset does. Prove in-bounds, and prove
    # the load-bearing claims this file's docstring makes.
    r = load_preset(out)
    assert r.seed == 3216
    assert r.sim.resolution == 4096

    # --- the composition -------------------------------------------------------
    assert r.storms.hero_count == 0
    # Both subject populations must be live, and the ovals must not be swamped by
    # speckle: ember_dwarf runs small_density 4.0, which is a different (fine,
    # uniform) texture from the countable ovals wanted here.
    assert r.storms.oval_density > 2.5, "no white-oval population in the zones"
    assert r.storms.barge_density > 2.0, "no dark-cigar population in the belts"
    assert r.storms.small_density < 2.0, r.storms.small_density

    # --- the mask, and the amplitude that has to move with it ------------------
    # The solver MODE is asserted first, and it is not a formality: the whole
    # composition rests on vort_inject_mask, which is a NO-OP in kinematic mode.
    # The mode is inherited from gas_giant_warm rather than set here, so nothing
    # else in this file would notice it changing -- the same class of silent
    # inheritance drift the coriolis_f0 == 3.0 assert exists to catch.
    assert r.solver.type == "vorticity", r.solver.type
    assert r.solver.vort_inject_mask == "belts"
    # A mask is a MULTIPLIER. belt_mask covers ~47.6% of latitudes against
    # shear_norm's ~3.9%, so carrying warm's 1.8 across would have applied ~4.9x
    # warm's integrated forcing (1.8 * 0.475 / 0.173) -- nearer GLOBAL, which the
    # project record says dissolves the planet. This assert is the one that would
    # catch a future "restore the inherited value" edit.
    assert r.solver.vort_inject < 1.0, (
        f"vort_inject={r.solver.vort_inject} is un-retuned for the BELTS mask"
    )

    # --- field: the premise moves L_d, and ONLY L_d ----------------------------
    # Asserted EQUAL to warm deliberately: this planet rotates at Jupiter's rate,
    # and the enrichment premise motivates the deformation radius alone. An earlier
    # draft moved f0 to 2.8 and then asserted it lay "strictly between" warm's 3.0
    # and ember_dwarf's 5.5 -- an assert that fails on its own value.
    assert r.solver.coriolis_f0 == 3.0
    assert 0.11 < r.solver.deformation_radius < 0.18, r.solver.deformation_radius

    # --- bands: seeded, EVEN, and actually alternating -------------------------
    assert warm_had_template and r.bands.template is None, "warm's Jovian template survived"
    assert r.bands.count % 2 == 0, f"count {r.bands.count} is odd"
    assert r.bands.value_contrast <= 1.3, (
        f"value_contrast={r.bands.value_contrast}: at 1.9 the layout collapses to a "
        "binary square wave and pins a wide polar band at exactly 1.000"
    )
    # The seeded path has NO alternation validator (the model has one, but it only
    # runs on the template path), and an odd count breaks alternation on ~half of
    # all seeds. Rather than trusting the count parity alone, run the real layout
    # and assert the CONSEQUENCE -- this also catches a future seed change.
    layout = generate_bands(r.seed, r.bands)
    flags = list(layout.is_belt)
    assert len(flags) == r.bands.count
    assert all(a != b for a, b in zip(flags, flags[1:], strict=False)), (
        f"zone/belt identity does not alternate at seed {r.seed}: {flags}"
    )

    # --- reflected light only --------------------------------------------------
    assert r.emission.thermal_strength == 0.0
    assert r.emission.lightning_strength == 0.0

    # --- inheritance pins ------------------------------------------------------
    assert r.bands.faded_sector == 0.0       # Jupiter's SEB-fade epoch feature
    assert r.bands.hue_jitter == 0.0
    assert r.storms.pearls_count == 0        # Jupiter's "string of pearls"
    assert r.storms.hero_latitude is None    # warm seats a GRS at -24
    assert r.waves.festoon_hero_strength == 0.0
    assert r.detail.hero_wake_braid == 0.0
    assert r.appearance.chroma_aging == 0.0  # a HUE push on dark material; clips green
    assert r.appearance.detail_chroma == 0.0

    # --- palette structure -----------------------------------------------------
    assert len(r.appearance.palette_rows) == 7
    lats = [row.latitude for row in r.appearance.palette_rows]
    # BOTH hemispheres must be anchored: bake_rows clamps outside the outermost
    # anchor, so a one-sided ladder paints a whole hemisphere with the row at 0.0.
    assert min(lats) <= -70.0 and max(lats) >= 70.0, lats

    # No polar row may reach a pale value -- what holds the unforced caps dark
    # independently of the band lottery and of sim resolution.
    #
    # Checks EVERY stop over its CHANNELS. GradientStop carries no ordering
    # validator (bake_lut sorts internally), so `stops[-1]` is list order rather
    # than the brightest stop; and `max(s.color for s in ...)` would compare colour
    # TUPLES lexicographically, i.e. pick the largest RED channel -- which on a
    # green ramp is not the brightest stop at all. Both are documented bugs in the
    # sibling cobalt_gale script.
    # TWO ceilings, because the unforced cap is WIDER than the >= 70 rows cover.
    # omega_force.comp zeroes relative vorticity from ~60-64 deg, and at this seed
    # the southernmost band spans -56.95..-90 with value 0.772 -- a 33-deg-wide
    # BRIGHT zone sitting almost entirely inside the unforced region. Between ~57
    # and ~78 deg bake_rows is dominated by SUBPOLAR, not POLAR, so guarding only
    # the +-78 rows leaves the equatorward half of that cap unguarded.
    #
    # The shipped render is fine -- measured zonal-mean luma runs 0.29 at -60 down
    # to 0.19 at -85, against 0.67 at the equator, so there is no bright ring
    # today. This is a REGRESSION guard, not a bug fix: without it a future
    # brightening of SUBPOLAR would repaint that cap with nothing objecting.
    # The subpolar ceiling is looser (0.66 vs 0.40) because those rows legitimately
    # carry mid-tones -- it only has to exclude a haze-crest-grade value.
    for row in r.appearance.palette_rows:
        if abs(row.latitude) >= 70.0:
            assert max(max(s.color) for s in row.stops) < 0.40, row.latitude
        elif abs(row.latitude) >= 55.0:
            assert max(max(s.color) for s in row.stops) < 0.66, row.latitude

    # Anti-frosting: a flat/pale ramp collapses structure to one pale colour. The
    # repo's own guard (tests/unit/test_presets.py::test_palette_has_value_contrast)
    # is hardcoded to three presets and does not cover this one, so it is asserted
    # here -- but SCOPED TO THE MID ROWS. Applying it to every row would contradict
    # the polar clamp directly: all channels <= 0.40 implies luma <= 0.40 < 0.5.
    # ember_dwarf's and cobalt_gale's polar rows measure 0.13 and 0.17.
    for row in r.appearance.palette_rows:
        if abs(row.latitude) <= 32.0:
            lumas = [0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
                     for c in (s.color for s in row.stops)]
            assert max(lumas) - min(lumas) > 0.5, (row.latitude, max(lumas) - min(lumas))

    # THE PRESET IS CALLED green_giant, AND THIS IS THE ASSERT THAT MAKES IT TRUE.
    # The first draft authored "olive" by eye and measured at hue 60-72 deg -- which
    # is yellow -- with a green-over-red margin of 0.005-0.06, an order of magnitude
    # below every other preset. Its brightest stop had R == G: zero green content.
    # Scoped to the upper half of the ramp because a near-black floor cannot carry
    # a large absolute channel separation, and to the mid rows because the polar
    # rows are deliberately desaturated.
    for row in r.appearance.palette_rows:
        if abs(row.latitude) <= 32.0:
            for s in (s for s in row.stops if s.pos >= 0.5):
                red, green, _ = s.color
                assert green - red >= 0.08, (row.latitude, s.pos, s.color)

    print(f"wrote + verified {out}", flush=True)


def main() -> None:
    build()


if __name__ == "__main__":
    main()

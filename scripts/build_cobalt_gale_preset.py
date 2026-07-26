"""Build the `cobalt_gale` factory preset — a scattering-blue hot Jupiter.

The library's second non-solar-system object, after `ember_dwarf`. A tidally
locked hot Jupiter in the manner of HD 189733b.

The premise is a MECHANISM CONTRAST with `neptune`, not just another blue:

  * **`neptune` is blue by ABSORPTION.** Methane eats the red end of the
    spectrum, so what returns to the eye is a bright blue ball.
  * **This is blue by RAYLEIGH SCATTERING** off a high-altitude MgSiO3
    (silicate) haze. Scattering goes as lambda^-4, so the haze throws back blue.
    HD 189733b is the planet this was actually measured on (Evans et al. 2013,
    HST secondary-eclipse photometry).

That difference inverts the tonal structure, which is the whole reason the two
presets are not one preset with a hue slider. The haze is the ONLY reflector
here, so **bright = thick silicate cloud** and **dark = clearer air**, seen down
into a hot, near-black, alkali-absorbing depth.

Emission is held at ZERO, deliberately and against the inherited warm value. At
roughly 1200 K this object glows in the INFRARED, not the visible, so every
bright pixel is reflected light. That is also what keeps it clear of
`ember_dwarf`, whose grammar is the opposite ("bright = a hole in the deck, with
hotter gas showing through").

**The composition is a JET, not a storm** — the first factory preset that is LED
by one. (NOT the first without a hero: `saturn_pale` also ships `hero_count=0`.
It merely lacks a focal feature, where this preset has a jet AS its subject.)
Tidally locked giants develop equatorial superrotation: one broad
prograde equatorial jet dominating rather than Jupiter's many alternating bands.
A hero storm would have made this read as the Great Red Spot in blue.

HD 189733b's rotation is tidally locked to its ~2.2-day orbit, i.e. FIVE TIMES
slower than Jupiter. That single fact drives most of the field tuning below and
it all points the same way: weaker Coriolis constraint (`coriolis_f0` down), a
larger deformation radius (bigger, fewer structures), and far fewer, much wider
bands. `ember_dwarf` is the mirror image of this preset in that respect -- a
2-5 hour rotator with f0 up, L_d down, count up.

Realism latitude, from the user and recorded so it is not "corrected" later:
"cool and plausible, absolute realism isn't required". The science above fixes
the PREMISE and the DIRECTION of every choice -- blue from scattering, bright
means cloud, no visible-light glow -- not the numbers. Contrast and chroma are
pushed past photometric defensibility where the globe looks better for it.

Accepted expressive limit: a tidally locked planet's signature features are
LONGITUDINAL (day/night contrast, the eastward-displaced hot spot).
`palette_rows` anchor on LATITUDE only and the renderer is full-lit by design,
so the color and the jet are in scope and a substellar hot spot is not.

Run: uv run python scripts/build_cobalt_gale_preset.py
"""
from __future__ import annotations

import statistics
from pathlib import Path

from gasgiant.params.model import BandTemplate, GradientStop, InjectMask, PaletteRow
from gasgiant.params.presets import load_factory_preset, load_preset, save_preset

PRESETS_DIR = Path("src/gasgiant/presets")

# Four latitude PROFILES (equator/midlat/subpolar/polar), mirrored into seven
# palette rows at build time -- see the comment on `rows` in build() for why the
# mirroring is mandatory rather than tidy.
#
# Every row climbs from a near-black indigo floor (clear air over an alkali-
# absorbing depth) through cobalt into a pale blue-white haze crest. What changes
# with latitude is HOW FAR UP IT GETS: haze production falls away from the
# equatorial upwelling, so the poleward rows are authored to stop at cold slate.
# That is what buys the equatorial jet its contrast -- from the palette's
# latitude structure rather than from a global contrast lever, which would have
# to flatten everything else to get it.
#
# The bright end stays faintly BLUE (lavender-white, never neutral white):
# Rayleigh scattering is what makes this planet visible at all, and a neutral
# white crest would quietly assert a grey cloud.
#
# The stop POSITIONS are measured, not guessed, and they are COUPLED to the field
# tuning below AND to sim resolution -- so RE-MEASURE if either moves, quoting the
# exact invocation (the tool defaults to sim-res 1024 with detail OFF, which is a
# materially different distribution):
#
#     uv run python scripts/probe_lut_usage.py --preset cobalt_gale \
#         --sim-res 4096 --keep-detail
#
# It swaps in a linear grayscale ramp so rendered luminance IS the palette index.
# MEASURED_HISTOGRAM below is the BASELINE to diff a re-measurement against --
# without it, "re-measure if either moves" has nothing to compare to.
#
# THE KNEE LADDER IS THE COMPOSITION, and it is set by measurement. The first
# build made this concrete: the equatorial jet did not read as the subject at all,
# and the reason was not a weak jet. Per-latitude index, sim-res 1024, detail off:
#
#     lat  +85    mean 0.843   p95 0.941      <- the BRIGHTEST place on the planet
#     lat  -40    mean 0.823   p95 0.905      <- a pale mid-latitude rail
#     lat   +0    mean 0.495   p95 0.731      <- the jet, sitting mid-ramp
#
# So the band skeleton sets the index and **the jet does not brighten its own
# latitude** -- it shapes flow, not tracer value. Authoring every row on the same
# knee ladder therefore hands the composition to wherever the seeded band lottery
# put its brightest zone, which was a southern mid-latitude rail, with the equator
# a middling cobalt and the north cap brighter still.
#
# The fix is the one the latitude rows exist for, and it is also the physical
# story: haze is THICKER at the equatorial upwelling, so the same tracer value
# must buy more cloud there. Hence EARLY knees at the equator (pale by ~0.75) and
# LATE knees poleward (the 0.82 rail at -40 lands mid-cobalt, not pale). The jet
# becomes the subject without touching the dynamics, and the mid-latitudes keep
# their structure instead of being flattened to get out of its way.
#
# The shipped baseline, both at `--keep-detail` (decile shares of the disc, in %).
# Diff a re-measurement against THIS, not against the ember_dwarf numbers.
MEASURED_HISTOGRAM = {
    #        0.0-0.1  .1-.2  .2-.3  .3-.4  .4-.5  .5-.6  .6-.7  .7-.8  .8-.9  .9-1
    1024: (0.0, 0.0, 5.2, 23.7, 15.4, 19.2, 18.7, 10.0, 4.8, 3.0),   # p50 0.540
    4096: (0.0, 0.2, 6.8, 22.9, 15.4, 17.3, 17.9, 11.0, 5.0, 3.5),   # p50 0.533
}
# The pair is also the evidence for NOT pinning the README render resolution (see
# SIM): above index 0.80 it is 7.8% -> 8.5%, only ~1.09x, where ember_dwarf's
# convection-driven premise moves ~1.55x.

EQUATOR = [   # the superrotation jet: EARLY knees, the only row reaching a crest
    (0.00, (0.035, 0.045, 0.105)),
    (0.25, (0.085, 0.130, 0.290)),
    (0.45, (0.190, 0.290, 0.545)),   # the jet's zonal MEAN (0.495) lands about here
    (0.60, (0.360, 0.470, 0.720)),
    (0.75, (0.620, 0.700, 0.870)),   # its p95 (0.731) lands here: pale crests
    (0.90, (0.860, 0.900, 0.975)),
    (1.00, (0.950, 0.965, 1.000)),
]
MIDLAT = [    # the deck proper: LATE knees, so no mid-latitude rail can go pale
    (0.00, (0.028, 0.034, 0.085)),
    (0.30, (0.062, 0.092, 0.215)),
    (0.55, (0.115, 0.180, 0.385)),
    (0.78, (0.205, 0.290, 0.520)),   # the measured 0.82 rail at -40: mid cobalt
    (0.92, (0.400, 0.490, 0.690)),
    (1.00, (0.620, 0.690, 0.845)),
]
SUBPOLAR = [  # cooling: haze thins, the bright end never reaches pale
    (0.00, (0.022, 0.027, 0.072)),
    (0.34, (0.048, 0.070, 0.175)),
    (0.60, (0.090, 0.135, 0.300)),
    (0.84, (0.150, 0.205, 0.395)),
    (1.00, (0.265, 0.330, 0.510)),
]
# The polar rows exist to CAP THE CAPS, and this is a borrowed fix rather than a
# guessed one. Poleward of ~65-70 deg the field is essentially unforced (the
# shear injection mask plus polar_decay leave almost no shear to inject against),
# so whatever value the outermost seeded bands happen to hold gets painted nearly
# flat across the cap. On `ember_dwarf` that produced blazing gold laminar streaks
# over the south pole, and it survived several dynamics-side fixes AND a
# strengthened polar tint -- because a tint is a hue push and the defect is
# LUMINANCE. Clamping the bright end of the polar rows is the only fix that is
# independent of the band lottery and of the sim resolution.
#
# This preset is MORE exposed to that failure than ember_dwarf was, not less: a
# high `polar_decay` is exactly what makes the jet-led composition work, and it is
# also what leaves the caps unforced.
#
# It was not a hypothetical here. The very first build measured the NORTH CAP as
# the brightest place on the planet -- lat +85 index mean 0.843, p95 0.941, versus
# 0.495 at the equator -- and it is invisible in the render only because these rows
# clamp it. The clamp is load-bearing on this preset, not belt-and-braces.
POLAR = [
    (0.00, (0.020, 0.024, 0.062)),
    (0.40, (0.038, 0.055, 0.135)),
    (0.70, (0.070, 0.100, 0.225)),
    (0.90, (0.105, 0.145, 0.290)),
    (1.00, (0.150, 0.195, 0.340)),
]

# Storm LUT. The vortices here are haze CONCENTRATIONS -- a coherent eddy piles
# cloud up and reads bright -- so the ramp runs from a cleared dark core to a
# bright crest, the same direction as the palette rows. (Contrast `ember_dwarf`,
# where a storm is a clearing and the bright end is exposed hot gas.)
STORM_TINTS = [
    (0.00, (0.035, 0.045, 0.105)),
    (0.55, (0.230, 0.320, 0.560)),
    (1.00, (0.880, 0.915, 0.980)),
]

SIM = {
    "dev_steps": 800,
    # Pinned, not inherited: 4096 matches the other modern presets.
    #
    # This preset deliberately does NOT get a SIM_RES_OVERRIDES entry in
    # render_readme_examples.py, and that is a measured decision rather than an
    # omission. Resolution sensitivity, LUT index above 0.80, detail on:
    #
    #     cobalt_gale   7.8% (1024) -> 8.5% (4096)   ~1.09x
    #     ember_dwarf  13.0% (1024) -> 20.2% (4096)  ~1.55x  <- hence ITS pin
    #
    # The reason is structural: ember_dwarf's premise is the FRACTION OF THE DISC
    # torn open by convective excursions, which is resolution-dependent, whereas
    # this preset's composition is an AUTHORED BAND TEMPLATE with verbatim values
    # -- resolution-independent by construction. Confirmed visually too: the
    # reduced grid renders the same planet with coarser filaments, not a different
    # one. So the README image is faithful at the default grid and there is no need
    # to make the doc-asset generator 16x more expensive.
    "resolution": 4096,
}

# Slow rotator (tidally locked, ~2.2 d), so everything gets BIGGER and there is
# less of it -- the exact inverse of ember_dwarf's fast-rotator tuning.
#
#   coriolis_f0 3.0 -> 2.0   weaker geostrophic constraint: fewer, wider jets
#   L_d         0.18 -> 0.26 larger deformation radius: fewer, larger eddies
#
# Raising L_d is the lever that most risks the inverse cascade piling every eddy
# into one gravest-mode blob, so vort_psi_drag carries the scale control -- it is
# the scale-SELECTIVE hypofriction (acting on the eddy streamfunction) rather than
# the flat-in-k vort_eddy_drag, which over-flattens. warm ships 0.06 and can
# afford to because its L_d is small; at 0.26 it cannot be left there.
SOLVER = {
    "coriolis_f0": 2.0,
    "deformation_radius": 0.26,
    "vort_inject": 1.5,
    # Coarser than warm's 2.5: this planet's structures are large, and fine
    # injection at a large L_d reads as noise rather than as chevrons.
    "vort_inject_scale": 2.0,
    # SHEAR-masked, never global. Unmasked injection is a known failure -- it
    # churns the band structure away, which is what made gas_giant_warm adopt the
    # mask in the first place. Here the mask does double duty: the shear IS the
    # jet flank, so confining eddies to it is what puts chevrons where they belong.
    "vort_inject_mask": InjectMask.SHEAR,
    "vort_hypervisc": 0.5,
    "vort_psi_drag": 0.45,
}

# An AUTHORED band template -- not warm's, and not the seeded path either.
#
# This reverses the preset's first attempt, so the reasoning is worth keeping.
# The argument against the template path is that the shipped template IS
# JUPITER'S band layout, and a 12-band Jovian skeleton is the most Jupiter-like
# thing this preset could inherit while trying not to look like Jupiter. (12, not
# 22: warm's `bands.count` is 22 but that value is INERT on the template path --
# exactly the conflation the BANDS block below warns about, and one this comment
# originally made.) That argument is against inheriting WARM'S template. It is not an argument against
# authoring one, and the seeded path turned out to be unable to deliver the
# composition:
#
#   * Bands are laid out in SIN-LATITUDE space with jittered widths
#     (`sim/bands.py`), so at count 10 the near-equator bands are only ~11 deg
#     wide and their EDGES fall wherever the seed puts them.
#   * Measured consequence: rendered luma +5 0.577, 0 0.436, -5 0.343, -10 0.607.
#     A band edge landed on the equator and split the jet into TWO bright lanes
#     with a dim one down the middle. The equator was already dominant over the
#     mid-latitudes by then (0.58-0.61 versus 0.38) -- it just was not a jet.
#   * No amount of seed roulette fixes that reliably. A composition whose whole
#     premise is one broad equatorial jet cannot be left to a width lottery.
#
# The template path also hands over the band VALUES verbatim, which is exactly
# what is wanted here: the values ARE the LUT index, so the equatorial zone's
# brightness becomes a number in this file rather than an emergent accident.
#
# Geometry: 9 bands. The count is forced, not chosen. Identity is derived as
# `values < median(values)` and must STRICTLY ALTERNATE, so a band CENTRED on the
# equator has to sit at an odd position -- which needs count = 1 (mod 4). 9 gives
# a centred band at position 5, with both polar caps landing in the
# zone-majority class so they are symmetric. (11 would put a BELT on the equator,
# i.e. a dark lane exactly where the jet belongs.)
BAND_TEMPLATE = {
    # +-13 deg equatorial zone: one broad bright lane, deliberately symmetric.
    # Everything else is slightly asymmetric on purpose -- real giants are not
    # mirror-symmetric, and a perfectly even skeleton reads mechanical (the same
    # failure that killed the festoon spiral-amplification idea).
    "edges_deg": [90.0, 76.0, 53.0, 31.0, 13.0, -13.0, -29.0, -50.0, -76.0, -90.0],
    # Color index per band, used VERBATIM -- so the equatorial zone's brightness
    # is a number in this file rather than an emergent accident.
    #
    # 0.74, down from 0.88. At 0.88 the measured jet ran mean 0.917 / p95 0.979
    # with max CLIPPING at 1.000, and it rendered as flat overexposed white: the
    # texture inside it collapsed to speckle and the planet lost the cobalt
    # identity that is the entire premise. Being the brightest thing on the planet
    # and being blown out are different requirements, and the palette's EQUATOR row
    # already reaches a pale crest by index 0.75 -- so the band value only has to
    # REACH that knee, not overshoot it.
    #
    # The belts came up (0.22-0.26 -> 0.29-0.32) at the same time: with the jet
    # saturated, the deck had gone so dark that the planet read as a black ball
    # with one white stripe.
    "values": [0.53, 0.30, 0.62, 0.32, 0.74, 0.31, 0.60, 0.29, 0.53],
    "heights": [0.48, 0.30, 0.55, 0.34, 0.86, 0.32, 0.52, 0.28, 0.48],
}

BANDS = {
    # value_contrast / count / width_jitter / width_tail / hue_jitter are all
    # INERT on the template path (values and edges are verbatim). They are pinned
    # to inert values rather than left carrying warm's numbers, so nothing reads
    # as a tuning decision that cannot have any effect. Warp, edge softness,
    # edge diversity and detail noise DO still apply.
    "count": 9,
    "value_contrast": 1.0,
    "width_jitter": 0.0,
    "width_tail": 0.0,
    "edge_softness": 0.02,
    "detail_amount": 0.14,
    # Collapses banding contrast poleward of ~45 deg toward mottle. Tried at 0.8
    # to calm a cap measured at index 0.94 (p95) and reverted to ~warm's 0.6: with
    # an AUTHORED template the cap band values are set directly, so there is no
    # band lottery left to calm, and 0.8 was flattening the mid-latitude deck into
    # a featureless surround for the jet.
    "contrast_envelope": 0.62,
    "edge_diversity": 0.5,
    "variance_amount": 0.22,
    # faded_sector is warm's 0.55 and is LIVE on the template path (the
    # faded-sector geometry is picked in _finish_layout, which both paths run).
    # It is Jupiter's SEB-FADE epoch feature -- a paled ~100-deg longitude sector
    # on one belt -- so inheriting a named Jovian epoch event on an exoplanet
    # would be accidental rather than authored.
    "faded_sector": 0.0,
    "hue_jitter": 0.0,
}

# The superrotation jet -- the subject of the preset.
#
# Deliberately BROAD and FAST: 0.33 rad is a ~19 deg half-width, versus
# ember_dwarf's 0.16 and the 0.12 model default. polar_decay 0.9 (warm 0.648 --
# 0.5 is the MODEL DEFAULT, not warm's value)
# quiets the mid-latitude jets so the equator is unmistakably dominant rather
# than merely the strongest of many.
#
# The polar_decay cost is real and paid for in POLAR above: quiet mid-to-high
# latitudes mean unforced caps.
JETS = {
    "strength": 1.0,
    "equatorial_speed": 3.2,
    "equatorial_width": 0.33,
    "polar_decay": 0.9,
    # No authored local jet or hero bracket: warm's are seated for a GRS at
    # -24 deg, and this preset has no hero at all.
    "local_jet_speed": 0.0,
    "hero_bracket_north": 0.0,
    "hero_bracket_south": 0.0,
}

# The haze streaks are made HERE, not by the palette: holding value_contrast
# moderate leaves the band skeleton mid-ramp, so what lifts a pixel to a bright
# crest is advected, stretched cloud.
#
# shear_coupling up to 0.9 (warm 0.7) is the lever that matters most for this
# composition -- it ties the tracer forcing to the local shear, and the dominant
# shear is the equatorial jet's flanks, so the texture gets DRAWN OUT along the
# jet instead of sitting in it as blobs. scale 5.0 (coarser than warm's 6.0) for
# the same reason ember_dwarf went coarse: fewer, longer features read as
# structure where finer noise reads as speckle.
TURBULENCE = {
    "intensity": 1.2,
    "shear_coupling": 0.9,
    "scale": 5.0,
    "replenish_rate": 0.4,
    "belt_boost": 1.2,
    "kh_amplitude": 0.8,
}

# NO HERO. This is the composition decision, expressed in one field.
#
# hero_count 0 makes every hero_* lever inert, which is exactly why they are
# pinned rather than left inherited: warm bakes ten of them (a -24 deg GRS with
# solid core, emergence, rim warp, wake), CLAUDE.md records hero_wake_braid as
# HELD pending a visual review, and an inert-but-inherited GRS would silently
# become a real one the moment anybody raised hero_count to debug something.
#
# The vortex population that remains is modest and large-scale: a slow rotator
# with a large L_d supports few, big eddies. small_density is RESET to 0.0 (warm
# ships 3.5) because fine speckle is the wrong texture at this deformation
# radius, and pearls/barges are Jovian belt furniture.
STORMS = {
    "hero_count": 0,
    "hero_latitude": None,
    "hero_longitude": None,
    # The AMPLITUDE-side hero levers, pinned for the reason stated above -- these
    # four were initially left inherited, which made the claim above broader than
    # the code: warm's GRS values (hero_strength 1.9, hero_radius 0.108,
    # wake_turbulence 3.2, companion_brightness 0.55) were still sitting in the
    # emitted JSON, inert at hero_count 0 but ready to become a real Great Red
    # Spot the moment anybody raised the count. hero_strength especially: it is
    # the single most consequential hero lever, and inheriting it by omission is
    # not a decision. Same set the sibling ember_dwarf script pins.
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
    # Mid-latitude eddies: few, large, bright (haze concentrations).
    "oval_density": 1.6,
    "oval_solid_core": 1.0,
    "barge_density": 0.0,
    "pearls_count": 0,
    "small_density": 0.0,
    "outbreak_count": 0,
    "accent_count": 0,
    "accent_latitude": None,
    "stamp_contrast": 1.8,
    "merge_rate": 0.0,
}

# Festoons are a wave train on a band edge -- Jupiter's are the dark scalloped
# clouds along the NEB's southern boundary. That is the right shape for the
# chevron-like disturbances GCMs show on a superrotation jet's flanks, so the
# lever is kept (moderately) rather than zeroed. festoon_hero_strength goes to 0
# with the hero.
WAVES = {
    "festoon_strength": 1.4,
    "festoon_hero_strength": 0.0,
    "ribbon_strength": 0.0,
}

DETAIL = {
    "intensity": 0.95,
    # Even coverage rather than band-gated: the repo's own lesson from shipping
    # this lever is that even texture density beats selectivity on gas giants.
    "spread": 0.42,
    "belt_texture": 1.1,
    "belt_texture_fine": 1.2,
    "mottle": 1.0,
    "cellular_amount": 0.9,
    "intermittency": 0.6,
    "polar_filaments": 1.0,
    "polar_stipple": 0.6,
    # All four hero-keyed detail levers pinned off with the hero. hero_wake_braid
    # is the highest-risk of them: CLAUDE.md records the warm bake as HELD at
    # ~1.0 pending a visual review, so an upstream bump is expected.
    "hero_calm": 0.0,
    "hero_spiral": 0.0,
    "hero_collar_wrap": 0.0,
    "hero_wake_braid": 0.0,
}

APPEARANCE = {
    # Above warm's 0.8, and this is one of the places the "cool over strictly
    # photometric" latitude is being spent: the honest reflected-light contrast
    # of a hazy planet is low, and a dark planet with low contrast reads muddy.
    "contrast": 1.12,
    # chroma_scale is the Oklab (perceptual) multiplier and the recommended
    # lever; `saturation` is an sRGB luma mix, left at 1.0.
    "saturation": 1.0,
    "chroma_scale": 1.2,
    # The haze lever is thematically the right one here -- this planet's premise
    # IS a high scattering haze -- but it is a global wash, so it stays light and
    # BLUE-WHITE. warm's (0.85, 0.78, 0.62) is a warm Jovian ammonia haze and
    # would put a yellow film over the entire point of the preset.
    "haze_amount": 0.07,
    "haze_color": (0.72, 0.80, 0.95),
    "chroma_variance": 0.35,
    "hue_variance": 0.18,
    # Both inherited chroma levers off, and the reasons are NOT symmetric.
    #
    # chroma_aging (warm bakes 0.35) is the measured blue-killer. It ties
    # saturation to the freshness tracer and deposits "reddish-brown chromophore"
    # in aged air. Measured on ember_dwarf, mean on-disc RGB across 0.35 -> 0.0:
    #   (0.308, 0.001, 0.061) -> (0.210, 0.004, 0.183)
    # -- it COLLAPSES BLUE, tripling it when removed. On a preset whose entire
    # premise is a scattering blue this is the most dangerous single value in the
    # inheritance, and it is also simply the wrong pigment: Jovian chromophores
    # are not present on a 1200 K silicate-haze planet.
    #
    # detail_chroma (warm bakes 0.6) is nearly inert by comparison (~1% on
    # ember_dwarf) because it keys on the SYNTHESIZED DETAIL field, not on scene
    # brightness. It is held at 0 for a specific reason rather than by analogy:
    # its strong side (1.0x vs 0.3x) pushes BRIGHT excursions COOL, and the bright
    # excursions here are the pale haze crests, which are already the coolest
    # thing on the planet. Pushing them cooler would drive them to cyan.
    "detail_chroma": 0.0,
    "chroma_aging": 0.0,
    # Polar treatment. Unusually, warm's inherited polar values SUIT this preset
    # rather than fighting it: polar_tint_color is already a slate blue and
    # polar_canvas_value deepens the cap toward a dark blue-TEAL floor. That teal
    # is exactly what ember_dwarf had to zero (it fought a magenta world) and it
    # is harmonious with a cobalt one -- so it is kept, and slightly raised.
    #
    # Read this as cosmetic hardening only. The real defense against a flat
    # bright cap is the clamped POLAR palette rows above: a tint is a hue push and
    # the failure mode is luminance. On ember_dwarf, strengthening the tint made
    # the cap look fixed at sim-res 1024 while both real causes were still live,
    # and that false confirmation cost two rounds.
    "polar_tint_strength": 0.7,
    "polar_tint_start_lat": 52.0,
    "polar_tint_color": (0.28, 0.36, 0.52),
    "polar_canvas_value": 0.9,
}

# Weak rotation means a weakly organised polar vortex, so the cyclone clusters
# are toned down from warm's 1.35/1.4 rather than inherited.
POLES = {"strength": 0.8, "field_density": 0.9}

# ZERO emission, pinned against warm's inherited thermal 0.35 / lightning 0.15.
# This is a load-bearing physical claim, not tidiness: at ~1200 K the object's
# thermal peak is in the near-infrared, so in the visible it is seen ENTIRELY in
# reflected light. It is also the cleanest separation from ember_dwarf, which
# re-keyed thermal up to 0.9 and made glow-through-the-gaps its whole point.
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
    # Captured BEFORE p.bands is replaced below, so the "not warm's template"
    # assert needs no second load from disk -- and so a warm without a template
    # would fail here on a clear AttributeError rather than inside the assert.
    warm_edges = p.bands.template.edges_deg
    # A distinct seed, so this preset's band/storm/turbulence draws are its own
    # rather than warm's 4201 replayed through a different tuning (ember_dwarf
    # kept 4201, which makes it the third preset drawing the same substreams).
    # 189733 is a nod to HD 189733b.
    p.seed = 189733
    p.sim = p.sim.model_copy(update=SIM)
    p.solver = p.solver.model_copy(update=SOLVER)
    p.bands = p.bands.model_copy(
        update={**BANDS, "template": BandTemplate(**BAND_TEMPLATE)}
    )
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
    # Rows MUST be mirrored across the equator. bake_rows blends on SIGNED
    # latitude (-90..+90) and CLAMPS outside the outermost anchor, so a
    # northern-only ladder (0/32/56/78) silently paints the ENTIRE southern
    # hemisphere with the row at 0.0 -- here the brightest, crest-capable EQUATOR
    # row, right over the south pole. On ember_dwarf that produced a bright orange
    # south cap that survived every dynamics-side fix and was misread for rounds
    # as a band-value seed lottery. `latitude` is signed, not an absolute value.
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

    # save_preset only writes the ENVELOPE name; without this the inner
    # params.name stays "gas_giant_warm" (a wart neptune.json also carries).
    p.name = "cobalt_gale"

    out = PRESETS_DIR / "cobalt_gale.json"
    save_preset(p, out, name="cobalt_gale")
    # save_preset does NOT re-validate; load_preset does. Prove in-bounds, and
    # prove the load-bearing claims this file's docstring makes.
    r = load_preset(out)
    assert r.seed == 189733
    assert r.sim.resolution == 4096
    # The composition: a jet, not a storm.
    assert r.storms.hero_count == 0
    assert r.jets.equatorial_speed == 3.2
    assert r.jets.equatorial_width == 0.33
    assert r.jets.polar_decay == 0.9
    # Slow rotator: the inverse of ember_dwarf's f0 5.5 / L_d 0.11.
    assert r.solver.coriolis_f0 == 2.0
    assert r.solver.deformation_radius == 0.26
    # Large L_d without scale-selective hypofriction piles the inverse cascade
    # into one gravest-mode blob; warm's 0.06 is only safe at warm's small L_d.
    assert r.solver.vort_psi_drag >= 0.4
    assert r.solver.vort_inject_mask == "shear"   # global churns the banding away
    # An AUTHORED template, and specifically NOT warm's Jovian one.
    t = r.bands.template
    assert t is not None
    assert t.edges_deg != warm_edges
    # 9 bands with a band CENTRED on the equator -- the geometric point of the
    # whole layout. Identity must strictly alternate, which forces count = 1 (mod 4)
    # for a centred band; at 11 a BELT lands on the equator instead.
    assert len(t.values) == 9
    assert t.edges_deg[4] == 13.0 and t.edges_deg[5] == -13.0, t.edges_deg
    # ...and the centred band must actually be a ZONE. This is the thing the
    # mod-4 rule exists to guarantee, so assert the CONSEQUENCE rather than the
    # rule: `len(values) % 4 == 1` would be a tautology under the length pin
    # above and could never fail on its own, whereas identity is derived from
    # the VALUES and a future retune could silently push the equatorial band
    # below the median and turn the jet into a dark lane.
    assert t.values[4] >= statistics.median(t.values), t.values
    # The equatorial zone must be the brightest band on the planet by a clear
    # margin -- but NOT by so much that it clips. 0.88 measured mean 0.917 / max
    # 1.000 and rendered as flat white; the upper bound is the real lesson.
    eq_value = t.values[4]
    assert eq_value == max(t.values)
    assert eq_value > sorted(t.values)[-2] * 1.15, t.values
    assert eq_value <= 0.80, f"{eq_value}: clips to flat white above ~0.8"
    # The two chroma levers, off for asymmetric reasons (see APPEARANCE).
    assert r.appearance.chroma_aging == 0.0
    assert r.appearance.detail_chroma == 0.0
    # Reflected light only -- warm bakes thermal 0.35 and lightning 0.15.
    assert r.emission.thermal_strength == 0.0
    assert r.emission.lightning_strength == 0.0
    # Inheritance pins actually landed.
    assert r.bands.faded_sector == 0.0        # Jupiter's SEB-fade epoch feature
    assert r.bands.hue_jitter == 0.0
    assert r.detail.hero_wake_braid == 0.0    # warm bake HELD upstream, expected to rise
    assert r.storms.hero_latitude is None     # warm seats a GRS at -24
    assert r.waves.festoon_hero_strength == 0.0
    # Palette structure.
    assert len(r.appearance.palette_rows) == 7
    lats = [row.latitude for row in r.appearance.palette_rows]
    # BOTH hemispheres must be anchored: bake_rows clamps outside the outermost
    # anchor, so a one-sided ladder paints a whole hemisphere with the row at 0.0.
    assert min(lats) <= -70.0 and max(lats) >= 70.0, lats
    # No polar row may reach a pale value. This is what holds the unforced caps
    # dark independently of the band lottery and of the sim resolution -- and this
    # preset is MORE exposed than ember_dwarf, because the high polar_decay that
    # makes the jet dominant is the same thing that leaves the caps unforced.
    # The 0.40 threshold is looser than ember_dwarf's 0.30 because a cold slate
    # blue at 0.34 is still unmistakably not a haze crest.
    #
    # Checks EVERY stop, not `stops[-1]`. GradientStop carries no ordering
    # validator (`params/model.py`; bake_lut sorts internally), so `stops[-1]` is
    # list order, not the brightest stop -- a polar ramp with an appended dark stop
    # or a non-monotone crest would sail through while the unforced cap still went
    # pale, i.e. the exact ember_dwarf defect this guard exists to catch. The
    # neighbouring knee-ladder guard was already written per-channel for the same
    # reason; this one was not, which made the file inconsistent with itself.
    for row in r.appearance.palette_rows:
        if abs(row.latitude) >= 70.0:
            assert max(max(s.color) for s in row.stops) < 0.40, row.latitude
    # The knee ladder IS the composition (see EQUATOR): the equator must reach a
    # pale crest EARLIER than the deck does, or the brightest zone of the seeded
    # band lottery takes over as the subject wherever it happens to land.
    #
    # Take max() over the CHANNELS, not over the colour tuples: `max(s.color ...)`
    # compares tuples LEXICOGRAPHICALLY, i.e. picks the largest RED channel rather
    # than the brightest stop. It happens to give the right answer on these two
    # ramps (both are monotone in R), so the guard would have passed for the wrong
    # reason and silently stopped meaning anything on a bluer ramp.
    by_lat = {row.latitude: row for row in r.appearance.palette_rows}
    eq_at_75 = max(max(s.color) for s in by_lat[0.0].stops if s.pos <= 0.75)
    mid_at_78 = max(max(s.color) for s in by_lat[-32.0].stops if s.pos <= 0.78)
    assert eq_at_75 > mid_at_78 * 1.5, (eq_at_75, mid_at_78)
    print(f"wrote + verified {out}", flush=True)


def main() -> None:
    build()


if __name__ == "__main__":
    main()

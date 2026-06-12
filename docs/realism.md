# Reference-closeness metrics (v1.3 pass)

How "close to the real thing" is measured, and the numbers across releases.
The comparison tool is `scripts/compare_reference.py`; ground truth is
Cassini's cylindrical Jupiter map (PIA07782, fetched into the gitignored
`refs/` by `scripts/fetch_references.py`).

## What is measured

Per-latitude statistics (90 bins) from `gasgiant.palette.reference`:

- **Level statistics** — `zone_rgb` / `belt_rgb` (median color of the
  brightest/darkest luminance quartile), `contrast` (p95 − p5 luminance),
  `zone_chroma` / `belt_chroma` (median Oklab chroma over the quartile's
  member pixels). Medians of a hue-spread population regress toward gray, so
  the member-chroma statistics see saturation the RGB medians cannot.
- **Variance statistics** — quartile-conditional `zone/belt_chroma_std` and
  `zone/belt_L_std` (within-band richness; full-bin stds would measure the
  zone-belt separation instead), `belt_chroma_p95` (saturated-pocket tail,
  the anti-cartoon guardrail), `hue_spread` (chroma-weighted circular hue
  spread), `texture_energy` (mean |∇L| — fine-texture density proxy).

**Common-resolution rule**: level statistics compare at width 1280;
variance statistics at width 640 — the reference is a 4:2:0 JPEG whose
chroma is native at half width, so 640 is where both images are equally
band-limited and block artifacts partially average out. Variance numbers
computed at other widths are not comparable to these tables.

## v1.2 baseline (raw view, jupiter_like @2048 vs PIA07782)

Distances (mean abs):

| metric | v1.2 |
|---|---|
| zone_rgb | 0.1016 |
| belt_rgb | 0.0858 |
| contrast | 0.0790 |
| zone_chroma | 0.0089 |
| belt_chroma | 0.0080 |
| zone_chroma_std | 0.0035 |
| belt_chroma_std | 0.0044 |
| zone_L_std | 0.0096 |
| belt_L_std | 0.0103 |
| belt_chroma_p95 | 0.0122 |
| hue_spread | 0.0489 |
| texture_energy | 0.0092 |

Signed means (ours − ref, |lat| ≤ 50°):

| metric | v1.2 | reading |
|---|---|---|
| zone_chroma | +0.0007 | zone saturation level: matched |
| belt_chroma | −0.0017 | belt saturation level: small deficit |
| belt_chroma_std | −0.0049 | within-belt chroma variance: ~half theirs |
| belt_L_std | −0.0070 | within-belt lightness patching: deficit |
| belt_chroma_p95 | −0.0097 | saturated-pocket tail: large deficit |
| hue_spread | −0.0835 | hue diversity: large deficit |
| texture_energy | −0.0070 | fine-texture density: deficit |

**Diagnosis the v1.3 pass acts on**: the *median* color levels were already
calibrated well in v1.1 — the visible "grayer, flatter" gap lives in the
variance statistics: the reference holds pockets of saturated rusty
material (chroma tail), slow longitudinal lightness patching, far more hue
diversity, and denser fine texture. The pass therefore targets variance
and tails (chroma_variance, bands.variance_amount, palette stop diversity,
detail intermittency) rather than pushing global saturation.

(Note: the v1.2 line in older docs quotes zone 0.1006 / belt 0.0843 /
contrast 0.0767 — those were computed at native resolutions; the small
shift here is the common-resolution protocol, not an output change.)

## v1.2 baseline (AgX view: agx_view(ours) vs the reference as-loaded)

`--view agx` applies `gasgiant.palette.agx.agx_view` (the repo's own AgX
approximation, ported from app/shaders/agx.glsl and pinned to it by a GPU
cross-validation test) to OUR render only. These are the confirmatory
acceptance denominators; raw view stays the cross-history series.

Distances: zone_rgb 0.0897, belt_rgb 0.0813, contrast 0.0856,
zone_chroma 0.0097, belt_chroma 0.0090, zone_chroma_std 0.0037,
belt_chroma_std 0.0044, zone_L_std 0.0093, belt_L_std 0.0103,
belt_chroma_p95 0.0145, hue_spread 0.0490, texture_energy 0.0101.

Signed (ours − ref, |lat| ≤ 50°): zone_chroma −0.0065, belt_chroma
−0.0083, belt_chroma_std −0.0057, belt_L_std −0.0120, belt_chroma_p95
−0.0179, hue_spread −0.0840, texture_energy −0.0136.

**Measured AgX chroma retention of the v1.2 render** (quartile medians,
|lat| ≤ 50°): **belt 0.677, zone 0.564**. Note this is lower than the
spot-color belt figures quoted below (0.92–1.16): a real render's belt
quartile is a mixture containing pastel material whose chroma does drop
through AgX. The render-quartile measurement is the one tuning uses —
and it must be re-measured whenever the palette changes.

## Measured perf calibration (16K all-maps, RTX 3070)

The detail retune (frequency 64, flow_stretch 1.3, striation 0.8@160,
intensity 0.75) is cost-neutral: 33.5 s vs the v1.2 33.8 s — frequency and
stretch change WHAT the same fbm calls sample, not how many. One extra
flow phase costs **+1.7 s** (35.2 s at phases=4), falsifying the planning
assumption of +4–5 s/phase — which is why jupiter_like ships flow_phases 4.

## v1.3 final (after the chroma-restored recalibration + variance tuning)

Raw view, jupiter_like @2048 vs PIA07782 (same protocol as the baselines):

| metric | v1.2 | v1.3 | note |
|---|---|---|---|
| zone_rgb | 0.1016 | 0.0679 | −33 % |
| belt_rgb | 0.0858 | 0.0747 | −13 % |
| contrast | 0.0790 | 0.0610 | −23 % |
| belt_chroma (signed) | −0.0017 | +0.0008 | deficit eliminated |
| zone_chroma (signed) | +0.0007 | −0.0013 | level matched |
| belt_chroma_std ratio | 0.38× ref | 0.65× ref | ±25 % target missed, see below |
| zone_chroma_std ratio | 0.31× | 0.46× | |
| belt_L_std ratio | ~0.60× | 0.63× | |
| belt_chroma_p95 ratio | — | 0.93× | guardrail ≤1.15 ✓ |
| hue_spread ratio | ~0.37× | 0.43× | advisory criterion, missed |
| clip fraction | 0 | 0 | anti-cartoon ✓ |

Honest findings recorded for the next pass:

- **The per-bin |belt_chroma| distance is band-ALIGNMENT-dominated**: the
  residual alternates sign band-by-band (±0.02 in adjacent 10° bands)
  because our procedural band layout doesn't coincide with the real
  NEB/SEB latitudes. No contiguous ≥15° span → the per-latitude envelope
  rule does NOT fire; the meaningful chroma-level statistic is the signed
  mean (now ≈ 0). Distance-style chroma criteria are not achievable
  without aligning band layout to the reference — out of scope.
- **Within-quartile std ratios plateau ≈ 0.65** (chroma_variance 0.35,
  variance_amount 0.18): the per-bin luminance-quartile membership
  re-selects under drift, absorbing part of the spread, and the
  reference's remaining within-quartile spread is storm-scale texture
  rather than slow drift. Pushing the drift knobs further blotches band
  definition without reading closer. The lever for the residual is
  finer storm-scale structure, not color drift.
- **AgX-view zone chroma deficit (−0.0079) is structural**: zone retention
  ≈ 0.55 through AgX; recovering it would need ~2× oversaturated raw
  zones, violating the raw guardrails. Documented, not chased.
- **hue_spread (0.43× ref)**: no shader knob rotates hue; the
  chroma-restored fit uses one hue direction per stop. The lever is
  per-stop hue diversity in the palette rows (future work).

Perf: 16K all-maps, all FX on (chroma + intermittency + spiral + emission):
**31.1 s median of 3** (v1.2: 33.8 s; gate 40 s).

## v1.4 feature audit — pass 1 (pre-template; seed 4257, 8K render)

Instrument: `scripts/audit_features.py` (dual-scale crop pairs, manifest in
`out/audit/`). Judged by the implementer + one independent blind judge
(crops + claims, no verdicts shared); two independent reverse-pass agents
enumerated the reference's feature classes by grid sweep, blind to the
codebase. Verdicts are per discrepancy; band-coupled verdicts are
PROVISIONAL (the band template re-audit re-grades them).

### Forward pass (formations.md claims vs reference)

| feature | verdict | evidence |
|---|---|---|
| band layout (count/widths/EZ) | BUILD (provisional) | ~12+ jittered bands vs ref's few broad belts + wide EZ; the known alignment residual |
| high-lat (40–60°) regime | BUILD | ref: dense lace-filament mottle with embedded ovals/rims; ours: featureless smooth gradient. Largest single gap after belts |
| belt interior (matched scale) | BUILD/LIMIT | ref belts are wall-to-wall folded filament chaos AT MATCHED SCALE (lum std 28 vs our 13); not a fine-texture issue |
| zone interior | TUNE | ours isotropic fuzz; ref smooth + long soft along-flow streaks |
| thin dark lanes | confirmed remove | uniform unbroken pencil lines; nothing like it in the ref (user-confirmed; commit 3) |
| GRS | TUNE+CODE | salmon level plausible; core near-circular vs ref 1.8:1; band edge runs straight THROUGH the storm (no deflection/moat); orange confined to small donut |
| GRS wake | BUILD (provisional) | absent at matched scale (hero sits in a pale quiet zone this seed) |
| GRS internal spiral | TUNE | lanes present but read annular/concentric; too faint; (a lane line slices the storm — fixed by commit 3) |
| white ovals | CODE-SMALL | largest ovals have near-zero visual signature: no compact edge, no dark rim. Ref: crisp rimmed white dots in lines |
| brown barges | CODE-SMALL | render as faint WARM blotches — wrong sign (ref: darker than belt), no cigar geometry |
| string of pearls | CODE-SMALL | invisible even at native res (confirmed by debug strip); placed but unrendered |
| merger debris | N/A | none alive at run end this seed |
| festoons | CODE-SMALL+TUNE | streamers read warm tan vs ref's blue-gray (hue offset wrong direction); droop instead of sweeping with shear |
| 5-µm hot spots | CODE-SMALL | no compact dark holes distinguishable |
| KH billows | TUNE | scallop train present but faint; no curl-over/rollup |
| meander | TUNE | wanders (claim satisfied) but amplitude low, spectrum jitter-dominated vs ref's coherent wavenumber 5–20 swings. Closest to MATCH |
| outbreaks | N/A-OFF | ref shows one live white plume + wake; argues for outbreak_count ≥ 1 |
| belt filaments (native) | TUNE | present; uniform/laminar — folded violence absent |
| zone cells (native) | TUNE | present, very low amplitude |
| striation (native) | TUNE | present faintly |
| intermittency (native) | TUNE | busy/calm alternation present; dynamic range a fraction of claim |
| polar south | BUILD | ref: cyclone ring with spiral arms, popcorn clusters, warm collar→blue transition; ours: near-featureless pale disk, faint smudges, off-center clump |
| polar north | BUILD | smooth gradient, zero discrete cyclones visible |
| global color balance | TUNE | belt-zone color offset ~half ref's (pale tan vs ochre-against-blue-white) |

### Reverse pass (reference classes with no/weak counterpart)

MISSING or visually absent: temperate lace-filament mottle regime (both
sweeps' top finding); braided "rope" band (~+38°); dark grey-blue spot
chains (+20/+35°); double/multi-stranded belts; band-hero deflection
(SEB hooks around the GRS + white moat); GRS elongation; large diffuse
color-anomaly patches (faded_sector exists but weak); zone laminar
streamline striations (floored too low); popcorn high-cloud clusters
(close-up scale). Implemented-but-OFF: convective plumes (ref shows one).
Both sweeps note: the ref map's featureless poles are partly projection
blur — PIA21641 is the polar ground truth, and ours is far from it.

### Pre-registered v1.4 targets (recorded at the template commit, BEFORE
### the post-template measurement; baseline = post-B0 @2048)

Counterfactual alignment ceilings (identity-matched monotone latitude warp
of the post-B0 profile onto the template's band positions — 10/12 ref bands
matched; the −57.5° belt and −78.5° polar zone are unmatched ceiling
limitations). Targets = 70% of each metric's OWN ceiling:

| metric | post-B0 | warp ceiling | target |
|---|---|---|---|
| zone_rgb | 0.0673 | −29.5 % | ≤ 0.0534 |
| belt_rgb | 0.0766 | −25.2 % | ≤ 0.0631 |
| contrast | 0.0618 | −2.7 % (alignment cannot move it) | no regression > 2 % |
| belt_chroma | 0.0117 | −14.3 % | ≤ 0.0105 |

(a) Band-level de-alternation: post-B0 residual sign-change count 6 over 9
template bands (|lat| ≤ 50°) → target ≤ 3; masked mean |belt_chroma|
residual 0.0137 → ≤ 0.0082 (−40%).
(c) Variance floors (ratio-to-ref, script-side): belt_chroma_std ≥ 0.58×,
zone_chroma_std ≥ 0.40×, belt_L_std ≥ 0.55×.
(d) hue_spread re-measured post-recalibration; the hue_variance shader knob
is built only if still < 60% of ref.

### v1.4 re-audit (post-template; seed 4201; full 3-judge panel, one
### self-cropping with independent quantitative probes)

Hypothesis test against the pre-registered targets:

| target | result | verdict |
|---|---|---|
| zone_rgb ≤ 0.0534 | 0.0359 | **HIT** (beat the warp ceiling — recalibration stacked) |
| belt_rgb ≤ 0.0631 | 0.0644 | marginal miss (+2 %) |
| contrast: no regression > 2 % | 0.0741 (+19.9 %) | **MISS** — diagnosis below |
| belt_chroma ≤ 0.0105 | 0.0100 | HIT |
| (a) sign changes ≤ 3 | 6 → 4 | near miss (de-alternation real, one flip left) |
| (a) masked mean residual ≤ 0.0082 | 0.0058 (−58 %) | **HIT** |
| (c) belt_chroma_std ≥ 0.58× | 0.612× | HIT |
| (c) zone_chroma_std ≥ 0.40× | 0.309× | MISS |
| (c) belt_L_std ≥ 0.55× | 0.472× | MISS |
| (d) hue_spread ≥ 0.60× | 0.579× | fires → hue_variance gets built |

**Shared diagnosis for the misses** (judge-confirmed + verified by direct
probes on the 8K render): (1) the recalibrated palette rows at anchors
inside the ref's blur/mottle-dominated regions (66°, −57.5°, −78.5°) are
near-constant — quartile windows there have tiny spread — which crushes
T0 dynamic range: high-lat banding, polar cyclones, and storm stamps all
render through nearly-flat gradients; (2) the detail layer's filament
masks are gated on jet speed/shear, and the template's broad bands put
their interiors far from any jet (the old 16-band layout had a jet every
~10°) — NEB-analog interior at native 8K is near-featureless where the
v1.3 render showed filaments. Both are levers in the authorized fix set,
not goal-shifts: contrast/L_std/zone_chroma_std re-measured at acceptance.

Re-audit verdict consensus (3 judges, majority per discrepancy):
band skeleton MATCH (all three: alignment achieved; "GRS sits at a
plausible SEB-flank latitude"); global color family close at low-mid lat
(judge-3 probes: NEB within ~10 BGR units, EZ within ~5); remaining gaps
ranked unanimously: (1) turbulent texture invisible planet-wide on the
new layout, (2) vortex rendering (GRS rim/moat/elongation; ovals/pearls/
barges at or below visibility; "concentric ripple rings" artifact = the
collar-streamline term standing out in the calm new context), (3) belt
saturation + within-band contrast, (4) polar clusters (flat palette rows
+ tint flatten the patch sim's content), (5) festoons orange-not-blue +
hotspots absent. GRS size graded undersized at matched scale (hero
radius vs the ref's ~12° oval). Lanes-gap verdict from pass 1 stands
(now removed). KH/meander: amplitude/wavenumber TUNE verdicts stand.

### Harness notes

Crops 20/22 shared one file (belt-filament and striation boxes
coincided); white_oval_2's crop was anomalously blurred. Both noted for
the B2.5 re-run; neither changes a verdict.

## Tuning protocol

- Texture judgement happens on **≥4K renders only**: the striation layer at
  frequency 160 is structurally invisible below ~4K (the across-flow
  wavelength attenuator floors it), and spiral jig detail shimmers at 1024.
  The small GUI preview cannot show what the export will look like.
- AgX chroma retention is color-dependent: re-measure it from the CURRENT
  render before deriving pre-AgX targets (a table measured on old colors is
  stale the moment the palette moves). Measured on v1.2 colors: dark belts
  retain ≈ 0.92–1.16 of Oklab chroma through AgX (belt saturation is a
  raw-space problem — do NOT pad belt targets for assumed AgX losses);
  pale zones retain ≈ 0.49 (zone chroma is AgX-limited); the AgX white
  ceiling ≈ 0.787 makes the AgX-view zone-lightness gap structural.

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

# Epoch recipes

Parameter recipes for planet **epochs** — documented historical states of the
real atmospheres that the factory presets deliberately do not bake in. Each
recipe is a small overlay on a factory preset: load the preset, apply the
listed fields, re-run. (Preset JSONs are sparse — see `docs/presets.md` — so
these overlays can also be saved as presets of their own.)

These two recipes come from the 2026-07-02 comprehensive review
(`docs/reviews/2026-07-02-comprehensive-review.md`): the faded-SEB scenario
(B5-2, Top-10 #8) previously required a template edit **plus** a palette-row
workaround that whitened the whole latitude ring; A11 (ochre EZ) was judged
"expressible but undocumented". Both now route through the W6 levers
`bands.belt_fade` + `bands.faded_band_index`.

Band indices below refer to the shipped Cassini-calibrated Jupiter template
(`jupiter_vorticity` / `gas_giant_warm`), index 0 = northernmost band:
band **4** = NEB (18.0..5.91 deg), band **5** = EZ (5.91..-7.31 deg),
band **6** = SEB (-7.31..-19.41 deg). Always set `faded_band_index`
explicitly — the auto (widest-belt) pick chooses the SEB over the NEB by only
0.01 deg on this template (review B5-4), so template tweaks can silently move
an unpinned fade.

## Faded SEB (Jupiter 2010 / pre-revival)

The South Equatorial Belt whitens to zone level around the Great Red Spot;
the revival then erupts as a convective plume train **inside** the faded
belt. Base: `jupiter_vorticity`.

```json
{
  "bands": { "belt_fade": 1.0, "faded_band_index": 6 },
  "storms": {
    "barge_density": 1.8,
    "outbreak_count": 2,
    "outbreak_latitude": -13.4,
    "outbreak_phase": 0.7,
    "outbreak_strength": 1.1
  }
}
```

- `belt_fade 1.0` = full fade (the belt's T0 stamp lands on its neighboring
  zones' mean). 0.5–0.7 gives the partially-faded intermediate epochs.
- The fade is **visual-only by design** (recorded LIMIT): the belt keeps its
  belt-class churn, storm seeding, and outbreak candidacy — which is the real
  SEB-fade phenomenology. Identity is frozen pre-fade (`BandLayout.is_belt`).
- The outbreak trio stages the **revival**: pinned to the SEB center
  (-13.4 deg), erupting at 70% of the dev run so the snapshot catches the
  fresh train plus its sheared streak. Drop the `outbreak_*` fields for the
  quiet pre-revival state.
- No template edit and no palette-row workaround: the old scenario's widened
  SEB edge (-19.41 → -19.9) and whitened -13.4 deg palette row are obsolete.

## Ochre EZ (Jupiter 2018-19 equatorial haze)

The normally white Equatorial Zone discolored to an ochre/tan haze. Point the
fade at the **zone**: it blends down-palette toward its belt neighbors, which
the warm equatorial palette rows render as ochre tans. Base:
`jupiter_vorticity`.

```json
{
  "bands": { "belt_fade": 0.45, "faded_band_index": 5 }
}
```

- `belt_fade 0.45` puts the EZ stamp at ~0.62 (between zone cream and belt
  tan). 0.3 is a light haze; beyond ~0.6 the EZ reads as a full belt.
- Optional deepeners: `appearance.chroma_aging 0.2` ties the discoloration to
  stagnant air; `appearance.haze_amount` warms the global cast instead — keep
  it at 0 if only the EZ should change.
- Festoons keep rooting on the NEB-S edge (their latitude comes from the band
  skeleton, which the fade never moves).

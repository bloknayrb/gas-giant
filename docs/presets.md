# Presets and the manifest contract

> Looking for the parameter overlays that reproduce documented historical
> atmosphere states (faded SEB, ochre EZ)? Those live in [recipes.md](recipes.md).

## Presets (STRICT policy)

A preset is a JSON envelope around the parameter tree:

```json
{
  "preset_format": 1,
  "app_version": "0.1.0",
  "name": "my_planet",
  "params": { "seed": 4257, "bands": { "count": 16 } }
}
```

- **Sparse is fine** — missing fields take model defaults.
- **Unknown keys are ERRORS** — a hand-edited typo (`"tubrulence"`) silently
  becoming a default is data loss, so loading fails with the offending path.
- `preset_format` bumps only on breaking changes; upgrades run through the
  registry in `params/migrations.py`. Files newer than the app are refused
  with a clear message.
- Factory presets are package data (`src/gasgiant/presets/*.json`); the GUI
  autosaves the session to `~/.gasgiant/session.json` on exit.

## User presets

User presets live in `~/.gasgiant/presets/` (`USER_PRESET_DIR` in
`params/presets.py`), one `<name>.json` per preset, and appear in the GUI's
preset dropdown merged after the factory entries (displayed as
`user/<name>`, so a user preset can never shadow a same-named factory
preset). The GUI's *Save As* dialog
defaults into that directory (created on demand); Ctrl+S overwrites the
active user preset after a confirm modal, or falls back to Save As when a
factory/file/unsaved identity is active. Enumeration never opens the files —
a corrupt preset cannot crash the dropdown; it only fails (with a toast) when
actually loaded. *Import preset…* validates an external `.json` through the
strict envelope path (running any format migrations) and installs it as a
durable user preset — unlike *Load*, whose file identity is transient — and
it never overwrites an existing user preset of the same name.

## Epoch recipes (parameter overlays)

An **epoch recipe** is a small overlay reproducing a documented historical
atmosphere state (faded SEB, ochre EZ — the phenomenology is in
[recipes.md](recipes.md)). Recipes ship as package data under
`src/gasgiant/presets/recipes/*.json`, and — unlike a preset — a recipe file
is a RAW overlay, **not** a preset envelope: `{ "base": <factory preset name>,
"overlay": <sparse nested dict>, "name"?, "description"? }`. `load_recipe`
returns `(base, overlay, meta)`; `apply_overlay` **deep-merges** the sparse
overlay onto the resolved base params (siblings of any group the overlay enters
are preserved) and re-validates through the strict model, so a typo'd overlay
key raises rather than silently defaulting — exactly like a hand-edited preset.

Precedence (CLI `--recipe`, on `export`/`checkpoint`/`sheet`): with `--preset`,
the overlay is applied on top of that preset; without `--preset`, the recipe's
own `base` is used; with neither, the subcommand's default base holds. In the
GUI the same overlays are the **Scenarios** menu. Because a merged overlay is
just a `PlanetParams`, the result can be saved as an ordinary user preset.

## Imported mask sidecars

`mask.file` points at an imported paint-mask PNG (art-directs POST output via
`band_fade`/`emission_gain`/`detail_gain`; see `docs/architecture.md`). Path
handling keeps presets portable: `load_preset` resolves a **relative**
`mask.file` against the preset's OWN folder and makes it absolute in memory
(`resolve_mask_path`); on `save_preset` the path is **re-relativized** to a
sidecar filename next to the saved JSON and the PNG is **copied there**
(`_relativized_for_save`), so a shared preset carries its mask. The session
autosave keeps the absolute path (`relativize_mask=False`) — it is not meant to
be moved. The CLI treats a missing `mask.file` as a hard error; the engine's
warn-and-disable path is for the checkpoint/GUI case where a portable preset
may outlive its sidecar.

## mapset.json (TOLERANT policy)

The exporter ↔ Blender contract. Canonical JSON Schema:
`src/gasgiant/export/mapset.schema.json`. Readers IGNORE unknown keys —
additive changes never bump `schema_version`; on a future major bump the
add-on warns and imports best-effort. The add-on vendors a stdlib-only
reader (`blender_addon/gasgiant_importer/manifest_schema.py`); a unit test
keeps it accepting exactly what the exporter writes.

Per-map entries carry `file`, `format` (`png16` | `exr32f`), `colorspace`
(`srgb` | `non-color`), and optional `channels`/`convention`. `physical`
carries `radius_km`, `height_scale` (full height-map range as a fraction of
radius), and `height_midlevel` — the importer derives the physically correct
displacement scale from these. The full generating preset is embedded under
`preset` for reproducibility. Optional additive maps a tolerant reader ignores:
`flow` (`flow.exr`, east/north velocity) and `rings` (`rings.exr`, radial
strip + `physical.ring_inner_km`/`ring_outer_km`) — both stay on
`schema_version 1`.

**Animated sequences** carry a `frames` block (`count`, plus a `frames.maps`
list naming which maps were written per frame — color always, height/emission
only with `--all-maps`); the importer loads those maps as image sequences (see
`docs/blender_addon.md`). A map set with no `frames` block imports as a still.

**Cube projection** (`export.projection = cube`) is the one **breaking**
manifest change: `schema_version` bumps to **2**, `projection` is `"cube"`, and
each map carries a `faces` block (`px/nx/py/ny/pz/nz`) instead of a single
`file`. Importers that only build equirect geometry reject a v2 manifest
cleanly (the current Blender importer does — see `docs/blender_addon.md`).

# Presets and the manifest contract

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
`preset` for reproducibility. A `frames` array is reserved for animated
sequences.

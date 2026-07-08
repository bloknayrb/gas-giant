# Blender Importer

## Install

1. Build the zip: `uv run python scripts/build_addon.py` → `dist/gasgiant_importer-1.1.0.zip`
2. Drag the zip into a Blender 4.2+ window (or *Preferences → Get
   Extensions → Install from Disk*).

## Use

*File → Import → Gas Giant Map Set (.json)* and pick the `mapset.json` in an
exported folder. Options in the import panel:

- **Radius** — planet radius in Blender units (default 1.0).
- **Mesh Segments** — sphere resolution (default 256, range 32–1024). High
  values shrink the pole fan; texture lookup is exact regardless under
  procedural mapping.
- **Mapping** — *Procedural spherical* (default) computes equirect UV per
  pixel from object coordinates: exact at the poles, no seam — best for
  Cycles. Use *Mesh UV* for EEVEE (the procedural mapping shows a one-pixel
  derivative seam at the dateline there).
- **True displacement** — off by default: real gas-giant cloud relief is
  ~3×10⁻⁴ of the radius, sub-pixel in any sane shot, so bump shading is
  usually correct. Turning it on adds a Subdivision modifier with adaptive
  subdivision (dicing 2.5 px — the 1 px default can exceed 8 GB VRAM on a
  screen-filling planet). *Relief exaggeration* multiplies the physically
  derived scale.
- **Atmosphere** — *Volume shell* (default, Cycles): a thin Volume Scatter
  shell, light-aware for free (correct terminator scattering, dark night
  limb). *Rim glow* is the fast EEVEE-safe fallback, gated by the sun
  direction captured at import. *None* skips the atmosphere entirely. The
  shell never casts shadows (`visible_shadow = False`).
- **Limb darkening / Limb haze** — facing-based grading on the surface
  material.
- **Longitude offset / Axial tilt** — put your hero storm where the camera
  is; tilt lives on a parent rig empty (animate the empty's Z for spin).
- **Create demo scene** — sun lamp at 0.25° angular size, black world,
  framed camera, AgX view transform: first renders look right.

## Emission (night-side glow)

If the map set was exported with any `emission.*_strength` nonzero, the
importer wires `emission.exr` into the Principled "Emission Color"
(channel-packed alpha — the EXR's alpha is an aurora data mask, not
coverage). Options:

- **Emission strength** — multiplier applied at import; night-side renders
  often want 2–10 without re-exporting. Cycles is required for the emission
  to actually light the scene (EEVEE shows it but doesn't illuminate
  without raytracing).
- **Aurora on shell** — lifts the aurora (alpha channel × the manifest's
  `aurora_color`) onto a transparent shell at ~1.03 R instead of the cloud
  surface; it stands off the limb like the real thing. The shell emission
  is not sun-gated: dayside-negligible at default strength, not
  dayside-clean. For lit-only shots simply export with strengths 0.

## Animated sequences

If the map set was exported with a `frames` block (the animation export), the
importer loads the colour map — and the height / emission maps when
`frames.maps` lists them — as an **image sequence** instead of a still:

- The frame-0 file (`frames/frame_0000.png`, `frames/height_0000.png`,
  `frames/emission_0000.exr`) is loaded and its `image.source` set to
  `SEQUENCE`; Blender discovers the remaining frames from the `_%04d`
  numbering.
- On each texture node's `image_user`: `frame_duration = frames.count`,
  `frame_start = 1`, and `frame_offset = -1`, with `use_auto_refresh` and
  `use_cyclic` on. Blender maps the scene frame to the on-disk picture number
  as `picture_number = scene_frame − frame_start + 1 + frame_offset`; our
  frames are 0000-based, so `frame_offset = -1` puts picture 0000 on scene
  frame 1 (the load-bearing off-by-one).
- Scrub the timeline (or render an animation) and the planet's clouds advect;
  auto-refresh keeps the viewport in sync. A map absent from `frames.maps`
  (and any map set with no `frames` block) imports as a still, unchanged.

## Version compatibility

`compat.py` is the only file that branches on `bpy.app.version`. Verified
live against Blender 5.1.2 (and written to the 4.2 LTS API for the rest):
the 5.x `displacement_method` enum (`BOTH`), the 5.x adaptive-subdivision
move onto the Subsurf modifier, the removed `feature_set` property, and
colorspace-name fallbacks for custom OCIO configs. Nodes are created by
type and sockets resolved through alias lists, never display-name lookups.

"""The ONLY file allowed to branch on bpy.app.version.

Verified against a live Blender 5.1.2 (API probe) and the 4.2 LTS docs:
- Material.displacement_method enum: 5.x uses 'BOTH', 4.x used
  'DISPLACEMENT_AND_BUMP'.
- scene.cycles.feature_set was REMOVED in 5.0 (adaptive subdivision is no
  longer experimental); the per-object cycles.use_adaptive_subdivision moved
  to the Subdivision Surface modifier (mod.use_adaptive_subdivision).
- image.colorspace_settings.name: 'Non-Color' / 'sRGB' exist in the default
  OCIO config; custom configs need alias fallbacks.
"""

from __future__ import annotations

import contextlib

import bpy

NONCOLOR_ALIASES = ("Non-Color", "Non-Colour Data", "Generic Data", "raw", "Raw", "Utility - Raw")
SRGB_ALIASES = ("sRGB", "sRGB - Texture", "Utility - sRGB - Texture", "srgb")


def set_displacement_method(mat: bpy.types.Material, displace: bool) -> None:
    if not displace:
        mat.displacement_method = "BUMP"
        return
    for value in ("BOTH", "DISPLACEMENT_AND_BUMP"):
        try:
            mat.displacement_method = value
            return
        except TypeError:
            continue
    mat.displacement_method = "DISPLACEMENT"


def enable_adaptive_subdivision(obj: bpy.types.Object, scene: bpy.types.Scene) -> None:
    mod = obj.modifiers.new("Subdivision", "SUBSURF")
    mod.levels = 2
    mod.render_levels = 3
    if hasattr(mod, "use_adaptive_subdivision"):  # 5.x
        mod.use_adaptive_subdivision = True
    elif hasattr(obj, "cycles") and hasattr(obj.cycles, "use_adaptive_subdivision"):  # 4.x
        obj.cycles.use_adaptive_subdivision = True
    if hasattr(scene, "cycles"):
        # 4.x required the experimental feature set; the property no longer
        # exists in 5.x (guarded — never written blindly).
        if hasattr(scene.cycles, "feature_set"):
            scene.cycles.feature_set = "EXPERIMENTAL"
        if hasattr(scene.cycles, "dicing_rate"):
            # Default 1px dicing on a screen-filling planet can exceed 8 GB
            # VRAM; 2.5px is visually fine for cloud-deck relief.
            scene.cycles.dicing_rate = 2.5


def set_colorspace(image: bpy.types.Image, kind: str) -> None:
    """kind: 'srgb' | 'non-color' (manifest values)."""
    aliases = SRGB_ALIASES if kind == "srgb" else NONCOLOR_ALIASES
    for name in aliases:
        try:
            image.colorspace_settings.name = name
            return
        except TypeError:
            continue
    # Custom OCIO config without any known alias: leave the default rather
    # than failing the whole import.


def set_channel_packed(image: bpy.types.Image) -> None:
    """Mark an image's alpha as CHANNEL-PACKED data. Blender defaults EXR to
    Premultiplied: with an independent data mask in A (the emission map's
    aurora), the alpha-association path would scale/divide RGB by that
    near-zero mask and corrupt the whole map."""
    with contextlib.suppress(TypeError, AttributeError):
        image.alpha_mode = "CHANNEL_PACKED"


def find_input(node: bpy.types.Node, *names: str) -> bpy.types.NodeSocket | None:
    """Socket lookup tolerant of the 4.0/5.0 socket renames."""
    for name in names:
        sock = node.inputs.get(name)
        if sock is not None:
            return sock
    return None


def configure_image_sequence(tex_node: bpy.types.Node, *, frame_duration: int) -> None:
    """Turn a TEX_IMAGE node into an animated image sequence.

    Blender resolves the on-disk picture number from the scene frame as::

        picture_number = scene_frame - frame_start + 1 + frame_offset

    Our exporter writes 0000-based frame files (``frame_0000.png`` ...), so at
    scene frame 1 we want picture 0000: ``frame_start = 1`` and the load-bearing
    ``frame_offset = -1``. ``frame_duration`` is the number of frames in the run.

    The ``image_user`` sub-struct and its ``use_auto_refresh`` / ``use_cyclic``
    booleans are stable across 4.2 LTS and 5.x, but each is written under a
    ``hasattr`` guard so a future rename degrades to a still frame rather than a
    hard failure (this file is the only place allowed to know API drift)."""
    image = getattr(tex_node, "image", None)
    if image is not None:
        image.source = "SEQUENCE"
    user = getattr(tex_node, "image_user", None)
    if user is None:
        return
    user.frame_duration = frame_duration
    user.frame_start = 1
    user.frame_offset = -1
    if hasattr(user, "use_auto_refresh"):
        user.use_auto_refresh = True
    if hasattr(user, "use_cyclic"):
        user.use_cyclic = True


def set_transparent_render_method(mat: bpy.types.Material) -> None:
    """EEVEE transparency: 4.2+ EEVEE-Next uses surface_render_method."""
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    elif hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"


def set_view_transform_agx(scene: bpy.types.Scene) -> None:
    with contextlib.suppress(TypeError):  # custom OCIO config: leave as-is
        scene.view_settings.view_transform = "AgX"

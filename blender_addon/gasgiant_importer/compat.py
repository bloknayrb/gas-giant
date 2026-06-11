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


def find_input(node: bpy.types.Node, *names: str) -> bpy.types.NodeSocket | None:
    """Socket lookup tolerant of the 4.0/5.0 socket renames."""
    for name in names:
        sock = node.inputs.get(name)
        if sock is not None:
            return sock
    return None


def set_transparent_render_method(mat: bpy.types.Material) -> None:
    """EEVEE transparency: 4.2+ EEVEE-Next uses surface_render_method."""
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    elif hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"


def set_view_transform_agx(scene: bpy.types.Scene) -> None:
    with contextlib.suppress(TypeError):  # custom OCIO config: leave as-is
        scene.view_settings.view_transform = "AgX"

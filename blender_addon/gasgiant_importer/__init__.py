"""Gas Giant Importer: Blender extension entry point.

Extension format only (blender_manifest.toml) — no legacy bl_info. Stdlib +
bpy only by design: zero wheel dependencies means minimal exposure to
Blender's Python version changes.
"""

from . import importer


def register() -> None:
    importer.register()


def unregister() -> None:
    importer.unregister()

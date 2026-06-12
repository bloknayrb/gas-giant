"""Headless Blender import test.

Run inside Blender:
    blender --background --factory-startup --python tests/blender/test_import.py -- <mapset_dir>

Registers the add-on from source, imports the map set with each combination
that matters, and asserts the scene contents. An optional second mapset dir
(exported with nonzero emission strengths) drives the emission-wiring and
aurora-shell checks. Writes PASS/FAIL JSON to tests/blender/result.json
(background Blender via the Store launcher has no usable stdout).
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

RESULT_PATH = Path(__file__).resolve().parent / "result.json"


def main() -> dict:
    import bpy

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not argv:
        raise RuntimeError("usage: ... -- <mapset_dir> [<emission_mapset_dir>]")
    mapset = Path(argv[0]) / "mapset.json"
    assert mapset.is_file(), f"missing {mapset}"
    emission_mapset = Path(argv[1]) / "mapset.json" if len(argv) > 1 else None

    addon_root = Path(__file__).resolve().parents[2] / "blender_addon"
    sys.path.insert(0, str(addon_root))
    import gasgiant_importer

    gasgiant_importer.register()

    # Clear the factory-startup scene (default Cube etc.).
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    checks: dict[str, bool] = {}

    # Default import: bump path, volume atmosphere, procedural mapping.
    result = bpy.ops.import_scene.gasgiant(filepath=str(mapset), demo_scene=True)
    checks["operator_finished"] = result == {"FINISHED"}

    objs = bpy.data.objects
    planet = next((o for o in objs if o.type == "MESH" and "atmosphere" not in o.name), None)
    checks["planet_created"] = planet is not None
    checks["planet_has_material"] = bool(planet and planet.data.materials)
    shell = next((o for o in objs if "atmosphere" in o.name), None)
    checks["atmosphere_created"] = shell is not None
    checks["atmosphere_shadow_off"] = bool(shell and shell.visible_shadow is False)
    checks["camera_created"] = any(o.type == "CAMERA" for o in objs)
    checks["sun_created"] = any(o.type == "LIGHT" and o.data.type == "SUN" for o in objs)

    mat = planet.data.materials[0]
    nodes = mat.node_tree.nodes
    checks["_diag_planet"] = planet.name
    checks["_diag_mat"] = mat.name
    checks["_diag_nodes"] = sorted({n.type for n in nodes})
    checks["material_displacement_bump"] = mat.displacement_method == "BUMP"
    checks["has_bump_node"] = any(n.type == "BUMP" for n in nodes)
    checks["no_normal_map_node"] = not any(n.type == "NORMAL_MAP" for n in nodes)
    checks["procedural_mapping"] = any(n.type == "TEX_COORD" for n in nodes)
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    spec = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    checks["specular_zero"] = bool(spec and spec.default_value == 0.0)
    shell_mat = shell.data.materials[0]
    checks["volume_scatter"] = any(
        n.type == "VOLUME_SCATTER" for n in shell_mat.node_tree.nodes
    )

    # Displacement variant.
    result = bpy.ops.import_scene.gasgiant(
        filepath=str(mapset), use_displacement=True, atmosphere_mode="NONE", mapping="UV"
    )
    checks["displacement_import_finished"] = result == {"FINISHED"}
    planet2 = bpy.context.active_object
    checks["displacement_method"] = planet2.data.materials[0].displacement_method in (
        "BOTH", "DISPLACEMENT_AND_BUMP", "DISPLACEMENT",
    )
    checks["subsurf_added"] = any(m.type == "SUBSURF" for m in planet2.modifiers)
    mod = next(m for m in planet2.modifiers if m.type == "SUBSURF")
    adaptive = getattr(mod, "use_adaptive_subdivision", None)
    if adaptive is None and hasattr(planet2, "cycles"):
        adaptive = getattr(planet2.cycles, "use_adaptive_subdivision", False)
    checks["adaptive_subdivision"] = bool(adaptive)

    # Emission map absent: no emission texture node, default emission black.
    em_nodes = [
        n for n in nodes
        if n.type == "TEX_IMAGE" and n.image
        and Path(n.image.filepath).name.lower() == "emission.exr"
    ]
    checks["no_emission_node_without_map"] = not em_nodes

    if emission_mapset is not None:
        result = bpy.ops.import_scene.gasgiant(
            filepath=str(emission_mapset), atmosphere_mode="NONE",
            emission_strength=2.0,
        )
        checks["emission_import_finished"] = result == {"FINISHED"}
        planet3 = bpy.context.active_object
        nodes3 = planet3.data.materials[0].node_tree.nodes
        em3 = [
            n for n in nodes3
            if n.type == "TEX_IMAGE" and n.image
            and Path(n.image.filepath).name.lower() == "emission.exr"
        ]
        checks["emission_node_present"] = len(em3) == 1
        if em3:
            img = em3[0].image
            checks["emission_channel_packed"] = img.alpha_mode == "CHANNEL_PACKED"
            bsdf3 = next(n for n in nodes3 if n.type == "BSDF_PRINCIPLED")
            em_color = bsdf3.inputs.get("Emission Color") or bsdf3.inputs.get("Emission")
            checks["emission_color_linked"] = bool(em_color and em_color.is_linked)
            em_str = bsdf3.inputs.get("Emission Strength")
            checks["emission_strength_set"] = bool(em_str and em_str.default_value == 2.0)

        # Aurora shell variant: the alpha-driven shell exists, and the
        # surface no longer adds the aurora term (no vector-math chain).
        result = bpy.ops.import_scene.gasgiant(
            filepath=str(emission_mapset), atmosphere_mode="NONE",
            aurora_shell=True,
        )
        checks["aurora_shell_import_finished"] = result == {"FINISHED"}
        shell_au = next((o for o in objs if "_aurora" in o.name), None)
        checks["aurora_shell_created"] = shell_au is not None
        if shell_au is not None:
            au_nodes = shell_au.data.materials[0].node_tree.nodes
            checks["aurora_shell_emission"] = any(n.type == "EMISSION" for n in au_nodes)
            checks["aurora_shell_transparent"] = any(
                n.type == "BSDF_TRANSPARENT" for n in au_nodes
            )

    return checks


try:
    checks = main()
    ok = all(v for k, v in checks.items() if not k.startswith("_diag"))
    RESULT_PATH.write_text(json.dumps({"ok": ok, "checks": checks}, indent=1), encoding="utf-8")
except Exception:
    RESULT_PATH.write_text(
        json.dumps({"ok": False, "error": traceback.format_exc()}, indent=1), encoding="utf-8"
    )
    raise

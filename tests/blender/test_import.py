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
import struct
import sys
import traceback
import zlib
from pathlib import Path

RESULT_PATH = Path(__file__).resolve().parent / "result.json"


# --- stdlib fixture writer (no bpy / numpy / cv2 — importable + testable GPU-free) ---

def _tiny_png(path: Path, width: int = 2, height: int = 1) -> None:
    """Write a minimal valid 8-bit RGB PNG (the reader/importer never inspect
    pixel data, so 2x1 black is enough for both the vendored-reader unit test
    and Blender's image loader). Pure stdlib zlib + struct."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # RGB, no interlace
    raw = (b"\x00" + b"\x00\x00\x00" * width) * height  # per-row filter byte 0
    idat = zlib.compress(raw)
    path.write_bytes(sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b""))


def write_sequence_fixture(root: Path, *, count: int = 3, steps_per_frame: int = 8) -> Path:
    """Create a tiny fake exported map set WITH a T7 `frames` block: still color/
    height maps plus per-frame color/height PNG sequences and an emission EXR
    sequence (placeholder bytes — emission frames aren't pixel-inspected). Pure
    stdlib, so the vendored reader unit test and the Blender background test share
    it. Returns ``root`` (containing ``mapset.json``)."""
    root = Path(root)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    _tiny_png(root / "color.png")
    _tiny_png(root / "height.png")
    color_files, height_files, emission_files = [], [], []
    for i in range(count):
        cf = f"frames/frame_{i:04d}.png"
        _tiny_png(root / cf)
        color_files.append(cf)
        hf = f"frames/height_{i:04d}.png"
        _tiny_png(root / hf)
        height_files.append(hf)
        ef = f"frames/emission_{i:04d}.exr"
        (root / ef).write_bytes(b"")  # placeholder; never pixel-inspected
        emission_files.append(ef)
    manifest = {
        "schema_version": 1,
        "generator": {"name": "gasgiant", "version": "test"},
        "name": "seqfix",
        "seed": 7,
        "projection": "equirectangular",
        "resolution": [4, 2],
        "physical": {"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
        "maps": {
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "height": {"file": "height.png", "format": "png16", "colorspace": "non-color"},
        },
        "preset": {},
        "frames": {
            "count": count,
            "steps_per_frame": steps_per_frame,
            "pattern": "frames/frame_%04d.png",
            "files": color_files,
            "maps": {"height": height_files, "emission": emission_files},
            "video": "sequence.mp4",
        },
    }
    (root / "mapset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def write_rings_fixture(root: Path) -> Path:
    """Create a tiny fake exported map set carrying a T16 `rings` map + the
    physical ring_*_km extent. Pure stdlib (the rings.exr is placeholder bytes;
    Blender's EXR loader is not exercised for pixels here -- the assertion is
    that a ring OBJECT is created and parented to the rig). Returns ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _tiny_png(root / "color.png")
    _tiny_png(root / "height.png")
    (root / "rings.exr").write_bytes(b"")  # placeholder; loaded but not pixel-checked
    manifest = {
        "schema_version": 1,
        "generator": {"name": "gasgiant", "version": "test"},
        "name": "ringfix",
        "seed": 5,
        "projection": "equirectangular",
        "resolution": [4, 2],
        "physical": {
            "radius_km": 60268.0, "height_scale": 0.004, "height_midlevel": 0.5,
            "ring_inner_km": 74500.0, "ring_outer_km": 136780.0,
        },
        "maps": {
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "height": {"file": "height.png", "format": "png16", "colorspace": "non-color"},
            "rings": {
                "file": "rings.exr", "format": "exr32f", "colorspace": "non-color",
                "channels": 4, "convention": "radial_inner_to_outer_alpha_coverage",
            },
        },
        "preset": {},
    }
    (root / "mapset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


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

    # Animated-sequence import (T9): build a tiny fixture WITH a frames block
    # and assert the colour (and height) map imports as an image SEQUENCE with
    # the off-by-one pin. frame_offset == -1 is the load-bearing assertion.
    import tempfile

    seq_dir = Path(tempfile.mkdtemp(prefix="gg_seq_"))
    write_sequence_fixture(seq_dir, count=4)
    result = bpy.ops.import_scene.gasgiant(
        filepath=str(seq_dir / "mapset.json"), atmosphere_mode="NONE",
    )
    checks["sequence_import_finished"] = result == {"FINISHED"}
    planet_seq = bpy.context.active_object
    nodes_seq = planet_seq.data.materials[0].node_tree.nodes
    color_seq_nodes = [
        n for n in nodes_seq
        if n.type == "TEX_IMAGE" and n.image
        and Path(n.image.filepath).name.lower() == "frame_0000.png"
    ]
    checks["sequence_color_node"] = len(color_seq_nodes) == 1
    if color_seq_nodes:
        node = color_seq_nodes[0]
        iu = node.image_user
        checks["sequence_source"] = node.image.source == "SEQUENCE"
        checks["sequence_frame_duration"] = iu.frame_duration == 4
        checks["sequence_frame_start"] = iu.frame_start == 1
        checks["sequence_frame_offset"] = iu.frame_offset == -1
        checks["sequence_auto_refresh"] = iu.use_auto_refresh is True
    height_seq_nodes = [
        n for n in nodes_seq
        if n.type == "TEX_IMAGE" and n.image
        and Path(n.image.filepath).name.lower() == "height_0000.png"
    ]
    checks["sequence_height_node"] = len(height_seq_nodes) == 1
    if height_seq_nodes:
        checks["sequence_height_source"] = height_seq_nodes[0].image.source == "SEQUENCE"
        checks["sequence_height_offset"] = height_seq_nodes[0].image_user.frame_offset == -1

    # Rings import (T16): a map set with a `rings` map builds a flat annulus
    # object parented to the rig empty, with a material sampling the ring image.
    ring_dir = Path(tempfile.mkdtemp(prefix="gg_ring_"))
    write_rings_fixture(ring_dir)
    result = bpy.ops.import_scene.gasgiant(
        filepath=str(ring_dir / "mapset.json"), atmosphere_mode="NONE",
    )
    checks["rings_import_finished"] = result == {"FINISHED"}
    ring_obj = next((o for o in bpy.data.objects if o.name.endswith("_rings")), None)
    checks["rings_object_created"] = ring_obj is not None
    if ring_obj is not None:
        checks["rings_parented_to_rig"] = (
            ring_obj.parent is not None and ring_obj.parent.type == "EMPTY"
        )
        checks["rings_has_material"] = bool(ring_obj.data.materials)
        checks["rings_has_faces"] = len(ring_obj.data.polygons) > 0
        if ring_obj.data.materials:
            rnodes = ring_obj.data.materials[0].node_tree.nodes
            checks["rings_samples_image"] = any(
                n.type == "TEX_IMAGE" and n.image
                and Path(n.image.filepath).name.lower() == "rings.exr"
                for n in rnodes
            )

    return checks


if __name__ == "__main__":
    # Blender runs this script as __main__ (blender --python ...); importing the
    # module (e.g. from the vendored-reader unit test) must NOT execute the bpy
    # path, so the run is guarded here.
    try:
        checks = main()
        ok = all(v for k, v in checks.items() if not k.startswith("_diag"))
        RESULT_PATH.write_text(
            json.dumps({"ok": ok, "checks": checks}, indent=1), encoding="utf-8"
        )
    except Exception:
        RESULT_PATH.write_text(
            json.dumps({"ok": False, "error": traceback.format_exc()}, indent=1), encoding="utf-8"
        )
        raise

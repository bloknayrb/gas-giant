"""The import operator: mapset.json -> ready-to-render planet."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import atmosphere, compat, manifest_schema, material


class IMPORT_SCENE_OT_gasgiant(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.gasgiant"
    bl_label = "Import Gas Giant Map Set"
    bl_description = "Import a Gas Giant Studio map set as a ready-to-render planet"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    radius: FloatProperty(
        name="Radius", default=1.0, min=0.001,
        description="Planet radius in Blender units",
    )
    mesh_segments: IntProperty(
        name="Mesh Segments", default=256, min=32, max=1024,
        description="Sphere resolution (high values shrink the pole fan; "
        "procedural mapping makes texture lookup exact regardless)",
    )
    mapping: EnumProperty(
        name="Mapping",
        items=(
            ("PROCEDURAL", "Procedural spherical",
             "Per-pixel equirect UV from object coordinates: exact at the "
             "poles, no seam (best for Cycles)"),
            ("UV", "Mesh UV",
             "Native sphere UVs (use for EEVEE, where the procedural "
             "mapping shows a derivative seam at the dateline)"),
        ),
        default="PROCEDURAL",
    )
    use_displacement: BoolProperty(
        name="True displacement", default=False,
        description="Adaptive-subdivision displacement (Cycles; heavy). "
        "Off uses bump shading — real cloud-deck relief is sub-pixel at "
        "planet scale, so bump is usually the right choice",
    )
    exaggeration: FloatProperty(
        name="Relief exaggeration", default=1.0, min=0.0, max=50.0,
        description="1.0 = physically derived relief from the manifest "
        "(visually nearly smooth, like the real planets)",
    )
    atmosphere_mode: EnumProperty(
        name="Atmosphere",
        items=(
            ("VOLUME", "Volume shell", "Light-aware scattering shell (Cycles)"),
            ("RIM", "Rim glow (fast)", "Stylized facing-based rim (EEVEE-safe)"),
            ("NONE", "None", ""),
        ),
        default="VOLUME",
    )
    limb_darkening: FloatProperty(name="Limb darkening", default=0.45, min=0.0, max=1.0)
    limb_haze: FloatProperty(name="Limb haze", default=0.3, min=0.0, max=1.0)
    emission_strength: FloatProperty(
        name="Emission strength", default=1.0, min=0.0, soft_max=10.0,
        description="Multiplier on the emission map (thermal glow, lightning, "
        "aurora). Night-side renders often want to push this without "
        "re-exporting. Needs Cycles to actually light the scene",
    )
    aurora_shell: BoolProperty(
        name="Aurora on shell", default=False,
        description="Lift the emission map's aurora (alpha channel) onto a "
        "transparent shell at ~1.03 R instead of the cloud surface — the "
        "real aurora sits ~1000 km up. Dayside-negligible, not sun-gated",
    )
    build_rings: BoolProperty(
        name="Build rings", default=True,
        description="If the map set carries a ring strip (rings.exr), build a flat "
        "Saturn-style annulus in the equatorial plane from ring_inner_km..ring_outer_km. "
        "Rings are a Blender-only feature (invisible in the GUI preview)",
    )
    longitude_offset: FloatProperty(
        name="Longitude offset", subtype="ANGLE", default=0.0,
        description="Rotate the planet so the feature you care about faces the camera",
    )
    axial_tilt: FloatProperty(name="Axial tilt", subtype="ANGLE", default=0.0)
    demo_scene: BoolProperty(
        name="Create demo scene", default=False,
        description="Sun lamp (small angular size), black world, framed camera, "
        "AgX view transform — so the first render looks right",
    )

    def execute(self, context):
        try:
            doc = manifest_schema.read_mapset(Path(self.filepath))
        except manifest_schema.MapsetError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        for warning in doc["_warnings"]:
            self.report({"WARNING"}, warning)

        # Animation: when the manifest carries a T7 `frames` block, a map is
        # imported as a SEQUENCE pointed at its frame-0 file; without one (or for
        # a map absent from `frames.maps`) the still map is used, unchanged.
        frame_count = manifest_schema.frame_count(doc)
        color_seq = manifest_schema.frame_zero_path(doc, "color")
        height_seq = manifest_schema.frame_zero_path(doc, "height")
        emission_seq = manifest_schema.frame_zero_path(doc, "emission")

        color_path = color_seq or manifest_schema.map_path(doc, "color")
        height_path = height_seq or manifest_schema.map_path(doc, "height")
        emission_path = emission_seq or manifest_schema.map_path(doc, "emission")
        color_img = bpy.data.images.load(str(color_path), check_existing=True)
        height_img = (
            bpy.data.images.load(str(height_path), check_existing=True)
            if height_path is not None
            else None
        )
        emission_img = (
            bpy.data.images.load(str(emission_path), check_existing=True)
            if emission_path is not None
            else None
        )
        aurora_color = tuple(
            doc["maps"].get("emission", {}).get("aurora_color", (0.85, 0.35, 0.60))
        )

        physical = doc["physical"]
        # Physically derived relief: height_scale is the full height-map range
        # as a fraction of the radius.
        displacement_scale = physical["height_scale"] * self.radius * self.exaggeration
        clearance = max(displacement_scale * (1.0 - physical["height_midlevel"]), 0.0)

        # Rig empty: tilt and spin live here.
        rig = bpy.data.objects.new(f"{doc['name']}_rig", None)
        context.collection.objects.link(rig)
        rig.rotation_euler = (self.axial_tilt, 0.0, 0.0)

        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=self.mesh_segments,
            ring_count=self.mesh_segments // 2,
            radius=self.radius,
        )
        planet = bpy.context.active_object
        planet.name = doc["name"]
        bpy.ops.object.shade_smooth()
        planet.rotation_euler = (0.0, 0.0, self.longitude_offset)
        planet.parent = rig

        built = material.build_planet_material(
            f"{doc['name']}_surface",
            color_img,
            height_img,
            use_displacement=self.use_displacement,
            displacement_scale=displacement_scale,
            height_midlevel=physical["height_midlevel"],
            procedural_mapping=self.mapping == "PROCEDURAL",
            limb_darkening=self.limb_darkening,
            limb_haze=self.limb_haze,
            haze_color=tuple(doc.get("atmosphere_hint", {}).get("rim_color", (0.6, 0.7, 1.0))),
            emission_img=emission_img,
            emission_strength=self.emission_strength,
            aurora_color=aurora_color,
            aurora_on_surface=not self.aurora_shell,
        )
        planet.data.materials.append(built.material)

        # Sequence wiring: reconfigure each animated texture node's image_user.
        # See compat.configure_image_sequence for the frame_start/offset formula
        # (frame_offset = -1 maps scene frame 1 -> 0000-based picture 0).
        if color_seq is not None:
            compat.configure_image_sequence(built.color_node, frame_duration=frame_count)
        if height_seq is not None and built.height_node is not None:
            compat.configure_image_sequence(built.height_node, frame_duration=frame_count)
        if emission_seq is not None and built.emission_node is not None:
            compat.configure_image_sequence(built.emission_node, frame_duration=frame_count)

        if emission_img is not None and self.aurora_shell:
            aurora = atmosphere.build_aurora_shell(
                f"{doc['name']}_aurora", self.radius, emission_img,
                aurora_color=aurora_color,
                strength=self.emission_strength,
                clearance=clearance,
            )
            aurora.parent = rig

        # Rings (T16): a flat annulus sampling the rings.exr radial strip, in the
        # planet's equatorial plane and parented to the rig so it tilts/spins with
        # the planet. Blender-only -- there is no equirect ring data.
        ring_extent = manifest_schema.ring_extent(doc)
        if self.build_rings and ring_extent is not None:
            ring_path = manifest_schema.map_path(doc, "rings")
            ring_img = bpy.data.images.load(str(ring_path), check_existing=True)
            inner_km, outer_km = ring_extent
            # Scale km -> Blender units by the planet's own radius (radius_km).
            radius_km = physical.get("radius_km", 69911.0)
            scale = self.radius / radius_km
            ring = atmosphere.build_ring_annulus(
                f"{doc['name']}_rings", ring_img,
                inner_km * scale, outer_km * scale,
            )
            ring.parent = rig

        if self.use_displacement:
            compat.enable_adaptive_subdivision(planet, context.scene)

        hint = doc.get("atmosphere_hint", {})
        rim_color = tuple(hint.get("rim_color", (0.55, 0.65, 1.0)))
        rim_strength = float(hint.get("rim_strength", 0.4))
        sun_dir = self._sun_direction(context)
        if self.atmosphere_mode == "VOLUME":
            shell = atmosphere.build_volume_atmosphere(
                f"{doc['name']}_atmosphere", self.radius,
                rim_color=rim_color, strength=rim_strength, clearance=clearance,
            )
            shell.parent = rig
        elif self.atmosphere_mode == "RIM":
            shell = atmosphere.build_rim_atmosphere(
                f"{doc['name']}_atmosphere", self.radius,
                rim_color=rim_color, strength=rim_strength,
                sun_direction=sun_dir, clearance=clearance,
            )
            shell.parent = rig

        if self.demo_scene:
            self._build_demo_scene(context, rig)

        self.report({"INFO"}, f"imported {doc['name']} ({doc['resolution'][0]}px)")
        return {"FINISHED"}

    @staticmethod
    def _sun_direction(context) -> tuple[float, float, float]:
        """Direction TOWARD the sun (a sun lamp shines along its local -Z)."""
        for obj in context.scene.objects:
            if obj.type == "LIGHT" and obj.data.type == "SUN":
                vec = obj.matrix_world.to_3x3() @ _NEG_Z
                return (-vec[0], -vec[1], -vec[2])
        return (0.5, -0.5, 0.7)

    def _build_demo_scene(self, context, rig) -> None:
        scene = context.scene
        world = scene.world or bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True
        bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
        if bg is not None:
            bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bg.inputs["Strength"].default_value = 0.0

        sun_data = bpy.data.lights.new("Sun", "SUN")
        sun_data.energy = 4.0
        sun_data.angle = math.radians(0.25)
        sun = bpy.data.objects.new("Sun", sun_data)
        context.collection.objects.link(sun)
        sun.rotation_euler = (math.radians(60.0), 0.0, math.radians(140.0))

        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        context.collection.objects.link(cam)
        cam.location = (0.0, -4.2 * self.radius, 0.9 * self.radius)
        cam.rotation_euler = (math.radians(78.0), 0.0, 0.0)
        scene.camera = cam

        compat.set_view_transform_agx(scene)


_NEG_Z = None  # set lazily; mathutils only exists inside Blender


def _init_constants():
    global _NEG_Z
    import mathutils

    _NEG_Z = mathutils.Vector((0.0, 0.0, -1.0))


def menu_func_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_gasgiant.bl_idname, text="Gas Giant Map Set (.json)")


def register():
    _init_constants()
    bpy.utils.register_class(IMPORT_SCENE_OT_gasgiant)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_SCENE_OT_gasgiant)

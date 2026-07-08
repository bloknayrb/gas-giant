"""Atmosphere builders.

Default for Cycles: a thin Volume Scatter shell — light-aware for free
(correct terminator scattering, dark night limb, soft falloff), cheap at
default volume bounces. The Facing->Emission rim is the EEVEE / fast-stylized
fallback; its emission is gated by a sun-direction dot product captured at
import time (a static gate: re-import or edit the node if the sun moves a lot).
VDB was evaluated and rejected: a homogeneous shell IS a volume-scatter mesh
shell — identical math, large files.
"""

from __future__ import annotations

import math

import bpy

from . import compat
from .material import _new, _spherical_uv_nodes


def _shell_object(name: str, radius: float, segments: int) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments, ring_count=segments // 2, radius=radius
    )
    shell = bpy.context.active_object
    shell.name = name
    bpy.ops.object.shade_smooth()
    shell.visible_shadow = False  # else: shadow ring artifact on the planet
    return shell


def build_volume_atmosphere(
    name: str,
    radius: float,
    *,
    rim_color: tuple[float, float, float],
    strength: float,
    clearance: float,
) -> bpy.types.Object:
    shell_radius = radius * (1.0 + 0.015) + clearance
    shell = _shell_object(name, shell_radius, 96)

    mat = bpy.data.materials.new(f"{name}_volume")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        nodes.remove(bsdf)
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")

    transparent = _new(nodes, "ShaderNodeBsdfTransparent", -200, 200)
    links.new(transparent.outputs[0], output.inputs["Surface"])

    scatter = _new(nodes, "ShaderNodeVolumeScatter", -200, -50)
    scatter.inputs["Color"].default_value = (*rim_color, 1.0)
    # Optical depth ~0.7 along the limb path (~0.31 * radius for this shell
    # thickness) reads as a real atmosphere without nuking render times.
    scatter.inputs["Density"].default_value = 2.2 / max(radius, 1e-3) * strength
    aniso = compat.find_input(scatter, "Anisotropy")
    if aniso is not None:
        aniso.default_value = 0.55
    links.new(scatter.outputs[0], output.inputs["Volume"])

    compat.set_transparent_render_method(mat)
    shell.data.materials.append(mat)
    return shell


def build_aurora_shell(
    name: str,
    radius: float,
    emission_img: bpy.types.Image,
    *,
    aurora_color: tuple[float, float, float],
    strength: float,
    clearance: float,
) -> bpy.types.Object:
    """A transparent shell at ~1.03 R emitting the emission map's
    alpha-channel aurora — the real aurora sits ~1000 km above the cloud
    deck, so baking it into the surface reads wrong at the terminator.
    NOTE: emission is not sun-gated; the arc is dayside-NEGLIGIBLE at
    default strength against the lit disc, not dayside-clean."""
    shell = _shell_object(name, radius * 1.03 + clearance, 96)

    mat = bpy.data.materials.new(f"{name}_aurora")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        nodes.remove(bsdf)
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")

    uv_socket = _spherical_uv_nodes(nt)
    tex = _new(nodes, "ShaderNodeTexImage", -250, 0)
    tex.image = emission_img
    tex.extension = "REPEAT"
    compat.set_colorspace(emission_img, "non-color")
    compat.set_channel_packed(emission_img)
    links.new(uv_socket, tex.inputs["Vector"])

    fac = _new(nodes, "ShaderNodeMath", -60, 0)
    fac.operation = "MINIMUM"
    links.new(tex.outputs["Alpha"], fac.inputs[0])
    fac.inputs[1].default_value = 1.0

    emission = _new(nodes, "ShaderNodeEmission", -60, -180)
    emission.inputs["Color"].default_value = (*aurora_color, 1.0)
    emission.inputs["Strength"].default_value = 3.0 * strength

    transparent = _new(nodes, "ShaderNodeBsdfTransparent", -60, 140)
    mix = _new(nodes, "ShaderNodeMixShader", 140, 0)
    links.new(fac.outputs[0], mix.inputs["Fac"])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])

    compat.set_transparent_render_method(mat)
    shell.data.materials.append(mat)
    return shell


def _annulus_mesh(
    name: str, inner: float, outer: float, segments: int
) -> bpy.types.Object:
    """A flat ring (annulus) in the planet's equatorial plane (XY, spin axis Z),
    built vertex-by-vertex so the UVs map radius -> V (the ring strip's long
    axis) and angle -> U. ``segments`` quads go around; two radial rings (inner,
    outer) close them."""
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []  # per-loop, filled alongside faces
    faces: list[tuple[int, int, int, int]] = []

    for i in range(segments):
        ang = 2.0 * math.pi * i / segments
        c, s = math.cos(ang), math.sin(ang)
        verts.append((inner * c, inner * s, 0.0))  # 2*i   : inner
        verts.append((outer * c, outer * s, 0.0))  # 2*i+1 : outer

    for i in range(segments):
        j = (i + 1) % segments
        a, b = 2 * i, 2 * i + 1        # this spoke: inner, outer
        c, d = 2 * j, 2 * j + 1        # next spoke: inner, outer
        faces.append((a, b, d, c))
        u0, u1 = i / segments, (i + 1) / segments
        # loop order matches the face vertex order (a, b, d, c):
        # V = 0 at inner radius, 1 at outer radius.
        uvs.extend([(u0, 0.0), (u0, 1.0), (u1, 1.0), (u1, 0.0)])

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], [f for f in faces])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop_idx, uv in enumerate(uvs):
        uv_layer.data[loop_idx].uv = uv

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def build_ring_annulus(
    name: str,
    ring_img: bpy.types.Image,
    inner_radius: float,
    outer_radius: float,
    *,
    segments: int = 192,
) -> bpy.types.Object:
    """A flat Saturn-style ring annulus sampling the ``rings.exr`` radial strip.

    The strip's alpha channel is coverage: transparent gaps (Cassini division,
    Encke) let light through, so with ``use_transparent_shadow`` the ring casts
    its structured shadow onto the planet in Cycles. RGB is the lit ice colour.
    Radius (mesh UV V) maps to the strip's long axis; U wraps tangentially."""
    ring = _annulus_mesh(name, inner_radius, outer_radius, segments)

    mat = bpy.data.materials.new(f"{name}_material")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")

    tex = _new(nodes, "ShaderNodeTexImage", -400, 0)
    tex.image = ring_img
    tex.extension = "CLIP"  # outside the annulus UV: transparent, not tiled
    compat.set_colorspace(ring_img, "non-color")
    compat.set_channel_packed(ring_img)  # alpha is coverage data, not premul

    base_color = compat.find_input(bsdf, "Base Color")
    if base_color is not None:
        links.new(tex.outputs["Color"], base_color)
    alpha_in = compat.find_input(bsdf, "Alpha")
    if alpha_in is not None:
        links.new(tex.outputs["Alpha"], alpha_in)
    rough = compat.find_input(bsdf, "Roughness")
    if rough is not None:
        rough.default_value = 0.9
    spec = compat.find_input(bsdf, "Specular IOR Level", "Specular")
    if spec is not None:
        spec.default_value = 0.1
    links.new(bsdf.outputs[0], output.inputs["Surface"])

    compat.set_transparent_render_method(mat)
    compat.set_transparent_shadow(mat)
    ring.data.materials.append(mat)
    return ring


def build_rim_atmosphere(
    name: str,
    radius: float,
    *,
    rim_color: tuple[float, float, float],
    strength: float,
    sun_direction: tuple[float, float, float],
    clearance: float,
) -> bpy.types.Object:
    shell = _shell_object(name, radius * 1.012 + clearance, 96)

    mat = bpy.data.materials.new(f"{name}_rim")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        nodes.remove(bsdf)
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")

    layer = _new(nodes, "ShaderNodeLayerWeight", -700, 0)
    layer.inputs["Blend"].default_value = 0.25
    power = _new(nodes, "ShaderNodeMath", -520, 0)
    power.operation = "POWER"
    links.new(layer.outputs["Facing"], power.inputs[0])
    power.inputs[1].default_value = 3.0

    # Sun gate: emission only where the surface faces the sun (a static
    # direction captured at import). Without this the night limb glows.
    normal = _new(nodes, "ShaderNodeNewGeometry", -700, -220)
    dot = _new(nodes, "ShaderNodeVectorMath", -520, -220)
    dot.operation = "DOT_PRODUCT"
    links.new(normal.outputs["Normal"], dot.inputs[0])
    dot.inputs[1].default_value = sun_direction
    gate = _new(nodes, "ShaderNodeMath", -340, -220)
    gate.operation = "MAXIMUM"
    links.new(dot.outputs["Value"], gate.inputs[0])
    gate.inputs[1].default_value = 0.0

    fac = _new(nodes, "ShaderNodeMath", -340, 0)
    fac.operation = "MULTIPLY"
    links.new(power.outputs[0], fac.inputs[0])
    links.new(gate.outputs[0], fac.inputs[1])

    emission = _new(nodes, "ShaderNodeEmission", -160, -80)
    emission.inputs["Color"].default_value = (*rim_color, 1.0)
    emission.inputs["Strength"].default_value = 2.0 * strength

    transparent = _new(nodes, "ShaderNodeBsdfTransparent", -160, 100)
    mix = _new(nodes, "ShaderNodeMixShader", 40, 0)
    links.new(fac.outputs[0], mix.inputs["Fac"])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])

    compat.set_transparent_render_method(mat)
    shell.data.materials.append(mat)
    return shell

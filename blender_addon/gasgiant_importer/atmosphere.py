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

import bpy

from . import compat
from .material import _new


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

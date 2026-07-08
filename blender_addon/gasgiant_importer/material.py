"""Planet material construction.

Hard rules (these are what survived Blender 4.0/5.0 API breakage):
- Nodes are created BY TYPE, never looked up by display name.
- Sockets are resolved through compat.find_input alias lists.
- Detail never double-counts: displacement ON uses the height-driven
  Displacement output (no bump from the same data); displacement OFF uses a
  Bump node (no tangent-space normal map — those degenerate at equirect
  poles).
- Specular ~0: a gas cloud deck has no meaningful specular reflection.
- 'Procedural spherical' mapping computes equirect UV per pixel from object
  coordinates (atan2/asin) — exact at the poles where mesh UV interpolation
  pinches; mesh UVs remain for EEVEE (screen-space-derivative seam at the
  atan2 wrap there).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import bpy

from . import compat


class PlanetMaterial(NamedTuple):
    """The built material plus handles to its image-texture nodes.

    The importer needs the texture-node handles to reconfigure them for an
    animated sequence (``image.source = 'SEQUENCE'`` + ``image_user`` fields);
    they are otherwise created internally and unreachable. ``height_node`` /
    ``emission_node`` are ``None`` when the corresponding map is absent.
    """

    material: bpy.types.Material
    color_node: bpy.types.Node
    height_node: bpy.types.Node | None
    emission_node: bpy.types.Node | None


def _new(nodes: bpy.types.Nodes, type_name: str, x: float, y: float) -> bpy.types.Node:
    node = nodes.new(type_name)
    node.location = (x, y)
    return node


def _spherical_uv_nodes(nt: bpy.types.NodeTree) -> bpy.types.NodeSocket:
    """Object coords -> equirect UV (u = atan2(y,x)/2pi + 0.5,
    v = 0.5 + asin(z)/pi, matching the exporter's texel convention)."""
    nodes, links = nt.nodes, nt.links
    coord = _new(nodes, "ShaderNodeTexCoord", -1500, 0)
    norm = _new(nodes, "ShaderNodeVectorMath", -1320, 0)
    norm.operation = "NORMALIZE"
    links.new(coord.outputs["Object"], norm.inputs[0])
    sep = _new(nodes, "ShaderNodeSeparateXYZ", -1140, 0)
    links.new(norm.outputs["Vector"], sep.inputs[0])

    atan = _new(nodes, "ShaderNodeMath", -960, 120)
    atan.operation = "ARCTAN2"
    links.new(sep.outputs["Y"], atan.inputs[0])
    links.new(sep.outputs["X"], atan.inputs[1])
    u_div = _new(nodes, "ShaderNodeMath", -780, 120)
    u_div.operation = "DIVIDE"
    links.new(atan.outputs[0], u_div.inputs[0])
    u_div.inputs[1].default_value = 2.0 * math.pi
    u_add = _new(nodes, "ShaderNodeMath", -600, 120)
    u_add.operation = "ADD"
    links.new(u_div.outputs[0], u_add.inputs[0])
    u_add.inputs[1].default_value = 0.5

    asin = _new(nodes, "ShaderNodeMath", -960, -80)
    asin.operation = "ARCSINE"
    links.new(sep.outputs["Z"], asin.inputs[0])
    v_div = _new(nodes, "ShaderNodeMath", -780, -80)
    v_div.operation = "DIVIDE"
    links.new(asin.outputs[0], v_div.inputs[0])
    v_div.inputs[1].default_value = math.pi
    v_add = _new(nodes, "ShaderNodeMath", -600, -80)
    v_add.operation = "ADD"
    links.new(v_div.outputs[0], v_add.inputs[0])
    v_add.inputs[1].default_value = 0.5

    combine = _new(nodes, "ShaderNodeCombineXYZ", -420, 0)
    links.new(u_add.outputs[0], combine.inputs["X"])
    links.new(v_add.outputs[0], combine.inputs["Y"])
    return combine.outputs["Vector"]


def build_planet_material(
    name: str,
    color_img: bpy.types.Image,
    height_img: bpy.types.Image | None,
    *,
    use_displacement: bool,
    displacement_scale: float,
    height_midlevel: float,
    procedural_mapping: bool,
    limb_darkening: float,
    limb_haze: float,
    haze_color: tuple[float, float, float],
    emission_img: bpy.types.Image | None = None,
    emission_strength: float = 1.0,
    aurora_color: tuple[float, float, float] = (0.85, 0.35, 0.60),
    aurora_on_surface: bool = True,
) -> PlanetMaterial:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")

    uv_socket = _spherical_uv_nodes(nt) if procedural_mapping else None

    tex_color = _new(nodes, "ShaderNodeTexImage", -250, 300)
    tex_color.image = color_img
    tex_color.extension = "REPEAT"
    compat.set_colorspace(color_img, "srgb")
    if uv_socket is not None:
        links.new(uv_socket, tex_color.inputs["Vector"])

    # Limb darkening + high-altitude haze tint toward the limb:
    # limb factor = (1 - LayerWeight.Facing), scaled per effect.
    color_src = tex_color.outputs["Color"]
    if limb_darkening > 0.0 or limb_haze > 0.0:
        layer = _new(nodes, "ShaderNodeLayerWeight", -250, 60)
        limb = _new(nodes, "ShaderNodeMath", -80, 60)
        limb.operation = "SUBTRACT"
        limb.inputs[0].default_value = 1.0
        links.new(layer.outputs["Facing"], limb.inputs[1])

        def _limb_factor(strength: float, y: float):
            mul = _new(nodes, "ShaderNodeMath", 80, y)
            mul.operation = "MULTIPLY"
            links.new(limb.outputs[0], mul.inputs[0])
            mul.inputs[1].default_value = strength
            return mul.outputs[0]

        if limb_haze > 0.0:
            mix_haze = _new(nodes, "ShaderNodeMix", 260, 240)
            mix_haze.data_type = "RGBA"
            links.new(_limb_factor(limb_haze, 280), mix_haze.inputs["Factor"])
            links.new(color_src, mix_haze.inputs["A"])
            mix_haze.inputs["B"].default_value = (*haze_color, 1.0)
            color_src = mix_haze.outputs["Result"]
        if limb_darkening > 0.0:
            dark = _new(nodes, "ShaderNodeMix", 440, 240)
            dark.data_type = "RGBA"
            dark.blend_type = "MULTIPLY"
            links.new(_limb_factor(limb_darkening, 120), dark.inputs["Factor"])
            links.new(color_src, dark.inputs["A"])
            dark.inputs["B"].default_value = (0.35, 0.35, 0.38, 1.0)
            color_src = dark.outputs["Result"]

    tex_height: bpy.types.Node | None = None
    tex_em: bpy.types.Node | None = None

    base_color = compat.find_input(bsdf, "Base Color")
    links.new(color_src, base_color)

    spec = compat.find_input(bsdf, "Specular IOR Level", "Specular")
    if spec is not None:
        spec.default_value = 0.0
    rough = compat.find_input(bsdf, "Roughness")
    if rough is not None:
        rough.default_value = 0.9

    if height_img is not None:
        tex_height = _new(nodes, "ShaderNodeTexImage", -250, -260)
        tex_height.image = height_img
        tex_height.extension = "REPEAT"
        compat.set_colorspace(height_img, "non-color")
        if uv_socket is not None:
            links.new(uv_socket, tex_height.inputs["Vector"])
        if use_displacement:
            disp = _new(nodes, "ShaderNodeDisplacement", 100, -260)
            links.new(tex_height.outputs["Color"], disp.inputs["Height"])
            disp.inputs["Midlevel"].default_value = height_midlevel
            disp.inputs["Scale"].default_value = displacement_scale
            links.new(disp.outputs["Displacement"], output.inputs["Displacement"])
        else:
            bump = _new(nodes, "ShaderNodeBump", 100, -260)
            links.new(tex_height.outputs["Color"], bump.inputs["Height"])
            bump.inputs["Strength"].default_value = 0.25
            bump.inputs["Distance"].default_value = displacement_scale
            normal_in = compat.find_input(bsdf, "Normal")
            if normal_in is not None:
                links.new(bump.outputs["Normal"], normal_in)

    if emission_img is not None:
        tex_em = _new(nodes, "ShaderNodeTexImage", -250, -560)
        tex_em.image = emission_img
        tex_em.extension = "REPEAT"
        compat.set_colorspace(emission_img, "non-color")
        # Alpha is an independent data mask (aurora intensity), not coverage:
        # without CHANNEL_PACKED the premultiplied path corrupts the RGB.
        compat.set_channel_packed(emission_img)
        if uv_socket is not None:
            links.new(uv_socket, tex_em.inputs["Vector"])
        em_src = tex_em.outputs["Color"]
        if aurora_on_surface:
            # + alpha * aurora_color (the importer's shell mode lifts the
            # aurora off the surface instead and sets aurora_on_surface=False).
            scale = _new(nodes, "ShaderNodeVectorMath", -60, -560)
            scale.operation = "SCALE"
            scale.inputs[0].default_value = aurora_color
            links.new(tex_em.outputs["Alpha"], scale.inputs["Scale"])
            add = _new(nodes, "ShaderNodeVectorMath", 100, -560)
            add.operation = "ADD"
            links.new(em_src, add.inputs[0])
            links.new(scale.outputs["Vector"], add.inputs[1])
            em_src = add.outputs["Vector"]
        em_color = compat.find_input(bsdf, "Emission Color", "Emission")
        em_strength = compat.find_input(bsdf, "Emission Strength")
        if em_color is not None:
            links.new(em_src, em_color)
            if em_strength is not None:
                em_strength.default_value = emission_strength
        else:
            # Exotic future socket rename: a separate Emission shader added
            # to the surface — never a silent drop of the map.
            emit = _new(nodes, "ShaderNodeEmission", 60, -700)
            links.new(em_src, emit.inputs["Color"])
            emit.inputs["Strength"].default_value = emission_strength
            add_sh = _new(nodes, "ShaderNodeAddShader", 260, -700)
            links.new(bsdf.outputs[0], add_sh.inputs[0])
            links.new(emit.outputs[0], add_sh.inputs[1])
            links.new(add_sh.outputs[0], output.inputs["Surface"])

    compat.set_displacement_method(mat, use_displacement)
    return PlanetMaterial(mat, tex_color, tex_height, tex_em)

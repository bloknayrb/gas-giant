#version 430
// Blits the engine's preview texture through the selected view transform
// into the display texture shown by imgui.

#include "agx.glsl"

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_image;
uniform int u_mode;     // 0 standard, 1 AgX approximation
uniform int u_channel;  // 0 rgb, 1 rrr, 2 ggg, 3 bbb, 4 aaa, 5 rgb + a*u_aurora
uniform vec3 u_aurora;  // Emission channel only (u_channel 5): composites the
                        // alpha-channel aurora intensity as a * u_aurora, the
                        // same alpha-times-aurora_color lift the Blender
                        // importer applies on its aurora shell. Zero for every
                        // other channel, so channels 0-4 are untouched.

void main() {
    // NORTH-UP: source texel row 0 is lat +90 (core/domain.py) and imgui.image
    // draws row 0 at the top of the image, so v_uv maps straight through -- NO
    // vertical flip. A `1.0 - v_uv.y` here rendered the whole equirect upside
    // down (pinned by test_view_transform_is_north_up).
    vec2 uv = v_uv;
    vec4 texel = texture(u_image, uv);
    vec3 color;
    if      (u_channel == 1) color = texel.rrr;
    else if (u_channel == 2) color = texel.ggg;
    else if (u_channel == 3) color = texel.bbb;
    else if (u_channel == 4) color = texel.aaa;
    else if (u_channel == 5) color = texel.rgb + texel.a * u_aurora;
    else                     color = texel.rgb;
    frag_color = vec4(viewTransform(color, u_mode), 1.0);
}

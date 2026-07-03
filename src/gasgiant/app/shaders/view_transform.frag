#version 430
// Blits the engine's preview texture through the selected view transform
// into the display texture shown by imgui.

#include "agx.glsl"

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_image;
uniform int u_mode;     // 0 standard, 1 AgX approximation
uniform int u_channel;  // 0 rgb, 1 rrr, 2 ggg, 3 bbb, 4 aaa

void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec4 texel = texture(u_image, uv);
    vec3 color;
    if      (u_channel == 1) color = texel.rrr;
    else if (u_channel == 2) color = texel.ggg;
    else if (u_channel == 3) color = texel.bbb;
    else if (u_channel == 4) color = texel.aaa;
    else                     color = texel.rgb;
    frag_color = vec4(viewTransform(color, u_mode), 1.0);
}

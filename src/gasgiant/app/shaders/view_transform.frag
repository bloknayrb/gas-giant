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
    // Sample the source unflipped. The source texture is north-up (render_maps
    // export + a cast-hero probe confirm +40 lat renders in the northern rows),
    // as are the marker overlay (viewport.py: top = +90) and the sphere QA view.
    // A `1.0 - v_uv.y` flip here made the DISPLAY alone south-up, so a placed
    // storm rendered opposite its marker. Confirmed by a single-storm on-screen
    // test: one hero placed at the top edge (toast lat +32.1) rendered in the
    // BOTTOM half under the flip. Unflipped, marker and storm coincide and the
    // preview matches the export. Do NOT re-add the flip.
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

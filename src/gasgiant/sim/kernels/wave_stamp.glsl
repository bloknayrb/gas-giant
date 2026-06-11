// Equatorial-wave tracer stamps (equirect domain only), shared by init.comp
// and the MacCormack correct pass: festoon plumes at Rossby-wave crests,
// 5-micron hot spots at the troughs, and the ribbon's thin dark line.
// These are continually regenerated (part of the relaxation target) — the
// jets and shear then advect them into the characteristic hooks.

uniform float u_fest_amp;     // 0 disables
uniform float u_fest_lat;     // belt-edge latitude of the wave train
uniform float u_fest_k;
uniform float u_fest_phase;
uniform float u_hotspot_depth;

uniform float u_rib_amp;      // 0 disables
uniform float u_rib_lat;
uniform float u_rib_k;
uniform float u_rib_phase;

// Returns (dT0, dT1, dT3) at (lon, lat).
vec3 waveStamp(vec2 ll) {
    vec3 d = vec3(0.0);

    if (u_fest_amp > 0.0) {
        float crest = sin(u_fest_k * ll.x + u_fest_phase);
        // Festoon plumes: blue-gray streamers (negative T3) rooted on the
        // belt edge, reaching ~equatorward of it at the wave crests.
        float plume_center = u_fest_lat - sign(u_fest_lat) * 0.045;
        float plume = exp(-pow((ll.y - plume_center) / 0.05, 2.0));
        float c = max(crest, 0.0);
        d.z -= u_fest_amp * 0.7 * plume * c * c;
        // Hot spots: compact cloud-free holes at the troughs — low cloud
        // tops, darkened color.
        float hole = exp(-pow((ll.y - u_fest_lat) / 0.025, 2.0));
        float t = max(-crest, 0.0);
        float spot = u_fest_amp * u_hotspot_depth * hole * t * t * t * t;
        d.y -= 0.5 * spot;
        d.x -= 0.35 * spot;
    }

    if (u_rib_amp > 0.0) {
        // Ribbon: a thin dark line riding the meandering jet; the meander in
        // psi folds it into the wave shape.
        float line = exp(-pow((ll.y - u_rib_lat) / 0.008, 2.0));
        d.x -= u_rib_amp * 0.14 * line;
    }

    return d;
}

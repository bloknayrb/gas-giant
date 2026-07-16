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

#ifdef FESTOON2
// Hero-adjacent festoon train (waves.festoon_hero_strength): a SECOND train
// rooted on the interior band edge nearest the hero storm, so its streamers
// weave through the hero's wake lane with tails brushing the collar (the
// reference's SEB-edge streamers next to the GRS). PLUMES ONLY, T3 ONLY:
// the hot-spot hole would center exactly on the root edge — INSIDE the hero
// plateau for a hollow-straddling hero — and T0/T1 writes would also
// pollute the wake-fold T0 statistics. Compiled as a preprocessor variant
// (predicate in solver._domain_defines: strength > 0 AND a facade-selected
// root edge exists) so the default program text is unchanged —
// byte-identical by construction. NOTE: unlike the primary train there is
// no psi-side meander for this one (deliberate: at storm latitudes the
// meander is feather-inert in vorticity mode anyway; the belt-edge jet
// supplies the shear that hooks the plumes).
uniform float u_fest2_amp;
uniform float u_fest2_lat;
uniform float u_fest2_k;
uniform float u_fest2_phase;
#endif

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

#ifdef FESTOON2
    {
        float crest2 = sin(u_fest2_k * ll.x + u_fest2_phase);
        // Per-plume seeded amplitude jitter (slow longitude envelope,
        // decorrelated from the crests by the 1.7x phase reuse): some plumes
        // assert, some nearly vanish — an even train reads as a mechanical
        // comb (the falsified-baroclinic tell this train must not repeat).
        float pj = 0.4 + 0.6 * pow(0.5 + 0.5 * sin(3.0 * ll.x
                                                   + u_fest2_phase * 1.7), 2.0);
        // Same dip convention as the primary train: plume center sits
        // equatorward of the root edge — INTO the belt and the wake lane.
        float pc2 = u_fest2_lat - sign(u_fest2_lat) * 0.045;
        float plume2 = exp(-pow((ll.y - pc2) / 0.05, 2.0));
        float c2 = max(crest2, 0.0);
        d.z -= u_fest2_amp * 0.7 * pj * plume2 * c2 * c2;
    }
#endif

    if (u_rib_amp > 0.0) {
        // Ribbon: a thin dark line riding the meandering jet; the meander in
        // psi folds it into the wave shape.
        float line = exp(-pow((ll.y - u_rib_lat) / 0.008, 2.0));
        d.x -= u_rib_amp * 0.14 * line;
    }

    return d;
}

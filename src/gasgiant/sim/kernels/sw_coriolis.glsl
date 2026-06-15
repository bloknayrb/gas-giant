// sw_coriolis.glsl — trapezoidal (Cayley) Coriolis rotation helpers, shared by
// the M2 SI Helmholtz-RHS and velocity-backsub kernels.  Include AFTER
// sw_common.glsl.  Requires the includer to declare:
//   uniform float u_omega;   // planetary rotation rate Omega
//   uniform float u_dt;      // time step
//
// Ports coriolis_trapezoidal / coriolis_sandwich from shallow_water_ref.py.
//
// The sandwich is: collapse v from v-faces to centres (v_c = 0.5*(v[j]+v[j+1])),
// trapezoidal-rotate (u, v_c) at the cell-centre latitude, then scatter v_c back
// to v-faces (v_new[k] = 0.5*(v_c_new[k-1]+v_c_new[k]); poles 0).

// f = 2*omega*sin(phi_c[row]); phi_c[row] = PI/2 - (row+0.5)*PI/H.
float cor_f(int row, int H) {
    float phi_c = 0.5 * PI - (float(row) + 0.5) * PI / float(H);
    return 2.0 * u_omega * sin(phi_c);
}

// Cayley rotation: u_new = ((1-a^2)u + 2a v_c)/denom, where a = 0.5*f*dt.
float cor_u_new(float u, float v_c, int row, int H) {
    float a = 0.5 * cor_f(row, H) * u_dt;
    float denom = 1.0 + a * a;
    return ((1.0 - a * a) * u + 2.0 * a * v_c) / denom;
}

// Cayley rotation: v_c_new = ((1-a^2)v_c - 2a u)/denom.
float cor_vc_new(float u, float v_c, int row, int H) {
    float a = 0.5 * cor_f(row, H) * u_dt;
    float denom = 1.0 + a * a;
    return ((1.0 - a * a) * v_c - 2.0 * a * u) / denom;
}

"""W13/W14 detail-CHARACTER crux spike (MEASUREMENT ONLY — touches no render path).

Hypothesis (docs/roadmap.md, "Research direction: detail CHARACTER = sim-advected
high-res tracer"): advecting ONE extra *high-resolution* passive tracer through the
`gas_giant_warm` vorticity solver's EVOLVING velocity field — decoupled from the
dynamics grid — folds an isotropic seed into ORIENTED (zonally-elongated filamentary)
structure, the morphology a frozen-field render trick cannot produce (F17, FALSIFIED).

The crux gate the roadmap pre-registers: measure the structure-tensor ORIENTATION
COHERENCE of the resulting tracer and check whether it crosses the F17 bar
    kinematic/isotropic control  ~0.14
    vorticity-solver tracer       0.384   (the go bar)
    Cassini reference             0.62    (strong target)
Go/no-go on that number BEFORE committing the multi-session subsystem build.

METRIC: identical operator to the calibrated project metric — we import
`scripts/measure_morphology.py::coher` (structure-tensor coherence c=(l1-l2)/(l1+l2),
horizontality-weighted, energy-weighted mean). Running the SAME operator on the seed
(isotropic control) and the advected tracer, plus a rot90 orientation control, is the
clean apples-to-apples separation this spike exists to produce.

METHOD:
  * Build a real `gas_giant_warm` Simulation (vorticity mode).
  * Allocate ONE extra high-res R32F tracer, decoupled from the dynamics grid
    (tracer width = TRACER_MULT x sim resolution), seeded with an ISOTROPIC
    band-pass noise field (radial Fourier annulus => provably isotropic).
  * Each dev step: advance the solver one step, then advect the high-res tracer by
    the solver's *current* equirect velocity texture with a THROWAWAY semi-Lagrangian
    (RK2 backtrace + bicubic Catmull-Rom) compute kernel compiled from a source string
    — the same backtrace math as sim/kernels/advect.comp (DOMAIN 0), inlined here so
    the spike is self-contained and imports no production kernel.
  * After the run, measure `coher` on a tropical-belt crop: advected vs seed control
    vs rot90(advected) orientation control.

FIDELITY: full gas_giant_warm is 4096 / 700 steps / 48 SOR iters x 3 domains — wholly
intractable under software GL (llvmpipe ~150x slower than native). This spike runs a
REDUCED-FIDELITY proxy (small dynamics grid, fewer steps) chosen to finish in minutes
under `xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1`. A reduced proxy that
cleanly separates advected-vs-control is a valid crux result; a full-res confirmation
needs a native GPU. Knobs below are CLI-overridable.

Usage:
    xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 \
        uv run python scripts/spike_detail_character.py --res 256 --steps 200 --tracer-mult 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_morphology import coher  # noqa: E402  the calibrated project metric

from gasgiant.engine.facade import Simulation  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402

# Throwaway advection kernel: RK2 semi-Lagrangian backtrace + bicubic Catmull-Rom,
# equirect (DOMAIN 0) backtrace math copied from sim/kernels/advect.comp. Single
# R32F channel; velocity sampled in normalized UV so the (coarser) dynamics velocity
# upsamples onto the high-res tracer grid — the roadmap's "1024-grid strain folds a
# 4K scalar" decoupling. Ping-ponged (read u_src sampler, write out_tracer image).
_ADVECT_SRC = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32f, binding = 0) writeonly uniform image2D out_tracer;
uniform sampler2D u_src;   // tracer scalar (r32f, linear, repeat_x)
uniform sampler2D u_vel;   // solver velocity (rg32f, linear, repeat_x) = (u east, v north)
uniform ivec2 u_size;      // TRACER size
uniform float u_dt;
const float PI = 3.14159265358979323846;

int wrapX(int x, int w) { return ((x % w) + w) % w; }
int clampY(int y, int h) { return clamp(y, 0, h - 1); }
float fetchT(int x, int y) {
    return texelFetch(u_src, ivec2(wrapX(x, u_size.x), clampY(y, u_size.y)), 0).r;
}
vec4 crW(float t) {
    float t2 = t * t; float t3 = t2 * t;
    return vec4(-0.5 * t3 + t2 - 0.5 * t,
                1.5 * t3 - 2.5 * t2 + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                0.5 * t3 - 0.5 * t2);
}
float sampleCR(vec2 pos) {
    vec2 grid = pos - 0.5; vec2 base = floor(grid); vec2 f = grid - base;
    vec4 wx = crW(f.x); vec4 wy = crW(f.y);
    float acc = 0.0;
    for (int j = 0; j < 4; ++j) {
        float row = 0.0; int y = int(base.y) + j - 1;
        for (int i = 0; i < 4; ++i) row += wx[i] * fetchT(int(base.x) + i - 1, y);
        acc += wy[j] * row;
    }
    return acc;
}
vec2 backtrace(vec2 pixPos, float dt) {
    vec2 size = vec2(u_size); vec2 uvScale = 1.0 / size;
    vec2 ll = vec2((pixPos.x / size.x) * 2.0 * PI - PI,
                   0.5 * PI - (pixPos.y / size.y) * PI);
    vec2 vel = texture(u_vel, pixPos * uvScale).rg;
    float cosl = max(cos(ll.y), 0.017);
    vec2 mid = ll + vec2(-0.5 * dt * vel.x / cosl, -0.5 * dt * vel.y);
    vec2 midPix = vec2((mid.x + PI) / (2.0 * PI) * size.x,
                       (0.5 * PI - mid.y) / PI * size.y);
    vec2 velMid = texture(u_vel, midPix * uvScale).rg;
    float coslMid = max(cos(mid.y), 0.017);
    vec2 dest = ll + vec2(-dt * velMid.x / coslMid, -dt * velMid.y);
    return vec2((dest.x + PI) / (2.0 * PI) * size.x,
                (0.5 * PI - dest.y) / PI * size.y);
}
void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= u_size.x || px.y >= u_size.y) return;
    vec2 pixPos = vec2(px) + 0.5;
    imageStore(out_tracer, px, vec4(sampleCR(backtrace(pixPos, u_dt)), 0.0, 0.0, 0.0));
}
"""


def isotropic_seed(h: int, w: int, seed: int, k_lo: float, k_hi: float) -> np.ndarray:
    """Provably-isotropic band-pass noise: white noise multiplied by a RADIAL
    Fourier annulus (k_lo..k_hi cycles across the shorter axis). Radial => no
    orientation bias, so its structure-tensor coherence is the isotropic control
    level by construction. Returns (h, w) float32 in ~[0,1]."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((h, w)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(white))
    cy, cx = h // 2, w // 2
    ky = (np.arange(h) - cy)[:, None].astype(np.float32) / h
    kx = (np.arange(w) - cx)[None, :].astype(np.float32) / w
    r = np.sqrt(kx * kx + ky * ky) * min(h, w)  # radial wavenumber in cycles
    annulus = ((r >= k_lo) & (r <= k_hi)).astype(np.float32)
    field = np.fft.ifft2(np.fft.ifftshift(f * annulus)).real.astype(np.float32)
    field -= field.mean()
    s = field.std()
    if s > 0:
        field /= s
    return (0.5 + 0.2 * field).astype(np.float32)  # center 0.5, moderate contrast


def belt_crop(field2d: np.ndarray, lat_half_deg: float = 30.0,
              fit_width: int = 640) -> np.ndarray:
    """Tropical/mid-latitude belt crop (|phi| < lat_half_deg), full longitude —
    the zonal-jet-dominated region where folded filaments live — resized to
    ``fit_width`` px so the structure-tensor pixel scale matches the calibrated
    metric (measure_morphology used 640-wide crops for the 0.14/0.384/0.62 bar)."""
    import cv2

    h = field2d.shape[0]
    lats = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    rows = np.where(np.abs(lats) < lat_half_deg)[0]
    crop = field2d[rows.min():rows.max() + 1, :].astype(np.float32)
    if fit_width and crop.shape[1] != fit_width:
        new_h = max(1, round(crop.shape[0] * fit_width / crop.shape[1]))
        crop = cv2.resize(crop, (fit_width, new_h), interpolation=cv2.INTER_AREA)
    return crop


def run(res: int, steps: int, tracer_mult: int, seed: int,
        k_lo: float, k_hi: float, control_translate: bool) -> dict:
    gpu = GpuContext.headless()
    gpu.make_current()
    print(f"GL renderer: {gpu.ctx.info.get('GL_RENDERER', '?')}")

    p = load_factory_preset("gas_giant_warm")
    p = p.model_copy(deep=True)
    p.sim = p.sim.model_copy(update={"resolution": res, "dev_steps": steps})
    assert p.solver.type.value == "vorticity", "spike requires vorticity mode"

    sim = Simulation(p, gpu)
    vel_tex = sim.solver.equirect.vel_tex
    dt = float(sim.solver.dt)
    print(f"sim: res={res} (equirect {vel_tex.size}) steps={steps} dt={dt:.5f} "
          f"poisson_iters={p.solver.poisson_iters}")

    tw = res * tracer_mult
    th = tw // 2
    seed_arr = isotropic_seed(th, tw, seed, k_lo, k_hi)
    print(f"tracer: {tw}x{th} (mult {tracer_mult}), isotropic band-pass k in "
          f"[{k_lo},{k_hi}] cyc")

    def r32f(data):
        t = gpu.texture2d((tw, th), 1, "f4", data=np.ascontiguousarray(data[..., None]),
                          linear=True)
        t.repeat_x = True
        return t

    cur = r32f(seed_arr)
    nxt = r32f(np.zeros((th, tw), np.float32))

    kernel = gpu.ctx.compute_shader(_ADVECT_SRC)
    kernel["u_size"].value = (tw, th)
    kernel["u_dt"].value = dt
    gx = (tw + 15) // 16
    gy = (th + 15) // 16

    # Optional zero-strain control velocity: a spatially-UNIFORM eastward flow
    # (pure translation, no strain) advected by the SAME kernel — isolates whether
    # coherence comes from the flow's STRAIN or merely from repeated interpolation.
    ctrl_vel = None
    if control_translate:
        umean = float(np.abs(gpu.read_texture(vel_tex)[..., 0]).mean())
        cv = np.zeros((vel_tex.height, vel_tex.width, 2), np.float32)
        cv[..., 0] = umean
        ctrl_vel = gpu.texture2d((vel_tex.width, vel_tex.height), 2, "f4",
                                 data=cv, linear=True)
        ctrl_vel.repeat_x = True
        ccur = r32f(seed_arr)
        cnxt = r32f(np.zeros((th, tw), np.float32))

    ctx = gpu.ctx
    t0 = time.time()
    for i in range(steps):
        sim.solver.step(1)  # advance the EVOLVING velocity field one step
        # advect high-res tracer by the solver's current velocity
        cur.use(location=0); kernel["u_src"].value = 0
        vel_tex.use(location=1); kernel["u_vel"].value = 1
        nxt.bind_to_image(0, read=False, write=True)
        kernel.run(gx, gy, 1); ctx.memory_barrier()
        cur, nxt = nxt, cur
        if ctrl_vel is not None:
            ccur.use(location=0); kernel["u_src"].value = 0
            ctrl_vel.use(location=1); kernel["u_vel"].value = 1
            cnxt.bind_to_image(0, read=False, write=True)
            kernel.run(gx, gy, 1); ctx.memory_barrier()
            ccur, cnxt = cnxt, ccur
        if (i + 1) % max(1, steps // 10) == 0:
            print(f"  step {i + 1}/{steps}  ({time.time() - t0:.1f}s)")

    advected = gpu.read_texture(cur)[..., 0]

    seed_belt = belt_crop(seed_arr).astype(np.float32)
    adv_belt = belt_crop(advected).astype(np.float32)
    rot_belt = np.rot90(adv_belt).copy()

    out = {
        "coher_seed_control": round(float(coher(seed_belt)), 4),
        "coher_advected": round(float(coher(adv_belt)), 4),
        "coher_advected_rot90": round(float(coher(rot_belt)), 4),
        "res": res, "steps": steps, "tracer": [tw, th], "dt": round(dt, 5),
        "wall_s": round(time.time() - t0, 1),
    }
    if control_translate:
        cadv = gpu.read_texture(ccur)[..., 0]
        out["coher_translate_control"] = round(float(coher(belt_crop(cadv))), 4)
        ctrl_vel.release()

    sim.release()
    cur.release(); nxt.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256, help="dynamics grid width (mult of 16)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tracer-mult", type=int, default=4, help="tracer width / dynamics width")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--k-lo", type=float, default=24.0, help="seed band low wavenumber (cyc)")
    ap.add_argument("--k-hi", type=float, default=96.0, help="seed band high wavenumber (cyc)")
    ap.add_argument("--no-translate-control", action="store_true")
    args = ap.parse_args()

    res = args.res
    result = run(
        res=res, steps=args.steps, tracer_mult=args.tracer_mult, seed=args.seed,
        k_lo=args.k_lo, k_hi=args.k_hi, control_translate=not args.no_translate_control,
    )

    print("\n==== detail-character crux result ====")
    for k, v in result.items():
        print(f"  {k:>26}: {v}")
    c_adv = result["coher_advected"]
    c_ctl = result["coher_seed_control"]
    print("\n  bar: isotropic control ~0.14 | GO 0.384 | strong 0.62")
    print(f"  advected {c_adv}  vs seed control {c_ctl}  "
          f"(separation x{c_adv / max(c_ctl, 1e-6):.2f})")
    print(f"  rot90(advected) {result['coher_advected_rot90']} "
          "(must collapse toward control if the signal is oriented HORIZONTAL structure)")
    if c_adv >= 0.62:
        verdict = "GO (strong): clears the 0.62 reference target"
    elif c_adv >= 0.384:
        verdict = "GO: clears the 0.384 bar"
    elif c_adv > 1.5 * c_ctl:
        verdict = "SEPARATES from control but below the 0.384 bar (see fidelity caveat)"
    else:
        verdict = "NO separation from the isotropic control"
    print(f"  VERDICT (this fidelity): {verdict}")


if __name__ == "__main__":
    main()

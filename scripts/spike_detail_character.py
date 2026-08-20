"""W13/W14 detail-CHARACTER crux spike (MEASUREMENT ONLY — touches no render path).

Hypothesis (docs/roadmap.md, "Research direction: detail CHARACTER = sim-advected
high-res tracer"): advecting ONE extra *high-resolution* passive tracer through the
`gas_giant_warm` vorticity solver's EVOLVING velocity field — decoupled from the
dynamics grid — folds an isotropic seed into ORIENTED (zonally-elongated filamentary)
structure, the morphology a frozen-field render trick cannot produce (F17, FALSIFIED).

MEASURE, do not grade. The roadmap pre-registered a go/no-go against 0.384, and
that framing does not survive scrutiny -- this script no longer prints a verdict
against it:
  * 0.384 is the SHIPPED jupiter_vorticity v1.6 RENDER's own score
    (docs/realism.md:558-564), so reaching it demonstrates PARITY with today, not
    improvement -- and the premise of the feature is that today reads as noise.
  * docs/realism.md:581-584 says so outright: "The blind judge panel is the gate;
    0.384 is an improvement benchmark, not a pass/fail threshold."
  * It was measured on RENDERED LUMINANCE, not a raw tracer, and on a belt crop
    3.6x finer than this one (100 deg of longitude fitted to 640px, against 360
    deg here). The same reference image scores 0.617 at that crop scale and 0.653
    at this one, so the anchors are not commensurable with this script's output.
  * The absolute drifts -18%..-29% with dynamics resolution for IDENTICAL content
    (the INTER_AREA resize in belt_crop), while the isotropic seed control stays
    flat at 0.083-0.085. So the SEPARATION RATIO is the resolution-robust
    statistic and the absolute is not.

Report the ratios, compare arms against each other, then RENDER and look.

FLOW-TIME is the axis that actually drives the absolute magnitude, and it is easy
to get wrong: dt ~ 1/resolution (sim/solver.py::compute_dt), so a FIXED step count
means LESS development at higher resolution. flow_time = dt * steps ~= 3.913 *
steps / res, which reproduces every row of the 2026-07-08 verdict table. The proxy
that scored 0.314 ran at flow-time 10.7; `--res 2048 --steps 700` is 1.34, i.e. 8x
LESS developed. Match flow-time when comparing across resolutions.

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
import json
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
        k_lo: float, k_hi: float, control_translate: bool,
        control_frozen: bool = True) -> dict:
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

    # H1 -- FROZEN-FIELD CONTROL. The distinction this whole line rests on is
    # "an EVOLVING field folds chaotically, a FROZEN one cannot" -- and that was
    # recorded as "FALSIFIED by analysis" (docs/roadmap.md:254) with no
    # measurement, against a spike whose three controls (seed / translate /
    # rot90) do not test it. Same seed, same kernel, same step count, advected
    # through the FINAL velocity held fixed: if this lands near the evolving
    # tracer, the premise is wrong and nothing downstream matters.
    fcur = fnxt = None
    if control_frozen:
        fcur = r32f(seed_arr)
        fnxt = r32f(np.zeros((th, tw), np.float32))

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

    if fcur is not None:
        # vel_tex now HOLDS the final state; simply not stepping the solver is
        # what makes this the frozen field. Same `steps`, so the two tracers see
        # the same number of interpolations -- only time-dependence differs.
        t_frozen = time.time()
        for i in range(steps):
            fcur.use(location=0); kernel["u_src"].value = 0
            vel_tex.use(location=1); kernel["u_vel"].value = 1
            fnxt.bind_to_image(0, read=False, write=True)
            kernel.run(gx, gy, 1); ctx.memory_barrier()
            fcur, fnxt = fnxt, fcur
            if (i + 1) % max(1, steps // 5) == 0:
                print(f"  frozen-control step {i + 1}/{steps} "
                      f"({time.time() - t_frozen:.1f}s)")

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
        ccur.release(); cnxt.release()
    if fcur is not None:
        frozen = gpu.read_texture(fcur)[..., 0]
        out["coher_frozen_control"] = round(float(coher(belt_crop(frozen))), 4)
        fcur.release(); fnxt.release()

    sim.release()
    cur.release(); nxt.release()
    return out




# Below this flow-time the premise test cannot discriminate: the velocity field
# has barely evolved over the run, so the frozen control IS approximately the
# evolving one. The proxy that scored 0.314 ran at flow-time 10.7.
_PREMISE_MIN_FLOW_TIME = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256, help="dynamics grid width (mult of 16)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tracer-mult", type=int, default=4, help="tracer width / dynamics width")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--k-lo", type=float, default=24.0, help="seed band low wavenumber (cyc)")
    ap.add_argument("--k-hi", type=float, default=96.0, help="seed band high wavenumber (cyc)")
    ap.add_argument("--no-translate-control", action="store_true")
    ap.add_argument("--no-frozen-control", action="store_true",
                    help="skip the frozen-field control (it costs `steps` extra "
                         "advections, though no extra solver steps)")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the result dict as JSON here, so arms can be "
                         "compared without retyping numbers")
    args = ap.parse_args()

    result = run(
        res=args.res, steps=args.steps, tracer_mult=args.tracer_mult, seed=args.seed,
        k_lo=args.k_lo, k_hi=args.k_hi,
        control_translate=not args.no_translate_control,
        control_frozen=not args.no_frozen_control,
    )
    # Flow-time is what the absolute number actually tracks: dt ~ 1/res, so a
    # FIXED step count means LESS development at higher resolution. Recording it
    # is what keeps arms comparable -- omitting it is what let the original gate
    # compare a run against a proxy carrying 8x its development.
    result["flow_time"] = round(result["dt"] * result["steps"], 3)

    print("\n==== detail-character crux result ====")
    for k, v in result.items():
        print(f"  {k:>26}: {v}")

    c_adv = result["coher_advected"]
    c_ctl = result["coher_seed_control"]
    sep = c_adv / max(c_ctl, 1e-6)

    # HEADLINE = the separation RATIO, not the absolute. The absolute drifts
    # -18%..-29% with dynamics resolution for IDENTICAL content (the belt crop's
    # INTER_AREA resize), while the seed control measures flat at 0.083-0.085
    # across res 256-2048 -- so the ratio is resolution-robust and the absolute
    # is not.
    print(f"\n  SEPARATION x{sep:.2f}   (advected {c_adv} / seed control {c_ctl})")
    print(f"  rot90(advected) {result['coher_advected_rot90']} -- must collapse "
          "toward the control if the signal is oriented HORIZONTAL structure")

    if "coher_frozen_control" in result:
        c_fro = result["coher_frozen_control"]
        fsep = c_adv / max(c_fro, 1e-6)
        print(f"\n  PREMISE TEST -- evolving {c_adv} vs frozen {c_fro}  (x{fsep:.2f})")
        if result["flow_time"] < _PREMISE_MIN_FLOW_TIME:
            # At short development the two fields ARE nearly the same field: the
            # velocity has barely evolved over the run, so frozen ~= evolving by
            # construction and a low ratio says nothing. Only a run with real
            # flow-time can discriminate.
            print(f"    (under-developed: flow-time {result['flow_time']} < "
                  f"{_PREMISE_MIN_FLOW_TIME}. The velocity barely changes over "
                  "this run, so")
            print("     frozen ~= evolving is EXPECTED and carries no verdict. "
                  "Re-run with more flow-time.)")
        elif fsep < 1.15:
            print("    ** evolving does NOT beat frozen. The premise that a "
                  "time-dependent field folds")
            print("       where a frozen one cannot is not showing up here. "
                  "STOP and re-derive before")
            print("       building anything on it.")
        else:
            print("    evolving beats frozen -- the time-dependence is doing the "
                  "work, as designed.")

    # Deliberately NO pass/fail verdict against 0.384. That number is the SHIPPED
    # jupiter_vorticity v1.6 RENDER's own score (docs/realism.md:558-564) -- i.e.
    # parity with today, not success. It was measured on rendered luminance
    # rather than a raw tracer, and on a belt crop 3.6x finer than this one
    # (100 deg of longitude fitted to 640px, vs 360 deg here; the same reference
    # image scores 0.617 there and 0.653 here). realism.md:581-584 states it
    # outright: "The blind judge panel is the gate; 0.384 is an improvement
    # benchmark, not a pass/fail threshold."
    print("\n  No pass/fail verdict: compare arms to each other, then render and look.")
    print("  (0.384 is the shipped v1.6 RENDER's score on a 3.6x finer crop -- not "
          "a bar this raw tracer can be held to.)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

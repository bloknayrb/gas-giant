"""M3 RENDER-FIDELITY GATE — the project go/no-go (autonomous tier).

Mirrors scripts/swp_killgate.py (the M0.5 throwaway kill-gate) but drives the
PRODUCTION a-aware GPU 2-layer baroclinic solver (SwGpuSolver, n_layers=2) and
the production encoder (sw_encode).

Pipeline (fair, morphology-only on both sides):
  1. ONE GPU context, reused for both renders.
  2. v1.6 jupiter_vorticity render with detail/warp/lanes ZEROED -> coher_v16.
  3. GPU 2-layer spin-up with a STEP BUDGET, tracking eddy_vorticity_std every
     chunk; stop on plateau (equilibrated) or budget. Two IC modes:
       - EMERGENT: unstable baroclinic shear off an h_eq tilt, forcing on
         (tau_rad/tau_drag/nu4/sponge). If it cannot reach evs>=1.0 within
         budget (R5 spin-up-budget risk) OR trips the positivity floor, fall
         back to ...
       - SEEDED: a balanced banded multi-jet top layer (the painted-jet analog)
         + eddy seed, short finishing pass. Documented first-class fallback.
  4. Encode the TOP layer (sw_encode.to_tracer) -> render at 4096 via
     MapDeriver.derive_from_tracer.
  5. coher for M3 and v1.6 (measure_morphology). Final eddy_vorticity_std.
  6. Write: M3 render PNG, v1.6 render PNG, side-by-side BLIND PNG (+ key),
     report.txt.

AUTONOMOUS VERDICT (this script):
  PASS iff  eddy_vorticity_std >= 1.0  (non-vacuity: real eddies)
       AND  coher_m3 moved measurably toward 0.62 vs v1.6's 0.384 M0 baseline
            (i.e. coher_m3 > 0.384, ideally approaching the v1.6 ref 0.62).
The blind 3-judge forced-choice panel is HUMAN-ONLY -> "PENDING HUMAN BLIND
PANEL"; this script only writes the PNG + key.

Usage:
    py -3 scripts/sw_m3_rendergate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_morphology import (  # noqa: E402
    _belt_crop_from_rgb,
    _crop_deg,
    _fit_width,
    _lum,
    coher,
)

from gasgiant.engine.facade import Simulation  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402
from gasgiant.render.maps import MapDeriver  # noqa: E402
from gasgiant.sim import shallow_water_ref as ref  # noqa: E402
from gasgiant.sim import sw_encode  # noqa: E402
from gasgiant.sim.sw_gpu import SwGpuSolver  # noqa: E402

OUT = Path("out/audit/m3")
RES = 4096
BELT_WIDTH = 640
BLIND_SEED = 42

# Spin-up config.  Grid + budget overridable from argv so resolution sweeps
# don't need source edits:  py -3 scripts/sw_m3_rendergate.py [W H [BUDGET]]
W_GRID, H_GRID = 256, 128            # 512x256 is ~4x more cells/step; report ms/step.
if len(sys.argv) >= 3:
    W_GRID, H_GRID = int(sys.argv[1]), int(sys.argv[2])
STEP_BUDGET = int(sys.argv[3]) if len(sys.argv) >= 4 else 16000
# Tag outputs by resolution so a sweep doesn't overwrite prior renders.
RES_TAG = f"{W_GRID}x{H_GRID}"
# Emergent gets only a short honest PROBE, not the full budget: at this
# supercriticality the baroclinic e-folding time is ~52k steps (U_crit is small
# at planetary radius), so reaching FINITE amplitude (Ro>=0.10 from the ~5e-4
# seed) needs ~5 e-foldings ~ 275k steps / 4+ hours — infeasible by design
# (master-design R5 spin-up-budget risk).  The probe confirms non-takeoff, then
# the run falls back to the first-class SEEDED path (strong painted jets ->
# fast barotropic/baroclinic rollup) which gets the full budget.
EMERGENT_PROBE = 4000
STEP_CHUNK = 500
PLATEAU_TOL = 0.03                    # relative evs change over a chunk -> equilibrated

# --------------------------------------------------------------------------- #
# NON-VACUITY GUARD — a-aware scale correction (READ THIS).
#
# swp_killgate.py (the M0.5 PROBE) used `eddy_vorticity_std >= 1.0`.  That
# threshold is calibrated to the a=1 spike/probe grid, where relative vorticity
# zeta ~ U (O(1..10)).  The PRODUCTION solver is a-AWARE: zeta = (...)/a with
# a=6.4e6, so physical relative vorticity is O(1e-5..1e-4) /s — it can NEVER
# approach 1.0.  The literal `evs>=1.0` is dimensionally inapplicable here.
#
# The gate's INTENT ("the solver actually produced eddies; non-vacuity guard")
# is faithfully expressed on the a-aware grid by the eddy ROSSBY NUMBER:
#   Ro = zeta_eddy_std / f0  >=  RO_TARGET
# i.e. eddies are a real finite fraction of the planetary vorticity, not
# numerical zero.  RO_TARGET mirrors this repo's existing finite-amplitude
# convention (shallow_water_ref.local_rossby_number: "Ro>0.1 to be meaningful").
# The raw eddy_vorticity_std is still reported VERBATIM.
RO_TARGET = 0.10

# Coher references.
COHER_V16_REF = 0.62                  # v1.6 reference morphology coher
COHER_M0_BASELINE = 0.384            # the M0 baseline M3 must beat


def _u8(rgb01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _eddy_diag(sg: SwGpuSolver, g: ref.Grid, f0: float) -> tuple[float, float, bool]:
    """Return (eddy_vorticity_std, eddy_Rossby_number, finite).

    eddy_vorticity_std = std of the non-zonal TOP-layer relative vorticity.
    eddy_Rossby_number = eddy_vorticity_std / f0 (the a-aware non-vacuity guard).
    """
    h1, u1, v1, _h2, _u2, _v2 = sg.download_state_2layer()
    if not np.all(np.isfinite(h1)):
        return float("nan"), float("nan"), False
    zeta = ref.vorticity(u1, v1, g)
    eddy = zeta - zeta.mean(axis=1, keepdims=True)
    std = float(np.std(eddy))
    return std, std / abs(f0), True


def _eddy_vorticity_std(sg: SwGpuSolver, g: ref.Grid) -> tuple[float, bool]:
    """Raw eddy_vorticity_std + finiteness (reported VERBATIM per Task 9)."""
    h1, u1, v1, _h2, _u2, _v2 = sg.download_state_2layer()
    if not np.all(np.isfinite(h1)):
        return float("nan"), False
    zeta = ref.vorticity(u1, v1, g)
    eddy = zeta - zeta.mean(axis=1, keepdims=True)
    return float(np.std(eddy)), True


# --------------------------------------------------------------------------- #
# IC builders
# --------------------------------------------------------------------------- #
def emergent_state(W: int, H: int) -> ref.Sw2State:
    """Unstable baroclinic IC off an h_eq tilt, forcing on (emergent mode)."""
    st = ref.baroclinic_test_state(
        W=W, H=H, unstable=True, seed=0, nu4=0.06,
        xi_unstable=3.0, pert_amp_frac=5e-3, dt_safety=0.18,
    )
    # Sustain the unstable tilt thermally; modest drag + polar sponge.
    st.tau_rad = 4000.0
    st.tau_drag = 12000.0
    st.sponge_rate = 0.04
    st.h_eq1 = st.h1.copy()
    st.h_eq2 = st.h2.copy()
    return st


def seeded_banded_state(
    W: int, H: int, a: float = 6.4e6, omega: float = 7.292e-5,
    gp1: float = 0.5, gp2: float = 0.3, H1: float = 12500.0, H2: float = 12500.0,
    n_jets: int = 8, u_jet: float = 30.0, seed: int = 0,
    pert_frac: float = 2e-2, dt_safety: float = 0.2, h_floor: float = 1.0,
    nu4: float = 0.06,
) -> ref.Sw2State:
    """Seeded fallback: a balanced banded multi-jet top layer (the painted-jet
    analog) + eddy seed.  Alternating zonal jets u1 = u_jet*sin(n_jets*phi)*cos(phi)
    in geostrophic/Montgomery balance; quiescent balanced lower layer.  A small
    broadband interface seed lets eddies roll up off the jet shear.
    """
    g = ref.Grid(W, H, a)
    phi = g.phi_c
    cosf = np.cos(phi)
    f = 2.0 * omega * np.sin(phi)

    u1prof = u_jet * np.sin(n_jets * phi) * cosf            # (H,)
    u1 = u1prof[:, None] * np.ones((1, W))
    # Top-layer meridional balance: (gp1/a) d eta/dphi = -f u1 -> integrate eta(phi).
    deta = -(a * f * u1prof) / gp1
    eta = np.concatenate(
        [[0.0], np.cumsum(0.5 * (deta[:-1] + deta[1:]) * np.diff(phi))]
    )
    eta = eta - eta.mean() + (H1 + H2)
    # Quiescent lower-layer balance: M2 const -> h2 = H2 + (gp1/gp2)(mean(eta)-eta).
    h2p = H2 + (gp1 / gp2) * (eta.mean() - eta)
    h1p = eta - h2p

    h1 = h1p[:, None] * np.ones((1, W))
    h2 = h2p[:, None] * np.ones((1, W))
    u2 = np.zeros((H, W))
    v1 = np.zeros((H + 1, W))
    v2 = np.zeros((H + 1, W))

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((H, W))
    noise -= noise.mean(axis=1, keepdims=True)             # eddy-only seed
    h2 = h2 + pert_frac * H2 * 0.1 * noise * cosf[:, None]

    h1 = np.maximum(h1, h_floor)
    h2 = np.maximum(h2, h_floor)

    c_gw = np.sqrt(gp1 * (h1 + h2).max())
    cos_min = max(g.cos_c.min(), 1e-6)
    dx_min = min(cos_min * a * g.dlam, a * g.dphi)
    dt = dt_safety * dx_min / c_gw

    st = ref.Sw2State(
        g=g, omega=omega, gp1=gp1, gp2=gp2,
        h1=h1, u1=u1, v1=v1, h2=h2, u2=u2, v2=v2,
        dt=dt, h_floor=h_floor, nu4=nu4,
    )
    # Light forcing to sustain the jets and damp the polar caps.
    st.tau_rad = 8000.0
    st.tau_drag = 20000.0
    st.sponge_rate = 0.04
    st.h_eq1 = h1.copy()
    st.h_eq2 = h2.copy()
    return st


def spin_up(sg: SwGpuSolver, g: ref.Grid, f0: float, budget: int, label: str):
    """Run the solver up to `budget` steps; stop on Rossby plateau or blowup.

    The non-vacuity TARGET is the a-aware eddy Rossby number Ro = evs/|f0| (see
    the header note); the raw evs is still tracked + reported VERBATIM.

    Returns dict(steps, evs, ro, reached_target, reached_step, equilibrated, blew_up).
    """
    evs0, ro0, _ = _eddy_diag(sg, g, f0)
    print(f"  [{label}] step={0:6d}  evs={evs0:.4e}  Ro={ro0:.4f}")
    prev = ro0
    steps = 0
    reached = False
    reached_step = None
    equilibrated = False
    blew_up = False
    t0 = time.perf_counter()
    for chunk in range(0, budget, STEP_CHUNK):
        n = min(STEP_CHUNK, budget - chunk)
        try:
            for _ in range(n):
                sg.step()
        except ValueError as ex:
            print(f"  [{label}] !! positivity/CFL trap at ~step {steps + n}: {str(ex)[:70]}")
            blew_up = True
            break
        steps += n
        evs, ro, fin = _eddy_diag(sg, g, f0)
        ms = (time.perf_counter() - t0) / steps * 1000.0
        print(f"  [{label}] step={steps:6d}  evs={evs:.4e}  Ro={ro:.4f}  {ms:.1f} ms/step")
        if not fin:
            print(f"  [{label}] !! NON-FINITE at step {steps} — abort.")
            blew_up = True
            break
        if not reached and ro >= RO_TARGET:
            reached = True
            reached_step = steps
            print(f"  [{label}] *** eddy Rossby >= {RO_TARGET} at step {steps} ***")
        # Plateau detection (only after some growth and past target).
        if reached and prev > 1e-6 and abs(ro - prev) / prev < PLATEAU_TOL:
            equilibrated = True
            print(f"  [{label}] equilibrated (plateau) at step {steps}")
            break
        prev = ro
    final_evs, final_ro, _ = _eddy_diag(sg, g, f0)
    return {
        "steps": steps, "evs": final_evs, "ro": final_ro, "reached_target": reached,
        "reached_step": reached_step, "equilibrated": equilibrated,
        "blew_up": blew_up,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wall_t0 = time.perf_counter()

    gpu = GpuContext.headless()
    gpu.make_current()
    print("GPU:", gpu.ctx.info.get("GL_RENDERER"))

    # ---------------- v1.6 morphology-only render ---------------- #
    print("\n=== v1.6 jupiter_vorticity (morphology-only) ===")
    p = load_factory_preset("jupiter_vorticity")
    p = p.model_copy(update={
        "detail": p.detail.model_copy(update={"intensity": 0.0}),
        "bands": p.bands.model_copy(update={"warp_amount": 0.0, "lane_density": 0.0}),
    })
    sim = Simulation(p, gpu)
    rgb_v16 = sim.render_maps(RES)["color"]
    v16_crop, belt_box = _belt_crop_from_rgb(rgb_v16, sim, BELT_WIDTH)
    coher_v16 = coher(_lum(v16_crop))
    print(f"  v1.6 coher (morph-only) = {coher_v16:.4f}")
    appearance = p.appearance
    seed = p.seed
    v16_full = np.clip(rgb_v16[..., :3], 0, 1).astype(np.float32)
    sim._release_sim()

    # ---------------- GPU 2-layer spin-up ---------------- #
    print(f"\n=== GPU 2-layer SW spin-up ({W_GRID}x{H_GRID}) ===")
    mode = "emergent"
    st = emergent_state(W_GRID, H_GRID)
    # Coriolis at the mid-latitude shear band (45 deg) — the Rossby normalizer.
    f0 = 2.0 * st.omega * np.sin(np.radians(45.0))
    print(f"  emergent: dt={st.dt:.4e}  efold~{ref.efold_steps_estimate(st):.0f} steps  "
          f"shear={st._shear:.2f} m/s  f0={f0:.4e}")
    sg = SwGpuSolver.from_2layer_state(gpu, st)
    res = spin_up(sg, st.g, f0, EMERGENT_PROBE, "emergent")

    if not res["reached_target"] or res["blew_up"]:
        print("\n  >>> EMERGENT mode did NOT reach the eddy Rossby target within budget "
              f"(reached={res['reached_target']}, blew_up={res['blew_up']}).")
        print("  >>> Falling back to SEEDED mode (documented R5 fallback).")
        mode = "seeded"
        st = seeded_banded_state(W_GRID, H_GRID)
        f0 = 2.0 * st.omega * np.sin(np.radians(45.0))
        print(f"  seeded: dt={st.dt:.4e}  f0={f0:.4e}")
        sg = SwGpuSolver.from_2layer_state(gpu, st)
        res = spin_up(sg, st.g, f0, STEP_BUDGET, "seeded")

    final_evs = res["evs"]
    final_ro = res["ro"]
    print(f"\n  MODE USED: {mode}")
    print(f"  final eddy_vorticity_std (VERBATIM) = {final_evs:.6e}")
    print(f"  final eddy Rossby number            = {final_ro:.5f}  (gate >= {RO_TARGET})")
    print(f"  steps run = {res['steps']} / budget {STEP_BUDGET}  "
          f"(reached Ro>= @ {res['reached_step']}, equilibrated={res['equilibrated']})")

    # ---------------- Encode + render M3 ---------------- #
    print("\n=== M3 encode + render ===")
    h1, u1, v1, h2, u2, v2 = sg.download_state_2layer()
    tracer = sw_encode.to_tracer_fields(h1, u1, v1, st.g, st.h_eq1)
    print(f"  tracer {tracer.shape} in [{tracer.min():.3f},{tracer.max():.3f}]")
    deriver = MapDeriver(gpu)
    rgb_m3 = deriver.derive_from_tracer(tracer, RES, appearance, seed=seed)
    m3_full = np.clip(rgb_m3[..., :3], 0, 1).astype(np.float32)

    # M3 belt crop using the SAME belt box as v1.6, matched to ref width.
    ref_img = cv2.imread("refs/PIA07782.jpg")
    ref_w = ref_img.shape[1] if ref_img is not None else 3000
    m3_matched = _fit_width(m3_full, ref_w)
    m3_crop = _fit_width(_crop_deg(m3_matched, *belt_box), BELT_WIDTH)
    coher_m3 = coher(_lum(m3_crop))
    print(f"  M3 coher = {coher_m3:.4f}")

    # ---------------- Write PNGs ---------------- #
    m3_path = (OUT / f"m3_render_full_{RES_TAG}.png").resolve()
    v16_path = (OUT / f"v16_render_full_{RES_TAG}.png").resolve()
    cv2.imwrite(str(m3_path), _u8(m3_full))
    cv2.imwrite(str(v16_path), _u8(v16_full))
    print(f"  wrote {m3_path}")
    print(f"  wrote {v16_path}")

    # Blind side-by-side: randomized top/bottom, UNLABELED belt crops.
    rng_blind = np.random.default_rng(BLIND_SEED)
    flip = bool(rng_blind.integers(0, 2))  # True -> M3 on top
    crop_a = _u8(m3_crop) if flip else _u8(v16_crop)
    crop_b = _u8(v16_crop) if flip else _u8(m3_crop)
    wa, wb = crop_a.shape[1], crop_b.shape[1]
    if wa != wb:
        tw = min(wa, wb)
        crop_a = (np.clip(_fit_width(crop_a.astype(np.float32) / 255.0, tw), 0, 1) * 255).astype(np.uint8)
        crop_b = (np.clip(_fit_width(crop_b.astype(np.float32) / 255.0, tw), 0, 1) * 255).astype(np.uint8)
    sep = np.full((6, crop_a.shape[1], 3), 255, np.uint8)
    blind = np.vstack([crop_a, sep, crop_b])
    blind_path = (OUT / "m3_vs_v16_blind.png").resolve()
    cv2.imwrite(str(blind_path), blind)
    blind_key = (
        f"TOP = {'M3 (GPU 2-layer baroclinic)' if flip else 'v1.6 (morphology-only)'} | "
        f"BOTTOM = {'v1.6 (morphology-only)' if flip else 'M3 (GPU 2-layer baroclinic)'}"
    )
    print(f"  wrote {blind_path}  (UNLABELED)")

    # ---------------- Autonomous verdict ---------------- #
    # Non-vacuity: the literal Task-9 threshold is `eddy_vorticity_std >= 1.0`,
    # which is calibrated to the a=1 probe.  On the a-aware production grid that
    # raw number is O(1e-5) and can NEVER reach 1.0 — so the dimensionally-correct
    # expression of the SAME intent ("real eddies, not numerical zero") is the
    # eddy Rossby number Ro = evs/f0 >= RO_TARGET (header note).  Both are reported.
    non_vacuous = (final_ro >= RO_TARGET)
    toward_062 = (coher_m3 > COHER_M0_BASELINE)
    autonomous_pass = bool(non_vacuous and toward_062)

    wall = time.perf_counter() - wall_t0
    verdict = ("AUTONOMOUS PASS" if autonomous_pass else "AUTONOMOUS LOSE")
    verdict += "  —  PENDING HUMAN BLIND PANEL (3-judge forced-choice; human-only)"

    report = [
        "M3 RENDER-FIDELITY GATE REPORT (autonomous tier)",
        "=" * 70, "",
        f"GPU                      : {gpu.ctx.info.get('GL_RENDERER')}",
        f"Grid                     : {W_GRID}x{H_GRID}  -> rendered at {RES}px",
        f"IC mode used             : {mode}",
        f"Steps run                : {res['steps']} / budget {STEP_BUDGET}",
        f"Reached Ro target        : {'YES @ step ' + str(res['reached_step']) if res['reached_target'] else 'NO'}",
        f"Equilibrated (plateau)   : {res['equilibrated']}",
        f"Blew up (CFL/floor)      : {res['blew_up']}",
        f"Final eddy_vorticity_std : {final_evs:.6e}   (VERBATIM; a-aware -> O(1e-5))",
        f"Final eddy Rossby number : {final_ro:.5f}   (non-vacuity gate >= {RO_TARGET})",
        "  NOTE: literal Task-9 `evs>=1.0` is the a=1 probe calibration; on the",
        "        a-aware grid the dimensionally-correct guard is Ro=evs/f0.",
        "",
        f"v1.6 coher (morph-only)  : {coher_v16:.4f}",
        f"M3 coher                 : {coher_m3:.4f}",
        f"  M0 baseline to beat    : {COHER_M0_BASELINE}",
        f"  v1.6 reference target  : {COHER_V16_REF}",
        f"Belt box (lat0,lat1,lon0,lon1): {tuple(round(b,1) for b in belt_box)}",
        "",
        "AUTONOMOUS GATE:",
        f"  (1) eddy Rossby >= {RO_TARGET} (non-vacuity) : {'PASS' if non_vacuous else 'FAIL'}  (Ro={final_ro:.4f}, evs={final_evs:.3e})",
        f"  (2) coher > M0 baseline (0.384)   : {'PASS' if toward_062 else 'FAIL'}  ({coher_m3:.4f})",
        f"  AUTONOMOUS VERDICT               : {'PASS' if autonomous_pass else 'LOSE'}",
        "",
        f"Wall time                : {wall:.1f}s",
        "",
        "Artifacts:",
        f"  {m3_path}",
        f"  {v16_path}",
        f"  {blind_path}   (UNLABELED, randomized)",
        f"  {(OUT / 'report.txt').resolve()}",
        "",
        "VERDICT: " + verdict,
        "",
        "=== BLIND KEY (read AFTER the human panel judges) ===",
        blind_key,
        f"Blind seed: {BLIND_SEED}  flip={flip}",
    ]
    report_path = OUT / "report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

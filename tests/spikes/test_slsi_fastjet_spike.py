"""M2-adv crux gate.

OUTCOME: NO-GO. The SLSI approach is FALSIFIED on the lat-lon grid (Earth-like
params). The accuracy gate below is marked xfail: step_slsi is unconditionally
unstable at the large dt the gate requires (the binding constraint after M2-core is
the polar gravity-wave/theta imbalance, NOT advection; the SL operators regress
stability vs the Eulerian scheme). See docs/superpowers/specs/m2-adv-verdict.md for
the full evidence. The CPU SL operators (Tasks 1-4) are correct/conserving and kept
for reuse; no GPU kernels were built.
"""
import numpy as np
import pytest
from gasgiant.sim.shallow_water_ref import (
    fast_jet_state, step_slsi, step_semi_implicit, total_mass)

def _run(step_fn, dt_mult, n_big):
    st = fast_jet_state(dt_mult=dt_mult)
    dh = None
    for _ in range(n_big):
        st = step_fn(st, theta=0.5, picard_iters=3, poisson_iters=300, dh_warm=dh)
    return st

def test_crux_setup_is_nonvacuous():
    st = fast_jet_state(dt_mult=8)
    u_c = 0.5 * (st.u + np.roll(st.u, 1, axis=1))
    C = np.max(np.abs(u_c) * st.dt / (st.g.a * st.g.cos_c[:, None] * st.g.dlam))
    assert C > 3.0, f"crux gate vacuous: max advective C={C:.2f} (need >>1)"
    assert np.max(np.abs(st.h - st.h.mean(axis=1, keepdims=True))) > 1.0

@pytest.mark.xfail(reason="M2-adv falsified: step_slsi unstable at large dt; "
                          "balanced reference does not survive. See m2-adv-verdict.md.",
                   strict=False)
def test_reference_is_self_converged():
    ref1 = _run(step_slsi, dt_mult=1, n_big=160)
    ref2 = _run(step_slsi, dt_mult=2, n_big=80)
    drift = float(np.sqrt(np.mean((ref2.h - ref1.h) ** 2)) / np.sqrt(np.mean(ref1.h ** 2)))
    assert drift < 0.02, f"reference not converged (self-L2={drift:.4f}); gate tol unsafe"

@pytest.mark.xfail(reason="M2-adv falsified: SLSI is unconditionally unstable at the "
                          "large dt this gate requires (premise does not hold on the "
                          "lat-lon grid). See m2-adv-verdict.md.",
                   strict=False)
def test_slsi_fastjet_accuracy_at_large_courant():
    ref = _run(step_slsi, dt_mult=1, n_big=160)
    big = _run(step_slsi, dt_mult=8, n_big=20)
    assert np.isfinite(big.h).all() and big.h.min() > 0.0
    def l2(a, b): return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
    err = l2(big.h, ref.h)
    print(f"\n[slsi-spike] 8x-dt vs fine-dt relative L2(h) = {err:.4f}")
    assert err < 0.05, f"SLSI accuracy gate FAILED at 8x dt: L2={err:.4f} (approach falsified)"
    assert abs(total_mass(big) - total_mass(ref)) / abs(total_mass(ref)) < 1e-6

def test_eulerian_path_fails_the_same_gate():
    try:
        big_eul = _run(step_semi_implicit, dt_mult=8, n_big=20)
        ref = _run(step_slsi, dt_mult=1, n_big=160)
        err = float(np.sqrt(np.mean((big_eul.h - ref.h) ** 2)) / np.sqrt(np.mean(ref.h ** 2)))
        assert err > 0.05 or not np.isfinite(big_eul.h).all(), (
            f"gate not discriminating: Eulerian path also passes (L2={err:.4f})")
    except (ValueError, FloatingPointError):
        pass

"""M2-T6: dual-path byte-identity gate.

Verifies that:
1. Adding the semi_implicit flag (defaulted False) does NOT perturb the explicit path.
2. Constructing with semi_implicit=True leaves explicit field textures and step()
   output byte-identical to semi_implicit=False (the SI kernels are not wired yet;
   step() must be the same explicit step regardless).
3. H_ref has shape (H,) when semi_implicit=True, and is None (absent from state)
   when semi_implicit=False (default).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest


_W, _H = 64, 32
_IC = dict(W=_W, H=_H, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
_N_STEPS = 30


def _sha1_state(h, u, v) -> str:
    m = hashlib.sha1()
    for f in (h, u, v):
        m.update(np.ascontiguousarray(f, dtype=np.float32).tobytes())
    return m.hexdigest()


# ---------------------------------------------------------------------------
# T6-A: explicit path byte-identical (flag present but off)
# ---------------------------------------------------------------------------

def test_explicit_path_byte_identical(gpu):
    """Two solvers with semi_implicit=False (default) must produce byte-identical
    results after N steps — mirrors the determinism discipline in test_gpu_deterministic.
    """
    from gasgiant.sim import sw_gpu

    def run():
        sg = sw_gpu.SwGpuSolver.from_williamson2(gpu, **_IC)
        for _ in range(_N_STEPS):
            sg.step()
        return sg.download_state()

    h_a, u_a, v_a = run()
    h_b, u_b, v_b = run()

    assert np.array_equal(h_a, h_b), "h not byte-identical between two explicit-path runs"
    assert np.array_equal(u_a, u_b), "u not byte-identical between two explicit-path runs"
    assert np.array_equal(v_a, v_b), "v not byte-identical between two explicit-path runs"
    assert _sha1_state(h_a, u_a, v_a) == _sha1_state(h_b, u_b, v_b), "SHA1 mismatch"


# ---------------------------------------------------------------------------
# T6-B: SI flag on does NOT change explicit fields or step() output
# ---------------------------------------------------------------------------

def test_si_construction_isolated(gpu):
    """Constructing with semi_implicit=True must not alter the explicit field
    textures or step() output relative to semi_implicit=False (the SI path is
    not wired yet — step() is still the explicit step in both cases).
    Also asserts H_ref shape and that it is None on the explicit solver.
    """
    from gasgiant.sim import sw_gpu

    # Build explicit-only solver.
    sg_exp = sw_gpu.SwGpuSolver.from_williamson2(gpu, **_IC, semi_implicit=False)

    # Build SI-flagged solver with non-default SI params (all inert for now).
    sg_si = sw_gpu.SwGpuSolver.from_williamson2(
        gpu, **_IC,
        semi_implicit=True,
        theta=0.6,
        sor_omega=1.8,
        helmholtz_iters=100,
        picard_iters=2,
        dt_multiplier=2.0,
    )

    # --- initial fields must be byte-identical ---
    h_exp0, u_exp0, v_exp0 = sg_exp.download_state()
    h_si0,  u_si0,  v_si0  = sg_si.download_state()
    assert np.array_equal(h_exp0, h_si0),  "h initial state differs when semi_implicit=True"
    assert np.array_equal(u_exp0, u_si0),  "u initial state differs when semi_implicit=True"
    assert np.array_equal(v_exp0, v_si0),  "v initial state differs when semi_implicit=True"

    # --- advance both solvers and compare ---
    for _ in range(_N_STEPS):
        sg_exp.step()
        sg_si.step()

    h_exp, u_exp, v_exp = sg_exp.download_state()
    h_si,  u_si,  v_si  = sg_si.download_state()

    assert np.array_equal(h_exp, h_si), "h differs after steps when semi_implicit=True"
    assert np.array_equal(u_exp, u_si), "u differs after steps when semi_implicit=True"
    assert np.array_equal(v_exp, v_si), "v differs after steps when semi_implicit=True"

    # --- H_ref shape gate ---
    assert sg_si.H_ref is not None, "H_ref should be set when semi_implicit=True"
    assert sg_si.H_ref.shape == (_H,), (
        f"H_ref shape {sg_si.H_ref.shape!r} != ({_H},)"
    )

    # --- explicit solver must NOT carry H_ref state ---
    assert sg_exp.H_ref is None, "H_ref should be None when semi_implicit=False"

    # --- verify SI param slots are stored correctly ---
    assert sg_si.semi_implicit is True
    assert sg_si.theta == pytest.approx(0.6)
    assert sg_si.sor_omega == pytest.approx(1.8)
    assert sg_si.helmholtz_iters == 100
    assert sg_si.picard_iters == 2
    assert sg_si.dt_multiplier == pytest.approx(2.0)

    # --- verify explicit solver defaults ---
    assert sg_exp.semi_implicit is False
    assert sg_exp.theta == pytest.approx(0.5)
    assert sg_exp.sor_omega == pytest.approx(1.7)
    assert sg_exp.helmholtz_iters == 200
    assert sg_exp.picard_iters == 3
    assert sg_exp.dt_multiplier == pytest.approx(1.0)

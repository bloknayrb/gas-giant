"""
M2-T0: Polar advective-CFL go/no-go spike
==========================================
Tests whether a Boyd/Vandeven-style zonal low-pass filter removes the advective
CFL constraint in a near-polar latitude row, allowing stable advection at
cell-Courant number C=1.5 (well above the upwind CFL limit of 1).

EXPERIMENT SETUP
----------------
- 1-D periodic zonal advection of a passive tracer q(lambda) on W=128 points.
- Smooth initial field: q0 = 1 + 0.5*sin(2*pi*i/W) + 0.3*sin(8*pi*i/W)
  (mix of wavenumber-1 and wavenumber-4, i.e. low + moderate frequency content).
- Constant rightward advecting velocity (u>0), donor-cell (first-order upwind) flux.
- Reference run: C_ref=0.3 (below CFL limit), no filter, 300 steps.
- Test run:      C_big=1.5 (above CFL limit), Boyd filter every step, 60 steps.
  Both cover identical physical advection time (N*C = 90 cell-widths).
- Boyd filter: qhat *= exp(-alpha*(k/k_max)^(2p)), p=4, alpha=36 (Nyquist ~ e^-36).

VERDICT
-------
FILTER ROUTE FALSIFIED — advective CFL is not removed by zonal filtering.

The filtered C=1.5 run remains bounded (no NaN/Inf; max|q| ~2.2 vs ~1.5 for
reference), BUT the retained low-wavenumber band (k < W//8 = 16) shows a
relative L2 error of ~0.41 against the C=0.3 reference.  The RETAINED mean
flow — i.e. the low-k content that survives the filter — is still advected at
the same over-large Courant number through the nonlinear upwind branch, and
that branch is both unstable (oscillations grow above q0 range) and highly
diffusive/inaccurate at C>1.  Filtering kills the high-wavenumber noise but
cannot rescue the large-Courant error in the modes it keeps.

ROUTING DECISION for M2-adv
----------------------------
The polar advective-CFL bottleneck CANNOT be fixed by cheap spectral filtering
alone.  M2-adv must adopt one of:
  (a) Semi-Lagrangian advection (unconditionally stable in advective CFL),
  (b) Implicit advection scheme,
  (c) Reduced-grid / lat-lon coarsening near the poles, or
  (d) A hybrid that genuinely removes or reduces the polar Courant number.
"""

import numpy as np
from numpy.fft import rfft, irfft

# ---------------------------------------------------------------------------
# Parameters (fixed, deterministic — no RNG)
# ---------------------------------------------------------------------------
W = 128
C_REF = 0.3
C_BIG = 1.5
N_REF = 300
N_TEST = round(N_REF * C_REF / C_BIG)   # = 60; same physical time
M_CUT = W // 8                           # retained wavenumber band: k < 16
BOYD_ALPHA = 36.0
BOYD_P = 4


# ---------------------------------------------------------------------------
# Core numerics
# ---------------------------------------------------------------------------

def make_initial_field():
    """Smooth tracer with wavenumber-1 and wavenumber-4 content."""
    i_arr = np.arange(W, dtype=float)
    return 1.0 + 0.5 * np.sin(2 * np.pi * i_arr / W) + 0.3 * np.sin(8 * np.pi * i_arr / W)


def upwind_step(q, C):
    """First-order upwind donor-cell update for u>0, periodic grid."""
    q_left = np.roll(q, 1)   # q[i-1] with periodic wrap
    return q - C * (q - q_left)


def boyd_filter(q, alpha=BOYD_ALPHA, p=BOYD_P):
    """
    Boyd/Vandeven-style spectral low-pass filter.
    qhat *= exp(-alpha * (k/k_max)^(2p))
    The Nyquist mode is damped by exp(-alpha) ~ e^-36 ~ 0.
    """
    qhat = rfft(q)
    k_max = float(len(qhat) - 1)          # index of Nyquist component
    k = np.arange(len(qhat), dtype=float)
    filt = np.exp(-alpha * (k / k_max) ** (2 * p))
    return irfft(qhat * filt, n=W)


def run_reference(q0):
    """C=0.3 reference: 300 upwind steps, no filter."""
    q = q0.copy()
    for _ in range(N_REF):
        q = upwind_step(q, C_REF)
    return q


def run_filtered_test(q0):
    """C=1.5 test: 60 upwind steps + Boyd filter every step."""
    q = q0.copy()
    for _ in range(N_TEST):
        q = upwind_step(q, C_BIG)
        q = boyd_filter(q)
    return q


def retained_band_relative_l2(q_test, q_ref, m_cut=M_CUT):
    """
    Relative L2 error in the retained low-wavenumber band (k < m_cut).
    Both fields are projected to wavenumbers [0, m_cut) before comparison.
    """
    def project(q):
        qhat = rfft(q)
        qhat_band = np.zeros_like(qhat)
        qhat_band[:m_cut] = qhat[:m_cut]
        return irfft(qhat_band, n=W)

    q_ref_band  = project(q_ref)
    q_test_band = project(q_test)
    norm_ref = np.sqrt(np.mean(q_ref_band ** 2))
    return np.sqrt(np.mean((q_test_band - q_ref_band) ** 2)) / (norm_ref + 1e-12)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_physical_time_matched():
    """Sanity: both runs cover the same physical advection distance."""
    assert abs(N_REF * C_REF - N_TEST * C_BIG) < 1e-10, (
        f"Physical time mismatch: ref={N_REF*C_REF}, test={N_TEST*C_BIG}"
    )


def test_reference_run_bounded():
    """Reference C=0.3 run must stay within a reasonable envelope."""
    q0 = make_initial_field()
    q_ref = run_reference(q0)
    assert not np.any(np.isnan(q_ref)), "Reference run produced NaN"
    assert not np.any(np.isinf(q_ref)), "Reference run produced Inf"
    max_q = np.max(np.abs(q_ref))
    # First-order upwind at C=0.3 is monotone; field stays in [min(q0), max(q0)]
    # with only diffusive shrinkage.  Allow a tiny floating-point margin.
    assert max_q <= np.max(np.abs(q0)) + 1e-10, (
        f"Reference run not bounded: max|q|={max_q:.6f}"
    )


def test_filtered_large_courant_stays_finite():
    """
    Filtered C=1.5 run must not produce NaN or Inf.
    The filter damps the instability enough to prevent blow-up — but does NOT
    produce an accurate solution (see retained-band error test below).

    Observed: max|q| ~ 2.20, bounded but outside q0 range.
    """
    q0 = make_initial_field()
    q_test = run_filtered_test(q0)
    assert not np.any(np.isnan(q_test)), "Filtered large-C run produced NaN"
    assert not np.any(np.isinf(q_test)), "Filtered large-C run produced Inf"
    max_q = np.max(np.abs(q_test))
    print(f"\n[spike] Filtered C={C_BIG} max|q| = {max_q:.6f}")
    # Observed ~2.20; allow up to 3.0 to give a stable characterization window.
    assert max_q < 3.0, f"Filtered run exceeded expected bound: max|q|={max_q:.6f}"


def test_filter_does_not_fix_retained_band_accuracy():
    """
    CORE VERDICT TEST.

    The retained low-wavenumber band (k < W//8 = 16) of the filtered C=1.5
    run has a LARGE relative L2 error vs the C=0.3 reference.  This means the
    filter does NOT rescue the advective-CFL problem: the mean flow itself is
    badly corrupted because the upwind branch still operates at C>1.

    Observed retained-band relative L2 error: ~0.41 (>> 0.10 threshold).

    VERDICT: FILTER ROUTE FALSIFIED — advective CFL is not removed by zonal
    filtering; M2-adv must use semi-Lagrangian/implicit/reduced-grid advection.
    """
    q0 = make_initial_field()
    q_ref  = run_reference(q0)
    q_test = run_filtered_test(q0)

    l2_band = retained_band_relative_l2(q_test, q_ref)
    l2_full = float(np.sqrt(np.mean((q_test - q_ref) ** 2)))

    print(f"\n[spike] Retained-band (k<{M_CUT}) relative L2 error: {l2_band:.6f}")
    print(f"[spike] Full-field L2 error:                         {l2_full:.6f}")
    print(f"[spike] VERDICT: FILTER ROUTE FALSIFIED — retained-band L2={l2_band:.3f} >> 0.10")

    # Lock in the observed large error.  If it ever shrinks below 0.10 something
    # changed in the numerics and the routing decision must be revisited.
    assert l2_band > 0.10, (
        f"Retained-band L2 error unexpectedly small ({l2_band:.4f}); "
        "filter may be rescuing advective CFL — routing verdict needs review."
    )

    # Also confirm it matches the observed ~0.41 within a generous tolerance
    # (the characterization window is [0.10, 0.80]).
    assert l2_band < 0.80, (
        f"Retained-band L2 error unexpectedly large ({l2_band:.4f}); "
        "the run may be more divergent than observed — check numerics."
    )

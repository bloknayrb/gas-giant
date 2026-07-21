"""Throwaway CPU pseudo-spectral 2D barotropic (equivalent-barotropic / screened QG)
vorticity solver for the billow-rollup spike.

State variable `w` is the ADVECTED quantity. With `screened=True` the inversion
psi_hat = -w_hat/(k2 + INV_LD2) makes `w` the screened potential vorticity
w = laplacian(psi) - psi/L_d^2, whose 2D advection J(psi, w) conserves BOTH the
energy  E = 0.5<|grad psi|^2> + 0.5*INV_LD2*<psi^2>  and the enstrophy 0.5<w^2>.
That pair is the conservation gate the whole spike stands on (Task 1 Step 2).

Doubly-periodic pseudo-spectral core on an (NY, NX) grid (y = rows, x = cols).
All constants come from config.py (the single source). Nothing here touches
src/gasgiant/** or git.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config  # noqa: E402  (spike-local single-source constants)

# ----------------------------------------------------------------------------
# Nyquist-targeted hyperviscous truncation filter strength.
#
# The state filter applied every step is  exp(-NU8 * (k2/k2max)^4) * DEALIAS.
# Its ONLY job is to keep the 2/3-truncated spectrum clean at the retained-band
# edge -- it is NOT physical dissipation. With the retained edge at (2/3) of the
# Nyquist per axis, (k2/k2max) at the edge is ~4/9 => (4/9)^4 ~ 0.039, so:
#   * retained-band edge (|k| ~ 0.44 k_nyq) decays exp(-NU8*0.039) ~ 0.89  (~11%/step, keeps the edge clean)
#   * the true Nyquist (already zeroed by DEALIAS) would decay exp(-NU8) ~ O(1)
#   * energy-containing modes (|k| < 20, ratio ~5e-5) decay exp(-~1e-16) == 1.0 (untouched)
# NU8 = 3.0 sits in the brief's suggested 2-4 band.
NU8 = 3.0


class Box:
    """Doubly-periodic pseudo-spectral barotropic vorticity box.

    Parameters
    ----------
    beta : float
        Planetary vorticity gradient (per-step tendency term -beta*v). Instance
        attribute so Gate 0a can build Box(beta=0.0) without monkey-patching.
    screened : bool
        True  -> equivalent-barotropic inversion psi_hat = -w_hat/(k2 + INV_LD2).
        False -> pure barotropic psi_hat = -w_hat/k2 with the k=0 mode zeroed.
    """

    def __init__(self, beta=config.BETA, screened=True):
        self.beta = float(beta)
        self.screened = bool(screened)

        self.nx = config.NX
        self.ny = config.NY
        self.dx = config.DX
        self.dt = config.DT
        self.ild2 = config.INV_LD2  # screening 1/L_d^2 (also the energy psi^2 weight)
        self.shape = (self.ny, self.nx)

        # ---- Spectral machinery (precomputed once) -------------------------
        # Angular wavenumbers. x is the rfft (halved) axis, y the full-fft axis.
        kx = np.fft.rfftfreq(self.nx, d=self.dx) * 2.0 * np.pi   # (NX//2+1,)  >= 0
        ky = np.fft.fftfreq(self.ny, d=self.dx) * 2.0 * np.pi    # (NY,)  signed
        KX = kx[np.newaxis, :]          # (1, NXh)
        KY = ky[:, np.newaxis]          # (NY, 1)
        self.KX, self.KY = KX, KY
        self.ikx = 1j * KX              # spectral d/dx operator
        self.iky = 1j * KY              # spectral d/dy operator

        k2 = KX * KX + KY * KY          # (NY, NXh)
        self.k2 = k2
        self.kmag = np.sqrt(k2)

        # Inversion denominator: psi_hat = -w_hat * inv_denom.
        if self.screened:
            self.inv_denom = 1.0 / (k2 + self.ild2)
        else:
            denom = k2.copy()
            denom[0, 0] = 1.0           # avoid divide-by-zero at the mean mode
            self.inv_denom = 1.0 / denom
            self.inv_denom[0, 0] = 0.0  # zero the k=0 (mean streamfunction) mode

        # State filter = 2/3 dealias mask * Nyquist-targeted nu_8 decay.
        kx_cut = (2.0 / 3.0) * np.abs(kx).max()
        ky_cut = (2.0 / 3.0) * np.abs(ky).max()
        dealias = (np.abs(KX) < kx_cut) & (np.abs(KY) < ky_cut)
        k2max = k2.max()
        nu8_fac = np.exp(-NU8 * (k2 / k2max) ** 4)
        self.dealias = dealias
        self.state_filter = nu8_fac * dealias   # applied to BOTH w_hat and tr_hat every step

        # ---- State -----------------------------------------------------------
        self.w = np.zeros(self.shape, dtype=np.float64)   # vorticity / screened PV
        self.tr = np.zeros(self.shape, dtype=np.float64)  # passive tracer

        # Task-8 hook: the bound advance method (ladder rungs swap this in).
        self.advector = self.step_spectral
        self.substep_count = 0

    # ------------------------------------------------------------------ helpers
    def _rfft(self, f):
        return np.fft.rfft2(f)

    def _irfft(self, fh):
        return np.fft.irfft2(fh, s=self.shape)

    def _psi_hat(self, w_hat):
        return -w_hat * self.inv_denom

    def invert(self, w=None):
        """Velocity (u, v) from vorticity. u = -dpsi/dy, v = +dpsi/dx."""
        if w is None:
            w = self.w
        psi_hat = self._psi_hat(self._rfft(w))
        u = self._irfft(-self.iky * psi_hat)
        v = self._irfft(self.ikx * psi_hat)
        return u, v

    def _psi(self, w=None):
        if w is None:
            w = self.w
        return self._irfft(self._psi_hat(self._rfft(w)))

    def _w_tend(self, w):
        """Return (u, v, tend) for a vorticity field. Shares one rfft2(w) between
        the inversion and the vorticity-gradient derivatives (5 transforms)."""
        w_hat = self._rfft(w)
        psi_hat = self._psi_hat(w_hat)
        u = self._irfft(-self.iky * psi_hat)
        v = self._irfft(self.ikx * psi_hat)
        wx = self._irfft(self.ikx * w_hat)
        wy = self._irfft(self.iky * w_hat)
        tend = -(u * wx + v * wy) - self.beta * v
        return u, v, tend

    def _tr_tend(self, tr, u, v):
        """Passive-tracer tendency -(u*tr_x + v*tr_y) through a FROZEN (u, v)."""
        tr_hat = self._rfft(tr)
        tx = self._irfft(self.ikx * tr_hat)
        ty = self._irfft(self.iky * tr_hat)
        return -(u * tx + v * ty)

    # ------------------------------------------------------------------ stepping
    def cfl(self):
        """Advective CFL number max(|u|,|v|)*DT/DX for the current state."""
        u, v = self.invert()
        return max(np.abs(u).max(), np.abs(v).max()) * self.dt / self.dx

    def step_spectral(self, forcing_fn=None):
        """Advance one production step (DT). RK4 for w; RK4 for tr through the
        frozen end-of-step velocity; then the 2/3+nu8 state filter; then forcing.

        CFL guard: RK4's imaginary-axis stability limit is 2*sqrt(2)=2.83. The
        pseudo-spectral advection eigenvalue magnitude is ~(2/3)*pi*CFL (2/3 from
        the dealias cutoff). If that exceeds 2.8 the step is split into two DT/2
        sub-steps and `substep_count` is incremented -- a high-amplitude blowup
        must not masquerade as physics.
        """
        # k1 stage doubles as the CFL probe (reuse its velocity; no extra transforms).
        u, v, k1 = self._w_tend(self.w)
        cfl = max(np.abs(u).max(), np.abs(v).max()) * self.dt / self.dx
        if (2.0 / 3.0) * np.pi * cfl > 2.8:
            self.substep_count += 1
            self._advance(0.5 * self.dt, k1=k1)   # k1 tendency is dt-independent
            self._advance(0.5 * self.dt)
        else:
            self._advance(self.dt, k1=k1)
        # Per-step forcing closure: applied ONCE per PRODUCTION step (rates are
        # per-step fractions -- applying per half-step would double them).
        if forcing_fn is not None:
            forcing_fn(self)

    def _advance(self, dt, k1=None):
        # ---- RK4 for vorticity ----------------------------------------------
        w = self.w
        if k1 is None:
            _, _, k1 = self._w_tend(w)
        _, _, k2 = self._w_tend(w + 0.5 * dt * k1)
        _, _, k3 = self._w_tend(w + 0.5 * dt * k2)
        _, _, k4 = self._w_tend(w + dt * k3)
        w_adv = w + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # ---- Truncation filter + frozen end-of-step velocity ----------------
        w_hat = self._rfft(w_adv) * self.state_filter
        self.w = self._irfft(w_hat)
        psi_hat = self._psi_hat(w_hat)
        u_f = self._irfft(-self.iky * psi_hat)   # frozen velocity for the tracer RK4
        v_f = self._irfft(self.ikx * psi_hat)

        # ---- RK4 for the passive tracer through the frozen velocity ---------
        tr = self.tr
        t1 = self._tr_tend(tr, u_f, v_f)
        t2 = self._tr_tend(tr + 0.5 * dt * t1, u_f, v_f)
        t3 = self._tr_tend(tr + 0.5 * dt * t2, u_f, v_f)
        t4 = self._tr_tend(tr + dt * t3, u_f, v_f)
        tr_adv = tr + (dt / 6.0) * (t1 + 2.0 * t2 + 2.0 * t3 + t4)
        self.tr = self._irfft(self._rfft(tr_adv) * self.state_filter)

    # ------------------------------------------------------------------ diagnostics
    def energy_total(self):
        """KE + screening energy = 0.5<u^2+v^2> + 0.5*INV_LD2*<psi^2>.
        The conserved quantity of the screened advection (domain-mean form)."""
        u, v = self.invert()
        psi = self._psi()
        return 0.5 * np.mean(u * u + v * v) + 0.5 * self.ild2 * np.mean(psi * psi)

    def enstrophy(self):
        return 0.5 * np.mean(self.w * self.w)


# ============================================================================
# Step 2 verification: energy + enstrophy conservation of the screened core.
# ============================================================================
def _random_band_limited_w(box, kcut=20.0, target_std=8.0, id_=None):
    """Random band-limited vorticity: every mode with |k| < kcut gets a random
    complex amplitude (stream `seed_noise`), normalized to a target RMS."""
    if id_ is None:
        id_ = config.SEEDS["seed_noise"]
    rng = np.random.default_rng(np.random.SeedSequence([777, id_]))
    mask = box.kmag < kcut
    w_hat = np.zeros(box.k2.shape, dtype=np.complex128)
    re = rng.standard_normal(box.k2.shape)
    im = rng.standard_normal(box.k2.shape)
    w_hat[mask] = (re + 1j * im)[mask]
    w = np.fft.irfft2(w_hat, s=box.shape)
    w *= target_std / w.std()
    return w


if __name__ == "__main__":
    # beta=0, screened=True, NO forcing, 500 steps; energy_total and enstrophy
    # must each drift < 1% relative. (KE alone exchanges ~24% with the psi^2
    # part under screening -- assert only on the TOTAL.)
    box = Box(beta=0.0, screened=True)
    box.w = _random_band_limited_w(box)

    def _ke_frac(b):
        u, v = b.invert()
        ke = 0.5 * np.mean(u * u + v * v)
        return ke, ke / b.energy_total()

    e0 = box.energy_total()
    z0 = box.enstrophy()
    ke0, kef0 = _ke_frac(box)

    n_steps = 500
    for _ in range(n_steps):
        box.step_spectral()

    e1 = box.energy_total()
    z1 = box.enstrophy()
    ke1, kef1 = _ke_frac(box)

    e_drift = abs(e1 - e0) / abs(e0) * 100.0
    z_drift = abs(z1 - z0) / abs(z0) * 100.0

    print(f"grid           : {box.ny} x {box.nx}  (DX={box.dx:.4e}, DT={box.dt:.4e})")
    print(f"beta={box.beta}  screened={box.screened}  steps={n_steps}")
    print(f"initial std(w) : {box.w.std():.4f} (evolved)   substeps={box.substep_count}")
    print(f"CFL (final)    : {box.cfl():.4f}   (2/3)pi*CFL={2.0/3.0*np.pi*box.cfl():.4f}")
    print("-" * 60)
    print(f"energy_total   : {e0:.10e} -> {e1:.10e}   drift = {e_drift:.3e} %")
    print(f"enstrophy      : {z0:.10e} -> {z1:.10e}   drift = {z_drift:.3e} %")
    print(f"KE fraction    : {kef0*100:.2f}% -> {kef1*100:.2f}%   "
          f"(KE moved {abs(kef1-kef0)*100:.2f} pts vs total)")
    print("-" * 60)
    ok = (e_drift < 1.0) and (z_drift < 1.0)
    print("CONSERVATION GATE:", "PASS" if ok else "FAIL")

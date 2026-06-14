"""The stepper: orchestrates one simulation step across the three domains.

Domains: the equirect main grid plus two square azimuthal-equidistant polar
patches (same kernels, compiled per-domain via the DOMAIN define). Per step:
drift vortex centers (CPU), rebuild psi and the frozen velocity per domain,
run the three MacCormack passes per domain, then the one-way nesting exchange
— equirect is authoritative equatorward of the exchange band, patches
poleward; per-step resampling keeps the overlap slaved so the final composite
feather has nothing to hide (the design review's anti-ghosting requirement).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gasgiant.gl import GpuContext
from gasgiant.params.model import PlanetParams, PoleParams, PoleStyle, SolverType
from gasgiant.params.seeds import subseed
from gasgiant.sim.advance import advance_registry
from gasgiant.sim.profiles import LatProfiles
from gasgiant.sim.tracers import TracerState
from gasgiant.sim.vortices import VortexRegistry

if TYPE_CHECKING:
    import moderngl

log = logging.getLogger(__name__)

_KERNELS = "gasgiant.sim.kernels"
_GROUP = 16

# Patch geometry and exchange/blend bands (radians).
RHO_MAX = np.deg2rad(34.0)            # patch covers poleward of 56 deg
EXCHANGE_TO_PATCH = (np.deg2rad(63.0), np.deg2rad(65.0))
EXCHANGE_TO_EQ = (np.deg2rad(65.0), np.deg2rad(67.0))
BLEND_BAND = (np.deg2rad(64.0), np.deg2rad(67.0))  # derive-time feather

# Estimated peak tangential speed contributed by vortices.
_VORTEX_SPEED_MARGIN = 0.45


def compute_dt(resolution: int, dt_scale: float, profiles_max_speed: float) -> float:
    """The advection timestep (~1.2 cells of jet displacement per step).
    Module-level so registry generation (seeded merger pairs) can target
    merge steps with the same dt the solver will actually use."""
    cell = 2.0 * np.pi / resolution
    max_speed = max(profiles_max_speed + _VORTEX_SPEED_MARGIN, 0.3)
    return float(dt_scale * 1.2 * cell / max_speed)

DOMAIN_EQUIRECT = 0
DOMAIN_NORTH = 1
DOMAIN_SOUTH = 2


def _set(prog: moderngl.ComputeShader, name: str, value) -> None:
    """Set a uniform if the compiler kept it (passes compile out unused ones)."""
    with contextlib.suppress(KeyError):
        prog[name].value = value


def patch_resolution(equirect_width: int) -> int:
    """Patch grid size matching the equirect angular resolution."""
    n = int(round(equirect_width * RHO_MAX / np.pi / _GROUP)) * _GROUP
    return max(n, 64)


@dataclass
class Domain:
    kind: int
    size: tuple[int, int]
    tracers: TracerState
    psi_tex: moderngl.Texture
    vel_tex: moderngl.Texture
    k_psi: moderngl.ComputeShader
    k_vel: moderngl.ComputeShader
    k_adv: list[moderngl.ComputeShader]
    k_init: moderngl.ComputeShader

    def groups(self) -> tuple[int, int]:
        return (
            (self.size[0] + _GROUP - 1) // _GROUP,
            (self.size[1] + _GROUP - 1) // _GROUP,
        )

    def release(self) -> None:
        self.tracers.release()
        self.psi_tex.release()
        self.vel_tex.release()


@dataclass
class _OmegaState:
    """Ping-pong textures and compiled kernels for the equirect ω field."""
    # Two R32F textures (ping-pong) holding absolute vorticity q.
    cur: moderngl.Texture
    fwd: moderngl.Texture
    back: moderngl.Texture
    out: moderngl.Texture
    # Scratch R32F for ∇²ω_rel (used between omega_force subpasses).
    lap_scratch: moderngl.Texture
    # Kernels
    k_init: moderngl.ComputeShader          # omega_init.comp
    k_adv: list[moderngl.ComputeShader]     # omega_advect.comp PASS 0,1,2
    k_force0: moderngl.ComputeShader        # omega_force.comp SUBPASS 0 (nudge)
    k_lap: moderngl.ComputeShader           # omega_lap.comp (∇²ω_rel)
    k_force1: moderngl.ComputeShader        # omega_force.comp SUBPASS 1 (hypervisc)

    def commit(self) -> None:
        self.cur, self.out = self.out, self.cur

    def release(self) -> None:
        for tex in (self.cur, self.fwd, self.back, self.out, self.lap_scratch):
            tex.release()


class Solver:
    def __init__(
        self,
        gpu: GpuContext,
        params: PlanetParams,
        profiles: LatProfiles,
        vortices: VortexRegistry,
        profile_dyn_tex: moderngl.Texture,
        profile_stamp_tex: moderngl.Texture,
        wave_lats: tuple[float, float] = (0.12, 0.82),
        events: object | None = None,
        profile_omega_tex: moderngl.Texture | None = None,
    ) -> None:
        self.gpu = gpu
        self.params = params
        self.profiles = profiles
        self.vortices = vortices
        self.profile_dyn = profile_dyn_tex
        self.profile_stamp = profile_stamp_tex
        self.wave_lats = wave_lats
        self.events = events
        self.profile_omega = profile_omega_tex

        w = params.sim.resolution
        n = patch_resolution(w)
        self.domains = [
            self._make_domain(DOMAIN_EQUIRECT, (w, w // 2)),
            self._make_domain(DOMAIN_NORTH, (n, n)),
            self._make_domain(DOMAIN_SOUTH, (n, n)),
        ]
        self.equirect = self.domains[0]
        self.north = self.domains[1]
        self.south = self.domains[2]

        self.k_x_to_patch = [
            gpu.compute(_KERNELS, "exchange_to_patch.comp", defines={"DOMAIN": str(d)})
            for d in (DOMAIN_NORTH, DOMAIN_SOUTH)
        ]
        self.k_x_to_eq = gpu.compute(
            _KERNELS, "exchange_to_equirect.comp", defines={"DOMAIN": "0"}
        )

        self.step_index = 0
        self.dt = self._compute_dt()
        self._ssbo = gpu.ssbo(vortices.pack_ssbo(), binding=2)

        # Vorticity-mode state (equirect domain only; built regardless of
        # solver.type so the uniforms path doesn't branch at compile time).
        self._omega_state: _OmegaState | None = None
        if params.solver.type == SolverType.VORTICITY:
            self._omega_state = self._build_omega_state()

        self._static_uniforms()

    def _make_domain(self, kind: int, size: tuple[int, int]) -> Domain:
        gpu = self.gpu
        defines = {"DOMAIN": str(kind)}
        wrap = kind == DOMAIN_EQUIRECT
        # Zero-filled: with dev_steps == 0 the first derive runs before any
        # step has written these, and an undefined-content texture would feed
        # the detail backtrace whatever VRAM held before.
        psi_tex = gpu.texture2d(size, 1, "f4", data=np.zeros((size[1], size[0], 1), np.float32))
        vel_tex = gpu.texture2d(
            size, 2, "f4", data=np.zeros((size[1], size[0], 2), np.float32), linear=True
        )
        vel_tex.repeat_x = wrap
        tracers = TracerState(gpu, size)
        for tex in (tracers.cur, tracers.fwd, tracers.back, tracers.out):
            tex.repeat_x = wrap
        return Domain(
            kind=kind,
            size=size,
            tracers=tracers,
            psi_tex=psi_tex,
            vel_tex=vel_tex,
            k_psi=gpu.compute(_KERNELS, "psi.comp", defines=defines),
            k_vel=gpu.compute(_KERNELS, "velocity.comp", defines=defines),
            k_adv=[
                gpu.compute(_KERNELS, "advect.comp", defines={**defines, "PASS": str(i)})
                for i in range(3)
            ],
            k_init=gpu.compute(_KERNELS, "init.comp", defines=defines),
        )

    def _build_omega_state(self) -> _OmegaState:
        """Build textures and kernels for the equirect vorticity field."""
        gpu = self.gpu
        size = self.equirect.size
        eq_defines = {"DOMAIN": "0"}

        def r32f(sz):
            return gpu.texture2d(sz, 1, "f4",
                                 data=np.zeros((sz[1], sz[0], 1), np.float32))

        cur        = r32f(size)
        fwd        = r32f(size)
        back       = r32f(size)
        out        = r32f(size)
        lap_scratch = r32f(size)
        # All omega textures periodic in x (equirect).
        for tex in (cur, fwd, back, out, lap_scratch):
            tex.repeat_x = True

        k_init = gpu.compute(_KERNELS, "omega_init.comp", defines=eq_defines)
        k_adv = [
            gpu.compute(_KERNELS, "omega_advect.comp",
                        defines={**eq_defines, "PASS": str(i)})
            for i in range(3)
        ]
        k_force0 = gpu.compute(_KERNELS, "omega_force.comp",
                               defines={**eq_defines, "SUBPASS": "0"})
        k_lap    = gpu.compute(_KERNELS, "omega_lap.comp",  defines=eq_defines)
        k_force1 = gpu.compute(_KERNELS, "omega_force.comp",
                               defines={**eq_defines, "SUBPASS": "1"})

        state = _OmegaState(
            cur=cur, fwd=fwd, back=back, out=out, lap_scratch=lap_scratch,
            k_init=k_init, k_adv=k_adv,
            k_force0=k_force0, k_lap=k_lap, k_force1=k_force1,
        )
        # Initialise q⁰ immediately.
        self._omega_init(state)
        return state

    def _omega_static_uniforms(self, state: _OmegaState) -> None:
        """Set size / f0 uniforms that never change after build."""
        size = self.equirect.size
        p = self.params
        for k in [state.k_init, *state.k_adv, state.k_force0, state.k_lap, state.k_force1]:
            _set(k, "u_size", size)
        for k in [state.k_init, state.k_force0, state.k_lap, state.k_force1]:
            _set(k, "u_coriolis_f0", p.solver.coriolis_f0)
        _set(state.k_force0, "u_relax_tau", p.solver.vort_relax_tau)
        _set(state.k_force1, "u_hypervisc", p.solver.vort_hypervisc)

    def _omega_init(self, state: _OmegaState) -> None:
        """Dispatch omega_init.comp to write q⁰ into state.cur."""
        ctx = self.gpu.ctx
        k = state.k_init
        size = self.equirect.size
        _set(k, "u_size", size)
        _set(k, "u_vortex_count", len(self.vortices.vortices))
        _set(k, "u_coriolis_f0", self.params.solver.coriolis_f0)
        if self.profile_omega is not None:
            self.profile_omega.use(location=0)
            _set(k, "u_profile_omega", 0)
        self._ssbo.bind_to_storage_buffer(2)
        state.cur.bind_to_image(0, read=False, write=True)
        gx, gy = self.equirect.groups()
        k.run(gx, gy, 1)
        ctx.memory_barrier()

    def _omega_step(self, state: _OmegaState) -> None:
        """Evolve ω one step: MacCormack advection + forcing + hyperviscosity."""
        ctx = self.gpu.ctx
        p = self.params
        gx, gy = self.equirect.groups()
        vel_tex = self.equirect.vel_tex  # frozen kinematic velocity (P3a)

        # Bind SSBO for vortex contributions.
        self._ssbo.bind_to_storage_buffer(2)

        # MacCormack pass 0 (forward).
        k0 = state.k_adv[0]
        state.cur.use(location=0)
        _set(k0, "u_src", 0)
        vel_tex.use(location=1)
        _set(k0, "u_vel", 1)
        # Unused samplers still bound (determinism).
        state.cur.use(location=2)
        _set(k0, "u_cur", 2)
        state.cur.use(location=3)
        _set(k0, "u_back", 3)
        _set(k0, "u_dt", +self.dt)
        state.fwd.bind_to_image(0, read=False, write=True)
        k0.run(gx, gy, 1)
        ctx.memory_barrier()

        # MacCormack pass 1 (backward).
        k1 = state.k_adv[1]
        state.fwd.use(location=0)
        _set(k1, "u_src", 0)
        vel_tex.use(location=1)
        _set(k1, "u_vel", 1)
        state.cur.use(location=2)
        _set(k1, "u_cur", 2)
        state.cur.use(location=3)
        _set(k1, "u_back", 3)
        _set(k1, "u_dt", -self.dt)
        state.back.bind_to_image(0, read=False, write=True)
        k1.run(gx, gy, 1)
        ctx.memory_barrier()

        # MacCormack pass 2 (correct).
        k2 = state.k_adv[2]
        state.fwd.use(location=0)
        _set(k2, "u_src", 0)
        vel_tex.use(location=1)
        _set(k2, "u_vel", 1)
        state.cur.use(location=2)
        _set(k2, "u_cur", 2)
        state.back.use(location=3)
        _set(k2, "u_back", 3)
        _set(k2, "u_dt", +self.dt)
        state.out.bind_to_image(0, read=False, write=True)
        k2.run(gx, gy, 1)
        ctx.memory_barrier()
        state.commit()  # cur <- out (out becomes scratch)

        # Forcing / nudging (SUBPASS 0): q += (q_target − q) / τ_ω.
        kf0 = state.k_force0
        state.cur.use(location=0)
        _set(kf0, "u_omega", 0)
        if self.profile_omega is not None:
            self.profile_omega.use(location=1)
            _set(kf0, "u_profile_omega", 1)
        # u_lap_omega_rel not used in SUBPASS 0; bind to something safe.
        state.cur.use(location=2)
        _set(kf0, "u_lap_omega_rel", 2)
        _set(kf0, "u_vortex_count", len(self.vortices.vortices))
        _set(kf0, "u_coriolis_f0", p.solver.coriolis_f0)
        _set(kf0, "u_relax_tau", p.solver.vort_relax_tau)
        _set(kf0, "u_hypervisc", p.solver.vort_hypervisc)
        state.out.bind_to_image(0, read=False, write=True)
        kf0.run(gx, gy, 1)
        ctx.memory_barrier()
        state.commit()  # cur <- out

        # Compute ∇²ω_rel into lap_scratch.
        kl = state.k_lap
        state.cur.use(location=0)
        _set(kl, "u_omega", 0)
        _set(kl, "u_coriolis_f0", p.solver.coriolis_f0)
        state.lap_scratch.bind_to_image(0, read=False, write=True)
        kl.run(gx, gy, 1)
        ctx.memory_barrier()

        # Hyperviscosity (SUBPASS 1): q += ν₄ · (−∇⁴ω_rel).
        kf1 = state.k_force1
        state.cur.use(location=0)
        _set(kf1, "u_omega", 0)
        # u_profile_omega not used in SUBPASS 1; bind to something safe.
        if self.profile_omega is not None:
            self.profile_omega.use(location=1)
            _set(kf1, "u_profile_omega", 1)
        state.lap_scratch.use(location=2)
        _set(kf1, "u_lap_omega_rel", 2)
        _set(kf1, "u_vortex_count", len(self.vortices.vortices))
        _set(kf1, "u_coriolis_f0", p.solver.coriolis_f0)
        _set(kf1, "u_relax_tau", p.solver.vort_relax_tau)
        _set(kf1, "u_hypervisc", p.solver.vort_hypervisc)
        state.out.bind_to_image(0, read=False, write=True)
        kf1.run(gx, gy, 1)
        ctx.memory_barrier()
        state.commit()  # cur is now the final advanced q

    # -- configuration ---------------------------------------------------------

    def _compute_dt(self) -> float:
        return compute_dt(
            self.equirect.size[0], self.params.sim.dt_scale, self.profiles.max_speed
        )

    def _wave_uniforms(self, prog: moderngl.ComputeShader) -> None:
        p = self.params
        fest_lat, rib_lat = self.wave_lats
        _set(prog, "u_fest_amp", p.waves.festoon_strength)
        _set(prog, "u_fest_lat", fest_lat)
        _set(prog, "u_fest_k", float(p.waves.festoon_wavenumber))
        _set(prog, "u_fest_phase", self._fest_phase)
        _set(prog, "u_hotspot_depth", p.waves.hotspot_depth)
        _set(prog, "u_rib_amp", p.waves.ribbon_strength)
        _set(prog, "u_rib_lat", rib_lat)
        _set(prog, "u_rib_k", float(p.waves.ribbon_wavenumber))
        _set(prog, "u_rib_phase", self._rib_phase)

    def _poly_uniforms(self, prog: moderngl.ComputeShader, pole: PoleParams) -> None:
        enabled = pole.style == PoleStyle.POLYGON_JET and pole.strength > 0.0
        _set(prog, "u_poly_amp", 0.016 * pole.strength if enabled else 0.0)
        _set(prog, "u_poly_k", float(pole.polygon_sides))
        _set(prog, "u_poly_rho", 0.21)
        _set(prog, "u_poly_eps", 0.12)
        _set(prog, "u_poly_phase", self._poly_phase)
        _set(prog, "u_poly_width", 0.03)

    def _band_mod_uniforms(self, prog: moderngl.ComputeShader) -> None:
        p = self.params
        _set(prog, "u_band_variance", p.bands.variance_amount)
        _set(prog, "u_variance_offset", self._variance_offset)
        _set(prog, "u_env_strength", p.bands.contrast_envelope)
        _set(prog, "u_fade_amp", p.bands.faded_sector)
        _set(prog, "u_fade_sector", tuple(self.profiles.fade_sector))

    def _static_uniforms(self) -> None:
        p = self.params
        warp_rng = subseed(p.seed, "warp-noise")
        detail_rng = subseed(p.seed, "detail-noise")
        turb_rng = subseed(p.seed, "turbulence")
        kh_rng = subseed(p.seed, "kh-wave")
        self._variance_offset = tuple(
            subseed(p.seed, "band-variance").uniform(-100.0, 100.0, 3)
        )
        self._warp_offset = tuple(warp_rng.uniform(-100.0, 100.0, 3))
        self._detail_offset = tuple(detail_rng.uniform(-100.0, 100.0, 3))
        self._turb_offset = tuple(turb_rng.uniform(-100.0, 100.0, 3))
        self._kh_phase = float(kh_rng.uniform(0.0, 2.0 * np.pi))
        self._poly_phase = float(subseed(p.seed, "poly-jet").uniform(0.0, 2.0 * np.pi))
        wave_rng = subseed(p.seed, "eq-waves")
        self._fest_phase = float(wave_rng.uniform(0.0, 2.0 * np.pi))
        self._rib_phase = float(wave_rng.uniform(0.0, 2.0 * np.pi))

        relax_k = 1.0 / max(p.turbulence.relax_tau, 1.0)
        for dom in self.domains:
            pole = p.poles.north if dom.kind == DOMAIN_NORTH else p.poles.south

            k = dom.k_psi
            _set(k, "u_size", dom.size)
            _set(k, "u_rho_max", RHO_MAX)
            _set(k, "u_warp_offset", self._warp_offset)
            _set(k, "u_warp_amount", p.bands.warp_amount)
            _set(k, "u_warp_freq", p.bands.warp_freq)
            if dom.kind != DOMAIN_EQUIRECT:
                self._poly_uniforms(k, pole)
            else:
                self._wave_uniforms(k)

            _set(dom.k_vel, "u_size", dom.size)
            _set(dom.k_vel, "u_rho_max", RHO_MAX)
            _set(dom.k_vel, "u_outbreak_count", 0)

            for i, prog in enumerate(dom.k_adv):
                _set(prog, "u_size", dom.size)
                _set(prog, "u_rho_max", RHO_MAX)
                if i == 2:
                    _set(prog, "u_relax_k", relax_k)
                    _set(prog, "u_replenish", p.turbulence.replenish_rate)
                    _set(prog, "u_rim_contrast", p.storms.rim_contrast)
                    _set(prog, "u_warp_offset", self._warp_offset)
                    _set(prog, "u_warp_amount", p.bands.warp_amount)
                    _set(prog, "u_warp_freq", p.bands.warp_freq)
                    _set(prog, "u_detail_offset", self._detail_offset)
                    _set(prog, "u_detail_freq", p.bands.detail_freq)
                    _set(prog, "u_belt_replenish", p.turbulence.belt_replenish)
                    _set(prog, "u_belt_scale", p.turbulence.belt_replenish_scale)
                    if dom.kind == DOMAIN_EQUIRECT and p.turbulence.belt_replenish > 0.0:
                        import math
                        scale = p.turbulence.belt_replenish_scale
                        finest_wavelen = 1.5 / (p.bands.detail_freq * 2.0 * scale)
                        nyquist = 8.0 * (2.0 * math.pi / p.sim.resolution)
                        if scale > 1.0 and finest_wavelen < nyquist:
                            import warnings
                            warnings.warn(
                                f"belt_replenish_scale={scale}: finest injection octave "
                                f"(~{finest_wavelen:.4f} noise-units) is below ~2 cells at "
                                f"resolution {p.sim.resolution} "
                                f"(detail_freq={p.bands.detail_freq}); "
                                f"sub-Nyquist energy will be dissipated/aliased.",
                                stacklevel=2,
                            )
                    self._band_mod_uniforms(prog)
                    if dom.kind != DOMAIN_EQUIRECT:
                        self._poly_uniforms(prog, pole)
                    else:
                        self._wave_uniforms(prog)

            k = dom.k_init
            _set(k, "u_size", dom.size)
            _set(k, "u_rho_max", RHO_MAX)
            _set(k, "u_rim_contrast", p.storms.rim_contrast)
            _set(k, "u_warp_offset", self._warp_offset)
            _set(k, "u_warp_amount", p.bands.warp_amount)
            _set(k, "u_warp_freq", p.bands.warp_freq)
            _set(k, "u_detail_offset", self._detail_offset)
            _set(k, "u_detail_amount", p.bands.detail_amount)
            _set(k, "u_detail_freq", p.bands.detail_freq)
            self._band_mod_uniforms(k)
            if dom.kind != DOMAIN_EQUIRECT:
                self._poly_uniforms(k, pole)
            else:
                self._wave_uniforms(k)

        for prog in self.k_x_to_patch:
            _set(prog, "u_rho_max", RHO_MAX)
            _set(prog, "u_ex_lo", EXCHANGE_TO_PATCH[0])
            _set(prog, "u_ex_hi", EXCHANGE_TO_PATCH[1])
        _set(self.k_x_to_eq, "u_size", self.equirect.size)
        _set(self.k_x_to_eq, "u_patch_rho_max", RHO_MAX)
        _set(self.k_x_to_eq, "u_ex_lo", EXCHANGE_TO_EQ[0])
        _set(self.k_x_to_eq, "u_ex_hi", EXCHANGE_TO_EQ[1])

        self.apply_velocity_params()

    def apply_velocity_params(self) -> None:
        """VELOCITY-tier uniforms; cheap to re-apply when those params change."""
        p = self.params
        for dom in self.domains:
            k = dom.k_psi
            _set(k, "u_turb_intensity", p.turbulence.intensity)
            _set(k, "u_turb_scale", p.turbulence.scale)
            _set(k, "u_turb_shear", p.turbulence.shear_coupling)
            _set(k, "u_turb_belt", p.turbulence.belt_boost)
            _set(k, "u_turb_offset", self._turb_offset)
            _set(k, "u_kh_amplitude", p.turbulence.kh_amplitude)
            _set(k, "u_kh_wavenumber", float(p.turbulence.kh_wavenumber))
            _set(k, "u_kh_phase", self._kh_phase)
            _set(k, "u_wake_gain", p.storms.wake_turbulence)

    def set_profiles(self, profiles: LatProfiles) -> None:
        self.profiles = profiles
        self.dt = self._compute_dt()

    @property
    def warp_offset(self) -> tuple[float, float, float]:
        """The seeded band-meander noise offset (derive-time lanes ride it)."""
        return self._warp_offset

    # -- initialization ----------------------------------------------------------

    def init_tracers(self) -> None:
        for dom in self.domains:
            k = dom.k_init
            _set(k, "u_vortex_count", len(self.vortices.vortices))
            self.profile_stamp.use(location=0)
            _set(k, "u_profile_stamp", 0)
            dom.tracers.cur.bind_to_image(0, read=False, write=True)
            gx, gy = dom.groups()
            k.run(gx, gy, 1)
            self.gpu.ctx.memory_barrier()
        self._exchange()

    # -- stepping -------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        ctx = self.gpu.ctx
        for _ in range(n):
            # 1. Events (outbreak spawn/decay), vortex drift, SSBO refresh.
            impulses = advance_registry(
                self.vortices, self.profiles, self.dt, self.step_index,
                self.events, self.params.storms,
            )
            ssbo_data = self.vortices.pack_ssbo()
            if ssbo_data.nbytes > self._ssbo.size:
                self._ssbo.orphan(ssbo_data.nbytes)
            self._ssbo.write(ssbo_data.tobytes())
            self._ssbo.bind_to_storage_buffer(2)

            turb_time = self.step_index * self.params.turbulence.evolution_rate

            for dom in self.domains:
                gx, gy = dom.groups()

                # 2. Produce psi for this step (kinematic analytic rebuild, or
                #    vorticity solve in vorticity mode — see _produce_psi).
                self._produce_psi(dom, turb_time, gx, gy)

                # 3. Frozen velocity for all three MacCormack passes.
                dom.psi_tex.use(location=0)
                _set(dom.k_vel, "u_psi", 0)
                self.profile_stamp.use(location=1)
                _set(dom.k_vel, "u_profile_stamp", 1)
                if dom.kind == DOMAIN_EQUIRECT:
                    _set(dom.k_vel, "u_outbreak_count", len(impulses))
                    if impulses:
                        flat = np.zeros((2, 4), dtype=np.float32)
                        for i, imp in enumerate(impulses[:2]):
                            flat[i] = imp
                        with contextlib.suppress(KeyError):
                            dom.k_vel["u_outbreaks"].write(flat.tobytes())
                dom.vel_tex.bind_to_image(0, read=False, write=True)
                dom.k_vel.run(gx, gy, 1)
                ctx.memory_barrier()

                # 4. MacCormack: forward, backward, correct.
                t = dom.tracers
                self._advect(dom, 0, src=t.cur, dst=t.fwd, dt=+self.dt)
                self._advect(dom, 1, src=t.fwd, dst=t.back, dt=-self.dt)
                self._correct(dom, turb_time)
                t.commit()

            # 5. One-way nesting exchange.
            self._exchange()
            self.step_index += 1

    def _produce_psi(self, dom, turb_time, gx, gy):
        """Write dom.psi_tex for this step.

        KINEMATIC mode: analytic streamfunction rebuild via psi.comp (v1.5).
        VORTICITY mode, equirect domain (P3a): advance ω one step, then write
          psi_tex via the same analytic kinematic path as KINEMATIC mode.
          This keeps velocity kinematic in P3a — the Poisson ψ-from-ω solve
          is deferred to P3b.  Polar patch domains always use the kinematic
          path regardless of solver.type.
        """
        ctx = self.gpu.ctx

        if (
            self.params.solver.type == SolverType.VORTICITY
            and dom.kind == DOMAIN_EQUIRECT
            and self._omega_state is not None
        ):
            self._omega_step(self._omega_state)

        # Write psi_tex analytically (kinematic path) for ALL domains in P3a.
        # This keeps velocity unchanged; the Poisson inversion (P3b) will
        # replace this line with a SOR solve for the equirect domain only.
        k = dom.k_psi
        _set(k, "u_vortex_count", len(self.vortices.vortices))
        _set(k, "u_turb_time", turb_time)
        self.profile_dyn.use(location=0)
        _set(k, "u_profile_dyn", 0)
        dom.psi_tex.bind_to_image(0, read=False, write=True)
        k.run(gx, gy, 1)
        ctx.memory_barrier()

    def _advect(
        self, dom: Domain, pass_index: int, src: moderngl.Texture,
        dst: moderngl.Texture, dt: float,
    ) -> None:
        prog = dom.k_adv[pass_index]
        src.use(location=0)
        _set(prog, "u_src", 0)
        dom.vel_tex.use(location=1)
        _set(prog, "u_vel", 1)
        _set(prog, "u_dt", dt)
        dst.bind_to_image(0, read=False, write=True)
        gx, gy = dom.groups()
        prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()

    def _correct(self, dom: Domain, turb_time: float) -> None:
        prog = dom.k_adv[2]
        t = dom.tracers
        t.fwd.use(location=0)
        _set(prog, "u_src", 0)
        dom.vel_tex.use(location=1)
        _set(prog, "u_vel", 1)
        t.cur.use(location=2)
        _set(prog, "u_cur", 2)
        t.back.use(location=3)
        _set(prog, "u_back", 3)
        self.profile_stamp.use(location=4)
        _set(prog, "u_profile_stamp", 4)
        self.profile_dyn.use(location=5)
        _set(prog, "u_profile_dyn", 5)
        _set(prog, "u_dt", +self.dt)
        _set(prog, "u_turb_time", turb_time)
        _set(prog, "u_vortex_count", len(self.vortices.vortices))
        t.out.bind_to_image(0, read=False, write=True)
        gx, gy = dom.groups()
        prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()

    def _exchange(self) -> None:
        ctx = self.gpu.ctx
        # equirect -> patches (outer boundary condition).
        for prog, dom in zip(self.k_x_to_patch, (self.north, self.south), strict=True):
            self.equirect.tracers.cur.use(location=0)
            _set(prog, "u_equirect", 0)
            _set(prog, "u_size", dom.size)
            dom.tracers.cur.bind_to_image(0, read=True, write=True)
            gx, gy = dom.groups()
            prog.run(gx, gy, 1)
        ctx.memory_barrier()
        # patches -> equirect (polar rows).
        prog = self.k_x_to_eq
        self.north.tracers.cur.use(location=0)
        _set(prog, "u_patch_n", 0)
        self.south.tracers.cur.use(location=1)
        _set(prog, "u_patch_s", 1)
        self.equirect.tracers.cur.bind_to_image(0, read=True, write=True)
        gx, gy = self.equirect.groups()
        prog.run(gx, gy, 1)
        ctx.memory_barrier()

    # -- diagnostics ------------------------------------------------------------------

    def exchange_band_rms(self) -> float:
        """RMS difference between equirect and north-patch T0 in the exchange
        band — the cross-domain consistency health metric."""
        eq = self.gpu.read_texture(self.equirect.tracers.cur)
        npatch = self.gpu.read_texture(self.north.tracers.cur)
        h, w = eq.shape[:2]
        lat_lo, lat_hi = np.deg2rad(64.0), np.deg2rad(66.0)
        lats = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
        rows = np.where((lats > lat_lo) & (lats < lat_hi))[0]
        if rows.size == 0:
            return 0.0
        diffs = []
        n = npatch.shape[0]
        for j in rows:
            lat = lats[j]
            rho = np.pi / 2 - lat
            lons = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
            s = rho * np.cos(lons)
            t = rho * np.sin(lons)
            xi = np.clip(((s / RHO_MAX) * 0.5 + 0.5) * n, 0, n - 1).astype(int)
            yi = np.clip(((t / RHO_MAX) * 0.5 + 0.5) * n, 0, n - 1).astype(int)
            diffs.append(eq[j, :, 0] - npatch[yi, xi, 0])
        return float(np.sqrt(np.mean(np.concatenate(diffs) ** 2)))

    def release(self) -> None:
        for dom in self.domains:
            dom.release()
        self._ssbo.release()
        if self._omega_state is not None:
            self._omega_state.release()
            self._omega_state = None

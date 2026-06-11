"""The stepper: orchestrates one simulation step.

Per step: drift vortex centers with the zonal flow (CPU), rebuild psi
(turbulence decorrelates via its time uniform — BETWEEN steps only, never
mid-step), derive the frozen velocity, then the three MacCormack passes.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import numpy as np

from gasgiant.gl import GpuContext
from gasgiant.params.model import PlanetParams
from gasgiant.params.seeds import subseed
from gasgiant.sim.profiles import LatProfiles
from gasgiant.sim.tracers import TracerState
from gasgiant.sim.vortices import VortexRegistry

if TYPE_CHECKING:
    import moderngl

log = logging.getLogger(__name__)

_KERNELS = "gasgiant.sim.kernels"
_GROUP = 16

# Estimated peak tangential speed contributed by vortices (Gaussian-psi
# vortex: u_peak ~ 0.858 * strength / r_core, bounded by construction).
_VORTEX_SPEED_MARGIN = 0.45


def _set(prog: moderngl.ComputeShader, name: str, value) -> None:
    """Set a uniform if the compiler kept it (passes compile out unused ones)."""
    with contextlib.suppress(KeyError):
        prog[name].value = value


class Solver:
    def __init__(
        self,
        gpu: GpuContext,
        params: PlanetParams,
        profiles: LatProfiles,
        vortices: VortexRegistry,
        tracers: TracerState,
        profile_dyn_tex: moderngl.Texture,
        profile_stamp_tex: moderngl.Texture,
    ) -> None:
        self.gpu = gpu
        self.params = params
        self.profiles = profiles
        self.vortices = vortices
        self.tracers = tracers
        self.profile_dyn = profile_dyn_tex
        self.profile_stamp = profile_stamp_tex
        size = tracers.size
        self.size = size

        self.psi_tex = gpu.texture2d(size, 1, "f4")
        self.vel_tex = gpu.texture2d(size, 2, "f4", linear=True)

        self.k_psi = gpu.compute(_KERNELS, "psi.comp")
        self.k_vel = gpu.compute(_KERNELS, "velocity.comp")
        self.k_adv = [
            gpu.compute(_KERNELS, "advect.comp", defines={"PASS": str(i)}) for i in range(3)
        ]

        self.step_index = 0
        self.dt = self._compute_dt()
        self._ssbo = gpu.ssbo(vortices.pack_ssbo(), binding=2)
        self._static_uniforms()

    def _compute_dt(self) -> float:
        cell = 2.0 * np.pi / self.size[0]
        max_speed = max(self.profiles.max_speed + _VORTEX_SPEED_MARGIN, 0.3)
        return float(self.params.sim.dt_scale * 1.2 * cell / max_speed)

    def _groups(self) -> tuple[int, int]:
        return (self.size[0] + _GROUP - 1) // _GROUP, (self.size[1] + _GROUP - 1) // _GROUP

    def _static_uniforms(self) -> None:
        p = self.params
        warp_rng = subseed(p.seed, "warp-noise")
        detail_rng = subseed(p.seed, "detail-noise")
        turb_rng = subseed(p.seed, "turbulence")
        kh_rng = subseed(p.seed, "kh-wave")
        self._warp_offset = tuple(warp_rng.uniform(-100.0, 100.0, 3))
        self._detail_offset = tuple(detail_rng.uniform(-100.0, 100.0, 3))
        self._turb_offset = tuple(turb_rng.uniform(-100.0, 100.0, 3))
        self._kh_phase = float(kh_rng.uniform(0.0, 2.0 * np.pi))

        k = self.k_psi
        _set(k, "u_size", self.size)
        _set(k, "u_warp_offset", self._warp_offset)
        _set(k, "u_warp_amount", p.bands.warp_amount)
        _set(k, "u_warp_freq", p.bands.warp_freq)
        self.apply_velocity_params()

        _set(self.k_vel, "u_size", self.size)

        relax_k = 1.0 / max(p.turbulence.relax_tau, 1.0)
        for i, prog in enumerate(self.k_adv):
            _set(prog, "u_size", self.size)
            if i == 2:
                _set(prog, "u_relax_k", relax_k)
                _set(prog, "u_replenish", p.turbulence.replenish_rate)
                _set(prog, "u_warp_offset", self._warp_offset)
                _set(prog, "u_warp_amount", p.bands.warp_amount)
                _set(prog, "u_warp_freq", p.bands.warp_freq)
                _set(prog, "u_detail_offset", self._detail_offset)
                _set(prog, "u_detail_freq", p.bands.detail_freq)

    def apply_velocity_params(self) -> None:
        """VELOCITY-tier uniforms; cheap to re-apply when those params change."""
        p = self.params
        k = self.k_psi
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

    # -- stepping -------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        gx, gy = self._groups()
        ctx = self.gpu.ctx
        for _ in range(n):
            # 1. Vortex drift (CPU registry -> SSBO).
            self.vortices.drift(self.profiles, self.dt)
            self._ssbo.write(self.vortices.pack_ssbo().tobytes())
            self._ssbo.bind_to_storage_buffer(2)

            # 2. Rebuild psi (turbulence time advances BETWEEN steps).
            k = self.k_psi
            _set(k, "u_vortex_count", len(self.vortices.vortices))
            _set(k, "u_turb_time", self.step_index * self.params.turbulence.evolution_rate)
            self.profile_dyn.use(location=0)
            _set(k, "u_profile_dyn", 0)
            self.psi_tex.bind_to_image(0, read=False, write=True)
            k.run(gx, gy, 1)
            ctx.memory_barrier()

            # 3. Frozen velocity for all three MacCormack passes.
            self.psi_tex.use(location=0)
            _set(self.k_vel, "u_psi", 0)
            self.profile_stamp.use(location=1)
            _set(self.k_vel, "u_profile_stamp", 1)
            self.vel_tex.bind_to_image(0, read=False, write=True)
            self.k_vel.run(gx, gy, 1)
            ctx.memory_barrier()

            # 4. MacCormack: forward, backward, correct.
            t = self.tracers
            self._advect(0, src=t.cur, dst=t.fwd, dt=+self.dt)
            self._advect(1, src=t.fwd, dst=t.back, dt=-self.dt)
            self._correct()
            t.commit()
            self.step_index += 1

    def _advect(
        self, pass_index: int, src: moderngl.Texture, dst: moderngl.Texture, dt: float
    ) -> None:
        prog = self.k_adv[pass_index]
        src.use(location=0)
        _set(prog, "u_src", 0)
        self.vel_tex.use(location=1)
        _set(prog, "u_vel", 1)
        _set(prog, "u_dt", dt)
        dst.bind_to_image(0, read=False, write=True)
        gx, gy = self._groups()
        prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()

    def _correct(self) -> None:
        prog = self.k_adv[2]
        t = self.tracers
        t.fwd.use(location=0)
        _set(prog, "u_src", 0)
        self.vel_tex.use(location=1)
        _set(prog, "u_vel", 1)
        t.cur.use(location=2)
        _set(prog, "u_cur", 2)
        t.back.use(location=3)
        _set(prog, "u_back", 3)
        self.profile_stamp.use(location=4)
        _set(prog, "u_profile_stamp", 4)
        _set(prog, "u_dt", +self.dt)
        _set(prog, "u_turb_time", self.step_index * self.params.turbulence.evolution_rate)
        _set(prog, "u_vortex_count", len(self.vortices.vortices))
        t.out.bind_to_image(0, read=False, write=True)
        gx, gy = self._groups()
        prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()

    def release(self) -> None:
        self.psi_tex.release()
        self.vel_tex.release()
        self._ssbo.release()

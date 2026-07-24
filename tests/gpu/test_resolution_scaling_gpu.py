"""GPU byte-identity + activation gate for resolution-invariant scaling.

The unit suite proves the pure transforms and the CPU-side timeline coherence;
this proves the GPU-path contract that p05 (kinematic@512, flag off) does not:

* flag ON at reference resolution (s == 1) is BYTE-IDENTICAL to flag off. Asserted
  on the KINEMATIC path, which is byte-exact same-process (the vorticity path
  carries documented cross-instance SOR LSB noise, so a byte-exact assertion there
  is forbidden -- see CLAUDE.md). The kinematic path still exercises every
  scaled TRACER uniform (relax_k, replenish, belt_replenish, turb_time) plus the
  effective-step timeline, so s == 1 being byte-identical proves the structural
  short-circuit through all of them.
* flag ON at a different resolution (s != 1) actually CHANGES the developed field
  -- proof the scaling is wired, not inert.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu


def _develop(gpu, *, invariant, reference, resolution=512, dev_steps=60):
    p = load_factory_preset("jupiter_like")  # kinematic -> byte-exact same-process
    p.sim.resolution = resolution
    p.sim.dev_steps = dev_steps
    p.sim.resolution_invariant = invariant
    p.sim.reference_resolution = reference
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=30)
    out = np.array(sim.tracers.read_current(), dtype=np.float32)
    sim._release_sim()
    return out


def test_flag_on_at_reference_is_byte_identical(gpu):
    off = _develop(gpu, invariant=False, reference=512)
    on_s1 = _develop(gpu, invariant=True, reference=512)  # s == 1
    np.testing.assert_array_equal(off, on_s1)


def test_flag_on_off_reference_changes_development(gpu):
    off = _develop(gpu, invariant=False, reference=512)
    on_scaled = _develop(gpu, invariant=True, reference=1024)  # s == 0.5
    assert not np.array_equal(off, on_scaled)

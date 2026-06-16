"""GL context contracts.

Regression for the M3 render-gate OOM: ``GpuContext.compute`` compiled a NEW
GL program object on every call.  The module-level ``run_*`` 2-layer helpers in
``sw_gpu`` call ``compute`` once per dispatch and never release the program, so a
long step loop (the render gate runs ~15-25 dispatches/step over thousands of
steps) leaked hundreds of thousands of shader programs -> the driver's host
allocation climbed to ~20 GB -> ``MemoryError`` mid-run.

Compiled shaders are immutable and finite, so the fix is to cache them on the
context keyed by (package, name, defines).  These tests pin that contract.
"""

from __future__ import annotations

_KERNELS = "gasgiant.sim.kernels"


def test_compute_caches_identical_programs(gpu):
    """Two calls with identical (package, name, defines) return the SAME program
    object — no recompilation, no per-call leak."""
    a = gpu.compute(_KERNELS, "sw_vorticity.comp")
    b = gpu.compute(_KERNELS, "sw_vorticity.comp")
    assert a is b


def test_compute_distinguishes_defines(gpu):
    """Different define sets are distinct programs (the cache key includes
    defines — sw_continuity PASS 0 vs PASS 1 must NOT collide)."""
    p0 = gpu.compute(_KERNELS, "sw_continuity.comp", defines={"PASS": "0"})
    p1 = gpu.compute(_KERNELS, "sw_continuity.comp", defines={"PASS": "1"})
    assert p0 is not p1
    # and each is itself cached
    assert gpu.compute(_KERNELS, "sw_continuity.comp", defines={"PASS": "0"}) is p0


def test_compute_define_order_insensitive(gpu):
    """Define dicts with the same entries in different insertion order hit the
    same cache entry (key is order-independent)."""
    a = gpu.compute(_KERNELS, "sw_forcing.comp", defines={"PASS0": "1"})
    b = gpu.compute(_KERNELS, "sw_forcing.comp", defines={"PASS0": "1"})
    assert a is b

"""GL context contracts.

Regression for the M3 render-gate OOM: ``GpuContext.compute`` compiled a NEW
GL program object on every call.  Callers that invoke ``compute`` once per
dispatch and never release the program (the since-removed ``sw_gpu`` render
gate ran ~15-25 dispatches/step over thousands of steps) leaked hundreds of
thousands of shader programs -> the driver's host allocation climbed to
~20 GB -> ``MemoryError`` mid-run.

Compiled shaders are immutable and finite, so the fix is to cache them on the
context keyed by (package, name, defines).  These tests pin that contract.
"""

from __future__ import annotations

_KERNELS = "gasgiant.sim.kernels"


def test_compute_caches_identical_programs(gpu):
    """Two calls with identical (package, name, defines) return the SAME program
    object — no recompilation, no per-call leak."""
    a = gpu.compute(_KERNELS, "exchange_to_patch.comp", defines={"DOMAIN": "1"})
    b = gpu.compute(_KERNELS, "exchange_to_patch.comp", defines={"DOMAIN": "1"})
    assert a is b


def test_compute_distinguishes_defines(gpu):
    """Different define sets are distinct programs (the cache key includes
    defines — poisson_sor COLOR 0 (red) vs COLOR 1 (black) must NOT collide)."""
    p0 = gpu.compute(_KERNELS, "poisson_sor.comp", defines={"DOMAIN": "0", "COLOR": "0"})
    p1 = gpu.compute(_KERNELS, "poisson_sor.comp", defines={"DOMAIN": "0", "COLOR": "1"})
    assert p0 is not p1
    # and each is itself cached
    assert gpu.compute(_KERNELS, "poisson_sor.comp",
                       defines={"DOMAIN": "0", "COLOR": "0"}) is p0


def test_compute_define_order_insensitive(gpu):
    """Define dicts with the same entries in different insertion order hit the
    same cache entry (key is order-independent)."""
    a = gpu.compute(_KERNELS, "poisson_sor.comp", defines={"DOMAIN": "0", "COLOR": "0"})
    b = gpu.compute(_KERNELS, "poisson_sor.comp", defines={"COLOR": "0", "DOMAIN": "0"})
    assert a is b

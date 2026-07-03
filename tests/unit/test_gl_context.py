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


# -- A2-7: include flattener cycle/duplicate guard (CPU-only, no GL) --------------


def _fake_sources(monkeypatch, sources: dict[str, str]) -> None:
    from gasgiant.gl import context as glctx

    monkeypatch.setattr(glctx, "_read_source", lambda pkg, name: sources[name])


def test_flattener_terminates_on_circular_include(monkeypatch):
    """a includes b, b includes a: pre-guard this recursed forever
    (RecursionError); with the include-once seen-set it terminates and each
    body is emitted exactly once."""
    from gasgiant.gl.context import _load_flattened

    _fake_sources(monkeypatch, {
        "a.comp": '#version 430\n#include "b.glsl"\nvoid main() {}\n',
        "b.glsl": 'float b_fn() { return 1.0; }\n#include "a.comp"\n',
    })
    source, _ = _load_flattened("pkg", "a.comp", {})
    assert source.count("float b_fn()") == 1
    assert source.count("void main()") == 1


def test_flattener_emits_duplicate_include_once(monkeypatch):
    """Including the same file twice (directly or via a diamond) must emit its
    text once -- a second copy would redefine every symbol and fail compile."""
    from gasgiant.gl.context import _load_flattened

    _fake_sources(monkeypatch, {
        "main.comp": '#version 430\n#include "common.glsl"\n#include "common.glsl"\nvoid main() {}\n',
        "common.glsl": "float c_fn() { return 2.0; }\n",
    })
    source, _ = _load_flattened("pkg", "main.comp", {})
    assert source.count("float c_fn()") == 1


def test_flattener_source_map_stays_aligned_after_skipped_include(monkeypatch):
    """A skipped duplicate include shifts the flattened text by one line; the
    SourceMap must keep mapping later lines to their true origin (error
    messages point at the right kernel line)."""
    from gasgiant.gl.context import _load_flattened

    _fake_sources(monkeypatch, {
        "main.comp": '#version 430\n#include "common.glsl"\n#include "common.glsl"\nBAD LINE\n',
        "common.glsl": "float c_fn() { return 2.0; }\n",
    })
    source, smap = _load_flattened("pkg", "main.comp", {})
    lines = source.splitlines()
    bad_flat_line = lines.index("BAD LINE") + 1  # 1-based, as drivers report
    assert smap.resolve(bad_flat_line) == ("main.comp", 4)


def test_flattener_diamond_include_via_nested_files(monkeypatch):
    """Diamond: top includes x and y; both include common. common emitted once,
    and both unique bodies present."""
    from gasgiant.gl.context import _load_flattened

    _fake_sources(monkeypatch, {
        "top.comp": '#version 430\n#include "x.glsl"\n#include "y.glsl"\nvoid main() {}\n',
        "x.glsl": '#include "common.glsl"\nfloat x_fn() { return 1.0; }\n',
        "y.glsl": '#include "common.glsl"\nfloat y_fn() { return 1.0; }\n',
        "common.glsl": "float c_fn() { return 2.0; }\n",
    })
    source, _ = _load_flattened("pkg", "top.comp", {})
    assert source.count("float c_fn()") == 1
    assert source.count("float x_fn()") == 1
    assert source.count("float y_fn()") == 1

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def gpu():
    """Headless GL context for @pytest.mark.gpu tests; skips if unavailable."""
    from gasgiant.gl import GpuContext

    try:
        ctx = GpuContext.headless()
    except Exception as exc:  # noqa: BLE001 - any context failure means skip
        pytest.skip(f"no OpenGL 4.3 context available: {exc}")
    yield ctx
    ctx.release()


@pytest.fixture(autouse=True)
def _isolate_baroclinic_cache(tmp_path, monkeypatch):
    """Point the baroclinic warm-state cache at a per-test temp directory.

    Two reasons this is autouse rather than opt-in. Writing: the facade opts
    into the disk cache, so any test that builds a baroclinic-enabled
    Simulation would otherwise deposit ~768 KiB in the developer's real
    ``~/.gasgiant/baro_cache``. Reading: a hit would let a test skip the warmup
    it is there to exercise, so a regression in the warmup could pass on a
    machine that had run the suite before and fail on a clean checkout -- the
    worst kind of flake.

    Per-test (not per-session) so no test can be made to pass by an entry a
    different test wrote.
    """
    from gasgiant.sim import baroclinic_cache as bcache
    monkeypatch.setattr(bcache, "BARO_CACHE_DIR", tmp_path / "baro_cache")


@pytest.fixture(autouse=True)
def _gpu_context_current(request):
    """Re-make the session GL context current before each test that uses it.
    Tests that run the CLI in-process create their own context and leave it
    current; moderngl then routes the fixture context's calls to the wrong
    context, corrupting later tests in subtle ways."""
    if "gpu" in request.fixturenames:
        request.getfixturevalue("gpu").make_current()
    yield

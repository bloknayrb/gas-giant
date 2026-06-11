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
def _gpu_context_current(request):
    """Re-make the session GL context current before each test that uses it.
    Tests that run the CLI in-process create their own context and leave it
    current; moderngl then routes the fixture context's calls to the wrong
    context, corrupting later tests in subtle ways."""
    if "gpu" in request.fixturenames:
        request.getfixturevalue("gpu").make_current()
    yield

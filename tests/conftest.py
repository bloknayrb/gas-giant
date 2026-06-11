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

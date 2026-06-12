"""Cross-validate the numpy AgX port against the actual GLSL.

A unit-test polynomial/range suite cannot catch a transposed matrix that
happens to preserve gray behavior on some inputs; running the REAL shader
on chromatic colors and comparing against agx_view pins the port to the
source of truth."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.gl.context import _read_source
from gasgiant.palette.agx import agx_view

pytestmark = pytest.mark.gpu

_COLORS = np.array(
    [
        [0.55, 0.36, 0.24],   # rusty belt
        [0.42, 0.28, 0.18],   # dark belt
        [0.85, 0.80, 0.72],   # pale zone
        [0.80, 0.49, 0.34],   # GRS salmon (storm tint)
        [0.10, 0.34, 0.55],   # festoon blue
        [0.5, 0.5, 0.5],      # gray anchor
    ],
    dtype=np.float32,
)


def test_numpy_port_matches_glsl(gpu):
    agx_src = _read_source("gasgiant.app.shaders", "agx.glsl")
    source = (
        "#version 430\n"
        "layout(local_size_x = 1) in;\n"
        + agx_src
        + """
layout(std430, binding = 7) buffer Out { vec4 results[]; };
uniform vec3 u_colors[8];
uniform int u_count;
void main() {
    for (int i = 0; i < u_count; ++i) {
        results[i] = vec4(viewTransform(u_colors[i], 1), 1.0);
    }
}
"""
    )
    prog = gpu.ctx.compute_shader(source)
    buf = gpu.ctx.buffer(np.zeros((8, 4), dtype=np.float32).tobytes())
    buf.bind_to_storage_buffer(7)
    padded = np.zeros((8, 3), dtype=np.float32)
    padded[: len(_COLORS)] = _COLORS
    prog["u_colors"].write(padded.tobytes())
    prog["u_count"].value = len(_COLORS)
    prog.run(1, 1, 1)
    gpu.ctx.memory_barrier()
    glsl = np.frombuffer(buf.read(), dtype=np.float32).reshape(8, 4)[: len(_COLORS), :3]
    ours = agx_view(_COLORS)
    assert np.allclose(ours, np.clip(glsl, 0.0, 1.0), atol=2e-3), (
        np.abs(ours - np.clip(glsl, 0.0, 1.0)).max()
    )
    buf.release()
    prog.release()

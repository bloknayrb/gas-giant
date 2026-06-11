"""GL context acquisition and shader loading.

Headless (CLI, tests) and windowed (GUI attaches to the GLFW-created context)
modes produce the same GpuContext; everything above this layer is identical in
both. Kernels target #version 430 — nothing in the project needs more, and 430
is what Mesa llvmpipe in CI guarantees.
"""

from __future__ import annotations

import logging
import re
from importlib import resources

import moderngl
import numpy as np

from gasgiant.diagnostics import SourceMap, format_glsl_error

log = logging.getLogger(__name__)

GL_REQUIRE = 430

_INCLUDE_RE = re.compile(r'^\s*#include\s+"(?P<name>[^"]+)"\s*$')


class ShaderError(RuntimeError):
    pass


_FULLSCREEN_VS = """#version 430
out vec2 v_uv;
void main() {
    vec2 pos = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""


class FullscreenPass:
    """A fragment shader over a fullscreen triangle, rendered into an FBO."""

    def __init__(self, gpu: GpuContext, package: str, frag_name: str) -> None:
        source, smap = _load_flattened(package, frag_name, {})
        try:
            self.prog = gpu.ctx.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=source)
        except moderngl.Error as exc:
            raise ShaderError(
                f"compile failed for {package}/{frag_name}:\n{format_glsl_error(str(exc), smap)}"
            ) from exc
        self._vao = gpu.ctx.vertex_array(self.prog, [])

    def render(self, fbo: moderngl.Framebuffer) -> None:
        fbo.use()
        self._vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        self._vao.release()
        self.prog.release()


class GpuContext:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx

    @classmethod
    def headless(cls) -> GpuContext:
        ctx = moderngl.create_standalone_context(require=GL_REQUIRE)
        log.info(
            "headless GL context: %s | %s",
            ctx.info.get("GL_VENDOR", "?"),
            ctx.info.get("GL_RENDERER", "?"),
        )
        return cls(ctx)

    @classmethod
    def attach(cls) -> GpuContext:
        """Attach to the current (window-owned) GL context."""
        return cls(moderngl.get_context())

    # -- resources ---------------------------------------------------------

    def texture2d(
        self,
        size: tuple[int, int],
        components: int = 4,
        dtype: str = "f4",
        data: np.ndarray | None = None,
    ) -> moderngl.Texture:
        raw = data.tobytes() if data is not None else None
        tex = self.ctx.texture(size, components, data=raw, dtype=dtype)
        # Sim kernels read via texelFetch; default to NEAREST so accidental
        # hardware filtering (8-bit fractional weights) can't sneak in.
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = True
        tex.repeat_y = False
        return tex

    def framebuffer(self, color_tex: moderngl.Texture) -> moderngl.Framebuffer:
        return self.ctx.framebuffer(color_attachments=[color_tex])

    def fullscreen_pass(self, package: str, frag_name: str) -> FullscreenPass:
        return FullscreenPass(self, package, frag_name)

    def lut_texture(self, lut: np.ndarray) -> moderngl.Texture:
        """(N, 4) float32 LUT -> Nx1 linear-filtered, clamped texture."""
        data = lut.astype(np.float32).tobytes()
        tex = self.ctx.texture((lut.shape[0], 1), 4, data=data, dtype="f4")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def read_texture(self, tex: moderngl.Texture) -> np.ndarray:
        """Texture -> (H, W, C) float32 array (synchronous; fine for one-shot export)."""
        dtype = {"f4": np.float32, "f2": np.float16}[tex.dtype]
        raw = np.frombuffer(tex.read(), dtype=dtype)
        h, w = tex.height, tex.width
        return raw.reshape(h, w, tex.components).astype(np.float32, copy=False)

    # -- shaders -----------------------------------------------------------

    def compute(
        self, package: str, name: str, defines: dict[str, str] | None = None
    ) -> moderngl.ComputeShader:
        """Load a compute shader from package data, expanding #include lines."""
        source, smap = _load_flattened(package, name, defines or {})
        try:
            return self.ctx.compute_shader(source)
        except moderngl.Error as exc:
            raise ShaderError(
                f"compile failed for {package}/{name}:\n{format_glsl_error(str(exc), smap)}"
            ) from exc

    def release(self) -> None:
        self.ctx.release()


def _load_flattened(package: str, name: str, defines: dict[str, str]) -> tuple[str, SourceMap]:
    smap = SourceMap()
    out: list[str] = []

    def emit(lines: list[str], source_name: str) -> None:
        i = 0
        while i < len(lines):
            line = lines[i]
            m = _INCLUDE_RE.match(line)
            if m:
                inc_name = m.group("name")
                inc_text = _read_source(package, inc_name)
                emit(inc_text.splitlines(), inc_name)
                smap.add_span(len(out) + 1, source_name, i + 2)  # resume after include
                i += 1
                continue
            if line.startswith("#version") and source_name == name and defines:
                out.append(line)
                smap.add_span(len(out) + 1, name, i + 2)
                for key, value in defines.items():
                    out.append(f"#define {key} {value}")
                i += 1
                continue
            out.append(line)
            i += 1

    smap.add_span(1, name, 1)
    emit(_read_source(package, name).splitlines(), name)
    return "\n".join(out) + "\n", smap


def _read_source(package: str, name: str) -> str:
    ref = resources.files(package) / name
    if not ref.is_file():
        raise ShaderError(f"shader source not found: {package}/{name}")
    return ref.read_text(encoding="utf-8")

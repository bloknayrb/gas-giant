"""GPU layer. The ONLY package allowed to import moderngl.

Owns context acquisition (windowed-attach or headless), all GL object creation,
and the shader loader. Higher layers receive and use handle objects from here
but never create raw GL resources.
"""

from gasgiant.gl.context import GpuContext, ShaderError

__all__ = ["GpuContext", "ShaderError"]

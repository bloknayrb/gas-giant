"""Logging configuration, GLSL error formatting, and lightweight perf counters.

This module sits outside the layer stack (no gasgiant imports) so every layer
may use it.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_NVIDIA_ERROR_RE = re.compile(r"^(?P<file>\d+)\((?P<line>\d+)\)\s*:", re.MULTILINE)
_MESA_ERROR_RE = re.compile(r"^(?P<file>\d+):(?P<line>\d+)\(\d+\):", re.MULTILINE)


def configure_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Set up root logging once: console always, optional structured log file."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_handler)


@dataclass
class SourceMap:
    """Maps post-#include-expansion line numbers back to (source name, line).

    Built by the shader loader as it expands includes; consumed when a driver
    reports compile errors against the flattened source.
    """

    entries: list[tuple[int, str, int]] = field(default_factory=list)
    # Each entry: (flattened start line, source name, line offset in that source)

    def add_span(self, flattened_start: int, source_name: str, source_start: int) -> None:
        self.entries.append((flattened_start, source_name, source_start))

    def resolve(self, flattened_line: int) -> tuple[str, int]:
        best: tuple[int, str, int] | None = None
        for start, name, src_start in self.entries:
            if start <= flattened_line and (best is None or start >= best[0]):
                best = (start, name, src_start)
        if best is None:
            return ("<unknown>", flattened_line)
        start, name, src_start = best
        return (name, src_start + (flattened_line - start))


def format_glsl_error(driver_log: str, source_map: SourceMap | None = None) -> str:
    """Rewrite driver compile-error line numbers to original file:line references.

    Handles NVIDIA ("0(123) :") and Mesa ("0:123(45):") formats; unknown
    formats pass through untouched.
    """
    if source_map is None:
        return driver_log

    def _rewrite(match: re.Match[str]) -> str:
        line = int(match.group("line"))
        name, src_line = source_map.resolve(line)
        return f"{name}:{src_line}:"

    rewritten = _NVIDIA_ERROR_RE.sub(_rewrite, driver_log)
    rewritten = _MESA_ERROR_RE.sub(_rewrite, rewritten)
    return rewritten


class PerfCounter:
    """Rolling average of recent timings, for the GUI perf HUD and CLI reporting."""

    def __init__(self, window: int = 120) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self._start: float | None = None

    def begin(self) -> None:
        self._start = time.perf_counter()

    def end(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.perf_counter() - self._start
        self._samples.append(elapsed)
        self._start = None
        return elapsed

    @property
    def mean_ms(self) -> float:
        if not self._samples:
            return 0.0
        return 1000.0 * sum(self._samples) / len(self._samples)

    @property
    def last_ms(self) -> float:
        return 1000.0 * self._samples[-1] if self._samples else 0.0

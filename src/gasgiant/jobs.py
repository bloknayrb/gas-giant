"""The job protocol: how long-running work (export, regeneration) yields
control in slices so both the GUI scheduler and the headless CLI can drive it.

Defined below app/ in the layer order on purpose: export produces jobs and
cli consumes them without importing the GUI. Fleshed out in Phase 4; the
protocol shape is fixed now so consumers can be written against it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Progress:
    """Yielded by job generators after each slice."""

    done: int
    total: int
    message: str = ""

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0

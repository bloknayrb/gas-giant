"""Named sub-seed streams.

Every stochastic subsystem draws from its own named stream derived from the
master seed, so changing e.g. storm parameters never reshuffles the bands.
"""

from __future__ import annotations

import zlib

import numpy as np


def subseed(seed: int, name: str) -> np.random.Generator:
    """Deterministic, independent generator for (master seed, subsystem name)."""
    tag = zlib.crc32(name.encode("utf-8"))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, tag])))

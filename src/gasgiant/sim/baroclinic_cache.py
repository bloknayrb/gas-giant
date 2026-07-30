"""On-disk cache for the baroclinic warm state.

Warming the 2-layer solver is the expensive half of enabling baroclinic
coupling: a measured 52 s to build a driver at the default `warmup_steps` 8000
on the fixed 192x96 source grid (~6.5 ms/step; ~130 s at the 20000 ceiling),
synchronously on the caller's thread. The result is a deterministic function of
a handful of inputs, so it only has to be computed once per configuration --
ever, not per session.

Measured end to end: 52.1 s cold, 0.017 s warm, 3083x. Entry size 768 KiB.

Mirrors the ``app/thumbnails.py`` cache idiom: a ``~/.gasgiant/<name>`` default
directory, a pure key function, and an injectable ``cache_dir`` so tests never
touch the user's real cache.

**Correctness rests entirely on the key.** An under-invalidating key silently
serves a state warmed under different physics, which is far worse than the
re-warmup it saves, so the key is deliberately over-broad in two ways:

1. It fingerprints the full source text of every module that shapes the warm
   state -- including ``baroclinic_driver`` itself, which owns the literals
   (``pert_amp_frac``, ``dt_safety``, ``nu4``) passed into the state builder.
   Editing a docstring in one of those files therefore invalidates the whole
   cache. That is the intended trade: a spurious re-warmup while you are
   actively editing the solver is exactly when you want one.
2. It carries the RUNTIME values of the ``baroclinic_source`` constants the
   driver reads, not just their on-disk text. Tests monkeypatch ``bsrc.XI`` and
   ``bsrc.GP2`` to force an outcrop; those never change the file, so a
   text-only fingerprint would hand a patched run a state warmed at the real
   constants.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BARO_CACHE_DIR = Path.home() / ".gasgiant" / "baro_cache"

#: Entries are ~740 KiB compressed, and the key space is continuous (an artist
#: sweeping a slider lands on a new one each time they stop), so unlike the
#: thumbnail cache -- keyed per preset, a bounded set -- this one needs a bound.
#: 64 MiB is ~88 configurations; well past any single tuning session.
BARO_CACHE_MAX_BYTES = 64 * 1024 * 1024

#: The complete evolving state. `step_2layer` writes exactly these six, and
#: `apply_forcing` writes the same six (it READS h_eq1/h_eq2 and never assigns
#: them). Everything else on Sw2State -- the grid, the reduced gravities, dt,
#: the floors, the forcing targets -- is rebuilt from the constructor inputs, so
#: a restored state can never disagree with current code about them.
#: `test_cached_warm_state_resumes_bit_identically` proves the set is complete
#: by advancing a restored driver against a freshly warmed one.
EVOLVING_FIELDS = ("h1", "u1", "v1", "h2", "u2", "v2")

_fingerprint: str | None = None


def solver_fingerprint() -> str:
    """sha256 over the source text of every module that shapes the warm state.

    Computed once per process. Read from disk rather than from a version
    constant so it cannot go stale: there is no step where someone edits the
    solver and forgets to bump something.
    """
    global _fingerprint
    if _fingerprint is None:
        here = Path(__file__).parent
        h = hashlib.sha256()
        for name in ("shallow_water_ref.py", "baroclinic_source.py",
                     "baroclinic_driver.py"):
            h.update((here / name).read_bytes())
        _fingerprint = h.hexdigest()
    return _fingerprint


def warm_cache_key(**inputs) -> str:
    """Stable content hash of the warm-state inputs plus the solver fingerprint.

    Pure: no filesystem writes, no GL. Callers pass every constructor input the
    warmup depends on; passing a derivation-time input (the output grid, the
    smoothing sigma) would be a bug -- see ``BaroclinicSourceDriver``.
    """
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"),
                           default=repr)
    return hashlib.sha256(
        f"{solver_fingerprint()}|{canonical}".encode()).hexdigest()


def warm_cache_path(key: str, cache_dir: Path = BARO_CACHE_DIR) -> Path:
    """On-disk path for a warm state (``<cache_dir>/<key>.npz``)."""
    return cache_dir / f"{key}.npz"


def load_warm_state(key: str, cache_dir: Path = BARO_CACHE_DIR
                    ) -> dict[str, np.ndarray] | None:
    """The cached evolving arrays, or None on any miss.

    A damaged entry (truncated write, half-copied directory, a file from a
    different numpy) must never be fatal -- it degrades to a re-warmup, and the
    bad file is removed so it cannot cost the same detour twice.
    """
    path = warm_cache_path(key, cache_dir)
    if not path.is_file():
        return None
    try:
        with np.load(path) as npz:
            state = {f: npz[f] for f in EVOLVING_FIELDS}
    except Exception as exc:  # noqa: BLE001 -- any unreadable file degrades alike
        log.warning("baroclinic warm-state cache: discarding unreadable %s (%s)",
                    path.name, exc)
        path.unlink(missing_ok=True)
        return None
    log.info("baroclinic warm-state cache: hit (%s)", path.name)
    return state


def save_warm_state(key: str, state: dict[str, np.ndarray],
                    cache_dir: Path = BARO_CACHE_DIR) -> None:
    """Write a warm state, then prune the directory back under its cap.

    Written to a temp file and renamed, so an interrupted write leaves the old
    entry (or nothing) rather than a truncated file that later reads as a
    corrupt hit. Never raises: a cache that cannot be written is a performance
    problem, not a correctness one, and must not take down a working sim.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = warm_cache_path(key, cache_dir)
        tmp = path.with_suffix(f".npz.{id(state):x}.tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **state)
        tmp.replace(path)
        _prune(cache_dir)
    except OSError as exc:
        log.warning("baroclinic warm-state cache: write failed (%s)", exc)


def _prune(cache_dir: Path, max_bytes: int | None = None) -> None:
    """Drop the oldest entries until the directory fits under ``max_bytes``.

    Oldest-first by mtime, and the NEWEST entry is never evicted, which is what
    protects the entry a caller just wrote. Without that floor a cap smaller
    than one entry makes the cache thrash -- warm, write, evict, re-warm --
    paying the full ~52 s on every single call and never serving a hit. One
    oversized entry is a far better failure than that.

    None sentinel rather than ``= BARO_CACHE_MAX_BYTES``: a default argument
    binds ONCE at def time, freezing the module constant so a later override
    would silently do nothing. Same trap the driver's sentinel defaults carry.
    """
    max_bytes = BARO_CACHE_MAX_BYTES if max_bytes is None else max_bytes
    entries = sorted(cache_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in entries)
    while total > max_bytes and len(entries) > 1:
        victim = entries.pop(0)
        total -= victim.stat().st_size
        victim.unlink(missing_ok=True)
        log.info("baroclinic warm-state cache: evicted %s", victim.name)

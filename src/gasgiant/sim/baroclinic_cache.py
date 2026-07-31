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

1. It fingerprints the full source text of the modules listed in
   ``_FINGERPRINTED`` -- including ``baroclinic_driver`` itself, which owns the
   literals (``pert_amp_frac``, ``dt_safety``, ``nu4``) passed into the state
   builder, and ``params/seeds.py``, whose ``subseed`` draws the
   phase_jitter/spectrum_width seeding realization. Editing a docstring in any
   of them invalidates the whole cache. That is the intended trade: a spurious
   re-warmup while you are actively editing the solver is exactly when you want
   one. The list is CURATED, not derived from the import graph -- see
   ``solver_fingerprint``.
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
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BARO_CACHE_DIR = Path.home() / ".gasgiant" / "baro_cache"

#: A temp file younger than this may still be an in-flight write by a CONCURRENT
#: process (a GUI session and a CLI export share the cache), whose `replace`
#: would then fail with FileNotFoundError. Only temps old enough to be certainly
#: abandoned get swept.
_TMP_SWEEP_AGE_S = 3600.0

#: Entries are 768 KiB compressed, and the key space is continuous (an artist
#: sweeping a slider lands on a new one each time they stop), so unlike the
#: thumbnail cache -- keyed per preset, a bounded set -- this one needs a bound.
#: 64 MiB is ~85 configurations; well past any single tuning session.
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


#: Every first-party module whose source text can change the warm state, relative
#: to this file. `params/seeds.py` is here because `shallow_water_ref._seed_pattern`
#: draws the seeding realization through `subseed(...)` whenever `phase_jitter` or
#: `spectrum_width` is on, and that realization lands in h2/u2/v2 at t=0 -- three
#: of the six EVOLVING_FIELDS. This module is here because it owns
#: EVOLVING_FIELDS, i.e. which arrays a restore puts back at all.
_FINGERPRINTED = (
    "shallow_water_ref.py",
    "baroclinic_source.py",
    "baroclinic_driver.py",
    "baroclinic_cache.py",
    "../params/seeds.py",
)


def solver_fingerprint() -> str | None:
    """sha256 over the source text of the modules listed in ``_FINGERPRINTED``.

    Computed once per process. Read from disk rather than from a version
    constant so it cannot go stale: there is no step where someone edits the
    solver and forgets to bump something.

    Note the list is a curated claim, not a derived one -- nothing walks the
    import graph, so a NEW first-party dependency of the warm state has to be
    added here by hand. Over-inclusion is free (a spurious re-warmup);
    under-inclusion is a false hit, so err toward adding.

    Returns None if the sources cannot be read -- the SAME Windows lock this
    module guards against for the cache file, but landing on a .py; or a frozen
    install with no .py on disk. This is the FIRST thing a cache lookup does, so
    an unguarded read would escape `_init_baroclinic`'s two-arm except and crash
    Simulation construction before any of the careful degrade paths below could
    run. A None fingerprint disables the cache: without trustworthy source
    identity a hit could be a state warmed under different physics.

    The memo only records a SUCCESS, so a transient lock costs one uncached
    construction rather than poisoning the cache for the whole process.
    """
    global _fingerprint
    if _fingerprint is None:
        here = Path(__file__).parent
        h = hashlib.sha256()
        try:
            for name in _FINGERPRINTED:
                h.update((here / name).read_bytes())
        except OSError as exc:
            log.warning("baroclinic warm-state cache: cannot fingerprint the "
                        "solver sources, disabling the cache (%s)", exc)
            return None
        _fingerprint = h.hexdigest()
    return _fingerprint


def warm_cache_key(**inputs) -> str | None:
    """Stable content hash of the warm-state inputs plus the solver fingerprint.

    None -- meaning "do not use the cache" -- when the fingerprint is
    unavailable; see ``solver_fingerprint``.

    Pure: no filesystem writes, no GL. Callers pass every constructor input the
    warmup depends on; passing a derivation-time input (the output grid, the
    smoothing sigma) would be a bug -- see ``BaroclinicSourceDriver``.
    """
    fingerprint = solver_fingerprint()
    if fingerprint is None:
        return None
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"),
                           default=repr)
    return hashlib.sha256(f"{fingerprint}|{canonical}".encode()).hexdigest()


def warm_cache_path(key: str, cache_dir: Path | None = None) -> Path:
    """On-disk path for a warm state (``<cache_dir>/<key>.npz``).

    None sentinel rather than ``= BARO_CACHE_DIR``, here and on the two
    functions below, for the reason spelled out on ``_prune``: a default
    argument binds ONCE at import, so the module constant would freeze. That is
    not academic for this particular constant -- the autouse ``conftest``
    fixture redirects the cache by monkeypatching exactly this attribute, so a
    frozen default would silently read and write the developer's real
    ``~/.gasgiant/baro_cache`` from inside the test suite.
    """
    cache_dir = BARO_CACHE_DIR if cache_dir is None else cache_dir
    return cache_dir / f"{key}.npz"


def load_warm_state(key: str, cache_dir: Path | None = None
                    ) -> dict[str, np.ndarray] | None:
    """The cached evolving arrays, or None on any miss.

    A damaged entry (truncated write, half-copied directory, a file from a
    different numpy) must never be fatal -- it degrades to a re-warmup, and the
    bad file is removed so it cannot cost the same detour twice.

    The REMOVAL is itself best-effort, and that is not paranoia. On Windows a
    perfectly valid entry that happens to be exclusively locked at that instant
    (an AV scan or a backup/sync agent touching ~/.gasgiant) makes np.load fail
    AND makes the unlink fail with WinError 32. An unguarded unlink here escapes
    every caller: the driver does not wrap the load, `_init_baroclinic` catches
    only ImportError and BaroclinicWarmupError, so a PermissionError would exit
    Simulation.__init__ and, from the GUI, throw through the imgui frame
    callback on a RESTART-tier edit. A locked cache file must cost a re-warmup,
    never the application.

    The except stays broad ON PURPOSE, and so does unlinking on any of it --
    reviewed and kept twice, so weigh it before narrowing. Narrowing to the
    errors that clearly mean "bad file" (BadZipFile, ValueError, KeyError,
    EOFError) would protect a valid entry from being destroyed by a transient
    failure whose unlink happens to succeed, at a cost of 52 s once. But an
    UNLISTED corruption signature then leaves a permanently unreadable file that
    costs a warning and a re-warmup on every construction, forever, with nothing
    that ever clears it. Bounded one-off loss beats unbounded repeat loss.
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
        try:
            path.unlink(missing_ok=True)
        except OSError as rm_exc:
            log.warning("baroclinic warm-state cache: could not remove %s (%s)",
                        path.name, rm_exc)
        return None
    log.info("baroclinic warm-state cache: hit (%s)", path.name)
    return state


def save_warm_state(key: str, state: dict[str, np.ndarray],
                    cache_dir: Path | None = None) -> None:
    """Write a warm state, then prune the directory back under its cap.

    Written to a temp file and renamed, so an interrupted write leaves the old
    entry (or nothing) rather than a truncated file that later reads as a
    corrupt hit. Never raises: a cache that cannot be written is a performance
    problem, not a correctness one, and must not take down a working sim.

    The except is deliberately broad, matching `load_warm_state`. OSError alone
    does not deliver on "never raises": np.savez_compressed goes through the zip
    layer, which can raise LargeZipFile or ValueError, and a compress under
    memory pressure can raise MemoryError. Any of those would exit
    Simulation.__init__ past `_init_baroclinic`'s narrow except -- the exact
    failure this promise exists to prevent.

    The `finally` matters: a write that dies partway (ENOSPC, quota, a failing
    replace) would otherwise strand its temp file forever. `_prune` globs
    ``*.npz`` and cannot match a ``.tmp`` suffix, so a stranded temp counts
    toward neither the total nor eviction -- the cap could never reclaim it. A
    successful `replace` has already consumed the temp, so the unlink is a
    no-op on the happy path.
    """
    cache_dir = BARO_CACHE_DIR if cache_dir is None else cache_dir
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = warm_cache_path(key, cache_dir)
        tmp = path.with_suffix(f".npz.{id(state):x}.tmp")
        try:
            with tmp.open("wb") as fh:
                np.savez_compressed(fh, **state)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        _prune(cache_dir)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
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
    # Sweep temps stranded by a crash (a kill between open and replace outruns
    # save_warm_state's `finally`). Harmless to read past -- nothing ever opens
    # them -- but invisible to the cap below, which globs *.npz.
    #
    # Guarded per file, and NOT because a stranded temp is precious. An
    # undeletable one (the same AV/sync lock as elsewhere) would otherwise abort
    # _prune before the eviction loop, so the 64 MiB cap would silently stop
    # being enforced for as long as that file existed -- and from
    # save_warm_state the abort is swallowed and logged as "write failed" even
    # though the write succeeded.
    cutoff = time.time() - _TMP_SWEEP_AGE_S
    for stale in cache_dir.glob("*.tmp"):
        try:
            if stale.stat().st_mtime <= cutoff:
                stale.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("baroclinic warm-state cache: could not sweep %s (%s)",
                        stale.name, exc)
    entries = sorted(cache_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in entries)
    while total > max_bytes and len(entries) > 1:
        victim = entries.pop(0)
        total -= victim.stat().st_size
        victim.unlink(missing_ok=True)
        log.info("baroclinic warm-state cache: evicted %s", victim.name)

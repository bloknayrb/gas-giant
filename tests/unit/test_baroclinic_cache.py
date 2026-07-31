"""On-disk warm-state cache: key soundness, round-trip, and failure tolerance.

Split by cost. Everything here is pure key/file work and runs in the fast loop;
the tests that actually warm a solver (a real ~700-step advance) live in
test_baroclinic_driver.py behind the `slow` marker.

The stakes are lopsided. A cache MISS costs a re-warmup -- slow, never wrong. A
false HIT serves a state warmed under different physics and every render
downstream is quietly wrong. So the key tests below are all miss-direction: each
asserts that some change DOES produce a different key.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.sim import baroclinic_cache as bcache


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "baro_cache"


def _state(seed: float = 1.0) -> dict[str, np.ndarray]:
    return {f: np.full((4, 8), seed + i, dtype=np.float64)
            for i, f in enumerate(bcache.EVOLVING_FIELDS)}


# -- the key ------------------------------------------------------------------


def test_same_inputs_give_the_same_key():
    assert bcache.warm_cache_key(a=1, b=2.5) == bcache.warm_cache_key(a=1, b=2.5)


def test_key_is_order_independent():
    """Keys are built from **kwargs, whose iteration order follows the call
    site. Sorting is what stops a harmless argument reshuffle in the driver from
    orphaning every cached entry."""
    assert bcache.warm_cache_key(a=1, b=2) == bcache.warm_cache_key(b=2, a=1)


@pytest.mark.parametrize("changed", [
    {"latitude": 46.0}, {"seed": 1}, {"gp2": 0.08}, {"m_zonal": 15},
    {"warmup_steps": 8001}, {"phase_jitter": 0.5}, {"spectrum_width": 1},
    {"width": 24.0}, {"gp1": 0.06}, {"xi": 3.1},
])
def test_every_input_changes_the_key(changed):
    base = dict(latitude=45.0, seed=0, gp2=0.075, m_zonal=14, warmup_steps=8000,
                phase_jitter=0.0, spectrum_width=0, width=25.0, gp1=0.05, xi=3.0)
    assert bcache.warm_cache_key(**base) != bcache.warm_cache_key(**{**base, **changed})


def test_key_distinguishes_int_and_float():
    """`spectrum_width` is an int and `phase_jitter` a float; a key built by
    stringifying without type information would collide 0 with 0.0 and serve a
    jittered state to an unjittered request."""
    assert bcache.warm_cache_key(x=0) != bcache.warm_cache_key(x=0.0)


def test_fingerprint_hashes_exactly_the_listed_files():
    """A PIN on the file list, not a proof of coverage -- nothing here walks the
    import graph, so this cannot tell you the list is complete. It tells you the
    list has not silently shrunk. Named accordingly: the earlier name claimed a
    property it could not check, and the list was in fact missing
    ``params/seeds.py`` at the time.

    Each entry earns its place:
      - shallow_water_ref: the solver itself
      - baroclinic_source: the grid size and the reduced-gravity constants
      - baroclinic_driver: the literals passed into the state builder
        (pert_amp_frac, dt_safety, nu4), which no caller supplies
      - baroclinic_cache: owns EVOLVING_FIELDS, i.e. what a restore puts back
      - params/seeds: subseed() draws the phase_jitter/spectrum_width seeding
        realization, which lands in h2/u2/v2 at t=0
    """
    import hashlib
    from pathlib import Path
    assert bcache._FINGERPRINTED == (
        "shallow_water_ref.py", "baroclinic_source.py", "baroclinic_driver.py",
        "baroclinic_cache.py", "../params/seeds.py")
    here = Path(bcache.__file__).parent
    h = hashlib.sha256()
    for name in bcache._FINGERPRINTED:
        h.update((here / name).read_bytes())
    assert bcache.solver_fingerprint() == h.hexdigest()


def test_every_fingerprinted_file_exists():
    """A typo'd path would raise FileNotFoundError from every cache lookup --
    and the first place that surfaces is Simulation construction."""
    from pathlib import Path
    here = Path(bcache.__file__).parent
    for name in bcache._FINGERPRINTED:
        assert (here / name).is_file(), name


def test_seeds_module_is_fingerprinted():
    """Pinned separately because it is the non-obvious one and the easiest to
    drop in a cleanup: it lives in a DIFFERENT layer (params) and is reached
    only indirectly, via shallow_water_ref._seed_pattern, and only when
    phase_jitter or spectrum_width is non-default. Its output lands in three of
    the six EVOLVING_FIELDS at t=0."""
    from pathlib import Path
    resolved = {(Path(bcache.__file__).parent / n).resolve()
                for n in bcache._FINGERPRINTED}
    from gasgiant.params import seeds
    assert Path(seeds.__file__).resolve() in resolved


def test_fingerprint_participates_in_the_key(monkeypatch):
    monkeypatch.setattr(bcache, "_fingerprint", "0" * 64)
    a = bcache.warm_cache_key(x=1)
    monkeypatch.setattr(bcache, "_fingerprint", "1" * 64)
    assert bcache.warm_cache_key(x=1) != a


# -- round-trip ---------------------------------------------------------------


def test_miss_on_empty_cache(cache_dir):
    assert bcache.load_warm_state("deadbeef", cache_dir) is None


def test_round_trip_is_bit_exact(cache_dir):
    """float64 in, the SAME float64 out. A lossy round-trip would put the
    resumed run on a different trajectory from the one that was warmed."""
    saved = _state()
    bcache.save_warm_state("k", saved, cache_dir)
    loaded = bcache.load_warm_state("k", cache_dir)
    assert loaded is not None
    for f in bcache.EVOLVING_FIELDS:
        assert np.array_equal(loaded[f], saved[f]), f
        assert loaded[f].dtype == saved[f].dtype, f


def test_save_creates_the_directory(cache_dir):
    assert not cache_dir.exists()
    bcache.save_warm_state("k", _state(), cache_dir)
    assert bcache.warm_cache_path("k", cache_dir).is_file()


def test_save_leaves_no_temp_files(cache_dir):
    bcache.save_warm_state("k", _state(), cache_dir)
    assert [p.name for p in cache_dir.iterdir()] == ["k.npz"]


# -- failure tolerance --------------------------------------------------------


def test_corrupt_entry_degrades_to_a_miss_and_is_removed(cache_dir):
    """A truncated write or a half-copied directory must cost a re-warmup, not
    an exception -- and must not cost it twice."""
    bcache.save_warm_state("k", _state(), cache_dir)
    path = bcache.warm_cache_path("k", cache_dir)
    path.write_bytes(b"not an npz")
    assert bcache.load_warm_state("k", cache_dir) is None
    assert not path.exists(), "a corrupt entry must be removed, not re-read"


def test_entry_missing_a_field_degrades_to_a_miss(cache_dir):
    """An entry written by an older build with a smaller EVOLVING_FIELDS would
    load but leave part of the state at its initial value -- a silently wrong
    resume. The KeyError must surface as a miss instead."""
    cache_dir.mkdir(parents=True)
    partial = _state()
    del partial["v2"]
    np.savez_compressed(bcache.warm_cache_path("k", cache_dir), **partial)
    assert bcache.load_warm_state("k", cache_dir) is None


def test_an_undeletable_corrupt_entry_is_not_fatal(cache_dir, monkeypatch):
    """The removal in the recovery path must itself be best-effort.

    Reachable without any corruption at all: on Windows a VALID entry that is
    exclusively locked at that instant (AV scan, backup/sync agent touching
    ~/.gasgiant) fails np.load AND fails the unlink with WinError 32. An
    unguarded unlink escapes every caller -- the driver does not wrap the load,
    _init_baroclinic catches only ImportError and BaroclinicWarmupError -- so it
    would exit Simulation.__init__ and, from the GUI, throw through the imgui
    frame callback on a RESTART-tier edit. A locked cache file must cost a
    re-warmup, never the application.
    """
    bcache.save_warm_state("k", _state(), cache_dir)
    bcache.warm_cache_path("k", cache_dir).write_bytes(b"not an npz")

    def locked(*a, **k):
        raise PermissionError(32, "being used by another process")

    monkeypatch.setattr(bcache.Path, "unlink", locked)
    assert bcache.load_warm_state("k", cache_dir) is None  # must NOT raise


def test_a_failed_write_strands_no_temp_file(cache_dir, monkeypatch):
    """_prune globs *.npz and cannot match a .tmp suffix, so a stranded temp
    counts toward neither the total nor eviction -- the 64 MiB cap could never
    reclaim it, and nothing else ever opens or removes it."""
    def die(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(bcache.np, "savez_compressed", die)
    bcache.save_warm_state("k", _state(), cache_dir)  # must NOT raise
    assert list(cache_dir.iterdir()) == [], "a failed write must leave nothing behind"


def test_prune_sweeps_temps_stranded_by_a_crash(cache_dir):
    """save_warm_state's `finally` cannot run if the process is killed between
    open and replace, so the sweep is the second line of defence."""
    import os
    cache_dir.mkdir(parents=True)
    tmp = cache_dir / "k.npz.deadbeef.tmp"
    tmp.write_bytes(b"x" * 100)
    os.utime(tmp, (1000, 1000))  # older than the sweep age
    bcache._prune(cache_dir)
    assert not list(cache_dir.glob("*.tmp"))


def test_prune_leaves_a_FRESH_temp_alone(cache_dir):
    """A temp younger than the sweep age may be a CONCURRENT process's in-flight
    write (a GUI session and a CLI export share ~/.gasgiant/baro_cache). Sweeping
    it turns that process's `replace` into a FileNotFoundError, so its warmup is
    thrown away -- 52 s lost to a cache that was supposed to save it."""
    cache_dir.mkdir(parents=True)
    tmp = cache_dir / "k.npz.deadbeef.tmp"
    tmp.write_bytes(b"x" * 100)
    bcache._prune(cache_dir)
    assert tmp.exists()


def test_an_undeletable_temp_does_not_stop_eviction(cache_dir, monkeypatch):
    """An unguarded sweep aborts _prune BEFORE the eviction loop, so one locked
    temp would silently switch the 64 MiB cap off for as long as it existed --
    and from save_warm_state the abort is swallowed as a 'write failed' warning
    even though the write succeeded."""
    import os
    cache_dir.mkdir(parents=True)
    tmp = cache_dir / "stuck.npz.beef.tmp"
    tmp.write_bytes(b"x" * 100)
    os.utime(tmp, (1000, 1000))
    for i, name in enumerate(["old", "new"]):
        p = cache_dir / f"{name}.npz"
        p.write_bytes(b"x" * 1000)
        os.utime(p, (2000 + i, 2000 + i))

    real_unlink = bcache.Path.unlink

    def locked(self, *a, **k):
        if self.name.endswith(".tmp"):
            raise PermissionError(32, "being used by another process")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(bcache.Path, "unlink", locked)
    bcache._prune(cache_dir, max_bytes=1500)  # must NOT raise
    assert [p.stem for p in cache_dir.glob("*.npz")] == ["new"], "cap must still bite"


def test_a_non_OSError_write_failure_is_not_fatal(cache_dir, monkeypatch):
    """'Never raises' has to mean it. np.savez_compressed reaches the zip layer
    (LargeZipFile, ValueError) and compresses under whatever memory is left
    (MemoryError); none of those are OSError, and any of them would exit
    Simulation.__init__ past _init_baroclinic's two-arm except."""
    def die(*a, **k):
        raise MemoryError("unable to allocate")

    monkeypatch.setattr(bcache.np, "savez_compressed", die)
    bcache.save_warm_state("k", _state(), cache_dir)  # must NOT raise
    assert bcache.load_warm_state("k", cache_dir) is None


def test_unreadable_solver_sources_disable_the_cache(cache_dir, monkeypatch):
    """The fingerprint is the FIRST thing a lookup does, ahead of every degrade
    path in this module, and it is reached from the driver's constructor. An
    unguarded read_bytes -- the same AV lock, or a frozen install with no .py on
    disk -- would crash Simulation construction. A cache may only ever cost a
    re-warmup."""
    monkeypatch.setattr(bcache, "_fingerprint", None)

    def locked(self, *a, **k):
        raise PermissionError(32, "being used by another process")

    monkeypatch.setattr(bcache.Path, "read_bytes", locked)
    assert bcache.solver_fingerprint() is None
    assert bcache.warm_cache_key(x=1) is None, "no source identity => no cache"


def test_a_failed_fingerprint_is_not_memoized(cache_dir, monkeypatch):
    """A transient lock must cost ONE uncached construction, not poison the
    cache for the rest of the process. Only a success is memoized."""
    monkeypatch.setattr(bcache, "_fingerprint", None)
    calls = {"n": 0}
    real = bcache.Path.read_bytes

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(32, "being used by another process")
        return real(self, *a, **k)

    monkeypatch.setattr(bcache.Path, "read_bytes", flaky)
    assert bcache.solver_fingerprint() is None
    assert bcache.solver_fingerprint() is not None, "the retry must not be blocked"


def test_the_default_cache_dir_is_read_at_CALL_time(tmp_path, monkeypatch):
    """As `cache_dir: Path = BARO_CACHE_DIR` the default binds at import, and the
    autouse conftest fixture -- which redirects the cache by monkeypatching this
    exact attribute -- would silently stop working: any caller that omitted
    cache_dir would read and write the developer's real ~/.gasgiant/baro_cache
    from inside the test suite."""
    monkeypatch.setattr(bcache, "BARO_CACHE_DIR", tmp_path / "elsewhere")
    assert bcache.warm_cache_path("k").parent == tmp_path / "elsewhere"
    bcache.save_warm_state("k", _state())
    assert (tmp_path / "elsewhere" / "k.npz").is_file()
    assert bcache.load_warm_state("k") is not None


def test_unwritable_cache_is_not_fatal(cache_dir, monkeypatch):
    """A read-only or full disk must not take down a working sim."""
    def boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(bcache.Path, "mkdir", boom)
    bcache.save_warm_state("k", _state(), cache_dir)  # must NOT raise
    assert bcache.load_warm_state("k", cache_dir) is None


# -- eviction -----------------------------------------------------------------


def test_prune_evicts_oldest_first_until_under_cap(cache_dir):
    cache_dir.mkdir(parents=True)
    for i, name in enumerate(["old", "mid", "new"]):
        p = cache_dir / f"{name}.npz"
        p.write_bytes(b"x" * 1000)
        import os
        os.utime(p, (1000 + i, 1000 + i))
    bcache._prune(cache_dir, max_bytes=2500)
    assert sorted(p.stem for p in cache_dir.glob("*.npz")) == ["mid", "new"]


def test_a_fresh_save_never_evicts_itself(cache_dir, monkeypatch):
    """The new entry is the newest by mtime, so oldest-first eviction cannot
    reach it -- otherwise a cap smaller than one entry would warm, write, delete,
    and re-warm forever, paying the full ~52 s every single time.

    Also pins that the cap is read at CALL time: as a default argument it would
    bind at import and this override would silently do nothing.
    """
    import os
    bcache.save_warm_state("first", _state(), cache_dir)
    os.utime(bcache.warm_cache_path("first", cache_dir), (1000, 1000))
    monkeypatch.setattr(bcache, "BARO_CACHE_MAX_BYTES", 1)
    bcache.save_warm_state("second", _state(2.0), cache_dir)
    assert bcache.load_warm_state("second", cache_dir) is not None
    assert bcache.load_warm_state("first", cache_dir) is None, "cap must still bite"

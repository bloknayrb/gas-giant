"""T6 preset thumbnails: the pure cache-key + cache hit/miss helpers (no GL)."""

from __future__ import annotations

from gasgiant.app.thumbnails import (
    is_cached,
    thumb_cache_key,
    thumb_cache_path,
)
from gasgiant.params.model import PlanetParams


def _params(seed=42):
    p = PlanetParams(seed=seed)
    p.sim.dev_steps = 50
    return p


def test_key_stable_for_equal_params():
    a = _params()
    b = a.model_copy(deep=True)
    assert thumb_cache_key(a) == thumb_cache_key(b)


def test_key_stable_across_reconstruction():
    # A round-trip through JSON must not change the key (cross-session stability).
    a = _params()
    b = PlanetParams.from_json(a.to_json())
    assert thumb_cache_key(a) == thumb_cache_key(b)


def test_key_changes_on_top_level_leaf():
    a = _params(seed=1)
    b = _params(seed=2)
    assert thumb_cache_key(a) != thumb_cache_key(b)


def test_key_changes_on_nested_leaf():
    a = _params()
    b = a.model_copy(deep=True)
    b.sim.dev_steps = a.sim.dev_steps + 1
    assert thumb_cache_key(a) != thumb_cache_key(b)


def test_key_is_hex_sha256():
    key = thumb_cache_key(_params())
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_cache_path_uses_key(tmp_path):
    p = _params()
    path = thumb_cache_path(p, cache_dir=tmp_path)
    assert path == tmp_path / f"{thumb_cache_key(p)}.png"


def test_is_cached_miss_then_hit(tmp_path):
    p = _params()
    assert not is_cached(p, cache_dir=tmp_path)
    thumb_cache_path(p, cache_dir=tmp_path).write_bytes(b"fake-png")
    assert is_cached(p, cache_dir=tmp_path)


def test_is_cached_distinguishes_presets(tmp_path):
    a, b = _params(seed=1), _params(seed=2)
    thumb_cache_path(a, cache_dir=tmp_path).write_bytes(b"x")
    assert is_cached(a, cache_dir=tmp_path)
    assert not is_cached(b, cache_dir=tmp_path)

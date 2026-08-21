"""Image-writer contracts (no GL).

The exporter assembles whole-map buffers on the host, so which channel order it
STORES decides whether cv2 has to materialise a contiguous duplicate (measured
at 1.00x the buffer -- 3.00 GiB on a 32768-wide color map). These pin that the
two entry points agree byte-for-byte, and pin the gray writer's clipping, which
becomes load-bearing once the exporter's redundant outer np.clip is dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.export.writers import (
    read_png16,
    write_png16_bgr_u16,
    write_png16_gray,
    write_png16_gray_u16,
    write_png16_rgb_u16,
)


def _rgb_u16(h: int = 7, w: int = 11) -> np.ndarray:
    """Channel-asymmetric content -- a red/blue swap must be detectable."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 65536, size=(h, w, 3), dtype=np.uint16)
    a[..., 0] //= 4          # force R != B everywhere the swap could hide
    a[..., 2] |= 0x8000
    return a


def test_bgr_writer_matches_rgb_writer_byte_for_byte(tmp_path):
    """Storing BGR and using the BGR entry point produces the IDENTICAL file to
    storing RGB and letting the RGB entry point reverse it. This is the whole
    correctness argument for the exporter storing BGR."""
    rgb = _rgb_u16()
    bgr = np.ascontiguousarray(rgb[..., ::-1])

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    write_png16_rgb_u16(a, rgb)
    write_png16_bgr_u16(b, bgr)
    assert a.read_bytes() == b.read_bytes()

    # ...and both decode back to the original RGB.
    np.testing.assert_array_equal(
        read_png16(a), rgb.astype(np.float32) / np.float32(65535.0)
    )


def test_bgr_and_rgb_writers_are_not_the_same_permutation(tmp_path):
    """Guard against the test above passing vacuously on symmetric content."""
    rgb = _rgb_u16()
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    write_png16_rgb_u16(a, rgb)
    write_png16_bgr_u16(b, rgb)          # deliberately NOT reversed
    assert a.read_bytes() != b.read_bytes()


@pytest.mark.parametrize("writer", [write_png16_rgb_u16, write_png16_bgr_u16])
@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4, 4), dtype=np.uint16),      # 4 channels
        np.zeros((4, 4), dtype=np.uint16),         # 2-D
        np.zeros((4, 4, 3), dtype=np.float32),     # wrong dtype
    ],
)
def test_u16_writers_reject_bad_shapes(tmp_path, writer, bad):
    with pytest.raises(ValueError):
        writer(tmp_path / "x.png", bad)


def test_gray_writer_honours_compression(tmp_path):
    """png_compression must actually reach write_png16_gray: the exporter
    called it with no argument, silently pinning level 2. The pfield
    description documented that limitation rather than the lever being broken,
    so nothing was inconsistent -- this pins the lever now that it is wired."""
    rng = np.random.default_rng(5)
    g = rng.random((256, 256), dtype=np.float32)

    fast, small = tmp_path / "f.png", tmp_path / "s.png"
    write_png16_gray(fast, g, 0)
    write_png16_gray(small, g, 9)
    assert fast.stat().st_size != small.stat().st_size
    # Lossless either way: identical decoded content.
    np.testing.assert_array_equal(read_png16(fast), read_png16(small))


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_gray_writer_clips_out_of_range_and_nan(tmp_path):
    """The writer's INTERNAL clip is load-bearing once the exporter's redundant
    outer np.clip is removed. NaN -> 0 is numpy's float->uint cast behaviour and
    is pinned here deliberately, not asserted as desirable."""
    g = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0, np.inf, -np.inf, np.nan]], dtype=np.float32)
    path = tmp_path / "g.png"
    write_png16_gray(path, g)
    out = read_png16(path)[0]

    assert out[0] == pytest.approx(0.0)      # -1 clipped low
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.5, abs=1e-4)
    assert out[3] == pytest.approx(1.0)
    assert out[4] == pytest.approx(1.0)      # 2.0 clipped high
    assert out[5] == pytest.approx(1.0)      # +inf clipped high
    assert out[6] == pytest.approx(0.0)      # -inf clipped low
    assert out[7] == pytest.approx(0.0)      # NaN -> 0


# -- C2: per-tile uint16 conversion must equal whole-map float conversion ------


def _bands(h: int, w: int, band: int):
    """Tile origins that deliberately do NOT divide the height evenly."""
    return [(y, min(band, h - y)) for y in range(0, h, band)]


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
@pytest.mark.parametrize("band", [7, 13, 64])
def test_banded_u16_height_matches_whole_map_float_write(tmp_path, band):
    """The sequence converts height to uint16 in the TILE loop and writes with
    write_png16_gray_u16; it used to hand a whole float32 map to
    write_png16_gray. Those must be bit-identical on disk, including the values
    that make rounding interesting: exact .5 steps, out-of-range, +-inf, -0.0
    and NaN. Band sizes are chosen NOT to divide the height."""
    h, w = 91, 37
    rng = np.random.default_rng(11)
    g = rng.random((h, w), dtype=np.float32)
    # Exact .5-of-a-step values: rounding must not diverge between the paths.
    steps = (np.arange(w, dtype=np.float32) + 0.5) / np.float32(65535.0)
    g[0] = steps
    g[1] = np.linspace(-0.25, 1.25, w, dtype=np.float32)   # out of range both ends
    g[2, :6] = [np.inf, -np.inf, np.nan, -0.0, 0.0, 1.0]

    # Old path: whole map, float in.
    ref = tmp_path / "ref.png"
    write_png16_gray(ref, g)

    # New path: convert per band, exactly as the tile scatter does.
    buf = np.empty((h, w), dtype=np.uint16)
    for y0, th in _bands(h, w, band):
        chunk = g[y0 : y0 + th]
        buf[y0 : y0 + th] = (np.clip(chunk, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    new = tmp_path / "new.png"
    write_png16_gray_u16(new, buf)

    assert ref.read_bytes() == new.read_bytes()


def test_gray_u16_writer_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        write_png16_gray_u16(tmp_path / "x.png", np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        write_png16_gray_u16(tmp_path / "x.png", np.zeros((4, 4, 3), dtype=np.uint16))


# -- PR D: the EXR writer must PRESERVE dtype ---------------------------------


def test_exr_rgba_preserves_half_and_full_precision(tmp_path):
    """write_exr_rgba used to coerce to float32 unconditionally. Handing it a
    float16 buffer would then have DOUBLED the peak instead of halving it -- the
    exact opposite of what export.emission_half exists to do."""
    import OpenEXR

    from gasgiant.export.writers import read_exr_rgba, write_exr_rgba

    rng = np.random.default_rng(17)
    a32 = (rng.random((64, 64, 4), dtype=np.float32) * 100.0)
    a16 = a32.astype(np.float16)

    p32, p16 = tmp_path / "f32.exr", tmp_path / "f16.exr"
    write_exr_rgba(p32, a32)
    write_exr_rgba(p16, a16)

    for path, want in ((p32, np.float32), (p16, np.float16)):
        with OpenEXR.File(str(path)) as f:
            assert f.channels()["RGBA"].pixels.dtype == want, f"{path.name} on-disk dtype"

    assert p16.stat().st_size < p32.stat().st_size

    # The reader always upcasts, and does so bit-exactly.
    back = read_exr_rgba(p16)
    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, a16.astype(np.float32))


def test_exr_rgba_rejects_other_dtypes(tmp_path):
    from gasgiant.export.writers import write_exr_rgba

    with pytest.raises(ValueError):
        write_exr_rgba(tmp_path / "x.exr", np.zeros((4, 4, 4), dtype=np.float64))

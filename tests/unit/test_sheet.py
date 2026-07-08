"""T6 contact sheet: pure grid composer, the 8-bit writer round-trip, and the
single-sim-reuse property of the seed loop (proven with a fake facade -- no GL)."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.export.sheet import compose_grid, run_sheet, sheet_job
from gasgiant.export.writers import decode_image, write_png8_rgb
from gasgiant.jobs import Progress
from gasgiant.params.model import PlanetParams

BG = (0.05, 0.05, 0.05)


# -- compose_grid ------------------------------------------------------------


def _cells(n, h=4, w=6):
    # Distinct constant color per cell so placement is checkable.
    return [np.full((h, w, 3), i + 1, dtype=np.float32) for i in range(n)]


def test_compose_grid_dims_and_padding():
    imgs = _cells(4)
    grid = compose_grid(imgs, cols=2, pad=1, bg=BG)
    # 2 rows x 2 cols of 4x6 cells with 1px padding all around and between.
    assert grid.shape == (2 * 4 + 3 * 1, 2 * 6 + 3 * 1, 3)
    # Outer border is bg.
    assert np.allclose(grid[0, :], BG)
    assert np.allclose(grid[:, 0], BG)
    # First cell sits at (pad, pad).
    assert np.allclose(grid[1:5, 1:7], 1.0)
    # Second cell is to its right, separated by a pad column of bg.
    assert np.allclose(grid[1:5, 7:8], BG)
    assert np.allclose(grid[1:5, 8:14], 2.0)


def test_compose_grid_last_row_padded_with_bg():
    # 3 images into a 2-wide grid -> the 4th cell (row 1, col 1) is empty (bg).
    grid = compose_grid(_cells(3), cols=2, pad=1, bg=BG)
    # Bottom-right cell region should be entirely bg (never written).
    assert np.allclose(grid[6:10, 8:14], BG)
    # And the third image did land at row 1, col 0.
    assert np.allclose(grid[6:10, 1:7], 3.0)


def test_compose_grid_deterministic():
    imgs = _cells(5)
    a = compose_grid(imgs, cols=3, pad=2, bg=BG)
    b = compose_grid(imgs, cols=3, pad=2, bg=BG)
    assert np.array_equal(a, b)


def test_compose_grid_rejects_mismatched_sizes():
    imgs = [np.zeros((4, 4, 3), np.float32), np.zeros((4, 5, 3), np.float32)]
    with pytest.raises(ValueError, match="uniform"):
        compose_grid(imgs, cols=2)


def test_compose_grid_rejects_empty():
    with pytest.raises(ValueError):
        compose_grid([], cols=2)


# -- write_png8_rgb round-trip ----------------------------------------------


def test_write_png8_rgb_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.random((16, 32, 3), dtype=np.float32)
    path = tmp_path / "sheet.png"
    write_png8_rgb(path, rgb)
    assert path.is_file()
    back = decode_image(path, color=True)
    assert back.shape == rgb.shape
    # 8-bit quantization: at most ~1/255 per channel.
    assert np.max(np.abs(back - rgb)) <= 1.001 / 255.0


def test_write_png8_rgb_drops_alpha(tmp_path):
    rgba = np.zeros((8, 8, 4), np.float32)
    rgba[..., 0] = 1.0  # red; alpha channel must be ignored
    path = tmp_path / "a.png"
    write_png8_rgb(path, rgba)
    back = decode_image(path, color=True)
    assert np.allclose(back[..., 0], 1.0, atol=1 / 255)
    assert np.allclose(back[..., 1:], 0.0, atol=1 / 255)


def test_write_png8_rgb_rejects_2d(tmp_path):
    with pytest.raises(ValueError):
        write_png8_rgb(tmp_path / "x.png", np.zeros((4, 4), np.float32))


# -- sheet_job single-sim reuse (fake facade, no GL) ------------------------


class _FakeSim:
    """Records that ONE facade is built, re-seeded per seed, released once."""

    def __init__(self, params, gpu=None):
        self.params = params
        self.gpu = gpu
        self.update_seeds: list[int] = []
        self.run_count = 0
        self.released = 0

    def update_params(self, new_params):
        self.params = new_params
        self.update_seeds.append(new_params.seed)

    def run_to_completion(self):
        self.run_count += 1

    def render_maps(self, width):
        h = width // 2
        val = np.float32((self.params.seed % 5) / 5.0)
        return {"color": np.full((h, width, 4), val, dtype=np.float32)}

    def release(self):
        self.released += 1


def _base_params():
    p = PlanetParams(seed=100)
    p.sim.dev_steps = 8
    return p


def test_sheet_job_reuses_one_sim(tmp_path):
    created: list[_FakeSim] = []

    def factory(params, gpu):
        sim = _FakeSim(params, gpu)
        created.append(sim)
        return sim

    seeds = [1, 2, 3, 4]
    run_sheet(factory, _base_params(), seeds, tmp_path / "sheet.png", width=8)

    # EXACTLY one Simulation constructed (no per-seed leak).
    assert len(created) == 1
    sim = created[0]
    # update_params called once per seed, in order.
    assert sim.update_seeds == seeds
    # run_to_completion once per seed.
    assert sim.run_count == len(seeds)
    # released exactly once, at the end.
    assert sim.released == 1
    assert (tmp_path / "sheet.png").is_file()


def test_sheet_job_releases_sim_on_error(tmp_path):
    class _Boom(_FakeSim):
        def render_maps(self, width):
            raise RuntimeError("boom")

    created: list[_Boom] = []

    def factory(params, gpu):
        sim = _Boom(params, gpu)
        created.append(sim)
        return sim

    with pytest.raises(RuntimeError, match="boom"):
        run_sheet(factory, _base_params(), [1, 2], tmp_path / "s.png", width=8)
    # Even on failure the single sim is torn down exactly once (no leak).
    assert len(created) == 1
    assert created[0].released == 1


def test_sheet_job_dev_steps_override_and_progress(tmp_path):
    created: list[_FakeSim] = []

    def factory(params, gpu):
        # dev_steps override is applied to the base BEFORE the sim is built.
        assert params.sim.dev_steps == 3
        sim = _FakeSim(params, gpu)
        created.append(sim)
        return sim

    progress = list(
        sheet_job(factory, _base_params(), [7, 8], tmp_path / "s.png",
                  width=8, dev_steps=3)
    )
    assert all(isinstance(p, Progress) for p in progress)
    # One slice per seed + one for the compose/write.
    assert progress[-1].done == progress[-1].total == 3


def test_sheet_job_default_cols_square(tmp_path):
    # cols=None -> ceil(sqrt(N)); N=4 -> 2 cols -> a 2x2 grid PNG.
    def factory(params, gpu):
        return _FakeSim(params, gpu)

    run_sheet(factory, _base_params(), [1, 2, 3, 4], tmp_path / "s.png", width=8)
    grid = decode_image(tmp_path / "s.png", color=True)
    # 2 cols x 2 rows of 4x8 cells + 8px default padding around/between.
    assert grid.shape == (2 * 4 + 3 * 8, 2 * 8 + 3 * 8, 3)


def test_sheet_job_rejects_empty_seeds(tmp_path):
    with pytest.raises(ValueError, match="at least one seed"):
        run_sheet(lambda p, g: _FakeSim(p, g), _base_params(), [], tmp_path / "s.png")

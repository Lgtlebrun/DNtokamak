import numpy as np
import pytest

from dntokamak import Currents, Grid, load_config, ring, save_config
from dntokamak.configs import Config


def test_default_grid_shape(grid):
    assert grid.mesh[0].shape == (324, 244)
    assert grid.staggered_mesh[1].shape == (324, 244)
    assert grid.x[0] == pytest.approx(-1.5 * grid.dx)
    assert grid.y[-1] == pytest.approx(grid.ly + 1.5 * grid.dy)


def test_inconsistent_grid_is_rejected():
    with pytest.raises(ValueError):
        Grid(lx=0.7, nx=7)


def test_separatrix_cannot_reach_the_shaping_ring(grid):
    """Justifies not filtering contour points by distance to the shaping ring."""
    cx, cy = grid.center
    corners = [(x, y) for x in grid.x[[0, -1]] for y in grid.y[[0, -1]]]
    farthest = max(np.hypot(x - cx, y - cy) for x, y in corners)
    assert farthest < 0.7 * max(grid.lx, grid.ly)


def test_ring_geometry(grid):
    cur = ring(grid, 3, 1.0, [0.0] * 8)
    assert len(cur) == 9
    cx, cy = grid.center
    assert (cur.x[0], cur.y[0]) == (cx, cy)
    radii = np.hypot(cur.x[1:] - cx, cur.y[1:] - cy)
    np.testing.assert_allclose(radii, 0.7 * max(grid.lx, grid.ly))


def test_ring_validates_arguments(grid):
    with pytest.raises(ValueError):
        ring(grid, 1, 1.0, [0.0, 0.0])
    with pytest.raises(ValueError):
        ring(grid, 2, 1.0, [0.0])


def test_currents_are_immutable():
    cur = Currents([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        cur.amp[0] = 5.0
    changed = cur.with_amp(0, 5.0)
    assert cur.amp[0] == 1.0 and changed.amp[0] == 5.0


def test_from_legacy_flattens_single_element_arrays():
    legacy = [[np.array([1.0]), 2.0], [3.0, 4.0], [np.array([5.0]), 6.0]]
    cur = Currents.from_legacy(legacy)
    assert cur.x.tolist() == [1.0, 2.0]
    assert cur.amp.tolist() == [5.0, 6.0]


def test_recentered_moves_main_current_to_center(grid):
    cur = Currents([10.0, 20.0], [30.0, 40.0], [1.0, 1.0])
    moved = cur.recentered(grid)
    assert moved.x[0] == pytest.approx(grid.center[0])
    assert moved.x[1] - moved.x[0] == pytest.approx(10.0)
    assert moved.y.tolist() == cur.y.tolist()
    both = cur.recentered(grid, y=True)
    assert both.y[0] == pytest.approx(grid.center[1])


def test_config_json_round_trip_is_exact(tmp_path, shipped_config):
    path = tmp_path / "copy.json"
    save_config(path, shipped_config)
    again = load_config(path)
    for name in ("x", "y", "amp"):
        assert np.array_equal(
            getattr(again.currents, name), getattr(shipped_config.currents, name)
        )
    assert again.grid == shipped_config.grid
    assert again.meta == shipped_config.meta


def test_config_rejects_unknown_kind(grid):
    with pytest.raises(ValueError):
        Config(ring(grid, 2, 1.0, [0.0] * 4), grid, "triple_null")

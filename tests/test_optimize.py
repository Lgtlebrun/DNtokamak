import numpy as np
import pytest

from dntokamak import Currents
from dntokamak.optimize import symmetric, up_down
from dntokamak.optimize.common import check_bounds
from dntokamak.optimize.symmetric import Goal


def test_mirror_pairs_of_a_ring_with_eight_shaping_currents():
    pairs = {i: symmetric.mirror_index(i, 8) for i in range(9)}
    assert pairs == {
        0: None,
        1: None,
        2: 8,
        3: 7,
        4: 6,
        5: None,
        6: None,
        7: None,
        8: None,
    }


def test_symmetric_amplitude_is_set_on_both_mirrors():
    cur = Currents(np.zeros(9), np.zeros(9), np.ones(9))
    out = symmetric.with_symmetric_amp(cur, 3, -2.5)
    assert out.amp[3] == out.amp[7] == 2.5
    assert cur.amp[3] == 1.0


def test_initial_currents_have_two_large_shaping_currents(grid):
    cur = symmetric.initial_currents(grid)
    assert len(cur) == 9
    assert np.flatnonzero(cur.amp[1:] > 1).tolist() == [2, 6]


def test_check_bounds_reorders_and_validates():
    assert check_bounds(155, 125, 800) == (125, 155)
    with pytest.raises(ValueError):
        check_bounds(-1, 100, 800)
    with pytest.raises(ValueError):
        check_bounds(100, 900, 800)


def test_goal_selection_follows_the_targets(grid):
    problem = symmetric._Problem(grid, 1.7, -0.4, 125, 155)
    shape = lambda k, d: symmetric.Shape(k, d, d, 0, 1, (0, 0), (0, 1))  # noqa: E731
    assert problem.choose_goal(shape(1.7, -0.2), 0.01) is Goal.DELTA
    assert problem.choose_goal(shape(1.5, -0.4), 0.01) is Goal.KAPPA
    assert problem.choose_goal(shape(1.7, -0.4), 0.01) is Goal.BOTH


def test_matching_a_window_that_already_holds_the_x_point_is_a_no_op(coarse_grid):
    start = symmetric.initial_currents(coarse_grid)
    out = symmetric.match_xpoint_bounds(start, coarse_grid, (1, 799))
    assert out is start


@pytest.mark.slow
def test_symmetric_fit_does_not_worsen_the_start(coarse_grid, symmetric_config):
    start = symmetric_config.currents
    problem = symmetric._Problem(coarse_grid, 1.7, -0.5, 100, 200)
    before = problem.cost(problem.measure(start))
    result = symmetric.fit(
        start, coarse_grid, 1.7, -0.5, (100, 200), max_rounds=2, nm_maxiter=20
    )
    assert result.cost <= before
    assert len(result.currents) == len(start)


@pytest.mark.slow
def test_up_down_triangularity_fit_improves_the_cost(coarse_grid, up_down_config):
    start = up_down_config.currents
    first = up_down.measure(start, coarse_grid)
    before = abs(0.05 - first.delta_up) ** 2 + abs(0.03 - first.delta_down) ** 2
    result = up_down.fit_triangularity(
        start, coarse_grid, 0.05, 0.03, (100, 200), nm_maxiter=60
    )
    assert result.cost < before


@pytest.mark.slow
def test_up_down_elongation_fit_reports_non_convergence(coarse_grid, up_down_config):
    result = up_down.fit_kappa(
        up_down_config.currents, coarse_grid, 1.2, (100, 200), max_iterations=1
    )
    assert not result.converged

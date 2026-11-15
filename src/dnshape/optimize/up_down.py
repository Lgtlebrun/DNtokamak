"""Fit of an up/down asymmetric double null to a target kappa, delta_up and delta_down.

The default strategy has two stages: elongation first, by varying one amplitude at a
time, then the triangularities with a Nelder Mead run over the position of the main
current and all amplitudes. The elongation is not constrained in the second stage.

fit_global is the earlier strategy of Albin Vernhes: one Nelder Mead run over every
coordinate of every current, with the elongation in the cost.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import fmin, minimize

from dnshape.currents import Currents, ring
from dnshape.equilibrium import Equilibrium, xpoint_lower_half
from dnshape.grid import Grid
from dnshape.optimize.common import PENALTY, Result, check_bounds
from dnshape.shape import Shape, up_down

log = logging.getLogger(__name__)


def initial_currents(grid: Grid, n: int = 2) -> Currents:
    """Main current and 2**n shaping currents with a hand tuned asymmetric start."""
    n_aux = 2**n
    prop = max(grid.lx, grid.ly) / 800
    amp_aux = [0.9 * prop if i == int(0.75 * n_aux) else 0 for i in range(n_aux)]
    amp_aux[0] = 0.2 * prop
    amp_aux[int(n_aux / 2)] = 0.2 * prop
    amp_aux[int(n_aux / 4)] = 0.8 * prop
    return ring(grid, n, 0.5 * prop, amp_aux)


def measure(currents: Currents, grid: Grid) -> Shape:
    return up_down(Equilibrium(currents, grid, xpoint_lower_half))


def _usable_shape(currents, grid, lower, upper) -> Shape | None:
    """Shape of a configuration, None if its X point is out of bounds or unusable."""
    eq = Equilibrium(currents, grid, xpoint_lower_half)
    if not eq.in_bounds(lower, upper):
        return None
    try:
        return up_down(eq)
    except ValueError:
        return None


def _options(nm_maxiter: int | None) -> dict | None:
    return None if nm_maxiter is None else {"maxiter": nm_maxiter}


def fit_kappa(
    start: Currents,
    grid: Grid,
    kappa: float,
    xpoint_bounds: tuple[float, float],
    tol: float = 1e-4,
    max_iterations: int = 1000,
) -> Result:
    """Fit the elongation by varying one amplitude at a time.

    Every 20 iterations the tolerance grows by tol.
    """
    lower, upper = check_bounds(*xpoint_bounds, grid.ly)

    def step(base: Currents, i: int) -> Currents:
        def cost(c):
            shape = _usable_shape(base.with_amp(i, c[0]), grid, lower, upper)
            return np.inf if shape is None else abs(kappa - shape.kappa)

        best = fmin(cost, base.amp[i], disp=False)
        return base.with_amp(i, best[0])

    current = start
    shape = measure(current, grid)
    level, iteration = tol, 1
    log.info("Target elongation: kappa = %s", kappa)
    while abs(kappa - shape.kappa) > level:
        if iteration > max_iterations:
            return Result(current, shape, abs(kappa - shape.kappa), iteration, False)
        if iteration % 20 == 0:
            level += tol
            log.warning("Tolerance on elongation raised to %s", level)
        for i in range(len(current)):
            log.debug("Elongation step, current %d", i)
            current = step(current, i)
            shape = measure(current, grid)
        iteration += 1
    return Result(current, shape, abs(kappa - shape.kappa), iteration - 1)


def fit_triangularity(
    start: Currents,
    grid: Grid,
    delta_up: float,
    delta_down: float,
    xpoint_bounds: tuple[float, float],
    tol: float = 1e-4,
    nm_maxiter: int | None = None,
) -> Result:
    """Fit both triangularities with Nelder Mead over x0, y0 and all amplitudes."""
    lower, upper = check_bounds(*xpoint_bounds, grid.ly)

    def to_currents(p) -> Currents:
        x0, y0, *amp = p
        return Currents(start.x, start.y, amp).with_main(x0 * grid.lx, y0 * grid.ly)

    def squared_error(shape: Shape) -> float:
        return (
            abs(delta_up - shape.delta_up) ** 2
            + abs(delta_down - shape.delta_down) ** 2
        )

    def cost(p):
        shape = _usable_shape(to_currents(p), grid, lower, upper)
        return PENALTY if shape is None else squared_error(shape)

    p0 = [start.x[0] / grid.lx, start.y[0] / grid.ly, *start.amp]
    result = minimize(
        cost, p0, tol=tol, method="Nelder-Mead", options=_options(nm_maxiter)
    )
    log.info("Optimisation status: %s", result.status)

    best = to_currents(result.x).recentered(grid, y=True)
    shape = measure(best, grid)
    return Result(best, shape, squared_error(shape), converged=bool(result.success))


def fit_global(
    start: Currents,
    grid: Grid,
    kappa: float,
    delta_up: float,
    delta_down: float,
    xpoint_bounds: tuple[float, float],
    nm_maxiter: int | None = None,
) -> Result:
    """Nelder Mead over every coordinate of every current, with kappa in the cost."""
    lower, upper = check_bounds(*xpoint_bounds, grid.ly)

    def to_currents(v) -> Currents:
        return Currents(*np.split(v, 3))

    def squared_error(shape: Shape) -> float:
        return (
            (delta_up - shape.delta_up) ** 2
            + (delta_down - shape.delta_down) ** 2
            + (kappa - shape.kappa) ** 2
        )

    def cost(v):
        candidate = to_currents(v)
        if not Equilibrium(candidate, grid, xpoint_lower_half).in_bounds(lower, upper):
            return PENALTY
        try:
            shape = measure(candidate.recentered(grid, y=True), grid)
        except ValueError:
            return PENALTY
        return squared_error(shape)

    v0 = np.concatenate([start.x, start.y, start.amp])
    result = minimize(cost, v0, method="Nelder-Mead", options=_options(nm_maxiter))
    best = to_currents(result.x).recentered(grid, y=True)
    shape = measure(best, grid)
    return Result(best, shape, squared_error(shape), converged=bool(result.success))

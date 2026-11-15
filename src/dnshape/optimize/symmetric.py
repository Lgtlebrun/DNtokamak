"""Fit of an up/down symmetric double null to a target kappa and delta.

The shaping currents are mirrored about the midplane. Free parameters are the radial
position and amplitude of the main current and half of the shaping amplitudes.

The fit starts with one Nelder Mead run over all free parameters. If the targets are
missed, rounds follow that alternate a 1D fit of the main current position with 1D fits
of each amplitude, choosing the quantity to correct (delta or kappa) after every step.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import fmin, minimize

from dnshape.currents import Currents, ring
from dnshape.equilibrium import Equilibrium, xpoint_fold
from dnshape.grid import Grid
from dnshape.optimize.common import PENALTY, Result, check_bounds
from dnshape.shape import Shape, symmetric

log = logging.getLogger(__name__)


class Goal(Enum):
    """Quantity a 1D step tries to correct."""

    DELTA = "delta"
    KAPPA = "kappa"
    BOTH = "both"
    XPOINT = "xpoint"


def initial_currents(grid: Grid, n: int = 3) -> Currents:
    """Main current and 2**n shaping currents, the top and bottom ones being large."""
    n_aux = 2**n
    amp_aux = [1.85 if (i + n_aux / 4) % (n_aux / 2) == 0 else 0 for i in range(n_aux)]
    return ring(grid, n, 1.0, amp_aux)


def mirror_index(i: int, n_aux: int) -> int | None:
    """Index of the current mirrored to current i, None if i has no distinct mirror."""
    return n_aux + 2 - i if 1 < i < (n_aux + 1) / 2 else None


def with_symmetric_amp(currents: Currents, i: int, value: float) -> Currents:
    """Set the amplitude of current i, and of its mirror, to |value|."""
    amp = currents.amp.copy()
    amp[i] = abs(value)
    mirror = mirror_index(i, len(currents) - 1)
    if mirror is not None:
        amp[mirror] = abs(value)
    return Currents(currents.x, currents.y, amp)


@dataclass
class _Problem:
    """Targets, X point window and the evaluations shared by the fitting steps."""

    grid: Grid
    kappa: float
    delta: float
    lower: float
    upper: float
    nm_maxiter: int | None = None

    def equilibrium(self, currents: Currents) -> Equilibrium:
        return Equilibrium(currents, self.grid, xpoint_fold)

    def measure(self, currents: Currents) -> Shape:
        return symmetric(self.equilibrium(currents))

    def in_bounds(self, currents: Currents) -> bool:
        return self.equilibrium(currents).in_bounds(self.lower, self.upper)

    def squared_errors(self, shape: Shape) -> tuple[float, float]:
        return (self.kappa - shape.kappa) ** 2, (self.delta - shape.delta) ** 2

    def cost(self, shape: Shape) -> float:
        k2, d2 = self.squared_errors(shape)
        return k2 + d2

    def reached(self, currents: Currents, shape: Shape, tol: float) -> bool:
        return (
            abs(shape.kappa - self.kappa) <= tol
            and abs(shape.delta - self.delta) <= tol
            and self.in_bounds(currents)
        )

    def choose_goal(self, shape: Shape, tol: float) -> Goal:
        delta_ok = abs(shape.delta - self.delta) <= tol
        if delta_ok and abs(shape.kappa - self.kappa) > tol:
            return Goal.KAPPA
        if not delta_ok:
            return Goal.DELTA
        return Goal.BOTH

    def vector_to_currents(self, base: Currents, p: np.ndarray) -> Currents:
        """Currents from [x0 / lx, free amplitudes], keeping the mirror symmetry."""
        x = base.x.copy()
        x[0] = p[0] * self.grid.lx
        amp = base.amp.copy()
        n_aux = len(base) - 1
        for i, value in enumerate(p[1:]):
            amp[i] = abs(value)
            mirror = mirror_index(i, n_aux)
            if mirror is not None:
                amp[mirror] = abs(value)
        return Currents(x, base.y, amp)

    def all_in_one(self, base: Currents) -> Currents:
        """Nelder Mead over x0 and the free amplitudes at once."""
        n_free = math.ceil((len(base) - 1) / 2 + 2)
        p0 = [base.x[0] / self.grid.lx, *base.amp[:n_free]]
        bounds = [(-1, 1)] + [(0, 10)] * n_free

        def cost(p):
            candidate = self.vector_to_currents(base, p)
            if not self.in_bounds(candidate):
                return PENALTY
            try:
                shape = self.measure(candidate.recentered(self.grid))
            except ValueError:
                return PENALTY
            return self.cost(shape)

        result = minimize(
            cost,
            p0,
            method="Nelder-Mead",
            bounds=bounds,
            options=self._options(),
        )
        log.info("All in one fit ended with parameters %s", result.x)
        return self.vector_to_currents(base, result.x).recentered(self.grid)

    def step_x0(self, base: Currents, goal: Goal) -> Currents:
        """1D fit of the main current position, relative to the shaping currents."""

        def cost(p):
            eq = self.equilibrium(base.with_main(x=p[0]).recentered(self.grid))
            if not eq.in_bounds(self.lower, self.upper):
                return np.inf
            try:
                k2, d2 = self.squared_errors(symmetric(eq))
            except ValueError:
                return np.inf
            return d2 if goal is Goal.DELTA else k2 + d2

        best = fmin(cost, base.x[0], maxiter=self.nm_maxiter, disp=False)
        return base.with_main(x=best[0]).recentered(self.grid)

    def step_current(self, base: Currents, i: int, goal: Goal) -> Currents:
        """1D fit of the amplitude of current i, and of its mirror."""

        def cost(c):
            eq = self.equilibrium(with_symmetric_amp(base, i, c[0]))
            inside = eq.in_bounds(self.lower, self.upper)
            if goal is Goal.XPOINT:
                z = eq.xpoint[1]
                return 0 if inside else (z - self.lower) ** 2 + (z - self.upper) ** 2
            if not inside:
                return np.inf
            try:
                k2, d2 = self.squared_errors(symmetric(eq))
            except ValueError:
                return np.inf
            return k2 if goal is Goal.KAPPA else d2 + k2

        best = fmin(cost, base.amp[i], maxiter=self.nm_maxiter, disp=False)
        return with_symmetric_amp(base, i, best[0])

    def _options(self) -> dict | None:
        return None if self.nm_maxiter is None else {"maxiter": self.nm_maxiter}


def match_xpoint_bounds(
    start: Currents,
    grid: Grid,
    xpoint_bounds: tuple[float, float],
    max_sweeps: int = 100,
) -> Currents:
    """Adjust amplitudes one by one until the X point height lies in the window."""
    lower, upper = check_bounds(*xpoint_bounds, grid.ly)
    problem = _Problem(grid, math.nan, math.nan, lower, upper)
    current = start
    for _ in range(max_sweeps):
        if problem.in_bounds(current):
            return current
        for i in range(len(current)):
            current = problem.step_current(current, i, Goal.XPOINT)
            if problem.in_bounds(current):
                return current
    raise RuntimeError("X point height did not reach its bounds, try widening them")


def fit(
    start: Currents,
    grid: Grid,
    kappa: float,
    delta: float,
    xpoint_bounds: tuple[float, float],
    tol: float = 0.01,
    max_rounds: int = 100,
    nm_maxiter: int | None = None,
) -> Result:
    """Fit kappa and delta, starting from a configuration inside the X point window.

    Every 20 rounds the tolerance grows by tol and the fit restarts from start.
    The best configuration seen is returned.
    """
    lower, upper = check_bounds(*xpoint_bounds, grid.ly)
    problem = _Problem(grid, kappa, delta, lower, upper, nm_maxiter)
    log.info("Targets: kappa = %s, delta = %s", kappa, delta)

    current = problem.all_in_one(start)
    shape = problem.measure(current)
    best, best_cost = current, problem.cost(shape)
    goal = problem.choose_goal(shape, tol)

    level, rounds, converged = tol, 0, True
    while not problem.reached(current, shape, level):
        rounds += 1
        if rounds % max_rounds == 0:
            log.warning("No convergence after %d rounds, aborting", rounds)
            converged = False
            break
        if rounds % 20 == 0:
            level += tol
            current = start
            log.warning("Tolerance raised to %s and configuration reset", level)

        log.info("Round %d, main current for %s", rounds, goal.value)
        current = problem.step_x0(current, goal)
        shape = problem.measure(current)
        if problem.cost(shape) < best_cost:
            best, best_cost = current, problem.cost(shape)
        goal = problem.choose_goal(shape, level)

        for i in range((len(start) - 1) // 2 + 2):
            log.debug("Round %d, current %d for %s", rounds, i, goal.value)
            current = problem.step_current(current, i, goal)
            shape = problem.measure(current)
            if problem.cost(shape) < best_cost:
                best, best_cost = current, problem.cost(shape)
            goal = problem.choose_goal(shape, level)

    return Result(best, problem.measure(best), best_cost, rounds, converged)


def center_legs(currents: Currents, grid: Grid, wall_margin: float = 20.0) -> Currents:
    """Shift all currents in x to move the separatrix legs away from the side walls."""
    eq = Equilibrium(currents, grid, xpoint_fold)
    shape = symmetric(eq)
    r = np.concatenate(eq.separatrix)[:, 0]

    if shape.delta < 0:
        shift = 0.75 * grid.lx - r.max()
    else:
        shift = 0.25 * grid.lx - r.min()

    iterations = 1
    while (shape.r_min + shift < wall_margin) or (
        shape.r_max + shift > grid.ly - wall_margin
    ):
        shift *= 1 + 1e-3
        iterations += 1
        if iterations > 1000:
            log.warning("Centering of the legs aborted, the shaping may be unsuitable")
            return currents
    log.info("Legs shifted by %s", shift)
    return currents.shifted(shift)

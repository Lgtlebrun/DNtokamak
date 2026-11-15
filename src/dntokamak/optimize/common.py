"""Pieces shared by the optimisers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from dntokamak.currents import Currents
from dntokamak.shape import Shape

log = logging.getLogger(__name__)

PENALTY = 9999999
"""Cost of a configuration that cannot be used, for the global Nelder Mead fits."""


@dataclass(frozen=True)
class Result:
    """Outcome of a fit: best configuration, its shape and cost."""

    currents: Currents
    shape: Shape
    cost: float
    rounds: int = 0
    converged: bool = True


def check_bounds(lower: float, upper: float, ly: float) -> tuple[float, float]:
    """Validate the window for the X point height, reordering it if reversed."""
    if not (0 <= lower <= ly and 0 <= upper <= ly):
        raise ValueError(f"bounds on the X point height must lie within [0, {ly}]")
    if lower > upper:
        lower, upper = upper, lower
    elif lower == upper:
        log.warning("X point bounds are equal, the fit may not converge")
    return lower, upper

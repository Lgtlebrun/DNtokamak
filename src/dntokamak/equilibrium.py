"""Flux map of a current configuration with its X point and separatrix."""

from __future__ import annotations

from functools import cached_property
from typing import Callable

import contourpy
import numpy as np

from dntokamak import flux
from dntokamak.currents import Currents
from dntokamak.grid import Grid

XPointRule = Callable[[tuple, Grid], tuple]


def _poloidal_field_squared(grad: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    dpdx, dpdy = grad
    return dpdx**2 + dpdy**2


def xpoint_fold(grad, grid: Grid) -> tuple[int, int]:
    """Mesh indices (ix, iy) of the lower X point, assuming up/down symmetry.

    The global minimum of the poloidal field is folded into the lower half.
    """
    bp2 = _poloidal_field_squared(grad)
    iy, ix = divmod(bp2.argmin(), bp2.shape[1])
    if iy > len(grid.y) / 2:
        iy = len(grid.y) - iy
    return ix, iy


def xpoint_lower_half(grad, grid: Grid) -> tuple[int, int]:
    """Mesh indices (ix, iy) of the minimum of the poloidal field in the lower half."""
    lower = _poloidal_field_squared(grad)[: grid.ny // 2 + 1]
    iy, ix = divmod(lower.argmin(), lower.shape[1])
    return ix, iy


XPOINT_RULES: dict[str, XPointRule] = {
    "symmetric": xpoint_fold,
    "up_down": xpoint_lower_half,
}


class Equilibrium:
    """Flux and derived quantities of a configuration, computed once on demand."""

    def __init__(
        self,
        currents: Currents,
        grid: Grid,
        xpoint: XPointRule = xpoint_fold,
    ) -> None:
        self.currents = currents
        self.grid = grid
        self._xpoint_rule = xpoint

    @classmethod
    def for_kind(cls, currents: Currents, grid: Grid, kind: str) -> Equilibrium:
        """Equilibrium using the X point rule of a configuration kind."""
        return cls(currents, grid, XPOINT_RULES[kind])

    @cached_property
    def psi(self) -> np.ndarray:
        return flux.psi(self.currents, self.grid)

    @cached_property
    def grad(self) -> tuple[np.ndarray, np.ndarray]:
        return flux.gradient(self.psi)

    @cached_property
    def hess(self) -> tuple[np.ndarray, np.ndarray]:
        return flux.hessian(self.grad)

    @cached_property
    def xpoint_index(self) -> tuple[int, int]:
        return self._xpoint_rule(self.grad, self.grid)

    @property
    def xpoint(self) -> tuple[float, float]:
        """Coordinates (R, Z) of the lower X point."""
        ix, iy = self.xpoint_index
        return self.grid.x[ix], self.grid.y[iy]

    @property
    def level(self) -> float:
        """Flux value on the separatrix."""
        ix, iy = self.xpoint_index
        return self.psi[iy, ix]

    @cached_property
    def separatrix(self) -> tuple[np.ndarray, ...]:
        """Vertices of the contour lines at the X point flux, one array per line."""
        X, Y = self.grid.mesh
        generator = contourpy.contour_generator(
            X,
            Y,
            np.ma.masked_invalid(self.psi),
            name="mpl2014",
            corner_mask=True,
            line_type=contourpy.LineType.SeparateCode,
        )
        vertices, _ = generator.lines(self.level)
        return tuple(vertices)

    def in_bounds(self, lower: float, upper: float) -> bool:
        """Whether the lower X point height lies strictly inside (lower, upper)."""
        return bool(lower < self.xpoint[1] < upper)

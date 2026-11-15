"""Computational box and meshes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Box [0, lx] by [0, ly] in units of rho_s0, padded by two guard cells per side.

    The mesh has nx by ny nodes. The guard cells take two cells on each side, so the
    box itself is covered by 4 cells fewer than nx and ny in each direction.
    """

    lx: float = 600.0
    ly: float = 800.0
    nx: int = 244
    ny: int = 324

    def __post_init__(self) -> None:
        if len(self.x) != self.nx or len(self.y) != self.ny:
            raise ValueError("lx, ly, nx and ny do not give a consistent mesh")

    @property
    def dx(self) -> float:
        return self.lx / (self.nx - 4)

    @property
    def dy(self) -> float:
        return self.ly / (self.ny - 4)

    @cached_property
    def x(self) -> np.ndarray:
        return np.arange(-3 / 2 * self.dx, self.lx + 5 * self.dx / 2, step=self.dx)

    @cached_property
    def y(self) -> np.ndarray:
        return np.arange(-3 / 2 * self.dy, self.ly + 5 * self.dy / 2, step=self.dy)

    @cached_property
    def y_staggered(self) -> np.ndarray:
        return np.arange(-1 / 2 * self.dy, self.ly + 7 * self.dy / 2, step=self.dy)

    @cached_property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Mesh arrays (X, Y) of shape (ny, nx)."""
        return np.meshgrid(self.x, self.y)

    @cached_property
    def staggered_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Mesh with the vertical coordinate on the staggered nodes."""
        return np.meshgrid(self.x, self.y_staggered)

    @property
    def center(self) -> tuple[float, float]:
        """Position of the main current, slightly off the box middle to dodge a node."""
        return 0.51 * self.lx, 0.5 * self.ly

    @property
    def current_scale(self) -> float:
        """Reference current amplitude I0."""
        return 2000 * self.ly / 400 * self.lx / 300

    @property
    def core_width(self) -> float:
        """Gaussian width s of the main current profile."""
        return 60 * self.ly / 400

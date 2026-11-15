"""Poloidal flux of a set of line currents and its finite difference derivatives.

The main current has a Gaussian profile, which adds an exponential integral term and
keeps the flux finite on the current. Shaping currents are filaments.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expn

from dnshape.currents import Currents
from dnshape.grid import Grid


def psi(
    currents: Currents,
    grid: Grid,
    mesh: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Flux on the grid mesh, or on another mesh with the same shape."""
    X, Y = grid.mesh if mesh is None else mesh
    i0, s = grid.current_scale, grid.core_width
    x, y, amp = currents.x, currents.y, currents.amp

    r2 = (X - x[0]) ** 2 + ((Y - y[0]) ** 2)
    out = i0 / 2 * amp[0] * (np.log(r2) + expn(1, r2 / (s**2)))
    for xi, yi, ai in zip(x[1:], y[1:], amp[1:]):
        out += i0 / 2 * ai * (np.log((X - xi) ** 2 + ((Y - yi) ** 2)))
    return out


def gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives along x and y per mesh index, without division by the spacing."""
    return np.gradient(field, axis=1), np.gradient(field, axis=0)


def hessian(grad: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Second derivatives d2/dx2 and d2/dy2 of a flux given its gradient."""
    dpdx, dpdy = grad
    return np.gradient(dpdx, axis=1), np.gradient(dpdy, axis=0)

"""Elongation and triangularity of the separatrix between the two X points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dntokamak.equilibrium import Equilibrium


class ShapeError(ValueError):
    """The separatrix has no usable points between the X points."""


@dataclass(frozen=True)
class Shape:
    """Shape of a double null separatrix.

    kappa is the elongation, delta_up and delta_down the triangularities of the upper
    and lower halves. bottom and top are (R, Z) points used to define them. For
    symmetric shapes the X points play that role and delta_up equals delta_down.
    """

    kappa: float
    delta_up: float
    delta_down: float
    r_min: float
    r_max: float
    bottom: tuple[float, float]
    top: tuple[float, float]

    @property
    def delta(self) -> float:
        """Triangularity of a symmetric shape."""
        return self.delta_up

    def to_dict(self, kind: str) -> dict[str, float]:
        """Scalar summary: kappa and delta for symmetric, both deltas otherwise."""
        if kind == "symmetric":
            return {"kappa": self.kappa, "delta": self.delta}
        return {
            "kappa": self.kappa,
            "delta_up": self.delta_up,
            "delta_down": self.delta_down,
        }


def _concatenate(paths) -> np.ndarray:
    if len(paths) == 0:
        raise ShapeError("no separatrix contour found")
    return np.concatenate(paths)


def _radial_extent(r: np.ndarray) -> tuple[float, float]:
    if r.size == 0:
        raise ShapeError("separatrix has no point between the X points")
    return float(r.min()), float(r.max())


def symmetric(eq: Equilibrium) -> Shape:
    """Shape of an up/down symmetric configuration.

    The elongation is the X point separation over the radial width, the triangularity
    is the offset of the X point from the radial midpoint over the half width.
    """
    grid = eq.grid
    ix, iy = eq.xpoint_index
    r_xpt, z_low = grid.x[ix], grid.y[iy]
    z_up = grid.y[-1] - z_low

    points = _concatenate(eq.separatrix)
    r, z = points[:, 0], points[:, 1]
    r = r[(z > z_low) & (z < z_up)]
    r_min, r_max = _radial_extent(r)

    r0 = (r_min + r_max) / 2
    a = (r_max - r_min) / 2
    kappa = (z_up - z_low) / (r_max - r_min)
    delta = (r0 - r_xpt) / a
    return Shape(
        float(kappa),
        float(delta),
        float(delta),
        r_min,
        r_max,
        (float(r_xpt), float(z_low)),
        (float(r_xpt), float(z_up)),
    )


def _drop_stray_branches(paths, z_up: float) -> list[np.ndarray]:
    """Drop contour branches reaching above the upper X point, keeping at least one.

    A branch following a dropped one is skipped by the scan, as in the original code.
    """
    kept = list(paths)
    i = 0
    while i < len(kept):
        if len(kept) == 1:
            break
        if np.any(kept[i][:, 1] > z_up):
            del kept[i]
        i += 1
    return kept


def up_down(eq: Equilibrium) -> Shape:
    """Shape of an up/down asymmetric configuration.

    The elongation is the vertical extent over the radial width of the separatrix
    between the X points. Each triangularity is the offset of the lowest or highest
    point from the radial midpoint over the half width.
    """
    grid = eq.grid
    _, iy = eq.xpoint_index
    z_low = grid.y[iy]
    z_up = grid.y[-1] - z_low

    points = _concatenate(_drop_stray_branches(eq.separatrix, z_up))
    keep = (points[:, 1] > z_low) & (points[:, 1] <= z_up)
    r, z = points[keep, 0], points[keep, 1]
    r_min, r_max = _radial_extent(r)

    i_bottom, i_top = np.argmin(z), np.argmax(z)
    z_bottom, z_top = z[i_bottom], z[i_top]
    r_bottom, r_top = r[i_bottom], r[i_top]

    r0 = (r_min + r_max) / 2
    a = (r_max - r_min) / 2
    return Shape(
        float((z_top - z_bottom) / (r_max - r_min)),
        float((r0 - r_top) / a),
        float((r0 - r_bottom) / a),
        r_min,
        r_max,
        (float(r_bottom), float(z_bottom)),
        (float(r_top), float(z_top)),
    )


MEASURES = {"symmetric": symmetric, "up_down": up_down}


def measure(eq: Equilibrium, kind: str) -> Shape:
    """Shape of an equilibrium with the definition of a configuration kind."""
    return MEASURES[kind](eq)

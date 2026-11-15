"""Line currents that generate the poloidal flux."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dntokamak.grid import Grid


def _frozen_array(values) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("expected a one dimensional sequence")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class Currents:
    """Positions and amplitudes of the line currents.

    Index 0 is the main plasma current, the others are shaping currents.
    """

    x: np.ndarray
    y: np.ndarray
    amp: np.ndarray

    def __post_init__(self) -> None:
        for name in ("x", "y", "amp"):
            object.__setattr__(self, name, _frozen_array(getattr(self, name)))
        if not len(self.x) == len(self.y) == len(self.amp):
            raise ValueError("x, y and amp must have the same length")

    def __len__(self) -> int:
        return len(self.x)

    @classmethod
    def from_legacy(cls, config: Sequence) -> Currents:
        """Build from the old [x, y, amp] lists, which may hold 1 element arrays."""
        x, y, amp = ([float(np.squeeze(v)) for v in column] for column in config)
        return cls(x, y, amp)

    def to_dict(self) -> dict[str, list[float]]:
        return {k: getattr(self, k).tolist() for k in ("x", "y", "amp")}

    def with_amp(self, index: int, value: float) -> Currents:
        amp = self.amp.copy()
        amp[index] = value
        return Currents(self.x, self.y, amp)

    def with_main(self, x: float | None = None, y: float | None = None) -> Currents:
        """Move the main current only."""
        xs, ys = self.x.copy(), self.y.copy()
        if x is not None:
            xs[0] = x
        if y is not None:
            ys[0] = y
        return Currents(xs, ys, self.amp)

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> Currents:
        return Currents(self.x + dx, self.y + dy, self.amp)

    def recentered(self, grid: Grid, y: bool = False) -> Currents:
        """Shift all currents so that the main one sits at the grid center.

        The vertical position is left alone unless y is True.
        """
        cx, cy = grid.center
        return self.shifted(cx - self.x[0], cy - self.y[0] if y else 0.0)


def ring(grid: Grid, n: int, amp_main: float, amp_aux: Sequence[float]) -> Currents:
    """Main current at the grid center plus 2**n shaping currents on a circle."""
    if n < 2:
        raise ValueError("n cannot be inferior to 2")
    n_aux = 2**n
    if len(amp_aux) != n_aux:
        raise ValueError(f"expected {n_aux} auxiliary amplitudes")

    x0, y0 = grid.center
    radius = max(grid.lx, grid.ly) * 0.7
    angles = [i * (2 * np.pi) / n_aux for i in range(n_aux)]
    xs = [x0] + [x0 + radius * np.cos(a) for a in angles]
    ys = [y0] + [y0 + radius * np.sin(a) for a in angles]
    return Currents(xs, ys, [amp_main, *amp_aux])

"""Contour plot of the flux with the separatrix and the shaping currents.

Requires matplotlib, install with the plot extra.
"""

from __future__ import annotations

import numpy as np

from dntokamak.currents import Currents
from dntokamak.equilibrium import Equilibrium
from dntokamak.grid import Grid
from dntokamak.shape import measure


def _contour_levels(psi: np.ndarray, currents: Currents, grid: Grid) -> np.ndarray:
    """Flux levels from the main current to the box side opposite to the shaping."""
    right = currents.x[0] >= grid.center[0]
    iy0 = int(np.round(currents.y[0] / grid.dy))
    ix0 = int(np.round(currents.x[0] / grid.dx))
    mid_row = psi[grid.ny // 2]
    stop = mid_row[6] if right else mid_row[grid.nx - 5]
    return np.arange(psi[iy0][ix0], stop, step=mid_row[grid.nx // 2] / 200)


def plot_fields(currents: Currents, grid: Grid, kind: str = "symmetric"):
    """Figure of the fields written for GBS: flux, its derivatives and |grad flux|.

    Derivatives are per mesh index, as in the exported file. The separatrix is red and
    the lower X point is marked.
    """
    import matplotlib.pyplot as plt

    eq = Equilibrium.for_kind(currents, grid, kind)
    (dpdx, dpdy), (d2pdx2, d2pdy2) = eq.grad, eq.hess
    panels = [
        (eq.psi, r"$\Psi$", "viridis"),
        (dpdx, r"$\partial_x \Psi$", "RdBu_r"),
        (dpdy, r"$\partial_y \Psi$", "RdBu_r"),
        (np.hypot(dpdx, dpdy), r"$|\nabla \Psi|$", "magma"),
        (d2pdx2, r"$\partial_x^2 \Psi$", "RdBu_r"),
        (d2pdy2, r"$\partial_y^2 \Psi$", "RdBu_r"),
    ]
    X, Y = grid.mesh
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 8.2), constrained_layout=True)
    for ax, (field, title, cmap) in zip(axes.flat, panels):
        limit = np.abs(field).max()
        diverging = cmap == "RdBu_r"
        image = ax.pcolormesh(
            X,
            Y,
            field,
            cmap=cmap,
            shading="auto",
            vmin=-limit if diverging else None,
            vmax=limit if diverging else None,
            rasterized=True,
        )
        ax.contour(X, Y, eq.psi, levels=[eq.level], colors="r", linewidths=1)
        ax.plot(*eq.xpoint, marker="x", color="k")
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    for ax in axes[1]:
        ax.set_xlabel(r"$R/\rho_{s0}$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$Z/\rho_{s0}$")
    return fig


def plot_equilibrium(
    currents: Currents,
    grid: Grid,
    kind: str = "symmetric",
    ax=None,
):
    """Draw flux contours, separatrix (red), currents and shape values on an axes."""
    import matplotlib.pyplot as plt

    eq = Equilibrium.for_kind(currents, grid, kind)
    shape = measure(eq, kind)
    X, Y = grid.mesh

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 6.5), constrained_layout=True)
    ax.set_aspect("equal")
    ax.contour(X, Y, eq.psi, levels=_contour_levels(eq.psi, currents, grid), colors="k")
    ax.contour(X, Y, eq.psi, levels=[eq.level], colors="r")
    for xi, yi in zip(currents.x, currents.y):
        ax.plot(xi, yi, marker="o")
    if kind == "up_down":
        ax.plot(*shape.bottom, marker="+", color="magenta")
        ax.plot(*shape.top, marker="+", color="cyan")
    ax.vlines([0, grid.lx], 0, grid.ly, colors="grey", linestyles="dashed")

    if kind == "symmetric":
        lines = [rf"$\delta = {shape.delta:.4}$"]
    else:
        lines = [
            rf"$\delta_{{up}} = {shape.delta_up:.4}$",
            rf"$\delta_{{down}} = {shape.delta_down:.4}$",
        ]
    lines += [rf"$\kappa = {shape.kappa:.4}$", rf"$Z_{{Xpt}} = {eq.xpoint[1]:.1f}$"]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightsteelblue", alpha=0.5),
    )
    ax.set_xlabel(r"$R/\rho_{s0}$")
    ax.set_ylabel(r"$Z/\rho_{s0}$")
    ax.set_title(r"$\Psi$")
    return ax

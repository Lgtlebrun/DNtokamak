"""HDF5 equilibrium files read by the GBS code."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dntokamak import flux
from dntokamak.currents import Currents
from dntokamak.equilibrium import Equilibrium, xpoint_fold
from dntokamak.grid import Grid


def gbs_filename(delta: float) -> str:
    """File name used for a target triangularity, for example Equil_DN_NT_d-0p4.h5."""
    sign = "NT" if delta < 0 else "PT"
    return f"Equil_DN_{sign}_d{str(delta).replace('.', 'p')}.h5"


def gbs_datasets(
    currents: Currents, grid: Grid, staggered: bool = True
) -> dict[str, np.ndarray]:
    """Flux, its derivatives and X point data of a symmetric configuration.

    Suffix _v marks fields on the mesh nodes, _n fields on the staggered mesh.
    The upper X point height assumes up/down symmetry.
    """
    eq = Equilibrium(currents, grid, xpoint_fold)
    _, iy = eq.xpoint_index
    data = {
        "psi_eq": eq.psi,
        "dpsidx_v": eq.grad[0],
        "dpsidy_v": eq.grad[1],
        "d2psidx2_v": eq.hess[0],
        "d2psidy2_v": eq.hess[1],
    }
    if staggered:
        grad_n = flux.gradient(flux.psi(currents, grid, grid.staggered_mesh))
        hess_n = flux.hessian(grad_n)
        data |= {
            "dpsidx_n": grad_n[0],
            "dpsidy_n": grad_n[1],
            "d2psidx2_n": hess_n[0],
            "d2psidy2_n": hess_n[1],
        }
    scalars = {
        "xmag1": currents.x[0],
        "y0_source": currents.y[0],
        "Yxpt_low": grid.y[iy],
        "Yxpt_up": grid.y[len(grid.y) - iy],
    }
    return data | {k: np.full((1, 1), v) for k, v in scalars.items()}


def write_h5(
    path: str | Path, currents: Currents, grid: Grid, staggered: bool = True
) -> None:
    """Write a configuration as a GBS equilibrium file.

    With staggered False the _n datasets are left out, as in the files delivered
    for the first GBS runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, values in gbs_datasets(currents, grid, staggered).items():
            f.create_dataset(name, data=values, dtype="float64")

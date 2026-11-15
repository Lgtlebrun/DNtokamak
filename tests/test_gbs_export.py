import h5py
import numpy as np
import pytest

from dntokamak import Equilibrium
from dntokamak.gbs_export import gbs_datasets, gbs_filename, write_h5

BASE = {
    "psi_eq", "dpsidx_v", "dpsidy_v", "d2psidx2_v", "d2psidy2_v",
    "xmag1", "y0_source", "Yxpt_low", "Yxpt_up",
}  # fmt: skip
STAGGERED = {"dpsidx_n", "dpsidy_n", "d2psidx2_n", "d2psidy2_n"}


@pytest.mark.parametrize(
    ("delta", "name"),
    [(-0.4, "Equil_DN_NT_d-0p4.h5"), (0.5, "Equil_DN_PT_d0p5.h5")],
)
def test_filename(delta, name):
    assert gbs_filename(delta) == name


def test_dataset_names_with_and_without_staggered_fields(grid, symmetric_config):
    cur = symmetric_config.currents
    assert set(gbs_datasets(cur, grid, staggered=False)) == BASE
    assert set(gbs_datasets(cur, grid, staggered=True)) == BASE | STAGGERED


def test_written_file_content(tmp_path, grid, symmetric_config):
    cur = symmetric_config.currents
    path = tmp_path / "sub" / "equil.h5"
    write_h5(path, cur, grid)
    eq = Equilibrium(cur, grid)
    with h5py.File(path) as f:
        assert set(f) == BASE | STAGGERED
        assert f["psi_eq"].shape == (grid.ny, grid.nx)
        assert f["psi_eq"].dtype == np.float64
        assert np.array_equal(f["psi_eq"][...], eq.psi)
        assert f["xmag1"].shape == (1, 1)
        assert f["xmag1"][0, 0] == cur.x[0]
        assert f["Yxpt_low"][0, 0] == eq.xpoint[1]
        assert not np.array_equal(f["dpsidx_n"][...], f["dpsidx_v"][...])

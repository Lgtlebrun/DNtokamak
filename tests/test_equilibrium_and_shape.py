import re
from types import SimpleNamespace

import numpy as np
import pytest

from dntokamak import Equilibrium, flux, measure, ring
from dntokamak.equilibrium import xpoint_fold, xpoint_lower_half
from dntokamak.shape import ShapeError, _drop_stray_branches, up_down


def test_flux_is_finite_on_the_main_current(grid, symmetric_config):
    eq = Equilibrium(symmetric_config.currents, grid)
    assert np.isfinite(eq.psi).all()


def test_symmetric_currents_give_a_midplane_symmetric_flux(grid):
    cur = ring(grid, 3, 1.0, [0, 0, 1.85, 0, 0, 0, 1.85, 0])
    psi = Equilibrium(cur, grid).psi
    np.testing.assert_allclose(psi, psi[::-1], rtol=1e-9)


def test_gradient_and_hessian_shapes(grid, symmetric_config):
    eq = Equilibrium(symmetric_config.currents, grid)
    assert eq.grad[0].shape == eq.hess[1].shape == (grid.ny, grid.nx)
    assert np.array_equal(eq.grad[0], flux.gradient(eq.psi)[0])


def test_xpoint_rules_agree_up_to_one_row_on_symmetric_currents(grid, symmetric_config):
    grad = Equilibrium(symmetric_config.currents, grid).grad
    (ix_a, iy_a), (ix_b, iy_b) = xpoint_fold(grad, grid), xpoint_lower_half(grad, grid)
    assert ix_a == ix_b and abs(iy_a - iy_b) <= 1


def test_separatrix_is_cached_and_not_empty(grid, symmetric_config):
    eq = Equilibrium(symmetric_config.currents, grid)
    assert len(eq.separatrix) >= 1
    assert eq.separatrix is eq.separatrix
    assert all(line.shape[1] == 2 for line in eq.separatrix)


def test_in_bounds_is_strict(grid, symmetric_config):
    eq = Equilibrium(symmetric_config.currents, grid)
    z = eq.xpoint[1]
    assert eq.in_bounds(z - 1, z + 1)
    assert not eq.in_bounds(z, z + 1)
    assert not eq.in_bounds(z - 1, z)


def test_shape_matches_the_stored_summary(shipped_config):
    eq = Equilibrium.for_kind(
        shipped_config.currents, shipped_config.grid, shipped_config.kind
    )
    measured = measure(eq, shipped_config.kind).to_dict(shipped_config.kind)
    stored = shipped_config.meta["shape"]
    assert measured.keys() == stored.keys()
    for key, value in stored.items():
        assert measured[key] == pytest.approx(value, rel=1e-10)


def test_shipped_configs_keep_their_x_point_near_the_stated_bounds(shipped_config):
    """Up/down fits recenter vertically after the window check, hence a margin."""
    eq = Equilibrium.for_kind(
        shipped_config.currents, shipped_config.grid, shipped_config.kind
    )
    low, high = shipped_config.meta["xpoint_bounds"]
    margin = 5.0 if shipped_config.kind == "up_down" else 0.0
    assert eq.in_bounds(low - margin, high + margin)


def test_symmetric_configs_are_close_to_their_targets(shipped_config):
    if shipped_config.kind != "symmetric":
        pytest.skip("up/down configs do not constrain kappa in the last stage")
    target, shape = shipped_config.meta["targets"], shipped_config.meta["shape"]
    assert shape["kappa"] == pytest.approx(target["kappa"], abs=0.03)
    assert shape["delta"] == pytest.approx(target["delta"], abs=0.03)


def test_up_down_filenames_encode_the_triangularities(shipped_config, request):
    name = request.node.callspec.id
    match = re.search(r"dup_([+-]\d\.\d+)_ddown_([+-]\d\.\d+)", name)
    if match is None:
        pytest.skip("name carries no triangularities")
    shape = shipped_config.meta["shape"]
    assert shape["delta_up"] == pytest.approx(float(match[1]), abs=6e-4)
    assert shape["delta_down"] == pytest.approx(float(match[2]), abs=6e-4)


def test_missing_separatrix_raises_shape_error(grid):
    stub = SimpleNamespace(grid=grid, xpoint_index=(10, 20), separatrix=())
    with pytest.raises(ShapeError):
        up_down(stub)


def test_stray_branch_filter_keeps_one_branch_and_skips_after_a_drop():
    low, high = np.array([[0.0, 1.0]]), np.array([[0.0, 99.0]])
    only = _drop_stray_branches([high, high], z_up=10.0)
    assert len(only) == 1
    kept = _drop_stray_branches([high, high, low], z_up=10.0)
    assert len(kept) == 2 and kept[1] is low

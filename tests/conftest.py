from pathlib import Path

import pytest

from dnshape import Grid, load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIG_PATHS = sorted(CONFIG_DIR.glob("*/*.json"))


@pytest.fixture(scope="session")
def grid() -> Grid:
    return Grid()


@pytest.fixture(scope="session")
def coarse_grid() -> Grid:
    """Same box with 5 times fewer cells per direction, for quick optimiser runs."""
    return Grid(nx=124, ny=164)


@pytest.fixture(params=CONFIG_PATHS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def shipped_config(request):
    return load_config(request.param)


@pytest.fixture(scope="session")
def symmetric_config():
    return load_config(CONFIG_DIR / "symmetric" / "delta_-0.5_kappa_1.7.json")


@pytest.fixture(scope="session")
def up_down_config():
    return load_config(CONFIG_DIR / "up_down" / "start_kappa_1.8.json")

"""Reading and writing configurations as JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from dntokamak.currents import Currents
from dntokamak.equilibrium import XPOINT_RULES
from dntokamak.grid import Grid


@dataclass(frozen=True)
class Config:
    """A current configuration with its grid, kind and free form metadata.

    kind is "symmetric" (one triangularity, up/down symmetric currents) or "up_down".
    """

    currents: Currents
    grid: Grid
    kind: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in XPOINT_RULES:
            raise ValueError(
                f"unknown kind {self.kind!r}, use one of {list(XPOINT_RULES)}"
            )


def _dumps(value, indent: int = 0) -> str:
    """JSON text with one key per line and lists of numbers on a single line."""
    pad = " " * indent
    if isinstance(value, dict):
        items = [
            f"{pad}  {json.dumps(k)}: {_dumps(v, indent + 2).lstrip()}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    return json.dumps(value)


def save_config(path: str | Path, config: Config) -> None:
    data = {
        "kind": config.kind,
        "grid": {
            "lx": config.grid.lx,
            "ly": config.grid.ly,
            "nx": config.grid.nx,
            "ny": config.grid.ny,
        },
        "currents": config.currents.to_dict(),
        **config.meta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(data) + "\n")


def load_config(path: str | Path) -> Config:
    data = json.loads(Path(path).read_text())
    meta = {k: v for k, v in data.items() if k not in ("kind", "grid", "currents")}
    return Config(
        Currents(**data["currents"]), Grid(**data["grid"]), data["kind"], meta
    )

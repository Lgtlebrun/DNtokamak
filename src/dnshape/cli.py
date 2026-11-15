"""Command line interface: dnshape symmetric | updown | export | plot."""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from pathlib import Path

from dnshape.configs import Config, load_config, save_config
from dnshape.currents import Currents
from dnshape.equilibrium import Equilibrium
from dnshape.gbs_export import gbs_filename, write_h5
from dnshape.grid import Grid
from dnshape.optimize import symmetric, up_down
from dnshape.shape import measure

log = logging.getLogger("dnshape")


def _save_plot(config: Config, path: Path, fields: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dnshape.plotting import plot_equilibrium, plot_fields

    if fields:
        fig = plot_fields(config.currents, config.grid, config.kind)
    else:
        fig = plot_equilibrium(config.currents, config.grid, config.kind).figure
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save(config: Config, stem: Path, plot: bool) -> None:
    """Write stem.json and optionally stem.png, the stem may contain dots."""
    json_path = stem.parent / f"{stem.name}.json"
    save_config(json_path, config)
    log.info("Saved %s", json_path)
    if plot:
        _save_plot(config, stem.parent / f"{stem.name}.png")


def _config(currents, grid, kind, targets, bounds, description) -> Config:
    shape = measure(Equilibrium.for_kind(currents, grid, kind), kind)
    meta = {
        "description": description,
        "targets": targets,
        "xpoint_bounds": list(bounds),
        "shape": shape.to_dict(kind),
    }
    return Config(currents, grid, kind, meta)


def run_symmetric(args: argparse.Namespace) -> None:
    grid, out = Grid(), Path(args.out)
    bounds = tuple(args.xpoint)
    start = symmetric.match_xpoint_bounds(
        symmetric.initial_currents(grid), grid, bounds
    )
    for delta in args.delta:
        t0 = time.time()
        result = symmetric.fit(start, grid, args.kappa, delta, bounds, tol=args.tol)
        final = symmetric.center_legs(result.currents, grid)
        log.info("Fit took %.1f minutes", (time.time() - t0) / 60)

        config = _config(
            final,
            grid,
            "symmetric",
            {"kappa": args.kappa, "delta": delta},
            bounds,
            f"Symmetric double null, kappa {args.kappa}, delta {delta}.",
        )
        _save(config, out / f"delta_{delta}_kappa_{args.kappa}", args.plot)
        if args.gbs:
            write_h5(out / gbs_filename(delta), final, grid)


def run_updown(args: argparse.Namespace) -> None:
    grid, out = Grid(), Path(args.out)
    bounds = tuple(args.xpoint)
    start = load_config(args.start).currents if args.start else None
    start = up_down.initial_currents(grid) if start is None else start

    stage_one = up_down.fit_kappa(start, grid, args.kappa, bounds)
    for d_up, d_down in itertools.product(args.delta_up, args.delta_down):
        if args.method == "two_stage":
            result = up_down.fit_triangularity(
                stage_one.currents, grid, d_up, d_down, bounds, tol=args.tol
            )
        else:
            result = up_down.fit_global(
                stage_one.currents, grid, args.kappa, d_up, d_down, bounds
            )
        config = _config(
            result.currents,
            grid,
            "up_down",
            {"kappa": args.kappa, "delta_up": d_up, "delta_down": d_down},
            bounds,
            f"Up/down asymmetric double null, kappa {args.kappa}, "
            f"delta up {d_up}, delta down {d_down}.",
        )
        name = f"kappa_{args.kappa}_dup_{d_up:+.3f}_ddown_{d_down:+.3f}"
        _save(config, out / name, args.plot)


def run_export(args: argparse.Namespace) -> None:
    for path in args.configs:
        config = load_config(path)
        if config.kind != "symmetric":
            raise SystemExit(
                f"{path}: GBS export is defined for symmetric configurations"
            )
        currents: Currents = config.currents.shifted(args.shift_x)
        delta = config.meta.get("targets", {}).get("delta")
        if delta is None:
            eq = Equilibrium.for_kind(currents, config.grid, config.kind)
            delta = round(measure(eq, config.kind).delta, 2)
        target = Path(args.out) / gbs_filename(delta)
        write_h5(target, currents, config.grid, staggered=not args.no_staggered)
        log.info("Wrote %s", target)


def run_plot(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.out:
        _save_plot(config, Path(args.out), args.fields)
        return
    import matplotlib.pyplot as plt

    from dnshape.plotting import plot_equilibrium, plot_fields

    if args.fields:
        plot_fields(config.currents, config.grid, config.kind)
    else:
        plot_equilibrium(config.currents, config.grid, config.kind)
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dnshape", description="Shape double null equilibria for GBS."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("symmetric", help="fit kappa and one delta, up/down symmetric")
    p.add_argument("--kappa", type=float, required=True)
    p.add_argument("--delta", type=float, nargs="+", required=True)
    p.add_argument(
        "--xpoint", type=float, nargs=2, metavar=("LOW", "HIGH"), required=True
    )
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--out", default="results/symmetric")
    p.add_argument("--gbs", action="store_true", help="also write the GBS HDF5 file")
    p.add_argument("--plot", action="store_true", help="also save a PNG")
    p.set_defaults(func=run_symmetric)

    p = sub.add_parser("updown", help="fit kappa, delta up and delta down")
    p.add_argument("--kappa", type=float, required=True)
    p.add_argument("--delta-up", type=float, nargs="+", required=True)
    p.add_argument("--delta-down", type=float, nargs="+", required=True)
    p.add_argument(
        "--xpoint", type=float, nargs=2, metavar=("LOW", "HIGH"), required=True
    )
    p.add_argument("--method", choices=["two_stage", "global"], default="two_stage")
    p.add_argument("--start", help="configuration file to start from")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--out", default="results/up_down")
    p.add_argument("--plot", action="store_true", help="also save a PNG")
    p.set_defaults(func=run_updown)

    p = sub.add_parser("export", help="write GBS HDF5 files from symmetric configs")
    p.add_argument("configs", nargs="+")
    p.add_argument("--out", default="results/gbs")
    p.add_argument("--shift-x", type=float, default=0.0)
    p.add_argument("--no-staggered", action="store_true")
    p.set_defaults(func=run_export)

    p = sub.add_parser("plot", help="plot a configuration")
    p.add_argument("config")
    p.add_argument("--out", help="PNG file, the plot is shown if omitted")
    p.add_argument("--fields", action="store_true", help="plot the exported fields")
    p.set_defaults(func=run_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )
    args.func(args)
    return 0

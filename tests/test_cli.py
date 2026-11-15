from pathlib import Path

import h5py
import pytest

from dntokamak.cli import build_parser, main

CONFIG = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "symmetric"
    / "delta_-0.4_kappa_1.7.json"
)


def test_export_writes_a_named_file(tmp_path):
    assert main(["export", str(CONFIG), "--out", str(tmp_path), "--no-staggered"]) == 0
    with h5py.File(tmp_path / "Equil_DN_NT_d-0p4.h5") as f:
        assert "dpsidx_n" not in f


def test_export_shift_moves_the_main_current(tmp_path):
    main(["export", str(CONFIG), "--out", str(tmp_path), "--shift-x", "10"])
    with h5py.File(tmp_path / "Equil_DN_NT_d-0p4.h5") as f:
        assert f["xmag1"][0, 0] == pytest.approx(186.0 + 10.0)


def test_export_refuses_up_down_configs(tmp_path):
    up_down = next((CONFIG.parents[1] / "up_down").glob("start_kappa_1.8.json"))
    with pytest.raises(SystemExit):
        main(["export", str(up_down), "--out", str(tmp_path)])


def test_plot_saves_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    target = tmp_path / "fig.png"
    main(["plot", str(CONFIG), "--out", str(target)])
    assert target.stat().st_size > 10_000


def test_parser_requires_a_subcommand_and_targets():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["symmetric", "--kappa", "1.7"])
    args = parser.parse_args(
        [
            "updown",
            "--kappa",
            "1.8",
            "--delta-up",
            "0.1",
            "0.2",
            "--delta-down",
            "0",
            "--xpoint",
            "120",
            "185",
        ]
    )
    assert args.delta_up == [0.1, 0.2] and args.method == "two_stage"

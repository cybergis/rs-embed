import argparse

import pytest

from rs_embed import cli
from rs_embed.core.specs import BBox, PointBuffer


# ══════════════════════════════════════════════════════════════════════
# _parse_bands / _parse_models
# ══════════════════════════════════════════════════════════════════════


def test_parse_bands_basic():
    assert cli._parse_bands("B4, B3, B2") == ("B4", "B3", "B2")


def test_parse_bands_single():
    assert cli._parse_bands("B1") == ("B1",)


def test_parse_bands_no_spaces():
    assert cli._parse_bands("B4,B3,B2") == ("B4", "B3", "B2")


def test_parse_bands_empty_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_bands("")


def test_parse_models_basic():
    assert cli._parse_models("m1, m2") == ["m1", "m2"]


def test_parse_models_single():
    assert cli._parse_models("tessera") == ["tessera"]


# ══════════════════════════════════════════════════════════════════════
# _parse_value_range
# ══════════════════════════════════════════════════════════════════════


def test_parse_value_range_ok():
    assert cli._parse_value_range("1,2") == (1.0, 2.0)


def test_parse_value_range_floats():
    assert cli._parse_value_range("0.0,10000.0") == (0.0, 10000.0)


def test_parse_value_range_none():
    assert cli._parse_value_range(None) is None


def test_parse_value_range_empty():
    assert cli._parse_value_range("") is None


def test_parse_value_range_bad():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_value_range("bad")


def test_parse_value_range_single_number():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_value_range("42")


# ══════════════════════════════════════════════════════════════════════
# _parse_spatial — bbox vs pointbuffer
# ══════════════════════════════════════════════════════════════════════


def test_parse_spatial_bbox():
    args = cli.build_parser().parse_args(
        [
            "inspect-gee",
            "--collection",
            "c",
            "--bands",
            "B1",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
        ]
    )
    spatial = cli._parse_spatial(args)
    assert isinstance(spatial, BBox)
    assert spatial.minlon == 0.0
    assert spatial.maxlon == 1.0


def test_parse_spatial_pointbuffer():
    args = cli.build_parser().parse_args(
        [
            "inspect-gee",
            "--collection",
            "c",
            "--bands",
            "B1",
            "--pointbuffer",
            "1",
            "2",
            "128",
        ]
    )
    spatial = cli._parse_spatial(args)
    assert isinstance(spatial, PointBuffer)
    assert spatial.lon == 1.0
    assert spatial.lat == 2.0
    assert spatial.buffer_m == 128.0


def test_bbox_and_pointbuffer_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "inspect-gee",
                "--collection",
                "c",
                "--bands",
                "B1",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
                "--pointbuffer",
                "1",
                "2",
                "128",
            ]
        )


def test_spatial_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["inspect-gee", "--collection", "c", "--bands", "B1"])


# ══════════════════════════════════════════════════════════════════════
# _parse_temporal — year vs range
# ══════════════════════════════════════════════════════════════════════


def test_parse_temporal_year():
    args = cli.build_parser().parse_args(
        [
            "inspect-gee",
            "--collection",
            "c",
            "--bands",
            "B1",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
            "--year",
            "2024",
        ]
    )
    t = cli._parse_temporal(args)
    assert t is not None
    assert t.mode == "year"
    assert t.year == 2024


def test_parse_temporal_range():
    args = cli.build_parser().parse_args(
        [
            "inspect-gee",
            "--collection",
            "c",
            "--bands",
            "B1",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
            "--range",
            "2022-06-01",
            "2022-09-01",
        ]
    )
    t = cli._parse_temporal(args)
    assert t is not None
    assert t.mode == "range"
    assert t.start == "2022-06-01"


def test_parse_temporal_none():
    args = cli.build_parser().parse_args(
        [
            "inspect-gee",
            "--collection",
            "c",
            "--bands",
            "B1",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
        ]
    )
    t = cli._parse_temporal(args)
    assert t is None


# ══════════════════════════════════════════════════════════════════════
# inspect-gee subcommand defaults
# ══════════════════════════════════════════════════════════════════════


def test_inspect_gee_defaults():
    args = cli.build_parser().parse_args(
        [
            "inspect-gee",
            "--collection",
            "COPERNICUS/S2_SR_HARMONIZED",
            "--bands",
            "B4,B3,B2",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
        ]
    )
    assert args.scale_m == 10
    assert args.cloudy_pct == 30
    assert args.fill_value == 0.0
    assert args.composite == "median"
    assert args.value_range is None
    assert args.save_dir is None


# ══════════════════════════════════════════════════════════════════════
# export-npz subcommand
# ══════════════════════════════════════════════════════════════════════


def test_export_npz_defaults():
    args = cli.build_parser().parse_args(
        [
            "export-npz",
            "--models",
            "tessera",
            "--out",
            "/tmp/out.npz",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
        ]
    )
    assert args.models == ["tessera"]
    assert args.out == "/tmp/out.npz"
    assert args.backend == "gee"
    assert args.device == "auto"
    assert args.output == "pooled"
    assert args.pooling == "mean"
    assert args.no_inputs is False
    assert args.no_embeddings is False
    assert args.no_json is False
    assert args.fail_on_bad_input is False


def test_export_npz_multiple_models():
    args = cli.build_parser().parse_args(
        [
            "export-npz",
            "--models",
            "tessera,remoteclip_s2rgb",
            "--out",
            "/tmp/out.npz",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
        ]
    )
    assert args.models == ["tessera", "remoteclip_s2rgb"]


# ══════════════════════════════════════════════════════════════════════
# missing required args
# ══════════════════════════════════════════════════════════════════════


def test_no_subcommand_fails():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_inspect_gee_missing_collection():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["inspect-gee", "--bands", "B1", "--bbox", "0", "0", "1", "1"]
        )


# ══════════════════════════════════════════════════════════════════════
# _parse_models / _parse_bands — edge cases
# ══════════════════════════════════════════════════════════════════════


def test_parse_models_empty_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_models("")


def test_parse_bands_whitespace():
    assert cli._parse_bands("  B4 , B3 , B2  ") == ("B4", "B3", "B2")


# ══════════════════════════════════════════════════════════════════════
# export-npz flag parsing
# ══════════════════════════════════════════════════════════════════════


def test_export_npz_flag_options():
    args = cli.build_parser().parse_args(
        [
            "export-npz",
            "--models",
            "tessera",
            "--out",
            "/tmp/out.npz",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
            "--no-inputs",
            "--no-embeddings",
            "--no-json",
            "--fail-on-bad-input",
        ]
    )
    assert args.no_inputs is True
    assert args.no_embeddings is True
    assert args.no_json is True
    assert args.fail_on_bad_input is True


def test_export_npz_grid_output():
    args = cli.build_parser().parse_args(
        [
            "export-npz",
            "--models",
            "tessera",
            "--out",
            "/tmp/out.npz",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
            "--output",
            "grid",
        ]
    )
    assert args.output == "grid"


def test_export_npz_custom_backend_device():
    args = cli.build_parser().parse_args(
        [
            "export-npz",
            "--models",
            "tessera",
            "--out",
            "/tmp/out.npz",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
            "--backend",
            "local",
            "--device",
            "cpu",
        ]
    )
    assert args.backend == "local"
    assert args.device == "cpu"


def test_cli_main_export_npz_maps_args(monkeypatch, capsys):
    captured = {}

    def _fake_export_npz(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "export_npz", _fake_export_npz)

    cli.main(
        [
            "export-npz",
            "--models",
            "tessera",
            "--out",
            "/tmp/out.npz",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
            "--collection",
            "COPERNICUS/S2_SR_HARMONIZED",
            "--bands",
            "B4,B3,B2",
            "--no-json",
            "--fail-on-bad-input",
        ]
    )

    assert captured["models"] == ["tessera"]
    assert captured["save_manifest"] is False
    assert captured["fail_on_bad_input"] is True
    assert captured["sensor"] is not None
    assert captured["sensor"].collection == "COPERNICUS/S2_SR_HARMONIZED"
    assert captured["sensor"].bands == ("B4", "B3", "B2")

    stdout = capsys.readouterr().out
    assert '"ok": true' in stdout.lower()


def test_cli_main_export_npz_value_range_rejected():
    with pytest.raises(SystemExit, match="supported only for inspect-gee"):
        cli.main(
            [
                "export-npz",
                "--models",
                "tessera",
                "--out",
                "/tmp/out.npz",
                "--bbox",
                "0",
                "0",
                "1",
                "1",
                "--value-range",
                "0,1",
            ]
        )

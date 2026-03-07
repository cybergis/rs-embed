import os

import numpy as np
import pytest

from rs_embed.tools.inspection import (
    inspect_chw,
    maybe_inspect_chw,
    checks_enabled,
    checks_should_raise,
    checks_save_dir,
    save_quicklook_rgb,
    _safe_float,
    _env_flag,
)
from rs_embed.core.specs import SensorSpec


# ══════════════════════════════════════════════════════════════════════
# _env_flag / _safe_float helpers
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "val,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
    ],
)
def test_env_flag_values(monkeypatch, val, expected):
    monkeypatch.setenv("TEST_FLAG", val)
    assert _env_flag("TEST_FLAG") is expected


def test_safe_float():
    assert _safe_float(3) == 3.0
    assert _safe_float("2.5") == 2.5
    assert _safe_float(None) is None
    assert _safe_float("not-a-number") is None


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — type / shape guards
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_non_array():
    report = inspect_chw("not-array")
    assert report["ok"] is False
    assert any("not a numpy array" in s for s in report["issues"])


def test_inspect_chw_wrong_ndim():
    report = inspect_chw(np.zeros((4, 4), dtype=np.float32))
    assert report["ok"] is False
    assert any("ndim" in s for s in report["issues"])


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — channel mismatch
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_channel_mismatch():
    x = np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)
    report = inspect_chw(x, expected_channels=6)
    assert report["ok"] is False
    assert any("channel mismatch" in s for s in report["issues"])


def test_inspect_chw_channel_match():
    x = np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)
    report = inspect_chw(x, expected_channels=3)
    assert report["ok"] is True


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — NaN / Inf detection
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_nan():
    x = np.ones((1, 4, 4), dtype=np.float32)
    x[0, 0, :] = np.nan
    report = inspect_chw(x)
    assert report["ok"] is False
    assert any("NaN" in s for s in report["issues"])


def test_inspect_chw_inf():
    x = np.ones((1, 4, 4), dtype=np.float32)
    x[0, 0, 0] = np.inf
    report = inspect_chw(x)
    assert report["ok"] is False
    assert report["finite_frac"] < 1.0


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — fill value and value range
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_flags_fill_and_range():
    x = np.zeros((2, 4, 4), dtype=np.float32)
    report = inspect_chw(x, value_range=(1.0, 2.0), fill_value=0.0)
    assert report["ok"] is False
    assert report.get("fill_frac", 0.0) > 0.98
    assert report.get("outside_range_frac", 0.0) > 0.9


def test_inspect_chw_fill_value_below_threshold():
    """fill_frac <= 0.98 should not trigger an issue for fill alone."""
    x = np.random.default_rng(1).random((1, 10, 10)).astype(np.float32)
    # Set ~50% to fill value
    x[0, :5, :] = 0.0
    report = inspect_chw(x, fill_value=0.0)
    fill_issues = [s for s in report["issues"] if "fill_value" in s]
    assert len(fill_issues) == 0


def test_inspect_chw_value_range_ok():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, 1.0, (2, 8, 8)).astype(np.float32)
    report = inspect_chw(x, value_range=(0.0, 1.0))
    range_issues = [s for s in report["issues"] if "outside range" in s]
    assert len(range_issues) == 0


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — constant band detection
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_constant_band():
    x = np.ones((2, 4, 4), dtype=np.float32)
    report = inspect_chw(x)
    assert report["ok"] is False
    assert any("near-constant" in s for s in report["issues"])


def test_inspect_chw_one_constant_one_varying():
    x = np.ones((2, 4, 4), dtype=np.float32)
    x[1] = np.random.default_rng(0).random((4, 4)).astype(np.float32)
    report = inspect_chw(x)
    assert report["ok"] is False
    assert any("near-constant" in s for s in report["issues"])


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — band stats and report fields
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_band_stats():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, (3, 16, 16)).astype(np.float32)
    report = inspect_chw(x)
    assert len(report["band_min"]) == 3
    assert len(report["band_max"]) == 3
    assert len(report["band_mean"]) == 3
    assert len(report["band_std"]) == 3
    for i in range(3):
        assert report["band_min"][i] <= report["band_mean"][i] <= report["band_max"][i]


def test_inspect_chw_has_shape_and_dtype():
    x = np.zeros((1, 2, 2), dtype=np.float64)
    report = inspect_chw(x)
    assert report["shape"] == (1, 2, 2)
    assert "float64" in report["dtype"]


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — downsampling on large arrays
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_downsample():
    rng = np.random.default_rng(7)
    x = rng.random((3, 200, 200)).astype(np.float32)
    report = inspect_chw(x, max_pixels_for_full_stats=1000)
    assert "downsample_stride" in report
    assert report["downsample_stride"] > 1


# ══════════════════════════════════════════════════════════════════════
# maybe_inspect_chw
# ══════════════════════════════════════════════════════════════════════


def test_maybe_inspect_chw_disabled(monkeypatch):
    monkeypatch.delenv("RS_EMBED_CHECK_INPUT", raising=False)
    report = maybe_inspect_chw(np.zeros((1, 2, 2), dtype=np.float32))
    assert report is None


def test_maybe_inspect_chw_enabled_meta(monkeypatch):
    monkeypatch.setenv("RS_EMBED_CHECK_INPUT", "1")
    meta = {}
    report = maybe_inspect_chw(
        np.zeros((1, 2, 2), dtype=np.float32),
        name="test",
        meta=meta,
    )
    assert report is not None
    assert "input_checks" in meta
    assert "test" in meta["input_checks"]
    assert meta.get("input_checks_config", {}).get("enabled") is True


def test_maybe_inspect_chw_sensor_flag():
    """Enable via SensorSpec.check_input without env var."""
    sensor = SensorSpec(
        collection="C",
        bands=("B1",),
        check_input=True,
        check_raise=False,
    )
    meta = {}
    report = maybe_inspect_chw(
        np.zeros((1, 2, 2), dtype=np.float32),
        sensor=sensor,
        meta=meta,
    )
    assert report is not None
    assert meta["input_checks_config"]["raise"] is False


# ══════════════════════════════════════════════════════════════════════
# checks_enabled / checks_should_raise / checks_save_dir
# ══════════════════════════════════════════════════════════════════════


def test_checks_flags_env_override(monkeypatch):
    monkeypatch.setenv("RS_EMBED_CHECK_INPUT", "1")
    monkeypatch.setenv("RS_EMBED_CHECK_RAISE", "0")
    assert checks_enabled() is True
    assert checks_should_raise() is False


def test_checks_should_raise_default_no_env(monkeypatch):
    monkeypatch.delenv("RS_EMBED_CHECK_RAISE", raising=False)
    # No sensor → defaults to True
    assert checks_should_raise() is True


def test_checks_should_raise_sensor_override(monkeypatch):
    monkeypatch.delenv("RS_EMBED_CHECK_RAISE", raising=False)
    sensor = SensorSpec(collection="C", bands=("B1",), check_raise=False)
    assert checks_should_raise(sensor) is False


def test_checks_save_dir_env(monkeypatch):
    monkeypatch.setenv("RS_EMBED_CHECK_SAVE_DIR", "/tmp/rs_embed_checks")
    assert checks_save_dir() == "/tmp/rs_embed_checks"


def test_checks_save_dir_sensor(monkeypatch):
    monkeypatch.delenv("RS_EMBED_CHECK_SAVE_DIR", raising=False)
    sensor = SensorSpec(collection="C", bands=("B1",), check_save_dir="/data/out")
    assert checks_save_dir(sensor) == "/data/out"


def test_checks_save_dir_none(monkeypatch):
    monkeypatch.delenv("RS_EMBED_CHECK_SAVE_DIR", raising=False)
    assert checks_save_dir() is None


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — histogram generation
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_histogram_keys():
    """Inspect report should include histogram bin edges and per-band counts."""
    rng = np.random.default_rng(10)
    x = rng.uniform(0, 10, (2, 16, 16)).astype(np.float32)
    report = inspect_chw(x, hist_bins=8)
    assert "hist_bins" in report
    assert "band_hist" in report
    assert len(report["hist_bins"]) == 9  # edges = bins + 1
    assert len(report["band_hist"]) == 2  # one list of counts per channel


def test_inspect_chw_histogram_with_clip_range():
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 100, (1, 8, 8)).astype(np.float32)
    report = inspect_chw(x, hist_bins=4, hist_clip_range=(10.0, 90.0))
    assert "hist_bins" in report
    assert report["hist_range"] == pytest.approx([10.0, 90.0])


def test_inspect_chw_no_histogram_when_bins_zero():
    rng = np.random.default_rng(12)
    x = rng.random((1, 4, 4)).astype(np.float32)
    report = inspect_chw(x, hist_bins=0)
    assert "hist_bins" not in report
    assert "band_hist" not in report


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — quantiles
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_quantiles_present():
    rng = np.random.default_rng(13)
    x = rng.random((2, 8, 8)).astype(np.float32)
    report = inspect_chw(x, quantiles=(0.25, 0.5, 0.75))
    assert "band_quantiles" in report
    assert "p25" in report["band_quantiles"]
    assert "p50" in report["band_quantiles"]
    assert "p75" in report["band_quantiles"]


def test_inspect_chw_no_quantiles():
    rng = np.random.default_rng(14)
    x = rng.random((1, 4, 4)).astype(np.float32)
    report = inspect_chw(x, quantiles=())
    assert "band_quantiles" not in report


# ══════════════════════════════════════════════════════════════════════
# inspect_chw — zero-size spatial
# ══════════════════════════════════════════════════════════════════════


def test_inspect_chw_zero_spatial():
    x = np.zeros((2, 0, 4), dtype=np.float32)
    report = inspect_chw(x)
    assert report["ok"] is False
    assert any("non-positive" in s for s in report["issues"])


# ══════════════════════════════════════════════════════════════════════
# save_quicklook_rgb
# ══════════════════════════════════════════════════════════════════════


def test_save_quicklook_rgb_creates_file(tmp_path):
    rng = np.random.default_rng(20)
    x = rng.random((3, 16, 16)).astype(np.float32)
    out = str(tmp_path / "ql.png")
    save_quicklook_rgb(x, path=out)
    assert os.path.isfile(out)


def test_save_quicklook_rgb_wrong_ndim():
    with pytest.raises(ValueError, match="Expected CHW"):
        save_quicklook_rgb(np.zeros((4, 4), dtype=np.float32), path="/tmp/bad.png")


def test_save_quicklook_rgb_band_out_of_range():
    x = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="out of range"):
        save_quicklook_rgb(x, path="/tmp/bad.png", bands=(0, 1, 2))


def test_save_quicklook_rgb_supports_vmin_vmax(tmp_path):
    x = np.random.default_rng(21).random((3, 16, 16)).astype(np.float32)
    out = str(tmp_path / "ql_vmin_vmax.png")
    save_quicklook_rgb(x, path=out, vmin=0.0, vmax=1.0)
    assert os.path.isfile(out)

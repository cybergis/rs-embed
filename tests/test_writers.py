"""Tests for rs_embed.writers — format-specific persistence."""

import os

import numpy as np
import pytest
import xarray as xr

from rs_embed.writers import (
    _infer_dims,
    _pick_engine,
    get_extension,
    write_arrays,
)


# ══════════════════════════════════════════════════════════════════════
# get_extension
# ══════════════════════════════════════════════════════════════════════


def test_get_extension_npz():
    assert get_extension("npz") == ".npz"


def test_get_extension_netcdf():
    assert get_extension("netcdf") == ".nc"


def test_get_extension_unknown():
    with pytest.raises(ValueError, match="Unknown format"):
        get_extension("parquet")


# ══════════════════════════════════════════════════════════════════════
# _infer_dims
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "key, shape, expected",
    [
        ("input_chw__model_a", (3, 64, 64), ("band", "y", "x")),
        ("inputs_bchw__model_a", (10, 3, 64, 64), ("point", "band", "y", "x")),
        ("embedding__model_a", (768,), ("dim",)),
        ("embedding__model_a", (3, 8, 8), ("band", "y", "x")),
        ("embeddings__model_a", (10, 768), ("point", "dim")),
        ("embeddings__model_a", (10, 3, 8, 8), ("point", "band", "y", "x")),
        ("some_other_key", (2, 3), ("d0", "d1")),
    ],
)
def test_infer_dims(key, shape, expected):
    arr = np.zeros(shape, dtype=np.float32)
    assert _infer_dims(key, arr) == expected


# ══════════════════════════════════════════════════════════════════════
# _pick_engine
# ══════════════════════════════════════════════════════════════════════


def test_pick_engine_returns_string():
    """At least scipy is available in the test environment."""
    engine = _pick_engine()
    assert engine in ("netcdf4", "h5netcdf", "scipy")


# ══════════════════════════════════════════════════════════════════════
# write_arrays — NPZ
# ══════════════════════════════════════════════════════════════════════


def test_write_npz_creates_files(tmp_path):
    arrays = {"embedding__m": np.arange(8, dtype=np.float32)}
    manifest = {"created_at": "2025-01-01T00:00:00Z", "backend": "gee", "device": "cpu"}

    out = str(tmp_path / "test.npz")
    result = write_arrays(
        fmt="npz", out_path=out, arrays=arrays, manifest=manifest, save_manifest=True
    )

    assert os.path.isfile(out)
    assert os.path.isfile(str(tmp_path / "test.json"))
    assert "npz_path" in result
    assert "npz_keys" in result
    assert result["npz_keys"] == ["embedding__m"]

    # verify array round-trips
    loaded = np.load(out)
    np.testing.assert_array_equal(loaded["embedding__m"], arrays["embedding__m"])


def test_write_npz_no_manifest(tmp_path):
    arrays = {"x": np.zeros(4, dtype=np.float32)}
    out = str(tmp_path / "t.npz")
    result = write_arrays(fmt="npz", out_path=out, arrays=arrays, manifest={}, save_manifest=False)
    assert os.path.isfile(out)
    assert not os.path.isfile(str(tmp_path / "t.json"))
    assert "manifest_path" not in result


def test_write_npz_appends_extension(tmp_path):
    out = str(tmp_path / "noext")
    write_arrays(
        fmt="npz",
        out_path=out,
        arrays={"a": np.zeros(1)},
        manifest={},
        save_manifest=False,
    )
    assert os.path.isfile(out + ".npz")


# ══════════════════════════════════════════════════════════════════════
# write_arrays — NetCDF
# ══════════════════════════════════════════════════════════════════════


def test_write_netcdf_creates_files(tmp_path):
    arrays = {
        "input_chw__model_a": np.random.rand(3, 4, 4).astype(np.float32),
        "embedding__model_a": np.random.rand(8).astype(np.float32),
    }
    manifest = {"created_at": "2025-01-01T00:00:00Z", "backend": "gee", "device": "cpu"}

    out = str(tmp_path / "test.nc")
    result = write_arrays(
        fmt="netcdf", out_path=out, arrays=arrays, manifest=manifest, save_manifest=True
    )

    assert os.path.isfile(out)
    assert os.path.isfile(str(tmp_path / "test.json"))
    assert "nc_path" in result
    assert "nc_variables" in result
    assert set(result["nc_variables"]) == {"embedding__model_a", "input_chw__model_a"}


def test_write_netcdf_roundtrip(tmp_path):
    """Verify arrays survive a write → read cycle with correct dimensions."""
    emb = np.arange(16, dtype=np.float32)
    inp = np.ones((3, 8, 8), dtype=np.float32)
    arrays = {
        "embedding__mx": emb,
        "input_chw__mx": inp,
    }
    out = str(tmp_path / "rt.nc")
    write_arrays(fmt="netcdf", out_path=out, arrays=arrays, manifest={}, save_manifest=False)

    ds = xr.open_dataset(out)
    np.testing.assert_array_almost_equal(ds["embedding__mx"].values, emb)
    np.testing.assert_array_almost_equal(ds["input_chw__mx"].values, inp)
    assert tuple(ds["embedding__mx"].dims) == ("dim",)
    assert tuple(ds["input_chw__mx"].dims) == ("band", "y", "x")
    ds.close()


def test_write_netcdf_global_attrs(tmp_path):
    manifest = {
        "created_at": "2025-06-01T12:00:00Z",
        "backend": "gee",
        "device": "auto",
    }
    out = str(tmp_path / "attrs.nc")
    write_arrays(
        fmt="netcdf",
        out_path=out,
        arrays={"embedding__m": np.zeros(4, dtype=np.float32)},
        manifest=manifest,
        save_manifest=False,
    )
    ds = xr.open_dataset(out)
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert ds.attrs["backend"] == "gee"
    assert ds.attrs["created_at"] == "2025-06-01T12:00:00Z"
    ds.close()


def test_write_netcdf_no_manifest(tmp_path):
    out = str(tmp_path / "nm.nc")
    result = write_arrays(
        fmt="netcdf",
        out_path=out,
        arrays={"embedding__m": np.zeros(4, dtype=np.float32)},
        manifest={},
        save_manifest=False,
    )
    assert os.path.isfile(out)
    assert not os.path.isfile(str(tmp_path / "nm.json"))
    assert "manifest_path" not in result


def test_write_netcdf_appends_extension(tmp_path):
    out = str(tmp_path / "noext")
    write_arrays(
        fmt="netcdf",
        out_path=out,
        arrays={"embedding__m": np.zeros(4, dtype=np.float32)},
        manifest={},
        save_manifest=False,
    )
    assert os.path.isfile(out + ".nc")


def test_write_netcdf_batch_embeddings(tmp_path):
    """Combined batch: (point, dim) embeddings round-trip with correct shape."""
    batch = np.random.rand(5, 32).astype(np.float32)
    arrays = {"embeddings__model_a": batch}
    out = str(tmp_path / "batch.nc")
    write_arrays(fmt="netcdf", out_path=out, arrays=arrays, manifest={}, save_manifest=False)

    ds = xr.open_dataset(out)
    np.testing.assert_array_almost_equal(ds["embeddings__model_a"].values, batch)
    assert tuple(ds["embeddings__model_a"].dims) == ("point", "dim")
    ds.close()


def test_write_netcdf_resolves_dim_name_conflicts(tmp_path):
    """Variables with different lengths should not be forced onto one shared dim."""
    arrays = {
        "embedding__model_a": np.zeros((8,), dtype=np.float32),
        "embedding__model_b": np.zeros((4,), dtype=np.float32),
        "inputs_bchw__model_c": np.zeros((2, 3, 4, 4), dtype=np.float32),
        "inputs_bchw__model_d": np.zeros((2, 5, 4, 4), dtype=np.float32),
    }
    out = str(tmp_path / "conflict.nc")
    write_arrays(fmt="netcdf", out_path=out, arrays=arrays, manifest={}, save_manifest=False)

    ds = xr.open_dataset(out)
    try:
        assert tuple(ds["embedding__model_a"].dims) == ("dim",)
        dim_b = ds["embedding__model_b"].dims[0]
        assert dim_b != "dim"
        assert dim_b.startswith("dim__")
        assert int(ds.sizes[dim_b]) == 4

        assert tuple(ds["inputs_bchw__model_c"].dims) == ("point", "band", "y", "x")
        band_d = ds["inputs_bchw__model_d"].dims[1]
        assert band_d != "band"
        assert band_d.startswith("band__")
        assert int(ds.sizes[band_d]) == 5
    finally:
        ds.close()


# ══════════════════════════════════════════════════════════════════════
# write_arrays — unknown format
# ══════════════════════════════════════════════════════════════════════


def test_write_arrays_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        write_arrays(
            fmt="csv",
            out_path=str(tmp_path / "x"),
            arrays={},
            manifest={},
            save_manifest=False,
        )

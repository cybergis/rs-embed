"""Tests for rs_embed.load — load_export() reader API."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rs_embed.load import (
    ModelResult,
    _build_combined_model_result,
    _find_per_item_files,
    _load_arrays,
    _resolve_combined_paths,
    _stack_per_item_embeddings,
    load_export,
)

# ══════════════════════════════════════════════════════════════════════
# Fixtures / helpers
# ══════════════════════════════════════════════════════════════════════

N_ITEMS = 3
N_DIM = 16
N_BANDS = 4
IMG_SIZE = 8


def _make_embeddings(n: int = N_ITEMS, dim: int = N_DIM) -> np.ndarray:
    return np.random.rand(n, dim).astype(np.float32)


def _make_inputs(n: int = N_ITEMS, c: int = N_BANDS, h: int = IMG_SIZE) -> np.ndarray:
    return np.random.rand(n, c, h, h).astype(np.float32)


def _combined_npz(
    directory: str,
    stem: str = "run",
    *,
    model: str = "remoteclip",
    save_inputs: bool = False,
    status: str = "ok",
) -> tuple[str, str]:
    """Write a minimal combined .npz + .json and return (npz_path, json_path)."""
    from rs_embed.tools.serialization import sanitize_key

    key = sanitize_key(model)
    embs = _make_embeddings()
    arrays: dict[str, np.ndarray] = {f"embeddings__{key}": embs}
    if save_inputs:
        arrays[f"inputs_bchw__{key}"] = _make_inputs()

    npz_path = os.path.join(directory, f"{stem}.npz")
    np.savez(npz_path, **arrays)

    metas = [{"sha1": f"abc{i}"} for i in range(N_ITEMS)]
    manifest = {
        "n_items": N_ITEMS,
        "status": status,
        "spatials": [
            {"type": "PointBuffer", "lon": 121.5 + i, "lat": 31.2} for i in range(N_ITEMS)
        ],
        "temporal": {"start": "2022-06-01", "end": "2022-09-01"},
        "models": [
            {
                "model": model,
                "status": status,
                "embeddings": {"npz_key": f"embeddings__{key}"},
                **({"inputs": {"npz_key": f"inputs_bchw__{key}"}} if save_inputs else {}),
                "metas": metas,
            }
        ],
    }
    json_path = os.path.join(directory, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(manifest, f)

    return npz_path, json_path


def _per_item_dir(
    directory: str,
    n: int = N_ITEMS,
    model: str = "remoteclip",
    save_inputs: bool = False,
) -> str:
    """Write n per-item .npz + .json files and return the directory."""
    from rs_embed.tools.serialization import sanitize_key

    key = sanitize_key(model)
    for i in range(n):
        emb = np.random.rand(N_DIM).astype(np.float32)
        arrays: dict[str, np.ndarray] = {f"embedding__{key}": emb}
        if save_inputs:
            arrays[f"input_chw__{key}"] = np.random.rand(N_BANDS, IMG_SIZE, IMG_SIZE).astype(
                np.float32
            )

        npz_path = os.path.join(directory, f"p{i:05d}.npz")
        np.savez(npz_path, **arrays)

        manifest = {
            "spatial": {"type": "PointBuffer", "lon": 121.5 + i, "lat": 31.2},
            "temporal": {"start": "2022-06-01", "end": "2022-09-01"},
            "models": [
                {
                    "model": model,
                    "status": "ok",
                    "embedding": {"npz_key": f"embedding__{key}"},
                    **({"input": {"npz_key": f"input_chw__{key}"}} if save_inputs else {}),
                    "meta": {"sha1": f"abc{i}"},
                }
            ],
        }
        json_path = os.path.join(directory, f"p{i:05d}.json")
        with open(json_path, "w") as f:
            json.dump(manifest, f)

    return directory


# ══════════════════════════════════════════════════════════════════════
# _resolve_combined_paths
# ══════════════════════════════════════════════════════════════════════


class TestResolveCombinedPaths:
    def test_npz_resolves(self, tmp_path):
        npz = tmp_path / "run.npz"
        npz.touch()
        js = tmp_path / "run.json"
        js.touch()
        arrays_p, json_p, fmt = _resolve_combined_paths(str(npz))
        assert arrays_p == str(npz)
        assert json_p == str(js)
        assert fmt == "npz"

    def test_nc_resolves(self, tmp_path):
        nc = tmp_path / "run.nc"
        nc.touch()
        js = tmp_path / "run.json"
        js.touch()
        arrays_p, json_p, fmt = _resolve_combined_paths(str(nc))
        assert fmt == "netcdf"

    def test_json_finds_npz(self, tmp_path):
        js = tmp_path / "run.json"
        js.touch()
        npz = tmp_path / "run.npz"
        npz.touch()
        arrays_p, json_p, fmt = _resolve_combined_paths(str(js))
        assert arrays_p == str(npz)
        assert fmt == "npz"

    def test_json_finds_nc_when_no_npz(self, tmp_path):
        js = tmp_path / "run.json"
        js.touch()
        nc = tmp_path / "run.nc"
        nc.touch()
        arrays_p, json_p, fmt = _resolve_combined_paths(str(js))
        assert fmt == "netcdf"

    def test_json_missing_array_raises(self, tmp_path):
        js = tmp_path / "run.json"
        js.touch()
        with pytest.raises(FileNotFoundError, match="paired"):
            _resolve_combined_paths(str(js))

    def test_npz_missing_json_raises(self, tmp_path):
        npz = tmp_path / "run.npz"
        npz.touch()
        with pytest.raises(FileNotFoundError, match="manifest"):
            _resolve_combined_paths(str(npz))

    def test_unknown_extension_raises(self, tmp_path):
        f = tmp_path / "run.csv"
        f.touch()
        with pytest.raises(ValueError, match="extension"):
            _resolve_combined_paths(str(f))


# ══════════════════════════════════════════════════════════════════════
# _load_arrays
# ══════════════════════════════════════════════════════════════════════


class TestLoadArrays:
    def test_npz_round_trip(self, tmp_path):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        path = str(tmp_path / "data.npz")
        np.savez(path, foo=arr)
        loaded = _load_arrays(path, "npz")
        assert "foo" in loaded
        np.testing.assert_array_equal(loaded["foo"], arr)

    def test_unknown_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="format"):
            _load_arrays(str(tmp_path / "x.bin"), "bin")


# ══════════════════════════════════════════════════════════════════════
# _stack_per_item_embeddings
# ══════════════════════════════════════════════════════════════════════


class TestStackPerItemEmbeddings:
    def test_all_present(self):
        arrs = [np.ones((4,), dtype=np.float32) * i for i in range(3)]
        stacked = _stack_per_item_embeddings(arrs)
        assert stacked.shape == (3, 4)
        np.testing.assert_array_equal(stacked[1], np.ones(4) * 1)

    def test_all_none_returns_none(self):
        assert _stack_per_item_embeddings([None, None]) is None

    def test_nan_fill_for_failed_point(self):
        arrs = [np.ones((4,), dtype=np.float32), None, np.ones((4,), dtype=np.float32) * 2]
        stacked = _stack_per_item_embeddings(arrs)
        assert stacked.shape == (3, 4)
        assert np.all(np.isnan(stacked[1]))
        np.testing.assert_array_almost_equal(stacked[2], np.full(4, 2.0))

    def test_output_dtype_is_float32(self):
        arrs = [np.ones((8,), dtype=np.float64)]
        stacked = _stack_per_item_embeddings(arrs)
        assert stacked.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════
# _build_combined_model_result
# ══════════════════════════════════════════════════════════════════════


class TestBuildCombinedModelResult:
    def _make_entry_and_arrays(self, model: str = "remoteclip", status: str = "ok"):
        from rs_embed.tools.serialization import sanitize_key

        key = sanitize_key(model)
        embs = _make_embeddings()
        arrays = {f"embeddings__{key}": embs}
        entry = {
            "model": model,
            "status": status,
            "embeddings": {"npz_key": f"embeddings__{key}"},
            "metas": [{"sha1": f"x{i}"} for i in range(N_ITEMS)],
        }
        return entry, arrays

    def test_basic(self):
        entry, arrays = self._make_entry_and_arrays()
        result = _build_combined_model_result(entry, arrays, N_ITEMS)
        assert result.name == "remoteclip"
        assert result.status == "ok"
        assert result.embeddings is not None
        assert result.embeddings.shape == (N_ITEMS, N_DIM)
        assert result.inputs is None
        assert len(result.meta) == N_ITEMS

    def test_inputs_loaded_when_present(self):
        from rs_embed.tools.serialization import sanitize_key

        model = "remoteclip"
        key = sanitize_key(model)
        embs = _make_embeddings()
        inps = _make_inputs()
        arrays = {f"embeddings__{key}": embs, f"inputs_bchw__{key}": inps}
        entry = {
            "model": model,
            "status": "ok",
            "embeddings": {"npz_key": f"embeddings__{key}"},
            "inputs": {"npz_key": f"inputs_bchw__{key}"},
            "metas": [{} for _ in range(N_ITEMS)],
        }
        result = _build_combined_model_result(entry, arrays, N_ITEMS)
        assert result.inputs is not None
        assert result.inputs.shape == (N_ITEMS, N_BANDS, IMG_SIZE, IMG_SIZE)

    def test_failed_status_propagated(self):
        entry, arrays = self._make_entry_and_arrays(status="failed")
        # Remove embeddings to simulate failed run
        result = _build_combined_model_result(entry, {}, N_ITEMS)
        assert result.status == "failed"
        assert result.embeddings is None

    def test_meta_padded_when_short(self):
        from rs_embed.tools.serialization import sanitize_key

        model = "prithvi"
        key = sanitize_key(model)
        embs = _make_embeddings()
        arrays = {f"embeddings__{key}": embs}
        entry = {
            "model": model,
            "status": "ok",
            "embeddings": {"npz_key": f"embeddings__{key}"},
            "metas": [{"sha1": "abc"}],  # only 1, but n_items=3
        }
        result = _build_combined_model_result(entry, arrays, N_ITEMS)
        assert len(result.meta) == N_ITEMS
        assert result.meta[1] == {}


# ══════════════════════════════════════════════════════════════════════
# _find_per_item_files
# ══════════════════════════════════════════════════════════════════════


class TestFindPerItemFiles:
    def test_finds_and_sorts(self, tmp_path):
        d = str(tmp_path)
        for i in [2, 0, 1]:
            (tmp_path / f"p{i:05d}.npz").touch()
            (tmp_path / f"p{i:05d}.json").touch()
        files = _find_per_item_files(d)
        indices = [int(os.path.basename(a).lstrip("p").split(".")[0]) for a, _, _, _ in files]
        assert indices == [0, 1, 2]

    def test_skips_orphaned_npz(self, tmp_path):
        (tmp_path / "p00000.npz").touch()
        # no p00000.json
        with pytest.raises(ValueError, match="No per-item"):
            _find_per_item_files(str(tmp_path))

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No per-item"):
            _find_per_item_files(str(tmp_path))

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "p00000.npz").touch()
        (tmp_path / "p00000.json").touch()
        (tmp_path / "README.md").touch()
        (tmp_path / "other.npz").touch()
        files = _find_per_item_files(str(tmp_path))
        assert len(files) == 1


# ══════════════════════════════════════════════════════════════════════
# load_export — path routing
# ══════════════════════════════════════════════════════════════════════


class TestLoadExportRouting:
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_export(str(tmp_path / "nonexistent.npz"))

    def test_directory_routes_to_per_item(self, tmp_path):
        _per_item_dir(str(tmp_path))
        result = load_export(str(tmp_path))
        assert result.layout == "per_item"

    def test_npz_file_routes_to_combined(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path))
        result = load_export(npz_path)
        assert result.layout == "combined"

    def test_json_file_routes_to_combined(self, tmp_path):
        _, json_path = _combined_npz(str(tmp_path))
        result = load_export(json_path)
        assert result.layout == "combined"


# ══════════════════════════════════════════════════════════════════════
# load_export — combined layout
# ══════════════════════════════════════════════════════════════════════


class TestLoadCombined:
    def test_basic_combined(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path))
        result = load_export(npz_path)
        assert result.layout == "combined"
        assert result.n_items == N_ITEMS
        assert result.status == "ok"
        assert len(result.spatials) == N_ITEMS
        assert result.temporal is not None

    def test_models_loaded(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path))
        result = load_export(npz_path)
        assert "remoteclip" in result.models
        mr = result.models["remoteclip"]
        assert mr.status == "ok"
        assert mr.embeddings.shape == (N_ITEMS, N_DIM)

    def test_ok_models_property(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path))
        result = load_export(npz_path)
        assert result.ok_models == ["remoteclip"]
        assert result.failed_models == []

    def test_embedding_accessor(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path))
        result = load_export(npz_path)
        emb = result.embedding("remoteclip")
        assert emb.shape == (N_ITEMS, N_DIM)

    def test_embedding_unknown_model_raises_keyerror(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path))
        result = load_export(npz_path)
        with pytest.raises(KeyError, match="prithvi"):
            result.embedding("prithvi")

    def test_inputs_loaded_when_saved(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path), save_inputs=True)
        result = load_export(npz_path)
        mr = result.models["remoteclip"]
        assert mr.inputs is not None
        assert mr.inputs.shape == (N_ITEMS, N_BANDS, IMG_SIZE, IMG_SIZE)

    def test_inputs_none_when_not_saved(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path), save_inputs=False)
        result = load_export(npz_path)
        assert result.models["remoteclip"].inputs is None

    def test_manifest_preserved(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path))
        result = load_export(npz_path)
        assert "n_items" in result.manifest

    def test_two_models(self, tmp_path):
        from rs_embed.tools.serialization import sanitize_key

        models = ["remoteclip", "prithvi"]
        arrays: dict[str, np.ndarray] = {}
        model_entries = []
        for m in models:
            key = sanitize_key(m)
            embs = _make_embeddings()
            arrays[f"embeddings__{key}"] = embs
            model_entries.append(
                {
                    "model": m,
                    "status": "ok",
                    "embeddings": {"npz_key": f"embeddings__{key}"},
                    "metas": [{} for _ in range(N_ITEMS)],
                }
            )

        npz_path = str(tmp_path / "run.npz")
        np.savez(npz_path, **arrays)
        manifest = {
            "n_items": N_ITEMS,
            "status": "ok",
            "spatials": [{} for _ in range(N_ITEMS)],
            "temporal": None,
            "models": model_entries,
        }
        with open(str(tmp_path / "run.json"), "w") as f:
            json.dump(manifest, f)

        result = load_export(npz_path)
        assert set(result.models) == {"remoteclip", "prithvi"}
        assert sorted(result.ok_models) == ["prithvi", "remoteclip"]


# ══════════════════════════════════════════════════════════════════════
# load_export — per-item layout
# ══════════════════════════════════════════════════════════════════════


class TestLoadPerItem:
    def test_basic_per_item(self, tmp_path):
        _per_item_dir(str(tmp_path))
        result = load_export(str(tmp_path))
        assert result.layout == "per_item"
        assert result.n_items == N_ITEMS
        assert result.status == "ok"

    def test_embeddings_stacked(self, tmp_path):
        _per_item_dir(str(tmp_path))
        result = load_export(str(tmp_path))
        mr = result.models["remoteclip"]
        assert mr.embeddings.shape == (N_ITEMS, N_DIM)

    def test_inputs_loaded_when_saved(self, tmp_path):
        _per_item_dir(str(tmp_path), save_inputs=True)
        result = load_export(str(tmp_path))
        mr = result.models["remoteclip"]
        assert mr.inputs is not None
        assert mr.inputs.shape == (N_ITEMS, N_BANDS, IMG_SIZE, IMG_SIZE)

    def test_spatials_collected(self, tmp_path):
        _per_item_dir(str(tmp_path))
        result = load_export(str(tmp_path))
        assert len(result.spatials) == N_ITEMS
        assert result.spatials[0]["type"] == "PointBuffer"

    def test_temporal_from_first_item(self, tmp_path):
        _per_item_dir(str(tmp_path))
        result = load_export(str(tmp_path))
        assert result.temporal is not None
        assert result.temporal["start"] == "2022-06-01"

    def test_partial_failure_nan_fill(self, tmp_path):
        """Failed middle point is NaN-filled in the stacked array."""
        from rs_embed.tools.serialization import sanitize_key

        model = "remoteclip"
        key = sanitize_key(model)
        d = str(tmp_path)

        for i in range(N_ITEMS):
            is_failed = i == 1
            emb = np.ones(N_DIM, dtype=np.float32) * i
            arrays: dict[str, np.ndarray] = {}
            if not is_failed:
                arrays[f"embedding__{key}"] = emb
            npz_path = os.path.join(d, f"p{i:05d}.npz")
            np.savez(npz_path, **arrays)

            manifest = {
                "spatial": {"type": "PointBuffer", "lon": 121.5 + i, "lat": 31.2},
                "temporal": None,
                "models": [
                    {
                        "model": model,
                        "status": "failed" if is_failed else "ok",
                        "embedding": {"npz_key": f"embedding__{key}"},
                        "meta": {},
                    }
                ],
            }
            json_path = os.path.join(d, f"p{i:05d}.json")
            with open(json_path, "w") as f:
                json.dump(manifest, f)

        result = load_export(d)
        mr = result.models[model]
        assert mr.status == "partial"
        assert mr.embeddings is not None
        assert np.all(np.isnan(mr.embeddings[1]))
        np.testing.assert_array_equal(mr.embeddings[0], np.zeros(N_DIM))

    def test_all_failed_model_has_none_embeddings(self, tmp_path):
        model = "remoteclip"
        d = str(tmp_path)

        for i in range(N_ITEMS):
            npz_path = os.path.join(d, f"p{i:05d}.npz")
            np.savez(npz_path)  # no embedding arrays
            manifest = {
                "spatial": {},
                "temporal": None,
                "models": [{"model": model, "status": "failed", "embedding": {}, "meta": {}}],
            }
            json_path = os.path.join(d, f"p{i:05d}.json")
            with open(json_path, "w") as f:
                json.dump(manifest, f)

        result = load_export(d)
        mr = result.models[model]
        assert mr.status == "failed"
        assert mr.embeddings is None

    def test_ok_and_failed_models_properties(self, tmp_path):
        from rs_embed.tools.serialization import sanitize_key

        d = str(tmp_path)
        for model, ok in [("remoteclip", True), ("prithvi", False)]:
            key = sanitize_key(model)
            for i in range(N_ITEMS):
                arrays: dict[str, np.ndarray] = {}
                if ok:
                    arrays[f"embedding__{key}"] = np.ones(N_DIM, dtype=np.float32)
                npz_path = os.path.join(d, f"p{i:05d}.npz")
                # Need to merge arrays across models
                if os.path.exists(npz_path):
                    with np.load(npz_path) as existing:
                        existing_arrays = dict(existing)
                    arrays.update(existing_arrays)
                np.savez(npz_path, **arrays)

                json_path = os.path.join(d, f"p{i:05d}.json")
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        manifest = json.load(f)
                    manifest["models"].append(
                        {
                            "model": model,
                            "status": "ok" if ok else "failed",
                            "embedding": {"npz_key": f"embedding__{key}"},
                            "meta": {},
                        }
                    )
                else:
                    manifest = {
                        "spatial": {},
                        "temporal": None,
                        "models": [
                            {
                                "model": model,
                                "status": "ok" if ok else "failed",
                                "embedding": {"npz_key": f"embedding__{key}"},
                                "meta": {},
                            }
                        ],
                    }
                with open(json_path, "w") as f:
                    json.dump(manifest, f)

        result = load_export(d)
        assert result.ok_models == ["remoteclip"]
        assert result.failed_models == ["prithvi"]


# ══════════════════════════════════════════════════════════════════════
# ExportResult — error cases
# ══════════════════════════════════════════════════════════════════════


class TestExportResultErrors:
    def test_embedding_failed_model_raises_valueerror(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path), status="failed")
        result = load_export(npz_path)
        # status=failed means no embeddings array
        mr_failed = ModelResult(
            name="remoteclip",
            status="failed",
            embeddings=None,
            inputs=None,
            meta=[],
            error="fetch error",
        )
        result.models["remoteclip"] = mr_failed
        with pytest.raises(ValueError, match="no embeddings"):
            result.embedding("remoteclip")

    def test_embedding_unknown_model_raises_keyerror(self, tmp_path):
        npz_path, _ = _combined_npz(str(tmp_path))
        result = load_export(npz_path)
        with pytest.raises(KeyError):
            result.embedding("no_such_model")


# ══════════════════════════════════════════════════════════════════════
# Public API surface
# ══════════════════════════════════════════════════════════════════════


def test_top_level_exports():
    """load_export, ExportResult, ModelResult are importable from rs_embed."""
    import rs_embed

    assert hasattr(rs_embed, "load_export")
    assert hasattr(rs_embed, "ExportResult")
    assert hasattr(rs_embed, "ModelResult")
    assert rs_embed.load_export is load_export


class TestRaggedAndAlignment:
    def test_combined_ragged_partial_loads_with_nan_rows(self, tmp_path):
        """Partial combined exports (npz_keys/indices) must load NaN-filled.

        Regression: the loader only understood dense npz_key references, so
        the ragged format the exporter writes on partial success loaded as
        embeddings=None, contradicting the documented NaN-fill promise.
        """
        d = str(tmp_path)
        arrays = {
            "embedding__m__00000": np.array([1.0, 2.0], dtype=np.float32),
            "embedding__m__00002": np.array([5.0, 6.0], dtype=np.float32),
        }
        npz_path = os.path.join(d, "run.npz")
        np.savez(npz_path, **arrays)
        manifest = {
            "n_items": 3,
            "status": "partial",
            "spatials": [{}, {}, {}],
            "models": [
                {
                    "model": "m",
                    "status": "partial",
                    "embeddings": {
                        "npz_keys": ["embedding__m__00000", "embedding__m__00002"],
                        "indices": [0, 2],
                    },
                    "metas": [{}, {}, {}],
                }
            ],
        }
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump(manifest, f)

        result = load_export(npz_path)
        embs = result.models["m"].embeddings
        assert embs is not None and embs.shape == (3, 2)
        np.testing.assert_allclose(embs[0], [1.0, 2.0])
        assert np.isnan(embs[1]).all()  # missing point -> NaN row
        np.testing.assert_allclose(embs[2], [5.0, 6.0])

    def test_per_item_custom_names_load(self, tmp_path):
        """target.names exports must be loadable (not just p<digits>)."""
        d = str(tmp_path)
        for i, name in enumerate(["siteA", "siteB"]):
            np.savez(
                os.path.join(d, f"{name}.npz"),
                embedding__m=np.array([float(i)], dtype=np.float32),
            )
            with open(os.path.join(d, f"{name}.json"), "w") as f:
                json.dump(
                    {
                        "point_index": i,
                        "status": "ok",
                        "models": [
                            {"model": "m", "status": "ok", "embedding": {"npz_key": "embedding__m"}}
                        ],
                    },
                    f,
                )
        result = load_export(d)
        assert result.n_items == 2
        np.testing.assert_allclose(
            result.models["m"].embeddings, np.array([[0.0], [1.0]], dtype=np.float32)
        )

    def test_per_item_missing_point_keeps_row_alignment(self, tmp_path):
        """A failed point (no file under continue_on_error) must leave a NaN
        row at its index, not densely renumber the survivors.

        Regression: row i of the result no longer corresponded to the
        caller's spatials[i] when any point failed.
        """
        d = str(tmp_path)
        for i in (0, 2):  # point 1 failed and wrote no file
            np.savez(
                os.path.join(d, f"p{i:05d}.npz"),
                embedding__m=np.array([float(i)], dtype=np.float32),
            )
            with open(os.path.join(d, f"p{i:05d}.json"), "w") as f:
                json.dump(
                    {
                        "point_index": i,
                        "status": "ok",
                        "models": [
                            {"model": "m", "status": "ok", "embedding": {"npz_key": "embedding__m"}}
                        ],
                    },
                    f,
                )
        result = load_export(d)
        assert result.n_items == 3
        embs = result.models["m"].embeddings
        np.testing.assert_allclose(embs[0], [0.0])
        assert np.isnan(embs[1]).all()
        np.testing.assert_allclose(embs[2], [2.0])


class TestPerItemFiltering:
    @staticmethod
    def _write_point(d, i):
        np.savez(
            os.path.join(d, f"p{i:05d}.npz"),
            embedding__m=np.array([float(i)], dtype=np.float32),
        )
        with open(os.path.join(d, f"p{i:05d}.json"), "w") as f:
            json.dump(
                {
                    "point_index": i,
                    "status": "ok",
                    "models": [
                        {"model": "m", "status": "ok", "embedding": {"npz_key": "embedding__m"}}
                    ],
                },
                f,
            )

    def test_ignores_non_point_artifacts_in_directory(self, tmp_path):
        """A combined run.npz/run.json dropped into a per-item dir must not
        be ingested as a point (it has no point_index and no p<digits> name)."""
        d = str(tmp_path)
        for i in (0, 1):
            self._write_point(d, i)
        np.savez(os.path.join(d, "run.npz"), embeddings__m=np.zeros((5, 2), dtype=np.float32))
        with open(os.path.join(d, "run.json"), "w") as f:
            json.dump({"n_items": 5, "status": "ok", "models": []}, f)

        result = load_export(d)
        assert result.n_items == 2
        np.testing.assert_allclose(
            result.models["m"].embeddings, np.array([[0.0], [1.0]], dtype=np.float32)
        )

    def test_duplicate_point_index_raises(self, tmp_path):
        """Two files claiming the same point_index must fail loudly, not
        silently overwrite a row."""
        d = str(tmp_path)
        self._write_point(d, 0)
        # A differently-named file claiming the same index.
        np.savez(
            os.path.join(d, "stale.npz"),
            embedding__m=np.array([9.0], dtype=np.float32),
        )
        with open(os.path.join(d, "stale.json"), "w") as f:
            json.dump({"point_index": 0, "status": "ok", "models": []}, f)

        with pytest.raises(ValueError, match="both claim point_index=0"):
            load_export(d)

"""Tests for the public API (get_embedding, get_embeddings_batch, export_batch).

These use a mock embedder registered in the test so they don't require
GEE, torch, or any real model weights.
"""

import numpy as np
import pytest

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.errors import ModelError
from rs_embed.core.specs import PointBuffer, TemporalSpec, OutputSpec, SensorSpec
from rs_embed.embedders.base import EmbedderBase


# ── mock embedder ──────────────────────────────────────────────────


class _MockEmbedder(EmbedderBase):
    """Returns a deterministic embedding without any I/O."""

    def describe(self):
        return {"type": "mock", "dim": 8}

    def get_embedding(
        self,
        *,
        spatial,
        temporal,
        sensor,
        output,
        backend,
        device="auto",
        input_chw=None,
    ):
        vec = np.arange(8, dtype=np.float32)
        return Embedding(
            data=vec, meta={"model": self.model_name, "output": output.mode}
        )


class _MockPrecomputedLocalEmbedder(EmbedderBase):
    def describe(self):
        return {
            "type": "precomputed",
            "backend": ["local", "auto"],
            "output": ["pooled"],
            "source": "mock.fixed.source",
        }

    def get_embedding(
        self,
        *,
        spatial,
        temporal,
        sensor,
        output,
        backend,
        device="auto",
        input_chw=None,
    ):
        return Embedding(
            data=np.arange(4, dtype=np.float32),
            meta={
                "model": self.model_name,
                "backend_used": backend,
                "source": "mock.fixed.source",
            },
        )


@pytest.fixture(autouse=True)
def register_mock():
    registry._REGISTRY.clear()
    registry.register("mock_model")(_MockEmbedder)
    yield
    registry._REGISTRY.clear()


# ── helpers ────────────────────────────────────────────────────────

_SPATIAL = PointBuffer(lon=0.0, lat=0.0, buffer_m=512)
_TEMPORAL = TemporalSpec.year(2024)


# ══════════════════════════════════════════════════════════════════════
# get_embedding
# ══════════════════════════════════════════════════════════════════════


def test_get_embedding_returns_embedding():
    from rs_embed.api import get_embedding

    emb = get_embedding("mock_model", spatial=_SPATIAL, temporal=_TEMPORAL)
    assert isinstance(emb, Embedding)
    assert emb.data.shape == (8,)
    assert emb.meta["model"] == "mock_model"


def test_get_embedding_output_modes():
    from rs_embed.api import get_embedding

    emb_pooled = get_embedding(
        "mock_model", spatial=_SPATIAL, output=OutputSpec.pooled()
    )
    assert emb_pooled.meta["output"] == "pooled"

    emb_grid = get_embedding("mock_model", spatial=_SPATIAL, output=OutputSpec.grid())
    assert emb_grid.meta["output"] == "grid"


def test_get_embedding_precomputed_default_backend_auto_resolves_to_auto():
    from rs_embed.api import get_embedding

    registry.register("mock_precomputed_local")(_MockPrecomputedLocalEmbedder)

    emb = get_embedding("mock_precomputed_local", spatial=_SPATIAL)
    assert emb.meta["backend_used"] == "auto"
    assert emb.meta["source"] == "mock.fixed.source"


def test_get_embedding_unknown_model():
    from rs_embed.api import get_embedding

    with pytest.raises(ModelError, match="Unknown model"):
        get_embedding("nonexistent", spatial=_SPATIAL)


# ══════════════════════════════════════════════════════════════════════
# get_embeddings_batch
# ══════════════════════════════════════════════════════════════════════


def test_get_embeddings_batch():
    from rs_embed.api import get_embeddings_batch

    spatials = [
        PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=1.0, lat=1.0, buffer_m=256),
        PointBuffer(lon=2.0, lat=2.0, buffer_m=256),
    ]
    results = get_embeddings_batch("mock_model", spatials=spatials, temporal=_TEMPORAL)
    assert len(results) == 3
    for emb in results:
        assert isinstance(emb, Embedding)


def test_get_embeddings_batch_empty():
    from rs_embed.api import get_embeddings_batch

    with pytest.raises(ModelError, match="non-empty"):
        get_embeddings_batch("mock_model", spatials=[], temporal=_TEMPORAL)


def test_get_embeddings_batch_with_sensor():
    """Ensures sensor param flows through _sensor_key without errors."""
    from rs_embed.api import get_embeddings_batch

    sensor = SensorSpec(collection="COLL", bands=("B1",))
    spatials = [PointBuffer(lon=0.0, lat=0.0, buffer_m=256)]
    results = get_embeddings_batch(
        "mock_model",
        spatials=spatials,
        temporal=_TEMPORAL,
        sensor=sensor,
    )
    assert len(results) == 1


def test_get_embeddings_batch_precomputed_default_backend_auto_resolves_to_auto():
    from rs_embed.api import get_embeddings_batch

    registry.register("mock_precomputed_local")(_MockPrecomputedLocalEmbedder)

    results = get_embeddings_batch(
        "mock_precomputed_local",
        spatials=[_SPATIAL, PointBuffer(lon=1.0, lat=0.0, buffer_m=512)],
    )
    assert len(results) == 2
    assert all(emb.meta["backend_used"] == "auto" for emb in results)


# ══════════════════════════════════════════════════════════════════════
# _validate_specs
# ══════════════════════════════════════════════════════════════════════


def test_validate_specs_invalid_spatial_type():
    from rs_embed.api import _validate_specs

    with pytest.raises(ModelError, match="Invalid spatial spec type"):
        _validate_specs(
            spatial="not-spatial", temporal=None, output=OutputSpec.pooled()
        )


def test_validate_specs_bad_output_mode():
    from rs_embed.api import _validate_specs

    bad_output = OutputSpec.__new__(OutputSpec)
    object.__setattr__(bad_output, "mode", "unknown")
    object.__setattr__(bad_output, "scale_m", 10)
    object.__setattr__(bad_output, "pooling", "mean")
    with pytest.raises(ModelError, match="Unknown output mode"):
        _validate_specs(spatial=_SPATIAL, temporal=None, output=bad_output)


def test_validate_specs_non_positive_scale():
    from rs_embed.api import _validate_specs

    bad_output = OutputSpec.__new__(OutputSpec)
    object.__setattr__(bad_output, "mode", "pooled")
    object.__setattr__(bad_output, "scale_m", 0)
    object.__setattr__(bad_output, "pooling", "mean")
    with pytest.raises(ModelError, match="scale_m must be positive"):
        _validate_specs(spatial=_SPATIAL, temporal=None, output=bad_output)


def test_validate_specs_bad_pooling():
    from rs_embed.api import _validate_specs

    bad_output = OutputSpec.__new__(OutputSpec)
    object.__setattr__(bad_output, "mode", "pooled")
    object.__setattr__(bad_output, "scale_m", 10)
    object.__setattr__(bad_output, "pooling", "median")
    with pytest.raises(ModelError, match="Unknown pooling"):
        _validate_specs(spatial=_SPATIAL, temporal=None, output=bad_output)


def test_validate_specs_ok():
    from rs_embed.api import _validate_specs

    _validate_specs(spatial=_SPATIAL, temporal=_TEMPORAL, output=OutputSpec.pooled())
    _validate_specs(spatial=_SPATIAL, temporal=None, output=OutputSpec.grid())


# ══════════════════════════════════════════════════════════════════════
# _assert_supported
# ══════════════════════════════════════════════════════════════════════


class _BackendLimitedEmbedder(EmbedderBase):
    """Embedder that only supports a specific backend."""

    def describe(self):
        return {
            "type": "mock",
            "dim": 8,
            "backend": ["gee"],
            "output": ["pooled"],
            "temporal": {"mode": "year"},
        }

    def get_embedding(
        self,
        *,
        spatial,
        temporal,
        sensor,
        output,
        backend,
        device="auto",
        input_chw=None,
    ):
        return Embedding(data=np.arange(8, dtype=np.float32), meta={})


class _BrokenDescribeEmbedder(EmbedderBase):
    """Embedder whose describe() raises — _assert_supported should not crash."""

    def describe(self):
        raise RuntimeError("broken")

    def get_embedding(self, **kw):
        return Embedding(data=np.zeros(4, dtype=np.float32), meta={})


def test_assert_supported_wrong_backend():
    from rs_embed.api import _assert_supported

    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    with pytest.raises(ModelError, match="does not support backend"):
        _assert_supported(
            emb, backend="local", output=OutputSpec.pooled(), temporal=None
        )


def test_assert_supported_wrong_output():
    from rs_embed.api import _assert_supported

    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    with pytest.raises(ModelError, match="does not support output.mode"):
        _assert_supported(emb, backend="gee", output=OutputSpec.grid(), temporal=None)


def test_assert_supported_wrong_temporal():
    from rs_embed.api import _assert_supported

    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    with pytest.raises(ModelError, match="expects TemporalSpec.mode='year'"):
        _assert_supported(
            emb,
            backend="gee",
            output=OutputSpec.pooled(),
            temporal=TemporalSpec.range("2022-01-01", "2022-06-01"),
        )


def test_assert_supported_ok():
    from rs_embed.api import _assert_supported

    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    _assert_supported(
        emb, backend="gee", output=OutputSpec.pooled(), temporal=TemporalSpec.year(2024)
    )


def test_assert_supported_broken_describe_raises_model_error():
    from rs_embed.api import _assert_supported

    emb = _BrokenDescribeEmbedder()
    emb.model_name = "broken"
    with pytest.raises(ModelError, match="describe\\(\\) failed"):
        _assert_supported(emb, backend="gee", output=OutputSpec.pooled(), temporal=None)


# ══════════════════════════════════════════════════════════════════════
# _sensor_key / _sensor_cache_key
# ══════════════════════════════════════════════════════════════════════


def test_sensor_key_none():
    from rs_embed.tools.runtime import sensor_key

    assert sensor_key(None) == ("__none__",)


def test_sensor_key_deterministic_and_differs():
    from rs_embed.tools.runtime import sensor_key

    s1 = SensorSpec(collection="A", bands=("B1",))
    s2 = SensorSpec(collection="B", bands=("B1",))
    assert sensor_key(s1) == sensor_key(s1)
    assert sensor_key(s1) != sensor_key(s2)


def test_sensor_cache_key_deterministic_and_differs():
    from rs_embed.tools.serialization import sensor_cache_key as _sensor_cache_key

    s1 = SensorSpec(collection="A", bands=("B1",))
    s2 = SensorSpec(collection="B", bands=("B1",))
    assert isinstance(_sensor_cache_key(s1), str)
    assert _sensor_cache_key(s1) == _sensor_cache_key(s1)
    assert _sensor_cache_key(s1) != _sensor_cache_key(s2)


# ══════════════════════════════════════════════════════════════════════
# export_batch — argument validation (no GEE needed)
# ══════════════════════════════════════════════════════════════════════


def test_export_batch_empty_spatials():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="non-empty"):
        export_batch(
            spatials=[], temporal=_TEMPORAL, models=["mock_model"], out_dir="/tmp"
        )


def test_export_batch_rejects_non_list_spatials():
    """_validate_spatials requires an actual list, not a tuple or single spec."""
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="non-empty"):
        export_batch(
            spatials=(_SPATIAL,),
            temporal=_TEMPORAL,
            models=["mock_model"],
            out_dir="/tmp",
        )

    with pytest.raises(ModelError, match="non-empty"):
        export_batch(
            spatials=_SPATIAL, temporal=_TEMPORAL, models=["mock_model"], out_dir="/tmp"
        )


def test_export_batch_empty_models():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="non-empty"):
        export_batch(spatials=[_SPATIAL], temporal=_TEMPORAL, models=[], out_dir="/tmp")


def test_export_batch_no_output_arg():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="out_dir or out_path"):
        export_batch(spatials=[_SPATIAL], temporal=_TEMPORAL, models=["mock_model"])


def test_export_batch_both_output_args():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="only one"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["mock_model"],
            out_dir="/tmp/a",
            out_path="/tmp/b.npz",
        )


def test_export_batch_decoupled_output_api_requires_out_and_layout():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="both out and layout"):
        export_batch(
            spatials=[_SPATIAL], temporal=_TEMPORAL, models=["mock_model"], out="/tmp/x"
        )


def test_export_batch_decoupled_output_api_disallows_mixing_with_legacy_args():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="either out\\+layout or out_dir/out_path"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["mock_model"],
            out="/tmp/x",
            layout="combined",
            out_path="/tmp/y.npz",
        )


def test_export_batch_unsupported_format():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="Unsupported export format"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["mock_model"],
            out_dir="/tmp",
            format="parquet",
        )


def test_export_batch_accepts_netcdf_format(tmp_path):
    """format='netcdf' should be accepted and export successfully with the mock embedder."""
    from rs_embed.api import export_batch

    results = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        out_dir=str(tmp_path),
        format="netcdf",
        backend="local",
        save_inputs=False,
        save_embeddings=True,
        save_manifest=False,
    )
    assert len(results) == 1
    nc_file = tmp_path / "p00000.nc"
    assert nc_file.exists()


def test_export_batch_names_length_mismatch(tmp_path):
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="same length"):
        export_batch(
            spatials=[_SPATIAL, _SPATIAL],
            temporal=_TEMPORAL,
            models=["mock_model"],
            out_dir=str(tmp_path),
            names=["only_one"],
        )


def test_export_batch_decoupled_layout_per_item(tmp_path):
    from rs_embed.api import export_batch

    export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        out=str(tmp_path / "dir_out"),
        layout="per_item",
        backend="local",
        save_inputs=False,
        save_embeddings=True,
        save_manifest=False,
        show_progress=False,
    )
    assert (tmp_path / "dir_out" / "p00000.npz").exists()


def test_export_batch_decoupled_layout_combined(tmp_path):
    from rs_embed.api import export_batch

    export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        out=str(tmp_path / "combined_out"),
        layout="combined",
        backend="local",
        save_inputs=False,
        save_embeddings=True,
        save_manifest=False,
        show_progress=False,
    )
    assert (tmp_path / "combined_out.npz").exists()


def test_public_list_models_uses_catalog_not_runtime_registry():
    from rs_embed import list_models

    models = list_models()
    assert "remoteclip" in models
    assert "remoteclip_s2rgb" not in models
    assert models == sorted(models)


def test_public_list_models_can_include_aliases():
    from rs_embed import list_models

    models = list_models(include_aliases=True)
    assert "remoteclip" in models
    assert "remoteclip_s2rgb" in models


def test_export_batch_infer_batch_size_is_independent_from_chunk_size(
    monkeypatch, tmp_path
):
    from rs_embed.api import export_batch

    captured = {}

    def _fake_run(self):
        captured["chunk_size"] = self.config.chunk_size
        captured["infer_batch_size"] = self.config.infer_batch_size
        return {"status": "ok"}

    monkeypatch.setattr("rs_embed.pipelines.exporter.BatchExporter.run", _fake_run)

    result = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        out_path=str(tmp_path / "combined"),
        chunk_size=32,
        infer_batch_size=5,
        show_progress=False,
    )

    assert result == {"status": "ok"}
    assert captured == {"chunk_size": 32, "infer_batch_size": 5}


# ══════════════════════════════════════════════════════════════════════
# normalize_embedding_output — idempotency / double-normalization guard
# ══════════════════════════════════════════════════════════════════════


def test_normalize_embedding_output_idempotent_for_pooled():
    """Pooled embeddings are passed through unchanged; calling twice is safe."""
    from rs_embed.tools.output import normalize_embedding_output

    data = np.arange(8, dtype=np.float32)
    emb = Embedding(data=data, meta={"y_axis_direction": "south_to_north"})
    output = OutputSpec.pooled()

    emb1 = normalize_embedding_output(emb=emb, output=output)
    emb2 = normalize_embedding_output(emb=emb1, output=output)
    np.testing.assert_array_equal(emb1.data, emb2.data)


def test_normalize_embedding_output_grid_south_to_north_applied_once():
    """
    A south-to-north grid embedding should be flipped exactly once.

    This is a regression guard for the double-normalization bug that existed
    in _run_embedding_request (prefetched path):  calling normalize_embedding_output
    a second time on an already-normalised embedding would set
    grid_orientation_applied=False, incorrectly claiming no flip occurred.
    """
    from rs_embed.tools.output import normalize_embedding_output

    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # 2×2 grid
    emb = Embedding(data=data, meta={"y_axis_direction": "south_to_north"})
    output = OutputSpec.grid()

    # First normalization — should flip the data
    emb1 = normalize_embedding_output(emb=emb, output=output)
    assert emb1.meta["grid_orientation_applied"] is True
    assert emb1.meta.get("y_axis_direction") == "north_to_south"
    # Rows should be reversed
    np.testing.assert_array_equal(emb1.data, np.flipud(data))

    # Second normalization — previously wrote grid_orientation_applied=False (bug!)
    emb2 = normalize_embedding_output(emb=emb1, output=output)
    # The second call sees native_dir="north_to_south" → no flip, applied=False.
    # That's acceptable semantics for this helper in isolation (the fix lives in
    # _run_embedding_request, which no longer calls it twice).
    # What matters: the DATA must not be double-flipped back to original.
    np.testing.assert_array_equal(emb2.data, np.flipud(data))


def test_run_embedding_request_prefetched_path_normalizes_once(monkeypatch):
    """
    get_embedding with a prefetched input should apply output normalisation
    exactly once, so grid_orientation_applied reflects truth for south-to-north
    models.
    """
    import rs_embed.api as api
    import rs_embed.tools.runtime as rt
    from rs_embed.core.embedding import Embedding

    class _SouthNorthGridEmbedder:
        model_name = "s2n_grid_mock"

        def describe(self):
            return {"type": "mock"}

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            # Simulate a model that reports south-to-north native orientation
            data = np.zeros((2, 2), dtype=np.float32)
            return Embedding(data=data, meta={"y_axis_direction": "south_to_north"})

    from rs_embed.core import registry

    registry.register("s2n_grid_mock")(_SouthNorthGridEmbedder)

    # Provide a fake prefetched input so the prefetched path is triggered
    fake_input = np.zeros((3, 4, 4), dtype=np.float32)

    monkeypatch.setattr(
        rt,
        "fetch_api_side_inputs",
        lambda *, spatials, temporal, **_: [fake_input],
    )

    emb = api.get_embedding(
        "s2n_grid_mock",
        spatial=_SPATIAL,
        temporal=_TEMPORAL,
        output=OutputSpec.grid(),
        backend="local",
        input_prep="resize",
    )

    # With the fix: normalised once → applied=True
    # Without the fix: second call set applied=False
    assert emb.meta.get("grid_orientation_applied") is True

    registry._REGISTRY.pop("s2n_grid_mock", None)


# ══════════════════════════════════════════════════════════════════════
# export_batch — assert_supported capability validation
# ══════════════════════════════════════════════════════════════════════


class _BackendOnlyGEEExportEmbedder:
    """Embedder that only supports backend='gee'."""

    model_name = "gee_only_export_mock"

    def describe(self):
        return {
            "type": "mock",
            "backend": ["gee"],
            "output": ["pooled"],
        }

    def get_embedding(
        self,
        *,
        spatial,
        temporal,
        sensor,
        output,
        backend,
        device="auto",
        input_chw=None,
    ):
        return Embedding(data=np.zeros(4, dtype=np.float32), meta={})


def test_export_batch_assert_supported_rejects_incompatible_backend(tmp_path):
    """export_batch should raise ModelError when a model doesn't support the backend."""
    from rs_embed.api import export_batch
    from rs_embed.core import registry
    from rs_embed.core.errors import ModelError

    registry.register("gee_only_export_mock")(_BackendOnlyGEEExportEmbedder)

    with pytest.raises(ModelError, match="does not support backend"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["gee_only_export_mock"],
            out_dir=str(tmp_path),
            backend="local",
            save_embeddings=False,
            show_progress=False,
        )

    registry._REGISTRY.pop("gee_only_export_mock", None)


def test_export_batch_assert_supported_rejects_incompatible_output_mode(tmp_path):
    """export_batch should raise ModelError when a model doesn't support the output mode."""
    from rs_embed.api import export_batch
    from rs_embed.core import registry
    from rs_embed.core.errors import ModelError

    registry.register("gee_only_export_mock")(_BackendOnlyGEEExportEmbedder)

    with pytest.raises(ModelError, match="does not support output.mode"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["gee_only_export_mock"],
            out_dir=str(tmp_path),
            backend="gee",
            output=OutputSpec.grid(),
            save_embeddings=False,
            show_progress=False,
        )

    registry._REGISTRY.pop("gee_only_export_mock", None)


def test_export_batch_assert_supported_passes_for_compatible_model(tmp_path):
    """export_batch should proceed normally when all model capabilities match."""
    from rs_embed.api import export_batch

    # _MockEmbedder (registered by autouse fixture) has no backend/output constraints
    results = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        out_dir=str(tmp_path),
        backend="local",
        save_inputs=False,
        save_embeddings=True,
        save_manifest=False,
        show_progress=False,
    )
    assert len(results) == 1
    assert results[0]["status"] == "ok"


def test_export_batch_backend_resolution_before_assert_supported(tmp_path, monkeypatch):
    """Backend is resolved per model BEFORE capability validation.

    A precomputed model declaring backend=["auto"] should pass when the user
    passes backend="gee", because _resolve_embedding_api_backend remaps "gee"
    to "auto" for precomputed models.  Without the per-model resolution fix
    (Finding 11), _assert_supported would see raw "gee" ∉ ["auto"] and raise.
    """
    import rs_embed.api as api
    from rs_embed.api import export_batch

    registry.register("mock_precomputed_local")(_MockPrecomputedLocalEmbedder)

    # Prevent real GEE initialization — the precomputed model uses backend="auto"
    # after remapping, so no provider is actually needed for inference.
    monkeypatch.setattr(api, "provider_factory_for_backend", lambda _b: None)

    # _MockPrecomputedLocalEmbedder declares backend=["local", "auto"]
    # User passes backend="gee" → _resolve_embedding_api_backend maps to "auto"
    # → _assert_supported sees "auto" ∈ ["local", "auto"] → passes
    results = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_precomputed_local"],
        out_dir=str(tmp_path),
        backend="gee",
        save_inputs=False,
        save_embeddings=True,
        save_manifest=False,
        show_progress=False,
    )
    assert len(results) == 1
    assert results[0]["status"] == "ok"

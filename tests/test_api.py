"""Tests for the public API (get_embedding, get_embeddings_batch, export_batch).

These use a mock embedder registered in the test so they don't require
GEE, torch, or any real model weights.
"""

import functools
import sys
import types

import numpy as np
import pytest

import rs_embed.api as api
import rs_embed.tools.runtime as rt
from rs_embed import list_models, reset_runtime
from rs_embed.api import (
    _assert_supported,
    _validate_specs,
    export_batch,
    get_embedding,
    get_embeddings_batch,
)
from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.errors import ModelError
from rs_embed.core.specs import FetchSpec, OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.core.types import (
    ExportConfig,
    ExportLayout,
    ExportModelRequest,
    ExportTarget,
    FetchResult,
)
from rs_embed.embedders.base import EmbedderBase
from rs_embed.tools.output import normalize_embedding_output
from rs_embed.tools.runtime import sensor_key
from rs_embed.tools.serialization import sensor_cache_key as _sensor_cache_key

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
        return Embedding(data=vec, meta={"model": self.model_name, "output": output.mode})


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


class _MockMultimodalEmbedder(EmbedderBase):
    def describe(self):
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "output": ["pooled"],
            "modalities": {
                "s2": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": ["B4", "B3", "B2"],
                },
                "s1": {
                    "collection": "COPERNICUS/S1_GRD_FLOAT",
                    "bands": ["VV", "VH"],
                    "defaults": {"use_float_linear": True},
                },
            },
            "defaults": {"modality": "s2", "scale_m": 10},
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
        vec = np.arange(4, dtype=np.float32)
        return Embedding(
            data=vec,
            meta={
                "model": self.model_name,
                "sensor": sensor,
                "backend_used": backend,
            },
        )


class _MockVariantEmbedder(EmbedderBase):
    def describe(self):
        return {
            "type": "mock",
            "backend": ["auto", "gee"],
            "output": ["pooled"],
            "model_config": {
                "variant": {
                    "type": "string",
                    "default": "base",
                    "choices": ["base", "large"],
                }
            },
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
        model_config=None,
    ):
        variant = (model_config or {}).get("variant", "base")
        return Embedding(
            data=np.arange(3, dtype=np.float32),
            meta={"model": self.model_name, "variant": variant, "backend_used": backend},
        )

    def get_embeddings_batch(
        self,
        *,
        spatials,
        temporal=None,
        sensor=None,
        model_config=None,
        output=OutputSpec.pooled(),
        backend="auto",
        device="auto",
    ):
        return [
            self.get_embedding(
                spatial=sp,
                temporal=temporal,
                sensor=sensor,
                model_config=model_config,
                output=output,
                backend=backend,
                device=device,
            )
            for sp in spatials
        ]


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
    emb = get_embedding("mock_model", spatial=_SPATIAL, temporal=_TEMPORAL)
    assert isinstance(emb, Embedding)
    assert emb.data.shape == (8,)
    assert emb.meta["model"] == "mock_model"


def test_get_embedding_output_modes():
    emb_pooled = get_embedding("mock_model", spatial=_SPATIAL, output=OutputSpec.pooled())
    assert emb_pooled.meta["output"] == "pooled"

    emb_grid = get_embedding("mock_model", spatial=_SPATIAL, output=OutputSpec.grid())
    assert emb_grid.meta["output"] == "grid"


def test_get_embedding_precomputed_default_backend_auto_resolves_to_auto():
    registry.register("mock_precomputed_local")(_MockPrecomputedLocalEmbedder)

    emb = get_embedding("mock_precomputed_local", spatial=_SPATIAL)
    assert emb.meta["backend_used"] == "auto"
    assert emb.meta["source"] == "mock.fixed.source"


def test_get_embedding_unknown_model():
    with pytest.raises(ModelError, match="Unknown model"):
        get_embedding("nonexistent", spatial=_SPATIAL)


def test_reset_runtime_clears_runtime_and_embedder_module_caches(monkeypatch):
    rt.get_embedder_bundle_cached("mock_model", "auto", "auto", sensor_key(None))
    rt.embedder_accepts_input_chw(type(_MockEmbedder))
    rt.embedder_accepts_model_config(type(_MockVariantEmbedder))
    registry._REGISTRY_IMPORT_ERRORS["remoteclip"] = RuntimeError("boom")

    fake_mod = types.ModuleType("rs_embed.embedders._reset_runtime_fake")

    @functools.lru_cache(maxsize=4)
    def _cached_loader(x):
        return x

    fake_mod._cached_loader = _cached_loader
    fake_mod._cached_loader(1)
    monkeypatch.setitem(sys.modules, fake_mod.__name__, fake_mod)

    summary = reset_runtime()

    assert summary["import_errors_cleared"] == 1
    assert summary["runtime_caches_cleared"] == 5
    assert summary["embedder_module_caches_cleared"] >= 1
    assert rt.get_embedder_bundle_cached.cache_info().currsize == 0
    assert rt.embedder_accepts_input_chw.cache_info().currsize == 0
    assert rt.embedder_accepts_model_config.cache_info().currsize == 0
    assert _cached_loader.cache_info().currsize == 0
    assert registry._REGISTRY_IMPORT_ERRORS == {}
    assert registry.get_embedder_cls("mock_model") is _MockEmbedder


def test_get_embedding_modality_resolves_default_sensor():
    registry.register("mock_multi")(_MockMultimodalEmbedder)

    emb = get_embedding("mock_multi", spatial=_SPATIAL, modality="s1", backend="gee")
    sensor = emb.meta["sensor"]
    assert sensor is not None
    assert sensor.modality == "s1"
    assert sensor.collection == "COPERNICUS/S1_GRD_FLOAT"
    assert sensor.bands == ("VV", "VH")


def test_get_embedding_fetch_resolves_default_sensor():
    registry.register("mock_multi")(_MockMultimodalEmbedder)

    emb = get_embedding(
        "mock_multi",
        spatial=_SPATIAL,
        fetch=FetchSpec(scale_m=30, cloudy_pct=5),
        backend="gee",
    )
    sensor = emb.meta["sensor"]
    assert sensor is not None
    assert sensor.modality == "s2"
    assert sensor.collection == "COPERNICUS/S2_SR_HARMONIZED"
    assert sensor.scale_m == 30
    assert sensor.cloudy_pct == 5


def test_get_embedding_rejects_sensor_and_fetch_together():
    with pytest.raises(ModelError, match="Use either sensor=... or fetch=..., not both"):
        get_embedding(
            "mock_model",
            spatial=_SPATIAL,
            sensor=SensorSpec(collection="COLL", bands=("B1",)),
            fetch=FetchSpec(scale_m=20),
        )


def test_get_embedding_rejects_unsupported_modality():
    with pytest.raises(ModelError, match="does not expose modality"):
        get_embedding("mock_model", spatial=_SPATIAL, modality="s1")


def test_get_embedding_rejects_model_kwargs_for_unsupported_model():
    with pytest.raises(ModelError, match="does not accept model-specific keyword arguments"):
        get_embedding("mock_model", spatial=_SPATIAL, variant="large")


def test_get_embedding_passes_model_kwargs_to_variant_aware_model():
    registry.register("mock_variant")(_MockVariantEmbedder)

    emb = get_embedding(
        "mock_variant",
        spatial=_SPATIAL,
        temporal=_TEMPORAL,
        variant="large",
    )
    assert emb.meta["variant"] == "large"


# ══════════════════════════════════════════════════════════════════════
# get_embeddings_batch
# ══════════════════════════════════════════════════════════════════════


def test_get_embeddings_batch():
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
    with pytest.raises(ModelError, match="non-empty"):
        get_embeddings_batch("mock_model", spatials=[], temporal=_TEMPORAL)


def test_get_embeddings_batch_with_sensor():
    """Ensures sensor param flows through _sensor_key without errors."""

    sensor = SensorSpec(collection="COLL", bands=("B1",))
    spatials = [PointBuffer(lon=0.0, lat=0.0, buffer_m=256)]
    results = get_embeddings_batch(
        "mock_model",
        spatials=spatials,
        temporal=_TEMPORAL,
        sensor=sensor,
    )
    assert len(results) == 1


def test_get_embeddings_batch_modality_merges_into_sensor():
    registry.register("mock_multi")(_MockMultimodalEmbedder)
    sensor = SensorSpec(collection="COPERNICUS/S1_GRD", bands=("VV", "VH"), scale_m=20)
    results = get_embeddings_batch(
        "mock_multi",
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        sensor=sensor,
        modality="s1",
        backend="gee",
    )
    out_sensor = results[0].meta["sensor"]
    assert out_sensor.modality == "s1"
    assert out_sensor.collection == "COPERNICUS/S1_GRD"
    assert out_sensor.scale_m == 20


def test_get_embeddings_batch_precomputed_default_backend_auto_resolves_to_auto():
    registry.register("mock_precomputed_local")(_MockPrecomputedLocalEmbedder)

    results = get_embeddings_batch(
        "mock_precomputed_local",
        spatials=[_SPATIAL, PointBuffer(lon=1.0, lat=0.0, buffer_m=512)],
    )
    assert len(results) == 2
    assert all(emb.meta["backend_used"] == "auto" for emb in results)


def test_get_embeddings_batch_passes_model_kwargs_to_variant_aware_model():
    registry.register("mock_variant")(_MockVariantEmbedder)

    results = get_embeddings_batch(
        "mock_variant",
        spatials=[_SPATIAL, PointBuffer(lon=1.0, lat=0.0, buffer_m=512)],
        temporal=_TEMPORAL,
        variant="large",
    )
    assert [emb.meta["variant"] for emb in results] == ["large", "large"]


# ══════════════════════════════════════════════════════════════════════
# _validate_specs
# ══════════════════════════════════════════════════════════════════════


def test_validate_specs_invalid_spatial_type():
    with pytest.raises(ModelError, match="Invalid spatial spec type"):
        _validate_specs(spatial="not-spatial", temporal=None, output=OutputSpec.pooled())


def test_validate_specs_bad_output_mode():
    bad_output = OutputSpec.__new__(OutputSpec)
    object.__setattr__(bad_output, "mode", "unknown")
    object.__setattr__(bad_output, "pooling", "mean")
    with pytest.raises(ModelError, match="Unknown output mode"):
        _validate_specs(spatial=_SPATIAL, temporal=None, output=bad_output)


def test_validate_specs_rejects_legacy_output_scale_m():
    bad_output = OutputSpec.__new__(OutputSpec)
    object.__setattr__(bad_output, "mode", "pooled")
    object.__setattr__(bad_output, "scale_m", 10)
    object.__setattr__(bad_output, "pooling", "mean")
    with pytest.raises(ModelError, match="output.scale_m is no longer supported"):
        _validate_specs(spatial=_SPATIAL, temporal=None, output=bad_output)


def test_validate_specs_bad_pooling():
    bad_output = OutputSpec.__new__(OutputSpec)
    object.__setattr__(bad_output, "mode", "pooled")
    object.__setattr__(bad_output, "pooling", "median")
    with pytest.raises(ModelError, match="Unknown pooling"):
        _validate_specs(spatial=_SPATIAL, temporal=None, output=bad_output)


def test_validate_specs_ok():
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
    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    with pytest.raises(ModelError, match="does not support backend"):
        _assert_supported(emb, backend="local", output=OutputSpec.pooled(), temporal=None)


def test_assert_supported_wrong_output():
    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    with pytest.raises(ModelError, match="does not support output.mode"):
        _assert_supported(emb, backend="gee", output=OutputSpec.grid(), temporal=None)


def test_assert_supported_wrong_temporal():
    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    with pytest.warns(UserWarning, match="only supports TemporalSpec.year"):
        _assert_supported(
            emb,
            backend="gee",
            output=OutputSpec.pooled(),
            temporal=TemporalSpec.range("2022-01-01", "2022-06-01"),
        )


def test_assert_supported_ok():
    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    _assert_supported(
        emb, backend="gee", output=OutputSpec.pooled(), temporal=TemporalSpec.year(2024)
    )


def test_assert_supported_broken_describe_raises_model_error():
    emb = _BrokenDescribeEmbedder()
    emb.model_name = "broken"
    with pytest.raises(ModelError, match="describe\\(\\) failed"):
        _assert_supported(emb, backend="gee", output=OutputSpec.pooled(), temporal=None)


# ══════════════════════════════════════════════════════════════════════
# _sensor_key / _sensor_cache_key
# ══════════════════════════════════════════════════════════════════════


def test_sensor_key_none():
    assert sensor_key(None) == ("__none__",)


def test_sensor_key_deterministic_and_differs():
    s1 = SensorSpec(collection="A", bands=("B1",), modality="s1")
    s2 = SensorSpec(collection="A", bands=("B1",), modality="s2")
    assert sensor_key(s1) == sensor_key(s1)
    assert sensor_key(s1) != sensor_key(s2)


def test_sensor_cache_key_deterministic_and_differs():
    s1 = SensorSpec(collection="A", bands=("B1",), modality="s1")
    s2 = SensorSpec(collection="A", bands=("B1",), modality="s2")
    assert isinstance(_sensor_cache_key(s1), str)
    assert _sensor_cache_key(s1) == _sensor_cache_key(s1)
    assert _sensor_cache_key(s1) != _sensor_cache_key(s2)


# ══════════════════════════════════════════════════════════════════════
# export_batch — argument validation (no GEE needed)
# ══════════════════════════════════════════════════════════════════════


def test_export_batch_empty_spatials():
    with pytest.raises(ModelError, match="non-empty"):
        export_batch(
            spatials=[],
            temporal=_TEMPORAL,
            models=["mock_model"],
            target=ExportTarget.per_item("/tmp"),
        )


def test_export_batch_rejects_non_list_spatials():
    """_validate_spatials requires an actual list, not a tuple or single spec."""

    with pytest.raises(ModelError, match="non-empty"):
        export_batch(
            spatials=(_SPATIAL,),
            temporal=_TEMPORAL,
            models=["mock_model"],
            target=ExportTarget.per_item("/tmp"),
        )

    with pytest.raises(ModelError, match="non-empty"):
        export_batch(
            spatials=_SPATIAL,
            temporal=_TEMPORAL,
            models=["mock_model"],
            target=ExportTarget.per_item("/tmp"),
        )


def test_export_batch_empty_models():
    with pytest.raises(ModelError, match="non-empty"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=[],
            target=ExportTarget.per_item("/tmp"),
        )


def test_export_batch_unsupported_format():
    with pytest.raises(ModelError, match="Unsupported export format"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["mock_model"],
            target=ExportTarget.per_item("/tmp"),
            config=ExportConfig(format="parquet"),
        )


def test_export_batch_accepts_netcdf_format(tmp_path):
    """format='netcdf' should be accepted and export successfully with the mock embedder."""

    results = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        target=ExportTarget.per_item(str(tmp_path)),
        config=ExportConfig(
            format="netcdf",
            save_inputs=False,
            save_embeddings=True,
            save_manifest=False,
        ),
        backend="local",
    )
    assert len(results) == 1
    nc_file = tmp_path / "p00000.nc"
    assert nc_file.exists()


def test_export_batch_names_length_mismatch(tmp_path):
    with pytest.raises(ModelError, match="same length"):
        export_batch(
            spatials=[_SPATIAL, _SPATIAL],
            temporal=_TEMPORAL,
            models=["mock_model"],
            target=ExportTarget.per_item(str(tmp_path), names=["only_one"]),
        )


def test_export_batch_combined_target_requires_out_file():
    with pytest.raises(ModelError, match="requires out_file"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["mock_model"],
            target=ExportTarget(layout=ExportLayout.COMBINED),
        )


def test_export_batch_per_item_target_requires_out_dir():
    with pytest.raises(ModelError, match="requires out_dir"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["mock_model"],
            target=ExportTarget(layout=ExportLayout.PER_ITEM),
        )


def test_export_batch_per_item_layout(tmp_path):
    export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        target=ExportTarget.per_item(str(tmp_path / "dir_out")),
        config=ExportConfig(
            save_inputs=False,
            save_embeddings=True,
            save_manifest=False,
            show_progress=False,
        ),
        backend="local",
    )
    assert (tmp_path / "dir_out" / "p00000.npz").exists()


def test_export_batch_combined_layout(tmp_path):
    export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        target=ExportTarget.combined(str(tmp_path / "combined_out.npz")),
        config=ExportConfig(
            save_inputs=False,
            save_embeddings=True,
            save_manifest=False,
            show_progress=False,
        ),
        backend="local",
    )
    assert (tmp_path / "combined_out.npz").exists()


def test_export_batch_object_style_target_and_config(tmp_path):
    export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        target=ExportTarget.per_item(str(tmp_path), names=["sample"]),
        config=ExportConfig(
            save_inputs=False,
            save_embeddings=True,
            save_manifest=False,
            show_progress=False,
        ),
        backend="local",
    )
    assert (tmp_path / "sample.npz").exists()


def test_public_list_models_uses_catalog_not_runtime_registry():
    models = list_models()
    assert "remoteclip" in models
    assert "remoteclip_s2rgb" not in models
    assert models == sorted(models)


def test_public_list_models_can_include_aliases():
    models = list_models(include_aliases=True)
    assert "remoteclip" in models
    assert "remoteclip_s2rgb" in models


def test_export_batch_infer_batch_size_is_independent_from_chunk_size(monkeypatch, tmp_path):
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
        target=ExportTarget.combined(str(tmp_path / "combined")),
        config=ExportConfig(chunk_size=32, infer_batch_size=5, show_progress=False),
    )

    assert result == {"status": "ok"}
    assert captured == {"chunk_size": 32, "infer_batch_size": 5}


def test_export_batch_modality_resolves_model_sensor(monkeypatch, tmp_path):
    registry.register("mock_multi")(_MockMultimodalEmbedder)
    captured = {}

    def _fake_run(self):
        captured["sensor"] = self.models[0].sensor
        return {"status": "ok"}

    monkeypatch.setattr("rs_embed.pipelines.exporter.BatchExporter.run", _fake_run)

    result = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_multi"],
        target=ExportTarget.combined(str(tmp_path / "combined")),
        config=ExportConfig(show_progress=False),
        modality="s1",
        backend="gee",
    )

    assert result == {"status": "ok"}
    sensor = captured["sensor"]
    assert sensor is not None
    assert sensor.modality == "s1"
    assert sensor.collection == "COPERNICUS/S1_GRD_FLOAT"


def test_export_batch_fetch_resolves_model_sensor(monkeypatch, tmp_path):
    registry.register("mock_multi")(_MockMultimodalEmbedder)
    captured = {}

    def _fake_run(self):
        captured["sensor"] = self.models[0].sensor
        return {"status": "ok"}

    monkeypatch.setattr("rs_embed.pipelines.exporter.BatchExporter.run", _fake_run)

    result = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_multi"],
        target=ExportTarget.combined(str(tmp_path / "combined")),
        config=ExportConfig(show_progress=False),
        fetch=FetchSpec(scale_m=30, cloudy_pct=5),
        backend="gee",
    )

    assert result == {"status": "ok"}
    sensor = captured["sensor"]
    assert sensor is not None
    assert sensor.modality == "s2"
    assert sensor.collection == "COPERNICUS/S2_SR_HARMONIZED"
    assert sensor.scale_m == 30
    assert sensor.cloudy_pct == 5


def test_export_batch_rejects_sensor_and_fetch_conflict(monkeypatch, tmp_path):
    registry.register("mock_multi")(_MockMultimodalEmbedder)

    monkeypatch.setattr(
        "rs_embed.pipelines.exporter.BatchExporter.run",
        lambda self: {"status": "unexpected"},
    )

    with pytest.raises(ModelError, match="Use either sensor=... or fetch=..., not both"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=[
                ExportModelRequest(
                    "mock_multi",
                    sensor=SensorSpec(
                        collection="COPERNICUS/S2_SR_HARMONIZED", bands=("B4", "B3", "B2")
                    ),
                    fetch=FetchSpec(scale_m=20),
                )
            ],
            target=ExportTarget.combined(str(tmp_path / "combined")),
            config=ExportConfig(show_progress=False),
            backend="gee",
        )


def test_export_batch_export_model_request_applies_per_model_overrides(monkeypatch, tmp_path):
    registry.register("mock_multi")(_MockMultimodalEmbedder)
    captured = {}

    def _fake_run(self):
        captured["sensor"] = self.models[0].sensor
        return {"status": "ok"}

    monkeypatch.setattr("rs_embed.pipelines.exporter.BatchExporter.run", _fake_run)

    result = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=[
            ExportModelRequest(
                "mock_multi",
                modality="s1",
                sensor=SensorSpec(
                    collection="COPERNICUS/S1_GRD_FLOAT",
                    bands=("VV", "VH"),
                    scale_m=20,
                ),
            )
        ],
        target=ExportTarget.combined(str(tmp_path / "combined")),
        backend="gee",
        config=ExportConfig(show_progress=False),
    )

    assert result == {"status": "ok"}
    sensor = captured["sensor"]
    assert sensor is not None
    assert sensor.modality == "s1"
    assert sensor.collection == "COPERNICUS/S1_GRD_FLOAT"
    assert sensor.scale_m == 20


def test_export_batch_export_model_request_preserves_model_config(monkeypatch, tmp_path):
    registry.register("mock_variant")(_MockVariantEmbedder)
    captured = {}

    def _fake_run(self):
        captured["model_config"] = self.models[0].model_config
        return {"status": "ok"}

    monkeypatch.setattr("rs_embed.pipelines.exporter.BatchExporter.run", _fake_run)

    result = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=[ExportModelRequest("mock_variant", model_config={"variant": "large"})],
        target=ExportTarget.combined(str(tmp_path / "combined")),
        backend="auto",
        config=ExportConfig(show_progress=False),
    )

    assert result == {"status": "ok"}
    assert captured["model_config"] == {"variant": "large"}


def test_export_batch_rejects_model_config_for_unsupported_model(tmp_path):
    with pytest.raises(ModelError, match="does not accept model-specific keyword arguments"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=[ExportModelRequest("mock_model", model_config={"variant": "large"})],
            target=ExportTarget.combined(str(tmp_path / "combined")),
            backend="auto",
            config=ExportConfig(show_progress=False),
        )


# ══════════════════════════════════════════════════════════════════════
# normalize_embedding_output — idempotency / double-normalization guard
# ══════════════════════════════════════════════════════════════════════


def test_normalize_embedding_output_idempotent_for_pooled():
    """Pooled embeddings are passed through unchanged; calling twice is safe."""

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

    registry.register("s2n_grid_mock")(_SouthNorthGridEmbedder)

    # Provide a fake prefetched input so the prefetched path is triggered
    fake_input = np.zeros((3, 4, 4), dtype=np.float32)

    monkeypatch.setattr(
        rt,
        "fetch_api_side_inputs",
        lambda *, spatials, temporal, **_: [FetchResult(data=fake_input, meta={})],
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

    registry.register("gee_only_export_mock")(_BackendOnlyGEEExportEmbedder)

    with pytest.raises(ModelError, match="does not support backend"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["gee_only_export_mock"],
            target=ExportTarget.per_item(str(tmp_path)),
            config=ExportConfig(save_embeddings=False, show_progress=False),
            backend="local",
        )

    registry._REGISTRY.pop("gee_only_export_mock", None)


def test_export_batch_assert_supported_rejects_incompatible_output_mode(tmp_path):
    """export_batch should raise ModelError when a model doesn't support the output mode."""

    registry.register("gee_only_export_mock")(_BackendOnlyGEEExportEmbedder)

    with pytest.raises(ModelError, match="does not support output.mode"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["gee_only_export_mock"],
            target=ExportTarget.per_item(str(tmp_path)),
            config=ExportConfig(save_embeddings=False, show_progress=False),
            backend="gee",
            output=OutputSpec.grid(),
        )

    registry._REGISTRY.pop("gee_only_export_mock", None)


def test_export_batch_assert_supported_passes_for_compatible_model(tmp_path):
    """export_batch should proceed normally when all model capabilities match."""

    # _MockEmbedder (registered by autouse fixture) has no backend/output constraints
    results = export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        target=ExportTarget.per_item(str(tmp_path)),
        config=ExportConfig(
            save_inputs=False,
            save_embeddings=True,
            save_manifest=False,
            show_progress=False,
        ),
        backend="local",
    )
    assert len(results) == 1
    assert results[0]["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════
# ExportModelRequest.configure — simplified model kwargs interface
# ══════════════════════════════════════════════════════════════════════


def test_export_model_request_configure_builds_model_config():
    req = ExportModelRequest.configure("dofa", variant="large")
    assert req.name == "dofa"
    assert req.model_config == {"variant": "large"}
    assert req.sensor is None
    assert req.fetch is None
    assert req.modality is None


def test_export_model_request_configure_no_model_kwargs_gives_none():
    req = ExportModelRequest.configure("prithvi")
    assert req.name == "prithvi"
    assert req.model_config is None


def test_export_model_request_configure_multiple_kwargs():
    req = ExportModelRequest.configure("dofa", variant="base", image_size=224)
    assert req.model_config == {"variant": "base", "image_size": 224}


def test_export_model_request_configure_passes_through_to_export_batch(monkeypatch, tmp_path):
    """ExportModelRequest.configure() model_config is forwarded into the BatchExporter."""
    registry.register("mock_variant")(_MockVariantEmbedder)
    captured = {}

    def _fake_run(self):
        captured["model_config"] = self.models[0].model_config
        return [{"status": "ok"}]

    monkeypatch.setattr("rs_embed.pipelines.exporter.BatchExporter.run", _fake_run)

    export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=[ExportModelRequest.configure("mock_variant", variant="large")],
        target=ExportTarget.per_item(str(tmp_path)),
        config=ExportConfig(show_progress=False),
    )

    assert captured["model_config"] == {"variant": "large"}


def test_export_batch_backend_resolution_before_assert_supported(tmp_path, monkeypatch):
    """Backend is resolved per model BEFORE capability validation.

    A precomputed model declaring backend=["auto"] should pass when the user
    passes backend="gee", because _resolve_embedding_api_backend remaps "gee"
    to "auto" for precomputed models.  Without the per-model resolution fix
    (Finding 11), _assert_supported would see raw "gee" ∉ ["auto"] and raise.
    """

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
        target=ExportTarget.per_item(str(tmp_path)),
        config=ExportConfig(
            save_inputs=False,
            save_embeddings=True,
            save_manifest=False,
            show_progress=False,
        ),
        backend="gee",
    )
    assert len(results) == 1
    assert results[0]["status"] == "ok"

"""Tests for the Model class (OOP API).

Mirrors test_api.py in philosophy: uses a mock embedder registered into
the registry so no GEE, torch, or real model weights are required.
"""

import numpy as np
import pytest

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.errors import ModelError
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.embedders.base import EmbedderBase
from rs_embed.model import Model
from rs_embed.tools.runtime import get_embedder_bundle_cached


# ── mock embedder ──────────────────────────────────────────────────


class _MockEmbedder(EmbedderBase):
    """Deterministic embedder with no I/O."""

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
            data=vec,
            meta={
                "model": self.model_name,
                "output": output.mode,
                "sensor": sensor,
            },
        )


class _MockPrecomputedLocalEmbedder(EmbedderBase):
    def describe(self):
        return {
            "type": "precomputed",
            "backend": ["local", "auto"],
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
        return Embedding(
            data=np.ones(4, dtype=np.float32),
            meta={"backend_used": backend},
        )


@pytest.fixture(autouse=True)
def _setup():
    registry._REGISTRY.clear()
    registry.register("mock_model")(_MockEmbedder)
    get_embedder_bundle_cached.cache_clear()
    yield
    registry._REGISTRY.clear()
    get_embedder_bundle_cached.cache_clear()


# ── helpers ────────────────────────────────────────────────────────

_SPATIAL = PointBuffer(lon=0.0, lat=0.0, buffer_m=512)
_TEMPORAL = TemporalSpec.year(2024)


# ══════════════════════════════════════════════════════════════════════
# Construction
# ══════════════════════════════════════════════════════════════════════


def test_model_init_happy_path():
    m = Model("mock_model")
    assert m._model_n == "mock_model"


def test_model_init_unknown_model_raises():
    with pytest.raises(ModelError, match="Unknown model"):
        Model("nonexistent_xyz")


def test_model_list_models_returns_sorted_list():
    models = Model.list_models()
    assert isinstance(models, list)
    assert "mock_model" in models
    assert models == sorted(models)


def test_model_list_models_include_aliases():
    # With include_aliases=True the result should be a superset
    without = set(Model.list_models())
    with_aliases = set(Model.list_models(include_aliases=True))
    assert without.issubset(with_aliases)


# ══════════════════════════════════════════════════════════════════════
# get_embedding
# ══════════════════════════════════════════════════════════════════════


def test_model_get_embedding_returns_embedding():
    m = Model("mock_model")
    emb = m.get_embedding(_SPATIAL, temporal=_TEMPORAL)
    assert isinstance(emb, Embedding)
    assert emb.data.shape == (8,)
    assert emb.meta["model"] == "mock_model"


def test_model_get_embedding_output_mode_pooled():
    m = Model("mock_model", output=OutputSpec.pooled())
    emb = m.get_embedding(_SPATIAL)
    assert emb.meta["output"] == "pooled"


def test_model_get_embedding_output_mode_grid():
    m = Model("mock_model", output=OutputSpec.grid())
    emb = m.get_embedding(_SPATIAL)
    assert emb.meta["output"] == "grid"


def test_model_get_embedding_without_temporal():
    m = Model("mock_model")
    emb = m.get_embedding(_SPATIAL)
    assert isinstance(emb, Embedding)


# ══════════════════════════════════════════════════════════════════════
# get_embeddings_batch
# ══════════════════════════════════════════════════════════════════════


def test_model_get_embeddings_batch_returns_ordered_results():
    m = Model("mock_model")
    spatials = [
        PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=1.0, lat=1.0, buffer_m=256),
        PointBuffer(lon=2.0, lat=2.0, buffer_m=256),
    ]
    results = m.get_embeddings_batch(spatials, temporal=_TEMPORAL)
    assert len(results) == 3
    for emb in results:
        assert isinstance(emb, Embedding)
        assert emb.data.shape == (8,)


def test_model_get_embeddings_batch_single_item():
    m = Model("mock_model")
    results = m.get_embeddings_batch([_SPATIAL])
    assert len(results) == 1
    assert isinstance(results[0], Embedding)


def test_model_get_embeddings_batch_empty_raises():
    m = Model("mock_model")
    with pytest.raises(ModelError, match="non-empty"):
        m.get_embeddings_batch([])


def test_model_get_embeddings_batch_not_list_raises():
    m = Model("mock_model")
    with pytest.raises(ModelError):
        m.get_embeddings_batch(_SPATIAL)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════
# describe
# ══════════════════════════════════════════════════════════════════════


def test_model_describe_returns_dict():
    m = Model("mock_model")
    d = m.describe()
    assert isinstance(d, dict)
    assert d.get("type") == "mock"


def test_model_describe_graceful_on_exception(monkeypatch):
    m = Model("mock_model")
    monkeypatch.setattr(m._embedder, "describe", lambda: (_ for _ in ()).throw(RuntimeError("oops")))
    assert m.describe() == {}


# ══════════════════════════════════════════════════════════════════════
# Functional vs. OOP API equivalence
# ══════════════════════════════════════════════════════════════════════


def test_model_and_functional_api_produce_same_shape():
    """Model.get_embedding and api.get_embedding must return same-shaped result."""
    from rs_embed.api import get_embedding

    m = Model("mock_model", output=OutputSpec.pooled())
    oop_emb = m.get_embedding(_SPATIAL, temporal=_TEMPORAL)
    fn_emb = get_embedding(
        "mock_model",
        spatial=_SPATIAL,
        temporal=_TEMPORAL,
        output=OutputSpec.pooled(),
    )
    assert oop_emb.data.shape == fn_emb.data.shape
    np.testing.assert_array_equal(oop_emb.data, fn_emb.data)


# ══════════════════════════════════════════════════════════════════════
# Embedder reuse / caching
# ══════════════════════════════════════════════════════════════════════


def test_model_reuses_same_embedder_across_calls():
    """Two calls on the same Model must use the same underlying embedder instance."""
    m = Model("mock_model")
    embedder_id_1 = id(m._embedder)

    m.get_embedding(_SPATIAL)
    m.get_embedding(_SPATIAL)

    assert id(m._embedder) == embedder_id_1


def test_two_model_instances_share_cached_embedder():
    """Two Model instances with the same args should share the lru_cache entry."""
    m1 = Model("mock_model")
    m2 = Model("mock_model")
    assert m1._embedder is m2._embedder


# ══════════════════════════════════════════════════════════════════════
# Sensor binding
# ══════════════════════════════════════════════════════════════════════


def test_model_sensor_flows_to_embedding():
    sensor = SensorSpec(collection="COPERNICUS/S2", bands=("B4", "B3", "B2"))
    m = Model("mock_model", sensor=sensor)
    emb = m.get_embedding(_SPATIAL)
    # The mock embedder echoes the sensor in meta
    assert emb.meta["sensor"] == sensor


def test_model_no_sensor_passes_none():
    m = Model("mock_model", sensor=None)
    emb = m.get_embedding(_SPATIAL)
    assert emb.meta["sensor"] is None


# ══════════════════════════════════════════════════════════════════════
# Precomputed model backend routing
# ══════════════════════════════════════════════════════════════════════


def test_model_precomputed_auto_resolves_to_auto():
    """Precomputed model with backend='auto' should route to 'auto' (local access)."""
    registry.register("mock_precomputed")(_MockPrecomputedLocalEmbedder)
    m = Model("mock_precomputed", backend="auto")
    emb = m.get_embedding(_SPATIAL)
    assert emb.meta["backend_used"] == "auto"

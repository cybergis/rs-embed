"""Tests for rs_embed.pipelines.combined_flow — run_pending_models.

These tests exercise the three-tier batch fallback:
  1. Batch with prefetched inputs (get_embeddings_batch_from_inputs)
  2. Batch without inputs (get_embeddings_batch)
  3. Per-item fallback (get_embedding)

All embedders are fakes — the goal is to verify the orchestration logic,
not real model inference.
"""

from threading import RLock
from typing import Any, Dict, List

import numpy as np
import pytest

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.pipelines import combined_flow
from rs_embed.pipelines import inference as inference_mod
from rs_embed.pipelines.combined_flow import run_pending_models


# ── Helpers ────────────────────────────────────────────────────────


class _NoOpProgress:
    """Minimal progress bar that does nothing."""

    def update(self, n=1):
        pass

    def close(self):
        pass


def _patch_deps(
    monkeypatch,
    embedder: Any,
    lock: Any = None,
    supports_batch: bool = False,
    supports_prefetched_batch: bool = False,
):
    if lock is None:
        lock = RLock()

    # ── Patches on combined_flow (describe, input gathering, progress) ──
    monkeypatch.setattr(combined_flow, "create_progress", lambda **kw: _NoOpProgress())
    monkeypatch.setattr(
        combined_flow, "drop_model_arrays", lambda arrays, m, sanitize_key=None: None
    )
    monkeypatch.setattr(combined_flow, "jsonable", lambda x: x)
    monkeypatch.setattr(
        combined_flow,
        "sensor_key",
        lambda s: ("__none__",) if s is None else (s.collection,),
    )
    monkeypatch.setattr(combined_flow, "normalize_model_name", lambda m: m)
    monkeypatch.setattr(
        combined_flow,
        "get_embedder_bundle_cached",
        lambda model, backend, device, sk: (embedder, lock),
    )
    monkeypatch.setattr(
        combined_flow, "sensor_cache_key", lambda s: s.collection if s else "__none__"
    )
    monkeypatch.setattr(
        combined_flow, "sanitize_key", lambda s: s.replace("/", "_").replace(" ", "_")
    )

    # ── Patches on inference module (for InferenceEngine.infer_model) ──
    monkeypatch.setattr(
        inference_mod,
        "get_embedder_bundle_cached",
        lambda model, backend, device, sk: (embedder, lock),
    )
    monkeypatch.setattr(
        inference_mod,
        "sensor_key",
        lambda s: ("__none__",) if s is None else (s.collection,),
    )
    monkeypatch.setattr(
        inference_mod,
        "sensor_cache_key",
        lambda s: s.collection if s else "__none__",
    )
    monkeypatch.setattr(
        inference_mod,
        "call_embedder_get_embedding",
        lambda **kw: kw["embedder"].get_embedding(
            **{k: v for k, v in kw.items() if k != "embedder"}
        ),
    )
    monkeypatch.setattr(
        inference_mod,
        "supports_prefetched_batch_api",
        lambda e: supports_prefetched_batch,
    )
    monkeypatch.setattr(inference_mod, "supports_batch_api", lambda e: supports_batch)
    monkeypatch.setattr(
        inference_mod,
        "embedding_to_numpy",
        lambda e: np.asarray(e.data, dtype=np.float32),
    )
    monkeypatch.setattr(
        inference_mod,
        "normalize_embedding_output",
        lambda emb, output: emb,
    )
    monkeypatch.setattr(inference_mod, "run_with_retry", lambda fn, **kw: fn())
    monkeypatch.setattr(inference_mod, "jsonable", lambda x: x)


def _make_spatials(n: int) -> list:
    return [PointBuffer(lon=-88.0 + i * 0.1, lat=40.0, buffer_m=500) for i in range(n)]


def _base_kwargs(
    *,
    models: List[str],
    spatials: list,
    provider_enabled: bool = True,
    save_embeddings: bool = True,
    continue_on_error: bool = False,
    inference_strategy: str = "auto",
) -> dict:
    manifest: Dict[str, Any] = {"models": []}
    return dict(
        pending_models=models,
        arrays={},
        manifest=manifest,
        spatials=spatials,
        temporal=TemporalSpec.range("2020-01-01", "2020-06-01"),
        output=OutputSpec.pooled(),
        resolved_sensor={m: None for m in models},
        model_type={m: "onthefly" for m in models},
        backend="gee",
        provider_enabled=provider_enabled,
        device="cpu",
        save_inputs=False,
        save_embeddings=save_embeddings,
        continue_on_error=continue_on_error,
        chunk_size=2,
        inference_strategy=inference_strategy,
        infer_batch_size=8,
        max_retries=0,
        retry_backoff_s=0.0,
        show_progress=False,
        input_refs_by_sensor={},
        get_or_fetch_input_fn=lambda i, sk, ss: np.ones((3, 4, 4), dtype=np.float32),
        write_checkpoint_fn=lambda **kw: manifest,
        progress=_NoOpProgress(),
    )


# ══════════════════════════════════════════════════════════════════════
# Per-item (single) inference — the default fallback
# ══════════════════════════════════════════════════════════════════════


class _SingleEmbedder:
    """Embedder that only supports per-item get_embedding."""

    calls = 0

    def describe(self):
        return {"type": "onthefly"}

    def get_embedding(self, *, spatial, temporal, sensor, output, backend, device, input_chw=None):
        _SingleEmbedder.calls += 1
        return Embedding(data=np.array([1.0, 2.0], dtype=np.float32), meta={})


def test_per_item_fallback(monkeypatch):
    """When no batch API is available, per-item inference runs for each spatial."""
    _SingleEmbedder.calls = 0
    embedder = _SingleEmbedder()
    spatials = _make_spatials(3)
    _patch_deps(monkeypatch, embedder)

    kw = _base_kwargs(models=["m1"], spatials=spatials, provider_enabled=False)
    manifest = run_pending_models(**kw)

    assert _SingleEmbedder.calls == 3
    m_entry = manifest["models"][0]
    assert m_entry["status"] == "ok"
    assert m_entry["embeddings"] is not None


# ══════════════════════════════════════════════════════════════════════
# Batch API (no input) succeeds
# ══════════════════════════════════════════════════════════════════════


class _BatchEmbedder:
    """Embedder that supports get_embeddings_batch."""

    batch_calls = 0
    single_calls = 0

    def describe(self):
        return {"type": "onthefly"}

    def get_embedding(self, **kw):
        _BatchEmbedder.single_calls += 1
        return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    def get_embeddings_batch(self, *, spatials, temporal, sensor, output, backend, device):
        _BatchEmbedder.batch_calls += 1
        return [
            Embedding(data=np.array([float(i)], dtype=np.float32), meta={})
            for i in range(len(spatials))
        ]


def test_batch_no_input_succeeds(monkeypatch):
    """When embedder supports batch API and no provider input needed, batch path is used."""
    _BatchEmbedder.batch_calls = 0
    _BatchEmbedder.single_calls = 0
    embedder = _BatchEmbedder()
    spatials = _make_spatials(4)
    _patch_deps(monkeypatch, embedder, supports_batch=True)

    kw = _base_kwargs(models=["m1"], spatials=spatials, provider_enabled=False)
    manifest = run_pending_models(**kw)

    assert _BatchEmbedder.batch_calls >= 1
    assert _BatchEmbedder.single_calls == 0
    m_entry = manifest["models"][0]
    assert m_entry["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════
# Batch fails → per-item fallback
# ══════════════════════════════════════════════════════════════════════


class _FailingBatchEmbedder:
    """Embedder whose batch API always fails, but single works."""

    batch_calls = 0
    single_calls = 0

    def describe(self):
        return {"type": "onthefly"}

    def get_embedding(self, **kw):
        _FailingBatchEmbedder.single_calls += 1
        return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    def get_embeddings_batch(self, **kw):
        _FailingBatchEmbedder.batch_calls += 1
        raise RuntimeError("batch OOM")


def test_batch_fails_falls_back_to_single(monkeypatch):
    """When batch API fails, all items are processed via per-item fallback."""
    _FailingBatchEmbedder.batch_calls = 0
    _FailingBatchEmbedder.single_calls = 0
    embedder = _FailingBatchEmbedder()
    spatials = _make_spatials(3)
    _patch_deps(monkeypatch, embedder, supports_batch=True)

    kw = _base_kwargs(models=["m1"], spatials=spatials, provider_enabled=False)
    manifest = run_pending_models(**kw)

    assert _FailingBatchEmbedder.batch_calls >= 1  # batch was attempted
    assert _FailingBatchEmbedder.single_calls == 3  # fell back to per-item
    m_entry = manifest["models"][0]
    assert m_entry["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════
# Prefetched batch succeeds
# ══════════════════════════════════════════════════════════════════════


class _PrefetchedBatchEmbedder:
    """Embedder that supports batch-from-prefetched inputs."""

    batch_calls = 0
    single_calls = 0

    def describe(self):
        return {"type": "onthefly", "inputs": {"collection": "C", "bands": ["B1"]}}

    def get_embedding(self, **kw):
        _PrefetchedBatchEmbedder.single_calls += 1
        return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    def get_embeddings_batch_from_inputs(
        self, *, spatials, input_chws, temporal, sensor, output, backend, device
    ):
        _PrefetchedBatchEmbedder.batch_calls += 1
        return [
            Embedding(data=np.array([float(i)], dtype=np.float32), meta={})
            for i in range(len(spatials))
        ]


def test_prefetched_batch_succeeds(monkeypatch):
    """When prefetched batch API is available and inputs are provided, it's used."""
    _PrefetchedBatchEmbedder.batch_calls = 0
    _PrefetchedBatchEmbedder.single_calls = 0
    embedder = _PrefetchedBatchEmbedder()
    sensor = SensorSpec(
        collection="C", bands=("B1",), scale_m=10, cloudy_pct=30, composite="median"
    )
    spatials = _make_spatials(3)
    _patch_deps(monkeypatch, embedder, supports_prefetched_batch=True)

    kw = _base_kwargs(models=["m1"], spatials=spatials, provider_enabled=True)
    kw["resolved_sensor"] = {"m1": sensor}
    manifest = run_pending_models(**kw)

    assert _PrefetchedBatchEmbedder.batch_calls >= 1
    assert _PrefetchedBatchEmbedder.single_calls == 0
    m_entry = manifest["models"][0]
    assert m_entry["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════
# Prefetched batch fails → per-item fallback
# ══════════════════════════════════════════════════════════════════════


class _FailingPrefetchedBatchEmbedder:
    batch_calls = 0
    single_calls = 0

    def describe(self):
        return {"type": "onthefly", "inputs": {"collection": "C", "bands": ["B1"]}}

    def get_embedding(self, **kw):
        _FailingPrefetchedBatchEmbedder.single_calls += 1
        return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    def get_embeddings_batch_from_inputs(self, **kw):
        _FailingPrefetchedBatchEmbedder.batch_calls += 1
        raise RuntimeError("prefetched batch failed")


def test_prefetched_batch_fails_falls_back_to_single(monkeypatch):
    _FailingPrefetchedBatchEmbedder.batch_calls = 0
    _FailingPrefetchedBatchEmbedder.single_calls = 0
    embedder = _FailingPrefetchedBatchEmbedder()
    sensor = SensorSpec(
        collection="C", bands=("B1",), scale_m=10, cloudy_pct=30, composite="median"
    )
    spatials = _make_spatials(2)
    _patch_deps(monkeypatch, embedder, supports_prefetched_batch=True)

    kw = _base_kwargs(models=["m1"], spatials=spatials, provider_enabled=True)
    kw["resolved_sensor"] = {"m1": sensor}
    manifest = run_pending_models(**kw)

    assert _FailingPrefetchedBatchEmbedder.batch_calls >= 1
    assert _FailingPrefetchedBatchEmbedder.single_calls == 2
    m_entry = manifest["models"][0]
    assert m_entry["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════
# Per-item errors with continue_on_error
# ══════════════════════════════════════════════════════════════════════


class _PartialFailEmbedder:
    """Embedder that fails on even indices."""

    def describe(self):
        return {"type": "onthefly"}

    def get_embedding(self, *, spatial, temporal, sensor, output, backend, device, input_chw=None):
        # Use lon to determine which index this is (our spatials have distinct lons)
        if round(spatial.lon, 1) == -88.0:  # index 0
            raise RuntimeError("fail on index 0")
        return Embedding(data=np.array([1.0], dtype=np.float32), meta={})


def test_continue_on_error_records_partial_failures(monkeypatch):
    """With continue_on_error=True, partial failures are recorded but don't stop the run."""
    embedder = _PartialFailEmbedder()
    spatials = _make_spatials(3)
    _patch_deps(monkeypatch, embedder)

    kw = _base_kwargs(
        models=["m1"],
        spatials=spatials,
        provider_enabled=False,
        continue_on_error=True,
    )
    manifest = run_pending_models(**kw)

    m_entry = manifest["models"][0]
    assert m_entry["status"] == "partial"
    assert 0 in m_entry["failed_indices"]


def test_continue_on_error_false_raises(monkeypatch):
    """With continue_on_error=False, first failure raises."""
    embedder = _PartialFailEmbedder()
    spatials = _make_spatials(3)
    _patch_deps(monkeypatch, embedder)

    kw = _base_kwargs(
        models=["m1"],
        spatials=spatials,
        provider_enabled=False,
        continue_on_error=False,
    )
    with pytest.raises(RuntimeError, match="fail on index 0"):
        run_pending_models(**kw)


# ══════════════════════════════════════════════════════════════════════
# All items fail
# ══════════════════════════════════════════════════════════════════════


class _AlwaysFailEmbedder:
    def describe(self):
        return {"type": "onthefly"}

    def get_embedding(self, **kw):
        raise RuntimeError("always fail")


def test_all_items_fail_marks_model_failed(monkeypatch):
    embedder = _AlwaysFailEmbedder()
    spatials = _make_spatials(2)
    _patch_deps(monkeypatch, embedder)

    kw = _base_kwargs(
        models=["m1"],
        spatials=spatials,
        provider_enabled=False,
        continue_on_error=True,
    )
    manifest = run_pending_models(**kw)

    m_entry = manifest["models"][0]
    assert m_entry["status"] == "failed"
    assert m_entry["embeddings"] is None


# ══════════════════════════════════════════════════════════════════════
# inference_strategy="single" bypasses batch
# ══════════════════════════════════════════════════════════════════════


def test_single_strategy_bypasses_batch(monkeypatch):
    _BatchEmbedder.batch_calls = 0
    _BatchEmbedder.single_calls = 0
    embedder = _BatchEmbedder()
    spatials = _make_spatials(2)
    _patch_deps(monkeypatch, embedder, supports_batch=True)

    kw = _base_kwargs(
        models=["m1"],
        spatials=spatials,
        provider_enabled=False,
        inference_strategy="single",
    )
    manifest = run_pending_models(**kw)

    assert _BatchEmbedder.batch_calls == 0
    assert _BatchEmbedder.single_calls == 2
    assert manifest["models"][0]["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════
# save_embeddings=False skips inference
# ══════════════════════════════════════════════════════════════════════


def test_save_embeddings_false_skips_inference(monkeypatch):
    _SingleEmbedder.calls = 0
    embedder = _SingleEmbedder()
    spatials = _make_spatials(2)
    _patch_deps(monkeypatch, embedder)

    kw = _base_kwargs(
        models=["m1"],
        spatials=spatials,
        provider_enabled=False,
        save_embeddings=False,
    )
    manifest = run_pending_models(**kw)

    assert _SingleEmbedder.calls == 0
    m_entry = manifest["models"][0]
    assert m_entry["embeddings"] is None

import threading

import numpy as np

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec
from rs_embed.core.types import ExportConfig, Status
from rs_embed.pipelines.inference import InferenceEngine


class _BatchEmbedder:
    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials,
        input_chws,
        temporal,
        sensor,
        output,
        backend,
        device,
    ):
        out = []
        for inp in input_chws:
            out.append(
                Embedding(
                    data=np.array([float(np.mean(inp))], dtype=np.float32),
                    meta={"source": "prefetched"},
                )
            )
        return out

    def get_embeddings_batch(
        self,
        *,
        spatials,
        temporal,
        sensor,
        output,
        backend,
        device,
    ):
        out = []
        for _ in spatials:
            out.append(
                Embedding(
                    data=np.array([1.0], dtype=np.float32),
                    meta={"source": "batch"},
                )
            )
        return out


def _engine() -> InferenceEngine:
    return InferenceEngine(device="cpu", output=OutputSpec.pooled(), config=ExportConfig())


def _spatials(n: int):
    return [PointBuffer(lon=float(i), lat=0.0, buffer_m=10.0) for i in range(n)]


def test_embedding_to_result_returns_ok_taskresult():
    engine = _engine()
    emb = Embedding(data=np.array([1.0, 2.0], dtype=np.float32), meta={"k": "v"})

    out = engine._embedding_to_result(emb)

    assert out.status == Status.OK
    assert np.allclose(out.embedding, np.array([1.0, 2.0], dtype=np.float32))
    assert out.meta == {"k": "v"}


def test_resolve_model_context_detects_provider_input(monkeypatch):
    engine = _engine()
    fake_embedder = object()
    fake_lock = object()

    monkeypatch.setattr(
        "rs_embed.pipelines.inference.get_embedder_bundle_cached",
        lambda *args, **kwargs: (fake_embedder, fake_lock),
    )

    sensor = SensorSpec(collection="C", bands=("B1", "B2", "B3"))
    ctx = engine._resolve_model_context(
        name="dummy_model",
        backend="gee",
        sensor=sensor,
        is_precomputed=False,
        provider_enabled=True,
    )

    assert ctx.embedder is fake_embedder
    assert ctx.lock is fake_lock
    assert ctx.skey is not None
    assert ctx.needs_provider_input is True


def test_evaluate_batch_capability_respects_gates(monkeypatch):
    engine = _engine()
    embedder = _BatchEmbedder()

    monkeypatch.setattr(
        "rs_embed.pipelines.inference.supports_prefetched_batch_api",
        lambda _e: True,
    )
    monkeypatch.setattr(
        "rs_embed.pipelines.inference.supports_batch_api",
        lambda _e: True,
    )

    can_prefetch, can_no_input = engine._evaluate_batch_capability(
        embedder=embedder,
        needs_provider_input=True,
        sensor=SensorSpec(collection="C", bands=("B1",)),
        skey="sensor::C",
        prefer_batch=True,
        allow_nonresize=True,
    )
    assert can_prefetch is True
    assert can_no_input is False


def test_run_batch_prefetched_returns_results_and_failures():
    engine = _engine()
    done = []

    def _get_input(i: int) -> np.ndarray:
        if i == 1:
            raise RuntimeError("prefetch failed")
        return np.full((3, 2, 2), float(i), dtype=np.float32)

    out, succeeded = engine._run_batch_prefetched(
        idxs=[0, 1, 2],
        spatials=_spatials(3),
        temporal=None,
        sensor=SensorSpec(collection="C", bands=("B1",)),
        embedder=_BatchEmbedder(),
        lock=threading.Lock(),
        backend="gee",
        get_input_fn=_get_input,
        batch_size=2,
        continue_on_error=True,
        on_done=lambda i: done.append(i),
        use_lock=False,
        model_name="dummy",
    )

    assert succeeded is True
    assert set(out.keys()) == {0, 1, 2}
    assert out[0].status == Status.OK
    assert out[2].status == Status.OK
    assert out[1].status == Status.FAILED
    assert set(done) == {0, 1, 2}


def test_run_batch_no_input_returns_false_on_length_mismatch():
    engine = _engine()

    class _MismatchEmbedder:
        def get_embeddings_batch(self, **kwargs):
            return [Embedding(data=np.array([1.0], dtype=np.float32), meta={})]

    out, succeeded = engine._run_batch_no_input(
        idxs=[0, 1],
        spatials=_spatials(2),
        temporal=None,
        sensor=None,
        embedder=_MismatchEmbedder(),
        lock=threading.Lock(),
        backend="gee",
        batch_size=8,
        on_done=lambda _i: None,
        use_lock=False,
        model_name="dummy",
    )

    assert succeeded is False
    assert out == {}


def test_run_single_fallback_skips_done_and_records_errors():
    engine = _engine()
    done = []

    def _infer_one(i: int) -> Embedding:
        if i == 2:
            raise RuntimeError("boom")
        return Embedding(data=np.array([float(i)], dtype=np.float32), meta={"i": i})

    out = engine._run_single_fallback(
        idxs=[0, 1, 2],
        already_done={1},
        infer_one_fn=_infer_one,
        continue_on_error=True,
        on_done=lambda i: done.append(i),
    )

    assert set(out.keys()) == {0, 2}
    assert out[0].status == Status.OK
    assert out[2].status == Status.FAILED
    assert set(done) == {0, 2}

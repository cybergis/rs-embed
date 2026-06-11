import numpy as np

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import BBox, InputPrepSpec, OutputSpec
from rs_embed.core.types import ExportConfig
from rs_embed.tools.tiling import (
    _call_embedder_get_embedding_with_input_prep,
    _tile_yx_starts,
)


class _FakeTileEmbedder:
    model_name = "fake_tile"

    def __init__(self):
        self.single_calls = 0
        self.batch_calls = 0
        self.batch_input_shapes = []
        self.batch_input_topleft = []

    def describe(self):
        return {"defaults": {"image_size": 4}}

    def get_embedding(
        self,
        *,
        spatial,
        temporal=None,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
        input_chw=None,
    ):
        self.single_calls += 1
        x = np.asarray(input_chw, dtype=np.float32)
        if output.mode == "grid":
            return Embedding(
                data=x[:1].copy(),
                meta={
                    "y_axis_direction": "north_to_south",
                    "grid_hw": (int(x.shape[-2]), int(x.shape[-1])),
                },
            )
        return Embedding(data=np.asarray([float(x.mean())], dtype=np.float32), meta={})

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials,
        input_chws,
        temporal=None,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
    ):
        self.batch_calls += 1
        self.batch_input_shapes.extend([tuple(np.asarray(x).shape) for x in input_chws])
        self.batch_input_topleft.extend([float(np.asarray(x)[0, 0, 0]) for x in input_chws])
        return [
            self.get_embedding(
                spatial=sp,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
                input_chw=x,
            )
            for sp, x in zip(spatials, input_chws, strict=True)
        ]


def _bbox():
    return BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0)


def test_tiled_grid_stitch_restores_shape_and_values():
    emb = _FakeTileEmbedder()
    x = np.arange(36, dtype=np.float32).reshape(1, 6, 6)

    out = _call_embedder_get_embedding_with_input_prep(
        embedder=emb,
        spatial=_bbox(),
        temporal=None,
        sensor=None,
        output=OutputSpec.grid(),
        backend="gee",
        device="cpu",
        input_chw=x,
        input_prep=InputPrepSpec.tile(tile_size=4, max_tiles=9, pad_edges=True),
    )

    arr = np.asarray(out.data, dtype=np.float32)
    assert arr.shape == (1, 6, 6)
    np.testing.assert_allclose(arr, x)
    assert out.meta["input_prep"]["resolved_mode"] == "tile"
    assert out.meta["input_prep"]["tile_count"] == 4
    assert out.meta["input_prep"]["tile_layout"] == "cover_shift"
    assert out.meta["input_prep"]["stitch_policy"] == "midpoint_cut"
    assert out.meta["grid_hw"] == (6, 6)
    assert emb.batch_calls >= 1


def test_tiled_pooled_mean_uses_area_weighted_merge():
    emb = _FakeTileEmbedder()
    x = np.arange(36, dtype=np.float32).reshape(1, 6, 6)

    out = _call_embedder_get_embedding_with_input_prep(
        embedder=emb,
        spatial=_bbox(),
        temporal=None,
        sensor=None,
        output=OutputSpec.pooled("mean"),
        backend="gee",
        device="cpu",
        input_chw=x,
        input_prep=InputPrepSpec.tile(tile_size=4, max_tiles=9, pad_edges=False),
    )

    arr = np.asarray(out.data, dtype=np.float32)
    assert arr.shape == (1,)
    np.testing.assert_allclose(arr[0], float(x.mean()), rtol=0, atol=1e-6)
    assert out.meta["input_prep"]["merged_output"] == "pooled_reduce"


def test_tiled_single_path_falls_back_when_batch_lacks_model_config():
    class _BatchWithoutModelConfig:
        model_name = "batch_without_model_config"

        def __init__(self):
            self.batch_calls = 0
            self.single_model_configs = []

        def describe(self):
            return {"defaults": {"image_size": 4}}

        def get_embedding(
            self,
            *,
            spatial,
            temporal=None,
            sensor=None,
            model_config=None,
            output=OutputSpec.pooled(),
            backend="gee",
            device="cpu",
            input_chw=None,
        ):
            self.single_model_configs.append(model_config)
            return Embedding(
                data=np.asarray([float(model_config["value"])], dtype=np.float32),
                meta={},
            )

        def get_embeddings_batch_from_inputs(
            self,
            *,
            spatials,
            input_chws,
            temporal=None,
            sensor=None,
            output=OutputSpec.pooled(),
            backend="gee",
            device="cpu",
        ):
            self.batch_calls += 1
            return [
                Embedding(data=np.asarray([0.0], dtype=np.float32), meta={}) for _ in input_chws
            ]

    emb = _BatchWithoutModelConfig()
    out = _call_embedder_get_embedding_with_input_prep(
        embedder=emb,
        spatial=_bbox(),
        temporal=None,
        sensor=None,
        output=OutputSpec.pooled("mean"),
        backend="gee",
        device="cpu",
        input_chw=np.ones((1, 5, 5), dtype=np.float32),
        input_prep=InputPrepSpec.tile(tile_size=4, max_tiles=16),
        model_config={"value": 7.0},
    )

    assert emb.batch_calls == 0
    assert emb.single_model_configs == [{"value": 7.0}] * 4
    np.testing.assert_allclose(np.asarray(out.data), np.asarray([7.0], dtype=np.float32))
    assert out.meta["input_prep"]["tile_count"] == 4


def test_auto_mode_falls_back_to_resize_when_tile_budget_exceeded():
    emb = _FakeTileEmbedder()
    x = np.arange(36, dtype=np.float32).reshape(1, 6, 6)

    out = _call_embedder_get_embedding_with_input_prep(
        embedder=emb,
        spatial=_bbox(),
        temporal=None,
        sensor=None,
        output=OutputSpec.grid(),
        backend="gee",
        device="cpu",
        input_chw=x,
        input_prep=InputPrepSpec.auto(tile_size=4, max_tiles=1),
    )

    arr = np.asarray(out.data, dtype=np.float32)
    assert arr.shape == (1, 6, 6)
    np.testing.assert_allclose(arr, x)
    assert "input_prep" not in (out.meta or {})


def test_cover_shift_tile_positions_avoid_padding_for_300_with_224():
    ys, xs = _tile_yx_starts(h=300, w=300, tile_size=224, stride=224)
    assert ys == [0, 76]
    assert xs == [0, 76]

    emb = _FakeTileEmbedder()
    x = np.arange(300 * 300, dtype=np.float32).reshape(1, 300, 300)
    out = _call_embedder_get_embedding_with_input_prep(
        embedder=emb,
        spatial=_bbox(),
        temporal=None,
        sensor=None,
        output=OutputSpec.grid(),
        backend="gee",
        device="cpu",
        input_chw=x,
        input_prep=InputPrepSpec.tile(tile_size=224),
    )

    assert out.meta["input_prep"]["tile_count"] == 4
    assert sorted(emb.batch_input_shapes) == [(1, 224, 224)] * 4
    assert sorted(int(v) for v in emb.batch_input_topleft) == [0, 76, 22800, 22876]


# ---------------------------------------------------------------------------
# Tiled batch inference across multiple spatial points
# ---------------------------------------------------------------------------


class _FakeTileEmbedderWithBase(_FakeTileEmbedder):
    """Subclass that satisfies EmbedderBase contract checks used by InferenceEngine."""

    def describe(self):
        return {"defaults": {"image_size": 4}}

    def get_embedding(
        self,
        *,
        spatial,
        temporal=None,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
        input_chw=None,
    ):
        self.single_calls += 1
        x = np.asarray(input_chw, dtype=np.float32)
        if output.mode == "grid":
            return Embedding(
                data=x[:1].copy(),
                meta={
                    "y_axis_direction": "north_to_south",
                    "grid_hw": (int(x.shape[-2]), int(x.shape[-1])),
                },
            )
        return Embedding(data=np.asarray([float(x.mean())], dtype=np.float32), meta={})


def _make_engine(tile_size: int = 4, max_tiles: int = 16) -> tuple:
    from rs_embed.pipelines.inference import InferenceEngine

    cfg = ExportConfig(input_prep=InputPrepSpec.tile(tile_size=tile_size, max_tiles=max_tiles))
    engine = InferenceEngine(device="cpu", output=OutputSpec.pooled("mean"), config=cfg)
    return engine


def test_run_batch_tiled_aggregates_multiple_points():
    """_run_batch_tiled should tile each image and return one result per spatial point."""
    import threading

    emb = _FakeTileEmbedderWithBase()
    lock = threading.RLock()
    engine = _make_engine(tile_size=4)

    n_points = 3
    # Each image is 6x6 → 4 tiles of size 4x4
    images = {
        i: np.arange(36, dtype=np.float32).reshape(1, 6, 6) * (i + 1) for i in range(n_points)
    }
    spatials = [
        BBox(minlon=float(i), minlat=0.0, maxlon=float(i + 1), maxlat=1.0) for i in range(n_points)
    ]

    done_indices: list[int] = []

    out, succeeded = engine._run_batch_tiled(
        idxs=list(range(n_points)),
        spatials=spatials,
        temporal=None,
        sensor=None,
        embedder=emb,
        lock=lock,
        backend="gee",
        get_input_fn=lambda i: images[i],
        batch_size=16,
        continue_on_error=False,
        on_done=done_indices.append,
        use_lock=False,
        model_name="fake_tile",
    )

    assert succeeded, "tiled batch should succeed"
    assert set(out.keys()) == set(range(n_points))
    for i in range(n_points):
        assert out[i].status.value == "ok", f"point {i} failed: {out[i].error}"
        assert out[i].embedding is not None

    assert sorted(done_indices) == list(range(n_points))
    # 3 points × 4 tiles each = 12 total tiles processed via batch API
    assert emb.batch_calls >= 1
    total_tiles = sum(len(s) for s in [emb.batch_input_shapes])
    assert total_tiles == n_points * 4


def test_run_batch_tiled_honors_max_tiles():
    """_run_batch_tiled should reject inputs that exceed input_prep.max_tiles."""
    import threading

    emb = _FakeTileEmbedderWithBase()
    lock = threading.RLock()
    engine = _make_engine(tile_size=4, max_tiles=1)
    image = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
    done_indices: list[int] = []

    out, succeeded = engine._run_batch_tiled(
        idxs=[0],
        spatials=[_bbox()],
        temporal=None,
        sensor=None,
        embedder=emb,
        lock=lock,
        backend="gee",
        get_input_fn=lambda i: image,
        batch_size=16,
        continue_on_error=True,
        on_done=done_indices.append,
        use_lock=False,
        model_name="fake_tile",
    )

    assert succeeded
    assert out[0].status.value == "failed"
    assert "would create 4 tiles (> max_tiles=1)" in str(out[0].error)
    assert done_indices == [0]
    assert emb.batch_calls == 0
    assert emb.batch_input_shapes == []


def test_run_batch_tiled_continue_on_error_skips_oversized_point():
    """A point exceeding max_tiles must not abort stitching for the valid points."""
    import threading

    emb = _FakeTileEmbedderWithBase()
    lock = threading.RLock()
    engine = _make_engine(tile_size=4, max_tiles=1)

    # Index 1 (5x5 → 4 tiles) exceeds max_tiles=1; indices 0 and 2 (4x4 → 1 tile) are valid.
    images = {
        0: np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        1: np.arange(25, dtype=np.float32).reshape(1, 5, 5),
        2: np.arange(16, dtype=np.float32).reshape(1, 4, 4) * 2,
    }
    spatials = [
        BBox(minlon=float(i), minlat=0.0, maxlon=float(i + 1), maxlat=1.0) for i in range(3)
    ]
    done_indices: list[int] = []

    out, succeeded = engine._run_batch_tiled(
        idxs=[0, 1, 2],
        spatials=spatials,
        temporal=None,
        sensor=None,
        embedder=emb,
        lock=lock,
        backend="gee",
        get_input_fn=lambda i: images[i],
        batch_size=16,
        continue_on_error=True,
        on_done=done_indices.append,
        use_lock=False,
        model_name="fake_tile",
    )

    assert succeeded, "tiled batch should succeed despite one oversized point"
    assert set(out.keys()) == {0, 1, 2}
    assert out[0].status.value == "ok", f"point 0 failed: {out[0].error}"
    assert out[2].status.value == "ok", f"point 2 failed: {out[2].error}"
    assert out[1].status.value == "failed"
    assert "would create 4 tiles (> max_tiles=1)" in str(out[1].error)
    assert sorted(done_indices) == [0, 1, 2]


def test_run_batch_tiled_falls_back_when_batch_lacks_model_config():
    """_run_batch_tiled should not silently drop model_config for tiled batch calls."""
    import threading

    class _BatchWithoutModelConfig(_FakeTileEmbedderWithBase):
        def __init__(self):
            super().__init__()
            self.batch_calls = 0

        def get_embedding(
            self,
            *,
            spatial,
            temporal=None,
            sensor=None,
            model_config=None,
            output=OutputSpec.pooled(),
            backend="gee",
            device="cpu",
            input_chw=None,
        ):
            return Embedding(
                data=np.asarray([float(model_config["value"])], dtype=np.float32),
                meta={},
            )

        def get_embeddings_batch_from_inputs(
            self,
            *,
            spatials,
            input_chws,
            temporal=None,
            sensor=None,
            output=OutputSpec.pooled(),
            backend="gee",
            device="cpu",
        ):
            self.batch_calls += 1
            return [
                Embedding(data=np.asarray([0.0], dtype=np.float32), meta={}) for _ in input_chws
            ]

    emb = _BatchWithoutModelConfig()
    lock = threading.RLock()
    engine = _make_engine(tile_size=4, max_tiles=16)
    fetched: list[int] = []

    out, succeeded = engine._run_batch_tiled(
        idxs=[0],
        spatials=[_bbox()],
        temporal=None,
        sensor=None,
        embedder=emb,
        lock=lock,
        backend="gee",
        get_input_fn=lambda i: fetched.append(i) or np.ones((1, 5, 5), dtype=np.float32),
        batch_size=16,
        continue_on_error=False,
        on_done=lambda i: None,
        use_lock=False,
        model_name="fake_tile",
        model_config={"value": 7.0},
    )

    assert not succeeded
    assert out == {}
    assert fetched == []
    assert emb.batch_calls == 0


def test_infer_chunk_tile_mode_uses_batch_on_gpu(monkeypatch):
    """infer_chunk with input_prep='tile' should reach _run_batch_tiled when GPU available."""
    import threading

    from rs_embed.pipelines import inference as inf_mod

    emb = _FakeTileEmbedderWithBase()
    lock = threading.RLock()

    # Force prefer_batch=True without a real GPU
    monkeypatch.setattr(inf_mod, "_device_has_gpu", lambda _: True)

    from rs_embed.core.specs import InputPrepSpec, OutputSpec
    from rs_embed.core.types import ExportConfig, ModelConfig
    from rs_embed.pipelines.inference import InferenceEngine

    cfg = ExportConfig(input_prep=InputPrepSpec.tile(tile_size=4, max_tiles=16))
    engine = InferenceEngine(device="cpu", output=OutputSpec.pooled("mean"), config=cfg)

    n = 2
    images = {i: np.ones((1, 6, 6), dtype=np.float32) * (i + 1) for i in range(n)}
    spatials = [
        BBox(minlon=float(i), minlat=0.0, maxlon=float(i + 1), maxlat=1.0) for i in range(n)
    ]

    tiled_called: list[bool] = []
    original_run_batch_tiled = engine._run_batch_tiled

    def _spy_tiled(**kwargs):
        tiled_called.append(True)
        return original_run_batch_tiled(**kwargs)

    engine._run_batch_tiled = _spy_tiled

    from rs_embed.core.specs import SensorSpec
    from rs_embed.tools.serialization import sensor_cache_key

    sensor = SensorSpec(collection="FAKE/S2", bands=("B2", "B3", "B4"), scale_m=10, fill_value=0.0)
    skey = sensor_cache_key(sensor)
    prefetch_cache = {(i, skey): images[i] for i in range(n)}

    # Patch resolve_model_context to return our fake embedder
    from rs_embed.pipelines.inference import _ModelContext

    monkeypatch.setattr(
        engine,
        "_resolve_model_context",
        lambda **kw: _ModelContext(
            embedder=emb,
            lock=lock,
            sensor_k=("s2",),
            skey=skey,
            needs_provider_input=True,
        ),
    )

    mc = ModelConfig(name="fake_tile", backend="gee", sensor=sensor)
    out = engine.infer_chunk(
        idxs=list(range(n)),
        spatials=spatials,
        temporal=None,
        models=[mc],
        prefetch_cache=prefetch_cache,
        prefetch_errors={},
    )

    assert len(tiled_called) >= 1, "_run_batch_tiled was not invoked"
    for i in range(n):
        key = (i, mc.name)
        assert key in out, f"missing result for {key}"
        assert out[key].status.value == "ok"

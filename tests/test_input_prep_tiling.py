import numpy as np
import pytest

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import BBox, InputPrepSpec, OutputSpec
from rs_embed.core.types import ExportConfig
from rs_embed.tools.tiling import (
    INPUT_PREP_VERSION,
    _call_embedder_get_embedding_with_input_prep,
    _slice_and_pad_tile,
    _snap_axis_to_tile,
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


def test_tiled_resize_model_skips_padding_on_short_axis():
    """Wide/flat ROI: resize-capable model tiles the long axis, resizes the short.

    When one dimension is shorter than ``tile_size`` and the other is long
    enough to tile, a model that resizes each tile to a fixed input size needs
    no padding at all — the short axis is fed at its natural size and upsampled.
    No fabricated pixels, no boundary "dead band". (regression: Prithvi grid
    over a wide ROI.)
    """
    emb = _FakeTileEmbedder()  # describe().defaults.image_size = 4 → resize-capable
    # H=3 (< tile_size 4 → single y-tile, fed unpadded), W=6 (> 4 → two x-tiles).
    x = np.arange(18, dtype=np.float32).reshape(1, 3, 6)

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
    # No padding occurred (model resizes tiles), so the original values come back
    # exactly and the short axis stays at its native 3 rows.
    assert arr.shape == (1, 3, 6)
    np.testing.assert_allclose(arr, x)
    assert out.meta["grid_hw"] == (3, 6)
    assert out.meta["input_prep"]["stitched_grid_shape"] == (3, 6)
    # Even though pad_edges=True was requested, the resize-capable model path
    # disables padding entirely.
    assert out.meta["input_prep"]["pad_edges"] is False
    assert out.meta["input_prep"]["pad_policy"] == "none_model_resizes_tiles"


def test_tiled_resize_model_anisotropic_tall_thin():
    """Tall/thin ROI is the symmetric case: tile the long (height) axis."""
    emb = _FakeTileEmbedder()
    # H=6 (> 4 → two y-tiles), W=3 (< 4 → single x-tile, fed unpadded).
    x = np.arange(18, dtype=np.float32).reshape(1, 6, 3)

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
    assert arr.shape == (1, 6, 3)
    np.testing.assert_allclose(arr, x)
    assert out.meta["input_prep"]["pad_policy"] == "none_model_resizes_tiles"


def test_slice_and_pad_tile_replicates_edge_not_constant_fill():
    """Edge padding must replicate boundary pixels, not write a constant fill.

    A patch/ViT model tokenizes the padded tile; constant (zero) padding turns
    the patch straddling the valid/pad boundary into an out-of-distribution
    token that renders as a flat "dead band". Edge replication keeps that patch
    on real surface values. (regression: residual band on Prithvi's short edge.)
    """
    # Distinct per-row values so a replicated edge is visibly different from a
    # constant fill of 0.0.
    x = np.arange(1, 4, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32)
    x = x[None]  # (1, 3, 3): rows are 1.0, 2.0, 3.0

    tile, meta = _slice_and_pad_tile(x, y0=0, x0=0, tile_size=5, pad_edges=True, fill_value=0.0)
    assert tile.shape == (1, 5, 5)
    assert meta["valid_h"] == 3 and meta["valid_w"] == 3
    # Bottom padding rows replicate the last valid row (3.0), not the 0.0 fill.
    np.testing.assert_allclose(tile[0, 3:, :3], 3.0)
    # Right padding cols replicate the last valid col of each row.
    np.testing.assert_allclose(tile[0, :3, 3:], np.broadcast_to(x[0, :, 2:3], (3, 2)))
    assert not np.any(tile == 0.0)


def test_tiled_nonresize_model_edge_pads_and_crops():
    """Models without a fixed input size keep the edge-replicate + crop fallback.

    A model that advertises no image_size may require exactly ``tile_size``
    tiles, so the short axis is edge-padded to ``tile_size`` and the stitcher
    crops the padded fraction back off (the replicated pad never reaches output).
    """

    class _NonResizePassthrough:
        model_name = "nonresize_passthrough"

        def describe(self):
            return {"defaults": {}}  # no image_size → not resize-capable

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
            xx = np.asarray(input_chw, dtype=np.float32)
            return Embedding(
                data=xx[:1].copy(),
                meta={"grid_hw": (int(xx.shape[-2]), int(xx.shape[-1]))},
            )

    emb = _NonResizePassthrough()
    # H=3 (< tile_size 4 → single y-tile, edge-padded to 4), W=6 (> 4 → two x-tiles).
    x = np.arange(1, 19, dtype=np.float32).reshape(1, 3, 6)

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
    # Padded 4th row cropped → native 3 rows; replicated pad never reaches output.
    assert arr.shape == (1, 3, 6)
    np.testing.assert_allclose(arr, x)
    assert out.meta["input_prep"]["pad_edges"] is True
    assert out.meta["input_prep"]["pad_policy"] == "edge_replicate"


def test_tiled_resize_model_warns_on_extreme_short_axis_upsample():
    """A very short axis (>4x upsample) warns that its resolution is interpolated."""
    emb = _FakeTileEmbedder()  # resize-capable
    # tile_size=20: H=4 (20/4 = 5x > 4x → warn), W=40 (> 20 → tiled).
    x = np.arange(4 * 40, dtype=np.float32).reshape(1, 4, 40)

    with pytest.warns(UserWarning, match="upsamples the height"):
        _call_embedder_get_embedding_with_input_prep(
            embedder=emb,
            spatial=_bbox(),
            temporal=None,
            sensor=None,
            output=OutputSpec.grid(),
            backend="gee",
            device="cpu",
            input_chw=x,
            input_prep=InputPrepSpec.tile(tile_size=20, max_tiles=16, pad_edges=True),
        )


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
        # tile_snap_frac=0 keeps the 5x5/tile-4 cover-shift overlap (4 tiles) so this
        # test exercises the multi-tile single-call fallback; snapping would otherwise
        # collapse the 1px overhang to a single 4x4 tile.
        input_prep=InputPrepSpec.tile(tile_size=4, max_tiles=16, tile_snap_frac=0.0),
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
    # Auto fell back to a plain resize, but the embedding is still stamped with a
    # versioned input_prep block for reproducibility provenance.
    prep = out.meta["input_prep"]
    assert prep["requested_mode"] == "auto"
    assert prep["resolved_mode"] == "resize"
    assert prep["prep_version"] == INPUT_PREP_VERSION
    assert "tile_count" not in prep


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


def _make_engine(tile_size: int = 4, max_tiles: int = 16, max_tiles_hard: int = 1024) -> tuple:
    from rs_embed.pipelines.inference import InferenceEngine

    cfg = ExportConfig(
        input_prep=InputPrepSpec.tile(
            tile_size=tile_size, max_tiles=max_tiles, max_tiles_hard=max_tiles_hard
        )
    )
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
    """_run_batch_tiled should reject inputs that exceed input_prep.max_tiles_hard."""
    import threading

    emb = _FakeTileEmbedderWithBase()
    lock = threading.RLock()
    engine = _make_engine(tile_size=4, max_tiles=1, max_tiles_hard=1)
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
    assert "would create 4 tiles (> max_tiles_hard=1)" in str(out[0].error)
    assert done_indices == [0]
    assert emb.batch_calls == 0
    assert emb.batch_input_shapes == []


def test_run_batch_tiled_continue_on_error_skips_oversized_point():
    """A point exceeding max_tiles must not abort stitching for the valid points."""
    import threading

    emb = _FakeTileEmbedderWithBase()
    lock = threading.RLock()
    engine = _make_engine(tile_size=4, max_tiles=1, max_tiles_hard=1)

    # Index 1 (5x5 → 4 tiles) exceeds max_tiles_hard=1; indices 0 and 2 (4x4 → 1 tile) are valid.
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
    assert "would create 4 tiles (> max_tiles_hard=1)" in str(out[1].error)
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


def test_snap_axis_to_tile_unit():
    # Small overhang past a tile multiple -> snap down to the multiple.
    assert _snap_axis_to_tile(68, tile_size=64, snap_frac=0.25) == 64
    assert _snap_axis_to_tile(130, tile_size=64, snap_frac=0.25) == 128  # 2px over 128
    assert _snap_axis_to_tile(5, tile_size=4, snap_frac=0.25) == 4
    # Large overhang -> already small overlap, keep native (no fabricated pixels).
    assert _snap_axis_to_tile(96, tile_size=64, snap_frac=0.25) == 96
    assert _snap_axis_to_tile(126, tile_size=64, snap_frac=0.25) == 126
    # Exact multiple and sub-tile dims are untouched.
    assert _snap_axis_to_tile(128, tile_size=64, snap_frac=0.25) == 128
    assert _snap_axis_to_tile(40, tile_size=64, snap_frac=0.25) == 40
    # frac=0 disables snapping entirely.
    assert _snap_axis_to_tile(68, tile_size=64, snap_frac=0.0) == 68


def test_snap_disabled_keeps_native_overlap_tiling():
    """With snapping off, a 1px overhang still spawns the cover-shift overlap tiles."""
    emb = _FakeTileEmbedder()
    out = _call_embedder_get_embedding_with_input_prep(
        embedder=emb,
        spatial=_bbox(),
        temporal=None,
        sensor=None,
        output=OutputSpec.grid(),
        backend="gee",
        device="cpu",
        input_chw=np.ones((1, 5, 5), dtype=np.float32),
        input_prep=InputPrepSpec.tile(tile_size=4, max_tiles=16, tile_snap_frac=0.0),
    )
    assert out.meta["input_prep"]["resolved_mode"] == "tile"
    assert out.meta["input_prep"]["tile_count"] == 4


def test_snap_collapses_small_overhang_to_single_tile():
    """A 1px overhang on both axes snaps away, collapsing 4 overlap tiles into one."""
    pytest.importorskip("torch")
    emb = _FakeTileEmbedder()
    out = _call_embedder_get_embedding_with_input_prep(
        embedder=emb,
        spatial=_bbox(),
        temporal=None,
        sensor=None,
        output=OutputSpec.grid(),
        backend="gee",
        device="cpu",
        input_chw=np.ones((1, 5, 5), dtype=np.float32),
        # snap_frac=0.25 -> round(0.25*4)=1px threshold catches the 1px overhang
        # (the default 0.1 rounds to a 0px threshold at this toy tile_size=4).
        input_prep=InputPrepSpec.tile(tile_size=4, max_tiles=16, tile_snap_frac=0.25),
    )
    arr = np.asarray(out.data, dtype=np.float32)
    # Single 4x4 tile fed after snapping 5x5 -> 4x4; no overlap-stitch inflation.
    assert arr.shape[-2:] == (4, 4)
    # Single-tile path reports a plain resize, and only one model call ran.
    assert emb.single_calls + emb.batch_calls >= 1
    assert out.meta["input_prep"]["resolved_mode"] == "resize"


def test_snap_reduces_tiles_and_records_meta_on_multitile_path():
    """When snapping still leaves >1 tile, the stitch meta records the snap."""
    pytest.importorskip("torch")
    emb = _FakeTileEmbedder()
    # H=5 (overhang 1 -> snap to 4), W=9 (overhang 1 -> snap to 8 -> two x-tiles).
    out = _call_embedder_get_embedding_with_input_prep(
        embedder=emb,
        spatial=_bbox(),
        temporal=None,
        sensor=None,
        output=OutputSpec.grid(),
        backend="gee",
        device="cpu",
        input_chw=np.ones((1, 5, 9), dtype=np.float32),
        input_prep=InputPrepSpec.tile(tile_size=4, max_tiles=16, tile_snap_frac=0.25),
    )
    prep = out.meta["input_prep"]
    assert prep["resolved_mode"] == "tile"
    assert prep["tile_count"] == 2
    assert prep["snapped_from_hw"] == (5, 9)
    assert prep["snapped_to_hw"] == (4, 8)
    assert np.asarray(out.data).shape[-2:] == (4, 8)

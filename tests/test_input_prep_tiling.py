import numpy as np

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import BBox, InputPrepSpec, OutputSpec
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
            for sp, x in zip(spatials, input_chws)
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

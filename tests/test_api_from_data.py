"""Tests for the bring-your-own-data API (get_embedding_from_data & friends).

These use mock embedders registered in the test so they don't require GEE,
torch, or any real model weights. list_models_for_data is additionally
exercised against the real catalog, which only reads static describe()
metadata.
"""

import numpy as np
import pytest

from rs_embed import (
    UserData,
    get_embedding_from_data,
    get_embeddings_batch_from_data,
    list_models_for_data,
)
from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.errors import ModelError, SpecError
from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders.base import EmbedderBase

S2 = "COPERNICUS/S2_SR_HARMONIZED"
_POINT = PointBuffer(lon=-88.2, lat=40.1, buffer_m=320)


class _MockFromDataEmbedder(EmbedderBase):
    """Captures the inputs it receives; no I/O."""

    model_name = "mock_from_data"
    last_input = None
    last_sensor = None
    last_model_config = None
    seen_temporals: list = []

    def describe(self):
        return {
            "type": "mock",
            "backend": ["gee", "auto"],
            "output": ["pooled", "grid"],
            "inputs": {"collection": S2, "bands": ["B4", "B3", "B2"]},
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
        type(self).last_input = input_chw
        type(self).last_sensor = sensor
        type(self).last_model_config = model_config
        type(self).seen_temporals.append(temporal)
        return Embedding(
            data=np.arange(4, dtype=np.float32),
            meta={"model": self.model_name, "output": output.mode, "temporal": temporal},
        )


class _MockPrecomputedFromDataEmbedder(EmbedderBase):
    model_name = "mock_from_data_precomputed"
    _is_precomputed = True

    def describe(self):
        return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}


class _MockGeorefFromDataEmbedder(_MockFromDataEmbedder):
    model_name = "mock_from_data_georef"
    _requires_georef = True


class _MockChunkTrackingEmbedder(_MockFromDataEmbedder):
    model_name = "mock_from_data_chunks"
    seen_chunk_sizes: list = []

    def get_embeddings_batch_from_inputs(self, *, spatials, input_chws, **kwargs):
        type(self).seen_chunk_sizes.append(len(input_chws))
        return super().get_embeddings_batch_from_inputs(
            spatials=spatials, input_chws=input_chws, **kwargs
        )


class _MockTiledFromDataEmbedder(_MockFromDataEmbedder):
    """Advertises a tile size (defaults.image_size) so the tiler engages."""

    model_name = "mock_from_data_tiled"
    seen_input_shapes: list = []
    from_inputs_calls: int = 0

    def describe(self):
        desc = super().describe()
        desc["defaults"] = {"image_size": 4}
        return desc

    def get_embedding(self, **kwargs):
        x = kwargs.get("input_chw")
        if x is not None:
            type(self).seen_input_shapes.append(tuple(np.asarray(x).shape))
        return super().get_embedding(**kwargs)

    def get_embeddings_batch_from_inputs(self, *, spatials, input_chws, **kwargs):
        type(self).from_inputs_calls += 1
        return super().get_embeddings_batch_from_inputs(
            spatials=spatials, input_chws=input_chws, **kwargs
        )


@pytest.fixture(autouse=True)
def register_mocks():
    registry.register("mock_from_data")(_MockFromDataEmbedder)
    registry.register("mock_from_data_precomputed")(_MockPrecomputedFromDataEmbedder)
    registry.register("mock_from_data_georef")(_MockGeorefFromDataEmbedder)
    registry.register("mock_from_data_chunks")(_MockChunkTrackingEmbedder)
    registry.register("mock_from_data_tiled")(_MockTiledFromDataEmbedder)
    _MockChunkTrackingEmbedder.seen_chunk_sizes = []
    _MockTiledFromDataEmbedder.seen_input_shapes = []
    _MockTiledFromDataEmbedder.from_inputs_calls = 0
    _MockFromDataEmbedder.last_input = None
    _MockFromDataEmbedder.last_sensor = None
    _MockFromDataEmbedder.last_model_config = None
    _MockFromDataEmbedder.seen_temporals = []
    yield


def _twelve_band_userdata(
    *, fill_by_channel=True, hw=(8, 8), temporal=None, declare_bands=True, spatial=_POINT
):
    bands = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")
    if fill_by_channel:
        arr = np.stack([np.full(hw, i * 1000.0, dtype=np.float32) for i in range(len(bands))])
    else:
        arr = np.full((len(bands), *hw), 5000.0, dtype=np.float32)
    return UserData(
        data=arr,
        collection="s2",
        spatial=spatial,
        bands=bands if declare_bands else None,
        temporal=temporal,
    )


# ── single ─────────────────────────────────────────────────────────


def test_superset_data_is_sliced_to_model_band_order():
    emb = get_embedding_from_data("mock_from_data", _twelve_band_userdata())
    x = _MockFromDataEmbedder.last_input
    assert x is not None and x.shape == (3, 8, 8)
    # model order (B4, B3, B2) -> channels 3, 2, 1 of the declared 12-band cube
    assert [float(x[i, 0, 0]) for i in range(3)] == [3000.0, 2000.0, 1000.0]
    assert emb.data.shape == (4,)


def test_omitted_bands_default_to_canonical_s2_order():
    emb = get_embedding_from_data("mock_from_data", _twelve_band_userdata(declare_bands=False))
    x = _MockFromDataEmbedder.last_input
    assert [float(x[i, 0, 0]) for i in range(3)] == [3000.0, 2000.0, 1000.0]
    assert emb.meta["user_input"]["declared_bands"][:3] == ["B1", "B2", "B3"]


def test_temporal_travels_with_the_data():
    t = TemporalSpec.year(2022)
    emb = get_embedding_from_data("mock_from_data", _twelve_band_userdata(temporal=t))
    assert emb.meta["temporal"] == t


def test_user_input_meta_records_declaration_and_selection():
    emb = get_embedding_from_data("mock_from_data", _twelve_band_userdata())
    ui = emb.meta["user_input"]
    assert ui["source"] == "user_data"
    assert ui["collection"] == S2
    assert ui["bands_used"] == ["B4", "B3", "B2"]
    assert ui["channel_indices"] == [3, 2, 1]


def test_missing_band_refuses_with_band_name():
    data = UserData(
        data=np.full((2, 8, 8), 5000.0, dtype=np.float32),
        collection="s2",
        spatial=_POINT,
        bands=("B2", "B3"),
    )
    with pytest.raises(ModelError, match=r"lacks \['B4'\]"):
        get_embedding_from_data("mock_from_data", data)


def test_collection_mismatch_refuses():
    data = UserData(
        data=np.full((3, 8, 8), 0.5, dtype=np.float32),
        collection="s1",
        spatial=_POINT,
        bands=("VV", "VH", "ANGLE"),
    )
    with pytest.raises(ModelError, match="collection"):
        get_embedding_from_data("mock_from_data", data)


def test_precomputed_model_refuses():
    with pytest.raises(ModelError, match="precomputed"):
        get_embedding_from_data("mock_from_data_precomputed", _twelve_band_userdata())


def test_missing_spatial_is_accepted_by_non_georef_models():
    emb = get_embedding_from_data("mock_from_data", _twelve_band_userdata(spatial=None))
    assert emb.data.shape == (4,)
    assert _MockFromDataEmbedder.last_input is not None


def test_missing_spatial_refuses_georef_conditioned_models():
    with pytest.raises(ModelError, match="conditions on request geometry"):
        get_embedding_from_data("mock_from_data_georef", _twelve_band_userdata(spatial=None))


def test_georef_model_accepts_data_with_spatial():
    emb = get_embedding_from_data("mock_from_data_georef", _twelve_band_userdata())
    assert emb.data.shape == (4,)


def test_invalid_userdata_raises_specerror():
    bad = UserData(
        data=np.zeros((4, 8, 8), dtype=np.float32),
        collection="s2",
        spatial=_POINT,
        bands=("B2", "B3"),
    )
    with pytest.raises(SpecError, match="channels"):
        get_embedding_from_data("mock_from_data", bad)


def test_model_kwargs_are_forwarded_as_model_config():
    get_embedding_from_data("mock_from_data", _twelve_band_userdata(), variant="large")
    assert _MockFromDataEmbedder.last_model_config == {"variant": "large"}


def test_output_mode_is_passed_through():
    emb = get_embedding_from_data(
        "mock_from_data",
        _twelve_band_userdata(),
        output=OutputSpec.pooled(),
    )
    assert emb.meta["output"] == "pooled"


# ── batch ──────────────────────────────────────────────────────────


def test_batch_returns_one_embedding_per_item():
    datas = [_twelve_band_userdata(), _twelve_band_userdata()]
    embs = get_embeddings_batch_from_data("mock_from_data", datas)
    assert len(embs) == 2
    assert all(e.meta["user_input"]["channel_indices"] == [3, 2, 1] for e in embs)


def test_batch_groups_by_temporal_and_preserves_order():
    t22, t23 = TemporalSpec.year(2022), TemporalSpec.year(2023)
    datas = [
        _twelve_band_userdata(temporal=t22),
        _twelve_band_userdata(temporal=t23),
        _twelve_band_userdata(temporal=t22),
    ]
    embs = get_embeddings_batch_from_data("mock_from_data", datas)
    assert [e.meta["temporal"] for e in embs] == [t22, t23, t22]
    # two distinct temporals -> two dispatches covering all three items
    assert sorted(_MockFromDataEmbedder.seen_temporals, key=str) == sorted([t22, t22, t23], key=str)


def test_batch_empty_refuses():
    with pytest.raises(ModelError, match="non-empty"):
        get_embeddings_batch_from_data("mock_from_data", [])


def test_batch_size_chunks_the_dispatch():
    datas = [_twelve_band_userdata() for _ in range(5)]
    embs = get_embeddings_batch_from_data("mock_from_data_chunks", datas, batch_size=2)
    assert len(embs) == 5
    assert _MockChunkTrackingEmbedder.seen_chunk_sizes == [2, 2, 1]


def test_batch_size_default_dispatches_once():
    datas = [_twelve_band_userdata() for _ in range(3)]
    get_embeddings_batch_from_data("mock_from_data_chunks", datas)
    assert _MockChunkTrackingEmbedder.seen_chunk_sizes == [3]


def test_invalid_batch_size_refuses():
    with pytest.raises(ModelError, match="batch_size"):
        get_embeddings_batch_from_data("mock_from_data", [_twelve_band_userdata()], batch_size=0)


# ── input_prep: tile default vs resize ─────────────────────────────


def test_large_input_is_tiled_by_default():
    # 8x8 input, model tile size 4 -> 2x2 grid of native-resolution 4x4 tiles.
    emb = get_embedding_from_data("mock_from_data_tiled", _twelve_band_userdata(hw=(8, 8)))
    assert _MockTiledFromDataEmbedder.seen_input_shapes == [(3, 4, 4)] * 4
    prep = emb.meta["input_prep"]
    assert prep["resolved_mode"] == "tile"
    assert emb.meta["user_input"]["bands_used"] == ["B4", "B3", "B2"]


def test_input_prep_resize_opts_out_of_tiling():
    emb = get_embedding_from_data(
        "mock_from_data_tiled", _twelve_band_userdata(hw=(8, 8)), input_prep="resize"
    )
    # One call with the full-size array; the embedder handles the downsample.
    assert _MockTiledFromDataEmbedder.seen_input_shapes == [(3, 8, 8)]
    assert emb.meta["input_prep"]["resolved_mode"] == "resize"


def test_tile_sized_inputs_keep_batched_dispatch_under_tile_default():
    # Items a single tile already covers cannot tile -> one batched dispatch.
    datas = [_twelve_band_userdata(hw=(4, 4)), _twelve_band_userdata(hw=(4, 4))]
    embs = get_embeddings_batch_from_data("mock_from_data_tiled", datas)
    assert _MockTiledFromDataEmbedder.from_inputs_calls == 1
    assert _MockTiledFromDataEmbedder.seen_input_shapes == [(3, 4, 4)] * 2
    assert all(e.meta["input_prep"]["resolved_mode"] == "resize" for e in embs)


# ── list_models_for_data over the real catalog ─────────────────────


def test_list_models_for_data_over_catalog_with_12_band_s2():
    report = list_models_for_data(_twelve_band_userdata(fill_by_channel=False))
    by_model = {r["model"]: r for r in report}

    # Precomputed models are incompatible with an explicit reason.
    for name in ("tessera", "gse", "copernicus"):
        assert not by_model[name]["compatible"]
        assert "precomputed" in by_model[name]["reason"]

    # MODIS-based SatVision cannot take S2 data.
    assert not by_model["satvision"]["compatible"]

    # A 12-band S2 L2A cube serves the S2-default models.
    for name in ("galileo", "prithvi", "clay", "scalemae", "satmae"):
        assert by_model[name]["compatible"], by_model[name]["reason"]
        assert by_model[name]["bands_used"]


def test_list_models_for_data_without_spatial_flags_georef_models():
    report = list_models_for_data(_twelve_band_userdata(fill_by_channel=False, spatial=None))
    by_model = {r["model"]: r for r in report}
    for name in ("clay", "prithvi"):
        assert not by_model[name]["compatible"]
        assert "geometry" in by_model[name]["reason"]
    assert by_model["galileo"]["compatible"]


def test_list_models_for_data_rgb_only_declaration():
    data = UserData(
        data=np.full((3, 8, 8), 5000.0, dtype=np.float32),
        collection="s2",
        spatial=_POINT,
        bands=("B4", "B3", "B2"),
    )
    report = list_models_for_data(data)
    by_model = {r["model"]: r for r in report}
    assert by_model["scalemae"]["compatible"]
    assert not by_model["galileo"]["compatible"]
    assert "lacks" in by_model["galileo"]["reason"]

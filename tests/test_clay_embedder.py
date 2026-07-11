"""Clay embedder unit tests (no network / weights required).

Covers: official preprocessing alignment (metadata.yaml stats + metadata
encodings), checkpoint state-dict extraction, ROI crop-back semantics of
``build_clay_embedding``, and the tensor/batch plumbing contracts.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from rs_embed.core.errors import ModelError
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.embedders.onthefly_clay import (
    _CLAY_S2_BANDS,
    _CLAY_S2_MEAN,
    _CLAY_S2_STD,
    _CLAY_S2_WAVELENGTHS_UM,
    ClayEmbedder,
    _clay_time_vec,
    _extract_clay_encoder_state_dict,
    _normalize_clay_input_chw,
    _normalize_clay_latlon,
    _normalize_clay_timestamp,
    build_clay_embedding,
)


# -----------------------------
# Preprocessing alignment (official Clay metadata.yaml, sentinel-2-l2a)
# -----------------------------
def test_clay_s2_normalization_matches_official_stats():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.0, 8000.0, size=(10, 6, 6)).astype(np.float32)
    x, waves, meta = _normalize_clay_input_chw(raw, bands=list(_CLAY_S2_BANDS), input_name="t")
    expected = (raw - _CLAY_S2_MEAN[:, None, None]) / _CLAY_S2_STD[:, None, None]
    np.testing.assert_allclose(x, expected, rtol=1e-6)
    np.testing.assert_allclose(waves, _CLAY_S2_WAVELENGTHS_UM)
    assert meta["normalization"] == "official_clay_metadata_s2_stats"


def test_clay_s2_normalization_band_subset_keeps_alignment():
    raw = np.full((3, 4, 4), 3000.0, dtype=np.float32)
    bands = ["B4", "B8", "B11"]  # red, nir, swir16
    x, waves, _ = _normalize_clay_input_chw(raw, bands=bands, input_name="t")
    np.testing.assert_allclose(waves, [0.665, 0.842, 1.61])
    np.testing.assert_allclose(
        x[:, 0, 0],
        (3000.0 - np.array([1552.0, 2743.0, 2388.0])) / np.array([1888.0, 1742.0, 1470.0]),
        rtol=1e-6,
    )


def test_clay_normalization_rejects_prenormalized_input():
    raw = np.full((10, 4, 4), 0.5, dtype=np.float32)
    with pytest.raises(ModelError, match="already normalized"):
        _normalize_clay_input_chw(raw, bands=list(_CLAY_S2_BANDS), input_name="t")


def test_clay_normalization_rejects_unknown_bands():
    raw = np.full((2, 4, 4), 3000.0, dtype=np.float32)
    with pytest.raises(ModelError, match="only defined for Sentinel-2"):
        _normalize_clay_input_chw(raw, bands=["B1", "B9"], input_name="t")


def test_clay_timestamp_encoding_matches_official_recipe():
    dt = datetime.fromisoformat("2023-07-15")
    vec = _normalize_clay_timestamp(dt)
    week = dt.isocalendar()[1] * 2 * math.pi / 52
    assert vec.shape == (4,)
    np.testing.assert_allclose(vec[0], math.sin(week), rtol=1e-6)
    np.testing.assert_allclose(vec[1], math.cos(week), rtol=1e-6)
    # midnight midpoint -> hour encoding is (sin 0, cos 0) = (0, 1)
    np.testing.assert_allclose(vec[2:], [0.0, 1.0], atol=1e-7)


def test_clay_latlon_encoding_matches_official_recipe():
    vec = _normalize_clay_latlon(0.0, 0.0)
    np.testing.assert_allclose(vec, [0.0, 1.0, 0.0, 1.0], atol=1e-7)
    vec = _normalize_clay_latlon(45.0, -90.0)
    np.testing.assert_allclose(
        vec,
        [
            math.sin(math.radians(45.0)),
            math.cos(math.radians(45.0)),
            math.sin(math.radians(-90.0)),
            math.cos(math.radians(-90.0)),
        ],
        rtol=1e-6,
    )


def test_clay_time_vec_from_year_temporal():
    vec, mid = _clay_time_vec(TemporalSpec.year(2021))
    assert mid == "2021-07-02"
    assert vec.shape == (4,)


# -----------------------------
# Checkpoint state-dict extraction
# -----------------------------
def test_extract_clay_encoder_state_dict_filters_lightning_ckpt():
    payload = {
        "state_dict": {
            "model.encoder.cls_token": 1,
            "model.encoder.transformer.norm.weight": 2,
            "model.decoder.mask_patch": 3,
            "model.teacher.blocks.0.attn.qkv.weight": 4,
            "model.proj.weight": 5,
        },
        "hyper_parameters": {"model_size": "large"},
    }
    sd = _extract_clay_encoder_state_dict(payload)
    assert sd == {"cls_token": 1, "transformer.norm.weight": 2}


def test_extract_clay_encoder_state_dict_accepts_bare_encoder_prefix():
    sd = _extract_clay_encoder_state_dict({"encoder.cls_token": 7})
    assert sd == {"cls_token": 7}


def test_extract_clay_encoder_state_dict_rejects_missing_encoder():
    with pytest.raises(ModelError, match="no encoder weights"):
        _extract_clay_encoder_state_dict({"state_dict": {"model.decoder.x": 1}})


# -----------------------------
# ROI crop-back semantics (fetch-square contract)
# -----------------------------
def _tokens_grid(side: int = 4, dim: int = 3) -> tuple[np.ndarray, np.ndarray]:
    n = side * side
    tokens = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    cls_vec = np.full((dim,), -1.0, dtype=np.float32)
    return tokens, cls_vec


def test_build_clay_embedding_full_frame_grid_and_cls_pooled():
    tokens, cls_vec = _tokens_grid()
    e = build_clay_embedding(
        tokens, cls_vec, geo_roi=None, output=OutputSpec.grid(), base_meta={"model": "clay"}
    )
    assert e.data.shape == (3, 4, 4)
    np.testing.assert_allclose(np.asarray(e.data).transpose(1, 2, 0).reshape(16, 3), tokens)

    e = build_clay_embedding(
        tokens, cls_vec, geo_roi=None, output=OutputSpec.pooled(), base_meta={"model": "clay"}
    )
    assert e.meta["pooling"] == "model_cls"
    np.testing.assert_allclose(e.data, cls_vec)


def test_build_clay_embedding_roi_crops_grid_and_pools_roi_tokens():
    tokens, cls_vec = _tokens_grid()
    roi = (0.0, 0.5, 0.0, 0.5)  # top-left quadrant
    e = build_clay_embedding(
        tokens, cls_vec, geo_roi=roi, output=OutputSpec.grid(), base_meta={"model": "clay"}
    )
    assert e.data.shape == (3, 2, 2)

    e_pool = build_clay_embedding(
        tokens, cls_vec, geo_roi=roi, output=OutputSpec.pooled(), base_meta={"model": "clay"}
    )
    assert e_pool.meta["pooling"] == "roi_grid_mean"
    grid = tokens.reshape(4, 4, 3)
    manual = grid[:2, :2, :].reshape(4, 3).mean(axis=0)
    np.testing.assert_allclose(e_pool.data, manual, rtol=1e-6)


def test_build_clay_embedding_rejects_non_square_tokens():
    tokens = np.zeros((15, 3), dtype=np.float32)
    with pytest.raises(ModelError, match="not square"):
        build_clay_embedding(
            tokens,
            np.zeros(3, dtype=np.float32),
            geo_roi=None,
            output=OutputSpec.grid(),
            base_meta={},
        )


# -----------------------------
# Embedder plumbing (fake model / no network)
# -----------------------------
class _FakeModel:
    pass


def _fake_load(*, model_size, device):
    return _FakeModel(), {"device": "cpu", "model_size": model_size, "patch_size": 8}


def _fake_forward(model, x_bchw, *, wavelengths_um, gsd_m, times_b4, latlons_b4, device):
    b = int(x_bchw.shape[0])
    tokens = np.ones((b, 16, 2), dtype=np.float32)
    cls_vec = np.stack(
        [np.array([float(latlons_b4[i, 0]), 2.0], dtype=np.float32) for i in range(b)]
    )
    return tokens, cls_vec, {"token_count": 16, "token_dim": 2}


def test_clay_tensor_get_embedding_uses_input_chw(monkeypatch):
    import rs_embed.embedders.onthefly_clay as clay

    emb = ClayEmbedder()
    seen = {}

    def _spy_forward(model, x_bchw, *, wavelengths_um, gsd_m, times_b4, latlons_b4, device):
        seen["shape"] = tuple(x_bchw.shape)
        seen["waves"] = [float(v) for v in wavelengths_um]
        seen["gsd"] = gsd_m
        seen["time"] = times_b4[0].tolist()
        seen["latlon"] = latlons_b4[0].tolist()
        return _fake_forward(
            model,
            x_bchw,
            wavelengths_um=wavelengths_um,
            gsd_m=gsd_m,
            times_b4=times_b4,
            latlons_b4=latlons_b4,
            device=device,
        )

    monkeypatch.setattr(clay, "_load_clay_model", _fake_load)
    monkeypatch.setattr(clay, "_clay_forward_tokens_and_cls_batch", _spy_forward)

    sensor = SensorSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_CLAY_S2_BANDS),
    )
    out = emb.get_embedding(
        spatial=PointBuffer(lon=-88.2, lat=40.1, buffer_m=256),
        temporal=TemporalSpec.range("2023-06-01", "2023-09-01"),
        sensor=sensor,
        output=OutputSpec.pooled(),
        backend="tensor",
        input_chw=np.full((10, 8, 8), 5000.0, dtype=np.float32),
    )

    assert seen["shape"] == (1, 10, 256, 256)
    assert seen["waves"] == [float(v) for v in _CLAY_S2_WAVELENGTHS_UM]
    assert seen["gsd"] == 10.0
    np.testing.assert_allclose(seen["latlon"], _normalize_clay_latlon(40.1, -88.2), rtol=1e-6)
    assert out.data.shape == (2,)
    assert out.meta["pooling"] == "model_cls"


def test_clay_prefetched_input_needs_no_provider(monkeypatch):
    """get_embedding(input_chw=...) on the provider backend must not acquire a
    live provider (lazy-provider convention from the batch-5 review fixes) and
    must record the resolved temporal window in meta."""
    import rs_embed.embedders.onthefly_clay as clay

    emb = ClayEmbedder()

    def _boom(_backend):
        raise AssertionError("prefetched input must not touch the provider")

    monkeypatch.setattr(emb, "_get_provider", _boom)
    monkeypatch.setattr(clay, "_load_clay_model", _fake_load)
    monkeypatch.setattr(clay, "_clay_forward_tokens_and_cls_batch", _fake_forward)

    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2021),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=np.full((10, 8, 8), 5000.0, dtype=np.float32),
        fetch_meta={"roi_window_geo": (0.0, 1.0, 0.0, 0.5)},
    )
    assert out.data.shape == (2,)
    assert out.meta["pooling"] == "roi_grid_mean"  # ROI window honored
    # meta records the resolved range, not the raw year request
    assert out.meta["temporal"]["mode"] == "range"
    assert out.meta["temporal"]["start"] == "2021-01-01"


def test_clay_get_embedding_uses_model_config_model_size(monkeypatch):
    import rs_embed.embedders.onthefly_clay as clay

    emb = ClayEmbedder()
    seen = {}

    def _spy_load(*, model_size, device):
        seen["model_size"] = model_size
        return _fake_load(model_size=model_size, device=device)

    monkeypatch.setattr(clay, "_load_clay_model", _spy_load)
    monkeypatch.setattr(clay, "_clay_forward_tokens_and_cls_batch", _fake_forward)

    emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=None,
        sensor=SensorSpec(collection="COPERNICUS/S2_SR_HARMONIZED", bands=tuple(_CLAY_S2_BANDS)),
        model_config={"model_size": "base"},
        output=OutputSpec.pooled(),
        backend="tensor",
        input_chw=np.full((10, 8, 8), 5000.0, dtype=np.float32),
    )
    assert seen["model_size"] == "base"


def test_clay_batch_from_inputs_crops_per_item_roi(monkeypatch):
    import rs_embed.embedders.onthefly_clay as clay

    emb = ClayEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(clay, "_load_clay_model", _fake_load)
    monkeypatch.setattr(clay, "_clay_forward_tokens_and_cls_batch", _fake_forward)

    spatials = [
        PointBuffer(lon=0.0, lat=10.0, buffer_m=256),
        PointBuffer(lon=1.0, lat=20.0, buffer_m=256),
    ]
    input_chws = [np.full((10, 8, 8), 5000.0, dtype=np.float32) for _ in spatials]

    out = emb.get_embeddings_batch_from_inputs(
        spatials=spatials,
        input_chws=input_chws,
        temporal=TemporalSpec.range("2023-06-01", "2023-09-01"),
        output=OutputSpec.grid(),
        backend="auto",
        _roi_windows_geo=[None, (0.0, 0.5, 0.0, 1.0)],
    )

    assert len(out) == 2
    assert out[0].data.shape == (2, 4, 4)  # full 4x4 token grid
    assert out[1].data.shape == (2, 2, 4)  # cropped to top half

    # pooled: item without ROI window uses the model CLS vector, and the CLS
    # rows must stay aligned per item (fake CLS encodes each item's latlon)
    out_pooled = emb.get_embeddings_batch_from_inputs(
        spatials=spatials,
        input_chws=input_chws,
        temporal=TemporalSpec.range("2023-06-01", "2023-09-01"),
        output=OutputSpec.pooled(),
        backend="auto",
    )
    np.testing.assert_allclose(
        out_pooled[0].data[0], _normalize_clay_latlon(10.0, 0.0)[0], rtol=1e-6
    )
    np.testing.assert_allclose(
        out_pooled[1].data[0], _normalize_clay_latlon(20.0, 1.0)[0], rtol=1e-6
    )


def test_clay_batch_from_inputs_reads_fetch_metas_roi(monkeypatch):
    import rs_embed.embedders.onthefly_clay as clay

    emb = ClayEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(clay, "_load_clay_model", _fake_load)
    monkeypatch.setattr(clay, "_clay_forward_tokens_and_cls_batch", _fake_forward)

    out = emb.get_embeddings_batch_from_inputs(
        spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=256)],
        input_chws=[np.full((10, 8, 8), 5000.0, dtype=np.float32)],
        temporal=None,
        output=OutputSpec.grid(),
        backend="auto",
        fetch_metas=[{"roi_window_geo": (0.0, 1.0, 0.0, 0.5)}],
    )
    assert out[0].data.shape == (2, 4, 2)  # cropped to left half


def test_clay_describe_is_serializable():
    import json

    d = ClayEmbedder().describe()
    json.dumps(d)
    assert d["type"] == "on_the_fly"
    assert d["defaults"]["model_size"] == "large"
    assert d["defaults"]["image_size"] == 256

"""Tests for OlmoEarth embedder (onthefly_olmoearth.py)."""

from __future__ import annotations

import numpy as np
import pytest

from rs_embed.core.embedding import Embedding
from rs_embed.core.errors import ModelError
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.embedders import onthefly_olmoearth as oe

# ---------------------------------------------------------------------------
# Variant normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("nano", "nano"),
        ("NANO", "nano"),
        ("Nano", "nano"),
        ("nano_v1", "nano"),
        ("tiny", "tiny"),
        ("base", "base"),
        ("large", "large"),
        ("large_v1", "large"),
        ("nano_v1_1", "nano_v1_1"),
        ("NANO_V1_1", "nano_v1_1"),
        ("nano_11", "nano_v1_1"),
        ("tiny_v1_1", "tiny_v1_1"),
        ("tiny_11", "tiny_v1_1"),
        ("base_v1_1", "base_v1_1"),
        ("base_11", "base_v1_1"),
    ],
)
def test_normalize_variant_valid(raw, expected):
    assert oe._normalize_variant(raw) == expected


def test_normalize_variant_raises_on_unknown():
    with pytest.raises(ModelError, match="Unknown OlmoEarth variant"):
        oe._normalize_variant("xlarge")


# ---------------------------------------------------------------------------
# _resolve_variant
# ---------------------------------------------------------------------------


def test_resolve_variant_from_model_config():
    assert oe._resolve_variant({"variant": "base"}) == "base"
    assert oe._resolve_variant({"variant": "nano_v1_1"}) == "nano_v1_1"


def test_resolve_variant_defaults_to_tiny_v1_1():
    assert oe._resolve_variant(None) == "tiny_v1_1"
    assert oe._resolve_variant({}) == "tiny_v1_1"


def test_resolve_variant_env_fallback(monkeypatch):
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_VARIANT", "tiny")
    assert oe._resolve_variant(None) == "tiny"


def test_resolve_variant_model_config_overrides_env(monkeypatch):
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_VARIANT", "tiny")
    assert oe._resolve_variant({"variant": "base"}) == "base"


def test_resolve_variant_env_auto_gives_default(monkeypatch):
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_VARIANT", "auto")
    assert oe._resolve_variant(None) == "tiny_v1_1"


# ---------------------------------------------------------------------------
# _resolve_patch_size / _resolve_image_size
# ---------------------------------------------------------------------------


def test_resolve_patch_size_defaults():
    assert oe._resolve_patch_size(None) == oe._DEFAULT_PATCH_SIZE


def test_resolve_patch_size_from_config():
    assert oe._resolve_patch_size({"patch_size": 8}) == 8
    assert oe._resolve_patch_size({"patch_size": "2"}) == 2


def test_resolve_patch_size_rejects_out_of_range():
    with pytest.raises(ModelError, match="patch_size must be 1"):
        oe._resolve_patch_size({"patch_size": 9})


def test_resolve_image_size_defaults():
    assert oe._resolve_image_size(None) == oe._DEFAULT_IMAGE_SIZE


def test_resolve_image_size_from_config():
    assert oe._resolve_image_size({"image_size": 128}) == 128


def test_resolve_patch_size_env_validated(monkeypatch):
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_PATCH_SIZE", "16")
    with pytest.raises(ModelError, match="patch_size must be 1"):
        oe._resolve_patch_size(None)
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_PATCH_SIZE", "abc")
    with pytest.raises(ModelError, match="must be an integer"):
        oe._resolve_patch_size(None)
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_PATCH_SIZE", "2")
    assert oe._resolve_patch_size(None) == 2


def test_resolve_patch_size_rejects_non_integer_config():
    with pytest.raises(ModelError, match="must be an integer"):
        oe._resolve_patch_size({"patch_size": "four"})


def test_resolve_image_size_env_validated(monkeypatch):
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_IMAGE_SIZE", "abc")
    with pytest.raises(ModelError, match="must be an integer"):
        oe._resolve_image_size(None)
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_IMAGE_SIZE", "128")
    assert oe._resolve_image_size(None) == 128


def test_resolve_image_size_rejects_non_positive():
    with pytest.raises(ModelError, match="must be positive"):
        oe._resolve_image_size({"image_size": 0})


def test_resolve_geometry_enforces_divisibility():
    assert oe._resolve_geometry({"image_size": 128, "patch_size": 8}) == (128, 8)
    with pytest.raises(ModelError, match="divisible by"):
        oe._resolve_geometry({"image_size": 257, "patch_size": 4})


# ---------------------------------------------------------------------------
# _date_to_timestamp
# ---------------------------------------------------------------------------


def test_date_to_timestamp_known_date():
    day, month, year = oe._date_to_timestamp("2022-07-15")
    assert day == 15
    assert month == 6  # 0-indexed: July = 6
    assert year == 2022


def test_date_to_timestamp_january():
    day, month, year = oe._date_to_timestamp("2020-01-01")
    assert month == 0  # January = 0


def test_date_to_timestamp_none_returns_default():
    day, month, year = oe._date_to_timestamp(None)
    assert isinstance(day, int)
    assert isinstance(month, int)
    assert isinstance(year, int)


# ---------------------------------------------------------------------------
# Modality helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("s2", "s2"),
        ("S2", "s2"),
        ("s1", "s1"),
        ("S1", "s1"),
        (None, "s2"),
        ("", "s2"),
    ],
)
def test_normalize_modality_valid(raw, expected):
    assert oe._normalize_modality(raw) == expected


def test_normalize_modality_raises_on_unknown():
    with pytest.raises(ModelError, match="modality must be"):
        oe._normalize_modality("landsat")


def test_modality_constants():
    assert oe._modality_n_bands("s2") == 12
    assert oe._modality_n_bands("s1") == 2
    assert oe._modality_n_band_sets("s2") == 3
    assert oe._modality_n_band_sets("s1") == 1
    assert oe._modality_field("s2") == "sentinel2_l2a"
    assert oe._modality_field("s1") == "sentinel1"


def test_s1_linear_to_db():
    lin = np.array([[[1.0, 10.0], [100.0, 0.0]]] * 2, dtype=np.float32)
    db = oe._s1_linear_to_db(lin)
    assert db.dtype == np.float32
    np.testing.assert_allclose(db[0, 0, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(db[0, 0, 1], 10.0, atol=1e-5)
    np.testing.assert_allclose(db[0, 1, 0], 20.0, atol=1e-5)
    assert np.isfinite(db).all()  # zeros clamp to eps, not -inf


# ---------------------------------------------------------------------------
# _normalize_chw
# ---------------------------------------------------------------------------


def test_normalize_chw_output_shape_and_dtype():
    raw = np.random.uniform(0, 3000, (12, 32, 32)).astype(np.float32)
    out = oe._normalize_chw(raw)
    assert out.shape == (12, 32, 32)
    assert out.dtype == np.float32


def test_normalize_chw_no_nans():
    raw = np.zeros((12, 16, 16), dtype=np.float32)
    out = oe._normalize_chw(raw)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


def test_normalize_chw_s1_shape_and_no_nans():
    raw = np.random.uniform(-25.0, 0.0, (2, 16, 16)).astype(np.float32)  # dB
    out = oe._normalize_chw(raw, modality="s1")
    assert out.shape == (2, 16, 16)
    assert out.dtype == np.float32
    assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# _prepare_chw
# ---------------------------------------------------------------------------


def test_prepare_chw_resizes_to_image_size():
    raw = np.random.uniform(0, 3000, (12, 64, 64)).astype(np.float32)
    out = oe._prepare_chw(raw, image_size=256, patch_size=4)
    assert out.shape == (12, 256, 256)


def test_prepare_chw_rejects_non_divisible_image_size():
    raw = np.random.uniform(0, 3000, (12, 64, 64)).astype(np.float32)
    with pytest.raises(ModelError, match="positive multiple"):
        oe._prepare_chw(raw, image_size=257, patch_size=4)


def test_prepare_chw_wrong_channels_raises():
    bad = np.zeros((6, 32, 32), dtype=np.float32)
    with pytest.raises(ModelError, match="12-band"):
        oe._prepare_chw(bad, image_size=256, patch_size=4)


def test_prepare_chw_s1_resizes_and_checks_channels():
    raw = np.random.uniform(-25.0, 0.0, (2, 64, 64)).astype(np.float32)
    out = oe._prepare_chw(raw, image_size=256, patch_size=4, modality="s1")
    assert out.shape == (2, 256, 256)
    with pytest.raises(ModelError, match="2-band"):
        oe._prepare_chw(
            np.zeros((12, 32, 32), dtype=np.float32),
            image_size=256,
            patch_size=4,
            modality="s1",
        )


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


def test_olmoearth_registered_in_catalog():
    from rs_embed.embedders.catalog import MODEL_SPECS

    assert "olmoearth" in MODEL_SPECS
    module, cls_name = MODEL_SPECS["olmoearth"]
    assert module == "onthefly_olmoearth"
    assert cls_name == "OlmoEarthEmbedder"


def test_olmoearth_lazy_loads_via_registry():
    from rs_embed.core.registry import get_embedder_cls

    cls = get_embedder_cls("olmoearth")
    assert cls is oe.OlmoEarthEmbedder


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


def test_describe_returns_expected_keys():
    emb = oe.OlmoEarthEmbedder()
    info = emb.describe()
    assert info["type"] == "onthefly"
    assert "pooled" in info["output"]
    assert "grid" in info["output"]
    assert "model_config" in info
    assert "variant" in info["model_config"]
    assert len(info["input_bands"]) == 12
    assert info["defaults"]["modality"] == "s2"
    assert set(info["modalities"]) == {"s2", "s1"}
    assert info["modalities"]["s1"]["bands"] == ["VV", "VH"]
    assert info["modalities"]["s1"]["defaults"]["use_float_linear"] is False


# ---------------------------------------------------------------------------
# get_embedding — mocked forward
# ---------------------------------------------------------------------------


def _fake_encoder_output(batch_size: int = 1, hw: int = 4, d: int = 128, modality: str = "s2"):
    """Return a fake tokens_and_masks NamedTuple-like object."""
    import torch
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import (  # type: ignore
        TokensAndMasks,
    )

    n_sets = 3 if modality == "s2" else 1
    tokens = torch.randn(batch_size, hw, hw, 1, n_sets, d)  # (B, H', W', T, S, D)
    mask = torch.zeros(batch_size, hw, hw, 1, n_sets)  # all ONLINE_ENCODER
    field = oe._modality_field(modality)
    return TokensAndMasks(**{field: tokens, f"{field}_mask": mask})


def _s1_sensor() -> SensorSpec:
    from rs_embed.tools.model_defaults import default_sensor_for_model

    sensor = default_sensor_for_model("olmoearth", modality="s1")
    assert sensor is not None
    return sensor


def test_get_embedding_pooled_returns_embedding(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())

    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128),
    )

    fake_model = object()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (fake_model, {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    out = emb.get_embedding(
        spatial=PointBuffer(lon=10.0, lat=20.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert isinstance(out, Embedding)
    assert out.data.ndim == 1
    assert out.data.dtype == np.float32
    assert out.meta["model"] == "olmoearth"
    assert "variant" in out.meta
    assert "hf_repo" in out.meta


def test_get_embedding_grid_returns_dataarray(monkeypatch):
    xr = pytest.importorskip("xarray")
    emb = oe.OlmoEarthEmbedder()

    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128),
    )
    fake_model = object()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (fake_model, {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    out = emb.get_embedding(
        spatial=PointBuffer(lon=10.0, lat=20.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=None,
        output=OutputSpec.grid(),
        backend="gee",
    )

    assert isinstance(out, Embedding)
    assert isinstance(out.data, xr.DataArray)
    assert set(out.data.dims) == {"d", "y", "x"}
    assert out.meta["grid_kind"] == "spatial_tokens"


def test_get_embedding_uses_model_config_variant(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 8, 768),
    )
    seen = {}

    def _fake_load(variant, device):
        seen["variant"] = variant
        return object(), {"hf_repo": "allenai/OlmoEarth-v1-Base"}, "cpu"

    monkeypatch.setattr(oe, "_load_olmoearth", _fake_load)

    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2021),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        model_config={"variant": "base"},
    )

    assert seen["variant"] == "base"
    assert out.meta["variant"] == "base"
    assert out.meta["model_size"] == "base"
    assert out.meta["model_version"] == "v1"


def test_get_embedding_accepts_input_chw(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128),
    )
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1_1-Tiny"}, "cpu"),
    )

    x_chw = np.full((12, 64, 64), 2000.0, dtype=np.float32)
    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=x_chw,
    )

    assert isinstance(out, Embedding)
    assert out.data.ndim == 1
    assert out.meta["temporal_mode"] == "single"
    assert "requested_temporal_mode" not in out.meta


def test_get_embedding_input_tchw_meta_reports_effective_mode(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128),
    )
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1_1-Tiny"}, "cpu"),
    )

    # TCHW override while temporal_mode resolves to the default "single":
    # meta must report the layout actually used, plus the requested mode.
    x_tchw = np.full((2, 12, 64, 64), 2000.0, dtype=np.float32)
    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-15", "2022-08-14"),  # 60d → 2 bins
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=x_tchw,
    )

    assert out.meta["temporal_mode"] == "multi"
    assert out.meta["requested_temporal_mode"] == "single"
    assert out.meta["n_frames"] == 2


def test_get_embedding_wrong_input_chw_raises(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    with pytest.raises(ModelError, match="12 bands"):
        emb.get_embedding(
            spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
            temporal=TemporalSpec.year(2022),
            sensor=None,
            output=OutputSpec.pooled(),
            backend="gee",
            input_chw=np.zeros((6, 32, 32), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# S1 modality — mocked paths
# ---------------------------------------------------------------------------


def test_fetch_input_s1_routes_to_s1_helper_and_keeps_db(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    sensor = _s1_sensor()
    assert sensor.modality == "s1"
    assert sensor.use_float_linear is False

    seen = {}

    def _fake_s1_fetch(provider, *, spatial, temporal, **kw):
        seen.update(kw)
        return np.full((2, 64, 64), -15.0, dtype=np.float32), {"iw_applied": True}

    monkeypatch.setattr(oe, "_fetch_s1_vvvh_raw_chw_with_meta", _fake_s1_fetch)

    fr = emb.fetch_input(
        object(),
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=sensor,
    )
    assert fr is not None
    # dB collection by default → values pass through unconverted
    np.testing.assert_allclose(fr.data, -15.0)
    assert fr.meta == {"iw_applied": True}
    assert seen["use_float_linear"] is False
    assert seen["require_iw"] is True


def test_fetch_input_s1_converts_linear_to_db(monkeypatch):
    from dataclasses import replace

    emb = oe.OlmoEarthEmbedder()
    sensor = replace(_s1_sensor(), use_float_linear=True)

    monkeypatch.setattr(
        oe,
        "_fetch_s1_vvvh_raw_chw_with_meta",
        lambda provider, **kw: (np.full((2, 8, 8), 0.1, dtype=np.float32), {}),
    )

    fr = emb.fetch_input(
        object(),
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=sensor,
    )
    assert fr is not None
    np.testing.assert_allclose(fr.data, -10.0, atol=1e-4)  # 10·log10(0.1)


def test_get_embedding_s1_pooled(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_s1_vvvh_raw_chw_with_meta",
        lambda provider, **kw: (
            np.full((2, 64, 64), -15.0, dtype=np.float32),
            {"iw_applied": True},
        ),
    )

    seen = {}

    def _fake_forward(model, sample, **kw):
        seen["sentinel1"] = sample.sentinel1
        seen["sentinel2_l2a"] = sample.sentinel2_l2a
        return _fake_encoder_output(1, 4, 128, modality="s1")

    monkeypatch.setattr(oe, "_encoder_forward", _fake_forward)
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    out = emb.get_embedding(
        spatial=PointBuffer(lon=10.0, lat=20.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=_s1_sensor(),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert isinstance(out, Embedding)
    assert out.data.ndim == 1
    assert out.meta["modality"] == "s1"
    assert out.meta["s1_fetch"] == {"iw_applied": True}
    # Sample must carry S1 tokens only (B,H,W,T,C=2)
    assert seen["sentinel2_l2a"] is None
    assert seen["sentinel1"] is not None
    assert seen["sentinel1"].shape[-1] == 2


def test_get_embedding_s1_grid(monkeypatch):
    xr = pytest.importorskip("xarray")
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_s1_vvvh_raw_chw_with_meta",
        lambda provider, **kw: (np.full((2, 64, 64), -15.0, dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128, modality="s1"),
    )
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    out = emb.get_embedding(
        spatial=PointBuffer(lon=10.0, lat=20.0, buffer_m=256),
        temporal=TemporalSpec.year(2022),
        sensor=_s1_sensor(),
        output=OutputSpec.grid(),
        backend="gee",
    )

    assert isinstance(out.data, xr.DataArray)
    assert set(out.data.dims) == {"d", "y", "x"}
    assert out.meta["modality"] == "s1"


def test_get_embedding_s1_wrong_input_chw_raises(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )
    with pytest.raises(ModelError, match="2 bands"):
        emb.get_embedding(
            spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
            temporal=TemporalSpec.year(2022),
            sensor=_s1_sensor(),
            output=OutputSpec.pooled(),
            backend="gee",
            input_chw=np.zeros((12, 32, 32), dtype=np.float32),
        )


def test_get_embeddings_batch_from_inputs_s1_pooled(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(
            sample.sentinel1.shape[0], 4, 128, modality="s1"
        ),
    )

    spatials = [PointBuffer(lon=float(i), lat=0.0, buffer_m=256) for i in range(3)]
    input_chws = [np.full((2, 64, 64), -15.0, dtype=np.float32) for _ in range(3)]

    out = emb.get_embeddings_batch_from_inputs(
        spatials=spatials,
        input_chws=input_chws,
        temporal=TemporalSpec.year(2022),
        sensor=_s1_sensor(),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 3
    for e in out:
        assert e.data.shape == (128,)
        assert e.meta["modality"] == "s1"


def test_api_modality_s1_resolves_sensor():
    from rs_embed.tools.model_defaults import (
        modality_profiles_for_model,
        supports_modality_for_model,
    )

    assert supports_modality_for_model("olmoearth", "s1")
    assert supports_modality_for_model("olmoearth", "s2")
    profiles = modality_profiles_for_model("olmoearth")
    s1 = profiles["s1"]
    assert s1.collection == "COPERNICUS/S1_GRD"
    assert s1.bands == ("VV", "VH")
    assert s1.modality == "s1"
    assert s1.use_float_linear is False


# ---------------------------------------------------------------------------
# get_embeddings_batch — mocked paths
# ---------------------------------------------------------------------------


def test_get_embeddings_batch_prefetch_and_dispatch(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )

    captured = {}

    def _fake_batch_from_inputs(**kwargs):
        captured["input_chws"] = kwargs["input_chws"]
        return [
            Embedding(data=np.array([float(i)], dtype=np.float32), meta={})
            for i in range(len(kwargs["spatials"]))
        ]

    monkeypatch.setattr(emb, "get_embeddings_batch_from_inputs", _fake_batch_from_inputs)

    spatials = [PointBuffer(lon=float(i), lat=0.0, buffer_m=256) for i in range(3)]
    out = emb.get_embeddings_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 3
    # Inputs should be raw DN arrays (not pre-normalized)
    assert len(captured["input_chws"]) == 3
    for arr in captured["input_chws"]:
        assert arr.shape == (12, 64, 64)


def test_get_embeddings_batch_from_inputs_pooled(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )
    call_count = {"n": 0}

    def _fake_encoder_forward(model, sample, *, patch_size, device):
        batch_size = sample.sentinel2_l2a.shape[0]
        call_count["n"] += batch_size
        return _fake_encoder_output(batch_size, 4, 128)

    monkeypatch.setattr(oe, "_encoder_forward", _fake_encoder_forward)

    spatials = [PointBuffer(lon=float(i), lat=0.0, buffer_m=256) for i in range(4)]
    input_chws = [np.full((12, 64, 64), float(i + 1) * 500, dtype=np.float32) for i in range(4)]

    out = emb.get_embeddings_batch_from_inputs(
        spatials=spatials,
        input_chws=input_chws,
        temporal=TemporalSpec.year(2022),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 4
    assert call_count["n"] == 4
    for emb_out in out:
        assert isinstance(emb_out, Embedding)
        assert emb_out.data.ndim == 1
        assert emb_out.data.shape == (128,)


def test_get_embeddings_batch_from_inputs_grid(monkeypatch):
    pytest.importorskip("xarray")
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(sample.sentinel2_l2a.shape[0], 4, 128),
    )

    spatials = [PointBuffer(lon=0.0, lat=0.0, buffer_m=256)]
    input_chws = [np.full((12, 64, 64), 2000.0, dtype=np.float32)]

    out = emb.get_embeddings_batch_from_inputs(
        spatials=spatials,
        input_chws=input_chws,
        temporal=TemporalSpec.year(2022),
        output=OutputSpec.grid(),
        backend="gee",
    )

    import xarray as xr

    assert len(out) == 1
    assert isinstance(out[0].data, xr.DataArray)
    assert out[0].data.dims == ("d", "y", "x")


def test_get_embeddings_batch_from_inputs_length_mismatch_raises():
    emb = oe.OlmoEarthEmbedder()
    with pytest.raises(ModelError, match="length mismatch"):
        emb.get_embeddings_batch_from_inputs(
            spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=256)],
            input_chws=[],
            temporal=TemporalSpec.year(2022),
            output=OutputSpec.pooled(),
            backend="gee",
        )


def test_get_embeddings_batch_returns_empty_for_empty_spatials():
    emb = oe.OlmoEarthEmbedder()
    out = emb.get_embeddings_batch(
        spatials=[],
        temporal=TemporalSpec.year(2022),
        output=OutputSpec.pooled(),
        backend="gee",
    )
    assert out == []


# ---------------------------------------------------------------------------
# Meta fields
# ---------------------------------------------------------------------------


def test_embedding_meta_has_required_fields(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())
    monkeypatch.setattr(
        oe,
        "_fetch_collection_patch_chw",
        lambda *a, **kw: np.full((12, 64, 64), 1500.0, dtype=np.float32),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(1, 4, 128),
    )
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    out = emb.get_embedding(
        spatial=PointBuffer(lon=5.0, lat=10.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
    )

    meta = out.meta
    assert meta["model"] == "olmoearth"
    assert meta["type"] == "on_the_fly"
    assert meta["backend"] == "gee"
    assert meta["variant"] == "tiny_v1_1"
    assert meta["model_size"] == "tiny"
    assert meta["model_version"] == "v1.1"
    assert meta["patch_size"] == oe._DEFAULT_PATCH_SIZE
    assert meta["hf_repo"] is not None
    assert "temporal_range" in meta
    assert "image_size" in meta


# ---------------------------------------------------------------------------
# temporal_mode='multi' — 30-day binning
# ---------------------------------------------------------------------------


def test_split_date_range_fixed_days_strides_and_truncates():
    from rs_embed.tools.temporal import split_date_range_fixed_days

    bins = split_date_range_fixed_days("2022-06-15", "2022-07-31", stride_days=30)
    assert bins == (("2022-06-15", "2022-07-15"), ("2022-07-15", "2022-07-31"))

    # full year → 13 bins of 30d, capped at 12
    bins = split_date_range_fixed_days("2022-01-01", "2023-01-01", stride_days=30, max_bins=12)
    assert len(bins) == 12
    assert bins[0] == ("2022-01-01", "2022-01-31")
    assert bins[-1][0] == "2022-11-27"


def test_resolve_temporal_mode_sources(monkeypatch):
    assert oe._resolve_temporal_mode(None) == "single"
    assert oe._resolve_temporal_mode({"temporal_mode": "multi"}) == "multi"
    monkeypatch.setenv("RS_EMBED_OLMOEARTH_TEMPORAL_MODE", "multi")
    assert oe._resolve_temporal_mode(None) == "multi"
    assert oe._resolve_temporal_mode({"temporal_mode": "single"}) == "single"
    with pytest.raises(ModelError, match="temporal_mode must be"):
        oe._resolve_temporal_mode({"temporal_mode": "monthly"})


def test_temporal_bins_caps_at_12():
    bins = oe._temporal_bins(TemporalSpec.range("2022-01-01", "2024-01-01"))
    assert len(bins) == 12
    assert bins[0] == ("2022-01-01", "2022-01-31")


def test_prepare_frames_drops_nan_frames_and_aligns_timestamps():
    bins = (
        ("2022-06-15", "2022-07-15"),
        ("2022-07-15", "2022-08-14"),
        ("2022-08-14", "2022-08-31"),
    )
    x = np.random.uniform(0, 3000, (3, 12, 32, 32)).astype(np.float32)
    x[1] = np.nan  # empty-bin sentinel
    out, ts = oe._prepare_frames(x, bins=bins, modality="s2", image_size=64, patch_size=4)
    assert out.shape == (2, 12, 64, 64)
    assert ts == [(15, 5, 2022), (14, 7, 2022)]  # June and August bin starts


def test_prepare_frames_all_nan_raises():
    bins = (("2022-06-15", "2022-07-15"),)
    x = np.full((1, 2, 8, 8), np.nan, dtype=np.float32)
    with pytest.raises(ModelError, match="All frames"):
        oe._prepare_frames(x, bins=bins, modality="s1", image_size=32, patch_size=4)


def test_prepare_frames_bin_count_mismatch_raises():
    bins = (("2022-06-15", "2022-07-15"), ("2022-07-15", "2022-07-31"))
    x = np.zeros((3, 12, 8, 8), dtype=np.float32)
    with pytest.raises(ModelError, match="30-day bins"):
        oe._prepare_frames(x, bins=bins, modality="s2", image_size=32, patch_size=4)


def test_build_sample_multi_frame_shapes():
    pytest.importorskip("torch")
    x = np.zeros((3, 2, 16, 16), dtype=np.float32)
    ts = [(15, 5, 2022), (15, 6, 2022), (14, 7, 2022)]
    sample = oe._build_sample(x, timestamps=ts, modality="s1")
    assert tuple(sample.sentinel1.shape) == (1, 16, 16, 3, 2)  # (B,H,W,T,C)
    assert tuple(sample.sentinel1_mask.shape) == (1, 16, 16, 3, 1)  # S=1 for s1
    assert tuple(sample.timestamps.shape) == (1, 3, 3)
    assert sample.timestamps[0, 2].tolist() == [14, 7, 2022]


def test_fetch_input_multi_s2_routes_to_binned_helper(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    seen = {}

    def _fake_binned(provider, *, spatial, bins, **kw):
        seen["bins"] = bins
        seen["collection"] = kw["collection"]
        return np.zeros((len(bins), 12, 8, 8), dtype=np.float32), {"frames": [], "n_empty": 0}

    monkeypatch.setattr(oe, "_fetch_collection_binned_raw_tchw", _fake_binned)
    fr = emb.fetch_input(
        object(),
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-15", "2022-07-31"),
        sensor=None or emb._default_sensor(),
        temporal_mode="multi",
    )
    assert fr is not None
    assert fr.data.shape == (2, 12, 8, 8)
    assert fr.meta["temporal_mode"] == "multi"
    assert seen["bins"][0] == ("2022-06-15", "2022-07-15")


def test_fetch_input_multi_s1_converts_linear_and_keeps_nan(monkeypatch):
    from dataclasses import replace

    emb = oe.OlmoEarthEmbedder()
    sensor = replace(_s1_sensor(), use_float_linear=True)

    def _fake_binned_s1(provider, *, spatial, bins, **kw):
        arr = np.full((len(bins), 2, 8, 8), 0.1, dtype=np.float32)
        arr[1] = np.nan  # empty bin sentinel
        return arr, {"frames": [], "n_empty": 1}

    monkeypatch.setattr(oe, "_fetch_s1_vvvh_binned_raw_tchw", _fake_binned_s1)
    fr = emb.fetch_input(
        object(),
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-01", "2022-08-30"),
        sensor=sensor,
        temporal_mode="multi",
    )
    assert fr is not None
    np.testing.assert_allclose(fr.data[0], -10.0, atol=1e-4)  # 10·log10(0.1)
    assert np.isnan(fr.data[1]).all()  # sentinel survives dB conversion


def test_get_embedding_multi_s1_pooled(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _: object())

    def _fake_binned_s1(provider, *, spatial, bins, **kw):
        arr = np.full((len(bins), 2, 64, 64), -15.0, dtype=np.float32)
        arr[1] = np.nan
        return arr, {"frames": [{"start": s, "end": e, "empty": False} for s, e in bins]}

    monkeypatch.setattr(oe, "_fetch_s1_vvvh_binned_raw_tchw", _fake_binned_s1)

    seen = {}

    def _fake_forward(model, sample, **kw):
        seen["t"] = sample.sentinel1.shape[3]
        seen["ts"] = sample.timestamps[0].tolist()
        return _fake_encoder_output(1, 4, 128, modality="s1")

    monkeypatch.setattr(oe, "_encoder_forward", _fake_forward)
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )

    out = emb.get_embedding(
        spatial=PointBuffer(lon=10.0, lat=20.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-15", "2022-09-15"),  # 92d → 4 bins
        sensor=_s1_sensor(),
        output=OutputSpec.pooled(),
        backend="gee",
        model_config={"temporal_mode": "multi"},
    )

    assert out.meta["temporal_mode"] == "multi"
    assert out.meta["n_frames"] == 3  # 4 bins, 1 empty → 3 frames
    assert seen["t"] == 3
    # timestamps = surviving bin starts: 06-15, 08-14, 09-13 (07-15 bin was empty)
    assert seen["ts"] == [[15, 5, 2022], [14, 7, 2022], [13, 8, 2022]]


def test_get_embeddings_batch_from_inputs_multi_variable_frames(monkeypatch):
    emb = oe.OlmoEarthEmbedder()
    monkeypatch.setattr(
        oe,
        "_load_olmoearth",
        lambda variant, device: (object(), {"hf_repo": "allenai/OlmoEarth-v1-Nano"}, "cpu"),
    )
    monkeypatch.setattr(
        oe,
        "_encoder_forward",
        lambda model, sample, **kw: _fake_encoder_output(
            sample.sentinel2_l2a.shape[0], 4, 128, modality="s2"
        ),
    )

    t = TemporalSpec.range("2022-06-15", "2022-08-14")  # 60d → 2 bins
    full = np.random.uniform(0, 3000, (2, 12, 64, 64)).astype(np.float32)
    partial = full.copy()
    partial[1] = np.nan  # second item lost its second bin

    out = emb.get_embeddings_batch_from_inputs(
        spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=256)] * 2,
        input_chws=[full, partial],
        temporal=t,
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert out[0].meta["n_frames"] == 2
    assert out[1].meta["n_frames"] == 1
    for e in out:
        assert e.data.shape == (128,)

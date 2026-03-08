import numpy as np

from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders.onthefly_dofa import DOFAEmbedder
from rs_embed.embedders.onthefly_satmae import SatMAERGBEmbedder
from rs_embed.embedders.onthefly_satmaepp import SatMAEPPEmbedder
from rs_embed.embedders.onthefly_satmaepp_s2 import SatMAEPPSentinel10Embedder


def test_satmaepp_channel_order_defaults_to_bgr_for_fmow_rgb(monkeypatch):
    import rs_embed.embedders.onthefly_satmaepp as satpp

    monkeypatch.delenv("RS_EMBED_SATMAEPP_CHANNEL_ORDER", raising=False)
    monkeypatch.delenv("RS_EMBED_SATMAEPP_BGR", raising=False)
    assert (
        satpp._resolve_satmaepp_channel_order("MVRL/satmaepp_ViT-L_pretrain_fmow_rgb")
        == "bgr"
    )


def test_satmaepp_channel_order_respects_env_override(monkeypatch):
    import rs_embed.embedders.onthefly_satmaepp as satpp

    monkeypatch.setenv("RS_EMBED_SATMAEPP_CHANNEL_ORDER", "rgb")
    assert (
        satpp._resolve_satmaepp_channel_order("MVRL/satmaepp_ViT-L_pretrain_fmow_rgb")
        == "rgb"
    )


def test_satmae_batch_loads_once_and_batches_forward(monkeypatch):
    import rs_embed.embedders.onthefly_satmae as sat

    emb = SatMAERGBEmbedder()
    calls = {"fetch": 0, "load": 0, "forward_batch": 0}

    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setenv("RS_EMBED_SATMAE_FETCH_WORKERS", "1")
    monkeypatch.setenv("RS_EMBED_SATMAE_BATCH_SIZE", "2")

    def _fake_fetch(*, spatial, temporal, sensor, out_size, provider):
        calls["fetch"] += 1
        v = int(spatial.lon) + 10
        return np.full((out_size, out_size, 3), v, dtype=np.uint8)

    def _fake_load(*, model_id, device):
        calls["load"] += 1
        return object(), {"device": "cpu"}

    def _fake_forward_batch(model, rgb_u8_batch, *, image_size, device):
        calls["forward_batch"] += 1
        out = []
        for rgb in rgb_u8_batch:
            val = float(rgb[0, 0, 0])
            # [N,D], with N=4 so pool_from_tokens uses all patch tokens.
            out.append(np.full((4, 2), val, dtype=np.float32))
        return out

    monkeypatch.setattr(sat, "fetch_s2_rgb_u8_from_provider", _fake_fetch)
    monkeypatch.setattr(sat, "_load_satmae", _fake_load)
    monkeypatch.setattr(sat, "_satmae_forward_tokens_batch", _fake_forward_batch)

    spatials = [
        PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=1.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=2.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=3.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=4.0, lat=0.0, buffer_m=256),
    ]

    out = emb.get_embeddings_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
        device="auto",
    )

    assert len(out) == 5
    assert calls["fetch"] == 5
    assert calls["load"] == 1
    assert calls["forward_batch"] == 3  # ceil(5 / batch_size=2)
    assert [float(e.data[0]) for e in out] == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_satmaepp_batch_loads_once_and_batches_forward(monkeypatch):
    import rs_embed.embedders.onthefly_satmaepp as satpp

    emb = SatMAEPPEmbedder()
    calls = {"fetch": 0, "load": 0, "forward_batch": 0}

    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setenv("RS_EMBED_SATMAEPP_FETCH_WORKERS", "1")
    monkeypatch.setenv("RS_EMBED_SATMAEPP_BATCH_SIZE", "2")

    def _fake_fetch(*, spatial, temporal, sensor, out_size, provider):
        calls["fetch"] += 1
        v = int(spatial.lon) + 20
        return np.full((out_size, out_size, 3), v, dtype=np.uint8)

    def _fake_load(*, model_id, device):
        calls["load"] += 1
        return object(), {"device": "cpu"}

    def _fake_forward_batch(model, rgb_u8_batch, *, image_size, device, model_id):
        calls["forward_batch"] += 1
        out = []
        for rgb in rgb_u8_batch:
            val = float(rgb[0, 0, 0])
            out.append(np.full((4, 2), val, dtype=np.float32))
        return out

    monkeypatch.setattr(satpp, "fetch_s2_rgb_u8_from_provider", _fake_fetch)
    monkeypatch.setattr(satpp, "_load_satmaepp", _fake_load)
    monkeypatch.setattr(satpp, "_satmaepp_forward_tokens_batch", _fake_forward_batch)

    spatials = [
        PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=1.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=2.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=3.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=4.0, lat=0.0, buffer_m=256),
    ]

    out = emb.get_embeddings_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
        device="auto",
    )

    assert len(out) == 5
    assert calls["fetch"] == 5
    assert calls["load"] == 1
    assert calls["forward_batch"] == 3  # ceil(5 / batch_size=2)
    assert [float(e.data[0]) for e in out] == [20.0, 21.0, 22.0, 23.0, 24.0]


def test_satmaepp_s2_batch_loads_once_and_batches_forward(monkeypatch):
    import rs_embed.embedders.onthefly_satmaepp_s2 as satpp_s2

    emb = SatMAEPPSentinel10Embedder()
    calls = {"fetch": 0, "load": 0, "forward_batch": 0}

    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setenv("RS_EMBED_SATMAEPP_S2_FETCH_WORKERS", "1")
    monkeypatch.setenv("RS_EMBED_SATMAEPP_S2_BATCH_SIZE", "2")

    def _fake_fetch(
        *, provider, spatial, temporal, scale_m, cloudy_pct, composite, fill_value
    ):
        calls["fetch"] += 1
        v = float(int(spatial.lon) + 30)
        return np.full((10, 8, 8), v, dtype=np.float32)

    def _fake_load(**kwargs):
        calls["load"] += 1
        return object(), {"device": "cpu"}

    def _fake_forward_batch(model, raw_chw_batch, *, image_size, device):
        calls["forward_batch"] += 1
        out = []
        # 1 + G*L with G=3, L=4 (square) to satisfy grouped-token assumptions.
        for raw in raw_chw_batch:
            val = float(raw[0, 0, 0])
            toks = np.full((13, 2), val, dtype=np.float32)
            out.append(toks)
        return out

    monkeypatch.setattr(satpp_s2, "_fetch_s2_sr_10_raw_chw", _fake_fetch)
    monkeypatch.setattr(satpp_s2, "_load_satmaepp_s2", _fake_load)
    monkeypatch.setattr(
        satpp_s2, "_satmaepp_s2_forward_tokens_batch", _fake_forward_batch
    )

    spatials = [
        PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=1.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=2.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=3.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=4.0, lat=0.0, buffer_m=256),
    ]

    out = emb.get_embeddings_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
        device="auto",
    )

    assert len(out) == 5
    assert calls["fetch"] == 5
    assert calls["load"] == 1
    assert calls["forward_batch"] == 3  # ceil(5 / batch_size=2)
    assert [float(e.data[0]) for e in out] == [30.0, 31.0, 32.0, 33.0, 34.0]


def test_dofa_auto_uses_cached_provider_path(monkeypatch):
    import rs_embed.embedders.onthefly_dofa as dofa

    emb = DOFAEmbedder()
    fake_provider = object()
    seen = {"provider_ok": False, "backend": None}

    def _fake_get_provider(_backend):
        seen["backend"] = _backend
        return fake_provider

    monkeypatch.setattr(emb, "_get_provider", _fake_get_provider)

    def _fake_fetch(
        provider,
        spatial,
        temporal,
        *,
        collection,
        bands,
        scale_m,
        cloudy_pct,
        composite,
        default_value,
    ):
        assert provider is fake_provider
        seen["provider_ok"] = True
        x = np.ones((len(bands), 8, 8), dtype=np.float32)
        return x, {"raw_chw_shape": tuple(x.shape)}

    def _fake_resize(x_chw, *, size=224):
        return x_chw.astype(np.float32, copy=False), {
            "orig_hw": x_chw.shape[-2:],
            "target_hw": x_chw.shape[-2:],
        }

    def _fake_load(*, variant, device):
        class _M:
            patch_size = 16

        return _M(), {"device": "cpu", "device_resolved": "cpu"}

    def _fake_forward(model, x_bchw, wavelengths_um, *, device):
        tokens = np.ones((4, 8), dtype=np.float32)
        pooled = np.arange(8, dtype=np.float32)
        return tokens, pooled, {"token_count": 4, "token_dim": 8}

    monkeypatch.setattr(dofa, "_fetch_gee_multiband_sr_chw", _fake_fetch)
    monkeypatch.setattr(dofa, "_resize_chw", _fake_resize)
    monkeypatch.setattr(dofa, "_load_dofa_model", _fake_load)
    monkeypatch.setattr(dofa, "_dofa_forward_tokens_and_pooled", _fake_forward)

    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        device="auto",
    )

    assert seen["provider_ok"] is True
    assert seen["backend"] == "auto"
    assert out.data.shape == (8,)

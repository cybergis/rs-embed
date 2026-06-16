import numpy as np

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders import onthefly_prithvi as pr


def test_normalize_prithvi_variant_accepts_full_model_keys():
    assert pr._normalize_prithvi_variant("prithvi_eo_v2_100_tl") == "prithvi_eo_v2_100_tl"
    assert pr._normalize_prithvi_variant("prithvi_eo_v2_300_tl") == "prithvi_eo_v2_300_tl"
    assert pr._normalize_prithvi_variant("prithvi_eo_v2_600_tl") == "prithvi_eo_v2_600_tl"


def test_prithvi_get_embedding_uses_model_config_variant(monkeypatch):
    emb = pr.PrithviEOV2S2_6B_Embedder()
    spatial = PointBuffer(lon=10.0, lat=20.0, buffer_m=256)
    temporal = TemporalSpec.range("2022-06-01", "2022-09-01")
    seen = {}

    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        pr,
        "_prepare_prithvi_chw",
        lambda x_chw, *, fill_value: (x_chw.astype(np.float32, copy=False), {"prep_mode": "test"}),
    )

    def _fake_load(model_key, *, pretrained, bands, num_frames, coords_encoding, device):
        seen["model_key"] = model_key
        return object(), {"loaded": True}, "cpu"

    monkeypatch.setattr(pr, "_load_prithvi", _fake_load)
    monkeypatch.setattr(
        pr,
        "_prithvi_forward_tokens",
        lambda *args, **kwargs: np.arange(8, dtype=np.float32).reshape(2, 4),
    )

    out = emb.get_embedding(
        spatial=spatial,
        temporal=temporal,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=np.full((6, 8, 8), 1000.0, dtype=np.float32),
        model_config={"variant": "prithvi_eo_v2_300_tl", "temporal_mode": "single"},
    )

    assert seen["model_key"] == "prithvi_eo_v2_300_tl"
    assert out.meta["model_key"] == "prithvi_eo_v2_300_tl"
    assert out.meta["variant"] == "prithvi_eo_v2_300_tl"


def test_prithvi_get_embeddings_batch_forwards_model_config(monkeypatch):
    emb = pr.PrithviEOV2S2_6B_Embedder()
    temporal = TemporalSpec.year(2022)
    seen = {}

    monkeypatch.setenv("RS_EMBED_PRITHVI_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        pr,
        "_fetch_s2_prithvi6_chw",
        lambda provider, spatial, temporal, **kw: np.full((6, 8, 8), 0.25, dtype=np.float32),
    )

    def _fake_batch_from_inputs(**kwargs):
        seen["model_config"] = kwargs.get("model_config")
        return [
            Embedding(data=np.array([sp.lon], dtype=np.float32), meta={})
            for sp in kwargs["spatials"]
        ]

    monkeypatch.setattr(emb, "get_embeddings_batch_from_inputs", _fake_batch_from_inputs)

    out = emb.get_embeddings_batch(
        spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=256)],
        temporal=temporal,
        sensor=None,
        model_config={"variant": "prithvi_eo_v2_600_tl", "temporal_mode": "single"},
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 1
    assert seen["model_config"] == {
        "variant": "prithvi_eo_v2_600_tl",
        "temporal_mode": "single",
    }


def test_prithvi_get_embeddings_batch_from_inputs_uses_model_config_variant(monkeypatch):
    emb = pr.PrithviEOV2S2_6B_Embedder()
    temporal = TemporalSpec.range("2022-06-01", "2022-09-01")
    seen = {}

    def _fake_load(model_key, *, pretrained, bands, num_frames, coords_encoding, device):
        seen["model_key"] = model_key
        return object(), {"loaded": True}, "cpu"

    monkeypatch.setattr(pr, "_load_prithvi", _fake_load)
    monkeypatch.setattr(
        pr,
        "_prepare_prithvi_chw",
        lambda x_chw, *, fill_value: (x_chw.astype(np.float32, copy=False), {"prep_mode": "test"}),
    )
    monkeypatch.setattr(
        pr,
        "_prithvi_forward_tokens_batch",
        lambda *args, **kwargs: [np.arange(8, dtype=np.float32).reshape(2, 4)],
    )

    out = emb.get_embeddings_batch_from_inputs(
        spatials=[PointBuffer(lon=1.0, lat=2.0, buffer_m=256)],
        input_chws=[np.full((6, 8, 8), 1000.0, dtype=np.float32)],
        temporal=temporal,
        sensor=None,
        model_config={"variant": "300_tl", "temporal_mode": "single"},
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert seen["model_key"] == "prithvi_eo_v2_300_tl"
    assert out[0].meta["model_key"] == "prithvi_eo_v2_300_tl"
    assert out[0].meta["variant"] == "prithvi_eo_v2_300_tl"

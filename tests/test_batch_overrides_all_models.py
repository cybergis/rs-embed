import numpy as np

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.embedders.onthefly_anysat import AnySatEmbedder
from rs_embed.embedders.onthefly_prithvi import PrithviEOV2S2_6B_Embedder
from rs_embed.embedders.onthefly_remoteclip import RemoteCLIPS2RGBEmbedder
from rs_embed.embedders.onthefly_scalemae import ScaleMAERGBEmbedder
from rs_embed.embedders.onthefly_galileo import GalileoEmbedder
from rs_embed.embedders.onthefly_wildsat import WildSATEmbedder
from rs_embed.embedders.onthefly_dofa import DOFAEmbedder
from rs_embed.embedders.onthefly_terrafm import TerraFMBEmbedder
from rs_embed.embedders.onthefly_terramind import TerraMindEmbedder
from rs_embed.embedders.onthefly_fomo import FoMoEmbedder
from rs_embed.embedders.onthefly_thor import THORBaseEmbedder
from rs_embed.embedders.onthefly_agrifm import AgriFMEmbedder
from rs_embed.embedders.onthefly_satvision_toa import SatVisionTOAEmbedder

from rs_embed.embedders.precomputed_copernicus_embed import CopernicusEmbedder
from rs_embed.embedders.precomputed_gse_annual import GSEAnnualEmbedder
from rs_embed.embedders.precomputed_tessera import TesseraEmbedder


def _spatials(n: int) -> list[PointBuffer]:
    return [PointBuffer(lon=float(i), lat=0.0, buffer_m=256) for i in range(n)]


def test_remoteclip_batch_prefetch_passes_input_chw(monkeypatch):
    import rs_embed.embedders.onthefly_remoteclip as rc

    emb = RemoteCLIPS2RGBEmbedder()
    monkeypatch.setenv("RS_EMBED_REMOTECLIP_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        rc,
        "_fetch_s2_rgb_chw",
        lambda provider, spatial, temporal, **kw: np.full((3, 8, 8), 0.5, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append(float(arr.max()))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(3),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 3
    assert all(v >= 4999.0 for v in seen)  # 0.5 * 10000


def test_scalemae_batch_prefetch_and_single_model_load(monkeypatch):
    import rs_embed.embedders.onthefly_scalemae as sm

    emb = ScaleMAERGBEmbedder()
    monkeypatch.setenv("RS_EMBED_SCALEMAE_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())

    calls = {"load": 0}

    def _fake_fetch(*, spatial, temporal, sensor, out_size, provider):
        return np.full((out_size, out_size, 3), int(spatial.lon) + 10, dtype=np.uint8)

    def _fake_load(*, model_id, device):
        calls["load"] += 1
        return object(), {"device": "cpu"}

    def _fake_forward(model, rgb_u8, *, image_size, device, input_res_m):
        val = float(rgb_u8[0, 0, 0])
        return np.full((4, 2), val, dtype=np.float32), {"tokens_kind": "tokens_forward"}

    monkeypatch.setattr(sm, "fetch_s2_rgb_u8_from_provider", _fake_fetch)
    monkeypatch.setattr(sm, "_load_scalemae", _fake_load)
    monkeypatch.setattr(sm, "_scalemae_forward_tokens_or_vec", _fake_forward)

    out = emb.get_embeddings_batch(
        spatials=_spatials(4),
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 4
    assert calls["load"] == 1
    assert [float(e.data[0]) for e in out] == [10.0, 11.0, 12.0, 13.0]


def test_prithvi_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_prithvi as pr

    emb = PrithviEOV2S2_6B_Embedder()
    monkeypatch.setenv("RS_EMBED_PRITHVI_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        pr,
        "_fetch_s2_prithvi6_chw",
        lambda provider, spatial, temporal, **kw: np.full((6, 8, 8), 0.25, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        seen.append(float(kw["input_chw"].max()))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(3),
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 3
    assert all(v >= 2499.0 for v in seen)  # 0.25 * 10000


def test_terrafm_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_terrafm as tf

    emb = TerraFMBEmbedder()
    monkeypatch.setenv("RS_EMBED_TERRAFM_FETCH_WORKERS", "1")
    seen_backend = {"value": None}

    def _fake_get_provider(_backend):
        seen_backend["value"] = _backend
        return object()

    monkeypatch.setattr(emb, "_get_provider", _fake_get_provider)
    monkeypatch.setattr(
        tf,
        "_fetch_s2_sr_12_chw",
        lambda provider, spatial, temporal, **kw: np.full((12, 8, 8), 0.1, dtype=np.float32),
    )

    seen = []

    def _fake_batch_from_inputs(**kw):
        seen.extend((arr.shape[0], float(arr.max())) for arr in kw["input_chws"])
        return [
            Embedding(data=np.array([sp.lon], dtype=np.float32), meta={}) for sp in kw["spatials"]
        ]

    monkeypatch.setattr(emb, "get_embeddings_batch_from_inputs", _fake_batch_from_inputs)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="auto",
    )

    assert len(out) == 2
    assert seen_backend["value"] == "auto"
    assert seen[0][0] == 12
    assert seen[0][1] >= 999.0  # 0.1 * 10000


def test_terramind_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_terramind as tm

    emb = TerraMindEmbedder()
    monkeypatch.setenv("RS_EMBED_TERRAMIND_FETCH_WORKERS", "1")
    seen_backend = {"value": None}

    def _fake_get_provider(_backend):
        seen_backend["value"] = _backend
        return object()

    monkeypatch.setattr(emb, "_get_provider", _fake_get_provider)
    monkeypatch.setattr(
        tm,
        "_fetch_s2_sr_12_raw_chw",
        lambda provider, spatial, temporal, **kw: np.full((12, 8, 8), 1234.0, dtype=np.float32),
    )

    seen = []

    def _fake_batch_from_inputs(**kw):
        seen.extend((arr.shape[0], float(arr.max())) for arr in kw["input_chws"])
        return [
            Embedding(data=np.array([sp.lon], dtype=np.float32), meta={}) for sp in kw["spatials"]
        ]

    monkeypatch.setattr(emb, "get_embeddings_batch_from_inputs", _fake_batch_from_inputs)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="auto",
    )

    assert len(out) == 2
    assert seen_backend["value"] == "auto"
    assert seen[0][0] == 12
    assert seen[0][1] >= 1234.0


def test_dofa_tensor_get_embedding_uses_input_chw(monkeypatch):
    import rs_embed.embedders.onthefly_dofa as dofa

    emb = DOFAEmbedder()
    seen = {}

    def _fake_resize(x_chw, *, size=224):
        return x_chw.astype(np.float32, copy=False), {
            "orig_hw": x_chw.shape[-2:],
            "target_hw": x_chw.shape[-2:],
        }

    class _M:
        patch_size = 16
        global_pool = True

    def _fake_load(*, variant, device):
        return _M(), {"device": "cpu"}

    def _fake_forward(model, x_bchw, wavelengths_um, *, device):
        seen["shape"] = tuple(x_bchw.shape)
        seen["wavelengths"] = list(wavelengths_um)
        return (
            np.ones((4, 2), dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
            {"token_count": 4, "token_dim": 2},
        )

    monkeypatch.setattr(dofa, "_resize_chw", _fake_resize)
    monkeypatch.setattr(dofa, "_load_dofa_model", _fake_load)
    monkeypatch.setattr(dofa, "_dofa_forward_tokens_and_pooled", _fake_forward)

    sensor = SensorSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]),
    )
    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=None,
        sensor=sensor,
        output=OutputSpec.pooled(),
        backend="tensor",
        input_chw=np.ones((12, 8, 8), dtype=np.float32),
    )

    assert seen["shape"] == (1, 12, 8, 8)
    assert len(seen["wavelengths"]) == 12
    assert out.data.shape == (2,)


def test_terrafm_tensor_get_embedding_uses_input_chw(monkeypatch):
    import rs_embed.embedders.onthefly_terrafm as tf

    emb = TerraFMBEmbedder()
    seen = {}

    monkeypatch.setattr(
        tf,
        "_resize_chw_to_224",
        lambda x_chw, *, size=224: x_chw.astype(np.float32, copy=False),
    )
    monkeypatch.setattr(
        tf,
        "_load_terrafm_b",
        lambda *, auto_download, cache_dir: (object(), {"device": "cpu"}),
    )

    def _fake_forward(model, x_bchw, *, device, want_grid):
        seen["shape"] = tuple(x_bchw.shape)
        return np.array([1.0, 2.0], dtype=np.float32), None

    monkeypatch.setattr(tf, "_terrafm_pooled_and_grid", _fake_forward)

    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=None,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="tensor",
        input_chw=np.ones((12, 8, 8), dtype=np.float32),
    )

    assert seen["shape"] == (1, 12, 8, 8)
    assert out.data.shape == (2,)


def test_terramind_tensor_get_embedding_uses_input_chw(monkeypatch):
    import rs_embed.embedders.onthefly_terramind as tm

    emb = TerraMindEmbedder()
    seen = {}

    monkeypatch.setattr(
        tm,
        "_resize_chw",
        lambda x_chw, *, size=224: x_chw.astype(np.float32, copy=False),
    )
    monkeypatch.setattr(
        tm,
        "_load_terramind",
        lambda **kwargs: (object(), {"device": "cpu"}, "cpu"),
    )

    def _fake_forward(model, x_bchw, *, modality, layer_index, device):
        seen["shape"] = tuple(x_bchw.shape)
        return np.ones((4, 2), dtype=np.float32), {"tokens_shape": (4, 2)}

    monkeypatch.setattr(tm, "_terramind_forward_tokens", _fake_forward)

    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=None,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="tensor",
        input_chw=np.ones((12, 8, 8), dtype=np.float32),
    )

    assert seen["shape"] == (1, 12, 8, 8)
    assert out.data.shape == (2,)


def test_fomo_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_fomo as fomo

    emb = FoMoEmbedder()
    monkeypatch.setenv("RS_EMBED_FOMO_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        fomo,
        "_fetch_s2_sr_12_raw_chw",
        lambda provider, spatial, temporal, **kw: np.full((12, 8, 8), 3456.0, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 12
    assert seen[0][1] >= 3456.0


def test_thor_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_thor as thor

    emb = THORBaseEmbedder()
    monkeypatch.setenv("RS_EMBED_THOR_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        thor,
        "_fetch_s2_sr_10_raw_chw",
        lambda provider, spatial, temporal, **kw: np.full((10, 8, 8), 5678.0, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 10
    assert seen[0][1] >= 5678.0


def test_anysat_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_anysat as anysat

    emb = AnySatEmbedder()
    monkeypatch.setenv("RS_EMBED_ANYSAT_FETCH_WORKERS", "1")
    monkeypatch.setenv("RS_EMBED_ANYSAT_FRAMES", "3")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        anysat,
        "_fetch_s2_10_raw_tchw",
        lambda provider, spatial, temporal, **kw: np.full((3, 10, 8, 8), 4321.0, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], arr.shape[1], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 3
    assert seen[0][1] == 10
    assert seen[0][2] >= 4321.0


def test_agrifm_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_agrifm as agri

    emb = AgriFMEmbedder()
    monkeypatch.setenv("RS_EMBED_AGRIFM_FETCH_WORKERS", "1")
    monkeypatch.setenv("RS_EMBED_AGRIFM_FRAMES", "4")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        agri,
        "_fetch_s2_10_raw_tchw",
        lambda provider, spatial, temporal, **kw: np.full((4, 10, 8, 8), 2222.0, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], arr.shape[1], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 4
    assert seen[0][1] == 10
    assert seen[0][2] >= 2222.0


def test_wildsat_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_wildsat as ws

    emb = WildSATEmbedder()
    monkeypatch.setenv("RS_EMBED_WILDSAT_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        ws,
        "_fetch_s2_rgb_chw",
        lambda provider, spatial, temporal, **kw: np.full((3, 8, 8), 0.4, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 3
    assert seen[0][1] >= 3999.0  # 0.4 * 10000


def test_galileo_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_galileo as gal

    emb = GalileoEmbedder()
    monkeypatch.setenv("RS_EMBED_GALILEO_FETCH_WORKERS", "1")
    monkeypatch.setenv("RS_EMBED_GALILEO_FRAMES", "5")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        gal,
        "_fetch_s2_10_raw_tchw",
        lambda provider, spatial, temporal, **kw: np.full((5, 10, 8, 8), 2222.0, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], arr.shape[1], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 5
    assert seen[0][1] == 10
    assert seen[0][2] >= 2222.0


def test_satvision_toa_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_satvision_toa as sv

    emb = SatVisionTOAEmbedder()
    monkeypatch.setenv("RS_EMBED_SATVISION_TOA_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        emb,
        "_resolve_runtime",
        lambda **kw: {
            "model": object(),
            "model_meta": {},
            "device": "cpu",
            "model_id": "MVRL/SatVision-TOA",
            "image_size": 8,
            "in_chans": 14,
            "norm_mode": "raw",
            "reflectance_indices": (0, 1, 2, 3, 4, 6),
            "emissive_indices": (5, 7, 8, 9, 10, 11, 12, 13),
            "reflectance_divisor": 10000.0,
            "emissive_mins": (175.0,) * 8,
            "emissive_maxs": (375.0,) * 8,
        },
    )
    monkeypatch.setattr(
        emb,
        "_prepare_input",
        lambda raw_chw, **kw: np.asarray(raw_chw, dtype=np.float32),
    )
    monkeypatch.setattr(
        sv,
        "_fetch_toa_raw_chw_from_gee",
        lambda provider, spatial, temporal, sensor: np.full(
            (14, 8, 8), 2000.0 + spatial.lon, dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        sv,
        "_satvision_forward_batch",
        lambda model, x_chw_batch, **kw: (
            [np.full((4,), float(x[0, 0, 0]), dtype=np.float32) for x in x_chw_batch],
            {"tokens_kind": "pooled"},
        ),
    )

    sensor = SensorSpec(
        collection="TEST/COLL",
        bands=tuple(f"B{i}" for i in range(14)),
        scale_m=500,
    )

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        sensor=sensor,
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert out[0].data.shape == (4,)
    assert float(out[0].data[0]) == 2000.0
    assert float(out[1].data[0]) == 2001.0


def test_precomputed_batch_overrides_call_single_embedding(monkeypatch):
    # gse_annual
    gse = GSEAnnualEmbedder()
    monkeypatch.setenv("RS_EMBED_GSE_BATCH_WORKERS", "1")
    monkeypatch.setattr(
        gse,
        "get_embedding",
        lambda **kw: Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={}),
    )
    out_gse = gse.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
    )
    assert len(out_gse) == 2

    # copernicus_embed
    cop = CopernicusEmbedder()
    monkeypatch.setenv("RS_EMBED_COPERNICUS_BATCH_WORKERS", "1")
    monkeypatch.setattr(
        cop,
        "get_embedding",
        lambda **kw: Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={}),
    )
    out_cop = cop.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.year(2021),
        output=OutputSpec.pooled(),
        backend="local",
    )
    assert len(out_cop) == 2

    # tessera
    tes = TesseraEmbedder()
    monkeypatch.setenv("RS_EMBED_TESSERA_BATCH_WORKERS", "1")
    monkeypatch.setattr(
        tes,
        "get_embedding",
        lambda **kw: Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={}),
    )
    out_tes = tes.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.year(2021),
        output=OutputSpec.pooled(),
        backend="local",
    )
    assert len(out_tes) == 2

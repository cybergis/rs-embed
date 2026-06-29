"""satmaepp folds the former standalone `satmaepp_s2_10b` model in as a modality.

`satmaepp` exposes two sensor configurations under one model name, selected via
`modality=`: the default fMoW-RGB 3-band path and the grouped-channel Sentinel-2
10-band path. These tests lock the describe() declaration, per-modality default
sensors, and that `get_embeddings_batch_from_inputs` routes to the right forward
path based on `sensor.modality`.
"""

import numpy as np

from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders.onthefly_satmaepp import SatMAEPPEmbedder
from rs_embed.embedders.onthefly_satmaepp_s2 import _S2_SR_10_BANDS
from rs_embed.tools.model_defaults import default_sensor_for_model

_SPATIAL = PointBuffer(lon=0.0, lat=0.0, buffer_m=256)
_TEMPORAL = TemporalSpec.year(2020)


def test_describe_exposes_rgb_and_s2_10b_modalities():
    desc = SatMAEPPEmbedder().describe()
    mods = desc["modalities"]
    assert set(mods) == {"rgb", "s2_10b"}
    assert desc["defaults"]["modality"] == "rgb"
    assert tuple(mods["rgb"]["bands"]) == ("B4", "B3", "B2")
    assert tuple(mods["s2_10b"]["bands"]) == tuple(_S2_SR_10_BANDS)


def test_default_sensor_resolves_per_modality():
    rgb = default_sensor_for_model("satmaepp")
    assert tuple(rgb.bands) == ("B4", "B3", "B2")

    s2 = default_sensor_for_model("satmaepp", modality="s2_10b")
    assert tuple(s2.bands) == tuple(_S2_SR_10_BANDS)
    assert str(getattr(s2, "modality", "")) == "s2_10b"


def test_s2_delegate_reports_satmaepp_model_name():
    # The unregistered Sentinel-2 class backs the s2_10b modality but labels its
    # metadata under the single public "satmaepp" model name.
    assert SatMAEPPEmbedder()._s2_delegate().model_name == "satmaepp"


def test_rgb_modality_routes_to_rgb_forward(monkeypatch):
    import rs_embed.embedders.onthefly_satmaepp as satpp

    emb = SatMAEPPEmbedder()
    monkeypatch.setattr(satpp, "_load_satmaepp", lambda **k: (object(), {"device": "cpu"}))

    def _fake_rgb_forward(model, rgb_u8_batch, *, image_size, device, model_id):
        # 1 CLS + 4 patch tokens -> exercises the plain patch-token pool path.
        return [np.full((5, 2), 1.0, dtype=np.float32) for _ in rgb_u8_batch]

    monkeypatch.setattr(satpp, "_satmaepp_forward_tokens_batch", _fake_rgb_forward)

    out = emb.get_embeddings_batch_from_inputs(
        spatials=[_SPATIAL],
        input_chws=[np.zeros((3, 2, 2), dtype=np.float32)],
        temporal=_TEMPORAL,
        sensor=default_sensor_for_model("satmaepp"),
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
    )
    assert out[0].meta["tokens_kind"] == "tokens_forward_encoder"


def test_s2_10b_modality_routes_to_grouped_channel_forward(monkeypatch):
    import rs_embed.embedders.onthefly_satmaepp_s2 as satpp_s2

    emb = SatMAEPPEmbedder()
    monkeypatch.setattr(satpp_s2, "_load_satmaepp_s2", lambda **k: (object(), {"device": "cpu"}))

    def _fake_s2_forward(model, raw_chw_batch, *, image_size, device):
        # 1 CLS + 12 group-channel tokens (3 groups x 4 spatial tokens).
        return [np.full((13, 2), 1.0, dtype=np.float32) for _ in raw_chw_batch]

    monkeypatch.setattr(satpp_s2, "_satmaepp_s2_forward_tokens_batch", _fake_s2_forward)

    out = emb.get_embeddings_batch_from_inputs(
        spatials=[_SPATIAL],
        input_chws=[np.zeros((10, 2, 2), dtype=np.float32)],
        temporal=_TEMPORAL,
        sensor=default_sensor_for_model("satmaepp", modality="s2_10b"),
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
    )
    assert out[0].meta["tokens_kind"] == "tokens_forward_encoder_group_channel"

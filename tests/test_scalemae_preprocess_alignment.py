import numpy as np
import pytest

from rs_embed.core.errors import ModelError
from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders.onthefly_scalemae import (
    ScaleMAERGBEmbedder,
    _resolve_scalemae_forward_features,
    _scalemae_effective_input_res_m,
    _scalemae_forward_tokens_or_vec,
)


def test_scalemae_effective_input_res_tracks_resize():
    # Single preprocessing path: the square input is resized to image_size (no
    # center crop), so the effective GSD scales by short_side / image_size.
    rgb_u8 = np.zeros((11, 13, 3), dtype=np.uint8)

    assert _scalemae_effective_input_res_m(
        rgb_u8,
        image_size=224,
        source_res_m=10.0,
    ) == pytest.approx(10.0 * (11.0 / 224.0))


def test_scalemae_input_override_skips_pre_resize_and_adjusts_input_res(monkeypatch):
    import rs_embed.embedders.onthefly_scalemae as sm

    emb = ScaleMAERGBEmbedder()
    seen = {}

    def _fake_load(*, model_id, device):
        return object(), {"device": "cpu"}

    def _fake_forward(model, rgb_u8, *, image_size, device, input_res_m):
        seen["shape"] = tuple(rgb_u8.shape)
        seen["input_res_m"] = float(input_res_m)
        return np.full((4, 2), 1.0, dtype=np.float32), {"tokens_kind": "tokens_forward"}

    monkeypatch.setattr(sm, "_load_scalemae", _fake_load)
    monkeypatch.setattr(sm, "_scalemae_forward_tokens_or_vec", _fake_forward)

    input_chw = np.full((3, 11, 13), 5000.0, dtype=np.float32)
    emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2020),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
        input_chw=input_chw,
    )

    assert seen["shape"] == (11, 13, 3)
    assert seen["input_res_m"] == pytest.approx(10.0 * (11.0 / 224.0))


def test_scalemae_resolves_nested_forward_features():
    class _Backbone:
        patch_size = 16

        def forward_features(self, x, input_res=None):
            return x

    class _Wrapper:
        def __init__(self):
            self.model = _Backbone()

    core, ff, owner = _resolve_scalemae_forward_features(_Wrapper())
    assert owner == "model"
    assert core.patch_size == 16
    assert getattr(ff, "__self__", None) is core
    assert getattr(ff, "__func__", None) is _Backbone.forward_features


def test_scalemae_requires_forward_features_path(monkeypatch):
    class _Model:
        patch_size = 16

        def __call__(self, x, patch_size, input_res):
            raise AssertionError("forward() should never be used for ScaleMAE extraction")

    with pytest.raises(ModelError, match="does not expose forward_features"):
        _scalemae_forward_tokens_or_vec(
            _Model(),
            np.zeros((16, 16, 3), dtype=np.uint8),
            image_size=224,
            device="cpu",
            input_res_m=10.0,
        )

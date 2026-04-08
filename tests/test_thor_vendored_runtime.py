import types

import numpy as np
import torch


class _FakeTHORModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.groups = {
            "group0": ["S2:Blue", "S2:Green", "S2:Red", "S2:NIR"],
            "group1": ["S2:RE1", "S2:RE2", "S2:RE3", "S2:RE4", "S2:SWIR1", "S2:SWIR2"],
        }
        self.out_channels = [8]
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x):
        batch = x.shape[0]
        channel_params = {
            "S2:Blue": {"num_patch": 18},
            "S2:Green": {"num_patch": 18},
            "S2:Red": {"num_patch": 18},
            "S2:NIR": {"num_patch": 18},
            "S2:RE1": {"num_patch": 9},
            "S2:RE2": {"num_patch": 9},
            "S2:RE3": {"num_patch": 9},
            "S2:RE4": {"num_patch": 9},
            "S2:SWIR1": {"num_patch": 9},
            "S2:SWIR2": {"num_patch": 9},
        }
        features = [torch.arange(batch * 405 * 8, dtype=torch.float32).reshape(batch, 405, 8)]
        return features, channel_params


def test_thor_loads_vendored_runtime_without_external_thor_dependency(monkeypatch):
    import rs_embed.embedders.onthefly_thor as thor

    fake_module = types.SimpleNamespace(
        load_thor_model=lambda **kwargs: _FakeTHORModel(),
    )
    thor._load_thor_module.cache_clear()
    thor._load_thor_cached.cache_clear()
    monkeypatch.setattr(thor, "_load_thor_module", lambda: fake_module)

    model, meta = thor._load_thor_cached(
        model_key="thor_v1_base",
        model_bands=tuple(thor._THOR_MODEL_BANDS),
        pretrained=False,
        ckpt_path=None,
        ground_cover=2880,
        patch_size=16,
        dev="cpu",
    )

    assert meta["model_key"] == "thor_v1_base"
    assert meta["pretrained"] is False

    tokens, grid, fmeta = thor._thor_forward_single(
        model,
        np.zeros((10, 288, 288), dtype=np.float32),
        device="cpu",
        group_merge="mean",
    )

    assert tokens.shape == (405, 8)
    assert grid is not None
    assert grid.shape == (8, 18, 18)
    assert fmeta["expected_patch_tokens"] == 405


def test_thor_runtime_config_defaults_patch_size_to_8(monkeypatch):
    import rs_embed.embedders.onthefly_thor as thor

    monkeypatch.delenv("RS_EMBED_THOR_PATCH_SIZE", raising=False)
    monkeypatch.delenv("RS_EMBED_THOR_IMG", raising=False)
    monkeypatch.delenv("RS_EMBED_THOR_MODEL_KEY", raising=False)

    cfg = thor._resolve_thor_runtime_config(
        model_config=None,
        default_model_key=thor.THORBaseEmbedder.DEFAULT_MODEL_KEY,
        default_image_size=thor.THORBaseEmbedder.DEFAULT_IMAGE_SIZE,
    )

    assert cfg["patch_size"] == 8
    assert cfg["image_size"] == 288


def test_enable_alibi_for_timm_patches_block_and_attention():
    from timm.models.vision_transformer import Attention, Block

    from rs_embed.embedders._vendor.thor.models.patch_timm import enable_alibi_for_timm

    enable_alibi_for_timm._done = False
    enable_alibi_for_timm()

    attn = Attention(dim=8, num_heads=2, qkv_bias=True)
    x = torch.randn(1, 4, 8)
    alibi = torch.zeros(1, 2, 4, 4)
    y = attn(x, alibi)
    assert y.shape == x.shape

    blk = Block(dim=8, num_heads=2, qkv_bias=True)
    z = blk(x, alibi)
    assert z.shape == x.shape

"""An ROI sub-window is pooled like the model's own pooled vector (issue #140).

Pooled output for an ROI sub-window must be comparable with the same model's
whole-input pooled vector: the same space and shape (through the same norm,
head or projection) and the same reduction (a CLS token or fixed mean ignores
``pooling``; a token pool honors it). Where no token grid exists the vector
covers the whole fetched square, and that is flagged, never silent.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from rs_embed.core.specs import OutputSpec

torch = pytest.importorskip("torch")

_FULL = (0.0, 1.0, 0.0, 1.0)
_TOP = (0.0, 0.5, 0.0, 1.0)  # rows 0..1 of a 4-row grid


def _row_tokens(side: int = 4, dim: int = 3) -> np.ndarray:
    """[N, D] tokens whose value is their grid row plus the channel index."""
    rows = np.repeat(np.arange(side, dtype=np.float32), side)
    return rows[:, None] + np.arange(dim, dtype=np.float32)[None, :]


def _top_mean(tokens: np.ndarray, side: int = 4) -> np.ndarray:
    return tokens.reshape(side, side, -1)[:2].reshape(-1, tokens.shape[-1]).mean(axis=0)


# --------------------------------------------------------------------------- dofa


def test_dofa_roi_mean_goes_through_fc_norm_under_global_pooling():
    from rs_embed.embedders.onthefly_dofa import _build_dofa_embedding

    torch.manual_seed(0)
    fc_norm = torch.nn.LayerNorm(3)
    with torch.no_grad():
        fc_norm.weight.uniform_(0.5, 2.0)
        fc_norm.bias.uniform_(-1.0, 1.0)
    model = types.SimpleNamespace(global_pool=True, patch_size=16, fc_norm=fc_norm)
    tokens = _row_tokens()
    native = np.array([7.0, 8.0, 9.0], dtype=np.float32)

    def build(geo_roi, pooling="mean"):
        return _build_dofa_embedding(
            tokens,
            native,
            model=model,
            geo_roi=geo_roi,
            shape_prep=None,
            output=OutputSpec.pooled(pooling),
            base_meta={},
        )

    np.testing.assert_array_equal(build(_FULL).data, native)
    expected = fc_norm(torch.from_numpy(_top_mean(tokens))[None])[0].detach().numpy()
    roi = build(_TOP)
    np.testing.assert_allclose(roi.data, expected, rtol=1e-6)
    assert roi.meta["pooling"] == "roi_grid_mean" and roi.meta["roi_pooled"] is True
    # DOFA's pooled vector is a fixed mean: max is recorded, not applied.
    roi_max = build(_TOP, "max")
    np.testing.assert_allclose(roi_max.data, expected, rtol=1e-6)
    assert roi_max.meta["requested_pooling"] == "max"


# --------------------------------------------------------------------------- clay / terrafm


def test_clay_roi_follows_its_cls_vector_and_records_max():
    from rs_embed.embedders.onthefly_clay import build_clay_embedding

    tokens = _row_tokens()
    out = build_clay_embedding(
        tokens,
        np.zeros(3, dtype=np.float32),
        geo_roi=_TOP,
        output=OutputSpec.pooled("max"),
        base_meta={},
    )
    np.testing.assert_allclose(out.data, _top_mean(tokens))
    assert out.meta["pooling"] == "roi_grid_mean"
    assert out.meta["requested_pooling"] == "max"


def test_terrafm_roi_mean_pools_like_its_cls_vector():
    from rs_embed.embedders.onthefly_terrafm import _build_terrafm_embedding

    tokens = _row_tokens()
    grid = tokens.reshape(4, 4, 3).transpose(2, 0, 1)
    native = np.array([7.0, 8.0, 9.0], dtype=np.float32)

    def build(geo_roi):
        return _build_terrafm_embedding(
            native, grid, geo_roi=geo_roi, output=OutputSpec.pooled("max"), base_meta={}
        )

    whole = build(_FULL)
    np.testing.assert_array_equal(whole.data, native)
    assert whole.meta["pooling"] == "model_cls"
    roi = build(_TOP)
    # It used to max-pool the ROI, a different statistic from the CLS vector.
    np.testing.assert_allclose(roi.data, _top_mean(tokens))
    assert roi.meta["pooling"] == "roi_grid_mean"
    assert roi.meta["requested_pooling"] == "max"


# --------------------------------------------------------------------------- remoteclip


def _fake_clip(width: int = 4, out_dim: int = 2):
    torch.manual_seed(0)
    visual = torch.nn.Module()
    visual.ln_post = torch.nn.LayerNorm(width)
    visual.proj = torch.nn.Parameter(torch.randn(width, out_dim))
    return types.SimpleNamespace(model=types.SimpleNamespace(visual=visual))


def _roi_pooled_remoteclip(monkeypatch, model, tokens, tokens_kind, geo_roi=_TOP):
    import rs_embed.embedders.onthefly_remoteclip as rc

    monkeypatch.setattr(
        rc, "_remoteclip_encode_tokens", lambda *a, **k: (tokens, {"tokens_kind": tokens_kind})
    )
    native = np.array([5.0, 6.0], dtype=np.float32)
    return rc._remoteclip_roi_pooled(
        model,
        np.zeros((8, 8, 3), np.uint8),
        native,
        geo_roi=geo_roi,
        shape_prep=None,
        output=OutputSpec.pooled("max"),
        image_size=224,
        device="cpu",
    )


def test_remoteclip_roi_pools_projected_tokens_in_the_clip_embedding_space(monkeypatch):
    model = _fake_clip()
    cls = np.full((1, 4), 9.0, dtype=np.float32)
    tokens = np.concatenate([cls, _row_tokens(dim=4)])  # CLS + 4x4 patch tokens, width 4
    vec, meta = _roi_pooled_remoteclip(monkeypatch, model, tokens, "tokens_intermediates")

    visual = model.model.visual
    with torch.no_grad():
        projected = (visual.ln_post(torch.from_numpy(tokens[1:])) @ visual.proj).numpy()
    # The intermediate tokens are 4-wide; the ROI vector is in the 2-wide embedding space.
    assert vec.shape == (2,)
    np.testing.assert_allclose(vec, _top_mean(projected), rtol=1e-5)
    assert meta["pooling"] == "roi_grid_mean" and meta["requested_pooling"] == "max"


@pytest.mark.parametrize(
    ("model", "tokens_kind"),
    [
        (types.SimpleNamespace(), "tokens_intermediates"),  # no ln_post / proj
        (_fake_clip(), "tokens"),  # forward_encoder tokens: unknown space
    ],
)
def test_remoteclip_warns_when_its_tokens_cannot_be_projected(monkeypatch, model, tokens_kind):
    with pytest.warns(UserWarning, match="no token grid"):
        vec, meta = _roi_pooled_remoteclip(monkeypatch, model, _row_tokens(dim=4), tokens_kind)
    np.testing.assert_array_equal(vec, [5.0, 6.0])
    assert meta["roi_pooled"] is False
    assert "tokens_kind" not in meta  # the unused token pass does not describe the vector


# --------------------------------------------------------------------------- wildsat


def _wildsat(fmeta, native_dim=2, geo_roi=_TOP):
    """A torchvision-ViT-like backbone: 3-wide encoder tokens, ``heads`` to 5,
    then an image head to 2 (real vitb16: 768 -> 1000 -> 256)."""
    from rs_embed.embedders.onthefly_wildsat import _build_wildsat_embedding

    torch.manual_seed(0)
    backbone = types.SimpleNamespace(heads=torch.nn.Linear(3, 5))
    image_head = torch.nn.Linear(5, 2)
    tokens = _row_tokens()
    out = _build_wildsat_embedding(
        np.zeros(native_dim, dtype=np.float32),
        tokens.reshape(4, 4, 3).transpose(2, 0, 1),
        fmeta,
        backbone=backbone,
        image_head=image_head,
        geo_roi=geo_roi,
        output=OutputSpec.pooled(),
        meta={},
    )
    roi_mean = torch.from_numpy(_top_mean(tokens))[None]
    return out, backbone.heads, image_head, roi_mean


def test_wildsat_roi_goes_through_the_backbone_heads_and_the_image_head():
    out, heads, image_head, roi_mean = _wildsat(
        {"feature_source": "image_head", "grid_kind": "vit_patch_tokens"}
    )
    expected = image_head(heads(roi_mean))[0].detach().numpy()
    assert out.data.shape == (2,)  # the image head's space, not the 3-wide encoder's
    np.testing.assert_allclose(out.data, expected, rtol=1e-6)


def test_wildsat_backbone_roi_goes_through_the_backbone_heads():
    out, heads, _, roi_mean = _wildsat(
        {"feature_source": "backbone", "grid_kind": "vit_patch_tokens"}, native_dim=5
    )
    np.testing.assert_allclose(out.data, heads(roi_mean)[0].detach().numpy(), rtol=1e-6)


def test_wildsat_token_pool_honors_max_on_the_roi():
    from rs_embed.embedders.onthefly_wildsat import _build_wildsat_embedding

    tokens = _row_tokens()
    fmeta = {"feature_source": "image_head", "grid_kind": "vit_patch_tokens"}
    # pooled_from_tokens: the vector is a backbone-token pool, so no head applies.
    out = _build_wildsat_embedding(
        tokens.max(axis=0),
        tokens.reshape(4, 4, 3).transpose(2, 0, 1),
        {**fmeta, "pooled_from_tokens": True},
        backbone=types.SimpleNamespace(heads=torch.nn.Linear(3, 5)),
        image_head=torch.nn.Linear(5, 2),
        geo_roi=_TOP,
        output=OutputSpec.pooled("max"),
        meta={},
    )
    np.testing.assert_allclose(out.data, tokens.reshape(4, 4, 3)[:2].max(axis=(0, 1)))
    assert out.meta["pooling"] == "roi_grid_max"


def test_wildsat_vector_as_1x1_is_not_an_roi_grid():
    with pytest.warns(UserWarning, match="no token grid"):
        out, _, _, _ = _wildsat({"feature_source": "image_head", "grid_kind": "vector_as_1x1"})
    np.testing.assert_array_equal(out.data, [0.0, 0.0])
    assert out.meta["pooling"] == "identity" and out.meta["roi_pooled"] is False


def test_module_vector_head_moves_the_vector_to_the_first_module_with_parameters():
    from rs_embed.embedders.shared import module_vector_head

    linear = torch.nn.Linear(3, 2).double()
    head = module_vector_head(lambda: torch.nn.Identity(), lambda: linear)
    expected = linear(torch.ones(1, 3, dtype=torch.float64))[0].detach().numpy()
    np.testing.assert_allclose(head(np.ones(3, np.float32)), expected)


# --------------------------------------------------------------------------- satvision


def test_satvision_roi_mean_of_normed_tokens_restricts_forward_features():
    from rs_embed.embedders.onthefly_satvision_toa import build_satvision_embedding

    tokens = _row_tokens()
    native = tokens.mean(axis=0)  # forward_features: the normed tokens' mean

    def build(geo_roi):
        return build_satvision_embedding(
            native, geo_roi=geo_roi, output=OutputSpec.pooled(), meta={}, roi_tokens=tokens
        )

    np.testing.assert_array_equal(build(_FULL).data, native)
    roi = build(_TOP)
    np.testing.assert_allclose(roi.data, _top_mean(tokens))
    assert roi.meta["pooling"] == "roi_grid_mean"


def test_satvision_pooled_forward_keeps_the_roi_tokens_from_one_forward(monkeypatch):
    import rs_embed.embedders.onthefly_satvision_toa as sv

    batches = []

    def forward_tokens(x):
        batches.append(int(x.shape[0]))
        return x.mean(dim=(2, 3))[:, None, :].repeat(1, 4, 1)  # [B, 4, C]

    model = types.SimpleNamespace(
        _forward_tokens=forward_tokens,
        norm=lambda t: t * 2.0,
        avgpool=torch.nn.AdaptiveAvgPool1d(1),
    )
    xs = [np.full((3, 8, 8), float(v), np.float32) for v in (1, 2, 3)]
    arrs, meta, toks = sv._satvision_forward(
        model, xs, [_TOP, _FULL, _TOP], output=OutputSpec.pooled(), device="cpu"
    )
    assert batches == [3]  # forward_features and the ROI tokens share one pass
    np.testing.assert_array_equal(np.stack(arrs), np.repeat([[2.0], [4.0], [6.0]], 3, axis=1))
    assert toks[1] is None  # the whole-ROI input needs no tokens
    np.testing.assert_array_equal(toks[2], np.full((4, 3), 6.0, np.float32))
    assert meta["forward_path"] == "forward_features"

    # Without a sub-window, or without the forward_features steps, it is the plain forward.
    calls = []
    monkeypatch.setattr(
        sv, "_satvision_forward_batch", lambda *a, **k: calls.append(1) or (["v"] * 3, {})
    )
    assert sv._satvision_forward(
        model, xs, [_FULL] * 3, output=OutputSpec.pooled(), device="cpu"
    ) == (["v"] * 3, {}, [None] * 3)
    sv._satvision_forward(
        types.SimpleNamespace(), xs, [_TOP] * 3, output=OutputSpec.pooled(), device="cpu"
    )
    assert calls == [1, 1] and batches == [3]


def test_satvision_warns_without_tokens():
    from rs_embed.embedders.onthefly_satvision_toa import build_satvision_embedding

    with pytest.warns(UserWarning, match="no token grid"):
        out = build_satvision_embedding(
            np.ones(3, np.float32),
            geo_roi=_TOP,
            output=OutputSpec.pooled(),
            meta={},
            roi_tokens=None,
        )
    assert out.meta["roi_pooled"] is False


# --------------------------------------------------------------------------- fomo / satmaepp_s2


def _spike_tokens(groups: int, side: int = 4) -> np.ndarray:
    """Group-major tokens [G*L, 1]: group 0 is 10 at cell (0, 0), all else 0."""
    tokens = np.zeros((groups, side * side, 1), dtype=np.float32)
    tokens[0, 0, 0] = 10.0
    return tokens.reshape(groups * side * side, 1)


def test_fomo_roi_max_takes_the_max_over_the_rois_modality_tokens():
    from rs_embed.embedders.onthefly_fomo import _fomo_pool_and_crop

    tokens = _spike_tokens(groups=2)

    def pool(geo_roi):
        return _fomo_pool_and_crop(
            tokens,
            n_modalities=2,
            patch_size=4,
            image_size=16,
            geo_roi=geo_roi,
            shape_prep=None,
            output=OutputSpec.pooled("max"),
        )

    whole, _, whole_meta, _ = pool(_FULL)
    roi, _, roi_meta, roi_gmeta = pool(_TOP)
    # Both are the max over tokens; a max over modality-mean cells gave 5.
    assert whole.tolist() == [10.0] and whole_meta["pooling"] == "token_max"
    assert roi.tolist() == [10.0] and roi_meta["pooling"] == "roi_grid_max"
    assert roi_gmeta["grid_kind"] == "spectral_max_patch_tokens"


def test_fomo_grid_rejects_an_unknown_modality_reduction():
    from rs_embed.core.errors import ModelError
    from rs_embed.embedders.onthefly_fomo import _tokens_to_grid

    with pytest.raises(ModelError, match="Unknown pooling"):
        _tokens_to_grid(
            _spike_tokens(groups=2), n_modalities=2, patch_size=4, image_size=16, reduce="median"
        )


def test_fomo_warns_when_its_tokens_form_no_grid():
    from rs_embed.embedders.onthefly_fomo import _fomo_pool_and_crop

    with pytest.warns(UserWarning, match="no token grid"):
        vec, _, meta, gmeta = _fomo_pool_and_crop(
            np.ones((7, 2), np.float32),
            n_modalities=2,
            patch_size=4,
            image_size=16,
            geo_roi=_TOP,
            shape_prep=None,
            output=OutputSpec.pooled(),
        )
    assert gmeta["grid_kind"] == "vector_as_1x1" and meta["roi_pooled"] is False


def test_satmaepp_s2_roi_max_takes_the_max_over_the_rois_group_tokens(monkeypatch):
    import rs_embed.embedders.onthefly_satmaepp_s2 as s2

    tokens = _spike_tokens(groups=len(s2._S2_10_CHANNEL_GROUPS))
    out = s2._build_satmaepp_s2_embedding(
        tokens, geo_roi=_TOP, output=OutputSpec.pooled("max"), meta={}, group_reduce="mean"
    )
    assert out.data.tolist() == [10.0]
    assert out.meta["pooling"] == "roi_grid_max"


# --------------------------------------------------------------------------- scalemae


def test_scalemae_pooled_vector_warns_for_an_roi_subwindow():
    from rs_embed.embedders.onthefly_scalemae import build_scalemae_embedding

    with pytest.warns(UserWarning, match="ScaleMAE has no token grid"):
        out = build_scalemae_embedding(
            np.ones(3, np.float32), geo_roi=_TOP, output=OutputSpec.pooled(), meta={}
        )
    assert out.meta["pooling"] == "model_pooled" and out.meta["roi_pooled"] is False

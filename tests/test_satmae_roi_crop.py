"""Fetch-square ROI crop-back behavior for the SatMAE RGB embedder.

These exercise :func:`build_token_embedding` directly (no provider / model load
needed): a full-frame ROI must reproduce the legacy token path, and a sub-window
ROI must crop the patch-token grid back to the ROI and pool only those tokens.
"""

from __future__ import annotations

import numpy as np

from rs_embed.core.specs import OutputSpec
from rs_embed.embedders.onthefly_satmae import build_token_embedding

_HW = 14  # 224 / 16 patch grid
_D = 8


def _tokens_with_cls() -> np.ndarray:
    patch = np.arange(_HW * _HW * _D, dtype=np.float32).reshape(_HW * _HW, _D)
    return np.concatenate([np.full((1, _D), -1.0, dtype=np.float32), patch], axis=0)


def test_full_frame_grid_is_unchanged():
    emb = build_token_embedding(
        _tokens_with_cls(), geo_roi=(0.0, 1.0, 0.0, 1.0), output=OutputSpec.grid(), meta={}
    )
    assert emb.meta["grid_hw"] == (_HW, _HW)
    assert emb.meta["cls_removed"] is True
    assert tuple(emb.data.shape) == (_D, _HW, _HW)


def test_subwindow_crops_grid_back_to_roi():
    roi = (0.0, 0.5, 0.0, 0.5)  # top-left quarter
    emb = build_token_embedding(_tokens_with_cls(), geo_roi=roi, output=OutputSpec.grid(), meta={})
    assert emb.meta["grid_hw"] == (7, 7)
    assert tuple(emb.data.shape) == (_D, 7, 7)


def test_pooled_full_vs_roi_differ_and_label():
    tokens = _tokens_with_cls()
    full = build_token_embedding(
        tokens, geo_roi=(0.0, 1.0, 0.0, 1.0), output=OutputSpec.pooled(), meta={}
    )
    roi = build_token_embedding(
        tokens, geo_roi=(0.0, 0.5, 0.0, 0.5), output=OutputSpec.pooled(), meta={}
    )
    assert full.meta["pooling"] == "patch_mean"
    assert roi.meta["pooling"] == "roi_grid_mean"
    assert full.data.shape == roi.data.shape == (_D,)
    assert not np.allclose(full.data, roi.data)


def test_roi_pooled_equals_manual_roi_token_mean():
    tokens = _tokens_with_cls()
    roi = (0.0, 0.5, 0.0, 0.5)
    emb = build_token_embedding(tokens, geo_roi=roi, output=OutputSpec.pooled(), meta={})
    # Manually: drop CLS, reshape to grid, take top-left 7x7, mean over tokens.
    patch = tokens[1:].reshape(_HW, _HW, _D)
    expected = patch[:7, :7, :].reshape(-1, _D).mean(axis=0).astype(np.float32)
    assert np.allclose(emb.data, expected)

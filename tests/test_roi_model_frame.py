"""ROI crops land in the model input's frame, not the fetched input's (issue #139).

A fetch-square ROI window (``roi_window_geo``) is relative to the fetched array.
Models that square that array (``prepare_square`` pad/crop, a center crop, a
bottom/right pad) feed the encoder a different frame, so each maps the window
through its preprocessing record (``shape.model_roi_window``) before cropping.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from rs_embed.core.specs import OutputSpec, TemporalSpec

# Input padded top and bottom: on an 8-row model grid it fills rows 2..5.
_PAD_ROWS = {"input_window": (0.25, 0.75, 0.0, 1.0)}
_FULL = (0.0, 1.0, 0.0, 1.0)


def _row_tokens(side: int = 8, dim: int = 2) -> np.ndarray:
    """[N, D] tokens whose value is their grid row, so crops are visible."""
    rows = np.repeat(np.arange(side, dtype=np.float32), side)
    return np.repeat(rows[:, None], dim, axis=1)


@pytest.mark.parametrize("model", ["clay", "dofa"])
def test_builder_crops_grid_in_the_model_frame(model):
    tokens, pooled = _row_tokens(), np.full(2, -1.0, dtype=np.float32)

    def build(geo_roi, shape_prep, output):
        if model == "clay":
            from rs_embed.embedders.onthefly_clay import build_clay_embedding

            return build_clay_embedding(
                tokens, pooled, geo_roi=geo_roi, output=output, base_meta={}, shape_prep=shape_prep
            )
        from rs_embed.embedders.onthefly_dofa import _build_dofa_embedding

        # Without global pooling DOFA's tokens are already normed: no ROI head.
        return _build_dofa_embedding(
            tokens,
            pooled,
            model=types.SimpleNamespace(global_pool=False, patch_size=16),
            geo_roi=geo_roi,
            shape_prep=shape_prep,
            output=output,
            base_meta={},
        )

    # Whole ROI on a padded input: grid drops the padding rows, pooled stays native.
    grid = np.asarray(build(_FULL, _PAD_ROWS, OutputSpec.grid()).data)
    assert grid.shape == (2, 4, 8)
    assert grid[0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    pooled_out = build(_FULL, _PAD_ROWS, OutputSpec.pooled())
    np.testing.assert_array_equal(pooled_out.data, pooled)

    # Sub-window ROI (the lower half of the fetched input) maps through the pad:
    # rows 4..5 of the model grid, not rows 4..7 as the raw window would give.
    sub = (0.5, 1.0, 0.0, 1.0)
    grid = np.asarray(build(sub, _PAD_ROWS, OutputSpec.grid()).data)
    assert grid[0, :, 0].tolist() == [4.0, 5.0]
    pooled_out = build(sub, _PAD_ROWS, OutputSpec.pooled())
    np.testing.assert_allclose(pooled_out.data, [4.5, 4.5])
    assert pooled_out.meta["pooling"] == "roi_grid_mean"


def test_olmoearth_merge_composes_the_geo_roi_with_the_pad():
    from rs_embed.embedders.onthefly_olmoearth import _merge_geo_roi

    shape_meta = {"shape_prep": dict(_PAD_ROWS, roi_window=(0.25, 0.75, 0.0, 1.0))}
    merged = _merge_geo_roi(shape_meta, (0.5, 1.0, 0.0, 1.0))
    # Used to overwrite the pad window with the raw geo window (0.5, 1.0, ...).
    assert merged["shape_prep"]["roi_window"] == (0.5, 0.75, 0.0, 1.0)
    assert merged["shape_prep"]["roi_source"] == "fetch_square"
    # A whole geo ROI keeps the pad window untouched.
    assert _merge_geo_roi(shape_meta, _FULL) is shape_meta


def test_olmoearth_merge_handles_a_cropped_input():
    from rs_embed.embedders.onthefly_olmoearth import _merge_geo_roi
    from rs_embed.tools.shape import prepare_square

    # 20x16 cropped to 16x16: 2 rows lost at each end of the input.
    _, shape_meta = prepare_square(np.ones((1, 20, 16), np.float32), size=8, shape_adjust="crop")
    merged = _merge_geo_roi(shape_meta, (0.5, 1.0, 0.0, 1.0))
    # Input row 10 of 20 is square row 8 of 16; the bottom 2 input rows were cropped.
    assert merged["shape_prep"]["roi_window"] == pytest.approx((0.5, 1.0, 0.0, 1.0))
    merged = _merge_geo_roi(shape_meta, (0.0, 0.25, 0.0, 1.0))
    assert merged["shape_prep"]["roi_window"] == pytest.approx((0.0, 3 / 16, 0.0, 1.0))


def test_remoteclip_center_crop_window():
    from rs_embed.embedders.onthefly_remoteclip import _remoteclip_shape_prep
    from rs_embed.tools.shape import model_roi_window

    # 100x200 RGB: Resize(short=100 -> 224) then CenterCrop keeps the middle half
    # of the width, so the input spans x in (-0.5, 1.5) of the model input.
    prep = _remoteclip_shape_prep(np.zeros((100, 200, 3), np.uint8))
    assert prep["input_window"] == (0.0, 1.0, -0.5, 1.5)
    # A ROI over the middle half of the width is the whole model input.
    assert model_roi_window(prep, (0.0, 1.0, 0.25, 0.75)) == (0.0, 1.0, 0.0, 1.0)
    # A square image is fed as-is.
    assert _remoteclip_shape_prep(np.zeros((64, 64, 3), np.uint8))["input_window"] == _FULL

    # A wrapper with its own transform has unknown geometry: no mapping.
    class _WithTransform:
        def transform(self, img, size):
            return img

    assert _remoteclip_shape_prep(np.zeros((100, 200, 3), np.uint8), _WithTransform()) is None


@pytest.mark.parametrize(
    ("prep_mode", "window"),
    [
        # 20x30 padded bottom/right to 32x32: the input keeps the top-left corner.
        ("pad", (0.0, 20 / 32, 0.0, 30 / 32)),
        # 20x30 reflect-padded to a centered 30x30 square, then resized.
        ("resize", (5 / 30, 25 / 30, 0.0, 1.0)),
    ],
)
def test_prithvi_prep_records_the_input_window(monkeypatch, prep_mode, window):
    from rs_embed.embedders.onthefly_prithvi import _prepare_prithvi_chw

    monkeypatch.setenv("RS_EMBED_PRITHVI_PREP", prep_mode)
    monkeypatch.setenv("RS_EMBED_PRITHVI_IMG", "32")
    _, meta = _prepare_prithvi_chw(np.ones((6, 20, 30), np.float32), fill_value=0.0)

    assert meta["shape_prep"]["input_window"] == pytest.approx(window)
    assert meta["shape_prep"]["applied"] in {"pad_bottom_right", "pad_to_square"}
    assert meta["shape_prep"]["orig_hw"] == (20, 30)


@pytest.mark.parametrize(
    ("module", "fn", "x"),
    [
        ("onthefly_agrifm", "_resize_tchw", np.ones((2, 10, 20, 30), np.float32)),
        ("onthefly_galileo", "_resize_tchw", np.ones((2, 10, 20, 30), np.float32)),
        ("onthefly_anysat", "_resize_tchw", np.ones((2, 10, 20, 30), np.float32)),
        ("onthefly_fomo", "_resize_chw", np.ones((12, 20, 30), np.float32)),
    ],
)
def test_squaring_helpers_return_their_input_window(module, fn, x):
    import importlib

    resize = getattr(importlib.import_module(f"rs_embed.embedders.{module}"), fn)
    out, shape_prep = resize(x, out_hw=16)

    assert out.shape[-2:] == (16, 16)
    # 20 rows reflect-padded to a centered 30-row square.
    assert shape_prep["input_window"] == pytest.approx((5 / 30, 25 / 30, 0.0, 1.0))


@pytest.mark.parametrize("transpose", [False, True])
def test_remoteclip_window_matches_its_preprocessing(transpose):
    """The recorded window is where the built-in preprocessing really puts the image."""
    pytest.importorskip("torchvision")
    from PIL import Image

    from rs_embed.embedders.onthefly_remoteclip import (
        _remoteclip_preprocess,
        _remoteclip_shape_prep,
    )

    img = np.zeros((100, 200, 3), np.uint8)
    img[20:40, 60:90] = 255
    if transpose:
        img = np.ascontiguousarray(img.transpose(1, 0, 2))
    h, w = img.shape[:2]
    x = _remoteclip_preprocess(224)(Image.fromarray(img, mode="RGB"))[0].numpy()

    # Centroid of the bright patch in the model input, in pixels.
    weight = np.clip(x - x.min(), 0.0, None)
    centers = np.arange(224) + 0.5
    cy = (weight.sum(axis=1) * centers).sum() / weight.sum()
    cx = (weight.sum(axis=0) * centers).sum() / weight.sum()

    # Where the recorded window puts the patch's centroid.
    rows, cols = np.nonzero(img[..., 0])
    y0, y1, x0, x1 = _remoteclip_shape_prep(img)["input_window"]
    want_y = (y0 + (rows + 0.5).mean() / h * (y1 - y0)) * 224
    want_x = (x0 + (cols + 0.5).mean() / w * (x1 - x0)) * 224
    assert cy == pytest.approx(want_y, abs=0.5)
    assert cx == pytest.approx(want_x, abs=0.5)


def test_remoteclip_records_shape_prep_in_meta(monkeypatch):
    from rs_embed.core.specs import PointBuffer
    from rs_embed.embedders import onthefly_remoteclip as rc

    monkeypatch.setattr(
        rc,
        "_remoteclip_encode_pooled_batch",
        lambda model, rgbs, **kw: np.zeros((len(rgbs), 4), np.float32),
    )
    monkeypatch.setattr(
        rc.RemoteCLIPS2RGBEmbedder, "_get_model", lambda self, **k: (object(), {}, "cpu")
    )
    emb = rc.RemoteCLIPS2RGBEmbedder()
    kw = dict(temporal=TemporalSpec.range("2020-01-01", "2020-02-01"), output=OutputSpec.pooled())
    chw = np.full((3, 16, 32), 1000.0, np.float32)
    spatial = PointBuffer(lon=0.0, lat=0.0, buffer_m=100)
    want = (0.0, 1.0, -0.5, 1.5)

    single = emb.get_embedding(spatial=spatial, sensor=None, backend="gee", input_chw=chw, **kw)
    assert single.meta["shape_prep"]["input_window"] == want
    (batch,) = emb.get_embeddings_batch_from_inputs(
        spatials=[spatial], input_chws=[chw], backend="gee", **kw
    )
    assert batch.meta["shape_prep"]["input_window"] == want


@pytest.mark.parametrize(("mode", "pooling"), [("grid", "token_mean"), ("pooled", "roi_grid_mean")])
def test_fomo_labels_the_vector_it_pooled(monkeypatch, mode, pooling):
    import rs_embed.embedders.onthefly_fomo as fomo

    monkeypatch.setenv("RS_EMBED_FOMO_IMG", "16")
    monkeypatch.setenv("RS_EMBED_FOMO_PATCH", "2")
    monkeypatch.setattr(fomo, "_resolve_fomo_ckpt_path", lambda: "unused")
    monkeypatch.setattr(fomo, "_load_fomo", lambda **kw: (object(), {}, "cpu"))
    # 12 modalities of an 8x8 token grid whose value is the token's row.
    monkeypatch.setattr(
        fomo,
        "_fomo_forward_tokens",
        lambda model, x, *, spectral_keys, device: (
            np.tile(_row_tokens(), (len(spectral_keys), 1)),
            {},
        ),
    )
    out = fomo.FoMoEmbedder().get_embedding(
        spatial=None,
        temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
        sensor=None,
        output=OutputSpec.grid() if mode == "grid" else OutputSpec.pooled(),
        backend="auto",
        input_chw=np.ones((12, 16, 16), np.float32),
        fetch_meta={"roi_window_geo": (0.5, 1.0, 0.0, 1.0)},
    )
    # Grid output keeps the whole-input token mean as its vector (rows 0..7);
    # pooled output pools the ROI's rows 4..7. The label must say which.
    assert out.meta["pooling"] == pooling
    if mode == "pooled":
        np.testing.assert_allclose(out.data, [5.5, 5.5])

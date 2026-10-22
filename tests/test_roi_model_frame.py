"""ROI crops land in the model input's frame, not the fetched input's (issue #139).

A fetch-square ROI window (``roi_window_geo``) is relative to the fetched array.
Models that square that array (``prepare_square`` pad/crop, a center crop, a
bottom/right pad) feed the encoder a different frame, so each maps the window
through its preprocessing record (``shape.model_roi_windows``) before cropping.
"""

from __future__ import annotations

import numpy as np
import pytest

from rs_embed.core.specs import OutputSpec

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

        return _build_dofa_embedding(
            tokens,
            pooled,
            geo_roi=geo_roi,
            shape_prep=shape_prep,
            output=output,
            base_meta={},
            patch_size=16,
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
    from rs_embed.tools.shape import model_roi_windows

    # 100x200 RGB: Resize(short=100 -> 224) then CenterCrop keeps the middle half
    # of the width, so the input spans x in (-0.5, 1.5) of the model input.
    prep = _remoteclip_shape_prep(np.zeros((100, 200, 3), np.uint8))
    assert prep["input_window"] == (0.0, 1.0, -0.5, 1.5)
    # A ROI over the middle half of the width is the whole model input.
    grid, _ = model_roi_windows(prep, (0.0, 1.0, 0.25, 0.75))
    assert grid == (0.0, 1.0, 0.0, 1.0)
    # A square image is fed as-is.
    assert _remoteclip_shape_prep(np.zeros((64, 64, 3), np.uint8))["input_window"] == _FULL


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

"""Tests for the unified square-input shape preparation (tools/shape.py)."""

from __future__ import annotations

import numpy as np
import pytest

from rs_embed.core.errors import ModelError
from rs_embed.tools.shape import (
    center_crop_to_square,
    center_fit,
    center_offset,
    center_pad_to_square,
    crop_grid_to_roi,
    model_roi_windows,
    prepare_square,
    resize_square,
    roi_in_frame,
    roi_is_full,
    roi_token_box,
)


def test_crop_to_square_chw():
    x = np.arange(3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)  # H=4, W=8
    y = center_crop_to_square(x)
    assert y.shape == (3, 4, 4)  # side = min(H, W) = 4
    # centered: columns 2..6 of the original
    np.testing.assert_array_equal(y, x[:, :, 2:6])


def test_pad_to_square_chw_shape():
    x = np.ones((3, 4, 8), dtype=np.float32)  # H=4, W=8
    y = center_pad_to_square(x)
    assert y.shape == (3, 8, 8)  # side = max(H, W) = 8


def test_pad_to_square_tchw():
    x = np.ones((2, 3, 5, 9), dtype=np.float32)  # T=2,C=3,H=5,W=9
    y = center_pad_to_square(x)
    assert y.shape == (2, 3, 9, 9)


def test_pad_constant_fill_value():
    x = np.ones((1, 2, 4), dtype=np.float32)  # H=2, W=4
    y = center_pad_to_square(x, fill_value=7.0, pad_mode="constant")
    assert y.shape == (1, 4, 4)
    # original content preserved in the centered band
    assert (y[:, 1:3, :] == 1.0).all()
    # padded rows carry the fill value
    assert (y[:, 0, :] == 7.0).all()
    assert (y[:, 3, :] == 7.0).all()


def test_pad_reflect_falls_back_to_edge_when_pad_too_large():
    # H=1, W=5 -> pad_h=4 >= h=1, reflect impossible, must not raise
    x = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
    y = center_pad_to_square(x, pad_mode="reflect")
    assert y.shape == (1, 5, 5)
    assert np.isfinite(y).all()


def test_resize_square_preserves_leading_dims():
    x = np.ones((2, 12, 7, 9), dtype=np.float32)  # T,C,H,W
    y = resize_square(x, size=16)
    assert y.shape == (2, 12, 16, 16)


def test_resize_square_noop_when_already_target():
    x = np.random.rand(3, 8, 8).astype(np.float32)
    y = resize_square(x, size=8)
    np.testing.assert_array_equal(y, x)


def test_prepare_square_already_square_just_resizes():
    x = np.random.rand(4, 32, 32).astype(np.float32)
    y, meta = prepare_square(x, size=64)
    assert y.shape == (4, 64, 64)
    assert meta["shape_prep"]["applied"] == "none"


def test_prepare_square_pad_default():
    x = np.random.rand(12, 33, 59).astype(np.float32)  # the reported field case
    y, meta = prepare_square(x, size=256)
    assert y.shape == (12, 256, 256)
    sp = meta["shape_prep"]
    assert sp["applied"] == "pad_to_square"
    assert sp["square_hw"] == (59, 59)
    assert sp["orig_hw"] == (33, 59)


def test_prepare_square_roi_window_pad():
    # 33 tall padded to 59 → ROI occupies the centered 33/59 band vertically,
    # full width horizontally.
    x = np.random.rand(12, 33, 59).astype(np.float32)
    _, meta = prepare_square(x, size=256)
    y0, y1, x0, x1 = meta["shape_prep"]["roi_window"]
    top = (59 - 33) // 2  # 13
    # Exact fractions: 6-decimal rounding put token-aligned edges a token off.
    assert y0 == top / 59
    assert y1 == (top + 33) / 59
    assert (x0, x1) == (0.0, 1.0)  # width unchanged
    assert meta["shape_prep"]["input_window"] == (y0, y1, x0, x1)


def test_prepare_square_roi_window_full_when_not_padded():
    x = np.random.rand(3, 32, 32).astype(np.float32)  # already square → none
    _, meta = prepare_square(x, size=64)
    assert roi_is_full(meta["shape_prep"]["roi_window"])


def test_roi_token_box_basic_and_clamping():
    # full window → whole grid
    assert roi_token_box((0.0, 1.0, 0.0, 1.0), grid_h=64, grid_w=64) == (0, 64, 0, 64)
    # vertical sub-band, rounds outward, full width
    y0, y1, x0, x1 = roi_token_box((13 / 59, 46 / 59, 0.0, 1.0), grid_h=64, grid_w=64)
    assert (x0, x1) == (0, 64)
    assert 0 < y0 < y1 <= 64
    # degenerate window still yields a 1-wide box
    assert roi_token_box((0.5, 0.5, 0.5, 0.5), grid_h=8, grid_w=8)[1] >= 1


def test_crop_grid_to_roi():
    grid = np.random.rand(16, 64, 64).astype(np.float32)
    # vertical band only
    out = crop_grid_to_roi(grid, (13 / 59, 46 / 59, 0.0, 1.0))
    assert out.shape[0] == 16
    assert out.shape[2] == 64  # full width
    assert out.shape[1] < 64  # cropped height
    # full window is a no-op (same object)
    assert crop_grid_to_roi(grid, (0.0, 1.0, 0.0, 1.0)) is grid


def test_prepare_square_crop_mode():
    x = np.random.rand(12, 33, 59).astype(np.float32)
    y, meta = prepare_square(x, size=64, shape_adjust="crop")
    assert y.shape == (12, 64, 64)
    assert meta["shape_prep"]["applied"] == "crop_to_square"
    assert meta["shape_prep"]["square_hw"] == (33, 33)


def test_prepare_square_extreme_aspect_pads_and_warns():
    # Extreme rectangles pad (never silently stretch) so the roi_window always
    # points back at the real ROI; a warning flags the synthetic-border cost.
    x = np.random.rand(3, 10, 90).astype(np.float32)  # aspect 9.0 >> tol
    with pytest.warns(UserWarning, match="aspect ratio"):
        y, meta = prepare_square(x, size=32)
    assert y.shape == (3, 32, 32)
    assert meta["shape_prep"]["applied"] == "pad_to_square"
    y0, y1, x0, x1 = meta["shape_prep"]["roi_window"]
    top = (90 - 10) // 2
    assert y0 == top / 90
    assert y1 == (top + 10) / 90
    assert (x0, x1) == (0.0, 1.0)


def test_prepare_square_aspect_tol_boundary():
    # aspect exactly 2.0 with default tol still pads (warns at >= tol)
    x = np.random.rand(3, 20, 40).astype(np.float32)
    with pytest.warns(UserWarning, match="aspect ratio"):
        _, meta = prepare_square(x, size=32)
    assert meta["shape_prep"]["applied"] == "pad_to_square"
    assert not roi_is_full(meta["shape_prep"]["roi_window"])


def test_prepare_square_invalid_shape_adjust():
    x = np.random.rand(3, 10, 10).astype(np.float32)
    with pytest.raises(ModelError):
        prepare_square(x, size=32, shape_adjust="bogus")


def test_prepare_square_pad_preserves_roi_content_after_resize():
    # A pad-to-square + resize must not horizontally/vertically stretch the ROI:
    # a vertically-striped input should stay vertically striped (no aspect flip).
    h, w = 20, 36
    base = np.zeros((1, h, w), dtype=np.float32)
    base[:, :, ::2] = 1.0  # vertical stripes
    y, meta = prepare_square(base, size=72, shape_adjust="pad")
    assert meta["shape_prep"]["applied"] == "pad_to_square"
    # variance across columns (x) should dominate variance across rows (y),
    # i.e. stripes remain vertical rather than being smeared into horizontal.
    col_var = float(y[0].mean(axis=0).var())
    row_var = float(y[0].mean(axis=1).var())
    assert col_var > row_var


def test_geo_roi_from_meta():
    from rs_embed.tools.shape import geo_roi_from_meta
    from rs_embed.tools.spatial import FULL_WINDOW

    assert geo_roi_from_meta(None) == FULL_WINDOW
    assert geo_roi_from_meta({}) == FULL_WINDOW
    assert geo_roi_from_meta({"roi_window_geo": (0.2, 0.8, 0.0, 1.0)}) == (0.2, 0.8, 0.0, 1.0)


def test_roi_fetch_meta():
    from rs_embed.tools.shape import roi_fetch_meta
    from rs_embed.tools.spatial import FULL_WINDOW

    assert roi_fetch_meta(FULL_WINDOW) is None
    assert roi_fetch_meta((0.2, 0.8, 0.0, 1.0)) == {"roi_window_geo": (0.2, 0.8, 0.0, 1.0)}


def test_crop_grid_and_pool_full_returns_fallback():
    from rs_embed.tools.shape import crop_grid_and_pool
    from rs_embed.tools.spatial import FULL_WINDOW

    grid = np.random.rand(8, 6, 6).astype(np.float32)
    fb = np.ones(8, np.float32)
    g, vec = crop_grid_and_pool(grid, FULL_WINDOW, pooling="mean", pooled_fallback=fb)
    assert g is grid
    assert vec is fb


def test_crop_grid_and_pool_crops_and_pools():
    from rs_embed.tools.shape import crop_grid_and_pool

    grid = np.random.rand(8, 8, 8).astype(np.float32)
    g, vec = crop_grid_and_pool(grid, (0.25, 0.75, 0.0, 1.0), pooling="mean")
    assert g.shape[0] == 8 and g.shape[1] < 8 and g.shape[2] == 8
    np.testing.assert_allclose(vec, g.mean(axis=(1, 2)), rtol=1e-5)


def test_square_fetch_batch():
    from rs_embed.core.specs import BBox, PointBuffer
    from rs_embed.tools.shape import square_fetch_batch

    def fetch(sq):
        return np.zeros((3, 4, 4), np.float32)

    spatials = [BBox(-88.23, 40.09, -88.22, 40.10), PointBuffer(10, 20, 256)]
    raws, geo_rois = square_fetch_batch(spatials, fetch, max_workers=1)
    assert len(raws) == 2 and len(geo_rois) == 2
    assert geo_rois[0] != (0.0, 1.0, 0.0, 1.0)
    assert geo_rois[1] == (0.0, 1.0, 0.0, 1.0)


def test_roi_in_frame_maps_a_window_through_padding_and_cropping():
    # Input padded 4 px each side into a 208 px array: its frame is (4/208, 204/208).
    frame = (4 / 208, 204 / 208, 4 / 208, 204 / 208)
    assert roi_in_frame(frame, (0.0, 1.0, 0.0, 1.0)) == pytest.approx(frame)
    assert roi_in_frame(frame, (0.5, 1.0, 0.0, 0.5)) == pytest.approx(
        (104 / 208, 204 / 208, 4 / 208, 104 / 208)
    )
    # A cropped axis extends past [0, 1]; the mapped window is clipped to the array.
    assert roi_in_frame((-0.1, 1.1, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)) == (0.0, 1.0, 0.0, 1.0)


def test_center_fit_crops_and_pads_each_axis_around_center_offset():
    x = np.arange(5 * 3, dtype=np.float32).reshape(1, 5, 3)
    y, window = center_fit(x, out_h=4, out_w=6, pad_mode="edge")

    assert y.shape == (1, 4, 6)
    top, left = center_offset(5, 4), center_offset(3, 6)
    assert (top, left) == (0, 1)  # 1 px cropped after on H; 1 before / 2 after on W
    np.testing.assert_array_equal(y[0, :, 1:4], x[0, 0:4, :])
    np.testing.assert_array_equal(y[0, :, 0], x[0, 0:4, 0])  # edge replicated
    assert window == pytest.approx((0.0, 5 / 4, 1 / 6, 4 / 6))
    same, full = center_fit(x, out_h=5, out_w=3)
    assert same is x
    assert full == (0.0, 1.0, 0.0, 1.0)


def test_center_fit_rejects_bad_targets_and_empty_edge_pads():
    x = np.ones((1, 4, 4), dtype=np.float32)
    with pytest.raises(ModelError, match="non-negative"):
        center_fit(x, out_h=-1, out_w=4)
    # An empty axis still crops to an empty square, as it always did.
    assert center_crop_to_square(np.ones((1, 0, 4), dtype=np.float32)).shape == (1, 0, 0)
    with pytest.raises(ModelError, match="empty input"):
        center_fit(np.ones((1, 0, 4), dtype=np.float32), out_h=4, out_w=4, pad_mode="edge")
    padded, _ = center_fit(x, out_h=6, out_w=6, pad_mode="constant", fill_value=-1.0)
    assert padded[0, 0, 0] == -1.0


def test_roi_token_box_ignores_float_round_off_at_token_edges():
    # 4/392 of a 98-token grid is exactly token 1, but evaluates to 0.99999...
    assert roi_token_box((4 / 392, 1.0, 0.0, 1.0), grid_h=98, grid_w=98) == (1, 98, 0, 98)


def test_prepare_square_records_the_input_window_for_crop_and_none():
    # crop: the input extends past the square on the cropped axis.
    x = np.ones((1, 20, 16), dtype=np.float32)
    _, meta = prepare_square(x, size=8, shape_adjust="crop")
    sp = meta["shape_prep"]
    assert sp["input_window"] == (-2 / 16, 18 / 16, 0.0, 1.0)
    assert sp["roi_window"] == (0.0, 1.0, 0.0, 1.0)
    # none: already square, so the input fills the square.
    _, meta = prepare_square(np.ones((1, 16, 16), dtype=np.float32), size=8)
    assert meta["shape_prep"]["input_window"] == (0.0, 1.0, 0.0, 1.0)


def test_model_roi_windows_maps_the_geo_roi_into_the_model_frame():
    pad = {"input_window": (0.25, 0.75, 0.0, 1.0)}  # input padded top and bottom
    # Whole ROI: crop grid output to the input (drop the pad), keep native pooling.
    assert model_roi_windows(pad, (0.0, 1.0, 0.0, 1.0)) == (
        (0.25, 0.75, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
    )
    # Sub-window ROI: both windows are the ROI mapped through the pad.
    grid, pool = model_roi_windows(pad, (0.5, 1.0, 0.0, 1.0))
    assert grid == pool == (0.5, 0.75, 0.0, 1.0)
    # A cropped axis is clipped to the model input.
    crop = {"input_window": (-0.125, 1.125, 0.0, 1.0)}
    assert model_roi_windows(crop, (0.0, 0.5, 0.0, 1.0))[0] == (0.0, 0.5, 0.0, 1.0)
    # No preprocessing record (as-is or stretched input): the frames agree.
    assert model_roi_windows(None, (0.2, 0.8, 0.0, 1.0)) == ((0.2, 0.8, 0.0, 1.0),) * 2


def test_square_spatial_window_is_exact():
    # The window must be the exact fraction, not a 6-decimal approximation:
    # rounding put token-aligned edges one token off on large grids.
    from rs_embed.core.specs import BBox
    from rs_embed.tools.projection import lonlat_to_web_mercator
    from rs_embed.tools.spatial import square_spatial

    bbox = BBox(minlon=0.0, minlat=-0.0013, maxlon=0.0098, maxlat=0.0011)
    _, geo = square_spatial(bbox)
    x0, y0 = lonlat_to_web_mercator(bbox.minlon, bbox.minlat)
    x1, y1 = lonlat_to_web_mercator(bbox.maxlon, bbox.maxlat)
    w, h = abs(x1 - x0), abs(y1 - y0)
    top = (1.0 - h / w) / 2.0
    assert geo == (top, top + h / w, 0.0, 1.0)
    assert geo[0] != round(geo[0], 6)

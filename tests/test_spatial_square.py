"""Tests for fetch-square ROI enlargement (tools/spatial.py)."""

from __future__ import annotations

from rs_embed.core.specs import BBox, PointBuffer
from rs_embed.tools.shape import roi_is_full
from rs_embed.tools.spatial import FULL_WINDOW, square_spatial


def _merc_wh(bbox: BBox) -> tuple[float, float]:
    from rs_embed.tools.spatial import _to_mercator

    x0, y0 = _to_mercator(bbox.minlon, bbox.minlat)
    x1, y1 = _to_mercator(bbox.maxlon, bbox.maxlat)
    return abs(x1 - x0), abs(y1 - y0)


def test_pointbuffer_unchanged_full_window():
    pb = PointBuffer(lon=10.0, lat=20.0, buffer_m=256)
    out, win = square_spatial(pb)
    assert out is pb
    assert win == FULL_WINDOW


def test_already_square_bbox_full_window():
    # A box that is square in EPSG:3857 stays unchanged.
    sq = BBox(minlon=-0.005, minlat=0.0, maxlon=0.005, maxlat=0.01)
    w, h = _merc_wh(sq)
    assert abs(w - h) / max(w, h) < 0.01  # roughly square near equator
    out, win = square_spatial(sq)
    assert roi_is_full(win)


def test_wide_bbox_enlarges_to_square_and_centers_roi():
    # The reported field: ~592 m wide × ~333 m tall near 40°N → wide box.
    bbox = BBox(minlon=-88.23131, minlat=40.09466, maxlon=-88.22436, maxlat=40.09767)
    w0, h0 = _merc_wh(bbox)
    assert w0 > h0  # wider than tall

    square, win = square_spatial(bbox)
    assert isinstance(square, BBox)
    w1, h1 = _merc_wh(square)
    # enlarged box is square in 3857
    assert abs(w1 - h1) / max(w1, h1) < 1e-3
    # width preserved (the long side), height grown to match
    assert abs(w1 - w0) / w0 < 1e-3
    assert h1 > h0

    y0, y1, x0, x1 = win
    # width axis unchanged → full; height axis is a centered sub-band
    assert (x0, x1) == (0.0, 1.0)
    assert 0.0 < y0 < y1 < 1.0
    assert abs((y0 + y1) / 2.0 - 0.5) < 1e-6  # centered
    # ROI height fraction matches original aspect
    assert abs((y1 - y0) - h0 / w0) < 1e-3


def test_tall_bbox_enlarges_width():
    # taller than wide → width grows, height-axis window stays full
    bbox = BBox(minlon=0.0, minlat=0.0, maxlon=0.001, maxlat=0.01)
    square, win = square_spatial(bbox)
    w1, h1 = _merc_wh(square)
    assert abs(w1 - h1) / max(w1, h1) < 1e-3
    y0, y1, x0, x1 = win
    assert (y0, y1) == (0.0, 1.0)  # height preserved
    assert 0.0 < x0 < x1 < 1.0  # width centered sub-band


def test_enlargement_out_of_bounds_falls_back():
    # A box hugging the antimeridian: enlarging longitude would exceed 180.
    bbox = BBox(minlon=179.99, minlat=0.0, maxlon=179.999, maxlat=10.0)
    out, win = square_spatial(bbox)
    # falls back to original spec + full window rather than producing invalid bbox
    assert out is bbox
    assert win == FULL_WINDOW

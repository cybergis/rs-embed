"""Live-GEE check that computePixels downloads equal the sampleRectangle ones (opt-in).

rs-embed used to download pixels with ``sampleRectangle(...).getInfo()``, which
sends every value as a JSON number; ``_compute_pixels_chw`` now asks
``ee.data.computePixels`` for binary float32 on the common grid. The fetched
values must not change. Each case builds one image expression the way a
provider path does, downloads it both ways — the replaced call is written out
here as the oracle — and compares the arrays value for value, including the
fill of masked pixels. Run with::

    RS_EMBED_LIVE_GEE=1 pytest tests/test_gee_compute_pixels_live.py

Requires authenticated Earth Engine credentials (``EE_PROJECT`` if needed).
"""

import os

import numpy as np
import pytest

from rs_embed.core.specs import PointBuffer
from rs_embed.providers.gee import GEEProvider
from rs_embed.providers.gee_utils import _compute_pixels_chw, _order_collection_for_mosaic
from rs_embed.tools.projection import COMMON_CRS, common_grid

pytestmark = pytest.mark.skipif(
    not os.environ.get("RS_EMBED_LIVE_GEE"),
    reason="live GEE test; set RS_EMBED_LIVE_GEE=1 to run",
)

_S2 = "COPERNICUS/S2_SR_HARMONIZED"
_S2_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")
_SCALE_M = 10


@pytest.fixture(scope="module")
def provider() -> GEEProvider:
    p = GEEProvider()
    p.ensure_ready()
    return p


def _oracle(img, region, bands, fill, *, south_up):
    """The replaced download: sampleRectangle JSON, rows flipped for the atScale form."""
    rect = img.sampleRectangle(region=region, defaultValue=fill).getInfo()
    arr = np.stack([np.array(rect["properties"][b], dtype=np.float32) for b in bands])
    return arr[:, ::-1] if south_up else arr


def _s2_median(region, bands):
    import ee

    col = ee.ImageCollection(_S2).filterDate("2022-06-01", "2022-09-01").filterBounds(region)
    return col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)).median().select(list(bands))


def _cases(region):
    """(case id, expression, bands, fill, south_up) — one per provider download form."""
    import ee

    s2 = _s2_median(region, _S2_BANDS)
    at_scale = ee.Projection(COMMON_CRS).atScale(_SCALE_M)
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate("2022-06-01", "2022-09-01")
        .filterBounds(region)
        .median()
        .select(["VV", "VH"])
    )
    gse = _order_collection_for_mosaic(
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate("2024-01-01", "2025-01-01")
        .filterBounds(region)
    ).mosaic()
    gse_bands = tuple(f"A{i:02d}" for i in range(64))
    inner = ee.Geometry.Point([-88.2272, 40.1020]).buffer(150)
    return [
        ("sensor-patch", s2.reproject(at_scale).clip(region), _S2_BANDS, 0.0, True),
        (
            "multi-frame",
            s2.reproject(crs=COMMON_CRS, scale=_SCALE_M),
            _S2_BANDS,
            0.0,
            False,
        ),
        ("s1", s1.reproject(crs=COMMON_CRS, scale=_SCALE_M), ("VV", "VH"), 0.0, False),
        (
            "all-bands",
            gse.reproject(crs=COMMON_CRS, scale=_SCALE_M).clip(region),
            gse_bands,
            -9999.0,
            False,
        ),
        (
            "masked-fill",
            _s2_median(region, ("B4", "B8")).reproject(crs=COMMON_CRS, scale=_SCALE_M).clip(inner),
            ("B4", "B8"),
            7.0,
            False,
        ),
    ]


@pytest.mark.parametrize("case", ["sensor-patch", "multi-frame", "s1", "all-bands", "masked-fill"])
def test_compute_pixels_matches_sample_rectangle(provider, case):
    spatial = PointBuffer(lon=-88.2272, lat=40.1020, buffer_m=300)
    region = provider.get_region(spatial)
    _, img, bands, fill, south_up = next(c for c in _cases(region) if c[0] == case)

    new = _compute_pixels_chw(img, spatial=spatial, bands=bands, scale_m=_SCALE_M, fill_value=fill)
    old = _oracle(img, region, bands, fill, south_up=south_up)

    assert new.shape == old.shape
    np.testing.assert_array_equal(new, old)
    if case == "masked-fill":
        assert (new == fill).any() and (new != fill).any(), "case must mix data and fill"


def test_south_edge_on_a_pixel_line_returns_the_common_grid(provider):
    """An ROI edge exactly on the lattice must not pull in a row beyond the ROI.

    sampleRectangle added one row south of an ROI whose south edge lay on a
    pixel line (latitude 0 here, with a buffer that is a multiple of the scale).
    """
    import ee

    spatial = PointBuffer(lon=10.2, lat=0.0, buffer_m=1000)
    grid = common_grid(spatial, scale_m=float(_SCALE_M))
    img = ee.Image.pixelCoordinates(COMMON_CRS).reproject(crs=COMMON_CRS, scale=_SCALE_M)

    arr = _compute_pixels_chw(
        img, spatial=spatial, bands=("x", "y"), scale_m=_SCALE_M, fill_value=0.0
    )

    assert arr.shape[1:] == grid.shape
    # pixel centers: the southern row sits half a pixel inside the ROI's south edge
    np.testing.assert_array_equal(arr[1, -1], -1000 + _SCALE_M / 2)
    np.testing.assert_array_equal(arr[1, 0], 1000 - _SCALE_M / 2)

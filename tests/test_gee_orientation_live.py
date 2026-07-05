"""Live-GEE row-order contract test (opt-in).

The provider layer's north-up guarantee rests on one empirical fact about
``sampleRectangle`` row order that mock tests cannot check:

- ``reproject(crs=..., scale=...)`` returns **north-up** rows, with or
  without ``.clip()`` (``_sample_image_bands_raw_chw``,
  ``_fetch_all_bands_impl`` — no flip applied);
- ``reproject(ee.Projection(...).atScale(...))`` returns **south-up** rows
  (``fetch_array_chw`` applies ``_flip_sample_tile_y``).

This test verifies that fact against real GEE using
``ee.Image.pixelLonLat()`` — each pixel's value is its own latitude, so row
order is read directly. Run with::

    RS_EMBED_LIVE_GEE=1 pytest tests/test_gee_orientation_live.py

Requires authenticated Earth Engine credentials.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RS_EMBED_LIVE_GEE"),
    reason="live GEE test; set RS_EMBED_LIVE_GEE=1 to run",
)


def _sample_latitude_rows(image, region) -> np.ndarray:
    rect = image.sampleRectangle(region=region, defaultValue=-999.0).getInfo()
    return np.array(rect["properties"]["latitude"], dtype=np.float64)


@pytest.mark.parametrize(
    "lon,lat,half_m,scale_m",
    [
        (0.0, 45.0, 1000, 100),  # northern hemisphere
        (151.2, -33.9, 500, 10),  # southern hemisphere, fine scale
    ],
)
def test_samplerectangle_row_order_contract(lon, lat, half_m, scale_m):
    import ee
    from pyproj import Transformer

    ee.Initialize()
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)
    region = ee.Geometry.Rectangle(
        [x - half_m, y - half_m, x + half_m, y + half_m], proj="EPSG:3857", geodesic=False
    )
    img = ee.Image.pixelLonLat().select("latitude")

    # reproject(crs=..., scale=...): north-up, with or without clip.
    for im in (
        img.reproject(crs="EPSG:3857", scale=scale_m),
        img.reproject(crs="EPSG:3857", scale=scale_m).clip(region),
    ):
        rows = _sample_latitude_rows(im, region)
        assert rows[0].mean() > rows[-1].mean(), "reproject(crs,scale) must be north-up"

    # Projection.atScale(): south-up (fetch_array_chw flips this).
    proj = ee.Projection("EPSG:3857").atScale(scale_m)
    rows = _sample_latitude_rows(img.reproject(proj).clip(region), region)
    assert rows[0].mean() < rows[-1].mean(), "atScale form must be south-up"

"""Live-GEE row-order contract test (opt-in).

Every GEE pixel download goes through ``_compute_pixels_chw``, which asks
``computePixels`` for the common grid with a negative ``scaleY`` (row 0 on the
north edge). The provider layer's north-up guarantee rests on GEE honoring
that grid whatever reprojection the image carries — both the
``reproject(crs=..., scale=...)`` form (S1, multi-frame, all-bands paths) and
the ``reproject(ee.Projection(...).atScale(...))`` form (``fetch_array_chw``),
with or without ``.clip()``. Mock tests cannot check that.

This test verifies it against real GEE using ``ee.Image.pixelLonLat()`` — each
pixel's value is its own latitude, so row order is read directly. Run with::

    RS_EMBED_LIVE_GEE=1 pytest tests/test_gee_orientation_live.py

Requires authenticated Earth Engine credentials.
"""

import os

import pytest

from rs_embed.core.specs import PointBuffer
from rs_embed.providers.gee import GEEProvider
from rs_embed.providers.gee_utils import _compute_pixels_chw
from rs_embed.tools.projection import COMMON_CRS

pytestmark = pytest.mark.skipif(
    not os.environ.get("RS_EMBED_LIVE_GEE"),
    reason="live GEE test; set RS_EMBED_LIVE_GEE=1 to run",
)


@pytest.mark.parametrize(
    "lon,lat,half_m,scale_m",
    [
        (0.0, 45.0, 1000, 100),  # northern hemisphere
        (151.2, -33.9, 500, 10),  # southern hemisphere, fine scale
    ],
)
def test_downloads_are_north_up_for_every_reprojection_form(lon, lat, half_m, scale_m):
    import ee

    provider = GEEProvider()
    provider.ensure_ready()
    spatial = PointBuffer(lon=lon, lat=lat, buffer_m=half_m)
    region = provider.get_region(spatial)
    img = ee.Image.pixelLonLat().select("latitude")
    at_scale = ee.Projection(COMMON_CRS).atScale(scale_m)

    for name, im in (
        ("reproject(crs, scale)", img.reproject(crs=COMMON_CRS, scale=scale_m)),
        ("reproject(crs, scale).clip", img.reproject(crs=COMMON_CRS, scale=scale_m).clip(region)),
        ("reproject(atScale).clip", img.reproject(at_scale).clip(region)),
    ):
        rows = _compute_pixels_chw(
            im, spatial=spatial, bands=("latitude",), scale_m=scale_m, fill_value=-999.0
        )[0]
        assert rows[0].mean() > rows[-1].mean(), f"{name} download must be north-up"

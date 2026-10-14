"""Live-GEE check that pixel-limit BBox splits are exact (opt-in, #134).

``_bbox_recursive_fallback`` cuts oversized requests on pixel-lattice lines
and concatenates the pieces. That is exact only if two facts about GEE hold,
which mock tests cannot check:

- both reprojection forms sample the EPSG:3857 lattice anchored at the
  projection origin (pixel edges at ``k * scale_m``), i.e. the common grid;
- a region whose edge sits just inside a lattice line gets exactly the pixels
  on its side of that line.

The ROI is small enough for one request, so each test compares a single
request against the concatenation of its split children, value for value.
Run with::

    RS_EMBED_LIVE_GEE=1 pytest tests/test_gee_bbox_split_live.py

Requires authenticated Earth Engine credentials (``EE_PROJECT`` if needed).
"""

import os

import numpy as np
import pytest

from rs_embed.core.specs import BBox, TemporalSpec
from rs_embed.providers.gee import GEEProvider
from rs_embed.providers.gee_utils import (
    _sample_image_bands_raw_chw,
    _split_bbox_for_recursive_fetch,
    _stitch_bbox_split_arrays,
)
from rs_embed.tools.projection import COMMON_CRS

pytestmark = pytest.mark.skipif(
    not os.environ.get("RS_EMBED_LIVE_GEE"),
    reason="live GEE test; set RS_EMBED_LIVE_GEE=1 to run",
)

# ~67 x 152 px at 100 m near Fairbanks: high latitude, where Mercator
# stretching made the old seams worst.
_BBOX = BBox(minlon=-149.03, minlat=63.97, maxlon=-148.97, maxlat=64.03)
_SCALE_M = 100


@pytest.fixture(scope="module")
def provider() -> GEEProvider:
    p = GEEProvider()
    p.ensure_ready()
    return p


def _split_matches_single(fetch, axis: str) -> None:
    a, b, got_axis = _split_bbox_for_recursive_fetch(_BBOX, scale_m=_SCALE_M, prefer_axis=axis)
    single = fetch(_BBOX)
    stitched = _stitch_bbox_split_arrays(fetch(a), fetch(b), got_axis)
    assert stitched.shape == single.shape
    np.testing.assert_array_equal(stitched, single)


def _pixel_coordinates():
    import ee

    return ee.Image.pixelCoordinates(COMMON_CRS)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_all_bands_split_matches_single_request(provider, axis):
    """The GSE path: reproject(crs, scale).clip(region).sampleRectangle()."""

    def fetch(sp):
        arr, _ = provider._fetch_all_bands_impl(
            spatial=sp,
            temporal=TemporalSpec.year(2024),
            collection="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            scale_m=_SCALE_M,
            fill_value=-9999.0,
            composite="mosaic",
        )
        return arr

    _split_matches_single(fetch, axis)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_array_chw_split_matches_single_request(provider, axis):
    """The on-the-fly path: reproject(Projection.atScale()).clip(region), flipped."""
    img = _pixel_coordinates()

    def fetch(sp):
        return provider.fetch_array_chw(
            image=img,
            bands=("x", "y"),
            region=provider.get_region(sp),
            scale_m=_SCALE_M,
            fill_value=0.0,
        )

    _split_matches_single(fetch, axis)


def test_both_reprojection_forms_sample_the_common_grid_lattice(provider):
    img = _pixel_coordinates()
    region = provider.get_region(_BBOX)
    fill = 0.0  # never a real 3857 coordinate this far from the origin
    for arr in (
        _sample_image_bands_raw_chw(
            img, region=region, bands=("x", "y"), scale_m=_SCALE_M, fill_value=fill
        ),
        provider.fetch_array_chw(
            image=img, bands=("x", "y"), region=region, scale_m=_SCALE_M, fill_value=fill
        ),
    ):
        arr = np.asarray(arr, dtype=np.float64)
        h, w = arr.shape[-2:]
        xs = arr[0, h // 2][arr[0, h // 2] != fill]
        ys = arr[1, :, w // 2][arr[1, :, w // 2] != fill]
        # Pixel centers sit at (k + 0.5) * scale_m on both axes ...
        for v in (xs, ys):
            np.testing.assert_array_equal(np.mod(v / _SCALE_M - 0.5, 1.0), 0.0)
        # ... and come back north-up, west to east, one pixel apart.
        np.testing.assert_array_equal(np.diff(xs), _SCALE_M)
        np.testing.assert_array_equal(np.diff(ys), -_SCALE_M)

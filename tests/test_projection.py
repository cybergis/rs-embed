"""Tests for the common grid and CRS helpers (tools/projection.py)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

from rs_embed.core.errors import SpecError
from rs_embed.core.specs import BBox, PointBuffer
from rs_embed.tools import projection as P


def test_web_mercator_roundtrip_and_pointbuffer_square():
    lon, lat = -88.2, 40.1
    x, y = P.lonlat_to_web_mercator(lon, lat)
    px, py = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)
    assert abs(x - px) < 1e-6 and abs(y - py) < 1e-6
    lon2, lat2 = P.web_mercator_to_lonlat(x, y)
    assert abs(lon2 - lon) < 1e-9 and abs(lat2 - lat) < 1e-9

    minx, miny, maxx, maxy = P.web_mercator_bounds(PointBuffer(lon=lon, lat=lat, buffer_m=320))
    assert abs((maxx - minx) - 640.0) < 1e-6 and abs((maxy - miny) - 640.0) < 1e-6


def test_common_grid_snaps_outward_to_the_lattice():
    def bbox(minx, miny, maxx, maxy):
        a = P.web_mercator_to_lonlat(minx, miny)
        b = P.web_mercator_to_lonlat(maxx, maxy)
        return BBox(minlon=a[0], minlat=a[1], maxlon=b[0], maxlat=b[1])

    g = P.common_grid(bbox(42.0, 12.0, 48.0, 18.0), scale_m=10)
    assert g.shape == (1, 1)
    assert g.transform == Affine(10, 0, 40, 0, -10, 20)
    assert g.crs == "EPSG:3857" and g.scale_m == 10

    g = P.common_grid(bbox(40.0, 10.0, 80.0, 40.0), scale_m=10)  # exactly on the lattice
    assert g.shape == (3, 4)
    assert g.bounds() == (40.0, 10.0, 80.0, 40.0)

    with pytest.raises(SpecError):
        P.common_grid(bbox(0.0, 0.0, 10.0, 10.0), scale_m=0)


def test_overlapping_requests_share_pixel_edges():
    a = P.common_grid(PointBuffer(lon=3.0, lat=45.0, buffer_m=100), scale_m=10)
    b = P.common_grid(PointBuffer(lon=3.0003, lat=45.0002, buffer_m=100), scale_m=10)
    for g in (a, b):
        assert g.transform.c % 10 == 0 and g.transform.f % 10 == 0


def test_resample_identity_and_fill():
    g = P.common_grid(PointBuffer(lon=3.0, lat=45.0, buffer_m=50), scale_m=10)
    src = np.random.default_rng(0).random((2, 3, g.height, g.width)).astype(np.float32)
    out = P.resample_to_common_grid(
        src, src_crs="EPSG:3857", src_transform=g.transform, grid=g, fill_value=-1.0
    )
    np.testing.assert_array_equal(out, src)

    # A source covering only the western half leaves the east filled.
    half = g.width // 2
    out = P.resample_to_common_grid(
        src[..., :half], src_crs="EPSG:3857", src_transform=g.transform, grid=g, fill_value=-1.0
    )
    np.testing.assert_array_equal(out[..., :half], src[..., :half])
    assert np.all(out[..., half:] == -1.0)


def test_resample_from_utm_keeps_orientation_and_values():
    crs = "EPSG:32631"
    lon, lat = 3.0, 45.0
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    n = 60
    rows = np.mgrid[0:n, 0:n][0].astype(np.float32)[None]
    src_transform = Affine(10, 0, x - n * 5, 0, -10, y + n * 5)
    g = P.common_grid(PointBuffer(lon=lon, lat=lat, buffer_m=100), scale_m=10)

    out = P.resample_to_common_grid(
        rows, src_crs=crs, src_transform=src_transform, grid=g, fill_value=np.nan
    )
    assert np.isfinite(out).all()
    col = out[0, :, g.width // 2]
    assert np.all(np.diff(col) >= 0) and col[-1] > col[0]  # north-up preserved

    fp = P.raster_bounds_4326(crs, src_transform, (n, n))
    assert fp.minlon < lon < fp.maxlon and fp.minlat < lat < fp.maxlat


def test_crs_label_accepts_strings_and_rasterio_objects():
    assert P.crs_label("epsg:32616") == "EPSG:32616"
    rasterio = pytest.importorskip("rasterio")
    assert P.crs_label(rasterio.crs.CRS.from_epsg(4326)) == "EPSG:4326"
    with pytest.raises(SpecError):
        P.crs_label("not-a-crs")


def test_utm_zone_helpers_and_boundary_warning():
    assert P.utm_zone_of_crs("EPSG:32616") == 16
    assert P.utm_zone_of_crs("EPSG:32716") == 16
    assert P.utm_zone_of_crs("EPSG:3857") is None
    assert P.utm_zones_spanned(BBox(minlon=-84.5, minlat=40, maxlon=-83.5, maxlat=41)) == {16, 17}
    assert P.utm_zones_spanned(BBox(minlon=-88.5, minlat=40, maxlon=-88.0, maxlat=41)) == {16}

    inside = BBox(minlon=-88.5, minlat=40, maxlon=-88.0, maxlat=41)
    across = BBox(minlon=-84.5, minlat=40, maxlon=-83.5, maxlat=41)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not P.warn_if_utm_boundary(crss=["EPSG:32616"], footprint=inside, context="t")
        assert not P.warn_if_utm_boundary(crss=["EPSG:3857"], footprint=across, context="t")
    with pytest.warns(UserWarning, match=r"zones \[16, 17\]"):
        assert P.warn_if_utm_boundary(crss=["EPSG:32616"], footprint=across, context="t")
    with pytest.warns(UserWarning, match="UTM zone boundary"):
        assert P.warn_if_utm_boundary(
            crss=["EPSG:32616", "EPSG:32617"], footprint=None, context="t"
        )

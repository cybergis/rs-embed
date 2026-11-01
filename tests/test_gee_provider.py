"""Tests for GEE provider helpers that don't require actual GEE auth.

Band alias resolution is pure logic — no network calls needed.
"""

import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rs_embed.core.errors import ProviderError
from rs_embed.core.specs import BBox, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.providers import gee_utils
from rs_embed.providers.gee import GEEProvider, _resolve_band_aliases
from rs_embed.providers.gee_utils import (
    _compute_pixels_chw,
    _fetch_with_bbox_fallback,
    _looks_like_gee_request_too_large,
)
from rs_embed.tools.projection import common_grid

# ══════════════════════════════════════════════════════════════════════
# Fake pixel download (ee.data.computePixels), shared by the fetch tests
# ══════════════════════════════════════════════════════════════════════

# Any real spatial works: the fake download ignores the requested grid.
_SPATIAL = PointBuffer(lon=0.0, lat=0.0, buffer_m=10.0)
_VV_VH = {"VV": [[1.0, 2.0], [3.0, 4.0]], "VH": [[5.0, 6.0], [7.0, 8.0]]}


class _FakePixels:
    """Fake ee image that computePixels can download: band name -> 2-D values.

    ``reproject``/``clip``/``unmask``/``toFloat`` pass through, ``select`` keeps
    the named bands in order, and ``on_download`` (if set) runs per download.
    """

    def __init__(self, bands, on_download=None):
        self.bands, self.on_download = dict(bands), on_download

    def select(self, names):
        kept = {n: self.bands[n] for n in names if n in self.bands}
        return _FakePixels(kept, self.on_download)

    def reproject(self, *_args, **_kwargs):
        return self

    def clip(self, _region):
        return self

    def unmask(self, *_args):
        return self

    def toFloat(self):
        return self

    def bandNames(self):
        return types.SimpleNamespace(getInfo=lambda: list(self.bands))


def _fake_compute_pixels(params):
    img = params["expression"]
    if img.on_download is not None:
        img.on_download()
    if not img.bands:
        return np.zeros((1, 1), dtype=np.dtype([]))
    planes = {n: np.asarray(v, dtype=np.float32) for n, v in img.bands.items()}
    shape = next(iter(planes.values())).shape
    out = np.zeros(shape, dtype=[(n, "f4") for n in planes])
    for n, v in planes.items():
        out[n] = v
    return out


_FAKE_EE_DATA = types.SimpleNamespace(computePixels=_fake_compute_pixels)

# ══════════════════════════════════════════════════════════════════════
# Sentinel-2 aliases
# ══════════════════════════════════════════════════════════════════════


def test_s2_rgb_aliases():
    result = _resolve_band_aliases("COPERNICUS/S2_SR_HARMONIZED", ("RED", "GREEN", "BLUE"))
    assert result == ("B4", "B3", "B2")


def test_s2_nir_aliases():
    result = _resolve_band_aliases("COPERNICUS/S2_SR_HARMONIZED", ("NIR", "NIR_NARROW"))
    assert result == ("B8", "B8A")


def test_s2_swir_aliases():
    result = _resolve_band_aliases("COPERNICUS/S2_SR_HARMONIZED", ("SWIR1", "SWIR2"))
    assert result == ("B11", "B12")


def test_s2_red_edge_aliases():
    result = _resolve_band_aliases("COPERNICUS/S2_SR_HARMONIZED", ("RE1", "RE2", "RE3", "RE4"))
    assert result == ("B5", "B6", "B7", "B8A")


def test_s2_passthrough_real_bands():
    result = _resolve_band_aliases("COPERNICUS/S2_SR_HARMONIZED", ("B4", "B3", "B2"))
    assert result == ("B4", "B3", "B2")


def test_s2_toa_also_resolves():
    result = _resolve_band_aliases("COPERNICUS/S2", ("RED", "GREEN", "BLUE"))
    assert result == ("B4", "B3", "B2")


# ══════════════════════════════════════════════════════════════════════
# Landsat 8/9 aliases
# ══════════════════════════════════════════════════════════════════════


def test_landsat89_rgb():
    result = _resolve_band_aliases("LANDSAT/LC08/C02/T1_L2", ("RED", "GREEN", "BLUE"))
    assert result == ("SR_B4", "SR_B3", "SR_B2")


def test_landsat89_nir_swir():
    result = _resolve_band_aliases("LANDSAT/LC09/C02/T1_L2", ("NIR", "SWIR1", "SWIR2"))
    assert result == ("SR_B5", "SR_B6", "SR_B7")


# ══════════════════════════════════════════════════════════════════════
# Landsat 4/5/7 aliases
# ══════════════════════════════════════════════════════════════════════


def test_landsat457_rgb():
    result = _resolve_band_aliases("LANDSAT/LE07/C02/T1_L2", ("RED", "GREEN", "BLUE"))
    assert result == ("SR_B3", "SR_B2", "SR_B1")


def test_landsat5_nir():
    result = _resolve_band_aliases("LANDSAT/LT05/C02/T1_L2", ("NIR",))
    assert result == ("SR_B4",)


# ══════════════════════════════════════════════════════════════════════
# Unknown collection — no aliasing
# ══════════════════════════════════════════════════════════════════════


def test_unknown_collection_passthrough():
    result = _resolve_band_aliases("SOME/OTHER/COLLECTION", ("RED", "GREEN", "BLUE"))
    # No mapping → returned as-is
    assert result == ("RED", "GREEN", "BLUE")


def test_empty_bands():
    result = _resolve_band_aliases("COPERNICUS/S2_SR_HARMONIZED", ())
    assert result == ()


# ══════════════════════════════════════════════════════════════════════
# Case insensitivity of aliases
# ══════════════════════════════════════════════════════════════════════


def test_alias_case_insensitive():
    result = _resolve_band_aliases("COPERNICUS/S2_SR_HARMONIZED", ("red", "green", "blue"))
    assert result == ("B4", "B3", "B2")


def test_mixed_alias_and_real():
    result = _resolve_band_aliases("COPERNICUS/S2_SR_HARMONIZED", ("RED", "B3", "BLUE"))
    assert result == ("B4", "B3", "B2")


def test_ensure_ready_allows_default_project_resolution(monkeypatch):
    calls = []

    class _FakeEE:
        @staticmethod
        def Initialize(**kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "ee", _FakeEE())

    provider = GEEProvider(auto_auth=False, project=None)
    provider.ensure_ready()

    assert calls == [{}]


def test_ensure_ready_passes_explicit_project(monkeypatch):
    calls = []

    class _FakeEE:
        @staticmethod
        def Initialize(**kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "ee", _FakeEE())

    provider = GEEProvider(auto_auth=False, project="demo-project")
    provider.ensure_ready()

    assert calls == [{"project": "demo-project"}]


def test_ensure_ready_missing_project_reports_setup_guidance(monkeypatch):
    class _FakeEE:
        @staticmethod
        def Initialize(**kwargs):  # noqa: ARG004
            raise Exception("no project found. Call with a quota project.")

    monkeypatch.setitem(sys.modules, "ee", _FakeEE())

    provider = GEEProvider(auto_auth=False, project=None)

    with pytest.raises(ProviderError, match="Earth Engine requires a Google Cloud project"):
        provider.ensure_ready()


def test_ensure_ready_auto_auth_fallback_omits_project_when_unset(monkeypatch):
    ee_calls = []
    geemap_calls = []

    class _FakeEE:
        @staticmethod
        def Initialize(**kwargs):
            ee_calls.append(kwargs)
            raise Exception("credentials missing")

    class _FakeGeemap:
        @staticmethod
        def ee_initialize(**kwargs):
            geemap_calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "ee", _FakeEE())
    monkeypatch.setitem(sys.modules, "geemap", _FakeGeemap())

    provider = GEEProvider(auto_auth=True, project=None)
    provider.ensure_ready()

    assert ee_calls == [{}]
    assert geemap_calls == [{}]


def test_build_image_empty_collection_raises_clear_error(monkeypatch):
    class _FakeSize:
        def getInfo(self):
            return 0

    class _FakeCollection:
        def filterBounds(self, _region):
            return self

        def filterDate(self, _start, _end):
            return self

        def size(self):
            return _FakeSize()

    fake_ee = types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Image=lambda _collection: (_ for _ in ()).throw(
            AssertionError("fallback should not be used")
        ),
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    sensor = SensorSpec(collection="COPERNICUS/S2_SR_HARMONIZED", bands=("B4",), cloudy_pct=None)
    temporal = TemporalSpec.range("2024-01-01", "2024-02-01")

    with pytest.raises(ProviderError, match="No images found"):
        provider.build_image(sensor=sensor, temporal=temporal, region=object())


def test_build_image_sorts_before_mosaic(monkeypatch):
    calls = []

    class _FakeSize:
        def getInfo(self):
            return 2

    class _FakeCollection:
        def filterBounds(self, _region):
            calls.append(("filterBounds",))
            return self

        def filterDate(self, _start, _end):
            calls.append(("filterDate",))
            return self

        def sort(self, key):
            calls.append(("sort", key))
            return self

        def size(self):
            return _FakeSize()

        def mosaic(self):
            calls.append(("mosaic",))
            return "mosaic-image"

    fake_ee = types.SimpleNamespace(ImageCollection=lambda _collection: _FakeCollection())
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    sensor = SensorSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4",),
        cloudy_pct=None,
        composite="mosaic",
    )
    temporal = TemporalSpec.range("2024-01-01", "2024-02-01")

    img = provider.build_image(sensor=sensor, temporal=temporal, region=object())

    assert img == "mosaic-image"
    assert ("sort", "system:time_start") in calls
    assert calls.index(("sort", "system:time_start")) < calls.index(("mosaic",))


def test_fetch_array_chw_empty_download_raises_clear_error(monkeypatch):
    class _FakeProjection:
        def atScale(self, _scale):
            return self

    fake_ee = types.SimpleNamespace(
        Projection=lambda *_args, **_kwargs: _FakeProjection(), data=_FAKE_EE_DATA
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())
    with pytest.raises(ProviderError, match="No images found"):
        provider.fetch_array_chw(
            image=_FakePixels({}),
            bands=("RED",),
            spatial=_SPATIAL,
            scale_m=10,
            fill_value=0.0,
            collection="COPERNICUS/S2_SR_HARMONIZED",
        )


def test_fetch_s1_vvvh_raw_chw_empty_collection_reports_filter_counts(monkeypatch):
    class _FakeSize:
        def __init__(self, n):
            self._n = n

        def getInfo(self):
            return self._n

    class _FakeFilter:
        @staticmethod
        def eq(field, value):
            return ("eq", field, value)

        @staticmethod
        def listContains(field, value):
            return ("listContains", field, value)

    class _FakeCollection:
        def __init__(self, count=5, stage="base"):
            self.count = count
            self.stage = stage

        def filterDate(self, _start, _end):
            return self

        def filterBounds(self, _region):
            return self

        def filter(self, filt):
            kind, field, value = filt
            if kind == "eq" and field == "instrumentMode" and value == "IW":
                return _FakeCollection(2, "iw")
            if (
                kind == "listContains"
                and field == "transmitterReceiverPolarisation"
                and value == "VV"
                and self.stage == "base"
            ):
                return _FakeCollection(1, "vv_no_iw")
            if (
                kind == "listContains"
                and field == "transmitterReceiverPolarisation"
                and value == "VH"
                and self.stage == "vv_no_iw"
            ):
                return _FakeCollection(0, "vh_no_iw")
            if (
                kind == "listContains"
                and field == "transmitterReceiverPolarisation"
                and value == "VV"
                and self.stage == "iw"
            ):
                return _FakeCollection(1, "vv")
            if (
                kind == "listContains"
                and field == "transmitterReceiverPolarisation"
                and value == "VH"
                and self.stage == "vv"
            ):
                return _FakeCollection(0, "vh")
            if kind == "eq" and field == "orbitProperties_pass":
                return _FakeCollection(0, "orbit")
            return self

        def size(self):
            return _FakeSize(self.count)

    fake_ee = types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Filter=_FakeFilter,
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())
    temporal = TemporalSpec.range("2024-01-01", "2024-02-01")

    with pytest.raises(ProviderError, match=r"base\(date\+bounds\)=5, iw=2, vv=1, vh=0"):
        provider.fetch_s1_vvvh_raw_chw(
            spatial=object(),
            temporal=temporal,
            orbit=None,
        )


def _make_s1_fake_ee(seen_filters: list):
    """Fake ee module for S1 fetch tests: records filters, returns data."""

    class _FakeSize:
        def getInfo(self):
            return 3

    class _FakeFilter:
        @staticmethod
        def eq(field, value):
            return ("eq", field, value)

        @staticmethod
        def listContains(field, value):
            return ("listContains", field, value)

    class _FakeCollection:
        def filterDate(self, _start, _end):
            return self

        def filterBounds(self, _region):
            return self

        def filter(self, filt):
            seen_filters.append(filt)
            return self

        def size(self):
            return _FakeSize()

        def median(self):
            return _FakePixels(_VV_VH)

    return types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Filter=_FakeFilter,
        data=_FAKE_EE_DATA,
    )


def test_fetch_s1_vvvh_raw_chw_applies_orbit_filter(monkeypatch):
    """An explicit orbit request must filter orbitProperties_pass and be
    recorded as applied in meta.

    Regression: orbit was accepted, stamped into meta as requested, and then
    silently ignored — an ASCENDING request produced a mixed ASC/DESC median.
    """
    seen_filters: list = []
    monkeypatch.setitem(sys.modules, "ee", _make_s1_fake_ee(seen_filters))

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    arr, meta = provider.fetch_s1_vvvh_raw_chw_with_meta(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
        orbit="ascending",
    )

    assert arr.shape == (2, 2, 2)
    assert ("eq", "orbitProperties_pass", "ASCENDING") in seen_filters
    assert meta["s1_orbit_requested"] == "ascending"
    assert meta["s1_orbit_applied"] == "ASCENDING"


def test_fetch_s1_vvvh_raw_chw_no_orbit_stays_unfiltered(monkeypatch):
    """The default orbit=None keeps the loosened mixed-pass behavior."""
    seen_filters: list = []
    monkeypatch.setitem(sys.modules, "ee", _make_s1_fake_ee(seen_filters))

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    _arr, meta = provider.fetch_s1_vvvh_raw_chw_with_meta(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
        orbit=None,
    )

    assert not any(f[1] == "orbitProperties_pass" for f in seen_filters)
    assert meta["s1_orbit_applied"] is None


def test_fetch_s1_vvvh_raw_chw_rejects_unknown_orbit(monkeypatch):
    seen_filters: list = []
    monkeypatch.setitem(sys.modules, "ee", _make_s1_fake_ee(seen_filters))

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    with pytest.raises(ProviderError, match="Unknown S1 orbit"):
        provider.fetch_s1_vvvh_raw_chw(
            spatial=object(),
            temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
            orbit="sideways",
        )


def test_fetch_s1_vvvh_raw_chw_mosaic_sorts_before_reduce(monkeypatch):
    calls = []

    class _FakeSize:
        def __init__(self, n):
            self._n = n

        def getInfo(self):
            return self._n

    class _FakeFilter:
        @staticmethod
        def eq(field, value):
            return ("eq", field, value)

        @staticmethod
        def listContains(field, value):
            return ("listContains", field, value)

    class _FakeCollection:
        def __init__(self, count=5, stage="base"):
            self.count = count
            self.stage = stage

        def filterDate(self, _start, _end):
            return self

        def filterBounds(self, _region):
            return self

        def filter(self, filt):
            kind, field, value = filt
            if kind == "eq" and field == "instrumentMode" and value == "IW":
                return _FakeCollection(2, "iw")
            if (
                kind == "listContains"
                and field == "transmitterReceiverPolarisation"
                and value == "VV"
                and self.stage == "iw"
            ):
                return _FakeCollection(1, "vv")
            if (
                kind == "listContains"
                and field == "transmitterReceiverPolarisation"
                and value == "VH"
                and self.stage == "vv"
            ):
                return _FakeCollection(1, "vh")
            return self

        def size(self):
            return _FakeSize(self.count)

        def sort(self, key):
            calls.append(("sort", key, self.stage))
            return self

        def mosaic(self):
            calls.append(("mosaic", self.stage))
            return _FakePixels(_VV_VH)

    fake_ee = types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Filter=_FakeFilter,
        data=_FAKE_EE_DATA,
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    arr = provider.fetch_s1_vvvh_raw_chw(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
        composite="mosaic",
    )

    assert arr.shape == (2, 2, 2)
    assert ("sort", "system:time_start", "vh") in calls
    assert calls.index(("sort", "system:time_start", "vh")) < calls.index(("mosaic", "vh"))


def test_fetch_s1_vvvh_raw_chw_relaxes_iw_and_reports_meta(monkeypatch):
    class _FakeSize:
        def __init__(self, n):
            self._n = n

        def getInfo(self):
            return self._n

    class _FakeFilter:
        @staticmethod
        def eq(field, value):
            return ("eq", field, value)

        @staticmethod
        def listContains(field, value):
            return ("listContains", field, value)

    class _FakeCollection:
        def __init__(self, count=5, stage="base"):
            self.count = count
            self.stage = stage

        def filterDate(self, _start, _end):
            return self

        def filterBounds(self, _region):
            return self

        def filter(self, filt):
            kind, field, value = filt
            if kind == "eq" and field == "instrumentMode" and value == "IW":
                return _FakeCollection(0, "iw")
            if (
                kind == "listContains"
                and field == "transmitterReceiverPolarisation"
                and value == "VV"
                and self.stage == "base"
            ):
                return _FakeCollection(1, "vv_no_iw")
            if (
                kind == "listContains"
                and field == "transmitterReceiverPolarisation"
                and value == "VH"
                and self.stage == "vv_no_iw"
            ):
                return _FakeCollection(1, "vh_no_iw")
            return self

        def size(self):
            return _FakeSize(self.count)

        def median(self):
            return _FakePixels(_VV_VH)

    fake_ee = types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Filter=_FakeFilter,
        data=_FAKE_EE_DATA,
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    arr, meta = provider.fetch_s1_vvvh_raw_chw_with_meta(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
        require_iw=True,
        relax_iw_on_empty=True,
    )

    assert arr.shape == (2, 2, 2)
    assert meta["s1_iw_requested"] is True
    assert meta["s1_iw_applied"] is False
    assert meta["s1_iw_relaxed_on_empty"] is True


# ══════════════════════════════════════════════════════════════════════
# _fetch_all_bands_impl — orientation contract
# ══════════════════════════════════════════════════════════════════════


def test_fetch_all_bands_impl_passes_through_downloaded_row_order(monkeypatch):
    """Orientation contract test for _fetch_all_bands_impl.

    The function applies **no** spatial flip: computePixels is asked for the
    common grid, whose affine transform puts row 0 on the north edge, so the
    downloaded rows are already north-up. This test locks in the pass-through
    behaviour so that any accidental flip addition would be caught immediately.

    The mock injects two distinguishable rows:
      row 0 → [1, 2]   (northernmost row)
      row 1 → [3, 4]   (southernmost row)
    """
    import numpy as np

    from rs_embed.core.specs import TemporalSpec

    # ── GEE pixel data: row 0 = [1, 2], row 1 = [3, 4] ────────────────
    _NORTH_ROW = [1.0, 2.0]
    _SOUTH_ROW = [3.0, 4.0]

    class _FakeCollection:
        def filterBounds(self, _r):
            return self

        def filterDate(self, _s, _e):
            return self

        def size(self):
            class _Size:
                def getInfo(self):
                    return 1

            return _Size()

        def median(self):
            return _FakePixels({"B1": [_NORTH_ROW, _SOUTH_ROW]})

    fake_ee = types.SimpleNamespace(
        ImageCollection=lambda _c: _FakeCollection(), data=_FAKE_EE_DATA
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    arr, names = provider._fetch_all_bands_impl(
        spatial=_SPATIAL,
        temporal=TemporalSpec.year(2024),
        collection="FAKE/COLL",
        scale_m=10,
        fill_value=0.0,
    )

    assert names == ("B1",)
    assert arr.shape == (1, 2, 2)
    # Row order is preserved as downloaded (north-up) — no flip applied.
    np.testing.assert_allclose(arr[0, 0], _NORTH_ROW)
    np.testing.assert_allclose(arr[0, 1], _SOUTH_ROW)


# ══════════════════════════════════════════════════════════════════════
# _compute_pixels_chw — the binary download every GEE fetch goes through
# ══════════════════════════════════════════════════════════════════════

_FIVE_BANDS = {f"B{i}": [[float(i), i + 0.5], [i + 0.25, i + 0.75]] for i in range(5)}
_SMALL_BBOX = BBox(minlon=-88.25, minlat=40.09, maxlon=-88.24, maxlat=40.10)


class _RecordingPixels(_FakePixels):
    def __init__(self, bands, log):
        super().__init__(bands)
        self.log = log

    def unmask(self, *args):
        self.log.append(("unmask", args))
        return self


def _install_compute_pixels(monkeypatch, requests, refuse=lambda _bands: False):
    """Fake ee whose computePixels records each request's bands and can refuse some."""

    def _compute(params):
        bands = list(params["expression"].bands)
        requests.append({**params, "bands": bands})
        if refuse(bands):
            raise RuntimeError(
                "Total request size (99 bytes) must be less than or equal to 50331648 bytes."
            )
        return _fake_compute_pixels(params)

    fake_ee = types.SimpleNamespace(data=types.SimpleNamespace(computePixels=_compute))
    monkeypatch.setitem(sys.modules, "ee", fake_ee)


def _band_bytes(spatial, scale_m=10):
    grid = common_grid(spatial, scale_m=float(scale_m))
    return grid.height * grid.width * gee_utils._GEE_REQUEST_BYTES_PER_VALUE


def test_compute_pixels_requests_the_common_grid_north_up(monkeypatch):
    requests: list = []
    _install_compute_pixels(monkeypatch, requests)
    log: list = []

    arr = _compute_pixels_chw(
        _RecordingPixels(_FIVE_BANDS, log),
        spatial=_SMALL_BBOX,
        bands=("B3", "B1"),
        scale_m=10,
        fill_value=-9999.0,
    )

    grid = common_grid(_SMALL_BBOX, scale_m=10.0)
    t = grid.transform
    (req,) = requests
    assert req["fileFormat"] == "NUMPY_NDARRAY"
    assert req["grid"]["crsCode"] == "EPSG:3857"
    assert req["grid"]["dimensions"] == {"width": grid.width, "height": grid.height}
    assert req["grid"]["affineTransform"] == {
        "scaleX": t.a,
        "shearX": t.b,
        "translateX": t.c,
        "shearY": t.d,
        "scaleY": t.e,
        "translateY": t.f,
    }
    assert t.e < 0  # row 0 is the north edge: downloads arrive north-up
    # masked pixels, inside or outside the image footprint, become the fill
    assert ("unmask", (-9999.0, False)) in log
    expected = np.stack([_FIVE_BANDS["B3"], _FIVE_BANDS["B1"]]).astype(np.float32)
    np.testing.assert_array_equal(arr, expected)


def test_point_buffer_bands_split_evenly_to_fit_the_request_limit(monkeypatch):
    # A PointBuffer cannot be split spatially, so its bands are: 4 bands fit per
    # request, and 5 go out as 3 + 2 rather than 4 + 1.
    requests: list = []
    _install_compute_pixels(monkeypatch, requests)
    monkeypatch.setattr(gee_utils, "_GEE_REQUEST_MAX_BYTES", 4 * _band_bytes(_SPATIAL))
    bands = tuple(_FIVE_BANDS)

    arr = _compute_pixels_chw(
        _FakePixels(_FIVE_BANDS), spatial=_SPATIAL, bands=bands, scale_m=10, fill_value=0.0
    )

    assert [r["bands"] for r in requests] == [["B0", "B1", "B2"], ["B3", "B4"]]
    np.testing.assert_array_equal(arr, np.stack([_FIVE_BANDS[b] for b in bands]))


def test_bbox_over_the_request_limit_raises_before_sending(monkeypatch):
    # A BBox is split spatially by the caller instead, keeping every band of a
    # piece in one request; the limit error says so before anything is sent.
    requests: list = []
    _install_compute_pixels(monkeypatch, requests)
    monkeypatch.setattr(gee_utils, "_GEE_REQUEST_MAX_BYTES", 4 * _band_bytes(_SMALL_BBOX))

    with pytest.raises(ProviderError) as err:
        _compute_pixels_chw(
            _FakePixels(_FIVE_BANDS),
            spatial=_SMALL_BBOX,
            bands=tuple(_FIVE_BANDS),
            scale_m=10,
            fill_value=0.0,
        )
    assert _looks_like_gee_request_too_large(err.value)
    assert requests == []


def test_grid_side_over_the_limit_raises_the_limit_error(monkeypatch):
    requests: list = []
    _install_compute_pixels(monkeypatch, requests)
    monkeypatch.setattr(gee_utils, "_GEE_REQUEST_MAX_GRID_SIDE", 1)

    with pytest.raises(ProviderError, match="Pixel grid dimensions") as err:
        _compute_pixels_chw(
            _FakePixels(_FIVE_BANDS), spatial=_SPATIAL, bands=("B0",), scale_m=10, fill_value=0.0
        )
    assert _looks_like_gee_request_too_large(err.value)  # BBox fetches split on it
    assert requests == []


def test_compute_pixels_halves_a_band_chunk_gee_refuses(monkeypatch):
    requests: list = []
    _install_compute_pixels(monkeypatch, requests, refuse=lambda bands: len(bands) > 2)
    bands = tuple(_FIVE_BANDS)

    arr = _compute_pixels_chw(
        _FakePixels(_FIVE_BANDS), spatial=_SMALL_BBOX, bands=bands, scale_m=10, fill_value=0.0
    )

    assert [r["bands"] for r in requests] == [
        ["B0", "B1", "B2", "B3", "B4"],
        ["B0", "B1"],
        ["B2", "B3", "B4"],
        ["B2"],
        ["B3", "B4"],
    ]
    np.testing.assert_array_equal(arr, np.stack([_FIVE_BANDS[b] for b in bands]))


def test_single_band_over_the_limit_raises_the_limit_error(monkeypatch):
    requests: list = []
    _install_compute_pixels(monkeypatch, requests)
    monkeypatch.setattr(gee_utils, "_GEE_REQUEST_MAX_BYTES", _band_bytes(_SPATIAL) - 1)

    def _fetch(sp):
        return _compute_pixels_chw(
            _FakePixels(_FIVE_BANDS), spatial=sp, bands=("B0",), scale_m=10, fill_value=0.0
        )

    # A PointBuffer is never split, so the limit error reaches the caller.
    with pytest.raises(ProviderError) as err:
        _fetch_with_bbox_fallback(spatial=_SPATIAL, scale_m=10, fetch_fn=_fetch)
    assert _looks_like_gee_request_too_large(err.value)  # BBox fetches split on it
    assert requests == []  # refused before anything is sent


def test_fetch_array_chw_does_not_flip_downloaded_rows(monkeypatch):
    class _FakeProjection:
        def atScale(self, _scale):
            return self

    fake_ee = types.SimpleNamespace(
        Projection=lambda *_args, **_kwargs: _FakeProjection(), data=_FAKE_EE_DATA
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    arr = provider.fetch_array_chw(
        image=_FakePixels({"B4": [[1.0, 2.0], [3.0, 4.0]]}),
        bands=("RED",),
        spatial=_SPATIAL,
        scale_m=10,
        fill_value=0.0,
        collection="COPERNICUS/S2_SR_HARMONIZED",
    )

    np.testing.assert_array_equal(arr, np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))


def test_cloud_property_for_collection_mapping():
    from rs_embed.providers.gee_utils import _cloud_property_for_collection

    assert (
        _cloud_property_for_collection("COPERNICUS/S2_SR_HARMONIZED") == "CLOUDY_PIXEL_PERCENTAGE"
    )
    assert _cloud_property_for_collection("LANDSAT/LC08/C02/T1_L2") == "CLOUD_COVER"
    assert _cloud_property_for_collection("LANDSAT/LE07/C02/T1_L2") == "CLOUD_COVER"
    assert _cloud_property_for_collection("COPERNICUS/S1_GRD") is None
    assert _cloud_property_for_collection("USGS/SRTMGL1_003") is None
    assert _cloud_property_for_collection("") is None


def test_filter_clouds_uses_collection_property_and_skips_unknown(monkeypatch):
    """Cloud filtering must use the collection's own property (inclusive lte)
    and must not empty collections that lack a cloud-cover property.

    Regression: the S2-only CLOUDY_PIXEL_PERCENTAGE filter was applied
    unconditionally with the default cloudy_pct=30; GEE property filters
    exclude images lacking the property, so every Landsat/S1/DEM request
    dropped all images and failed with a misleading 'No images found'.
    """
    import sys
    import types

    from rs_embed.providers.gee_utils import _filter_clouds

    seen_filters = []

    class _FakeCollection:
        def filter(self, f):
            seen_filters.append(f)
            return self

    fake_ee = types.SimpleNamespace(
        Filter=types.SimpleNamespace(lte=lambda prop, val: ("lte", prop, val))
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    col = _FakeCollection()

    # S2 -> its own property, inclusive threshold.
    out = _filter_clouds(col, collection="COPERNICUS/S2_SR_HARMONIZED", cloudy_pct=30)
    assert out is col
    assert seen_filters == [("lte", "CLOUDY_PIXEL_PERCENTAGE", 30)]

    # Landsat -> CLOUD_COVER, not the S2 property.
    seen_filters.clear()
    _filter_clouds(col, collection="LANDSAT/LC08/C02/T1_L2", cloudy_pct=20)
    assert seen_filters == [("lte", "CLOUD_COVER", 20)]

    # No cloud property (SAR/DEM) -> unfiltered.
    seen_filters.clear()
    out = _filter_clouds(col, collection="COPERNICUS/S1_GRD", cloudy_pct=30)
    assert out is col
    assert seen_filters == []

    # cloudy_pct=None -> unfiltered.
    _filter_clouds(col, collection="COPERNICUS/S2_SR_HARMONIZED", cloudy_pct=None)
    assert seen_filters == []


def test_build_image_default_cloudy_pct_does_not_empty_landsat(monkeypatch):
    """A default-constructed SensorSpec (cloudy_pct=30) for Landsat must
    filter on CLOUD_COVER, not filter out every image via the S2 property."""
    import sys
    import types

    seen_filters = []

    class _FakeSize:
        def getInfo(self):
            return 3

    class _FakeCollection:
        def filterBounds(self, _region):
            return self

        def filterDate(self, _start, _end):
            return self

        def filter(self, f):
            seen_filters.append(f)
            return self

        def size(self):
            return _FakeSize()

        def median(self):
            return "median_image"

    fake_ee = types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Filter=types.SimpleNamespace(lte=lambda prop, val: ("lte", prop, val)),
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    sensor = SensorSpec(collection="LANDSAT/LC08/C02/T1_L2", bands=("SR_B4",))
    temporal = TemporalSpec.range("2024-01-01", "2024-02-01")

    img = provider.build_image(sensor=sensor, temporal=temporal, region=object())
    assert img == "median_image"
    assert seen_filters == [("lte", "CLOUD_COVER", 30)]


def _make_multiframe_fake_ee(
    *,
    bin_values: list[float | None],
    fallback_value: float | None = 1.0,
    bin_size_raises: bool = False,
    bin_delays: list[float] | None = None,
):
    """Fake ee for multiframe fetch: one value per temporal bin (None = empty).

    ``fallback_value=None`` makes sampling the whole-window fallback composite
    fail; ``bin_size_raises`` makes the per-bin image count query raise;
    ``bin_delays`` sleeps that long while sampling bin i (to reorder completion).
    ``.sampled`` records each sampled value (the fallback included).
    """
    sampled: list[float | None] = []

    class _FakeSize:
        def __init__(self, n=1, raises=False):
            self._n, self._raises = n, raises

        def getInfo(self):
            if self._raises:
                raise RuntimeError("transient getInfo failure")
            return self._n

    class _FakeList:
        def __init__(self, items):
            self._items = items

        def getInfo(self):
            return [item.getInfo() for item in self._items]

    def _pixels(value, delay=0.0):
        def _on_download():
            time.sleep(delay)
            sampled.append(value)
            if value is None:
                raise RuntimeError("whole-window composite failed")

        return _FakePixels({"B1": [[value]]}, on_download=_on_download)

    class _FakeBin:
        def __init__(self, value, delay):
            self.value, self.delay = value, delay

        def size(self):
            if bin_size_raises:
                return _FakeSize(raises=True)
            return _FakeSize(1 if self.value is not None else 0)

        def median(self):
            return _pixels(self.value, self.delay)

    class _FakeCollection:
        def __init__(self):
            self.bin_idx = 0

        def filterDate(self, _start, _end):
            # First filterDate builds col_all; later ones select bins in order.
            if not hasattr(self, "_dated"):
                self._dated = True
                return self
            i = self.bin_idx
            self.bin_idx += 1
            return _FakeBin(bin_values[i], bin_delays[i] if bin_delays else 0.0)

        def filterBounds(self, _region):
            return self

        def filter(self, _filt):
            return self

        def size(self):
            return _FakeSize(1)

        def median(self):
            return _pixels(fallback_value)

    return types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        List=_FakeList,
        data=_FAKE_EE_DATA,
        Filter=types.SimpleNamespace(
            lte=lambda f, v: ("lte", f, v),
            eq=lambda f, v: ("eq", f, v),
            listContains=lambda f, v: ("listContains", f, v),
        ),
        sampled=sampled,
    )


def _multiframe_provider(monkeypatch, fake_ee):
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())
    return provider


def test_multiframe_fetch_does_not_clip_non_s2_collections(monkeypatch):
    """The [0,10000] S2 DN clip must not destroy other collections' values.

    Regression: the clip was baked into the generic sampler, so Landsat SR
    (DN up to ~65455) and signed products were silently truncated.
    """
    provider = _multiframe_provider(
        monkeypatch,
        _make_multiframe_fake_ee(bin_values=[-5.0, 20000.0], fallback_value=1.0),
    )
    arr = provider.fetch_multiframe_collection_raw_tchw(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-03-01"),
        collection="FAKE/LANDSAT_LIKE",
        bands=("B1",),
        n_frames=2,
    )
    assert arr.shape == (2, 1, 1, 1)
    assert arr[0, 0, 0, 0] == -5.0
    assert arr[1, 0, 0, 0] == 20000.0


def test_multiframe_fetch_clips_s2_to_dn_range(monkeypatch):
    provider = _multiframe_provider(
        monkeypatch,
        _make_multiframe_fake_ee(bin_values=[-5.0, 20000.0], fallback_value=1.0),
    )
    arr = provider.fetch_multiframe_collection_raw_tchw(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-03-01"),
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B1",),
        n_frames=2,
    )
    assert arr[0, 0, 0, 0] == 0.0
    assert arr[1, 0, 0, 0] == 10000.0


def test_multiframe_fetch_bin_count_failure_raises(monkeypatch):
    """A failed per-bin image count is an error, not an empty bin.

    Regression: a transient getInfo failure was treated as 'no data' and the
    whole-window composite was silently substituted for a bin that had data.
    """
    provider = _multiframe_provider(
        monkeypatch,
        _make_multiframe_fake_ee(bin_values=[2.0, 3.0], bin_size_raises=True),
    )
    with pytest.raises(ProviderError, match="Failed to query image count"):
        provider.fetch_multiframe_collection_raw_tchw(
            spatial=_SPATIAL,
            temporal=TemporalSpec.range("2024-01-01", "2024-03-01"),
            collection="FAKE/COLL",
            bands=("B1",),
            n_frames=2,
        )


def test_multiframe_fetch_empty_bin_keeps_alignment_without_fallback(monkeypatch):
    """An empty bin with no whole-window fallback must not shift later frames.

    Regression: the empty bin was skipped, so frame i no longer corresponded
    to bin i (prithvi's midpoint-date logic relies on that alignment) and the
    tail was padded with duplicates.
    """
    provider = _multiframe_provider(
        monkeypatch,
        _make_multiframe_fake_ee(bin_values=[2.0, None, 4.0], fallback_value=None),
    )
    arr = provider.fetch_multiframe_collection_raw_tchw(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-04-01"),
        collection="FAKE/COLL",
        bands=("B1",),
        n_frames=3,
    )
    vals = arr[:, 0, 0, 0].tolist()
    # bin 2's frame stays at index 2; the empty bin 1 back-fills from the
    # nearest non-empty bin instead of shifting everything left.
    assert vals[0] == 2.0
    assert vals[2] == 4.0
    assert vals[1] in (2.0, 4.0)


def test_multiframe_fetch_concurrent_bins_keep_bin_order(monkeypatch):
    """Bins are sampled concurrently; frame i must still be bin i.

    Later bins finish first here, so a completion-ordered result would reverse them.
    """
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    provider = _multiframe_provider(
        monkeypatch,
        _make_multiframe_fake_ee(
            bin_values=values, bin_delays=[0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        ),
    )
    arr = provider.fetch_multiframe_collection_raw_tchw(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-07-01"),
        collection="FAKE/COLL",
        bands=("B1",),
        n_frames=6,
    )
    assert arr[:, 0, 0, 0].tolist() == values


def test_multiframe_fetch_samples_fallback_only_for_empty_bins(monkeypatch):
    """The whole-window composite only stands in for empty bins.

    With every bin filled it is never sampled; with an empty bin it is sampled
    once and fills that bin.
    """
    full = _make_multiframe_fake_ee(bin_values=[2.0, 3.0], fallback_value=9.0)
    _multiframe_provider(monkeypatch, full).fetch_multiframe_collection_raw_tchw(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-03-01"),
        collection="FAKE/COLL",
        bands=("B1",),
        n_frames=2,
    )
    assert sorted(full.sampled) == [2.0, 3.0]

    gap = _make_multiframe_fake_ee(bin_values=[2.0, None], fallback_value=9.0)
    arr = _multiframe_provider(monkeypatch, gap).fetch_multiframe_collection_raw_tchw(
        spatial=_SPATIAL,
        temporal=TemporalSpec.range("2024-01-01", "2024-03-01"),
        collection="FAKE/COLL",
        bands=("B1",),
        n_frames=2,
    )
    assert sorted(gap.sampled) == [2.0, 9.0]
    assert arr[:, 0, 0, 0].tolist() == [2.0, 9.0]


def test_ensure_ready_initializes_once_per_ee_module_and_project(monkeypatch):
    """ee.Initialize is not idempotent, so it must run once — also under threads.

    A different project, or a replaced ee module, initializes again.
    """
    calls = []

    class _FakeEE:
        @staticmethod
        def Initialize(**kwargs):
            time.sleep(0.01)
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "ee", _FakeEE())
    provider = GEEProvider(auto_auth=False, project="p1")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: provider.ensure_ready(), range(16)))
    provider.ensure_ready()
    GEEProvider(auto_auth=False, project="p1").ensure_ready()
    assert calls == [{"project": "p1"}]

    GEEProvider(auto_auth=False, project="p2").ensure_ready()
    assert calls[-1] == {"project": "p2"}

    monkeypatch.setitem(sys.modules, "ee", _FakeEE())
    GEEProvider(auto_auth=False, project="p2").ensure_ready()
    assert len(calls) == 3


def test_ensure_ready_initializes_again_after_ee_reset(monkeypatch):
    """ee.Reset() ends the session behind rs-embed's back; ask ee whether it is live."""
    calls = []
    live = {"on": False}

    def _initialize(**kwargs):
        calls.append(kwargs)
        live["on"] = True

    fake_ee = types.SimpleNamespace(
        Initialize=_initialize, data=types.SimpleNamespace(is_initialized=lambda: live["on"])
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    provider = GEEProvider(auto_auth=False, project="p1")

    provider.ensure_ready()
    provider.ensure_ready()
    assert len(calls) == 1
    live["on"] = False  # ee.Reset()
    provider.ensure_ready()
    assert len(calls) == 2


def test_normalize_s1_vvvh_rejects_db_scale_input():
    """dB-scale S1 input must fail loudly, not normalize to constant zero.

    Regression: use_float_linear=False fetches return negative dB values;
    max(x,0) zeroed them all and the embedder ran on a blank image.
    """
    import numpy as np

    from rs_embed.core.errors import ModelError
    from rs_embed.providers.fetch import normalize_s1_vvvh_chw

    db = np.random.uniform(-25.0, -5.0, size=(2, 4, 4)).astype(np.float32)
    with pytest.raises(ModelError, match="dB-scaled"):
        normalize_s1_vvvh_chw(db)

    # Linear-scale values (small non-negative floats) still normalize.
    linear = np.random.uniform(0.0, 0.5, size=(2, 4, 4)).astype(np.float32)
    out = normalize_s1_vvvh_chw(linear)
    assert out.shape == (2, 4, 4)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ── provider registry diagnostics ───────────────────────────────────────────


def test_get_provider_unknown_reports_recorded_import_error(monkeypatch):
    """A provider whose import failed must surface the cause instead of
    silently looking unregistered (M11)."""
    import rs_embed.providers as providers

    monkeypatch.setitem(
        providers._PROVIDER_IMPORT_ERRORS, "fakeprov", ImportError("no optional dep")
    )
    with pytest.raises(ValueError, match="no optional dep"):
        providers.get_provider("fakeprov")


def test_get_provider_unknown_mentions_other_import_errors(monkeypatch):
    """Even for a name with no recorded failure, known import errors are listed."""
    import rs_embed.providers as providers

    monkeypatch.setitem(providers._PROVIDER_IMPORT_ERRORS, "otherprov", ImportError("dep missing"))
    with pytest.raises(ValueError, match="otherprov: ImportError: dep missing"):
        providers.get_provider("definitely-not-registered")

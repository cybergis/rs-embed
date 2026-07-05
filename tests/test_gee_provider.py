"""Tests for GEE provider helpers that don't require actual GEE auth.

Band alias resolution is pure logic — no network calls needed.
"""

import sys
import types

import pytest

from rs_embed.core.errors import ProviderError
from rs_embed.core.specs import SensorSpec, TemporalSpec
from rs_embed.providers.gee import GEEProvider, _resolve_band_aliases

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


def test_fetch_array_chw_empty_sample_props_raises_clear_error(monkeypatch):
    class _FakeProjection:
        def atScale(self, _scale):
            return self

    class _FakeRect:
        def getInfo(self):
            return {"properties": {}}

    class _FakeImage:
        def select(self, _bands):
            return self

        def reproject(self, _proj):
            return self

        def clip(self, _region):
            return self

        def sampleRectangle(self, *, region, defaultValue):  # noqa: ARG002
            return _FakeRect()

    fake_ee = types.SimpleNamespace(Projection=lambda *_args, **_kwargs: _FakeProjection())
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    with pytest.raises(ProviderError, match="No images found"):
        provider.fetch_array_chw(
            image=_FakeImage(),
            bands=("RED",),
            region=object(),
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

    class _FakeRect:
        def getInfo(self):
            return {
                "properties": {
                    "VV": [[1.0, 2.0], [3.0, 4.0]],
                    "VH": [[5.0, 6.0], [7.0, 8.0]],
                }
            }

    class _FakeImage:
        def select(self, _bands):
            return self

        def reproject(self, **_kwargs):
            return self

        def sampleRectangle(self, *, region, defaultValue):  # noqa: ARG002
            return _FakeRect()

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
            return _FakeImage()

    return types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Filter=_FakeFilter,
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
        spatial=object(),
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
        spatial=object(),
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

    class _FakeRect:
        def getInfo(self):
            return {
                "properties": {
                    "VV": [[1.0, 2.0], [3.0, 4.0]],
                    "VH": [[5.0, 6.0], [7.0, 8.0]],
                }
            }

    class _FakeImage:
        def select(self, _bands):
            return self

        def reproject(self, **_kwargs):
            return self

        def sampleRectangle(self, *, region, defaultValue):  # noqa: ARG002
            return _FakeRect()

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
            return _FakeImage()

    fake_ee = types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Filter=_FakeFilter,
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    arr = provider.fetch_s1_vvvh_raw_chw(
        spatial=object(),
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

    class _FakeRect:
        def getInfo(self):
            return {
                "properties": {
                    "VV": [[1.0, 2.0], [3.0, 4.0]],
                    "VH": [[5.0, 6.0], [7.0, 8.0]],
                }
            }

    class _FakeImage:
        def select(self, _bands):
            return self

        def reproject(self, **_kwargs):
            return self

        def sampleRectangle(self, *, region, defaultValue):  # noqa: ARG002
            return _FakeRect()

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
            return _FakeImage()

    fake_ee = types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Filter=_FakeFilter,
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    arr, meta = provider.fetch_s1_vvvh_raw_chw_with_meta(
        spatial=object(),
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


def test_fetch_all_bands_impl_passes_through_gee_row_order(monkeypatch):
    """Orientation contract test for _fetch_all_bands_impl.

    The function applies **no** spatial flip — it passes through whatever row
    order GEE's ``sampleRectangle`` returns.  For the
    ``reproject(crs=..., scale=...) + .clip()`` call pattern used here, GEE
    empirically returns north-up rows (row 0 = northernmost).  This test locks
    in the pass-through behaviour so that any accidental flip addition would be
    caught immediately.

    The mock injects two distinguishable rows:
      row 0 → [1, 2]   (northernmost row, per the empirical GEE convention)
      row 1 → [3, 4]   (southernmost row)

    We assert the output preserves this exact order.  If GEE is ever found to
    return south-up for this pattern, add ``_flip_sample_tile_y`` to
    ``_fetch_all_bands_impl`` **and** update this test to expect the flipped
    order.
    """
    import numpy as np

    from rs_embed.core.specs import BBox, TemporalSpec

    # ── GEE pixel data: row 0 = [1, 2], row 1 = [3, 4] ────────────────
    _NORTH_ROW = [1.0, 2.0]
    _SOUTH_ROW = [3.0, 4.0]

    class _FakeImage:
        def reproject(self, **_kw):
            return self

        def clip(self, _region):
            return self

        def bandNames(self):
            class _Names:
                def getInfo(self):
                    return ["B1"]

            return _Names()

        def sampleRectangle(self, **_kw):
            class _Rect:
                def getInfo(self):
                    return {"properties": {"B1": [_NORTH_ROW, _SOUTH_ROW]}}

            return _Rect()

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
            return _FakeImage()

    fake_ee = types.SimpleNamespace(ImageCollection=lambda _c: _FakeCollection())
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    provider = GEEProvider(auto_auth=False)
    monkeypatch.setattr(provider, "ensure_ready", lambda: None)
    monkeypatch.setattr(provider, "get_region", lambda _spatial: object())

    arr, names = provider._fetch_all_bands_impl(
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=TemporalSpec.year(2024),
        collection="FAKE/COLL",
        scale_m=10,
        fill_value=0.0,
    )

    assert names == ("B1",)
    assert arr.shape == (1, 2, 2)
    # Row order is preserved as-is from GEE — no flip applied.
    # Empirically GEE returns north-up for this call pattern, so row 0 = northernmost.
    np.testing.assert_allclose(arr[0, 0], _NORTH_ROW)
    np.testing.assert_allclose(arr[0, 1], _SOUTH_ROW)


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
):
    """Fake ee for multiframe fetch: one value per temporal bin (None = empty).

    ``fallback_value=None`` makes the whole-window fallback probe fail;
    ``bin_size_raises`` makes the per-bin image count query raise.
    """

    class _FakeSize:
        def __init__(self, n=1, raises=False):
            self._n, self._raises = n, raises

        def getInfo(self):
            if self._raises:
                raise RuntimeError("transient getInfo failure")
            return self._n

    class _FakeImage:
        def __init__(self, value):
            self.value = value

        def select(self, _bands):
            return self

        def reproject(self, **_kwargs):
            return self

        def sampleRectangle(self, *, region, defaultValue):  # noqa: ARG002
            img = self

            class _Rect:
                def getInfo(self):
                    return {"properties": {"B1": [[img.value]]}}

            return _Rect()

    class _FakeBin:
        def __init__(self, value):
            self.value = value

        def size(self):
            if bin_size_raises:
                return _FakeSize(raises=True)
            return _FakeSize(1 if self.value is not None else 0)

        def median(self):
            return _FakeImage(self.value)

    class _FakeCollection:
        def __init__(self):
            self.bin_idx = 0

        def filterDate(self, _start, _end):
            # First filterDate builds col_all; later ones select bins in order.
            if not hasattr(self, "_dated"):
                self._dated = True
                return self
            b = _FakeBin(bin_values[self.bin_idx])
            self.bin_idx += 1
            return b

        def filterBounds(self, _region):
            return self

        def filter(self, _filt):
            return self

        def size(self):
            if fallback_value is None:
                return _FakeSize(raises=True)
            return _FakeSize(1)

        def median(self):
            return _FakeImage(fallback_value)

    return types.SimpleNamespace(
        ImageCollection=lambda _collection: _FakeCollection(),
        Filter=types.SimpleNamespace(
            lte=lambda f, v: ("lte", f, v),
            eq=lambda f, v: ("eq", f, v),
            listContains=lambda f, v: ("listContains", f, v),
        ),
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
        spatial=object(),
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
        spatial=object(),
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
            spatial=object(),
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
        spatial=object(),
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

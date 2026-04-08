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


def test_fetch_s1_vvvh_raw_chw_ignores_orbit_filter(monkeypatch):
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
            if kind == "eq" and field == "orbitProperties_pass":
                raise AssertionError("orbit filter should not be applied")
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

    arr = provider.fetch_s1_vvvh_raw_chw(
        spatial=object(),
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
        orbit="ASCENDING",
    )

    assert arr.shape == (2, 2, 2)


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

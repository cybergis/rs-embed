import numpy as np

import rs_embed.embedders.runtime_utils as ru
from rs_embed.core.specs import BBox, SensorSpec, TemporalSpec


def test_runtime_utils_fetch_sensor_patch_uses_shared_helper(monkeypatch):
    """runtime_utils.fetch_sensor_patch_chw delegates to provider.fetch_sensor_patch_chw."""
    calls = {"provider": 0}

    class _FakeProvider:
        def fetch_sensor_patch_chw(self, *, spatial, temporal, sensor, to_float_image=False):
            calls["provider"] += 1
            assert isinstance(spatial, BBox)
            assert sensor.collection == "FAKE/COLL"
            assert bool(to_float_image) is False
            return np.ones((1, 2, 2), dtype=np.float32)

    out = ru.fetch_sensor_patch_chw(
        _FakeProvider(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=None,
        sensor=SensorSpec(collection="FAKE/COLL", bands=("B1",)),
        to_float_image=False,
    )

    assert out.shape == (1, 2, 2)
    assert calls["provider"] == 1


def test_runtime_utils_fetch_sensor_patch_to_float_uses_shared_helper(monkeypatch):
    """runtime_utils.fetch_sensor_patch_chw passes to_float_image through to provider."""
    calls = {"provider": 0}

    class _FakeProvider:
        def fetch_sensor_patch_chw(self, *, spatial, temporal, sensor, to_float_image=False):
            calls["provider"] += 1
            assert isinstance(spatial, BBox)
            assert bool(to_float_image) is True
            return np.ones((1, 2, 2), dtype=np.float32)

    out = ru.fetch_sensor_patch_chw(
        _FakeProvider(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=None,
        sensor=SensorSpec(collection="FAKE/COLL", bands=("B1",)),
        to_float_image=True,
    )
    assert out.shape == (1, 2, 2)
    assert calls["provider"] == 1


def test_runtime_utils_fetch_s1_uses_bbox_fallback_wrapper(monkeypatch):
    calls = {"wrapper": 0}

    class _FakeProvider:
        def fetch_s1_vvvh_raw_chw(self, **kwargs):
            raise AssertionError("Should be invoked through wrapper callback, not directly in test")

    def _fake_wrapper(provider, *, spatial, scale_m, fill_value, fetch_fn, split_depth=0):  # noqa: ARG001
        calls["wrapper"] += 1
        assert isinstance(provider, _FakeProvider)
        assert isinstance(spatial, BBox)
        return np.ones((2, 3, 4), dtype=np.float32)

    monkeypatch.setattr(ru, "_fetch_spatial_array_with_bbox_fallback", _fake_wrapper)
    out = ru.fetch_s1_vvvh_raw_chw(
        _FakeProvider(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
    )
    assert out.shape == (2, 3, 4)
    assert calls["wrapper"] == 1


def test_runtime_utils_fetch_multiframe_uses_bbox_fallback_wrapper(monkeypatch):
    calls = {"wrapper": 0}

    class _FakeProvider:
        def fetch_multiframe_collection_raw_tchw(self, **kwargs):
            raise AssertionError("Should be invoked through wrapper callback, not directly in test")

    def _fake_wrapper(provider, *, spatial, scale_m, fill_value, fetch_fn, split_depth=0):  # noqa: ARG001
        calls["wrapper"] += 1
        assert isinstance(provider, _FakeProvider)
        assert isinstance(spatial, BBox)
        return np.ones((4, 3, 2, 5), dtype=np.float32)

    monkeypatch.setattr(ru, "_fetch_spatial_array_with_bbox_fallback", _fake_wrapper)
    out = ru.fetch_s2_multiframe_raw_tchw(
        _FakeProvider(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
        bands=("B4", "B3", "B2"),
        n_frames=4,
    )
    assert out.shape == (4, 3, 2, 5)
    assert calls["wrapper"] == 1


def test_runtime_utils_fetch_all_bands_uses_provider_and_returns_names():
    class _FakeProvider:
        def fetch_collection_patch_all_bands_chw(self, **kwargs):
            return np.ones((3, 2, 2), dtype=np.float32), ("E1", "E2", "E3")

    arr, names = ru.fetch_collection_patch_all_bands_chw(
        _FakeProvider(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=TemporalSpec.year(2024),
        collection="FAKE/COLL",
        scale_m=10,
        fill_value=0.0,
        composite="mosaic",
    )
    assert arr.shape == (3, 2, 2)
    assert names == ("E1", "E2", "E3")

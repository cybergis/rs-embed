import numpy as np

import rs_embed.providers.fetch as ru
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


def test_runtime_utils_fetch_s1_delegates_to_provider():
    calls = {"provider": 0}

    class _FakeProvider:
        def fetch_s1_vvvh_raw_chw(self, *, spatial, **kwargs):
            calls["provider"] += 1
            assert isinstance(spatial, BBox)
            return np.ones((2, 3, 4), dtype=np.float32)

    out = ru.fetch_s1_vvvh_raw_chw(
        _FakeProvider(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
    )
    assert out.shape == (2, 3, 4)
    assert calls["provider"] == 1


def test_runtime_utils_fetch_multiframe_delegates_to_provider():
    calls = {"provider": 0}

    class _FakeProvider:
        def fetch_multiframe_collection_raw_tchw(self, *, spatial, **kwargs):
            calls["provider"] += 1
            assert isinstance(spatial, BBox)
            return np.ones((4, 3, 2, 5), dtype=np.float32)

    out = ru.fetch_multiframe_patch_raw_tchw(
        _FakeProvider(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
        bands=("B4", "B3", "B2"),
        collection="COPERNICUS/S2_SR_HARMONIZED",
        n_frames=4,
    )
    assert out.shape == (4, 3, 2, 5)
    assert calls["provider"] == 1


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


def test_fetch_collection_binned_raw_tchw_stacks_and_flags_empty_bins():
    bins = [
        ("2022-01-01", "2022-01-31"),
        ("2022-01-31", "2022-03-02"),
        ("2022-03-02", "2022-04-01"),
    ]
    calls = {"n": 0}

    class _FakeProvider:
        def fetch_sensor_patch_chw(self, *, spatial, temporal, sensor, to_float_image=False):
            calls["n"] += 1
            if str(temporal.start) == "2022-01-31":  # second bin has no imagery
                from rs_embed.core.errors import ProviderError

                raise ProviderError("no images found")
            return np.full((3, 4, 4), float(calls["n"]), dtype=np.float32)

    arr, meta = ru.fetch_collection_binned_raw_tchw(
        _FakeProvider(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        bins=bins,
        collection="FAKE/COLL",
        bands=("B1", "B2", "B3"),
    )
    assert arr.shape == (3, 3, 4, 4)
    assert not np.isnan(arr[0]).any()
    assert np.isnan(arr[1]).all()  # NaN sentinel for empty bin
    assert not np.isnan(arr[2]).any()
    assert [m["empty"] for m in meta["frames"]] == [False, True, False]
    assert meta["n_empty"] == 1
    assert meta["frames"][1]["start"] == "2022-01-31"


def test_fetch_collection_binned_raw_tchw_raises_when_all_bins_empty():
    import pytest

    from rs_embed.core.errors import ModelError, ProviderError

    class _FakeProvider:
        def fetch_sensor_patch_chw(self, **kwargs):
            raise ProviderError("no images found")

    with pytest.raises(ModelError, match="no imagery found in any"):
        ru.fetch_collection_binned_raw_tchw(
            _FakeProvider(),
            spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
            bins=[("2022-01-01", "2022-01-31"), ("2022-01-31", "2022-03-02")],
            collection="FAKE/COLL",
            bands=("B1",),
        )


def test_fetch_s1_vvvh_binned_raw_tchw_reuses_single_frame_path():
    bins = [("2022-01-01", "2022-01-31"), ("2022-01-31", "2022-03-02")]
    seen = []

    class _FakeProvider:
        def fetch_s1_vvvh_raw_chw_with_meta(self, *, spatial, temporal, **kwargs):
            seen.append((str(temporal.start), kwargs["require_iw"], kwargs["use_float_linear"]))
            if str(temporal.start) == "2022-01-31":
                from rs_embed.core.errors import ProviderError

                raise ProviderError("no S1 imagery")
            return np.full((2, 4, 4), -15.0, dtype=np.float32), {"iw_applied": True}

    arr, meta = ru.fetch_s1_vvvh_binned_raw_tchw(
        _FakeProvider(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        bins=bins,
        use_float_linear=False,
        require_iw=True,
    )
    assert arr.shape == (2, 2, 4, 4)
    assert not np.isnan(arr[0]).any()
    assert np.isnan(arr[1]).all()
    assert meta["frames"][0]["fetch"] == {"iw_applied": True}
    assert meta["frames"][1]["empty"] is True
    # per-bin calls carry the S1-specific options through
    assert seen == [("2022-01-01", True, False), ("2022-01-31", True, False)]


# ---------------------------------------------------------------------------
# Frame-diversity helpers (back-filled / duplicate multi-frame detection)
# ---------------------------------------------------------------------------


def test_count_distinct_frames_collapses_duplicates():
    f0 = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    f2 = (f0 + 1000.0).astype(np.float32)
    # frames 1 and 3 are exact back-filled copies of frame 0 -> 2 distinct
    x = np.stack([f0, f0.copy(), f2, f0.copy()], axis=0)
    assert ru.count_distinct_frames(x) == 2


def test_count_distinct_frames_all_distinct_and_edge_shapes():
    x = np.stack([np.full((2, 2, 2), float(i), dtype=np.float32) for i in range(3)], axis=0)
    assert ru.count_distinct_frames(x) == 3
    assert ru.count_distinct_frames(np.zeros((1, 2, 2, 2), dtype=np.float32)) == 1
    assert ru.count_distinct_frames(np.zeros((3, 4, 4), dtype=np.float32)) == 1  # non-4D -> 1


def test_frame_diversity_meta_keys():
    assert ru.frame_diversity_meta(n_requested=4, n_distinct=2) == {
        "n_frames_requested": 4,
        "n_distinct_frames": 2,
        "n_backfilled_frames": 2,
    }
    # all distinct -> nothing back-filled
    assert ru.frame_diversity_meta(n_requested=3, n_distinct=3)["n_backfilled_frames"] == 0


def test_multiframe_fetch_warns_on_backfilled_frames():
    import warnings

    f0 = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    f2 = (f0 + 5000.0).astype(np.float32)
    backfilled = np.stack([f0, f0.copy(), f2, f0.copy()], axis=0)  # 2 distinct of 4

    class _FakeProvider:
        def fetch_multiframe_collection_raw_tchw(self, **kwargs):
            return backfilled

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ru.fetch_multiframe_patch_raw_tchw(
            _FakeProvider(),
            spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
            temporal=TemporalSpec.range("2024-01-01", "2024-04-01"),
            bands=("B4", "B3", "B2"),
            collection="COPERNICUS/S2_SR_HARMONIZED",
            n_frames=4,
        )
    msgs = [str(m.message) for m in w if issubclass(m.category, UserWarning)]
    assert any("only 2 of 4 temporal sub-window" in m for m in msgs)


def test_multiframe_fetch_silent_when_all_frames_distinct():
    import warnings

    distinct = np.stack([np.full((3, 2, 2), float(i), dtype=np.float32) for i in range(4)], axis=0)

    class _FakeProvider:
        def fetch_multiframe_collection_raw_tchw(self, **kwargs):
            return distinct

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ru.fetch_multiframe_patch_raw_tchw(
            _FakeProvider(),
            spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
            temporal=TemporalSpec.range("2024-01-01", "2024-04-01"),
            bands=("B4", "B3", "B2"),
            collection="COPERNICUS/S2_SR_HARMONIZED",
            n_frames=4,
        )
    assert [m for m in w if "sub-window" in str(m.message)] == []

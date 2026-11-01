import numpy as np

from rs_embed.core.specs import BBox, SensorSpec, TemporalSpec
from rs_embed.providers.base import ProviderBase
from rs_embed.providers.gee_utils import (
    _stitch_bbox_split_arrays,
    fetch_provider_patch_raw,
)


class _FakeBBoxSplitProvider(ProviderBase):
    name = "fake_bbox_split"

    def __init__(self):
        self.full_bbox = BBox(minlon=0.0, minlat=0.0, maxlon=4.0, maxlat=1.0)
        self.full_h = 2
        self.full_w = 6
        self.max_pixels = 8
        self.build_calls = 0
        self.fetch_calls = 0

    def ensure_ready(self) -> None:  # pragma: no cover - unused
        return None

    def get_region(self, spatial):
        return spatial

    def build_image(self, *, sensor, temporal, region=None):  # noqa: ARG002
        self.build_calls += 1
        return object()

    def fetch_array_chw(self, *, image, bands, spatial, scale_m, fill_value, collection=None):  # noqa: ARG002
        self.fetch_calls += 1
        bbox = spatial
        assert isinstance(bbox, BBox)
        h, w, y0, x0 = self._dims_and_offsets(bbox)
        if h * w > self.max_pixels:
            raise RuntimeError(
                f"Total request size ({h * w * 5} bytes) must be less than or equal to "
                f"{self.max_pixels * 5} bytes."
            )
        # fetch_array_chw contract: returns north-up CHW (row 0 = northernmost).
        yy = np.arange(y0, y0 + h, dtype=np.float32)[:, None]
        xx = np.arange(x0, x0 + w, dtype=np.float32)[None, :]
        base = yy * 100.0 + xx
        out = np.stack([base + float(i) * 1000.0 for i in range(len(bands))], axis=0)
        return out.astype(np.float32)

    def _dims_and_offsets(self, bbox: BBox):
        lon_span = float(self.full_bbox.maxlon - self.full_bbox.minlon)
        lat_span = float(self.full_bbox.maxlat - self.full_bbox.minlat)
        x0 = int(
            round(((float(bbox.minlon) - float(self.full_bbox.minlon)) / lon_span) * self.full_w)
        )
        x1 = int(
            round(((float(bbox.maxlon) - float(self.full_bbox.minlon)) / lon_span) * self.full_w)
        )
        y0 = int(
            round(((float(self.full_bbox.maxlat) - float(bbox.maxlat)) / lat_span) * self.full_h)
        )
        y1 = int(
            round(((float(self.full_bbox.maxlat) - float(bbox.minlat)) / lat_span) * self.full_h)
        )
        h = max(1, y1 - y0)
        w = max(1, x1 - x0)
        return h, w, y0, x0


class _FakeVerticalSouthUpProvider(_FakeBBoxSplitProvider):
    """Force y-splits on south-up tiles."""

    def __init__(self):
        super().__init__()
        self.full_bbox = BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=4.0)
        self.full_h = 6
        self.full_w = 2
        self.max_pixels = 8  # 12 px full patch -> split; each half 6 px -> success


class _FakeDeepVerticalSouthUpProvider(_FakeBBoxSplitProvider):
    """Force multiple recursive y-splits on south-up tiles."""

    def __init__(self):
        super().__init__()
        # Pick a bbox whose Web Mercator height estimates split cleanly as
        # 16 -> 8 -> 4 rows at 75 km resolution, matching this fake provider's
        # linear 16-row grid during recursive y-splits.
        self.full_bbox = BBox(minlon=0.0, minlat=0.0, maxlon=0.5, maxlat=10.06)
        self.full_h = 16
        self.full_w = 1
        self.max_pixels = 4  # 16 px full patch -> 8 -> 4, requiring 2 y-stitch levels


def test_fetch_provider_patch_raw_recursively_splits_bbox_on_gee_pixel_limit():
    provider = _FakeBBoxSplitProvider()
    sensor = SensorSpec(collection="FAKE/COLL", bands=("B1",), scale_m=75000, fill_value=0.0)

    arr = fetch_provider_patch_raw(
        provider,
        spatial=provider.full_bbox,
        temporal=None,
        sensor=sensor,
    )

    assert arr.shape == (1, provider.full_h, provider.full_w)
    expected = np.array(
        [[[0, 1, 2, 3, 4, 5], [100, 101, 102, 103, 104, 105]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(arr, expected)
    assert provider.build_calls == 1
    # One failed full fetch + two successful sub-fetches
    assert provider.fetch_calls >= 3


def test_fetch_provider_patch_raw_uses_generic_path_for_s1():
    """After the model-input-contract refactor, S1 VV/VH no longer has a
    special case in ``fetch_provider_patch_raw`` — it goes through the
    generic ``get_region`` → ``build_image`` → ``fetch_array_chw`` path
    like every other sensor.  Model-specific S1 fetch logic is now owned
    by ``TerraFMBEmbedder.fetch_input()``.
    """

    class _FakeGenericS1Provider(ProviderBase):
        name = "fake_s1_generic"

        def __init__(self):
            self.build_calls = 0
            self.fetch_calls = 0

        def ensure_ready(self) -> None:
            return None

        def get_region(self, spatial):
            return spatial

        def build_image(self, *, sensor, temporal, region=None):  # noqa: ARG002
            self.build_calls += 1
            return object()

        def fetch_array_chw(self, *, image, bands, spatial, scale_m, fill_value, collection=None):
            self.fetch_calls += 1
            # fetch_array_chw contract: return north-up CHW.
            return np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    provider = _FakeGenericS1Provider()
    sensor = SensorSpec(
        collection="COPERNICUS/S1_GRD_FLOAT",
        bands=("VV", "VH"),
        modality="s1",
        s1_require_iw=True,
        s1_relax_iw_on_empty=True,
    )

    arr = fetch_provider_patch_raw(
        provider,
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=0.001, maxlat=0.001),
        temporal=TemporalSpec.range("2024-01-01", "2024-02-01"),
        sensor=sensor,
    )

    assert arr.shape == (2, 3, 4)
    assert provider.build_calls == 1
    assert provider.fetch_calls == 1


def test_fetch_provider_patch_raw_returns_north_up_tile():
    provider = _FakeBBoxSplitProvider()
    sensor = SensorSpec(collection="FAKE/COLL", bands=("B1",), scale_m=75000, fill_value=0.0)
    spatial = BBox(minlon=0.0, minlat=0.0, maxlon=2.0, maxlat=1.0)

    arr = fetch_provider_patch_raw(
        provider,
        spatial=spatial,
        temporal=None,
        sensor=sensor,
    )

    expected = np.array([[[0, 1, 2], [100, 101, 102]]], dtype=np.float32)
    assert arr.shape == expected.shape
    np.testing.assert_allclose(arr, expected)


def test_fetch_provider_patch_raw_correctly_stitches_y_split_tiles():
    provider = _FakeVerticalSouthUpProvider()
    sensor = SensorSpec(collection="FAKE/COLL", bands=("B1",), scale_m=75000, fill_value=0.0)

    arr = fetch_provider_patch_raw(
        provider,
        spatial=provider.full_bbox,
        temporal=None,
        sensor=sensor,
    )

    expected = np.array(
        [[[0, 1], [100, 101], [200, 201], [300, 301], [400, 401], [500, 501]]],
        dtype=np.float32,
    )
    assert arr.shape == expected.shape
    np.testing.assert_allclose(arr, expected)


def test_fetch_provider_patch_raw_correctly_stitches_recursive_y_split_tiles():
    provider = _FakeDeepVerticalSouthUpProvider()
    sensor = SensorSpec(collection="FAKE/COLL", bands=("B1",), scale_m=75000, fill_value=0.0)

    arr = fetch_provider_patch_raw(
        provider,
        spatial=provider.full_bbox,
        temporal=None,
        sensor=sensor,
    )

    expected = (
        np.arange(provider.full_h, dtype=np.float32).reshape(1, provider.full_h, provider.full_w)
        * 100.0
    )
    assert arr.shape == expected.shape
    np.testing.assert_allclose(arr, expected)


def test_stitch_bbox_split_arrays_keeps_normalized_y_tiles_in_spatial_order():
    arr_a = np.array(
        [[[0, 1], [100, 101], [200, 201]]],
        dtype=np.float32,
    )
    arr_b = np.array(
        [[[300, 301], [400, 401], [500, 501]]],
        dtype=np.float32,
    )

    out = _stitch_bbox_split_arrays(arr_a, arr_b, "y")

    expected = np.array(
        [[[0, 1], [100, 101], [200, 201], [300, 301], [400, 401], [500, 501]]],
        dtype=np.float32,
    )
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected)

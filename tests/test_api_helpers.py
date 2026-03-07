import numpy as np

from rs_embed.core.specs import BBox, SensorSpec
from rs_embed.providers.gee_utils import fetch_provider_patch_raw
from rs_embed.providers.base import ProviderBase


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

    def fetch_array_chw(
        self, *, image, bands, region, scale_m, fill_value, collection=None
    ):  # noqa: ARG002
        self.fetch_calls += 1
        bbox = region
        assert isinstance(bbox, BBox)
        h, w, y0, x0 = self._dims_and_offsets(bbox)
        if h * w > self.max_pixels:
            raise RuntimeError(
                "Image.sampleRectangle: Too many pixels in sample; must be <= 8. Got "
                f"{h * w}."
            )
        yy = np.arange(y0, y0 + h, dtype=np.float32)[:, None]
        xx = np.arange(x0, x0 + w, dtype=np.float32)[None, :]
        base = yy * 100.0 + xx
        out = np.stack([base + float(i) * 1000.0 for i in range(len(bands))], axis=0)
        return out.astype(np.float32)

    def _dims_and_offsets(self, bbox: BBox):
        lon_span = float(self.full_bbox.maxlon - self.full_bbox.minlon)
        lat_span = float(self.full_bbox.maxlat - self.full_bbox.minlat)
        x0 = int(
            round(
                ((float(bbox.minlon) - float(self.full_bbox.minlon)) / lon_span)
                * self.full_w
            )
        )
        x1 = int(
            round(
                ((float(bbox.maxlon) - float(self.full_bbox.minlon)) / lon_span)
                * self.full_w
            )
        )
        y0 = int(
            round(
                ((float(self.full_bbox.maxlat) - float(bbox.maxlat)) / lat_span)
                * self.full_h
            )
        )
        y1 = int(
            round(
                ((float(self.full_bbox.maxlat) - float(bbox.minlat)) / lat_span)
                * self.full_h
            )
        )
        h = max(1, y1 - y0)
        w = max(1, x1 - x0)
        return h, w, y0, x0


class _FakeBoundaryOverlapProvider(_FakeBBoxSplitProvider):
    """Simulate GEE returning one duplicated boundary column on the left child tile."""

    def fetch_array_chw(
        self, *, image, bands, region, scale_m, fill_value, collection=None
    ):  # noqa: ARG002
        self.fetch_calls += 1
        bbox = region
        assert isinstance(bbox, BBox)
        h, w, y0, x0 = self._dims_and_offsets(bbox)
        if h * w > self.max_pixels:
            raise RuntimeError(
                "Image.sampleRectangle: Too many pixels in sample; must be <= 8. Got "
                f"{h * w}."
            )

        # Left child tile includes one extra boundary column, creating a 1-pixel overlap
        # with the right child tile after splitting the full bbox.
        if np.isclose(float(bbox.minlon), float(self.full_bbox.minlon)) and float(
            bbox.maxlon
        ) < float(self.full_bbox.maxlon):
            if x0 + w < self.full_w:
                w += 1

        yy = np.arange(y0, y0 + h, dtype=np.float32)[:, None]
        xx = np.arange(x0, x0 + w, dtype=np.float32)[None, :]
        base = yy * 100.0 + xx
        out = np.stack([base + float(i) * 1000.0 for i in range(len(bands))], axis=0)
        return out.astype(np.float32)


class _FakeVerticalSouthUpProvider(_FakeBBoxSplitProvider):
    """Force y-splits and return each tile in local south->north row order."""

    def __init__(self):
        super().__init__()
        self.full_bbox = BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=4.0)
        self.full_h = 6
        self.full_w = 2
        self.max_pixels = 8  # 12 px full patch -> split; each half 6 px -> success

    def fetch_array_chw(
        self, *, image, bands, region, scale_m, fill_value, collection=None
    ):  # noqa: ARG002
        self.fetch_calls += 1
        bbox = region
        assert isinstance(bbox, BBox)
        h, w, y0, x0 = self._dims_and_offsets(bbox)
        if h * w > self.max_pixels:
            raise RuntimeError(
                "Image.sampleRectangle: Too many pixels in sample; must be <= 8. Got "
                f"{h * w}."
            )
        # Local south-up: top row corresponds to the tile's southern edge.
        yy = np.arange(y0 + h - 1, y0 - 1, -1, dtype=np.float32)[:, None]
        xx = np.arange(x0, x0 + w, dtype=np.float32)[None, :]
        base = yy * 100.0 + xx
        out = np.stack([base + float(i) * 1000.0 for i in range(len(bands))], axis=0)
        return out.astype(np.float32)


def test_fetch_provider_patch_raw_recursively_splits_bbox_on_gee_pixel_limit():
    provider = _FakeBBoxSplitProvider()
    sensor = SensorSpec(
        collection="FAKE/COLL", bands=("B1",), scale_m=75000, fill_value=0.0
    )

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


def test_fetch_provider_patch_raw_trims_boundary_overlap_when_stitching():
    provider = _FakeBoundaryOverlapProvider()
    sensor = SensorSpec(
        collection="FAKE/COLL", bands=("B1",), scale_m=75000, fill_value=0.0
    )

    arr = fetch_provider_patch_raw(
        provider,
        spatial=provider.full_bbox,
        temporal=None,
        sensor=sensor,
    )

    expected = np.array(
        [[[0, 1, 2, 3, 4, 5], [100, 101, 102, 103, 104, 105]]],
        dtype=np.float32,
    )
    assert arr.shape == expected.shape
    np.testing.assert_allclose(arr, expected)


def test_fetch_provider_patch_raw_flips_each_tile_before_y_stitch():
    provider = _FakeVerticalSouthUpProvider()
    sensor = SensorSpec(
        collection="FAKE/COLL", bands=("B1",), scale_m=75000, fill_value=0.0
    )

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

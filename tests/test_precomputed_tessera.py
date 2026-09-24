import warnings

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

from rs_embed.core.errors import ModelError
from rs_embed.core.specs import BBox, OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders.precomputed_tessera import TesseraEmbedder
from rs_embed.tools.projection import web_mercator_to_lonlat


class _FakeRegistry:
    def __init__(self, n_tiles: int):
        self._n_tiles = n_tiles

    def load_blocks_for_region(self, bounds, year):
        return list(range(self._n_tiles))


class _FakeGeoTessera:
    def __init__(self, rows):
        self._rows = rows
        self.registry = _FakeRegistry(len(rows))

    def fetch_embeddings(self, tiles):
        for i in tiles:
            yield self._rows[i]


def _fake_rows():
    """2x2 tiles already on the common EPSG:3857 lattice (10 m pixels, 4x4 px each).

    Together they cover x in [0, 80], y in [0, 80] meters; tile value = tile id.
    """
    d = 64
    h = w = 4
    return [
        (
            2021,
            0.0,
            0.0,
            np.full((h, w, d), 1.0, np.float32),
            "EPSG:3857",
            Affine(10, 0, 0, 0, -10, 80),
        ),
        (
            2021,
            0.0,
            0.0,
            np.full((h, w, d), 2.0, np.float32),
            "EPSG:3857",
            Affine(10, 0, 40, 0, -10, 80),
        ),
        (
            2021,
            0.0,
            0.0,
            np.full((h, w, d), 3.0, np.float32),
            "EPSG:3857",
            Affine(10, 0, 0, 0, -10, 40),
        ),
        (
            2021,
            0.0,
            0.0,
            np.full((h, w, d), 4.0, np.float32),
            "EPSG:3857",
            Affine(10, 0, 40, 0, -10, 40),
        ),
    ]


def _bbox_from_mercator(minx, miny, maxx, maxy) -> BBox:
    minlon, minlat = web_mercator_to_lonlat(minx, miny)
    maxlon, maxlat = web_mercator_to_lonlat(maxx, maxy)
    return BBox(minlon=minlon, minlat=minlat, maxlon=maxlon, maxlat=maxlat)


# One common-grid pixel inside tile 4 (x 40..80, y 0..40).
_ROI = _bbox_from_mercator(42.0, 12.0, 48.0, 18.0)


def _embedder(monkeypatch, rows):
    embedder = TesseraEmbedder()
    embedder.model_name = "tessera"
    monkeypatch.setattr(embedder, "_get_gt", lambda _cache: _FakeGeoTessera(rows))
    return embedder


def _get(embedder, spatial, output=OutputSpec.pooled(), **kw):
    return embedder.get_embedding(
        spatial=spatial,
        temporal=TemporalSpec.year(2021),
        sensor=None,
        output=output,
        backend="auto",
        **kw,
    )


def _utm_tile(crs: str, lon: float, lat: float, *, half_px: int, value_fn):
    """A north-up 10 m tile in *crs* centered on (lon, lat); values from value_fn(row, col)."""
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    n = 2 * half_px
    rows, cols = np.mgrid[0:n, 0:n]
    emb = np.repeat(value_fn(rows, cols)[..., None].astype(np.float32), 64, axis=-1)
    transform = Affine(10, 0, x - half_px * 10, 0, -10, y + half_px * 10)
    return (2021, lon, lat, emb, crs, transform)


def test_tessera_lattice_tiles_land_on_common_grid_without_mosaic(monkeypatch):
    import rs_embed.embedders.precomputed_tessera as tessera_mod

    embedder = _embedder(monkeypatch, _fake_rows())
    zeros_calls = []
    real_zeros = tessera_mod.np.zeros

    def _zeros_probe(shape, *args, **kwargs):
        zeros_calls.append(tuple(int(v) for v in shape))
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(tessera_mod.np, "zeros", _zeros_probe)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no projection / UTM warning on a clean request
        emb = _get(embedder, _ROI)

    assert emb.meta["input_crs"] == "EPSG:4326"
    assert emb.meta["output_crs"] == "EPSG:3857"
    assert emb.meta["projection_mode"] == "common_grid"
    assert emb.meta["scale_m"] == 10
    assert emb.meta["grid_hw"] == (1, 1)
    assert emb.meta["tile_crs"] == ["EPSG:3857"]
    assert emb.meta["coverage"] == 1.0
    assert emb.meta["transform"] == Affine(10, 0, 40, 0, -10, 20)
    # Only the output canvas is allocated, never the 8x8 mosaic.
    assert (1, 1, 64) in zeros_calls
    assert (8, 8, 64) not in zeros_calls
    np.testing.assert_allclose(emb.data, np.full((64,), 4.0, dtype=np.float32))


def test_tessera_grid_output_is_north_up_on_common_grid(monkeypatch):
    embedder = _embedder(monkeypatch, _fake_rows())
    # Two rows straddling the tile-1 / tile-3 seam (y=40): row 0 is tile 1, row 1 is tile 3.
    roi = _bbox_from_mercator(12.0, 32.0, 18.0, 48.0)

    emb = _get(embedder, roi, output=OutputSpec.grid())

    assert emb.meta["grid_hw"] == (2, 1)
    assert emb.meta["transform"].e < 0
    assert tuple(emb.data.dims) == ("d", "y", "x")
    np.testing.assert_allclose(emb.data.values[0, :, 0], [1.0, 3.0])


def test_tessera_utm_tile_is_reprojected_onto_common_grid(monkeypatch):
    # Tile in UTM 31N; ROI is a PointBuffer well inside it. Values grow with the
    # tile row, so the output must grow southward (north-up) after reprojection.
    tile = _utm_tile("EPSG:32631", 3.0, 45.0, half_px=60, value_fn=lambda r, c: r)
    embedder = _embedder(monkeypatch, [tile])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emb = _get(embedder, PointBuffer(lon=3.0, lat=45.0, buffer_m=200), output=OutputSpec.grid())

    assert emb.meta["output_crs"] == "EPSG:3857"
    assert emb.meta["tile_crs"] == ["EPSG:32631"]
    assert emb.meta["coverage"] == 1.0
    h, w = emb.meta["grid_hw"]
    # 400 m EPSG:3857 square at 10 m -> 40 px (plus lattice snapping).
    assert 40 <= h <= 42 and 40 <= w <= 42
    col = emb.data.values[0, :, w // 2]
    assert np.all(np.diff(col) >= 0) and col[-1] > col[0]


def test_tessera_roi_across_utm_zones_is_served_with_warning(monkeypatch):
    # Zone 31 ends at lon 6. West tile in 32631, east tile in 32632, each
    # ~2.4 km wide and ending at the boundary (a few px overlap), so the ROI
    # straddling lon 6 is filled by both.
    west = _utm_tile("EPSG:32631", 5.985, 45.0, half_px=120, value_fn=lambda r, c: 0 * r + 1.0)
    east = _utm_tile("EPSG:32632", 6.015, 45.0, half_px=120, value_fn=lambda r, c: 0 * r + 2.0)
    embedder = _embedder(monkeypatch, [west, east])
    roi = BBox(minlon=5.997, minlat=44.999, maxlon=6.003, maxlat=45.001)

    with pytest.warns(UserWarning, match="UTM zone boundary"):
        emb = _get(embedder, roi, output=OutputSpec.grid())

    assert emb.meta["tile_crs"] == ["EPSG:32631", "EPSG:32632"]
    assert emb.meta["coverage"] > 0.99
    values = emb.data.values[0]
    assert np.all(values[:, 0] == 1.0)
    assert np.all(values[:, -1] == 2.0)


def test_tessera_roi_outside_tiles_raises(monkeypatch):
    embedder = _embedder(monkeypatch, _fake_rows())
    with pytest.raises(ModelError, match="does not overlap"):
        _get(embedder, _bbox_from_mercator(1000.0, 1000.0, 1100.0, 1100.0))


def _embedder_with_captured_cache(monkeypatch):
    embedder = TesseraEmbedder()
    embedder.model_name = "tessera"
    captured: dict[str, str] = {}

    def _get_gt(cache_key):
        captured["cache_key"] = cache_key
        return _FakeGeoTessera(_fake_rows())

    monkeypatch.setattr(embedder, "_get_gt", _get_gt)
    return embedder, captured


def test_tessera_model_config_cache_dir_channel(monkeypatch):
    embedder, captured = _embedder_with_captured_cache(monkeypatch)

    emb = _get(embedder, _ROI, model_config={"cache_dir": "/tmp/tessera-mc-cache"})

    assert captured["cache_key"] == "/tmp/tessera-mc-cache"
    assert emb.meta["cache_dir"] == "/tmp/tessera-mc-cache"


def test_tessera_cache_collection_prefix_is_deprecated_but_works(monkeypatch):
    from rs_embed.core.specs import SensorSpec

    embedder, captured = _embedder_with_captured_cache(monkeypatch)

    with pytest.warns(DeprecationWarning, match="cache_dir"):
        emb = embedder.get_embedding(
            spatial=_ROI,
            temporal=TemporalSpec.year(2021),
            sensor=SensorSpec(collection="cache:/tmp/tessera-legacy-cache", bands=()),
            output=OutputSpec.pooled(),
            backend="auto",
        )

    assert captured["cache_key"] == "/tmp/tessera-legacy-cache"
    assert emb.meta["cache_dir"] == "/tmp/tessera-legacy-cache"


def test_tessera_model_config_wins_over_collection_prefix_and_env(monkeypatch):
    from rs_embed.core.specs import SensorSpec

    embedder, captured = _embedder_with_captured_cache(monkeypatch)
    monkeypatch.setenv("RS_EMBED_TESSERA_CACHE", "/tmp/tessera-env-cache")

    embedder.get_embedding(
        spatial=_ROI,
        temporal=TemporalSpec.year(2021),
        sensor=SensorSpec(collection="cache:/tmp/tessera-legacy-cache", bands=()),
        output=OutputSpec.pooled(),
        backend="auto",
        model_config={"cache_dir": "/tmp/tessera-mc-cache"},
    )

    assert captured["cache_key"] == "/tmp/tessera-mc-cache"


def test_tessera_batch_forwards_model_config(monkeypatch):
    embedder, captured = _embedder_with_captured_cache(monkeypatch)
    monkeypatch.setenv("RS_EMBED_TESSERA_BATCH_WORKERS", "1")

    out = embedder.get_embeddings_batch(
        spatials=[_ROI],
        temporal=TemporalSpec.year(2021),
        output=OutputSpec.pooled(),
        backend="auto",
        model_config={"cache_dir": "/tmp/tessera-mc-cache"},
    )

    assert len(out) == 1
    assert captured["cache_key"] == "/tmp/tessera-mc-cache"

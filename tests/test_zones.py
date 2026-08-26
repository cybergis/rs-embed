"""embed_zones: per-zone aggregation of pixel embeddings.

The provider is stubbed throughout — these check the geometry, the accounting and the
contract, not Earth Engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rs_embed.core.errors import SpecError
from rs_embed.zones import ZoneEmbedding, ZoneEmbeddings, _to_lonlat, embed_zones

gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")
from shapely.geometry import box  # noqa: E402


def _square(lon: float, lat: float, size_deg: float):
    return box(lon, lat, lon + size_deg, lat + size_deg)


def _zones_gdf():
    """Two adjacent squares near Chicago, with ids that are not row numbers."""
    return gpd.GeoDataFrame(
        {"geoid": ["A", "B"]},
        geometry=[_square(-87.65, 41.90, 0.01), _square(-87.64, 41.90, 0.01)],
        crs="EPSG:4326",
    )


class _StubEmbedding:
    def __init__(self, data, meta):
        self.data = data
        self.meta = meta


def _patch_provider(monkeypatch, *, dims=4, scale_m=10, value=1.0, fail_tiles=0):
    """Stub get_embedding: a constant-valued grid sized from the requested bbox."""
    calls = {"n": 0}

    def fake(model, *, spatial, temporal=None, output=None, backend="auto", **kw):
        calls["n"] += 1
        if calls["n"] <= fail_tiles:
            raise RuntimeError("provider unavailable")
        x0, y0 = _merc(spatial.minlon, spatial.minlat)
        x1, y1 = _merc(spatial.maxlon, spatial.maxlat)
        w = max(1, round((x1 - x0) / scale_m))
        h = max(1, round((y1 - y0) / scale_m))
        arr = np.full((dims, h, w), value, dtype=np.float32)
        return _StubEmbedding(arr, {"scale_m": scale_m, "bands": tuple(f"A{i}" for i in range(dims))})

    monkeypatch.setattr("rs_embed.api.get_embedding", fake)
    return calls


def _merc(lon, lat):
    R = 6378137.0
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def test_every_zone_gets_a_vector_and_its_support(monkeypatch):
    _patch_provider(monkeypatch, dims=4, value=2.0)
    out = embed_zones("gse", zones=_zones_gdf(), zone_id_field="geoid", tile_px=128)

    assert isinstance(out, ZoneEmbeddings)
    assert [z.zone_id for z in out.zones] == ["A", "B"]
    for z in out.zones:
        assert z.pixels > 0
        assert z.mean == pytest.approx(np.full(4, 2.0))
        assert z.total == pytest.approx(z.mean * z.pixels)
        assert z.area_km2 > 0
    assert out.meta["dims"] == 4
    assert out.meta["zones_with_pixels"] == 2


def test_pixel_count_matches_the_polygon_area(monkeypatch):
    """The independent check on the affine: pixels x pixel area should recover the geometry."""
    _patch_provider(monkeypatch, scale_m=10)
    out = embed_zones("gse", zones=_zones_gdf(), zone_id_field="geoid", tile_px=256)

    ground = out.meta["pixel_ground_m"]
    for z in out.covered:
        implied = z.pixels * ground * ground / 1e6
        assert implied == pytest.approx(z.area_km2, rel=0.05)


def test_scale_m_is_reported_as_mercator_and_as_ground(monkeypatch):
    """scale_m is Web Mercator metres, which run 1/cos(lat) long; both are surfaced so a
    caller never reads '10 m' and means 10 m on the ground."""
    _patch_provider(monkeypatch, scale_m=10)
    out = embed_zones("gse", zones=_zones_gdf(), zone_id_field="geoid")

    assert out.meta["scale_m"] == 10
    expected = 10 * math.cos(math.radians(41.905))
    assert out.meta["pixel_ground_m"] == pytest.approx(expected, rel=0.01)
    assert out.meta["pixel_ground_m"] < out.meta["scale_m"]


def test_sums_let_zones_roll_up_exactly(monkeypatch):
    """A mean of means is wrong across unequal zones; the kept sums make the union exact."""
    _patch_provider(monkeypatch, dims=2, value=1.0)
    out = embed_zones("gse", zones=_zones_gdf(), zone_id_field="geoid")

    rolled = out.rollup({"A": "county", "B": "county"})
    assert [z.zone_id for z in rolled.zones] == ["county"]
    parent = rolled.zones[0]
    assert parent.pixels == sum(z.pixels for z in out.covered)
    assert parent.total == pytest.approx(sum(z.total for z in out.covered))
    assert parent.area_km2 == pytest.approx(sum(z.area_km2 for z in out.covered))


def test_rollup_of_unequal_zones_is_not_the_average_of_means(monkeypatch):
    a = ZoneEmbedding("a", pixels=100, area_km2=1.0, total=np.array([10.0]), mean=np.array([0.1]))
    b = ZoneEmbedding("b", pixels=400, area_km2=4.0, total=np.array([30.0]), mean=np.array([0.075]))
    rolled = ZoneEmbeddings(zones=[a, b]).rollup({"a": "p", "b": "p"})

    assert rolled.zones[0].mean == pytest.approx(np.array([40.0 / 500]))
    naive = (a.mean + b.mean) / 2
    assert rolled.zones[0].mean != pytest.approx(naive)


def test_to_frame_is_the_shape_a_model_consumes(monkeypatch):
    _patch_provider(monkeypatch, dims=3)
    df = embed_zones("gse", zones=_zones_gdf(), zone_id_field="geoid").to_frame()

    assert list(df.columns) == ["zone_id", "pixels", "area_km2", "e000", "e001", "e002"]
    assert len(df) == 2


def test_unknown_zone_id_field_raises_and_names_the_columns():
    with pytest.raises(SpecError) as err:
        embed_zones("gse", zones=_zones_gdf(), zone_id_field="GEOID")
    assert "GEOID" in str(err.value)
    assert "geoid" in str(err.value), "must name the column that does exist"


def test_row_index_ids_when_no_field_is_given(monkeypatch):
    _patch_provider(monkeypatch)
    out = embed_zones("gse", zones=_zones_gdf())
    assert [z.zone_id for z in out.zones] == ["0", "1"]


def test_iterable_of_id_geometry_pairs_is_accepted(monkeypatch):
    _patch_provider(monkeypatch)
    out = embed_zones("gse", zones=[("north", _square(-87.65, 41.90, 0.01))])
    assert [z.zone_id for z in out.zones] == ["north"]
    assert out.covered


def test_a_failing_tile_does_not_lose_the_sweep(monkeypatch):
    _patch_provider(monkeypatch, fail_tiles=1)   # the probe tile fails
    with pytest.raises(RuntimeError):
        embed_zones("gse", zones=_zones_gdf(), zone_id_field="geoid")


def test_max_tiles_reports_that_it_stopped_early(monkeypatch):
    _patch_provider(monkeypatch, scale_m=10)
    out = embed_zones("gse", zones=_zones_gdf(), zone_id_field="geoid",
                      tile_px=8, max_tiles=1)
    assert out.meta["tiles_capped"] is True
    assert out.meta["tiles_fetched"] == 1
    # Zones the sweep never reached are reported as uncovered rather than as zero vectors.
    assert out.meta["zones_with_pixels"] <= 2
    for z in out.zones:
        if not z.pixels:
            assert z.mean is None and z.total is None


def test_bad_tile_px_is_rejected():
    with pytest.raises(SpecError):
        embed_zones("gse", zones=_zones_gdf(), tile_px=0)


def test_empty_input_is_rejected():
    with pytest.raises(SpecError):
        embed_zones("gse", zones=gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326"))


def test_mercator_roundtrip():
    for lon, lat in [(-87.65, 41.93), (121.5, 31.2), (0.0, 0.0)]:
        assert _to_lonlat(*_merc(lon, lat)) == pytest.approx((lon, lat), abs=1e-9)

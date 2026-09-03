"""`max_tiles`: what it bounds, and when a sweep is honestly truncated.

`tiles_capped` used to be `planned > max_tiles` — a comparison of the CAP against the whole
bounding grid. Most of a bounding box is empty and skipped for free, so a sweep that fetched
every tile its zones touched still reported itself truncated. These pin the accounting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rs_embed.zones import embed_zones

gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")
from shapely.geometry import box  # noqa: E402

# Coarse tiles over a wide, mostly-empty bounding box: the grid is ~90 cells but only a
# handful hold a zone. That gap between `planned` and `needed` is the whole subject here.
TILE_PX = 800
SCALE_M = 10


def _far_apart_zones():
    """Two small squares at opposite corners, so most of the bounding grid is empty."""
    return gpd.GeoDataFrame(
        {"geoid": ["A", "B"]},
        geometry=[box(-87.65, 41.90, -87.64, 41.91), box(-87.05, 42.40, -87.04, 42.41)],
        crs="EPSG:4326",
    )


class _StubEmbedding:
    def __init__(self, data, meta):
        self.data, self.meta = data, meta


def _merc(lon, lat):
    R = 6378137.0
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _patch_provider(monkeypatch, *, dims=2):
    calls = {"n": 0}

    def fake(model, *, spatial, temporal=None, output=None, backend="auto", **kw):
        calls["n"] += 1
        x0, y0 = _merc(spatial.minlon, spatial.minlat)
        x1, y1 = _merc(spatial.maxlon, spatial.maxlat)
        w = max(1, round((x1 - x0) / SCALE_M))
        h = max(1, round((y1 - y0) / SCALE_M))
        return _StubEmbedding(
            np.full((dims, h, w), 1.0, dtype=np.float32),
            {"scale_m": SCALE_M, "bands": tuple(f"A{i}" for i in range(dims))},
        )

    monkeypatch.setattr("rs_embed.api.get_embedding", fake)
    return calls


def _sweep(monkeypatch, **kw):
    _patch_provider(monkeypatch)
    return embed_zones("gse", zones=_far_apart_zones(), zone_id_field="geoid",
                       tile_px=TILE_PX, **kw).meta


def test_tiles_needed_counts_only_the_cells_a_zone_touches(monkeypatch):
    m = _sweep(monkeypatch)
    # The grid is mostly empty; `planned` counts it all, `needed` counts the real work.
    assert m["tiles_needed"] < m["tiles_planned"]
    assert m["tiles_fetched"] == m["tiles_needed"]


def test_an_uncapped_sweep_is_never_reported_as_capped(monkeypatch):
    m = _sweep(monkeypatch, max_tiles=None)
    assert m["tiles_capped"] is False
    assert m["tiles_skipped_by_cap"] == 0


def test_a_cap_larger_than_the_work_does_not_report_truncation(monkeypatch):
    """The fix. Old behaviour compared the cap against `planned` and cried wolf here."""
    m = _sweep(monkeypatch, max_tiles=20)
    assert m["tiles_planned"] > 20, "fixture must have a grid LARGER than the cap"
    assert m["tiles_needed"] <= 20, "fixture must need FEWER tiles than the cap"
    # Every tile any zone touches was fetched, so nothing was lost and nothing is claimed.
    assert m["tiles_capped"] is False
    assert m["tiles_skipped_by_cap"] == 0
    assert m["tiles_fetched"] == m["tiles_needed"]


def test_a_cap_that_bites_reports_exactly_how_many_tiles_it_dropped(monkeypatch):
    m = _sweep(monkeypatch, max_tiles=1)
    assert m["tiles_capped"] is True
    assert m["tiles_fetched"] == 1
    assert m["tiles_skipped_by_cap"] == m["tiles_needed"] - 1
    assert m["tiles_skipped_by_cap"] >= 1


def test_max_tiles_zero_fetches_nothing_rather_than_everything(monkeypatch):
    """0 is a cap of zero. It was falsy upstream and silently meant 'uncapped'."""
    m = _sweep(monkeypatch, max_tiles=0)
    assert m["tiles_fetched"] == 0
    assert m["tiles_capped"] is True
    assert m["tiles_skipped_by_cap"] == m["tiles_needed"]
    assert m["zones_with_pixels"] == 0

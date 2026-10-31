"""Pixel-limit BBox splitting reproduces the single-request grid exactly (#134).

The fake ``sampleRectangle`` below samples the EPSG:3857 lattice the way GEE
does for ``reproject(crs="EPSG:3857", scale=s).clip(region)``, and each
pixel's two "bands" are its global (col, row) on that lattice, so a
duplicated, dropped or padded pixel shows up exactly. GEE's boundary rule is
not something a mock can pin down (``test_gee_bbox_split_live.py`` checks it),
so both plausible rules are covered: pixel-center inclusion, and rounding the
region outward to whole pixels.
"""

import math

import numpy as np
import pytest

from rs_embed.core.specs import BBox, OutputSpec, SensorSpec, TemporalSpec
from rs_embed.embedders import precomputed_gse_annual as gse_mod
from rs_embed.providers import gee_utils
from rs_embed.providers.gee import GEEProvider
from rs_embed.providers.gee_utils import (
    _SPLIT_EDGE_NUDGE_PX,
    _fetch_with_bbox_fallback,
    _split_bbox_for_recursive_fetch,
)
from rs_embed.tools.projection import web_mercator_bounds

# Keeps lattice indices positive and well away from any fill value.
_OFFSET = 1_000_000.0

_ALASKA = BBox(minlon=-170.0, minlat=52.0, maxlon=-130.0, maxlat=72.0)

# (bbox, scale_m, pixel limit): Alaska is the reported ROI at GEE's real limit;
# the rest use a small limit so many recursive splits stay cheap.
_CASES = [
    pytest.param(_ALASKA, 4000, 262144, id="alaska-4000m"),
    pytest.param(
        BBox(minlon=-88.5, minlat=40.0, maxlon=-88.0, maxlat=40.4), 100, 4096, id="midlat"
    ),
    pytest.param(BBox(minlon=10.0, minlat=-0.3, maxlon=10.4, maxlat=0.2), 250, 4096, id="equator"),
    pytest.param(BBox(minlon=20.0, minlat=69.0, maxlon=21.0, maxlat=69.5), 200, 4096, id="arctic"),
]


def _pixel_range(lo: float, hi: float, s: float, rule: str) -> tuple[int, int]:
    if rule == "center":
        return math.ceil(lo / s - 0.5), math.ceil(hi / s - 0.5)
    return math.floor(lo / s), math.ceil(hi / s)


class _LatticeSampler:
    def __init__(self, rule: str, limit: float = math.inf):
        self.rule = rule
        self.limit = limit
        self.ok = 0
        self.failed = 0

    def __call__(self, spatial, scale_m: int) -> np.ndarray:
        minx, miny, maxx, maxy = web_mercator_bounds(spatial)
        s = float(scale_m)
        c0, c1 = _pixel_range(minx, maxx, s, self.rule)
        r0, r1 = _pixel_range(-maxy, -miny, s, self.rule)
        n = (r1 - r0) * (c1 - c0)
        if n > self.limit:
            self.failed += 1
            raise RuntimeError(
                f"Image.sampleRectangle: Too many pixels in sample; must be <= {self.limit}. Got {n}."
            )
        self.ok += 1
        cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        return np.stack([cols, rows]).astype(np.float32) + _OFFSET


def _fake_provider(sampler: _LatticeSampler) -> GEEProvider:
    provider = GEEProvider.__new__(GEEProvider)  # skip ee initialization

    def _fetch_all_bands_impl(*, spatial, scale_m, **_kw):
        return sampler(spatial, scale_m), ("col", "row")

    provider._fetch_all_bands_impl = _fetch_all_bands_impl
    return provider


@pytest.mark.parametrize("rule", ["center", "outward"])
@pytest.mark.parametrize("presplit", [True, False], ids=["presplit", "split-on-error"])
@pytest.mark.parametrize("bbox,scale_m,limit", _CASES)
def test_split_fetch_matches_single_request(monkeypatch, bbox, scale_m, limit, rule, presplit):
    # presplit: the provider knows GEE's limit and splits before asking.
    # split-on-error: it thinks everything fits, so only GEE's error drives splits.
    monkeypatch.setattr(gee_utils, "_GEE_SAMPLE_RECT_MAX_PIXELS", limit if presplit else 10**12)
    sampler = _LatticeSampler(rule, limit)

    arr, names = _fake_provider(sampler).fetch_collection_patch_all_bands_chw(
        spatial=bbox,
        temporal=None,
        collection="FAKE",
        scale_m=scale_m,
        fill_value=-9999.0,
        composite="mosaic",
    )

    expected = _LatticeSampler(rule)(bbox, scale_m)
    assert names == ("col", "row")
    assert sampler.ok > 1, "case must actually exercise splitting"
    np.testing.assert_array_equal(arr, expected)


def test_presplit_avoids_failing_requests():
    sampler = _LatticeSampler("center", 262144)

    arr, _ = _fake_provider(sampler).fetch_collection_patch_all_bands_chw(
        spatial=_ALASKA,
        temporal=None,
        collection="FAKE",
        scale_m=4000,
        fill_value=-9999.0,
        composite="mosaic",
    )

    np.testing.assert_array_equal(arr, _LatticeSampler("center")(_ALASKA, 4000))
    # 1238 x 1113 px needs at least 6 requests under 262144; none may fail.
    assert sampler.failed == 0
    assert sampler.ok <= 8


@pytest.mark.parametrize("rule", ["center", "outward"])
def test_array_fetch_fallback_matches_single_request(monkeypatch, rule):
    monkeypatch.setattr(gee_utils, "_GEE_SAMPLE_RECT_MAX_PIXELS", 4096)
    bbox = BBox(minlon=-88.5, minlat=40.0, maxlon=-88.0, maxlat=40.4)
    sampler = _LatticeSampler(rule, 4096)

    arr = _fetch_with_bbox_fallback(spatial=bbox, scale_m=100, fetch_fn=lambda sp: sampler(sp, 100))

    np.testing.assert_array_equal(arr, _LatticeSampler(rule)(bbox, 100))


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("scale_m", [10, 4000])
def test_split_edges_sit_just_inside_a_lattice_line(axis, scale_m):
    a, b, got_axis = _split_bbox_for_recursive_fetch(_ALASKA, scale_m=scale_m, prefer_axis=axis)
    assert got_axis == axis

    s = float(scale_m)
    nudge = s * _SPLIT_EDGE_NUDGE_PX
    pminx, pminy, pmaxx, pmaxy = web_mercator_bounds(_ALASKA)
    ax0, ay0, ax1, ay1 = web_mercator_bounds(a)
    bx0, by0, bx1, by1 = web_mercator_bounds(b)
    if axis == "x":  # a = west, b = east
        a_cut, b_cut, lo, hi = ax1, bx0, pminx, pmaxx
        assert (ax0, ay0, ay1) == (pminx, pminy, pmaxy)
        assert (bx1, by0, by1) == (pmaxx, pminy, pmaxy)
    else:  # a = north, b = south
        a_cut, b_cut, lo, hi = ay0, by1, pminy, pmaxy
        assert (ay1, ax0, ax1) == (pmaxy, pminx, pmaxx)
        assert (by0, bx0, bx1) == (pminy, pminx, pmaxx)

    edge = round(0.5 * (a_cut + b_cut) / s) * s
    assert abs(edge - 0.5 * (lo + hi)) <= 0.5 * s
    tol = nudge / 100.0
    if axis == "x":
        assert math.isclose(a_cut, edge - nudge, abs_tol=tol)
        assert math.isclose(b_cut, edge + nudge, abs_tol=tol)
    else:
        assert math.isclose(a_cut, edge + nudge, abs_tol=tol)
        assert math.isclose(b_cut, edge - nudge, abs_tol=tol)


def test_gse_large_high_latitude_roi_is_not_duplicated():
    """#134: GSE used to tile this ROI itself and return a 2x2 duplicated mosaic."""
    sampler = _LatticeSampler("center", 262144)
    embedder = gse_mod.GSEAnnualEmbedder()
    embedder.model_name = "gse"
    embedder._get_provider = lambda _backend: _fake_provider(sampler)

    emb = embedder.get_embedding(
        spatial=_ALASKA,
        temporal=TemporalSpec.year(2024),
        sensor=SensorSpec(collection="GSE", bands=(), scale_m=4000),
        output=OutputSpec.grid(),
        backend="gee",
    )

    np.testing.assert_array_equal(np.asarray(emb.data), _LatticeSampler("center")(_ALASKA, 4000))

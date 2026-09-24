"""The package's common raster grid, and the CRS helpers that put data on it.

Every provider-backed model samples imagery on one grid: **EPSG:3857 (Web
Mercator) at the request's ``scale_m``**, with pixel edges on the lattice
anchored at the projection origin — the grid ``ee.Projection("EPSG:3857")
.atScale(scale_m)`` produces. Data that arrives in another CRS (Tessera's UTM
tiles, a user-provided raster) is resampled onto that same grid by
:func:`resample_to_common_grid`, so grid outputs for one ROI line up
pixel-for-pixel across models instead of each product keeping its own frame.

The Web Mercator formulas live here once; :mod:`rs_embed.tools.spatial`
(fetch-square) and :mod:`rs_embed.providers.gee` (request regions) build on
them, so the three never drift apart.

UTM zones are the one place where "reproject to the common grid" hides a real
seam: a ROI that straddles a zone boundary is assembled from rasters in two
different projections, and distortion grows past the zone edge.
:func:`warn_if_utm_boundary` surfaces that instead of stitching silently.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine

from ..core.errors import SpecError
from ..core.specs import BBox, PointBuffer, SpatialSpec

__all__ = [
    "COMMON_CRS",
    "CommonGrid",
    "common_grid",
    "crs_label",
    "lonlat_to_web_mercator",
    "project_pixel_centers",
    "raster_bounds_4326",
    "resample_to_common_grid",
    "source_pixel_indices",
    "utm_zone_of_crs",
    "utm_zones_spanned",
    "warn_if_utm_boundary",
    "web_mercator_bounds",
    "web_mercator_to_lonlat",
]

COMMON_CRS = "EPSG:3857"

WEB_MERCATOR_R = 6378137.0
WEB_MERCATOR_MAX_LAT = 85.05112878

# Guard against lattice snapping growing the grid by one pixel over float fuzz.
_LATTICE_EPS = 1e-9


# ── Web Mercator (spherical) ─────────────────────────────────────────────────


def _clamp_lat(lat_deg: float) -> float:
    return max(-WEB_MERCATOR_MAX_LAT, min(WEB_MERCATOR_MAX_LAT, float(lat_deg)))


def lonlat_to_web_mercator(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """EPSG:4326 lon/lat degrees → EPSG:3857 meters (latitude clamped to the valid range)."""
    lon = math.radians(float(lon_deg))
    lat = math.radians(_clamp_lat(lat_deg))
    x = WEB_MERCATOR_R * lon
    y = WEB_MERCATOR_R * math.log(math.tan((math.pi / 4.0) + (lat / 2.0)))
    return (float(x), float(y))


def web_mercator_to_lonlat(x_m: float, y_m: float) -> tuple[float, float]:
    """EPSG:3857 meters → EPSG:4326 lon/lat degrees."""
    lon = math.degrees(float(x_m) / WEB_MERCATOR_R)
    lat = math.degrees((2.0 * math.atan(math.exp(float(y_m) / WEB_MERCATOR_R))) - (math.pi / 2.0))
    return (float(lon), _clamp_lat(lat))


def web_mercator_bounds(spatial: SpatialSpec) -> tuple[float, float, float, float]:
    """ROI → ``(minx, miny, maxx, maxy)`` in EPSG:3857.

    A ``BBox`` maps its lon/lat corners; a ``PointBuffer`` is the square of
    half-side ``buffer_m`` around the projected center. This is the rectangle
    every provider request is sampled over, so anything else that claims to
    cover the same ROI derives its extent from here.
    """
    if isinstance(spatial, PointBuffer):
        spatial.validate()
        x, y = lonlat_to_web_mercator(spatial.lon, spatial.lat)
        half = float(spatial.buffer_m)
        return (x - half, y - half, x + half, y + half)
    if isinstance(spatial, BBox):
        spatial.validate()
        minx, miny = lonlat_to_web_mercator(spatial.minlon, spatial.minlat)
        maxx, maxy = lonlat_to_web_mercator(spatial.maxlon, spatial.maxlat)
        return (minx, miny, maxx, maxy)
    raise SpecError(f"Unsupported spatial type: {type(spatial)}")


# ── The common grid ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CommonGrid:
    """A north-up EPSG:3857 pixel grid at ``scale_m`` meters per pixel.

    ``transform`` is the usual raster affine (pixel ``(col, row)`` →
    projected ``(x, y)`` of the pixel's top-left corner), so
    ``transform * (0, 0)`` is the grid's north-west corner.
    """

    transform: Affine
    height: int
    width: int

    @property
    def crs(self) -> str:
        return COMMON_CRS

    @property
    def scale_m(self) -> float:
        return float(self.transform.a)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.height), int(self.width))

    def bounds(self) -> tuple[float, float, float, float]:
        """``(minx, miny, maxx, maxy)`` in EPSG:3857."""
        x0, y0 = self.transform * (0, 0)
        x1, y1 = self.transform * (self.width, self.height)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def bounds_4326(self) -> BBox:
        """The grid's footprint as an EPSG:4326 ``BBox``."""
        minx, miny, maxx, maxy = self.bounds()
        minlon, minlat = web_mercator_to_lonlat(minx, miny)
        maxlon, maxlat = web_mercator_to_lonlat(maxx, maxy)
        return BBox(minlon=minlon, minlat=minlat, maxlon=maxlon, maxlat=maxlat)

    def meta(self) -> dict[str, Any]:
        """Provenance keys describing this grid, for embedding ``meta``."""
        return {
            "output_crs": self.crs,
            "scale_m": self.scale_m,
            "transform": self.transform,
            "grid_hw": self.shape,
        }


def common_grid(spatial: SpatialSpec, *, scale_m: float) -> CommonGrid:
    """The common grid covering *spatial* at *scale_m*.

    The ROI's EPSG:3857 rectangle is snapped **outward** to the pixel lattice
    anchored at the projection origin, so two requests at the same scale share
    pixel edges wherever they overlap, and a request always covers its ROI.
    """
    s = float(scale_m)
    if not (s > 0.0):
        raise SpecError(f"scale_m must be positive, got {scale_m!r}.")
    minx, miny, maxx, maxy = web_mercator_bounds(spatial)
    col0 = math.floor(minx / s + _LATTICE_EPS)
    col1 = math.ceil(maxx / s - _LATTICE_EPS)
    row0 = math.floor(-maxy / s + _LATTICE_EPS)
    row1 = math.ceil(-miny / s - _LATTICE_EPS)
    width = max(1, col1 - col0)
    height = max(1, row1 - row0)
    transform = Affine(s, 0.0, col0 * s, 0.0, -s, -row0 * s)
    return CommonGrid(transform=transform, height=height, width=width)


# ── Resampling onto the common grid ──────────────────────────────────────────


def crs_label(crs: Any) -> str:
    """A stable, comparable string for a CRS given in any pyproj-accepted form."""
    from pyproj import CRS

    try:
        return CRS.from_user_input(crs).to_string()
    except Exception as e:
        raise SpecError(f"Unrecognized CRS {crs!r}: {e}") from e


def project_pixel_centers(grid: CommonGrid, crs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of *grid*'s pixel centers expressed in *crs*, as two ``[H, W]`` arrays."""
    t = grid.transform
    xs = t.c + (np.arange(grid.width, dtype=np.float64) + 0.5) * t.a
    ys = t.f + (np.arange(grid.height, dtype=np.float64) + 0.5) * t.e
    gx, gy = np.meshgrid(xs, ys)
    if crs_label(crs) == COMMON_CRS:
        return gx, gy
    from pyproj import Transformer

    tfm = Transformer.from_crs(COMMON_CRS, crs, always_xy=True)
    sx, sy = tfm.transform(gx.ravel(), gy.ravel())
    return (
        np.asarray(sx, dtype=np.float64).reshape(gx.shape),
        np.asarray(sy, dtype=np.float64).reshape(gy.shape),
    )


def source_pixel_indices(
    src_transform: Affine,
    x: np.ndarray,
    y: np.ndarray,
    src_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest source pixel for each ``(x, y)`` in the source CRS.

    Returns ``(rows, cols, inside)``; ``inside`` marks the positions that fall
    within the ``src_hw`` raster (the others hold clipped, meaningless indices).
    """
    inv = ~Affine(*src_transform[:6])
    col_f = inv.a * x + inv.b * y + inv.c
    row_f = inv.d * x + inv.e * y + inv.f
    h, w = int(src_hw[0]), int(src_hw[1])
    finite = np.isfinite(col_f) & np.isfinite(row_f)
    cols = np.floor(np.where(finite, col_f, -1.0)).astype(np.int64)
    rows = np.floor(np.where(finite, row_f, -1.0)).astype(np.int64)
    inside = finite & (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return rows, cols, inside


def resample_to_common_grid(
    array: np.ndarray,
    *,
    src_crs: Any,
    src_transform: Affine,
    grid: CommonGrid,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Nearest-neighbour resample a ``[..., H, W]`` raster onto *grid*.

    Nearest neighbour keeps every output pixel an actual source value — the
    right choice for embeddings and for raw sensor units alike (no invented
    intermediate values). Output pixels the source does not cover are set to
    *fill_value*.
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim < 2:
        raise SpecError(f"Expected a raster with spatial last two dims, got shape={arr.shape}.")
    x, y = project_pixel_centers(grid, src_crs)
    rows, cols, inside = source_pixel_indices(src_transform, x, y, arr.shape[-2:])
    out = np.full(arr.shape[:-2] + grid.shape, float(fill_value), dtype=np.float32)
    out[..., inside] = arr[..., rows[inside], cols[inside]]
    return out


def raster_bounds_4326(crs: Any, transform: Affine, hw: tuple[int, int]) -> BBox:
    """Footprint of a georeferenced raster as an EPSG:4326 ``BBox``.

    The four corners are projected to lon/lat and boxed; for a raster whose
    edges bow in the target projection the box is the outer hull of the
    corners, which is what the common grid needs to cover it.
    """
    from pyproj import Transformer

    t = Affine(*transform[:6])
    h, w = int(hw[0]), int(hw[1])
    corners = [t * (0, 0), t * (w, 0), t * (0, h), t * (w, h)]
    xs, ys = zip(*corners, strict=True)
    tfm = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = tfm.transform(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    return BBox(
        minlon=float(np.min(lons)),
        minlat=float(np.min(lats)),
        maxlon=float(np.max(lons)),
        maxlat=float(np.max(lats)),
    )


# ── UTM zone boundaries ──────────────────────────────────────────────────────

_UTM_EPSG_RE = re.compile(r"^EPSG:32[67](\d{2})$")


def utm_zone_of_crs(crs: Any) -> int | None:
    """UTM zone number for an ``EPSG:326xx`` / ``EPSG:327xx`` CRS, else ``None``."""
    m = _UTM_EPSG_RE.match(crs_label(crs))
    if m is None:
        return None
    zone = int(m.group(1))
    return zone if 1 <= zone <= 60 else None


def utm_zones_spanned(bbox: BBox) -> set[int]:
    """Nominal 6° UTM zones a lon/lat box touches (ignores the Norway/Svalbard exceptions)."""

    def _zone(lon: float) -> int:
        return max(1, min(60, int(math.floor((float(lon) + 180.0) / 6.0)) + 1))

    return set(range(_zone(bbox.minlon), _zone(bbox.maxlon) + 1))


def warn_if_utm_boundary(
    *,
    crss: Iterable[Any],
    footprint: BBox | None,
    context: str,
) -> bool:
    """Warn when UTM-projected data for one ROI straddles a zone boundary.

    Triggers when the rasters involved come from more than one UTM zone, or
    when a UTM raster's footprint extends into a neighbouring zone. Returns
    whether a warning was issued. Non-UTM CRSs never trigger.
    """
    zones = {z for z in (utm_zone_of_crs(c) for c in crss) if z is not None}
    if not zones:
        return False
    if footprint is not None:
        zones |= utm_zones_spanned(footprint)
    if len(zones) <= 1:
        return False
    warnings.warn(
        f"{context}: the ROI straddles a UTM zone boundary (zones {sorted(zones)}). "
        f"Data from different zones is resampled onto the common {COMMON_CRS} grid "
        "independently, so expect a resampling seam along the boundary and growing "
        "UTM distortion past a zone's edge.",
        UserWarning,
        stacklevel=3,
    )
    return True

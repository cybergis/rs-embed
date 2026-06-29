"""Expand a rectangular ROI to a square fetch window (the "fetch-square" strategy).

Square-input encoders (ViT-style models whose 2-D positional encoding needs a
square token grid) cannot take a non-square ROI. Two ways to make the input
square:

- **pad** the fetched rectangle to square — cheap, but the padded border is
  synthetic (reflected) data, which blurs object boundaries near the pad seam;
- **fetch a larger square** of *real* imagery centered on the ROI, encode that,
  then crop the token grid back to the ROI — no synthetic pixels, real spatial
  context on every side, boundaries stay crisp.

This module implements the second approach's geometry. :func:`square_spatial`
takes a :class:`SpatialSpec` and returns an enlarged **square** spec plus the
ROI's normalized window within it (``(y0, y1, x0, x1)`` in ``[0, 1]``, the same
convention as :mod:`rs_embed.tools.shape`), so callers fetch the square, encode,
and crop the output back to the ROI with ``shape.crop_grid_to_roi`` /
``shape.roi_token_box``.

The square is built in **EPSG:3857** — the projection GEE samples in — so the
enlarged box maps to square *pixels*, matching the provider exactly. A
:class:`PointBuffer` is already a centered square, so it is returned unchanged
with a full window. If the enlargement would fall outside valid lon/lat bounds
(near the poles / antimeridian), we fall back to the original spec and a full
window, leaving any squaring to the downstream pad path.
"""

from __future__ import annotations

import math

from ..core.specs import BBox, PointBuffer, SpatialSpec

__all__ = ["FULL_WINDOW", "square_spatial"]

FULL_WINDOW: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)

# Web Mercator (EPSG:3857) constants — must match providers.gee_utils so the
# enlarged square lines up with how GEE samples pixels.
_WEB_MERCATOR_R = 6378137.0
_WEB_MERCATOR_MAX_LAT = 85.05112878


def _clamp_lat(lat_deg: float) -> float:
    return max(-_WEB_MERCATOR_MAX_LAT, min(_WEB_MERCATOR_MAX_LAT, float(lat_deg)))


def _to_mercator(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    lon = math.radians(float(lon_deg))
    lat = math.radians(_clamp_lat(lat_deg))
    x = _WEB_MERCATOR_R * lon
    y = _WEB_MERCATOR_R * math.log(math.tan((math.pi / 4.0) + (lat / 2.0)))
    return x, y


def _to_lonlat(x_m: float, y_m: float) -> tuple[float, float]:
    lon = math.degrees(float(x_m) / _WEB_MERCATOR_R)
    lat = math.degrees((2.0 * math.atan(math.exp(float(y_m) / _WEB_MERCATOR_R))) - (math.pi / 2.0))
    return lon, _clamp_lat(lat)


def _is_bbox(spatial: SpatialSpec) -> bool:
    return isinstance(spatial, BBox) or all(
        hasattr(spatial, a) for a in ("minlon", "minlat", "maxlon", "maxlat")
    )


def square_spatial(
    spatial: SpatialSpec, *, tol_frac: float = 1e-4
) -> tuple[SpatialSpec, tuple[float, float, float, float]]:
    """Enlarge a rectangular ``BBox`` to a centered square in EPSG:3857.

    Returns ``(square_spatial, roi_window)``. ``roi_window`` is the ROI's
    normalized ``(y0, y1, x0, x1)`` position inside the returned square, ready
    for :func:`rs_embed.tools.shape.crop_grid_to_roi`. The window's y-axis is
    image order (row 0 = north); because the enlargement is centered the window
    is symmetric, so the y-direction convention does not matter.

    Non-``BBox`` specs (e.g. ``PointBuffer``) are already square and returned
    unchanged with :data:`FULL_WINDOW`. ``tol_frac`` treats near-square boxes as
    already square (no enlargement) to avoid pointless 1-pixel growth.
    """
    if isinstance(spatial, PointBuffer) or not _is_bbox(spatial):
        return spatial, FULL_WINDOW

    minlon, minlat = float(spatial.minlon), float(spatial.minlat)
    maxlon, maxlat = float(spatial.maxlon), float(spatial.maxlat)
    x0, y0 = _to_mercator(minlon, minlat)
    x1, y1 = _to_mercator(maxlon, maxlat)
    w, h = abs(x1 - x0), abs(y1 - y0)
    if w <= 0.0 or h <= 0.0:
        return spatial, FULL_WINDOW

    side = max(w, h)
    if abs(w - h) <= tol_frac * side:
        return spatial, FULL_WINDOW  # already (near) square

    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = side / 2.0
    nminlon, nminlat = _to_lonlat(cx - half, cy - half)
    nmaxlon, nmaxlat = _to_lonlat(cx + half, cy + half)

    square = BBox(
        minlon=nminlon,
        minlat=nminlat,
        maxlon=nmaxlon,
        maxlat=nmaxlat,
        crs=str(getattr(spatial, "crs", "EPSG:4326")),
    )
    try:
        square.validate()
    except Exception:
        # Enlargement ran past valid lon/lat bounds (pole/antimeridian) — leave
        # the ROI as-is and let the downstream pad path handle squaring.
        return spatial, FULL_WINDOW

    # ROI window within the centered square (symmetric on each enlarged axis).
    x0f = (1.0 - w / side) / 2.0
    y0f = (1.0 - h / side) / 2.0
    roi = (y0f, y0f + h / side, x0f, x0f + w / side)
    return square, tuple(round(float(v), 6) for v in roi)

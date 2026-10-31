from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.errors import ModelError, ProviderError, SpecError
from ..core.specs import BBox, SensorSpec, SpatialSpec, TemporalSpec
from ..providers.base import ProviderBase
from ..tools.normalization import normalize_input_chw
from ..tools.projection import (
    COMMON_CRS,
    common_grid,
    web_mercator_bounds,
    web_mercator_to_lonlat,
)
from ..tools.temporal import split_date_range as _split_date_range_core

_GEE_SAMPLE_RECT_TOO_MANY_PIXELS = "Too many pixels in sample"
_GEE_SAMPLE_RECT_OP = "sampleRectangle"
_GEE_SAMPLE_RECT_MUST_BE = "must be <="
_MAX_GEE_BBOX_SPLIT_DEPTH = 12
# GEE's ceiling for one sampleRectangle call; larger BBoxes are split up front.
_GEE_SAMPLE_RECT_MAX_PIXELS = 262144
# How far (in pixels) a split child's cut edge sits inside its own side of the
# lattice line: far above lon/lat round-trip error, far below half a pixel.
_SPLIT_EDGE_NUDGE_PX = 1e-6


def _iter_exception_messages(exc: BaseException) -> tuple[str, ...]:
    msgs: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    depth = 0
    while cur is not None and id(cur) not in seen and depth < 6:
        seen.add(id(cur))
        try:
            msgs.append(str(cur))
        except Exception as _e:
            pass
        try:
            msgs.append(repr(cur))
        except Exception as _e:
            pass
        try:
            for a in getattr(cur, "args", ()) or ():
                msgs.append(str(a))
        except Exception as _e:
            pass
        nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        cur = nxt if isinstance(nxt, BaseException) else None
        depth += 1
    return tuple(m for m in msgs if m)


def _looks_like_gee_sample_too_many_pixels(exc: BaseException) -> bool:
    msgs = " | ".join(_iter_exception_messages(exc))
    if _GEE_SAMPLE_RECT_TOO_MANY_PIXELS not in msgs:
        return False
    return (_GEE_SAMPLE_RECT_OP in msgs) or (_GEE_SAMPLE_RECT_MUST_BE in msgs)


def _looks_like_bbox_spatial(spatial: SpatialSpec) -> bool:
    if isinstance(spatial, BBox):
        return True
    needed = ("minlon", "minlat", "maxlon", "maxlat")
    return all(hasattr(spatial, k) for k in needed) and (
        getattr(spatial, "crs", "EPSG:4326") == "EPSG:4326"
    )


def _coerce_bbox_like(spatial: SpatialSpec) -> BBox:
    if isinstance(spatial, BBox):
        return spatial
    if not _looks_like_bbox_spatial(spatial):
        raise ModelError(f"Expected BBox-like spatial for GEE fallback, got {type(spatial)}")
    return BBox(
        minlon=float(spatial.minlon),
        minlat=float(spatial.minlat),
        maxlon=float(spatial.maxlon),
        maxlat=float(spatial.maxlat),
        crs=str(getattr(spatial, "crs", "EPSG:4326")),
    )


def _validated_mid(candidate: float, lo: float, hi: float, coord_name: str) -> float:
    mid = min(max(candidate, lo), hi)
    if not (lo < mid < hi):
        mid = 0.5 * (lo + hi)
    if not (lo < mid < hi):
        raise ModelError(
            f"Failed to split BBox along {coord_name} for GEE sampleRectangle fallback."
        )
    return mid


def _lattice_split_edge(lo: float, hi: float, *, scale_m: float, nudge: float) -> float:
    """The EPSG:3857 pixel-lattice line nearest the middle of ``(lo, hi)``.

    Falls back to the plain middle when no lattice line lies inside the span
    (a sub-pixel span, which never trips the pixel limit in practice).
    """
    mid = 0.5 * (lo + hi)
    edge = round(mid / scale_m) * scale_m
    return edge if (lo + nudge) < edge < (hi - nudge) else mid


def _split_bbox_for_recursive_fetch(
    bbox: BBox, *, scale_m: int, prefer_axis: str
) -> tuple[BBox, BBox, str]:
    """Split *bbox* in two along a pixel edge of the common EPSG:3857 grid.

    The cut sits on the ``k * scale_m`` lattice every GEE request samples on,
    and each child's cut edge is pulled ``_SPLIT_EDGE_NUDGE_PX`` of a pixel
    inside its own side. Every pixel therefore lies wholly within exactly one
    child, whether GEE assigns boundary pixels by center, by overlap, or by
    rounding the region outward, so the children concatenate to the
    single-request grid with no overlap or gap. Outer edges are untouched.
    """
    minx, miny, maxx, maxy = web_mercator_bounds(bbox)
    dx, dy = maxx - minx, maxy - miny
    s = float(scale_m)
    nudge = s * _SPLIT_EDGE_NUDGE_PX

    axis = str(prefer_axis).lower()
    if axis not in {"x", "y"}:
        axis = "x" if dx >= dy else "y"
    if axis == "x" and dx <= 0.0 and dy > 0.0:
        axis = "y"
    if axis == "y" and dy <= 0.0 and dx > 0.0:
        axis = "x"
    if dx <= 0.0 and dy <= 0.0:
        raise ModelError(
            "Cannot split degenerate BBox while handling GEE sampleRectangle pixel-limit error."
        )

    lo_lon, lo_lat = float(bbox.minlon), float(bbox.minlat)
    hi_lon, hi_lat = float(bbox.maxlon), float(bbox.maxlat)
    if axis == "x":
        edge = _lattice_split_edge(minx, maxx, scale_m=s, nudge=nudge)
        west_max, _ = web_mercator_to_lonlat(edge - nudge, miny)
        east_min, _ = web_mercator_to_lonlat(edge + nudge, miny)
        west = BBox(
            minlon=lo_lon,
            minlat=lo_lat,
            maxlon=_validated_mid(west_max, lo_lon, hi_lon, "longitude"),
            maxlat=hi_lat,
            crs=bbox.crs,
        )
        east = BBox(
            minlon=_validated_mid(east_min, lo_lon, hi_lon, "longitude"),
            minlat=lo_lat,
            maxlon=hi_lon,
            maxlat=hi_lat,
            crs=bbox.crs,
        )
        return (west, east, "x")

    edge = _lattice_split_edge(miny, maxy, scale_m=s, nudge=nudge)
    _, north_min = web_mercator_to_lonlat(minx, edge + nudge)
    _, south_max = web_mercator_to_lonlat(minx, edge - nudge)
    north = BBox(
        minlon=lo_lon,
        minlat=_validated_mid(north_min, lo_lat, hi_lat, "latitude"),
        maxlon=hi_lon,
        maxlat=hi_lat,
        crs=bbox.crs,
    )
    south = BBox(
        minlon=lo_lon,
        minlat=lo_lat,
        maxlon=hi_lon,
        maxlat=_validated_mid(south_max, lo_lat, hi_lat, "latitude"),
        crs=bbox.crs,
    )
    return (north, south, "y")


# ── Band alias tables ────────────────────────────────────────────────────────

_ALIAS_S2 = {
    "BLUE": "B2",
    "GREEN": "B3",
    "RED": "B4",
    "NIR": "B8",
    "NIR_BROAD": "B8",
    "NIR_WIDE": "B8",
    "NIR_NARROW": "B8A",
    "NIRN": "B8A",
    "NIRNARROW": "B8A",
    "NIR_N": "B8A",
    "RE1": "B5",
    "RED_EDGE_1": "B5",
    "RE2": "B6",
    "RED_EDGE_2": "B6",
    "RE3": "B7",
    "RED_EDGE_3": "B7",
    "RE4": "B8A",
    "RED_EDGE_4": "B8A",
    "SWIR1": "B11",
    "SWIR_1": "B11",
    "SWIR2": "B12",
    "SWIR_2": "B12",
}
_ALIAS_LS89_SR = {
    "BLUE": "SR_B2",
    "GREEN": "SR_B3",
    "RED": "SR_B4",
    "NIR": "SR_B5",
    "SWIR1": "SR_B6",
    "SWIR2": "SR_B7",
}
_ALIAS_LS457_SR = {
    "BLUE": "SR_B1",
    "GREEN": "SR_B2",
    "RED": "SR_B3",
    "NIR": "SR_B4",
    "SWIR1": "SR_B5",
    "SWIR2": "SR_B7",
}

_NO_IMAGES_FOUND_MSG = "No images found for the selected region/time window."


# ── GEE auth / init helpers ───────────────────────────────────────────────────


def _gee_init_kwargs(project: str | None) -> dict[str, str]:
    return {"project": project} if project else {}


def _gee_error_message(exc: Exception) -> str:
    msg = str(exc).strip()
    low = msg.lower()
    if any(
        token in low
        for token in (
            "quota project",
            "cloud project",
            "project is required",
            "user project",
            "quota_project_id",
            "no project",
        )
    ):
        return (
            "Earth Engine requires a Google Cloud project. Pass "
            "project='your-gcp-project-id', set EE_PROJECT or "
            "GOOGLE_CLOUD_PROJECT, or configure a default project with "
            "`earthengine set_project your-gcp-project-id` or "
            "`gcloud auth application-default set-quota-project your-gcp-project-id`."
        )
    if any(
        token in low
        for token in (
            "authenticate",
            "authorization",
            "credentials",
            "credential",
            "login required",
            "invalid_grant",
            "refresh token",
            "reauth",
        )
    ):
        return (
            "Earth Engine authentication is required. Run `earthengine authenticate` and try again."
        )
    detail = msg or repr(exc)
    return f"Failed to initialize GEE: {detail}"


# ── Band alias resolution ─────────────────────────────────────────────────────


def resolve_band_aliases(collection: str, bands: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve human-friendly band aliases to collection-native band names.

    Pure lookup over static alias tables (no provider access); unknown names
    and unknown collections pass through unchanged.
    """
    if not bands:
        return bands
    c = (collection or "").upper()
    if "COPERNICUS/S2" in c:
        amap = _ALIAS_S2
    elif "LANDSAT/LC08/C02/T1_L2" in c or "LANDSAT/LC09/C02/T1_L2" in c:
        amap = _ALIAS_LS89_SR
    elif (
        "LANDSAT/LE07/C02/T1_L2" in c
        or "LANDSAT/LT05/C02/T1_L2" in c
        or "LANDSAT/LT04/C02/T1_L2" in c
    ):
        amap = _ALIAS_LS457_SR
    else:
        amap = {}
    return tuple(amap.get((b or "").upper(), b) for b in bands)


# Backwards-compatible alias kept for existing imports/tests.
_resolve_band_aliases = resolve_band_aliases


# ── Cloud-cover filtering ─────────────────────────────────────────────────────

# Scene-level cloud-cover property per collection family. GEE property filters
# EXCLUDE images that lack the property, so applying the S2 property to any
# other collection silently filters out every image.
_CLOUD_PROPERTY_BY_COLLECTION: tuple[tuple[str, str], ...] = (
    ("COPERNICUS/S2", "CLOUDY_PIXEL_PERCENTAGE"),
    ("LANDSAT/", "CLOUD_COVER"),
)


def _cloud_property_for_collection(collection: str) -> str | None:
    c = (collection or "").upper()
    for marker, prop in _CLOUD_PROPERTY_BY_COLLECTION:
        if marker in c:
            return prop
    return None


def _filter_clouds(col: Any, *, collection: str, cloudy_pct: int | None) -> Any:
    """Apply the collection's scene cloud-cover filter (inclusive threshold).

    Collections without a known cloud-cover property (SAR, DEMs, ...) are
    returned unfiltered rather than emptied by a filter on a missing property.
    """
    if cloudy_pct is None:
        return col
    prop = _cloud_property_for_collection(collection)
    if prop is None:
        return col
    import ee

    return col.filter(ee.Filter.lte(prop, int(cloudy_pct)))


# ── Temporal split helper ─────────────────────────────────────────────────────


def _split_date_range(start: str, end: str, n_parts: int) -> tuple[tuple[str, str], ...]:
    try:
        return _split_date_range_core(start, end, n_parts)
    except SpecError as e:
        raise ProviderError(str(e)) from e


# ── Collection utilities ──────────────────────────────────────────────────────


def _no_images_found_message(*, collection: str | None = None) -> str:
    if collection:
        return f"{_NO_IMAGES_FOUND_MSG} collection={collection!r}"
    return _NO_IMAGES_FOUND_MSG


def _collection_size_or_none(col: Any) -> int | None:
    try:
        return int(col.size().getInfo())
    except Exception:
        return None


def _raise_if_empty_collection(col: Any, *, collection: str | None = None) -> None:
    if _collection_size_or_none(col) == 0:
        raise ProviderError(_no_images_found_message(collection=collection))


def _order_collection_for_mosaic(col: Any) -> Any:
    """Stabilize mosaic priority by preferring chronological collection order."""
    try:
        return col.sort("system:time_start")
    except Exception:
        return col


def _format_s1_empty_collection_message(
    *,
    collection_id: str,
    temporal: TemporalSpec,
    counts: dict[str, int | None],
    require_iw: bool,
    relax_iw_on_empty: bool,
) -> str:
    detail_parts = [
        f"base(date+bounds)={counts.get('base')}",
        f"iw={counts.get('iw')}",
        f"vv={counts.get('vv')}",
        f"vh={counts.get('vh')}",
    ]
    if "vh_no_iw" in counts:
        detail_parts.append(f"vh_no_iw={counts.get('vh_no_iw')}")
    details = ", ".join(detail_parts)
    return (
        f"{_NO_IMAGES_FOUND_MSG} collection={collection_id!r} "
        f"time=({temporal.start!r}, {temporal.end!r}) "
        f"filters=[{details}]. "
        f"requested_iw={require_iw} relax_iw_on_empty={relax_iw_on_empty}. "
        "This request needs dual-pol (VV/VH) S1 scenes; try a wider time window or a different AOI."
    )


def _build_s1_dualpol_collection(base: Any, *, require_iw: bool) -> Any:
    import ee

    col = base
    if require_iw:
        col = col.filter(ee.Filter.eq("instrumentMode", "IW"))
    col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    return col


def _sample_image_bands_raw_chw(
    img: Any,
    *,
    region: Any,
    bands: Sequence[str],
    scale_m: int,
    fill_value: float,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Sample a GEE image as **north-up** CHW float32.

    *value_range*, when given, clips the sampled values — callers pass the
    collection's valid data range (e.g. ``(0, 10000)`` for S2 DN). ``None``
    (the default) returns values unclipped, so collections with other ranges
    (Landsat SR, signed products) are not silently destroyed.

    Row-order contract (verified against live GEE with ``ee.Image.pixelLonLat``
    across hemispheres and scales; see ``tests/test_gee_orientation_live.py``):
    ``sampleRectangle`` row order depends on the *reprojection form*, not on
    clipping —

    - ``reproject(crs=..., scale=...)`` → north-up, with or without ``.clip()``
      (this function and ``_fetch_all_bands_impl``; no flip applied).
    - ``reproject(ee.Projection(...).atScale(...))`` → south-up
      (``fetch_array_chw``, which applies ``_flip_sample_tile_y``).
    """
    img = img.select(list(bands)).reproject(crs=COMMON_CRS, scale=int(scale_m))
    rect = img.sampleRectangle(region=region, defaultValue=float(fill_value)).getInfo()
    props = rect.get("properties", {}) if isinstance(rect, dict) else {}
    if not props:
        raise ProviderError(_NO_IMAGES_FOUND_MSG)
    arrs = [np.array(props.get(str(b), []), dtype=np.float32) for b in bands]
    try:
        raw = np.stack(arrs, axis=0).astype(np.float32)
    except Exception as e:
        raise ProviderError("Failed to sample rectangle from GEE image.") from e
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if value_range is not None:
        raw = np.clip(raw, float(value_range[0]), float(value_range[1]))
    return raw.astype(np.float32)


# ── Tile fetch helpers ────────────────────────────────────────────────────────


def _flip_sample_tile_y(arr: np.ndarray) -> np.ndarray:
    """Flip the penultimate (height) axis of a CHW or TCHW array from south-up to north-up.

    Called inside ``GEEProvider.fetch_array_chw``, whose
    ``reproject(ee.Projection(...).atScale(...))`` form makes ``sampleRectangle``
    return south-up rows. The ``reproject(crs=..., scale=...)`` form returns
    north-up rows (clipping does not affect row order), so
    ``_sample_image_bands_raw_chw`` and ``_fetch_all_bands_impl`` must NOT apply
    this flip. Contract verified against live GEE; see
    ``_sample_image_bands_raw_chw`` and ``tests/test_gee_orientation_live.py``.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim < 2:
        raise ModelError(f"Expected fetched tile with spatial last2 dims, got shape={a.shape}")
    return np.flip(a, axis=a.ndim - 2).astype(np.float32, copy=False)


def _exceeds_sample_pixel_limit(bbox: BBox, *, scale_m: int) -> bool:
    try:
        h, w = common_grid(bbox, scale_m=float(scale_m)).shape
    except SpecError:
        # Invalid bounds: let fetch_fn report them through the provider's own error path.
        return False
    return h * w > _GEE_SAMPLE_RECT_MAX_PIXELS


def _bbox_recursive_fallback(
    *,
    spatial: SpatialSpec,
    scale_m: int,
    fetch_fn: Any,
    stitch_fn: Any,
    _depth: int = 0,
) -> Any:
    """Generic recursive BBox splitting for GEE's sampleRectangle pixel limit.

    A BBox whose common-grid footprint already exceeds the limit is split up
    front; anything else is fetched with fetch_fn(spatial) and split only if
    GEE still reports a pixel-limit error. Splits follow pixel-lattice lines
    (see :func:`_split_bbox_for_recursive_fetch`), so
    stitch_fn(result_a, result_b, axis) only has to concatenate.
    """
    cause: BaseException | None = None
    is_bbox = _looks_like_bbox_spatial(spatial)
    if not (is_bbox and _exceeds_sample_pixel_limit(_coerce_bbox_like(spatial), scale_m=scale_m)):
        try:
            return fetch_fn(spatial)
        except Exception as e:
            if not (is_bbox and _looks_like_gee_sample_too_many_pixels(e)):
                raise
            cause = e
    if _depth >= _MAX_GEE_BBOX_SPLIT_DEPTH:
        raise ModelError(
            "GEE sampleRectangle pixel-limit fallback exceeded max recursive BBox splits "
            f"({_MAX_GEE_BBOX_SPLIT_DEPTH})."
        ) from cause
    spatial_bbox = _coerce_bbox_like(spatial)
    grid_h, grid_w = common_grid(spatial_bbox, scale_m=float(scale_m)).shape
    a_sp, b_sp, axis = _split_bbox_for_recursive_fetch(
        spatial_bbox, scale_m=scale_m, prefer_axis="x" if grid_w >= grid_h else "y"
    )
    kw: dict[str, Any] = dict(
        scale_m=scale_m,
        fetch_fn=fetch_fn,
        stitch_fn=stitch_fn,
        _depth=_depth + 1,
    )
    return stitch_fn(
        _bbox_recursive_fallback(spatial=a_sp, **kw),
        _bbox_recursive_fallback(spatial=b_sp, **kw),
        axis,
    )


def _fetch_with_bbox_fallback(
    *,
    spatial: SpatialSpec,
    scale_m: int,
    fetch_fn: Any,
    split_depth: int = 0,
) -> np.ndarray:
    """BBox-splitting fallback for fetch functions returning np.ndarray."""
    return _bbox_recursive_fallback(
        spatial=spatial,
        scale_m=scale_m,
        fetch_fn=lambda sp: np.asarray(fetch_fn(sp), dtype=np.float32),
        stitch_fn=_stitch_bbox_split_arrays,
        _depth=split_depth,
    )


def _fetch_provider_array_chw_with_bbox_fallback(
    provider: ProviderBase,
    *,
    image: Any,
    spatial: SpatialSpec,
    bands: tuple[str, ...],
    scale_m: int,
    fill_value: float,
    collection: str | None,
) -> np.ndarray:
    def _do(sp: SpatialSpec) -> np.ndarray:
        region = provider.get_region(sp)
        # fetch_array_chw owns its orientation contract: always returns north-up.
        return np.asarray(
            provider.fetch_array_chw(
                image=image,
                bands=bands,
                region=region,
                scale_m=int(scale_m),
                fill_value=float(fill_value),
                collection=collection,
            ),
            dtype=np.float32,
        )

    return _fetch_with_bbox_fallback(
        spatial=spatial,
        scale_m=int(scale_m),
        fetch_fn=_do,
    )


def _stitch_bbox_split_arrays(arr_a: Any, arr_b: Any, axis: str) -> np.ndarray:
    """Join two bbox-split arrays along a spatial axis.

    The recursive splitter returns children in spatial order:
    - ``axis="x"``: west, east
    - ``axis="y"``: north, south

    Children are cut on pixel-lattice lines, so they share no pixels and leave
    no gap: a plain concatenation reproduces the single-request grid.
    """
    arr_a = np.asarray(arr_a, dtype=np.float32)
    arr_b = np.asarray(arr_b, dtype=np.float32)
    if arr_a.ndim < 2 or arr_b.ndim < 2:
        raise ModelError(
            f"Expected arrays with spatial last2 dims for bbox stitching, got {arr_a.shape} and {arr_b.shape}"
        )
    if tuple(arr_a.shape[:-2]) != tuple(arr_b.shape[:-2]):
        raise ModelError(
            f"Leading shape mismatch while stitching bbox tiles: {arr_a.shape} vs {arr_b.shape}"
        )

    axis = str(axis).lower()
    if axis not in {"x", "y"}:
        raise ModelError(f"Invalid bbox fallback stitch axis={axis!r}")
    split_axis = arr_a.ndim - 1 if axis == "x" else arr_a.ndim - 2
    nonsplit_axis = arr_a.ndim - 2 if axis == "x" else arr_a.ndim - 1

    if int(arr_a.shape[nonsplit_axis]) != int(arr_b.shape[nonsplit_axis]):
        raise ModelError(
            f"Non-split spatial dim mismatch while stitching bbox tiles: {arr_a.shape} vs {arr_b.shape}"
        )
    return np.concatenate([arr_a, arr_b], axis=split_axis).astype(np.float32, copy=False)


def fetch_provider_patch_raw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    sensor: SensorSpec,
    to_float_image: bool = False,
) -> np.ndarray:
    region = provider.get_region(spatial)
    img = provider.build_image(sensor=sensor, temporal=temporal, region=region)
    if bool(to_float_image):
        try:
            img = img.toFloat()
        except Exception as _e:
            pass
    x = _fetch_provider_array_chw_with_bbox_fallback(
        provider,
        image=img,
        spatial=spatial,
        bands=tuple(sensor.bands),
        scale_m=int(sensor.scale_m),
        fill_value=float(sensor.fill_value),
        collection=sensor.collection,
    )
    x = normalize_input_chw(
        x,
        expected_channels=len(sensor.bands),
        name=f"gee_input[{sensor.collection}]",
    )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


# Backwards-compatible alias kept for existing imports/tests.
fetch_gee_patch_raw = fetch_provider_patch_raw

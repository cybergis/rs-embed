from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.errors import ModelError
from ..core.specs import BBox, SensorSpec, SpatialSpec, TemporalSpec
from ..providers.base import ProviderBase
from ..tools.normalization import normalize_input_chw
from ..tools.serialization import jsonable as _jsonable

_WEB_MERCATOR_R = 6378137.0
_WEB_MERCATOR_MAX_LAT = 85.05112878
_GEE_SAMPLE_RECT_TOO_MANY_PIXELS = "Too many pixels in sample"
_GEE_SAMPLE_RECT_OP = "sampleRectangle"
_GEE_SAMPLE_RECT_MUST_BE = "must be <="
_MAX_GEE_BBOX_SPLIT_DEPTH = 12
_GEE_BBOX_STITCH_LEN_TOLERANCE_PX = 4

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

def _clamp_lat_for_web_mercator(lat_deg: float) -> float:
    return max(-_WEB_MERCATOR_MAX_LAT, min(_WEB_MERCATOR_MAX_LAT, float(lat_deg)))

def _lonlat_to_web_mercator_xy(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    lon = math.radians(float(lon_deg))
    lat = math.radians(_clamp_lat_for_web_mercator(lat_deg))
    x = _WEB_MERCATOR_R * lon
    y = _WEB_MERCATOR_R * math.log(math.tan((math.pi / 4.0) + (lat / 2.0)))
    return (float(x), float(y))

def _web_mercator_xy_to_lonlat(x_m: float, y_m: float) -> tuple[float, float]:
    lon = math.degrees(float(x_m) / _WEB_MERCATOR_R)
    lat = math.degrees((2.0 * math.atan(math.exp(float(y_m) / _WEB_MERCATOR_R))) - (math.pi / 2.0))
    lat = _clamp_lat_for_web_mercator(lat)
    return (float(lon), float(lat))

def _bbox_span_pixels_estimate(bbox: BBox, *, scale_m: int) -> tuple[int, int]:
    x0, y0 = _lonlat_to_web_mercator_xy(bbox.minlon, bbox.minlat)
    x1, y1 = _lonlat_to_web_mercator_xy(bbox.maxlon, bbox.maxlat)
    s = max(1.0, float(scale_m))
    w = max(1, int(math.ceil(abs(x1 - x0) / s)))
    h = max(1, int(math.ceil(abs(y1 - y0) / s)))
    return (h, w)

def _split_bbox_for_recursive_fetch(bbox: BBox, *, prefer_axis: str) -> tuple[BBox, BBox, str]:
    x0, y0 = _lonlat_to_web_mercator_xy(bbox.minlon, bbox.minlat)
    x1, y1 = _lonlat_to_web_mercator_xy(bbox.maxlon, bbox.maxlat)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

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

    if axis == "x":
        xm = 0.5 * (x0 + x1)
        lon_mid, _ = _web_mercator_xy_to_lonlat(xm, 0.5 * (y0 + y1))
        lon_mid = min(max(lon_mid, float(bbox.minlon)), float(bbox.maxlon))
        if not (float(bbox.minlon) < lon_mid < float(bbox.maxlon)):
            lon_mid = 0.5 * (float(bbox.minlon) + float(bbox.maxlon))
        if not (float(bbox.minlon) < lon_mid < float(bbox.maxlon)):
            raise ModelError(
                "Failed to split BBox along longitude for GEE sampleRectangle fallback."
            )
        west = BBox(
            minlon=float(bbox.minlon),
            minlat=float(bbox.minlat),
            maxlon=float(lon_mid),
            maxlat=float(bbox.maxlat),
            crs=bbox.crs,
        )
        east = BBox(
            minlon=float(lon_mid),
            minlat=float(bbox.minlat),
            maxlon=float(bbox.maxlon),
            maxlat=float(bbox.maxlat),
            crs=bbox.crs,
        )
        return (west, east, "x")

    ym = 0.5 * (y0 + y1)
    _, lat_mid = _web_mercator_xy_to_lonlat(0.5 * (x0 + x1), ym)
    lat_mid = min(max(lat_mid, float(bbox.minlat)), float(bbox.maxlat))
    if not (float(bbox.minlat) < lat_mid < float(bbox.maxlat)):
        lat_mid = 0.5 * (float(bbox.minlat) + float(bbox.maxlat))
    if not (float(bbox.minlat) < lat_mid < float(bbox.maxlat)):
        raise ModelError("Failed to split BBox along latitude for GEE sampleRectangle fallback.")

    north = BBox(
        minlon=float(bbox.minlon),
        minlat=float(lat_mid),
        maxlon=float(bbox.maxlon),
        maxlat=float(bbox.maxlat),
        crs=bbox.crs,
    )
    south = BBox(
        minlon=float(bbox.minlon),
        minlat=float(bbox.minlat),
        maxlon=float(bbox.maxlon),
        maxlat=float(lat_mid),
        crs=bbox.crs,
    )
    return (north, south, "y")

def _flip_sample_tile_y(arr: np.ndarray) -> np.ndarray:
    """Normalize a fetched tile to north-up row order once at the leaf fetch."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim < 2:
        raise ModelError(f"Expected fetched tile with spatial last2 dims, got shape={a.shape}")
    return np.flip(a, axis=a.ndim - 2).astype(np.float32, copy=False)

def _fetch_provider_array_chw_with_bbox_fallback(
    provider: ProviderBase,
    *,
    image: Any,
    spatial: SpatialSpec,
    bands: tuple[str, ...],
    scale_m: int,
    fill_value: float,
    collection: str | None,
    split_depth: int = 0,
) -> np.ndarray:
    region = provider.get_region(spatial)
    try:
        arr = provider.fetch_array_chw(
            image=image,
            bands=bands,
            region=region,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            collection=collection,
        )
        return _flip_sample_tile_y(
            np.asarray(arr, dtype=np.float32)
        )  # _flip_sample_tile_y(np.asarray(arr, dtype=np.float32))
    except Exception as e:
        if not (_looks_like_gee_sample_too_many_pixels(e) and _looks_like_bbox_spatial(spatial)):
            raise
        if int(split_depth) >= _MAX_GEE_BBOX_SPLIT_DEPTH:
            raise ModelError(
                "GEE sampleRectangle pixel-limit fallback exceeded max recursive BBox splits "
                f"({_MAX_GEE_BBOX_SPLIT_DEPTH})."
            ) from e

        spatial_bbox = _coerce_bbox_like(spatial)
        h_est, w_est = _bbox_span_pixels_estimate(spatial_bbox, scale_m=int(scale_m))
        prefer_axis = "x" if int(w_est) >= int(h_est) else "y"
        a_sp, b_sp, axis = _split_bbox_for_recursive_fetch(spatial_bbox, prefer_axis=prefer_axis)

        arr_a = _fetch_provider_array_chw_with_bbox_fallback(
            provider,
            image=image,
            spatial=a_sp,
            bands=bands,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            collection=collection,
            split_depth=int(split_depth) + 1,
        )
        arr_b = _fetch_provider_array_chw_with_bbox_fallback(
            provider,
            image=image,
            spatial=b_sp,
            bands=bands,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            collection=collection,
            split_depth=int(split_depth) + 1,
        )

        return _stitch_bbox_split_arrays(
            arr_a=arr_a,
            arr_b=arr_b,
            parent_spatial=spatial_bbox,
            axis=axis,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
        )

def _stitch_bbox_split_arrays(
    *,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    parent_spatial: Any,
    axis: str,
    scale_m: int,
    fill_value: float,
) -> np.ndarray:
    """Stitch two bbox-split arrays along a spatial axis.

    The recursive splitter returns children in spatial order:
    - ``axis="x"``: west, east
    - ``axis="y"``: north, south
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

    spatial_bbox = _coerce_bbox_like(parent_spatial)
    axis = str(axis).lower()
    split_axis = arr_a.ndim - 1 if axis == "x" else arr_a.ndim - 2
    nonsplit_axis = arr_a.ndim - 2 if axis == "x" else arr_a.ndim - 1

    if int(arr_a.shape[nonsplit_axis]) != int(arr_b.shape[nonsplit_axis]):
        raise ModelError(
            f"Non-split spatial dim mismatch while stitching bbox tiles: {arr_a.shape} vs {arr_b.shape}"
        )

    if axis not in {"x", "y"}:
        raise ModelError(f"Invalid bbox fallback stitch axis={axis!r}")

    target_h, target_w = _bbox_span_pixels_estimate(spatial_bbox, scale_m=int(scale_m))
    target_len = int(target_w if axis == "x" else target_h)
    len_a = int(arr_a.shape[split_axis])
    len_b = int(arr_b.shape[split_axis])
    combined_len = int(len_a + len_b)
    delta = int(combined_len - target_len)

    if delta > 0:
        if delta > _GEE_BBOX_STITCH_LEN_TOLERANCE_PX:
            raise ModelError(
                "Excessive overlap while stitching bbox tiles: "
                f"combined={combined_len}, target~={target_len}, delta={delta}"
            )
        trim_a = int(delta // 2)
        trim_b = int(delta - trim_a)
        if trim_a > 0:
            slicer = [slice(None)] * arr_a.ndim
            slicer[split_axis] = slice(0, max(0, len_a - trim_a))
            arr_a = arr_a[tuple(slicer)]
            len_a = int(arr_a.shape[split_axis])
        if trim_b > 0:
            slicer = [slice(None)] * arr_b.ndim
            slicer[split_axis] = slice(min(trim_b, len_b), None)
            arr_b = arr_b[tuple(slicer)]
            len_b = int(arr_b.shape[split_axis])
    elif delta < 0:
        gap = int(-delta)
        if gap > _GEE_BBOX_STITCH_LEN_TOLERANCE_PX:
            raise ModelError(
                "Excessive gap while stitching bbox tiles: "
                f"combined={combined_len}, target~={target_len}, gap={gap}"
            )
        pad_shape = list(arr_a.shape)
        pad_shape[split_axis] = gap
        gap_arr = np.full(tuple(pad_shape), float(fill_value), dtype=np.float32)
        return np.concatenate([arr_a, gap_arr, arr_b], axis=split_axis).astype(
            np.float32, copy=False
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

def inspect_input_raw(x_chw: np.ndarray, *, sensor: SensorSpec, name: str) -> dict[str, Any]:
    from ..tools.inspection import inspect_chw

    x = normalize_input_chw(
        x_chw,
        expected_channels=len(sensor.bands),
        name=name,
    )
    rep = inspect_chw(
        x,
        name=name,
        expected_channels=len(sensor.bands),
        value_range=None,
        fill_value=float(sensor.fill_value),
    )
    return {
        "ok": bool(rep.get("ok", False)),
        "report": rep,
        "sensor": _jsonable(sensor),
    }

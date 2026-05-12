from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.errors import ModelError, ProviderError, SpecError
from ..core.specs import BBox, SensorSpec, SpatialSpec, TemporalSpec
from ..providers.base import ProviderBase
from ..tools.normalization import normalize_input_chw
from ..tools.temporal import split_date_range as _split_date_range_core

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


def _validated_mid(candidate: float, lo: float, hi: float, coord_name: str) -> float:
    mid = min(max(candidate, lo), hi)
    if not (lo < mid < hi):
        mid = 0.5 * (lo + hi)
    if not (lo < mid < hi):
        raise ModelError(
            f"Failed to split BBox along {coord_name} for GEE sampleRectangle fallback."
        )
    return mid


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
    dx, dy = abs(x1 - x0), abs(y1 - y0)

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

    xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if axis == "x":
        lon_raw, _ = _web_mercator_xy_to_lonlat(xm, ym)
        lon_mid = _validated_mid(lon_raw, float(bbox.minlon), float(bbox.maxlon), "longitude")
        west = BBox(
            minlon=float(bbox.minlon),
            minlat=float(bbox.minlat),
            maxlon=lon_mid,
            maxlat=float(bbox.maxlat),
            crs=bbox.crs,
        )
        east = BBox(
            minlon=lon_mid,
            minlat=float(bbox.minlat),
            maxlon=float(bbox.maxlon),
            maxlat=float(bbox.maxlat),
            crs=bbox.crs,
        )
        return (west, east, "x")

    _, lat_raw = _web_mercator_xy_to_lonlat(xm, ym)
    lat_mid = _validated_mid(lat_raw, float(bbox.minlat), float(bbox.maxlat), "latitude")
    north = BBox(
        minlon=float(bbox.minlon),
        minlat=lat_mid,
        maxlon=float(bbox.maxlon),
        maxlat=float(bbox.maxlat),
        crs=bbox.crs,
    )
    south = BBox(
        minlon=float(bbox.minlon),
        minlat=float(bbox.minlat),
        maxlon=float(bbox.maxlon),
        maxlat=lat_mid,
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


def _resolve_band_aliases(collection: str, bands: tuple[str, ...]) -> tuple[str, ...]:
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
        "TerraFM S1 expects dual-pol VV/VH input; try a wider time window or a different AOI."
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
) -> np.ndarray:
    img = img.select(list(bands)).reproject(crs="EPSG:3857", scale=int(scale_m))
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
    return np.clip(raw, 0.0, 10000.0).astype(np.float32)


# ── Tile fetch helpers ────────────────────────────────────────────────────────


def _flip_sample_tile_y(arr: np.ndarray) -> np.ndarray:
    """Normalize a fetched tile to north-up row order once at the leaf fetch."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim < 2:
        raise ModelError(f"Expected fetched tile with spatial last2 dims, got shape={a.shape}")
    return np.flip(a, axis=a.ndim - 2).astype(np.float32, copy=False)


def _bbox_recursive_fallback(
    *,
    spatial: SpatialSpec,
    scale_m: int,
    fill_value: float,
    fetch_fn: Any,
    stitch_fn: Any,
    _depth: int = 0,
) -> Any:
    """Generic recursive BBox-splitting fallback for GEE pixel-limit errors.

    Calls fetch_fn(spatial); on a pixel-limit error splits the bbox, recurses,
    then calls stitch_fn(result_a, result_b, bbox, axis, scale_m, fill_value).
    """
    try:
        return fetch_fn(spatial)
    except Exception as e:
        if not (_looks_like_gee_sample_too_many_pixels(e) and _looks_like_bbox_spatial(spatial)):
            raise
        if _depth >= _MAX_GEE_BBOX_SPLIT_DEPTH:
            raise ModelError(
                "GEE sampleRectangle pixel-limit fallback exceeded max recursive BBox splits "
                f"({_MAX_GEE_BBOX_SPLIT_DEPTH})."
            ) from e
        spatial_bbox = _coerce_bbox_like(spatial)
        h_est, w_est = _bbox_span_pixels_estimate(spatial_bbox, scale_m=scale_m)
        prefer_axis = "x" if w_est >= h_est else "y"
        a_sp, b_sp, axis = _split_bbox_for_recursive_fetch(spatial_bbox, prefer_axis=prefer_axis)
        kw: dict[str, Any] = dict(
            scale_m=scale_m,
            fill_value=fill_value,
            fetch_fn=fetch_fn,
            stitch_fn=stitch_fn,
            _depth=_depth + 1,
        )
        return stitch_fn(
            _bbox_recursive_fallback(spatial=a_sp, **kw),
            _bbox_recursive_fallback(spatial=b_sp, **kw),
            spatial_bbox,
            axis,
            scale_m,
            fill_value,
        )


def _fetch_with_bbox_fallback(
    *,
    spatial: SpatialSpec,
    scale_m: int,
    fill_value: float,
    fetch_fn: Any,
    split_depth: int = 0,
) -> np.ndarray:
    """BBox-splitting fallback for fetch functions returning np.ndarray."""

    def _stitch(
        a: Any, b: Any, bbox: Any, axis: str, scale_m: int, fill_value: float
    ) -> np.ndarray:
        return _stitch_bbox_split_arrays(
            arr_a=np.asarray(a, dtype=np.float32),
            arr_b=np.asarray(b, dtype=np.float32),
            parent_spatial=bbox,
            axis=axis,
            scale_m=scale_m,
            fill_value=fill_value,
        )

    return _bbox_recursive_fallback(
        spatial=spatial,
        scale_m=scale_m,
        fill_value=fill_value,
        fetch_fn=lambda sp: np.asarray(fetch_fn(sp), dtype=np.float32),
        stitch_fn=_stitch,
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
        arr = provider.fetch_array_chw(
            image=image,
            bands=bands,
            region=region,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            collection=collection,
        )
        return _flip_sample_tile_y(np.asarray(arr, dtype=np.float32))

    return _fetch_with_bbox_fallback(
        spatial=spatial,
        scale_m=int(scale_m),
        fill_value=float(fill_value),
        fetch_fn=_do,
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

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar

import numpy as np

from ..core.errors import ModelError
from ..core.specs import SensorSpec, SpatialSpec, TemporalSpec
from ..providers import get_provider, has_provider, list_providers
from ..providers.base import ProviderBase

_T = TypeVar("_T")


from ..internal.api.api_helpers import normalize_backend_name


def default_provider_backend_name() -> Optional[str]:
    configured = normalize_backend_name(os.environ.get("RS_EMBED_DEFAULT_PROVIDER", ""))
    if configured:
        return configured if has_provider(configured) else None
    providers = list_providers()
    if not providers:
        return None
    if "gee" in providers:
        return "gee"
    return str(providers[0]).strip().lower()


def resolve_provider_backend_name(
    backend: str,
    *,
    allow_auto: bool = True,
    auto_backend: Optional[str] = None,
) -> Optional[str]:
    b = normalize_backend_name(backend)
    if allow_auto and b == "auto":
        resolved_auto = (
            normalize_backend_name(auto_backend)
            if auto_backend is not None
            else default_provider_backend_name()
        )
        if not resolved_auto:
            return None
        b = resolved_auto
    if has_provider(b):
        return b
    return None


def is_provider_backend(
    backend: str,
    *,
    allow_auto: bool = True,
    auto_backend: Optional[str] = None,
) -> bool:
    return (
        resolve_provider_backend_name(
            backend,
            allow_auto=allow_auto,
            auto_backend=auto_backend,
        )
        is not None
    )


def get_cached_provider(
    provider_cache: Dict[str, ProviderBase],
    *,
    backend: str,
    allow_auto: bool = True,
    auto_backend: Optional[str] = None,
) -> ProviderBase:
    b = resolve_provider_backend_name(
        backend,
        allow_auto=allow_auto,
        auto_backend=auto_backend,
    )
    if b is None:
        raise ModelError(f"Unsupported provider backend={backend!r}.")
    p = provider_cache.get(b)
    if p is None:
        kwargs = provider_init_kwargs(b)
        p = get_provider(b, **kwargs)
        provider_cache[b] = p
    p.ensure_ready()
    return p


def provider_init_kwargs(backend: str) -> Dict[str, Any]:
    """Provider-specific constructor kwargs, centralized outside embedders."""
    b = normalize_backend_name(backend)
    if b == "gee":
        return {"auto_auth": True}
    return {}


def create_provider_for_backend(
    backend: str,
    *,
    allow_auto: bool = True,
    auto_backend: Optional[str] = None,
) -> ProviderBase:
    b = resolve_provider_backend_name(
        backend,
        allow_auto=allow_auto,
        auto_backend=auto_backend,
    )
    if b is None:
        raise ModelError(f"Unsupported provider backend={backend!r}.")
    p = get_provider(b, **provider_init_kwargs(b))
    p.ensure_ready()
    return p


def resolve_device_auto_torch(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_cached_with_device(
    cached_loader: Callable[..., _T],
    *,
    device: str,
    **kwargs: Any,
) -> Tuple[_T, str]:
    """Resolve device once and call a cached loader that accepts `dev=...`."""
    dev = resolve_device_auto_torch(device)
    loaded = cached_loader(dev=dev, **kwargs)
    return loaded, dev


def fetch_collection_patch_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    collection: str,
    bands: Tuple[str, ...],
    scale_m: int = 10,
    cloudy_pct: Optional[int] = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch a provider patch as CHW float32 using shared SensorSpec logic."""
    sensor = SensorSpec(
        collection=str(collection),
        bands=tuple(str(b) for b in bands),
        scale_m=int(scale_m),
        cloudy_pct=(int(cloudy_pct) if cloudy_pct is not None else None),  # type: ignore[arg-type]
        fill_value=float(fill_value),
        composite=str(composite),
    )
    return fetch_sensor_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
    )


def fetch_sensor_patch_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    sensor: SensorSpec,
    to_float_image: bool = False,
) -> np.ndarray:
    """Fetch a CHW patch from a concrete SensorSpec.

    Delegates to ``provider.fetch_sensor_patch_chw`` (which already validates
    ndim, channel count, and calls nan_to_num).  Errors are re-raised as
    ModelError so embedder-layer callers see a consistent exception type.
    """
    from ..core.errors import ProviderError

    try:
        return provider.fetch_sensor_patch_chw(
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            to_float_image=to_float_image,
        )
    except ProviderError as exc:
        raise ModelError(str(exc)) from exc


def _stitch_spatial_last2_arrays(
    *,
    a: np.ndarray,
    b: np.ndarray,
    parent_spatial: Any,
    axis: str,
    scale_m: int,
    fill_value: float,
) -> np.ndarray:
    from ..internal.api.api_helpers import _stitch_bbox_split_arrays

    return _stitch_bbox_split_arrays(
        arr_a=np.asarray(a, dtype=np.float32),
        arr_b=np.asarray(b, dtype=np.float32),
        parent_spatial=parent_spatial,
        axis=axis,
        scale_m=scale_m,
        fill_value=fill_value,
    )


def _fetch_spatial_array_with_bbox_fallback(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    scale_m: int,
    fill_value: float,
    fetch_fn: Callable[[SpatialSpec], np.ndarray],
    split_depth: int = 0,
) -> np.ndarray:
    from ..internal.api import api_helpers as _ah

    try:
        return np.asarray(fetch_fn(spatial), dtype=np.float32)
    except Exception as e:
        if not (
            _ah._looks_like_gee_sample_too_many_pixels(e)
            and _ah._looks_like_bbox_spatial(spatial)
        ):
            raise
        max_depth = int(getattr(_ah, "_MAX_GEE_BBOX_SPLIT_DEPTH", 12))
        if int(split_depth) >= max_depth:
            raise ModelError(
                f"GEE bbox fallback exceeded max recursive splits ({max_depth})."
            ) from e

        spatial_bbox = _ah._coerce_bbox_like(spatial)
        h_est, w_est = _ah._bbox_span_pixels_estimate(
            spatial_bbox, scale_m=int(scale_m)
        )
        prefer_axis = "x" if int(w_est) >= int(h_est) else "y"
        a_sp, b_sp, axis = _ah._split_bbox_for_recursive_fetch(
            spatial_bbox, prefer_axis=prefer_axis
        )
        arr_a = _fetch_spatial_array_with_bbox_fallback(
            provider,
            spatial=a_sp,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            fetch_fn=fetch_fn,
            split_depth=int(split_depth) + 1,
        )
        arr_b = _fetch_spatial_array_with_bbox_fallback(
            provider,
            spatial=b_sp,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            fetch_fn=fetch_fn,
            split_depth=int(split_depth) + 1,
        )
        return _stitch_spatial_last2_arrays(
            a=arr_a,
            b=arr_b,
            parent_spatial=spatial_bbox,
            axis=axis,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
        )


def fetch_collection_patch_all_bands_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    collection: str,
    scale_m: int = 10,
    fill_value: float = 0.0,
    composite: str = "median",
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Fetch all bands for a collection with BBox fallback stitching for large GEE samples."""

    def _fetch_once(sp: SpatialSpec) -> Tuple[np.ndarray, Tuple[str, ...]]:
        arr, names = provider.fetch_collection_patch_all_bands_chw(
            spatial=sp,
            temporal=temporal,
            collection=str(collection),
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            composite=str(composite),
        )
        return np.asarray(arr, dtype=np.float32), tuple(str(b) for b in names)

    try:
        arr, names = _fetch_once(spatial)
        return np.asarray(arr, dtype=np.float32), tuple(names)
    except Exception as e:
        from ..internal.api import api_helpers as _ah

        if not (
            _ah._looks_like_gee_sample_too_many_pixels(e)
            and _ah._looks_like_bbox_spatial(spatial)
        ):
            raise

        def _rec(sp: SpatialSpec, depth: int = 0) -> Tuple[np.ndarray, Tuple[str, ...]]:
            max_depth = int(getattr(_ah, "_MAX_GEE_BBOX_SPLIT_DEPTH", 12))
            try:
                return _fetch_once(sp)
            except Exception as ee:
                if not (
                    _ah._looks_like_gee_sample_too_many_pixels(ee)
                    and _ah._looks_like_bbox_spatial(sp)
                ):
                    raise
                if int(depth) >= max_depth:
                    raise ModelError(
                        f"GEE bbox fallback exceeded max recursive splits ({max_depth})."
                    ) from ee
                sp_bbox = _ah._coerce_bbox_like(sp)
                h_est, w_est = _ah._bbox_span_pixels_estimate(
                    sp_bbox, scale_m=int(scale_m)
                )
                prefer_axis = "x" if int(w_est) >= int(h_est) else "y"
                a_sp, b_sp, axis = _ah._split_bbox_for_recursive_fetch(
                    sp_bbox, prefer_axis=prefer_axis
                )
                arr_a, names_a = _rec(a_sp, depth + 1)
                arr_b, names_b = _rec(b_sp, depth + 1)
                if tuple(names_a) != tuple(names_b):
                    raise ModelError(
                        "Band names mismatch while stitching all-band bbox tiles."
                    )
                stitched = _stitch_spatial_last2_arrays(
                    a=arr_a,
                    b=arr_b,
                    parent_spatial=sp_bbox,
                    axis=axis,
                    scale_m=int(scale_m),
                    fill_value=float(fill_value),
                )
                return stitched, tuple(names_a)

        return _rec(spatial, 0)


def fetch_s2_rgb_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
) -> np.ndarray:
    """Fetch Sentinel-2 RGB as float32 CHW in [0,1]."""
    raw = fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=int(scale_m),
        cloudy_pct=int(cloudy_pct),
        composite=str(composite),
        fill_value=0.0,
    )
    return np.clip(raw / 10000.0, 0.0, 1.0).astype(np.float32)


def fetch_s1_vvvh_raw_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    orbit: Optional[str] = None,
    use_float_linear: bool = True,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch Sentinel-1 VV/VH as raw float32 CHW."""
    arr = _fetch_spatial_array_with_bbox_fallback(
        provider,
        spatial=spatial,
        scale_m=int(scale_m),
        fill_value=float(fill_value),
        fetch_fn=lambda sp: provider.fetch_s1_vvvh_raw_chw(
            spatial=sp,
            temporal=temporal,
            scale_m=int(scale_m),
            orbit=orbit,
            use_float_linear=bool(use_float_linear),
            composite=str(composite),
            fill_value=float(fill_value),
        ),
    )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ModelError(
            f"Expected S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_s1_vvvh_chw(raw_chw: np.ndarray) -> np.ndarray:
    """Convert raw S1 VV/VH to numerically stable [0,1] CHW."""
    arr = np.asarray(raw_chw, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ModelError(
            f"Expected raw S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}"
        )
    x = np.log1p(np.maximum(arr, 0.0))
    denom = np.percentile(x, 99) if np.isfinite(x).all() else 1.0
    denom = float(denom) if float(denom) > 0 else 1.0
    return np.clip(x / denom, 0.0, 1.0).astype(np.float32)


def fetch_s2_multiframe_raw_tchw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    bands: Sequence[str],
    n_frames: int = 8,
    collection: str = "COPERNICUS/S2_SR_HARMONIZED",
    scale_m: int = 10,
    cloudy_pct: Optional[int] = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch an S2 time series as raw float32 [T,C,H,W] in [0,10000]."""
    arr = _fetch_spatial_array_with_bbox_fallback(
        provider,
        spatial=spatial,
        scale_m=int(scale_m),
        fill_value=float(fill_value),
        fetch_fn=lambda sp: provider.fetch_multiframe_collection_raw_tchw(
            spatial=sp,
            temporal=temporal,
            collection=str(collection),
            bands=tuple(str(b) for b in bands),
            n_frames=int(n_frames),
            scale_m=int(scale_m),
            cloudy_pct=(int(cloudy_pct) if cloudy_pct is not None else None),
            composite=str(composite),
            fill_value=float(fill_value),
        ),
    )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise ModelError(
            f"Expected TCHW array, got shape={getattr(arr, 'shape', None)}"
        )
    if int(arr.shape[1]) != len(tuple(bands)):
        raise ModelError(
            f"Time series channel mismatch: got C={int(arr.shape[1])}, expected C={len(tuple(bands))}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def coerce_input_to_tchw(
    input_chw: np.ndarray,
    *,
    expected_channels: int,
    n_frames: int,
    model_name: str,
) -> np.ndarray:
    """Normalize user-provided CHW/TCHW into clipped float32 [T,C,H,W]."""
    raw = np.asarray(input_chw, dtype=np.float32)
    t = max(1, int(n_frames))

    if raw.ndim == 3:
        if int(raw.shape[0]) != int(expected_channels):
            raise ModelError(
                f"input_chw must be CHW with C={int(expected_channels)} for {model_name}, "
                f"got {tuple(int(v) for v in raw.shape)}"
            )
        raw_tchw = np.repeat(raw[None, ...], repeats=t, axis=0).astype(np.float32)
    elif raw.ndim == 4:
        if int(raw.shape[1]) != int(expected_channels):
            raise ModelError(
                f"input_chw must be TCHW with C={int(expected_channels)} for {model_name}, "
                f"got {tuple(int(v) for v in raw.shape)}"
            )
        raw_tchw = raw.astype(np.float32, copy=False)
        if raw_tchw.shape[0] < t:
            raw_tchw = np.concatenate(
                [raw_tchw] + [raw_tchw[-1:]] * (t - raw_tchw.shape[0]),
                axis=0,
            )
        elif raw_tchw.shape[0] > t:
            raw_tchw = raw_tchw[:t]
    else:
        raise ModelError(
            f"input_chw must be CHW (C,H,W) or TCHW (T,C,H,W) for {model_name}, "
            f"got {tuple(int(v) for v in raw.shape)}"
        )

    raw_tchw = np.nan_to_num(raw_tchw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(raw_tchw, 0.0, 10000.0).astype(np.float32)

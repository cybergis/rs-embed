"""Provider fetch helpers and satellite-data normalization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.errors import ModelError, ProviderError
from ..core.specs import SensorSpec, SpatialSpec, TemporalSpec
from .base import ProviderBase


def fetch_sensor_patch_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    sensor: SensorSpec,
    to_float_image: bool = False,
) -> np.ndarray:
    """Fetch a CHW patch from a concrete SensorSpec, re-raising ProviderError as ModelError."""
    try:
        return provider.fetch_sensor_patch_chw(
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            to_float_image=to_float_image,
        )
    except ProviderError as exc:
        raise ModelError(str(exc)) from exc


def fetch_collection_patch_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    collection: str,
    bands: tuple[str, ...],
    scale_m: int = 10,
    cloudy_pct: int | None = 30,
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


def fetch_collection_patch_all_bands_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    collection: str,
    scale_m: int = 10,
    fill_value: float = 0.0,
    composite: str = "median",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Fetch all bands for a collection as CHW float32."""
    try:
        arr, names = provider.fetch_collection_patch_all_bands_chw(
            spatial=spatial,
            temporal=temporal,
            collection=str(collection),
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            composite=str(composite),
        )
        return np.asarray(arr, dtype=np.float32), tuple(str(b) for b in names)
    except ProviderError as exc:
        raise ModelError(str(exc)) from exc


def fetch_s2_rgb_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
) -> np.ndarray:
    """Fetch Sentinel-2 RGB (B4/B3/B2) as float32 CHW in raw DN [0, 10000].

    Normalization to model input range is the caller's responsibility.
    """
    return fetch_collection_patch_chw(
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


def _require_s1_support(provider: ProviderBase, method: str) -> None:
    if not hasattr(provider, method):
        raise ModelError(
            f"Provider '{getattr(provider, 'name', type(provider).__name__)}' does not support "
            "Sentinel-1 VV/VH fetch. Use fetch_sensor_patch_chw with an S1 SensorSpec instead."
        )


def fetch_s1_vvvh_raw_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    orbit: str | None = None,
    use_float_linear: bool = True,
    composite: str = "median",
    fill_value: float = 0.0,
    require_iw: bool = True,
    relax_iw_on_empty: bool = True,
) -> np.ndarray:
    """Fetch Sentinel-1 VV/VH as raw float32 CHW."""
    _require_s1_support(provider, "fetch_s1_vvvh_raw_chw")
    try:
        arr = provider.fetch_s1_vvvh_raw_chw(  # type: ignore[attr-defined]
            spatial=spatial,
            temporal=temporal,
            scale_m=int(scale_m),
            orbit=orbit,
            use_float_linear=bool(use_float_linear),
            composite=str(composite),
            fill_value=float(fill_value),
            require_iw=bool(require_iw),
            relax_iw_on_empty=bool(relax_iw_on_empty),
        )
    except ProviderError as exc:
        raise ModelError(str(exc)) from exc
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ModelError(f"Expected S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fetch_s1_vvvh_raw_chw_with_meta(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    orbit: str | None = None,
    use_float_linear: bool = True,
    composite: str = "median",
    fill_value: float = 0.0,
    require_iw: bool = True,
    relax_iw_on_empty: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch Sentinel-1 VV/VH as raw float32 CHW together with fetch metadata."""
    _require_s1_support(provider, "fetch_s1_vvvh_raw_chw_with_meta")
    try:
        arr, meta = provider.fetch_s1_vvvh_raw_chw_with_meta(  # type: ignore[attr-defined]
            spatial=spatial,
            temporal=temporal,
            scale_m=int(scale_m),
            orbit=orbit,
            use_float_linear=bool(use_float_linear),
            composite=str(composite),
            fill_value=float(fill_value),
            require_iw=bool(require_iw),
            relax_iw_on_empty=bool(relax_iw_on_empty),
        )
    except ProviderError as exc:
        raise ModelError(str(exc)) from exc
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ModelError(f"Expected S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), dict(meta or {})


def fetch_s2_multiframe_raw_tchw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    bands: Sequence[str],
    n_frames: int = 8,
    collection: str = "COPERNICUS/S2_SR_HARMONIZED",
    scale_m: int = 10,
    cloudy_pct: int | None = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch an S2 time series as raw float32 [T,C,H,W] in [0,10000]."""
    try:
        arr = provider.fetch_multiframe_collection_raw_tchw(
            spatial=spatial,
            temporal=temporal,
            collection=str(collection),
            bands=tuple(str(b) for b in bands),
            n_frames=int(n_frames),
            scale_m=int(scale_m),
            cloudy_pct=(int(cloudy_pct) if cloudy_pct is not None else None),
            composite=str(composite),
            fill_value=float(fill_value),
            # fetch_fn=_fetch,
        )
    except ProviderError as exc:
        raise ModelError(str(exc)) from exc
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise ModelError(f"Expected TCHW array, got shape={getattr(arr, 'shape', None)}")
    if int(arr.shape[1]) != len(tuple(bands)):
        raise ModelError(
            f"Time series channel mismatch: got C={int(arr.shape[1])}, expected C={len(tuple(bands))}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _stack_binned_frames(
    frames: list[np.ndarray | None],
    bins: Sequence[tuple[str, str]],
    *,
    expected_channels: int | None,
    context: str,
    last_error: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stack per-bin frames into [T,C,H,W]; empty bins become all-NaN sentinel frames.

    The NaN sentinel keeps ``T == len(bins)`` so frame/bin alignment survives
    array-only transport (e.g. through ``input_chw`` pipelines); consumers
    detect empty frames via ``np.isnan(frame).all()`` or the returned meta.
    """
    shapes = {f.shape for f in frames if f is not None}
    if not shapes:
        detail = f" Last fetch error: {last_error}" if last_error else ""
        raise ModelError(
            f"{context}: no imagery found in any of the {len(bins)} time bins.{detail}"
        )
    if len(shapes) > 1:
        raise ModelError(
            f"{context}: inconsistent frame shapes across time bins: {sorted(shapes)}."
        )
    shape = shapes.pop()
    if expected_channels is not None and int(shape[0]) != int(expected_channels):
        raise ModelError(f"{context}: expected C={expected_channels} per frame, got shape={shape}.")

    out: list[np.ndarray] = []
    frame_meta: list[dict[str, Any]] = []
    for (start, end), f in zip(bins, frames, strict=True):
        empty = f is None
        out.append(np.full(shape, np.nan, dtype=np.float32) if empty else f.astype(np.float32))
        frame_meta.append({"start": start, "end": end, "empty": empty})
    arr = np.stack(out, axis=0)
    meta: dict[str, Any] = {
        "frames": frame_meta,
        "n_empty": int(sum(1 for m in frame_meta if m["empty"])),
    }
    return arr, meta


def fetch_collection_binned_raw_tchw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    bins: Sequence[tuple[str, str]],
    collection: str,
    bands: Sequence[str],
    scale_m: int = 10,
    cloudy_pct: int | None = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch one composite per explicit time bin as raw float32 [T,C,H,W].

    Unlike :func:`fetch_s2_multiframe_raw_tchw` (equal division of one window),
    the caller supplies the exact ``(start, end)`` date bins. Bins with no
    imagery yield all-NaN sentinel frames flagged in the returned meta instead
    of failing or duplicating neighbours; at least one bin must have data.
    """
    if not bins:
        raise ModelError("fetch_collection_binned_raw_tchw requires at least one time bin.")
    frames: list[np.ndarray | None] = []
    last_error: str | None = None
    for start, end in bins:
        try:
            frames.append(
                fetch_collection_patch_chw(
                    provider,
                    spatial=spatial,
                    temporal=TemporalSpec.range(start, end),
                    collection=str(collection),
                    bands=tuple(str(b) for b in bands),
                    scale_m=int(scale_m),
                    cloudy_pct=cloudy_pct,
                    composite=str(composite),
                    fill_value=float(fill_value),
                )
            )
        except ModelError as exc:
            last_error = str(exc)
            frames.append(None)
    return _stack_binned_frames(
        frames,
        bins,
        expected_channels=len(tuple(bands)),
        context=f"binned fetch of {collection}",
        last_error=last_error,
    )


def fetch_s1_vvvh_binned_raw_tchw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    bins: Sequence[tuple[str, str]],
    scale_m: int = 10,
    orbit: str | None = None,
    use_float_linear: bool = True,
    composite: str = "median",
    fill_value: float = 0.0,
    require_iw: bool = True,
    relax_iw_on_empty: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch one S1 VV/VH composite per explicit time bin as raw float32 [T,2,H,W].

    Reuses the single-frame S1 path per bin, so IW filtering, dual-pol
    selection, and the relaxed retry all apply within each bin. Bins with no
    S1 coverage yield all-NaN sentinel frames flagged in the returned meta;
    at least one bin must have data.
    """
    if not bins:
        raise ModelError("fetch_s1_vvvh_binned_raw_tchw requires at least one time bin.")
    _require_s1_support(provider, "fetch_s1_vvvh_raw_chw_with_meta")
    frames: list[np.ndarray | None] = []
    frame_fetch_meta: list[dict[str, Any] | None] = []
    last_error: str | None = None
    for start, end in bins:
        try:
            arr, fmeta = fetch_s1_vvvh_raw_chw_with_meta(
                provider,
                spatial=spatial,
                temporal=TemporalSpec.range(start, end),
                scale_m=int(scale_m),
                orbit=orbit,
                use_float_linear=bool(use_float_linear),
                composite=str(composite),
                fill_value=float(fill_value),
                require_iw=bool(require_iw),
                relax_iw_on_empty=bool(relax_iw_on_empty),
            )
            frames.append(arr)
            frame_fetch_meta.append(fmeta)
        except ModelError as exc:
            last_error = str(exc)
            frames.append(None)
            frame_fetch_meta.append(None)
    arr, meta = _stack_binned_frames(
        frames,
        bins,
        expected_channels=2,
        context="binned S1 VV/VH fetch",
        last_error=last_error,
    )
    for m, fm in zip(meta["frames"], frame_fetch_meta, strict=True):
        if fm:
            m["fetch"] = fm
    return arr, meta


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


def inspect_fetch_result(
    x_chw: np.ndarray,
    *,
    sensor: SensorSpec,
    name: str,
) -> dict[str, Any]:
    """Inspect a prefetched CHW (or TCHW) array and return a structured quality report."""
    from ..tools.inspection import inspect_chw
    from ..tools.normalization import normalize_input_array
    from ..tools.serialization import jsonable as _jsonable

    x = normalize_input_array(x_chw, expected_channels=len(sensor.bands), name=name)
    x_inspect = x[0] if x.ndim == 4 else x
    rep = inspect_chw(
        x_inspect,
        name=name,
        expected_channels=len(sensor.bands),
        value_range=None,
        fill_value=float(sensor.fill_value),
    )
    return {
        "ok": bool(rep.get("ok", False)),
        "report": rep,
        "sensor": _jsonable(sensor),
        "input_ndim": int(x.ndim),
        "n_frames": (int(x.shape[0]) if x.ndim == 4 else None),
    }

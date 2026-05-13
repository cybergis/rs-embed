from __future__ import annotations

import importlib
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import (
    ModelInputSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from ..core.types import FetchResult
from ..providers import ProviderBase
from ..providers.fetch import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
)
from ..providers.fetch import (
    fetch_s1_vvvh_raw_chw_with_meta as _fetch_s1_vvvh_raw_chw_with_meta,
)
from ..providers.fetch import (
    normalize_s1_vvvh_chw as _normalize_s1_vvvh_chw,
)
from ..providers.resolution import (
    is_provider_backend,
)
from ..tools.runtime import (
    load_cached_with_device as _load_cached_with_device,
)
from ..tools.runtime import (
    resolve_device_auto_torch as _resolve_device,
)
from .base import EmbedderBase
from .config import model_config_value
from .meta import build_meta, temporal_to_range


def ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise ModelError("This embedder requires torch installed.") from e


def pool_from_tokens(tokens, pooling):
    n = len(tokens)
    h2 = int((n - 1) ** 0.5)
    has_cls = n > 1 and h2 * h2 == n - 1
    patch = tokens[1:] if has_cls else tokens
    if len(patch) == 0:
        return tokens[0].astype("float32"), has_cls
    if pooling == "mean":
        return patch.mean(axis=0).astype("float32"), has_cls
    if pooling == "max":
        return patch.max(axis=0).astype("float32"), has_cls
    raise ModelError(f"Unknown pooling={pooling!r} (expected 'mean' or 'max').")


def tokens_to_grid_dhw(tokens):
    n = len(tokens)
    h2 = int((n - 1) ** 0.5)
    has_cls = n > 1 and h2 * h2 == n - 1
    patch = tokens[1:] if has_cls else tokens
    p, d = patch.shape
    hw = int(p**0.5)
    if hw * hw != p:
        raise ModelError(f"Patch token count {p} is not a perfect square.")
    return patch.reshape(hw, hw, d).transpose(2, 0, 1).astype("float32"), (hw, hw), has_cls


_S2_SR_10_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
_S1_VVVH_BANDS = ["VV", "VH"]
_THOR_MODEL_BANDS = [
    "BLUE",
    "GREEN",
    "RED",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
    "NIR_BROAD",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]
_THOR_S1_MODEL_BANDS = ["VV", "VH"]
_THOR_MODALITY_GSDS = {
    "s2": (10, 20),
    "s1": (10,),
}
_THOR_MODALITY_SENSOR_BANDS = {
    "s2": tuple(_S2_SR_10_BANDS),
    "s1": tuple(_S1_VVVH_BANDS),
}
_THOR_MODALITY_MODEL_BANDS = {
    "s2": tuple(_THOR_MODEL_BANDS),
    "s1": tuple(_THOR_S1_MODEL_BANDS),
}

_THOR_S2_MEAN = np.array(
    [
        0.176620,
        0.195923,
        0.213948,
        0.263378,
        0.300818,
        0.313144,
        0.308133,
        0.320993,
        0.221550,
        0.175772,
    ],
    dtype=np.float32,
)
_THOR_S2_STD = np.array(
    [
        0.264520,
        0.252949,
        0.259180,
        0.272771,
        0.248175,
        0.235432,
        0.226434,
        0.223274,
        0.171606,
        0.156223,
    ],
    dtype=np.float32,
)

_THOR_VARIANT_TO_MODEL_KEY = {
    "tiny": "thor_v1_tiny",
    "small": "thor_v1_small",
    "base": "thor_v1_base",
    "large": "thor_v1_large",
}


def _normalize_thor_variant(variant: Any) -> str:
    raw = str(variant).strip().lower()
    aliases = {
        "t": "tiny",
        "tiny": "tiny",
        "s": "small",
        "small": "small",
        "b": "base",
        "base": "base",
        "l": "large",
        "large": "large",
    }
    resolved = aliases.get(raw)
    if resolved is None:
        raise ModelError(
            f"Unknown THOR variant='{variant}' (expected one of: tiny, small, base, large)."
        )
    return resolved


def _normalize_thor_model_key(value: Any) -> str:
    raw = str(value).strip().lower()
    aliases = {
        "thor_v1_tiny": "thor_v1_tiny",
        "thor_1_0_tiny": "thor_v1_tiny",
        "thor_v1_small": "thor_v1_small",
        "thor_1_0_small": "thor_v1_small",
        "thor_v1_base": "thor_v1_base",
        "thor_1_0_base": "thor_v1_base",
        "thor_v1_large": "thor_v1_large",
        "thor_1_0_large": "thor_v1_large",
        "tiny": "thor_v1_tiny",
        "small": "thor_v1_small",
        "base": "thor_v1_base",
        "large": "thor_v1_large",
    }
    resolved = aliases.get(raw)
    if resolved is None:
        raise ModelError(
            f"Unknown THOR model_key='{value}' "
            "(expected one of: thor_v1_tiny, thor_v1_small, thor_v1_base, thor_v1_large)."
        )
    return resolved


def _thor_variant_from_model_key(model_key: str) -> str:
    for variant, candidate in _THOR_VARIANT_TO_MODEL_KEY.items():
        if candidate == model_key:
            return variant
    return model_key


def _thor_hf_id_from_model_key(model_key: str) -> str | None:
    variant = _thor_variant_from_model_key(model_key)
    if variant in _THOR_VARIANT_TO_MODEL_KEY:
        return f"FM4CS/THOR-1.0-{variant}"
    return None


def _resolve_thor_runtime_config(
    *,
    model_config: dict[str, Any] | None,
    default_model_key: str,
    default_image_size: int,
) -> dict[str, Any]:
    variant_v = model_config_value(model_config, "variant")
    if variant_v is not None:
        model_key = _THOR_VARIANT_TO_MODEL_KEY[_normalize_thor_variant(variant_v)]
    else:
        model_key = _normalize_thor_model_key(
            os.environ.get("RS_EMBED_THOR_MODEL_KEY", default_model_key).strip()
            or default_model_key
        )

    image_size = int(os.environ.get("RS_EMBED_THOR_IMG", str(default_image_size)))

    ckpt_path = os.environ.get("RS_EMBED_THOR_CKPT")
    ckpt_path = ckpt_path or None

    pretrained = os.environ.get("RS_EMBED_THOR_PRETRAINED", "1").strip() not in {
        "0",
        "false",
        "False",
    }

    normalize_mode = os.environ.get("RS_EMBED_THOR_NORMALIZE", "thor_stats").strip()

    group_merge = os.environ.get("RS_EMBED_THOR_GROUP_MERGE", "mean").strip().lower()

    patch_size = int(os.environ.get("RS_EMBED_THOR_PATCH_SIZE", "8"))
    resize_mode = str(os.environ.get("RS_EMBED_THOR_RESIZE_MODE", "native_snap")).strip().lower()
    shape_adjust = str(os.environ.get("RS_EMBED_THOR_SHAPE_ADJUST", "crop")).strip().lower()
    shape_tol_px = int(os.environ.get("RS_EMBED_THOR_SHAPE_TOL_PX", "8"))
    max_native_side = int(os.environ.get("RS_EMBED_THOR_MAX_NATIVE_SIDE", "384"))
    max_native_tokens = int(os.environ.get("RS_EMBED_THOR_MAX_NATIVE_TOKENS", "3000"))
    input_prep_mode = (
        str(model_config_value(model_config, "_input_prep_mode") or "").strip().lower()
    )

    if resize_mode not in {"fixed", "native_snap"}:
        raise ModelError(
            f"Unknown THOR resize mode '{resize_mode}'. Use one of: fixed, native_snap."
        )
    if shape_adjust not in {"crop", "pad"}:
        raise ModelError(f"Unknown THOR shape adjust mode '{shape_adjust}'. Use one of: crop, pad.")
    if shape_tol_px < 0:
        raise ModelError(f"RS_EMBED_THOR_SHAPE_TOL_PX must be >= 0, got {shape_tol_px}.")
    if max_native_side <= 0:
        raise ModelError(f"RS_EMBED_THOR_MAX_NATIVE_SIDE must be > 0, got {max_native_side}.")
    if max_native_tokens <= 0:
        raise ModelError(f"RS_EMBED_THOR_MAX_NATIVE_TOKENS must be > 0, got {max_native_tokens}.")

    return {
        "model_key": model_key,
        "variant": _thor_variant_from_model_key(model_key),
        "image_size": int(image_size),
        "ckpt_path": ckpt_path,
        "pretrained": bool(pretrained),
        "normalize_mode": normalize_mode,
        "group_merge": group_merge,
        "patch_size": int(patch_size),
        "resize_mode": resize_mode,
        "shape_adjust": shape_adjust,
        "shape_tol_px": int(shape_tol_px),
        "max_native_side": int(max_native_side),
        "max_native_tokens": int(max_native_tokens),
        "input_prep_mode": input_prep_mode,
        "hf_id": _thor_hf_id_from_model_key(model_key),
    }


def _resize_chw(x_chw: np.ndarray, *, out_hw: int) -> np.ndarray:
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW array, got {x_chw.shape}")
    x = torch.from_numpy(x_chw.astype(np.float32, copy=False)).unsqueeze(0)
    y = F.interpolate(x, size=(int(out_hw), int(out_hw)), mode="bilinear", align_corners=False)
    return y[0].detach().cpu().numpy().astype(np.float32)


def _normalize_thor_modality(modality: Any) -> str:
    raw = str(modality or "s2").strip().lower().replace("-", "_")
    aliases = {
        "sentinel1": "s1",
        "sentinel_1": "s1",
        "sentinel2": "s2",
        "sentinel_2": "s2",
    }
    resolved = aliases.get(raw, raw)
    if resolved not in {"s1", "s2"}:
        raise ModelError(f"Unknown THOR modality='{modality}' (expected 's2' or 's1').")
    return resolved


def _thor_band_gsds_for_modality(modality: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in _THOR_MODALITY_GSDS[_normalize_thor_modality(modality)])


def _thor_valid_side_unit(
    *,
    scale_m: int,
    patch_size: int,
    band_gsds: tuple[int, ...],
) -> int:
    if scale_m <= 0:
        raise ModelError(f"THOR scale_m must be > 0, got {scale_m}.")
    if patch_size <= 0:
        raise ModelError(f"THOR patch_size must be > 0, got {patch_size}.")
    if not band_gsds:
        raise ModelError("THOR band_gsds must be non-empty.")

    units = []
    for gsd in tuple(sorted(set(int(v) for v in band_gsds))):
        if gsd <= 0:
            raise ModelError(f"THOR band GSD must be > 0, got {gsd}.")
        ground_unit = int(patch_size) * int(gsd)
        units.append(ground_unit // math.gcd(int(scale_m), ground_unit))
    return int(math.lcm(*units))


def _thor_estimated_patch_tokens(
    *,
    side: int,
    scale_m: int,
    patch_size: int,
    band_gsds: tuple[int, ...],
) -> int | None:
    if side <= 0:
        return None
    ground_cover = int(side) * int(scale_m)
    total = 0
    for gsd in tuple(sorted(set(int(v) for v in band_gsds))):
        denom = int(patch_size) * int(gsd)
        if ground_cover % denom != 0:
            return None
        n = ground_cover // denom
        total += int(n * n)
    return int(total)


def _thor_center_crop_to_square_chw(x_chw: np.ndarray) -> np.ndarray:
    h, w = int(x_chw.shape[-2]), int(x_chw.shape[-1])
    side = min(h, w)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    return np.ascontiguousarray(x_chw[:, y0 : y0 + side, x0 : x0 + side])


def _thor_center_pad_to_square_chw(
    x_chw: np.ndarray,
    *,
    fill_value: float,
) -> np.ndarray:
    c, h, w = int(x_chw.shape[0]), int(x_chw.shape[1]), int(x_chw.shape[2])
    side = max(h, w)
    out = np.full((c, side, side), float(fill_value), dtype=np.float32)
    y0 = max(0, (side - h) // 2)
    x0 = max(0, (side - w) // 2)
    out[:, y0 : y0 + h, x0 : x0 + w] = x_chw
    return out


def _thor_center_crop_or_pad_to_side_chw(
    x_chw: np.ndarray,
    *,
    side: int,
    fill_value: float,
) -> np.ndarray:
    cur = int(x_chw.shape[-1])
    if int(x_chw.shape[-2]) != cur:
        raise ModelError(
            "THOR crop/pad-to-side expects a square CHW tensor before snapping to target side."
        )
    side = int(side)
    if side <= 0:
        raise ModelError(f"THOR target side must be > 0, got {side}.")
    if side == cur:
        return np.ascontiguousarray(x_chw)
    if side < cur:
        off = max(0, (cur - side) // 2)
        return np.ascontiguousarray(x_chw[:, off : off + side, off : off + side])

    c = int(x_chw.shape[0])
    out = np.full((c, side, side), float(fill_value), dtype=np.float32)
    off = max(0, (side - cur) // 2)
    out[:, off : off + cur, off : off + cur] = x_chw
    return out


def _thor_snap_side_to_valid(
    side: int,
    *,
    scale_m: int,
    patch_size: int,
    band_gsds: tuple[int, ...],
) -> int:
    unit = _thor_valid_side_unit(
        scale_m=scale_m,
        patch_size=patch_size,
        band_gsds=tuple(band_gsds),
    )
    lower = max(unit, (int(side) // unit) * unit)
    upper = max(unit, ((int(side) + unit - 1) // unit) * unit)
    if abs(int(side) - lower) <= abs(upper - int(side)):
        return int(lower)
    return int(upper)


def _prepare_thor_raw_chw(
    raw_chw: np.ndarray,
    *,
    scale_m: int,
    patch_size: int,
    band_gsds: tuple[int, ...],
    image_size: int,
    resize_mode: str,
    shape_adjust: str,
    shape_tol_px: int,
    max_native_side: int,
    max_native_tokens: int,
    input_prep_mode: str,
    fill_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(raw_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ModelError(f"THOR expected CHW input, got {getattr(x, 'shape', None)}")

    h, w = int(x.shape[-2]), int(x.shape[-1])
    meta: dict[str, Any] = {
        "resize_mode_requested": str(resize_mode),
        "input_prep_mode": str(input_prep_mode or "resize"),
        "raw_input_hw": (h, w),
        "requested_image_size": int(image_size),
        "shape_adjust": str(shape_adjust),
        "shape_tol_px": int(shape_tol_px),
        "max_native_side": int(max_native_side),
        "max_native_tokens": int(max_native_tokens),
        "shape_adjust_applied": "none",
    }

    if image_size <= 0:
        raise ModelError(f"RS_EMBED_THOR_IMG must be > 0, got {image_size}.")

    force_fixed = False
    fallback_reason = None
    input_prep_mode_l = str(input_prep_mode or "").lower().strip()
    resize_mode_l = str(resize_mode).lower().strip()

    if input_prep_mode_l == "tile":
        force_fixed = True
        fallback_reason = "tile_preserve_stitch_geometry"
    elif resize_mode_l != "native_snap":
        force_fixed = True
        fallback_reason = "fixed_mode"
    else:
        diff = abs(h - w)
        meta["raw_aspect_diff_px"] = int(diff)
        if diff > int(shape_tol_px):
            force_fixed = True
            fallback_reason = "rectangular_above_tolerance"

    if force_fixed:
        meta["preprocess_strategy"] = "fixed_resize"
        meta["preprocess_reason"] = str(fallback_reason)
        meta["final_hw"] = (int(image_size), int(image_size))
        meta["effective_image_size"] = int(image_size)
        meta["ground_cover_m"] = int(scale_m) * int(image_size)
        meta["estimated_patch_tokens"] = _thor_estimated_patch_tokens(
            side=int(image_size),
            scale_m=int(scale_m),
            patch_size=int(patch_size),
            band_gsds=tuple(band_gsds),
        )
        return x, meta

    if shape_adjust == "crop":
        x_sq = _thor_center_crop_to_square_chw(x)
        if h != w:
            meta["shape_adjust_applied"] = "crop_to_square"
    else:
        x_sq = _thor_center_pad_to_square_chw(x, fill_value=float(fill_value))
        if h != w:
            meta["shape_adjust_applied"] = "pad_to_square"

    square_side = int(x_sq.shape[-1])
    side_unit = _thor_valid_side_unit(
        scale_m=int(scale_m),
        patch_size=int(patch_size),
        band_gsds=tuple(band_gsds),
    )
    snapped_side = _thor_snap_side_to_valid(
        square_side,
        scale_m=int(scale_m),
        patch_size=int(patch_size),
        band_gsds=tuple(band_gsds),
    )
    est_tokens = _thor_estimated_patch_tokens(
        side=int(snapped_side),
        scale_m=int(scale_m),
        patch_size=int(patch_size),
        band_gsds=tuple(band_gsds),
    )

    meta["square_side"] = int(square_side)
    meta["side_unit"] = int(side_unit)
    meta["snapped_side"] = int(snapped_side)
    meta["estimated_patch_tokens"] = est_tokens

    if int(snapped_side) > int(max_native_side):
        meta["preprocess_strategy"] = "fixed_resize"
        meta["preprocess_reason"] = "native_side_limit"
        meta["final_hw"] = (int(image_size), int(image_size))
        meta["effective_image_size"] = int(image_size)
        meta["ground_cover_m"] = int(scale_m) * int(image_size)
        meta["estimated_patch_tokens"] = _thor_estimated_patch_tokens(
            side=int(image_size),
            scale_m=int(scale_m),
            patch_size=int(patch_size),
            band_gsds=tuple(band_gsds),
        )
        return x, meta

    if est_tokens is None or int(est_tokens) > int(max_native_tokens):
        meta["preprocess_strategy"] = "fixed_resize"
        meta["preprocess_reason"] = "native_token_limit"
        meta["final_hw"] = (int(image_size), int(image_size))
        meta["effective_image_size"] = int(image_size)
        meta["ground_cover_m"] = int(scale_m) * int(image_size)
        meta["estimated_patch_tokens"] = _thor_estimated_patch_tokens(
            side=int(image_size),
            scale_m=int(scale_m),
            patch_size=int(patch_size),
            band_gsds=tuple(band_gsds),
        )
        return x, meta

    x_native = _thor_center_crop_or_pad_to_side_chw(
        x_sq,
        side=int(snapped_side),
        fill_value=float(fill_value),
    )
    if int(snapped_side) > int(square_side):
        meta["snap_adjust_applied"] = "pad_to_valid_side"
    elif int(snapped_side) < int(square_side):
        meta["snap_adjust_applied"] = "crop_to_valid_side"
    else:
        meta["snap_adjust_applied"] = "none"

    meta["preprocess_strategy"] = "native_snap"
    meta["preprocess_reason"] = "native_snap_applied"
    meta["final_hw"] = (int(snapped_side), int(snapped_side))
    meta["effective_image_size"] = int(snapped_side)
    meta["ground_cover_m"] = int(scale_m) * int(snapped_side)
    return x_native, meta


def _normalize_s2_for_thor(raw_chw: np.ndarray, *, mode: str) -> np.ndarray:
    if raw_chw.ndim != 3 or int(raw_chw.shape[0]) != len(_S2_SR_10_BANDS):
        raise ModelError(f"Expected CHW with 10 S2 bands, got {getattr(raw_chw, 'shape', None)}")

    x = np.asarray(raw_chw, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 10000.0)

    m = str(mode).lower().strip()
    if m in {"raw", "none", "off"}:
        return x.astype(np.float32)

    x = x / 10000.0
    if m in {"unit", "unit_scale", "reflectance"}:
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    if m in {"thor_stats", "zscore", "thor_zscore"}:
        std = np.maximum(_THOR_S2_STD, 1e-6)
        x = (x - _THOR_S2_MEAN[:, None, None]) / std[:, None, None]
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    raise ModelError(
        f"Unknown THOR normalization mode '{mode}'. Use one of: thor_stats, unit_scale, none."
    )


def _normalize_s1_for_thor(raw_chw: np.ndarray, *, mode: str) -> np.ndarray:
    if raw_chw.ndim != 3 or int(raw_chw.shape[0]) != len(_S1_VVVH_BANDS):
        raise ModelError(f"Expected CHW with 2 S1 bands, got {getattr(raw_chw, 'shape', None)}")

    x = np.asarray(raw_chw, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    m = str(mode).lower().strip()
    if m in {"raw", "none", "off"}:
        return x.astype(np.float32)
    if m in {
        "thor_stats",
        "default",
        "auto",
        "s1_log_normalize",
        "log_normalize",
        "log1p",
        "log1p_p99",
    }:
        return _normalize_s1_vvvh_chw(x)

    raise ModelError(
        "Unknown THOR S1 normalization mode "
        f"'{mode}'. Use one of: thor_stats, s1_log_normalize, none."
    )


def _fetch_s2_sr_10_raw_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    raw = _fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_S2_SR_10_BANDS),
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        fill_value=fill_value,
    )
    return np.clip(raw, 0.0, 10000.0).astype(np.float32)


def _extract_feature_and_channel_params(
    out: Any,
) -> tuple[Any, dict[str, Any] | None]:
    channel_params = None
    features = out
    if isinstance(out, tuple) and len(out) >= 2:
        features = out[0]
        if isinstance(out[1], dict):
            channel_params = out[1]
    if not isinstance(features, (list, tuple)) or len(features) == 0:
        raise ModelError(f"THOR forward expected list/tuple of features, got type={type(features)}")
    feat_t = features[-1]
    return feat_t, channel_params


def _group_patch_sizes(
    *,
    channel_params: dict[str, Any],
    groups: dict[str, list[str]],
) -> tuple[list[int], list[str]]:
    patch_sizes: list[int] = []
    used_groups: list[str] = []
    for gname, members in groups.items():
        member = next(
            (
                m
                for m in members
                if m in channel_params
                and isinstance(channel_params[m], dict)
                and channel_params[m].get("num_patch") is not None
            ),
            None,
        )
        if member is None:
            continue
        p = int(channel_params[member]["num_patch"])
        if p <= 0:
            continue
        patch_sizes.append(p)
        used_groups.append(str(gname))
    return patch_sizes, used_groups


def _thor_group_grid_from_tokens(
    tokens_bnd,
    *,
    channel_params: dict[str, Any],
    groups: dict[str, list[str]],
    merge: str,
):
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(tokens_bnd) or tokens_bnd.ndim != 3:
        raise ModelError(f"Expected THOR tokens [B,N,D], got {getattr(tokens_bnd, 'shape', None)}")

    patch_sizes, used_groups = _group_patch_sizes(channel_params=channel_params, groups=groups)
    if not patch_sizes:
        raise ModelError("THOR returned no usable group patch sizes in channel_params.")

    expected_patch_tokens = int(sum(p * p for p in patch_sizes))
    n_tok = int(tokens_bnd.shape[1])
    start = 1 if n_tok == expected_patch_tokens + 1 else 0
    cls_removed = bool(start == 1)
    if n_tok < expected_patch_tokens + start:
        raise ModelError(
            f"THOR token count mismatch. got N={n_tok}, expected at least {expected_patch_tokens + start}"
        )

    patch_tokens = tokens_bnd[:, start : start + expected_patch_tokens, :]
    b, _, d = patch_tokens.shape
    max_p = max(patch_sizes)
    maps = []
    idx = 0
    for p in patch_sizes:
        pp = int(p * p)
        t = patch_tokens[:, idx : idx + pp, :]
        idx += pp
        t = t.reshape(b, p, p, d).permute(0, 3, 1, 2)  # [B,D,H,W]
        if p != max_p:
            t = F.interpolate(t, size=(max_p, max_p), mode="bilinear", align_corners=False)
        maps.append(t)

    merge_l = str(merge).lower().strip()
    if merge_l == "concat":
        grid = torch.cat(maps, dim=1)
    elif merge_l == "sum":
        grid = torch.stack(maps, dim=0).sum(dim=0)
    else:
        if merge_l not in {"mean", "avg", "average"}:
            raise ModelError(f"Unknown THOR group merge '{merge}'. Use mean/sum/concat.")
        grid = torch.stack(maps, dim=0).mean(dim=0)

    meta = {
        "expected_patch_tokens": expected_patch_tokens,
        "group_patch_sizes": tuple(int(p) for p in patch_sizes),
        "groups_used": tuple(used_groups),
        "cls_removed": cls_removed,
        "group_merge": "mean" if merge_l in {"mean", "avg", "average"} else merge_l,
    }
    return grid, meta


def _pool_thor_tokens(
    tokens: np.ndarray,
    *,
    pooling: str,
    expected_patch_tokens: int | None,
) -> tuple[np.ndarray, bool]:
    if (
        expected_patch_tokens is not None
        and tokens.ndim == 2
        and int(tokens.shape[0]) in {int(expected_patch_tokens), int(expected_patch_tokens) + 1}
    ):
        cls_removed = int(tokens.shape[0]) == int(expected_patch_tokens) + 1
        patch_tokens = tokens[1:] if cls_removed else tokens
        if pooling == "mean":
            return patch_tokens.mean(axis=0).astype(np.float32), bool(cls_removed)
        if pooling == "max":
            return patch_tokens.max(axis=0).astype(np.float32), bool(cls_removed)
        raise ModelError(f"Unknown pooling='{pooling}' (expected 'mean' or 'max').")
    return pool_from_tokens(tokens, pooling)


@lru_cache(maxsize=1)
def _load_thor_module():
    try:
        return importlib.import_module("rs_embed.embedders._vendor.thor_vit")
    except Exception as e:
        raise ModelError(f"Failed to import vendored THOR runtime: {type(e).__name__}: {e}") from e


@lru_cache(maxsize=8)
def _load_thor_cached(
    model_key: str,
    model_bands: tuple[str, ...],
    pretrained: bool,
    ckpt_path: str | None,
    ground_cover: int,
    patch_size: int,
    dev: str,
) -> tuple[Any, dict[str, Any]]:
    ensure_torch()
    import torch

    try:
        mod = _load_thor_module()
        build_kwargs: dict[str, Any] = {
            "model_name": str(model_key),
            "model_bands": list(model_bands),
            "out_indices": [-1],
            "return_channel_params": True,
            "pretrained": bool(pretrained),
            "input_params": {
                "ground_covers": [int(ground_cover)],
                "flexivit_patch_size_seqs": [int(patch_size)],
            },
        }
        if ckpt_path:
            build_kwargs["ckpt"] = os.path.expanduser(str(ckpt_path))
        model = mod.load_thor_model(**build_kwargs)
    except ModuleNotFoundError as e:
        raise ModelError(
            "Failed to import vendored THOR runtime while loading THOR. "
            f"Missing dependency: {getattr(e, 'name', None) or e}. "
            "Check project runtime dependencies like torch/timm/einops."
        ) from e
    except Exception as e:
        raise ModelError(
            f"Failed to build vendored THOR backbone '{model_key}'. "
            "Check the THOR package installation and model key."
        ) from e

    try:
        model = model.to(dev).eval()
    except Exception:
        pass

    p0 = None
    for _, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError("THOR model has no parameters; cannot verify weights.")
    if not torch.isfinite(p0).all():
        raise ModelError("THOR parameters contain NaN/Inf; load likely failed.")

    p0f = p0.float()
    out_channels = getattr(model, "out_channels", None)
    embed_dim = None
    if isinstance(out_channels, (list, tuple)) and len(out_channels) > 0:
        try:
            embed_dim = int(out_channels[-1])
        except Exception as _e:
            embed_dim = None

    meta = {
        "device": str(dev),
        "model_key": str(model_key),
        "model_bands": tuple(model_bands),
        "ground_cover_m": int(ground_cover),
        "patch_size": int(patch_size),
        "pretrained": bool(pretrained),
        "ckpt_path": os.path.expanduser(str(ckpt_path)) if ckpt_path else None,
        "embed_dim": embed_dim,
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
    }
    return model, meta


def _load_thor(
    *,
    model_key: str,
    model_bands: tuple[str, ...],
    pretrained: bool,
    ckpt_path: str | None,
    ground_cover: int,
    patch_size: int,
    device: str,
) -> tuple[Any, dict[str, Any], str]:
    (loaded, dev) = _load_cached_with_device(
        _load_thor_cached,
        device=device,
        model_key=str(model_key),
        model_bands=tuple(model_bands),
        pretrained=bool(pretrained),
        ckpt_path=(os.path.expanduser(ckpt_path) if ckpt_path else None),
        ground_cover=int(ground_cover),
        patch_size=int(patch_size),
    )
    model, meta = loaded
    return model, meta, dev


def _thor_forward_single(
    model: Any,
    x_chw: np.ndarray,
    *,
    device: str,
    group_merge: str,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    ensure_torch()
    import torch

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW input for THOR, got {getattr(x_chw, 'shape', None)}")

    dev = _resolve_device(device)
    model = model.to(dev).eval()
    x = torch.from_numpy(x_chw.astype(np.float32, copy=False)).unsqueeze(0).to(dev)

    with torch.no_grad():
        out = model(x)

    feat_t, channel_params = _extract_feature_and_channel_params(out)
    if not torch.is_tensor(feat_t) or feat_t.ndim != 3:
        raise ModelError(
            f"THOR feature tensor must be [B,N,D], got {getattr(feat_t, 'shape', None)}"
        )
    if int(feat_t.shape[0]) != 1:
        raise ModelError(
            f"THOR embedder expects B=1 in single inference, got B={int(feat_t.shape[0])}"
        )

    tokens = feat_t[0].detach().float().cpu().numpy().astype(np.float32)
    grid: np.ndarray | None = None
    expected_patch_tokens: int | None = None
    grid_meta: dict[str, Any] = {}

    groups = getattr(model, "groups", None)
    if isinstance(channel_params, dict) and isinstance(groups, dict):
        try:
            grid_bdhw, gmeta = _thor_group_grid_from_tokens(
                feat_t,
                channel_params=channel_params,
                groups=groups,
                merge=group_merge,
            )
            grid = grid_bdhw[0].detach().float().cpu().numpy().astype(np.float32)
            expected_patch_tokens = int(gmeta["expected_patch_tokens"])
            grid_meta = {
                "grid_kind": "thor_group_grid",
                "grid_group_merge": gmeta["group_merge"],
                "group_patch_sizes": gmeta["group_patch_sizes"],
                "groups_used": gmeta["groups_used"],
                "cls_removed": bool(gmeta["cls_removed"]),
                "expected_patch_tokens": expected_patch_tokens,
            }
        except Exception as _e:
            # fall back to square-token reshape (works for simple ViT-style outputs)
            grid = None

    if grid is None:
        try:
            g, (gh, gw), cls_removed = tokens_to_grid_dhw(tokens)
            grid = g.astype(np.float32)
            grid_meta = {
                "grid_kind": "patch_tokens",
                "grid_hw": (int(gh), int(gw)),
                "cls_removed": bool(cls_removed),
            }
        except Exception as _e:
            grid = None

    meta = {
        "tokens_shape": tuple(tokens.shape),
        "expected_patch_tokens": expected_patch_tokens,
        **grid_meta,
    }
    return tokens, grid, meta


@register("thor")
class THORBaseEmbedder(EmbedderBase):
    DEFAULT_MODEL_KEY = "thor_v1_base"
    DEFAULT_IMAGE_SIZE = 288
    DEFAULT_FETCH_WORKERS = 8

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_S2_SR_10_BANDS),
        scale_m=10,
        cloudy_pct=30,
        image_size=288,
        expected_channels=10,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "s2_sr": {
                    "collection": self.input_spec.collection,
                    "bands": list(self.input_spec.bands),
                },
                "s1": {
                    "collection": "COPERNICUS/S1_GRD_FLOAT (default) or COPERNICUS/S1_GRD",
                    "bands": list(_S1_VVVH_BANDS),
                },
            },
            "modalities": {
                "s2": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": tuple(_S2_SR_10_BANDS),
                },
                "s1": {
                    "collection": "COPERNICUS/S1_GRD_FLOAT",
                    "bands": tuple(_S1_VVVH_BANDS),
                    "defaults": {
                        "use_float_linear": True,
                        "s1_require_iw": True,
                        "s1_relax_iw_on_empty": True,
                    },
                },
            },
            "output": ["pooled", "grid"],
            "defaults": {
                "modality": "s2",
                "model_key": self.DEFAULT_MODEL_KEY,
                "variant": _thor_variant_from_model_key(self.DEFAULT_MODEL_KEY),
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "patch_size": 8,
                "resize_mode": "native_snap",
                "shape_adjust": "crop",
                "shape_tol_px": 8,
                "max_native_side": 384,
                "max_native_tokens": 3000,
                "normalization": "thor_stats",
                "scale_m": self.input_spec.scale_m,
                "cloudy_pct": self.input_spec.cloudy_pct,
                "composite": self.input_spec.composite,
                "group_merge": "mean",
                "use_float_linear": True,
                "s1_require_iw": True,
                "s1_relax_iw_on_empty": True,
            },
            "model_config": {
                "variant": {
                    "type": "string",
                    "default": _thor_variant_from_model_key(self.DEFAULT_MODEL_KEY),
                    "choices": ["tiny", "small", "base", "large"],
                }
            },
            "notes": [
                "Loads THOR through a fully vendored local runtime.",
                "Default weights come from Hugging Face FM4CS/THOR-1.0-base when pretrained=true.",
                "modality='s1' uses a VV/VH Sentinel-1 branch with shared S1 log-style normalization.",
            ],
        }

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_THOR_FETCH_WORKERS",
                str(THORBaseEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _default_sensor(modality: str | None = None) -> SensorSpec:
        modality_l = _normalize_thor_modality(modality)
        if modality_l == "s1":
            return SensorSpec(
                collection="COPERNICUS/S1_GRD_FLOAT",
                bands=tuple(_S1_VVVH_BANDS),
                scale_m=10,
                cloudy_pct=30,
                composite="median",
                fill_value=0.0,
                modality="s1",
                use_float_linear=True,
                s1_require_iw=True,
                s1_relax_iw_on_empty=True,
            )
        return SensorSpec(
            collection=THORBaseEmbedder.input_spec.collection,
            bands=tuple(_S2_SR_10_BANDS),
            scale_m=THORBaseEmbedder.input_spec.scale_m,
            cloudy_pct=THORBaseEmbedder.input_spec.cloudy_pct,
            composite=THORBaseEmbedder.input_spec.composite,
            fill_value=THORBaseEmbedder.input_spec.fill_value,
            modality="s2",
        )

    def fetch_input(
        self,
        provider: ProviderBase,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec,
    ) -> FetchResult | None:
        t = temporal_to_range(temporal)
        modality = _normalize_thor_modality(getattr(sensor, "modality", "s2"))
        if modality == "s1":
            raw, meta = _fetch_s1_vvvh_raw_chw_with_meta(
                provider,
                spatial=spatial,
                temporal=t,
                scale_m=int(getattr(sensor, "scale_m", 10)),
                orbit=getattr(sensor, "orbit", None),
                use_float_linear=bool(getattr(sensor, "use_float_linear", True)),
                composite=str(getattr(sensor, "composite", "median")),
                fill_value=float(getattr(sensor, "fill_value", 0.0)),
                require_iw=bool(getattr(sensor, "s1_require_iw", True)),
                relax_iw_on_empty=bool(getattr(sensor, "s1_relax_iw_on_empty", True)),
            )
            return FetchResult(data=raw, meta=meta)

        raw = _fetch_s2_sr_10_raw_chw(
            provider,
            spatial,
            t,
            scale_m=int(getattr(sensor, "scale_m", 10)),
            cloudy_pct=int(getattr(sensor, "cloudy_pct", 30)),
            composite=str(getattr(sensor, "composite", "median")),
            fill_value=float(getattr(sensor, "fill_value", 0.0)),
        )
        return FetchResult(data=raw, meta={})

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: np.ndarray | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("thor expects a provider backend (or 'auto').")

        ss = sensor or self._default_sensor()
        t = temporal_to_range(temporal)
        modality = _normalize_thor_modality(getattr(ss, "modality", "s2"))
        runtime_cfg = _resolve_thor_runtime_config(
            model_config=model_config,
            default_model_key=self.DEFAULT_MODEL_KEY,
            default_image_size=self.DEFAULT_IMAGE_SIZE,
        )
        image_size = int(runtime_cfg["image_size"])
        model_key = str(runtime_cfg["model_key"])
        ckpt_path = runtime_cfg["ckpt_path"]
        pretrained = bool(runtime_cfg["pretrained"])
        normalize_mode = str(runtime_cfg["normalize_mode"])
        group_merge = str(runtime_cfg["group_merge"])
        patch_size = int(runtime_cfg["patch_size"])
        resize_mode = str(runtime_cfg["resize_mode"])
        shape_adjust = str(runtime_cfg["shape_adjust"])
        shape_tol_px = int(runtime_cfg["shape_tol_px"])
        max_native_side = int(runtime_cfg["max_native_side"])
        max_native_tokens = int(runtime_cfg["max_native_tokens"])
        input_prep_mode = str(runtime_cfg["input_prep_mode"])

        source = str(getattr(ss, "collection", "COPERNICUS/S2_SR_HARMONIZED"))
        scale_m = int(getattr(ss, "scale_m", 10))
        cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
        composite = str(getattr(ss, "composite", "median"))
        fill_value = float(getattr(ss, "fill_value", 0.0))
        use_float_linear = bool(getattr(ss, "use_float_linear", True))
        s1_require_iw = bool(getattr(ss, "s1_require_iw", True))
        s1_relax_iw_on_empty = bool(getattr(ss, "s1_relax_iw_on_empty", True))
        orbit = getattr(ss, "orbit", None)
        band_gsds = _thor_band_gsds_for_modality(modality)
        sensor_bands = _THOR_MODALITY_SENSOR_BANDS[modality]
        model_bands = _THOR_MODALITY_MODEL_BANDS[modality]

        fetch_meta: dict[str, Any] = {}
        if input_chw is None:
            result = self.fetch_input(
                self._get_provider(backend),
                spatial=spatial,
                temporal=t,
                sensor=ss,
            )
            assert result is not None
            raw_chw = result.data
            fetch_meta = result.meta
        else:
            if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(sensor_bands):
                raise ModelError(
                    "input_chw must be CHW with "
                    f"{len(sensor_bands)} bands for THOR {modality.upper()}, "
                    f"got {getattr(input_chw, 'shape', None)}"
                )
            raw_chw = np.asarray(input_chw, dtype=np.float32)
            raw_chw = np.nan_to_num(raw_chw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            if modality == "s2":
                raw_chw = np.clip(raw_chw, 0.0, 10000.0).astype(np.float32)

        from ..tools.inspection import checks_should_raise, maybe_inspect_chw

        check_meta: dict[str, Any] = {}
        if modality == "s2":
            report = maybe_inspect_chw(
                raw_chw,
                sensor=ss,
                name="provider_s2_sr_10_raw_chw",
                expected_channels=len(_S2_SR_10_BANDS),
                value_range=(0.0, 10000.0),
                fill_value=fill_value,
                meta=check_meta,
            )
        else:
            report = maybe_inspect_chw(
                raw_chw,
                sensor=ss,
                name="provider_s1_vvvh_raw_chw",
                expected_channels=len(_S1_VVVH_BANDS),
                fill_value=fill_value,
                meta=check_meta,
            )
        if report is not None and (not report.get("ok", True)) and checks_should_raise(ss):
            raise ModelError(
                "Provider input inspection failed: " + "; ".join(report.get("issues", []))
            )

        raw_prepped, prep_meta = _prepare_thor_raw_chw(
            raw_chw,
            scale_m=scale_m,
            patch_size=patch_size,
            band_gsds=tuple(band_gsds),
            image_size=image_size,
            resize_mode=resize_mode,
            shape_adjust=shape_adjust,
            shape_tol_px=shape_tol_px,
            max_native_side=max_native_side,
            max_native_tokens=max_native_tokens,
            input_prep_mode=input_prep_mode,
            fill_value=fill_value,
        )

        if modality == "s1":
            x_chw = _normalize_s1_for_thor(raw_prepped, mode=normalize_mode)
        else:
            x_chw = _normalize_s2_for_thor(raw_prepped, mode=normalize_mode)
        effective_image_size = int(prep_meta["effective_image_size"])
        ground_cover = int(prep_meta["ground_cover_m"])
        if prep_meta["preprocess_strategy"] == "fixed_resize" and (
            x_chw.shape[-1] != image_size or x_chw.shape[-2] != image_size
        ):
            x_chw = _resize_chw(x_chw, out_hw=image_size)

        model, wmeta, dev = _load_thor(
            model_key=model_key,
            model_bands=tuple(model_bands),
            pretrained=pretrained,
            ckpt_path=ckpt_path,
            ground_cover=ground_cover,
            patch_size=patch_size,
            device=device,
        )
        tokens, grid, fmeta = _thor_forward_single(
            model,
            x_chw,
            device=dev,
            group_merge=group_merge,
        )

        meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=str(backend).lower(),
            source=source,
            sensor={
                "collection": source,
                "bands": tuple(sensor_bands),
                "bands_thor": tuple(model_bands),
                "scale_m": scale_m,
                "cloudy_pct": cloudy_pct,
                "composite": composite,
                "fill_value": fill_value,
                "modality": modality,
                "orbit": orbit if modality == "s1" else None,
                "use_float_linear": use_float_linear if modality == "s1" else None,
                "s1_require_iw": s1_require_iw if modality == "s1" else None,
                "s1_relax_iw_on_empty": s1_relax_iw_on_empty if modality == "s1" else None,
            },
            temporal=t,
            image_size=effective_image_size,
            extra={
                "hf_id": runtime_cfg["hf_id"],
                "model_source": "vendored_rs_embed_runtime",
                "modality": modality,
                "variant": str(runtime_cfg["variant"]),
                "normalization": normalize_mode,
                "group_merge": group_merge,
                "requested_image_size": image_size,
                "resize_mode": resize_mode,
                "shape_adjust": shape_adjust,
                "shape_tol_px": shape_tol_px,
                "max_native_side": max_native_side,
                "max_native_tokens": max_native_tokens,
                "ground_cover_m": ground_cover,
                "patch_size": patch_size,
                "use_float_linear": use_float_linear if modality == "s1" else None,
                "s1_require_iw": s1_require_iw if modality == "s1" else None,
                "s1_relax_iw_on_empty": s1_relax_iw_on_empty if modality == "s1" else None,
                "orbit": orbit if modality == "s1" else None,
                **fetch_meta,
                **check_meta,
                **prep_meta,
                **wmeta,
                **fmeta,
            },
        )

        if output.mode == "pooled":
            vec, cls_removed = _pool_thor_tokens(
                tokens,
                pooling=output.pooling,
                expected_patch_tokens=fmeta.get("expected_patch_tokens"),
            )
            out_meta = {
                **meta,
                "pooling": output.pooling,
                "cls_removed": bool(cls_removed),
            }
            return Embedding(data=vec.astype(np.float32), meta=out_meta)

        if output.mode == "grid":
            if grid is None:
                raise ModelError(
                    "THOR grid output is unavailable for this configuration. "
                    "Try pooled output, or use default model/input settings."
                )
            gmeta = {
                **meta,
                "grid_shape": tuple(grid.shape),
            }
            da = xr.DataArray(
                grid.astype(np.float32),
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(grid.shape[0]),
                    "y": np.arange(grid.shape[1]),
                    "x": np.arange(grid.shape[2]),
                },
                name="embedding",
                attrs=gmeta,
            )
            return Embedding(data=da, meta=gmeta)

        raise ModelError(f"Unknown output mode: {output.mode}")

    def get_embeddings_batch(
        self,
        *,
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []

        backend_l = backend.lower().strip()
        if not is_provider_backend(backend_l, allow_auto=True):
            raise ModelError("thor expects a provider backend (or 'auto').")

        t = temporal_to_range(temporal)
        ss = sensor or self._default_sensor()
        modality = _normalize_thor_modality(getattr(ss, "modality", "s2"))
        provider = self._get_provider(backend_l)

        scale_m = int(getattr(ss, "scale_m", 10))
        cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
        composite = str(getattr(ss, "composite", "median"))
        fill_value = float(getattr(ss, "fill_value", 0.0))
        use_float_linear = bool(getattr(ss, "use_float_linear", True))
        s1_require_iw = bool(getattr(ss, "s1_require_iw", True))
        s1_relax_iw_on_empty = bool(getattr(ss, "s1_relax_iw_on_empty", True))
        orbit = getattr(ss, "orbit", None)

        n = len(spatials)
        prefetched_raw: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            if modality == "s1":
                raw, _meta = _fetch_s1_vvvh_raw_chw_with_meta(
                    provider,
                    spatial=sp,
                    temporal=t,
                    scale_m=scale_m,
                    orbit=orbit,
                    use_float_linear=use_float_linear,
                    composite=composite,
                    fill_value=fill_value,
                    require_iw=s1_require_iw,
                    relax_iw_on_empty=s1_relax_iw_on_empty,
                )
            else:
                raw = _fetch_s2_sr_10_raw_chw(
                    provider,
                    sp,
                    t,
                    scale_m=scale_m,
                    cloudy_pct=cloudy_pct,
                    composite=composite,
                    fill_value=fill_value,
                )
            return i, raw

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                ii, raw = _fetch_one(i, sp)
                prefetched_raw[ii] = raw
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, raw = fut.result()
                    prefetched_raw[i] = raw

        out: list[Embedding] = []
        for i, sp in enumerate(spatials):
            raw = prefetched_raw[i]
            if raw is None:
                raise ModelError(f"Missing prefetched input at index={i} for thor.")
            out.append(
                self.get_embedding(
                    spatial=sp,
                    temporal=temporal,
                    sensor=ss,
                    output=output,
                    backend=backend,
                    device=device,
                    input_chw=raw,
                    model_config=model_config,
                )
            )
        return out

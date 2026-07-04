from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np

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
    fetch_collection_binned_raw_tchw as _fetch_collection_binned_raw_tchw,
)
from ..providers.fetch import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
)
from ..providers.fetch import (
    fetch_s1_vvvh_binned_raw_tchw as _fetch_s1_vvvh_binned_raw_tchw,
)
from ..providers.fetch import (
    fetch_s1_vvvh_raw_chw_with_meta as _fetch_s1_vvvh_raw_chw_with_meta,
)
from ..providers.resolution import is_provider_backend
from ..tools.runtime import load_cached_with_device as _load_cached_with_device
from ..tools.shape import (
    crop_grid_to_roi,
    prepare_square,
    roi_fetch_meta,
    roi_is_full,
    roi_token_box,
)
from ..tools.spatial import FULL_WINDOW, square_spatial
from .base import EmbedderBase
from .config import model_config_value
from .meta import build_meta, temporal_midpoint_str, temporal_to_range

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# S2 L2A band order expected by OlmoEarth (matches Modality.SENTINEL2_L2A):
#   BandSet-0 (10 m): B02, B03, B04, B08
#   BandSet-1 (20 m): B05, B06, B07, B8A, B11, B12
#   BandSet-2 (60 m): B01, B09
# GEE COPERNICUS/S2_SR_HARMONIZED uses names without leading zeros.
_S2_BANDS_GEE: tuple[str, ...] = (
    "B2",
    "B3",
    "B4",
    "B8",  # 10 m
    "B5",
    "B6",
    "B7",
    "B8A",
    "B11",
    "B12",  # 20 m
    "B1",
    "B9",  # 60 m
)
_N_BANDS = len(_S2_BANDS_GEE)  # 12
_N_BAND_SETS = 3  # matches OlmoEarth S2 L2A

# S1 GRD VV/VH order matches OlmoEarth Modality.SENTINEL1 band_order ['vv', 'vh'].
# OlmoEarth S1 normalization stats are in dB (COPERNICUS/S1_GRD units).
_S1_BANDS_GEE: tuple[str, ...] = ("VV", "VH")
_N_BANDS_S1 = len(_S1_BANDS_GEE)  # 2
_N_BAND_SETS_S1 = 1  # matches OlmoEarth Modality.SENTINEL1 (single band set)

# Map canonical variant names to (ModelID enum string, size, version)
_VARIANT_SPECS: dict[str, tuple[str, str, str]] = {
    "nano": ("OlmoEarth-v1-Nano", "nano", "v1"),
    "tiny": ("OlmoEarth-v1-Tiny", "tiny", "v1"),
    "base": ("OlmoEarth-v1-Base", "base", "v1"),
    "large": ("OlmoEarth-v1-Large", "large", "v1"),
    "nano_v1_1": ("OlmoEarth-v1_1-Nano", "nano", "v1.1"),
    "tiny_v1_1": ("OlmoEarth-v1_1-Tiny", "tiny", "v1.1"),
    "base_v1_1": ("OlmoEarth-v1_1-Base", "base", "v1.1"),
    "nano_v1_2": ("OlmoEarth-v1_2-Nano", "nano", "v1.2"),
    "tiny_v1_2": ("OlmoEarth-v1_2-Tiny", "tiny", "v1.2"),
    "small_v1_2": ("OlmoEarth-v1_2-Small", "small", "v1.2"),
    "base_v1_2": ("OlmoEarth-v1_2-Base", "base", "v1.2"),
}

_VARIANT_ALIASES: dict[str, str] = {
    "nano_v1": "nano",
    "tiny_v1": "tiny",
    "base_v1": "base",
    "large_v1": "large",
    "nano_11": "nano_v1_1",
    "tiny_11": "tiny_v1_1",
    "base_11": "base_v1_1",
    "nano_12": "nano_v1_2",
    "tiny_12": "tiny_v1_2",
    "small_12": "small_v1_2",
    "base_12": "base_v1_2",
    # "small" exists only in v1.2, so the bare size name is unambiguous.
    "small": "small_v1_2",
}

_DEFAULT_VARIANT = "base_v1_2"
_DEFAULT_IMAGE_SIZE = 256  # training tile size; model accepts any size divisible by patch_size
_DEFAULT_PATCH_SIZE = 4
# OlmoEarth's positional encoding requires a square token grid, so a rectangular
# ROI must be made square before encoding. Default "pad" keeps the whole ROI
# (no data discarded) and avoids the aspect-ratio stretch that smears a non-square
# field into striped, distorted embeddings; "crop" center-crops to square instead.
_DEFAULT_SHAPE_ADJUST = "pad"
_DEFAULT_SCALE_M = 10
_DEFAULT_CLOUDY_PCT = 30
# "auto" (default) picks single/multi from the window: multi when the range spans
# ≥2 temporal bins, single otherwise. Use "single"/"multi" to force a mode.
_DEFAULT_TEMPORAL_MODE = "auto"

# Multi-frame mode mirrors OlmoEarth pretraining: fixed 30-day bins anchored at
# the range start (offsets in the official rslearn config are 30d strides, not
# calendar months), at most 12 frames per sample (YEAR regime).
_FRAME_STRIDE_DAYS = 30
_MAX_FRAMES = 12


# ---------------------------------------------------------------------------
# Package guard
# ---------------------------------------------------------------------------


def _ensure_olmoearth() -> Any:
    """Import and return olmoearth_pretrain_minimal, raising a clear error if absent."""
    try:
        import olmoearth_pretrain_minimal as om  # type: ignore[import-untyped]

        return om
    except ImportError as exc:
        raise ModelError(
            "OlmoEarth requires olmoearth-pretrain-minimal. "
            "Install: uv pip install olmoearth-pretrain-minimal"
        ) from exc


def _ensure_torch() -> Any:
    try:
        import torch  # noqa: F401

        return torch
    except ImportError as exc:
        raise ModelError("OlmoEarth requires torch. Install: uv pip install torch") from exc


# ---------------------------------------------------------------------------
# Variant helpers
# ---------------------------------------------------------------------------


def _normalize_variant(variant: Any) -> str:
    raw = str(variant).strip().lower().replace("-", "_").replace(".", "_")
    if raw in _VARIANT_SPECS:
        return raw
    if raw in _VARIANT_ALIASES:
        return _VARIANT_ALIASES[raw]
    raise ModelError(
        f"Unknown OlmoEarth variant='{variant}'. Valid choices: {sorted(_VARIANT_SPECS)}."
    )


def _resolve_variant(model_config: dict[str, Any] | None) -> str:
    v = model_config_value(model_config, "variant")
    if v is not None:
        return _normalize_variant(v)
    env = os.environ.get("RS_EMBED_OLMOEARTH_VARIANT", "").strip()
    if env and env.lower() not in ("", "auto"):
        return _normalize_variant(env)
    return _DEFAULT_VARIANT


def _coerce_patch_size(v: Any, *, source: str) -> int:
    try:
        ps = int(v)
    except (TypeError, ValueError) as exc:
        raise ModelError(
            f"OlmoEarth patch_size from {source} must be an integer, got {v!r}."
        ) from exc
    if ps < 1 or ps > 8:
        raise ModelError(f"OlmoEarth patch_size must be 1–8, got {ps}.")
    return ps


def _resolve_patch_size(model_config: dict[str, Any] | None) -> int:
    v = model_config_value(model_config, "patch_size")
    if v is not None:
        return _coerce_patch_size(v, source="model_config")
    env = os.environ.get("RS_EMBED_OLMOEARTH_PATCH_SIZE", "").strip()
    if env:
        return _coerce_patch_size(env, source="RS_EMBED_OLMOEARTH_PATCH_SIZE")
    return _DEFAULT_PATCH_SIZE


def _coerce_image_size(v: Any, *, source: str) -> int:
    try:
        s = int(v)
    except (TypeError, ValueError) as exc:
        raise ModelError(
            f"OlmoEarth image_size from {source} must be an integer, got {v!r}."
        ) from exc
    if s < 1:
        raise ModelError(f"OlmoEarth image_size must be positive, got {s}.")
    return s


def _resolve_image_size(model_config: dict[str, Any] | None) -> int:
    v = model_config_value(model_config, "image_size")
    if v is not None:
        return _coerce_image_size(v, source="model_config")
    env = os.environ.get("RS_EMBED_OLMOEARTH_IMAGE_SIZE", "").strip()
    if env:
        return _coerce_image_size(env, source="RS_EMBED_OLMOEARTH_IMAGE_SIZE")
    return _DEFAULT_IMAGE_SIZE


def _resolve_geometry(model_config: dict[str, Any] | None) -> tuple[int, int]:
    """Resolve (image_size, patch_size), enforcing the divisibility contract."""
    patch_size = _resolve_patch_size(model_config)
    image_size = _resolve_image_size(model_config)
    if image_size % patch_size != 0:
        raise ModelError(
            f"OlmoEarth image_size ({image_size}) must be divisible by patch_size ({patch_size})."
        )
    return image_size, patch_size


def _normalize_temporal_mode(mode: Any) -> str:
    m = str(mode or _DEFAULT_TEMPORAL_MODE).strip().lower()
    if m not in ("single", "multi", "auto"):
        raise ModelError(
            f"olmoearth temporal_mode must be 'single', 'multi', or 'auto', got {mode!r}."
        )
    return m


def _resolve_temporal_mode(model_config: dict[str, Any] | None) -> str:
    """Resolve the *configured* temporal mode (may be ``"auto"``)."""
    v = model_config_value(model_config, "temporal_mode")
    if v is not None:
        return _normalize_temporal_mode(v)
    env = os.environ.get("RS_EMBED_OLMOEARTH_TEMPORAL_MODE", "").strip()
    if env:
        return _normalize_temporal_mode(env)
    return _DEFAULT_TEMPORAL_MODE


def _expand_auto_mode(mode: str, t: TemporalSpec) -> str:
    """Expand ``"auto"`` to ``"single"``/``"multi"`` from the window.

    Resolves to ``"multi"`` when the range spans ≥2 temporal bins (it genuinely
    covers multiple time steps), else ``"single"`` — a single bin where multi
    would add nothing but extra GEE fetches. ``"single"``/``"multi"`` pass through.
    """
    if mode != "auto":
        return mode
    bins, _ = _temporal_bins(t)
    return "multi" if len(bins) >= 2 else "single"


def _temporal_bins(t: TemporalSpec) -> tuple[tuple[tuple[str, str], ...], bool]:
    """Temporal bins for the requested window, returned as ``(bins, stretched)``.

    Up to ``_MAX_FRAMES`` (12) fixed ``_FRAME_STRIDE_DAYS`` (30-day) bins, mirroring
    OlmoEarth's YEAR pretraining regime. Windows longer than that capacity are
    **equal-divided into 12 frames** instead of dropping the trailing time, so the
    whole window is covered (``stretched=True``); see
    :func:`rs_embed.tools.temporal.fixed_or_equal_bins`.
    """
    from ..tools.temporal import fixed_or_equal_bins  # noqa: PLC0415

    return fixed_or_equal_bins(
        str(t.start), str(t.end), stride_days=_FRAME_STRIDE_DAYS, max_bins=_MAX_FRAMES
    )


def _temporal_sampling_meta(bins: tuple[tuple[str, str], ...], stretched: bool) -> dict[str, Any]:
    """Pure metadata describing the temporal binning mode (no side effects)."""
    meta: dict[str, Any] = {
        "temporal_sampling": "equal_divided" if stretched else "fixed_stride",
        "temporal_spacing_stretched": bool(stretched),
    }
    if stretched and bins:
        from datetime import date as _date  # noqa: PLC0415

        span = (_date.fromisoformat(bins[-1][1]) - _date.fromisoformat(bins[0][0])).days
        meta["effective_stride_days"] = int(round(span / max(1, len(bins))))
    return meta


def _warn_stretched_sampling(sampling_meta: dict[str, Any]) -> None:
    """Emit a single warning when equal-division (stretched) sampling was used."""
    if not sampling_meta.get("temporal_spacing_stretched"):
        return
    warnings.warn(
        f"OlmoEarth window exceeds {_MAX_FRAMES} × {_FRAME_STRIDE_DAYS}-day frames; "
        f"switched to equal division (~{sampling_meta.get('effective_stride_days')}d apart) "
        "to cover the whole window instead of dropping the trailing time. Frame spacing is "
        "outside OlmoEarth's monthly training cadence, so embeddings are extrapolated — "
        "narrow the temporal window to stay in-distribution.",
        UserWarning,
        stacklevel=2,
    )


def _warn_dropped_bins(dropped: list[tuple[str, str]], *, n_bins: int, n_frames: int) -> None:
    """Warn when 30-day bins were dropped from the series for lack of imagery."""
    if not dropped:
        return
    ranges = ", ".join(f"{s}→{e}" for s, e in dropped)
    warnings.warn(
        f"OlmoEarth dropped {len(dropped)} of {n_bins} temporal bin(s) with no usable "
        f"imagery (no scene passed the cloud filter): {ranges}. Encoded {n_frames} "
        "frame(s) instead. To recover them, raise cloudy_pct, widen/shift the window, "
        "or pass temporal_mode='single' if a single composite is acceptable.",
        UserWarning,
        stacklevel=2,
    )


def _multi_meta_extra(
    mode: str,
    sampling_meta: dict[str, Any],
    n_bins: int,
    dropped: list[tuple[str, str]],
) -> dict[str, Any]:
    """Per-item multi-frame meta: sampling info, bin count, and any dropped bins."""
    if mode != "multi":
        return {}
    extra: dict[str, Any] = {**sampling_meta, "n_bins": n_bins}
    if dropped:
        extra["dropped_bins"] = [list(b) for b in dropped]
    return extra


# ---------------------------------------------------------------------------
# Modality helpers
# ---------------------------------------------------------------------------


def _normalize_modality(modality: Any) -> str:
    m = str(modality or "s2").strip().lower()
    if m not in ("s2", "s1"):
        raise ModelError(f"olmoearth modality must be 's2' or 's1', got {modality!r}.")
    return m


def _modality_n_bands(modality: str) -> int:
    return _N_BANDS if modality == "s2" else _N_BANDS_S1


def _modality_n_band_sets(modality: str) -> int:
    return _N_BAND_SETS if modality == "s2" else _N_BAND_SETS_S1


def _modality_field(modality: str) -> str:
    """MaskedOlmoEarthSample / tokens_and_masks field name for a modality."""
    return "sentinel2_l2a" if modality == "s2" else "sentinel1"


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_olmoearth_cached(model_id_str: str, dev: str):
    om = _ensure_olmoearth()
    model_id = om.ModelID(model_id_str)
    try:
        model = om.load_model_from_id(model_id, load_weights=True)
    except Exception as exc:
        raise ModelError(
            f"Failed to load OlmoEarth model '{model_id_str}': {type(exc).__name__}: {exc}"
        ) from exc
    model = model.to(dev).eval()
    meta = {
        "model_id": model_id_str,
        "hf_repo": model_id.repo_id(),
        "device": dev,
    }
    return model, meta


def _load_olmoearth(variant: str, *, device: str = "auto"):
    model_id_str, _, _ = _VARIANT_SPECS[variant]
    (model, meta), dev = _load_cached_with_device(
        _load_olmoearth_cached,
        device=device,
        model_id_str=model_id_str,
    )
    return model, meta, dev


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _normalize_chw(x_chw: np.ndarray, *, modality: str = "s2") -> np.ndarray:
    """Normalize CHW raw values (S2 DN or S1 dB) via OlmoEarth mean±2σ clipping."""
    om = _ensure_olmoearth()
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import (
        Modality,  # type: ignore
    )

    om_modality = Modality.SENTINEL2_L2A if modality == "s2" else Modality.SENTINEL1
    # Normalizer expects shape (..., C) — transpose to HWC, normalize, back to CHW
    hwc = np.moveaxis(x_chw, 0, -1).astype(np.float32)  # [H, W, C]
    norm = om.Normalizer(std_multiplier=2.0)
    hwc = norm.normalize(om_modality, hwc).astype(np.float32)
    hwc = np.nan_to_num(hwc, nan=0.0, posinf=1.0, neginf=0.0)
    return np.moveaxis(hwc, -1, 0)  # [C, H, W]


def _s1_linear_to_db(x_chw: np.ndarray) -> np.ndarray:
    """Convert linear-power S1 backscatter to dB (OlmoEarth S1 stats are in dB).

    Mirrors the official ``convert_to_db`` in olmoearth_pretrain
    (10·log10 with values clipped to 1e-10 to avoid log(0)).
    """
    return (10.0 * np.log10(np.clip(x_chw, 1e-10, None))).astype(np.float32)


def _normalize_shape_adjust(v: Any) -> str:
    s = str(v or _DEFAULT_SHAPE_ADJUST).strip().lower()
    if s not in ("pad", "crop"):
        raise ModelError(f"olmoearth shape_adjust must be 'pad' or 'crop', got {v!r}.")
    return s


def _resolve_shape_adjust(model_config: dict[str, Any] | None) -> str:
    v = model_config_value(model_config, "shape_adjust")
    if v is not None:
        return _normalize_shape_adjust(v)
    env = os.environ.get("RS_EMBED_OLMOEARTH_SHAPE_ADJUST", "").strip()
    if env:
        return _normalize_shape_adjust(env)
    return _DEFAULT_SHAPE_ADJUST


def _prepare_chw(
    x_chw: np.ndarray,
    *,
    image_size: int,
    patch_size: int,
    modality: str = "s2",
    shape_adjust: str = _DEFAULT_SHAPE_ADJUST,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize a CHW patch and make it a square ``image_size`` input.

    A rectangular ROI is padded/cropped to square *before* resizing (no
    aspect-ratio stretch); see :mod:`rs_embed.tools.shape`. Returns
    ``(x, shape_prep_meta)``.
    """
    n_bands = _modality_n_bands(modality)
    if x_chw.ndim != 3 or x_chw.shape[0] != n_bands:
        raise ModelError(
            f"OlmoEarth ({modality}) expects {n_bands}-band CHW input, got {x_chw.shape}."
        )
    if image_size < 1 or image_size % patch_size != 0:
        raise ModelError(
            f"OlmoEarth image_size ({image_size}) must be a positive multiple of "
            f"patch_size ({patch_size})."
        )
    x = _normalize_chw(x_chw, modality=modality)
    x, shape_meta = prepare_square(
        x, size=image_size, shape_adjust=shape_adjust, fill_value=0.0, pad_mode="reflect"
    )
    return x, shape_meta


def _date_to_timestamp(date_str: str | None) -> tuple[int, int, int]:
    """Convert ISO date string to OlmoEarth timestamp tuple (day, month_0idx, year)."""
    if not date_str:
        return (1, 0, 2022)
    from datetime import date as _date  # noqa: PLC0415

    d = _date.fromisoformat(date_str)
    return (d.day, d.month - 1, d.year)  # month is 0-indexed in OlmoEarth


def _prepare_frames(
    x_tchw: np.ndarray,
    *,
    bins: tuple[tuple[str, str], ...],
    modality: str,
    image_size: int,
    patch_size: int,
    shape_adjust: str = _DEFAULT_SHAPE_ADJUST,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[str, str]], dict[str, Any]]:
    """Prepare a binned [T,C,H,W] stack: drop empty frames, normalize, square, resize.

    Empty bins arrive as all-NaN sentinel frames (see the binned fetch
    helpers); they are dropped here — under ``fast_pass=True`` the encoder
    ignores attention masks, so excluding empty frames is the only way to keep
    them out of the forward pass. Timestamps (frame-start dates, mirroring the
    official pipeline) are kept aligned with the surviving frames.

    Returns ``(stack, timestamps, dropped, shape_prep)`` where ``dropped`` lists
    the ``(start, end)`` ranges of the bins that had no imagery (callers surface
    it via ``meta['dropped_bins']`` and a ``UserWarning``), and ``shape_prep`` is
    the square-prep metadata (shared by all frames, which have identical H/W).
    """
    n_bands = _modality_n_bands(modality)
    if x_tchw.ndim != 4 or x_tchw.shape[1] != n_bands:
        raise ModelError(
            f"OlmoEarth ({modality}) multi-frame input must be [T,{n_bands},H,W], "
            f"got {getattr(x_tchw, 'shape', None)}."
        )
    if x_tchw.shape[0] != len(bins):
        raise ModelError(
            f"Multi-frame input has T={x_tchw.shape[0]} frames but the temporal range "
            f"yields {len(bins)} 30-day bins; frames must align with the binning."
        )

    frames: list[np.ndarray] = []
    timestamps: list[tuple[int, int, int]] = []
    dropped: list[tuple[str, str]] = []
    shape_meta: dict[str, Any] = {}
    for i, (start, end) in enumerate(bins):
        frame = x_tchw[i]
        if np.isnan(frame).all():
            dropped.append((start, end))
            continue  # empty bin sentinel
        prepped, shape_meta = _prepare_chw(
            np.nan_to_num(frame, nan=0.0).astype(np.float32),
            image_size=image_size,
            patch_size=patch_size,
            modality=modality,
            shape_adjust=shape_adjust,
        )
        frames.append(prepped)
        timestamps.append(_date_to_timestamp(start))
    if not frames:
        raise ModelError("All frames in the multi-frame OlmoEarth input are empty.")
    return np.stack(frames, axis=0), timestamps, dropped, shape_meta


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _build_sample(
    x_tchw: np.ndarray,
    *,
    timestamps: list[tuple[int, int, int]],
    modality: str = "s2",
):
    """Wrap a [T,C,H,W] stack as a batched MaskedOlmoEarthSample (B=1, T>=1)."""
    torch = _ensure_torch()
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (  # type: ignore
        MaskedOlmoEarthSample,
    )

    t, _, h, w = x_tchw.shape
    if len(timestamps) != t:
        raise ModelError(f"timestamps length {len(timestamps)} != T={t}.")
    # Model expects (B, H, W, T, C)
    data = torch.from_numpy(x_tchw).permute(2, 3, 0, 1).unsqueeze(0)  # [1,H,W,T,C]
    # Mask: all ONLINE_ENCODER (0)
    mask = torch.zeros(1, h, w, t, _modality_n_band_sets(modality), dtype=torch.float32)
    ts = torch.tensor(
        [[list(s) for s in timestamps]], dtype=torch.long
    )  # [1, T, 3]; must be int for month embedding
    field = _modality_field(modality)
    return MaskedOlmoEarthSample(
        timestamps=ts,
        **{field: data, f"{field}_mask": mask},
    )


def _build_batch_sample(
    x_btchw: np.ndarray,
    *,
    timestamps: list[list[tuple[int, int, int]]],
    modality: str = "s2",
):
    """Wrap a [B,T,C,H,W] batch as a batched MaskedOlmoEarthSample (B, T>=1)."""
    torch = _ensure_torch()
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (  # type: ignore
        MaskedOlmoEarthSample,
    )

    b, t, _, h, w = x_btchw.shape
    if len(timestamps) != b or any(len(item) != t for item in timestamps):
        raise ModelError(f"timestamps must be {b} lists of length T={t}.")
    # (B, H, W, T, C)
    data = torch.from_numpy(x_btchw).permute(0, 3, 4, 1, 2)
    mask = torch.zeros(b, h, w, t, _modality_n_band_sets(modality), dtype=torch.float32)
    ts = torch.tensor(
        [[list(s) for s in item] for item in timestamps], dtype=torch.long
    )  # [B, T, 3]; int for month embedding
    field = _modality_field(modality)
    return MaskedOlmoEarthSample(
        timestamps=ts,
        **{field: data, f"{field}_mask": mask},
    )


def _encoder_forward(model, sample, *, patch_size: int, device: str) -> Any:
    """Run the encoder forward pass. Returns the tokens_and_masks output."""
    torch = _ensure_torch()

    # Move sample tensors to device
    kwargs: dict[str, Any] = {}
    for field in sample._fields:
        val = getattr(sample, field)
        if val is not None:
            kwargs[field] = val.to(device)
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (  # type: ignore
        MaskedOlmoEarthSample,
    )

    sample_dev = MaskedOlmoEarthSample(**kwargs)
    with torch.no_grad():
        # load_model_from_id returns LatentMIM directly (.encoder is a direct attr).
        # OlmoEarthPretrain_v1 delegates via __getattr__, so .encoder works for both.
        encoder = model.encoder if hasattr(model, "encoder") else model.model.encoder
        output = encoder(sample_dev, fast_pass=True, patch_size=patch_size)
    return output["tokens_and_masks"]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _pool_tokens(tokens_and_masks, pooling: str) -> np.ndarray:
    """Pool spatial + temporal + band-set dims → (B, D) numpy array."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import (
        PoolingType,  # type: ignore
    )

    pt = PoolingType.MEAN if pooling == "mean" else PoolingType.MAX
    pooled = tokens_and_masks.pool_unmasked_tokens(pooling_type=pt)  # (B, D)
    return pooled.detach().float().cpu().numpy().astype(np.float32)


_FULL_ROI = (0.0, 1.0, 0.0, 1.0)


def _pool_with_roi(
    tokens_and_masks,
    pooling: str,
    *,
    modality: str,
    roi_window: tuple[float, float, float, float],
) -> np.ndarray:
    """Pool tokens to (B, D), restricting to the ROI's spatial tokens when the
    square was padded.

    When the ROI fills the frame (no padding) this is exactly the built-in
    ``pool_unmasked_tokens`` path. When the ROI is a centered sub-window of a
    padded square, the padded border tokens are excluded from the pool so they
    do not bias the embedding — the pooled equivalent of cropping the grid back
    to the ROI.
    """
    if roi_is_full(roi_window):
        return _pool_tokens(tokens_and_masks, pooling)
    tokens = _grid_tokens(tokens_and_masks, modality)  # (B, H', W', T, S, D)
    if pooling == "mean":
        sp = tokens.mean(dim=[3, 4])  # (B, H', W', D)
    else:
        sp = tokens.max(dim=4).values.max(dim=3).values
    _, h, w, _ = sp.shape
    y0, y1, x0, x1 = roi_token_box(roi_window, grid_h=h, grid_w=w)
    sp = sp[:, y0:y1, x0:x1, :]
    if pooling == "mean":
        v = sp.mean(dim=[1, 2])  # (B, D)
    else:
        v = sp.max(dim=2).values.max(dim=1).values
    return v.detach().float().cpu().numpy().astype(np.float32)


def _grid_tokens(tokens_and_masks, modality: str):
    """Return the (B, H', W', T, S, D) token tensor for a modality, or raise."""
    field = _modality_field(modality)
    tokens = getattr(tokens_and_masks, field, None)
    if tokens is None:
        raise ModelError(f"No {field} tokens in OlmoEarth encoder output.")
    return tokens


def _tokens_to_grid(tokens_and_masks, pooling: str, *, modality: str = "s2") -> np.ndarray:
    """Return spatial grid (D, H', W') by averaging over T and band-set dims."""
    tokens = _grid_tokens(tokens_and_masks, modality)  # (B, H', W', T, S, D)
    # pool over T (dim 3) and band-sets (dim 4)
    if pooling == "mean":
        spatial = tokens.mean(dim=[3, 4])  # (B, H', W', D)
    else:
        spatial = tokens.max(dim=4).values.max(dim=3).values  # (B, H', W', D)
    # Take first batch item and move dim to (D, H', W')
    grid = spatial[0].permute(2, 0, 1).detach().float().cpu().numpy().astype(np.float32)
    return grid


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


@register("olmoearth")
class OlmoEarthEmbedder(EmbedderBase):
    """OlmoEarth v1/v1.1/v1.2 on-the-fly embeddings from S2 L2A 12-band or S1 VV/VH patches.

    Inputs:
      - spatial : BBox / PointBuffer (EPSG:4326)
      - temporal: range / year (year → full year composite)
      - sensor  : controls provider fetch (scale/cloudy/composite) and
                  modality routing (``modality="s2"`` default, or ``"s1"``)

    Outputs:
      - pooled: global mean/max over spatial, temporal and band-set token dims
      - grid  : spatial token map [D, H', W'] averaged over temporal/band-set dims

    Modalities (via ``modality="s1"`` at the API level, like TerraFM):
      s2 : 12-band S2 L2A SR DN patches (default)
      s1 : 2-band S1 GRD VV/VH in dB; ``input_chw`` overrides must be dB

    Temporal modes (via ``temporal_mode="multi"``):
      single : one composite over the whole range, T=1 (default)
      multi  : 30-day bins anchored at range start (max 12 frames), mirroring
               OlmoEarth pretraining; per-frame timestamps are bin start dates
               and bins without imagery are dropped from the sequence

    Model variants (via ``model_config={"variant": "..."}``):
      v1  : nano (128-d), tiny (192-d), base (768-d), large (1024-d)
      v1.1: nano_v1_1 (128-d), tiny_v1_1 (192-d), base_v1_1 (768-d)
      v1.2: nano_v1_2, tiny_v1_2, small_v1_2, base_v1_2 (default; ``small``
            is v1.2-only). Embedding width is read from the model output, so
            new sizes need no dim wiring here.
    """

    # Square-input model: marks it for the API tiling path (its own fetch_input
    # squares the ROI for the single-input case and skips it when tiling).
    _requires_square_input = True
    DEFAULT_VARIANT = _DEFAULT_VARIANT
    DEFAULT_IMAGE_SIZE = _DEFAULT_IMAGE_SIZE
    DEFAULT_PATCH_SIZE = _DEFAULT_PATCH_SIZE
    DEFAULT_SCALE_M = _DEFAULT_SCALE_M
    DEFAULT_CLOUDY_PCT = _DEFAULT_CLOUDY_PCT
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 4
    DEFAULT_BATCH_CUDA = 16

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=_S2_BANDS_GEE,
        scale_m=_DEFAULT_SCALE_M,
        cloudy_pct=_DEFAULT_CLOUDY_PCT,
        expected_channels=_N_BANDS,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["provider"],
            "input_bands": list(_S2_BANDS_GEE),
            "modalities": {
                "s2": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": list(_S2_BANDS_GEE),
                },
                "s1": {
                    "collection": "COPERNICUS/S1_GRD",
                    "bands": list(_S1_BANDS_GEE),
                    "defaults": {
                        # OlmoEarth S1 normalization stats are in dB → fetch
                        # the dB collection by default (linear gets converted).
                        "use_float_linear": False,
                        "s1_require_iw": True,
                        "s1_relax_iw_on_empty": True,
                    },
                },
            },
            "output": ["pooled", "grid"],
            "defaults": {
                "variant": self.DEFAULT_VARIANT,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "patch_size": self.DEFAULT_PATCH_SIZE,
                "scale_m": self.DEFAULT_SCALE_M,
                "cloudy_pct": self.DEFAULT_CLOUDY_PCT,
                "modality": "s2",
            },
            "model_config": {
                "variant": {
                    "type": "string",
                    "default": self.DEFAULT_VARIANT,
                    "choices": sorted(_VARIANT_SPECS),
                    "description": (
                        "Model size/version. v1: nano, tiny, base, large. "
                        "v1.1: nano_v1_1, tiny_v1_1, base_v1_1."
                    ),
                },
                "patch_size": {
                    "type": "int",
                    "default": self.DEFAULT_PATCH_SIZE,
                    "description": "Patch size for FlexiViT encoder (1–8). Smaller = more tokens.",
                },
                "image_size": {
                    "type": "int",
                    "default": self.DEFAULT_IMAGE_SIZE,
                    "description": "Image resize target before encoding (pixels). Must be divisible by patch_size.",
                },
                "shape_adjust": {
                    "type": "string",
                    "default": _DEFAULT_SHAPE_ADJUST,
                    "choices": ["pad", "crop"],
                    "description": (
                        "How a non-square ROI is made square before encoding (OlmoEarth "
                        "needs a square token grid). 'pad' (default): reflect-pad the short "
                        "side to square, keeping the whole ROI; 'crop': center-crop to square. "
                        "Either avoids the aspect-ratio stretch that distorts rectangular ROIs. "
                        "Extremely rectangular inputs (aspect >= 2) fall back to a plain resize; "
                        "see meta['shape_prep']."
                    ),
                },
                "temporal_mode": {
                    "type": "string",
                    "default": _DEFAULT_TEMPORAL_MODE,
                    "choices": ["auto", "single", "multi"],
                    "description": (
                        "auto (default): single when the range spans one temporal bin, "
                        "else multi (≥2 bins). single: one composite over the whole range "
                        "(T=1). multi: 30-day bins anchored at range start (max 12 frames, "
                        "mirroring OlmoEarth pretraining; longer windows are equal-divided "
                        "into 12 frames with a warning); empty bins are dropped."
                    ),
                },
            },
            "notes": [
                "OlmoEarth is a Vision Transformer trained on the Major TOM dataset.",
                "Requires olmoearth-pretrain-minimal (pip install olmoearth-pretrain-minimal).",
                "Weights are downloaded automatically from Hugging Face on first use.",
                "Normalization: per-band mean±2σ clipping (OlmoEarth COMPUTED strategy).",
                "S1 modality via modality='s1' (VV/VH dB, COPERNICUS/S1_GRD).",
                "temporal_mode='multi' fetches one composite per 30-day bin (S1 and S2).",
                "Multi-frame bins with no usable imagery are dropped from the series; "
                "meta records n_bins/n_frames/dropped_bins and a UserWarning is emitted.",
            ],
        }

    def _default_sensor(self) -> SensorSpec:
        assert self.input_spec is not None
        return self.input_spec.to_sensor_spec()

    def fetch_input(
        self,
        provider: ProviderBase,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec,
        temporal_mode: str | None = None,
        square_input: bool = True,
    ) -> FetchResult | None:
        """Fetch raw OlmoEarth input, routing by ``sensor.modality`` (s2/s1).

        S1 values are always returned in dB: with ``use_float_linear=True``
        the linear-power fetch is converted via 10·log10, matching the dB
        statistics used by the OlmoEarth normalizer.

        ``temporal_mode="multi"`` fetches one composite per 30-day bin and
        returns ``[T,C,H,W]`` (empty bins are all-NaN sentinel frames; see
        the binned fetch helpers). When not given, the mode falls back to
        ``RS_EMBED_OLMOEARTH_TEMPORAL_MODE`` / ``"single"`` so generic
        pipeline callers keep working unchanged.
        """
        modality = _normalize_modality(getattr(sensor, "modality", "s2"))
        t = temporal_to_range(temporal)
        mode = (
            _normalize_temporal_mode(temporal_mode)
            if temporal_mode is not None
            else _resolve_temporal_mode(None)
        )
        mode = _expand_auto_mode(mode, t)

        # Fetch-square: enlarge a rectangular ROI to a square of real imagery so
        # the encoder gets a square, in-distribution input with real spatial
        # context (no synthetic pad). The ROI's window within the square travels
        # in meta['roi_window_geo']; the output is cropped back to it. Skipped
        # when the API tiles the input (square_input=False): tiling squares per
        # tile, so fetching the rectangular ROI directly avoids extra imagery.
        geo_roi = FULL_WINDOW
        if square_input:
            spatial, geo_roi = square_spatial(spatial)

        if mode == "multi":
            bins, _stretched = _temporal_bins(t)
            if modality == "s2":
                raw_t, meta = _fetch_collection_binned_raw_tchw(
                    provider,
                    spatial=spatial,
                    bins=bins,
                    collection=sensor.collection,
                    bands=_S2_BANDS_GEE,
                    scale_m=int(sensor.scale_m),
                    cloudy_pct=int(sensor.cloudy_pct),
                    composite=str(sensor.composite),
                    fill_value=float(sensor.fill_value),
                )
            else:
                raw_t, meta = _fetch_s1_vvvh_binned_raw_tchw(
                    provider,
                    spatial=spatial,
                    bins=bins,
                    scale_m=int(sensor.scale_m),
                    use_float_linear=bool(sensor.use_float_linear),
                    composite=str(sensor.composite),
                    fill_value=float(sensor.fill_value),
                    require_iw=bool(sensor.s1_require_iw),
                    relax_iw_on_empty=bool(sensor.s1_relax_iw_on_empty),
                )
                if bool(sensor.use_float_linear):
                    raw_t = _s1_linear_to_db(raw_t)  # NaN sentinel frames stay NaN
            meta["temporal_mode"] = "multi"
            meta.update(roi_fetch_meta(geo_roi) or {})
            return FetchResult(data=raw_t, meta=meta)

        if modality == "s2":
            raw = _fetch_collection_patch_chw(
                provider,
                spatial=spatial,
                temporal=t,
                collection=sensor.collection,
                bands=_S2_BANDS_GEE,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
            )
            s2_meta = roi_fetch_meta(geo_roi) or {}
            return FetchResult(data=raw, meta=s2_meta)
        raw, fmeta = _fetch_s1_vvvh_raw_chw_with_meta(
            provider,
            spatial=spatial,
            temporal=t,
            scale_m=int(sensor.scale_m),
            use_float_linear=bool(sensor.use_float_linear),
            composite=str(sensor.composite),
            fill_value=float(sensor.fill_value),
            require_iw=bool(sensor.s1_require_iw),
            relax_iw_on_empty=bool(sensor.s1_relax_iw_on_empty),
        )
        if bool(sensor.use_float_linear):
            raw = _s1_linear_to_db(raw)
        fmeta.update(roi_fetch_meta(geo_roi) or {})
        return FetchResult(data=raw, meta=fmeta)

    @staticmethod
    def _resolve_fetch_workers(n: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_OLMOEARTH_FETCH_WORKERS",
                str(OlmoEarthEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(n, v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default = (
            OlmoEarthEmbedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else OlmoEarthEmbedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_OLMOEARTH_BATCH_SIZE", str(default)))
        return max(1, v)

    # ------------------------------------------------------------------
    # Single embedding
    # ------------------------------------------------------------------

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
        fetch_meta: dict[str, Any] | None = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("olmoearth expects a provider backend (or 'auto').")

        # When the API prefetched the input (input_chw set), the fetch-square ROI
        # window rides along in fetch_meta['roi_window_geo']; without it a padded
        # prefetched square would not be cropped back to the ROI.
        prefetch_roi = (fetch_meta or {}).get("roi_window_geo")

        if sensor is None:
            sensor = self._default_sensor()
        modality = _normalize_modality(getattr(sensor, "modality", "s2"))
        t = temporal_to_range(temporal)
        temporal_mode = _expand_auto_mode(_resolve_temporal_mode(model_config), t)

        variant = _resolve_variant(model_config)
        image_size, patch_size = _resolve_geometry(model_config)
        shape_adjust = _resolve_shape_adjust(model_config)
        _, size, version = _VARIANT_SPECS[variant]

        model, wmeta, dev = _load_olmoearth(variant, device=device)

        fetch_meta_local: dict[str, Any] = {}
        geo_roi: tuple[float, float, float, float] | None = None
        if input_chw is None:
            provider = self._get_provider(backend)
            fr = self.fetch_input(
                provider,
                spatial=spatial,
                temporal=t,
                sensor=sensor,
                temporal_mode=temporal_mode,
            )
            assert fr is not None
            x_raw = np.asarray(fr.data, dtype=np.float32)
            fetch_meta_local = dict(fr.meta or {})
            geo_roi = fetch_meta_local.pop("roi_window_geo", None)
        else:
            n_bands = _modality_n_bands(modality)
            ok_3d = input_chw.ndim == 3 and input_chw.shape[0] == n_bands
            ok_4d = input_chw.ndim == 4 and input_chw.shape[1] == n_bands
            if not (ok_3d or ok_4d):
                raise ModelError(
                    f"input_chw must be CHW (or TCHW) with {n_bands} bands for "
                    f"olmoearth ({modality}), got {getattr(input_chw, 'shape', None)}."
                )
            x_raw = input_chw.astype(np.float32)
            # Prefetched square (API input-prep path): crop back to the ROI it covers.
            geo_roi = prefetch_roi

        # Frame layout is decided by the array shape so that arrays survive
        # array-only transport (prefetch pipelines, user-provided input_chw).
        date_str = temporal_midpoint_str(t)
        sampling_meta: dict[str, Any] = {}
        dropped_bins: list[tuple[str, str]] = []
        n_bins = 0
        if x_raw.ndim == 4:
            bins, stretched = _temporal_bins(t)
            n_bins = len(bins)
            sampling_meta = _temporal_sampling_meta(bins, stretched)
            x_t, timestamps, dropped_bins, shape_meta = _prepare_frames(
                x_raw,
                bins=bins,
                modality=modality,
                image_size=image_size,
                patch_size=patch_size,
                shape_adjust=shape_adjust,
            )
        else:
            x_prep, shape_meta = _prepare_chw(
                x_raw,
                image_size=image_size,
                patch_size=patch_size,
                modality=modality,
                shape_adjust=shape_adjust,
            )
            x_t = x_prep[np.newaxis]
            timestamps = [_date_to_timestamp(date_str)]

        # Fetch-square ROI (if the fetch enlarged a rectangle) overrides the
        # residual pad window as the effective crop target.
        shape_meta = _merge_geo_roi(shape_meta, geo_roi)

        sample = _build_sample(x_t, timestamps=timestamps, modality=modality)
        tokens_and_masks = _encoder_forward(model, sample, patch_size=patch_size, device=dev)

        # Report the mode actually used (decided by the array shape above), which
        # can differ from the configured one when input_chw overrides the fetch.
        effective_mode = "multi" if x_raw.ndim == 4 else "single"
        extra: dict[str, Any] = {"temporal_mode": effective_mode, "n_frames": len(timestamps)}
        extra.update(shape_meta)  # {"shape_prep": {...}}
        if effective_mode != temporal_mode:
            extra["requested_temporal_mode"] = temporal_mode
        if effective_mode == "multi":
            extra.update(sampling_meta)
            extra["n_bins"] = n_bins
            if dropped_bins:
                extra["dropped_bins"] = [list(b) for b in dropped_bins]
            _warn_stretched_sampling(sampling_meta)
            _warn_dropped_bins(dropped_bins, n_bins=n_bins, n_frames=len(timestamps))
        if fetch_meta_local:
            extra["s1_fetch" if modality == "s1" else "fetch"] = fetch_meta_local
        meta = _build_embedding_meta(
            model_name=self.model_name,
            wmeta=wmeta,
            variant=variant,
            size=size,
            version=version,
            backend=backend,
            sensor=sensor,
            temporal=t,
            image_size=int(x_t.shape[-1]),
            patch_size=patch_size,
            date_str=date_str,
            modality=modality,
            extra=extra,
        )

        roi_window = _roi_window_from_meta(shape_meta)

        if output.mode == "pooled":
            vec = _pool_with_roi(
                tokens_and_masks, output.pooling, modality=modality, roi_window=roi_window
            )[0]  # (D,)
            meta["pooling"] = f"spatial_temporal_bandset_{output.pooling}"
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            return _make_grid_embedding(
                tokens_and_masks, output, meta, modality=modality, roi_window=roi_window
            )

        raise ModelError(f"Unknown output mode: {output.mode!r}.")

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("olmoearth expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()
        t = temporal_to_range(temporal)
        temporal_mode = _expand_auto_mode(_resolve_temporal_mode(model_config), t)
        provider = self._get_provider(backend)
        n = len(spatials)
        prefetched: list[np.ndarray | None] = [None] * n
        geo_rois: list[tuple[float, float, float, float] | None] = [None] * n

        def _fetch_one(
            i: int, sp: SpatialSpec
        ) -> tuple[int, np.ndarray, tuple[float, float, float, float] | None]:
            fr = self.fetch_input(
                provider, spatial=sp, temporal=t, sensor=sensor, temporal_mode=temporal_mode
            )
            assert fr is not None
            return i, np.asarray(fr.data, dtype=np.float32), (fr.meta or {}).get("roi_window_geo")

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                ii, raw, gr = _fetch_one(i, sp)
                prefetched[ii] = raw
                geo_rois[ii] = gr
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    ii, raw, gr = fut.result()
                    prefetched[ii] = raw
                    geo_rois[ii] = gr

        raw_inputs: list[np.ndarray] = []
        for i, raw in enumerate(prefetched):
            if raw is None:
                raise ModelError(f"Missing prefetched input at index {i} for olmoearth.")
            raw_inputs.append(raw)

        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            temporal=temporal,
            sensor=sensor,
            model_config=model_config,
            output=output,
            backend=backend,
            device=device,
            _roi_windows_geo=geo_rois,
        )

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
        _roi_windows_geo: list[tuple[float, float, float, float] | None] | None = None,
        fetch_metas: list[dict[str, Any] | None] | None = None,
    ) -> list[Embedding]:
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}."
            )
        if not spatials:
            return []
        # Prefetched square inputs carry the ROI window in fetch_meta (the
        # export pipeline passes it via ``fetch_metas``); fold it into the
        # internal per-item ROI list so the output is cropped back to the ROI.
        if _roi_windows_geo is None and fetch_metas is not None:
            _roi_windows_geo = [(m or {}).get("roi_window_geo") for m in fetch_metas]
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("olmoearth expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()
        modality = _normalize_modality(getattr(sensor, "modality", "s2"))
        t = temporal_to_range(temporal)

        variant = _resolve_variant(model_config)
        image_size, patch_size = _resolve_geometry(model_config)
        shape_adjust = _resolve_shape_adjust(model_config)
        _, size, version = _VARIANT_SPECS[variant]

        model, wmeta, dev = _load_olmoearth(variant, device=device)
        infer_bs = self._resolve_infer_batch(dev)

        date_str = temporal_midpoint_str(t)

        # Prepare all inputs (normalize + resize) and group by shape.
        # Frame layout is decided per item by array shape: CHW → single frame,
        # TCHW → 30-day binned frames (empty bins arrive as all-NaN sentinels
        # and are dropped, so prepared T may vary per item).
        n_bands = _modality_n_bands(modality)
        bins: tuple[tuple[str, str], ...] | None = None
        sampling_meta: dict[str, Any] = {}  # shared by all multi items (same window)
        prepared: list[np.ndarray] = []  # each (T_i, C, H, W)
        item_timestamps: list[list[tuple[int, int, int]]] = []
        item_modes: list[str] = []  # effective temporal mode per item (by array shape)
        item_dropped: list[list[tuple[str, str]]] = []  # dropped bins per item
        item_shape_meta: list[dict[str, Any]] = []  # square-prep meta per item
        n_bins = 0  # number of 30-day bins from the window (shared by multi items)
        for i, x_raw in enumerate(input_chws):
            if x_raw.ndim == 3 and x_raw.shape[0] == n_bands:
                prep, sm = _prepare_chw(
                    x_raw.astype(np.float32),
                    image_size=image_size,
                    patch_size=patch_size,
                    modality=modality,
                    shape_adjust=shape_adjust,
                )
                prepared.append(prep[np.newaxis])
                item_timestamps.append([_date_to_timestamp(date_str)])
                item_modes.append("single")
                item_dropped.append([])
                item_shape_meta.append(sm)
            elif x_raw.ndim == 4 and x_raw.shape[1] == n_bands:
                if bins is None:
                    bins, stretched = _temporal_bins(t)
                    n_bins = len(bins)
                    sampling_meta = _temporal_sampling_meta(bins, stretched)
                    _warn_stretched_sampling(sampling_meta)  # once per batch
                x_t, ts, dropped, sm = _prepare_frames(
                    x_raw.astype(np.float32),
                    bins=bins,
                    modality=modality,
                    image_size=image_size,
                    patch_size=patch_size,
                    shape_adjust=shape_adjust,
                )
                prepared.append(x_t)
                item_timestamps.append(ts)
                item_modes.append("multi")
                item_dropped.append(dropped)
                item_shape_meta.append(sm)
            else:
                raise ModelError(
                    f"input_chw at index {i} must be CHW (or TCHW) with {n_bands} bands "
                    f"({modality}), got {getattr(x_raw, 'shape', None)}."
                )

        # Fetch-square ROIs (from the batch fetch path) override the residual pad
        # window per item; absent (user-supplied input_chws) leaves pad windows.
        if _roi_windows_geo is not None:
            for i in range(len(item_shape_meta)):
                item_shape_meta[i] = _merge_geo_roi(item_shape_meta[i], _roi_windows_geo[i])

        affected = [d for d in item_dropped if d]
        if affected and len({tuple(d) for d in affected}) == 1:
            # All affected inputs dropped the same bins — the common case when one
            # ROI is tiled into many sub-inputs. Report at the bin level so the
            # message doesn't leak tile/item counts.
            _warn_dropped_bins(affected[0], n_bins=n_bins, n_frames=n_bins - len(affected[0]))
        elif affected:
            # Genuinely different inputs (a multi-ROI batch) dropped different bins.
            warnings.warn(
                f"OlmoEarth dropped empty temporal bins for {len(affected)} of "
                f"{len(input_chws)} input(s) (no usable imagery in parts of the "
                "window); see each embedding's meta['dropped_bins'].",
                UserWarning,
                stacklevel=2,
            )

        shape_groups: dict[tuple[int, ...], list[int]] = {}
        for i, x in enumerate(prepared):
            shape_groups.setdefault(x.shape, []).append(i)

        xr_mod = None
        if output.mode == "grid":
            try:
                import xarray as xr  # noqa: PLC0415

                xr_mod = xr
            except ImportError as exc:
                raise ModelError(
                    "grid output requires xarray. Install: pip install xarray"
                ) from exc

        out: list[Embedding | None] = [None] * len(spatials)

        for idxs in shape_groups.values():
            for s0 in range(0, len(idxs), infer_bs):
                chunk = idxs[s0 : s0 + infer_bs]
                xb = np.stack([prepared[i] for i in chunk], axis=0)  # (B, T, C, H, W)
                sample = _build_batch_sample(
                    xb, timestamps=[item_timestamps[i] for i in chunk], modality=modality
                )
                tokens_and_masks = _encoder_forward(
                    model, sample, patch_size=patch_size, device=dev
                )

                if output.mode == "pooled":
                    pooled = _pool_tokens(tokens_and_masks, output.pooling)  # (B, D)
                    # Only compute spatial tokens if some item needs ROI cropping
                    # (a padded non-square ROI); full-frame items keep the exact
                    # built-in pool, matching the single-item path.
                    chunk_rois = [_roi_window_from_meta(item_shape_meta[i]) for i in chunk]
                    spatial_b = None
                    if any(not roi_is_full(rw) for rw in chunk_rois):
                        tk = _grid_tokens(tokens_and_masks, modality)  # (B,H',W',T,S,D)
                        if output.pooling == "mean":
                            spatial_b = tk.mean(dim=[3, 4]).detach().float().cpu()
                        else:
                            spatial_b = (
                                tk.max(dim=4).values.max(dim=3).values.detach().float().cpu()
                            )
                    for j, i in enumerate(chunk):
                        rw = chunk_rois[j]
                        if roi_is_full(rw):
                            vec = pooled[j]
                        else:
                            assert spatial_b is not None
                            sp = spatial_b[j]  # (H', W', D)
                            gh, gw, _ = sp.shape
                            y0, y1, x0, x1 = roi_token_box(rw, grid_h=gh, grid_w=gw)
                            sp = sp[y0:y1, x0:x1, :]
                            if output.pooling == "mean":
                                vec = sp.mean(dim=[0, 1]).numpy().astype(np.float32)
                            else:
                                vec = (
                                    sp.max(dim=1)
                                    .values.max(dim=0)
                                    .values.numpy()
                                    .astype(np.float32)
                                )
                        meta = _build_embedding_meta(
                            model_name=self.model_name,
                            wmeta=wmeta,
                            variant=variant,
                            size=size,
                            version=version,
                            backend=backend,
                            sensor=sensor,
                            temporal=t,
                            image_size=int(prepared[i].shape[-1]),
                            patch_size=patch_size,
                            date_str=date_str,
                            modality=modality,
                            extra={
                                "batch_infer": True,
                                "temporal_mode": item_modes[i],
                                "n_frames": len(item_timestamps[i]),
                                **item_shape_meta[i],
                                **_multi_meta_extra(
                                    item_modes[i], sampling_meta, n_bins, item_dropped[i]
                                ),
                            },
                        )
                        meta["pooling"] = f"spatial_temporal_bandset_{output.pooling}"
                        out[i] = Embedding(data=vec, meta=meta)
                    continue

                if output.mode == "grid":
                    tokens = _grid_tokens(tokens_and_masks, modality)  # (B, H', W', T, S, D)
                    if output.pooling == "mean":
                        spatial_b = tokens.mean(dim=[3, 4])  # (B, H', W', D)
                    else:
                        spatial_b = tokens.max(dim=4).values.max(dim=3).values
                    spatial_b = spatial_b.detach().float().cpu()
                    for j, i in enumerate(chunk):
                        grid_np = spatial_b[j].permute(2, 0, 1).numpy().astype(np.float32)
                        grid_np = crop_grid_to_roi(
                            grid_np, _roi_window_from_meta(item_shape_meta[i])
                        )
                        meta = _build_embedding_meta(
                            model_name=self.model_name,
                            wmeta=wmeta,
                            variant=variant,
                            size=size,
                            version=version,
                            backend=backend,
                            sensor=sensor,
                            temporal=t,
                            image_size=int(prepared[i].shape[-1]),
                            patch_size=patch_size,
                            date_str=date_str,
                            modality=modality,
                            extra={
                                "batch_infer": True,
                                "temporal_mode": item_modes[i],
                                "n_frames": len(item_timestamps[i]),
                                **item_shape_meta[i],
                                **_multi_meta_extra(
                                    item_modes[i], sampling_meta, n_bins, item_dropped[i]
                                ),
                            },
                        )
                        d, h, w = grid_np.shape
                        meta.update({"grid_hw": (h, w), "grid_kind": "spatial_tokens"})
                        assert xr_mod is not None
                        da = xr_mod.DataArray(
                            grid_np,
                            dims=("d", "y", "x"),
                            coords={"d": np.arange(d), "y": np.arange(h), "x": np.arange(w)},
                            name="embedding",
                            attrs=meta,
                        )
                        out[i] = Embedding(data=da, meta=meta)
                    continue

                raise ModelError(f"Unknown output mode: {output.mode!r}.")

        if any(e is None for e in out):
            raise ModelError("OlmoEarth batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]


# ---------------------------------------------------------------------------
# Internal helpers shared between single and batch paths
# ---------------------------------------------------------------------------


def _build_embedding_meta(
    *,
    model_name: str,
    wmeta: dict[str, Any],
    variant: str,
    size: str,
    version: str,
    backend: str,
    sensor: SensorSpec,
    temporal: TemporalSpec,
    image_size: int,
    patch_size: int,
    date_str: str | None,
    modality: str = "s2",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = build_meta(
        model=model_name,
        kind="on_the_fly",
        backend=str(backend).lower(),
        source=sensor.collection,
        sensor=sensor,
        temporal=temporal,
        image_size=image_size,
    )
    meta.update(
        {
            "hf_repo": wmeta.get("hf_repo"),
            "variant": variant,
            "model_size": size,
            "model_version": version,
            "patch_size": patch_size,
            "modality": modality,
            "temporal_range": (temporal.start, temporal.end),
            "date_str": date_str,
        }
    )
    if extra:
        meta.update(extra)
    return meta


def _roi_window_from_meta(extra: dict[str, Any] | None) -> tuple[float, float, float, float]:
    """Pull the ROI window out of the ``shape_prep`` meta, defaulting to full."""
    sp = (extra or {}).get("shape_prep") or {}
    rw = sp.get("roi_window")
    if rw is None or len(tuple(rw)) != 4:
        return _FULL_ROI
    return tuple(float(v) for v in rw)  # type: ignore[return-value]


def _merge_geo_roi(
    shape_meta: dict[str, Any], geo_roi: tuple[float, float, float, float] | None
) -> dict[str, Any]:
    """Make a fetch-square geo ROI the effective ``roi_window`` in shape_meta.

    When the fetch enlarged a rectangular ROI to a square of real imagery, the
    ROI's window within that square (``geo_roi``) is the crop target — it
    overrides the residual pad window from squaring the fetched pixels. A full or
    missing geo ROI leaves shape_meta untouched (input_chw / already-square fetch).
    """
    if geo_roi is None or roi_is_full(tuple(geo_roi)):
        return shape_meta
    sp = dict(shape_meta.get("shape_prep") or {})
    sp["roi_window"] = tuple(float(v) for v in geo_roi)
    sp["roi_source"] = "fetch_square"
    return {**shape_meta, "shape_prep": sp}


def _make_grid_embedding(
    tokens_and_masks: Any,
    output: OutputSpec,
    meta: dict[str, Any],
    *,
    modality: str = "s2",
    roi_window: tuple[float, float, float, float] = _FULL_ROI,
) -> Embedding:
    try:
        import xarray as xr  # noqa: PLC0415
    except ImportError as exc:
        raise ModelError("grid output requires xarray. Install: pip install xarray") from exc

    grid = _tokens_to_grid(tokens_and_masks, output.pooling, modality=modality)  # (D, H', W')
    grid = crop_grid_to_roi(grid, roi_window)  # crop padded border back to the ROI
    d, h, w = grid.shape
    meta.update({"grid_hw": (h, w), "grid_kind": "spatial_tokens"})
    da = xr.DataArray(
        grid,
        dims=("d", "y", "x"),
        coords={"d": np.arange(d), "y": np.arange(h), "x": np.arange(w)},
        name="embedding",
        attrs=meta,
    )
    return Embedding(data=da, meta=meta)

from __future__ import annotations

import json
import math
import os
import warnings
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
    count_distinct_frames,
    frame_diversity_meta,
)
from ..providers.fetch import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
)
from ..providers.fetch import (
    fetch_s2_multiframe_raw_tchw as _fetch_s2_multiframe_raw_tchw,
)
from ..providers.resolution import (
    is_provider_backend,
)
from ..tools.runtime import (
    load_cached_with_device as _load_cached_with_device,
)
from ..tools.shape import (
    crop_grid_to_roi,
    geo_roi_from_meta,
    prepare_square,
    roi_fetch_meta,
    roi_is_full,
    square_fetch_batch,
)
from ..tools.spatial import FULL_WINDOW, square_spatial
from ..tools.temporal import temporal_frame_midpoints as _temporal_frame_midpoints
from .base import EmbedderBase
from .config import model_config_value
from .meta import build_meta, temporal_midpoint_str, temporal_to_range


def ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise ModelError("This embedder requires torch installed.") from e


def base_meta(
    *,
    model_name,
    hf_id,
    backend,
    image_size,
    sensor,
    temporal=None,
    source=None,
    embed_type="on_the_fly",
    extra=None,
):
    m = build_meta(
        model=model_name,
        kind=embed_type,
        backend=backend,
        source=source or getattr(sensor, "collection", None),
        sensor=sensor,
        temporal=temporal,
        image_size=image_size,
    )
    m["hf_id"] = hf_id
    if extra:
        m.update(extra)
    return m


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


def _split_prithvi_patch_tokens(tokens, n_frames: int):
    """Split a token sequence into per-frame square patch grids.

    Prithvi returns CLS + ``T·(h·w)`` patch tokens in ``(t h w)`` order (see the
    vendored ``prepare_features_for_image_model``). Returns
    ``(patch[T, h, w, D], has_cls, (h, w))``.
    """
    n = int(tokens.shape[0])
    d = int(tokens.shape[1])
    nf = max(1, int(n_frames))
    for has_cls in (True, False):
        n_patch = n - 1 if has_cls else n
        if n_patch <= 0 or n_patch % nf != 0:
            continue
        per_frame = n_patch // nf
        hw = int(round(per_frame**0.5))
        if hw >= 1 and hw * hw == per_frame:
            patch = tokens[1:] if has_cls else tokens
            return patch.reshape(nf, hw, hw, d).astype("float32"), has_cls, (hw, hw)
    raise ModelError(f"Cannot split Prithvi tokens (N={n}) into {nf} frames of square patches.")


def pool_from_tokens_tchw(tokens, n_frames: int, pooling: str):
    """Pool patch tokens over time + space → (D,). Excludes CLS if present."""
    patch, has_cls, _ = _split_prithvi_patch_tokens(tokens, n_frames)  # [T,h,w,D]
    flat = patch.reshape(-1, patch.shape[-1])
    if flat.shape[0] == 0:
        return tokens[0].astype("float32"), has_cls
    if pooling == "mean":
        return flat.mean(axis=0).astype("float32"), has_cls
    if pooling == "max":
        return flat.max(axis=0).astype("float32"), has_cls
    raise ModelError(f"Unknown pooling={pooling!r} (expected 'mean' or 'max').")


def tokens_to_grid_dhw_tchw(tokens, n_frames: int, pooling: str):
    """Reduce per-frame patch grids over time → grid [D,h,w] and (h,w)."""
    patch, has_cls, (h, w) = _split_prithvi_patch_tokens(tokens, n_frames)  # [T,h,w,D]
    if pooling == "mean":
        g = patch.mean(axis=0)  # [h,w,D]
    elif pooling == "max":
        g = patch.max(axis=0)
    else:
        raise ModelError(f"Unknown pooling={pooling!r} (expected 'mean' or 'max').")
    return g.transpose(2, 0, 1).astype("float32"), (h, w), has_cls


# -------------------------
# Provider: Sentinel-2 -> Prithvi 6-band (CHW float32 in [0,1])
# -------------------------
PRITHVI_S2_BANDS_SRC = ["B2", "B3", "B4", "B8", "B11", "B12"]
PRITHVI_S2_BANDS_DST = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
_PRITHVI_VARIANT_TO_MODEL_KEY = {
    "prithvi_eo_v2_100_tl": "prithvi_eo_v2_100_tl",
    "100_tl": "prithvi_eo_v2_100_tl",
    "100m_tl": "prithvi_eo_v2_100_tl",
    "prithvi_eo_v2_300_tl": "prithvi_eo_v2_300_tl",
    "300_tl": "prithvi_eo_v2_300_tl",
    "300m_tl": "prithvi_eo_v2_300_tl",
    "prithvi_eo_v2_600_tl": "prithvi_eo_v2_600_tl",
    "600_tl": "prithvi_eo_v2_600_tl",
    "600m_tl": "prithvi_eo_v2_600_tl",
}
_PRITHVI_HF_SPECS = {
    "prithvi_eo_v2_100_tl": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL",
        "checkpoint": "Prithvi_EO_V2_100M_TL.pt",
    },
    "prithvi_eo_v2_300_tl": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
        "checkpoint": "Prithvi_EO_V2_300M_TL.pt",
    },
    "prithvi_eo_v2_600_tl": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
        "checkpoint": "Prithvi_EO_V2_600M_TL.pt",
    },
}


# -------------------------
# Temporal (multi-frame) configuration
# -------------------------
# Prithvi-EO-2.0 was pretrained on 4-timestep HLS series with consecutive-frame
# gaps of ~1–6 months (≈28–184 days; arXiv:2412.02732). We therefore:
#   - default to single-frame (median composite) for backward compatibility;
#   - in multi mode, derive T from the requested window with a ~28-day minimum
#     spacing (low end of the training interval), capped at 4 frames;
#   - never duplicate frames to reach a target T — identical frames returned by
#     the provider (empty sub-windows are back-filled with the whole-window
#     composite) are collapsed, so windows lacking temporal diversity degrade
#     gracefully to T=1;
#   - flag (not silently fix) frame gaps beyond the ~184-day training maximum,
#     which can only occur for very long windows once T is capped at 4.
# "auto" (default) picks single/multi from the window: multi when the window
# yields ≥2 frames, single otherwise. Use "single"/"multi" to force a mode.
_DEFAULT_TEMPORAL_MODE = "auto"
_DEFAULT_MAX_FRAMES = 4  # matches Prithvi-EO-2.0 pretraining (4 timesteps)
_DEFAULT_FRAME_STRIDE_DAYS = 28  # min spacing = low end of training's 1–6 month gap
_DEFAULT_MAX_FRAME_STRIDE_DAYS = 184  # max gap (~6 months) seen in pretraining


def _normalize_temporal_mode(mode: Any) -> str:
    m = str(mode if mode is not None else _DEFAULT_TEMPORAL_MODE).strip().lower()
    if m not in ("single", "multi", "auto"):
        raise ModelError(
            f"prithvi temporal_mode must be 'single', 'multi', or 'auto', got {mode!r}."
        )
    return m


def _resolve_temporal_mode(model_config: dict[str, Any] | None) -> str:
    """Resolve the *configured* temporal mode (may be ``"auto"``)."""
    v = model_config_value(model_config, "temporal_mode")
    if v is not None:
        return _normalize_temporal_mode(v)
    env = os.environ.get("RS_EMBED_PRITHVI_TEMPORAL_MODE", "").strip()
    if env:
        return _normalize_temporal_mode(env)
    return _DEFAULT_TEMPORAL_MODE


def _effective_temporal_mode(model_config: dict[str, Any] | None, temporal: TemporalSpec) -> str:
    """Resolve single/multi, expanding ``"auto"`` from the window.

    ``auto`` → ``"multi"`` when the window yields ≥2 frames
    (``T = clamp(window_days // stride, 1, max_frames) >= 2``), else ``"single"``
    (a 1-frame window where multi adds nothing but fetch cost).
    """
    mode = _resolve_temporal_mode(model_config)
    if mode != "auto":
        return mode
    n = _auto_num_frames(
        temporal,
        max_frames=_resolve_max_frames(model_config),
        stride_days=_resolve_frame_stride_days(),
    )
    return "multi" if n >= 2 else "single"


def _resolve_max_frames(model_config: dict[str, Any] | None) -> int:
    v = model_config_value(model_config, "max_frames")
    if v is None:
        v = model_config_value(model_config, "n_frames")
    if v is None:
        env = os.environ.get("RS_EMBED_PRITHVI_MAX_FRAMES", "").strip()
        v = env or _DEFAULT_MAX_FRAMES
    try:
        n = int(v)
    except (TypeError, ValueError) as exc:
        raise ModelError(f"prithvi max_frames must be an integer, got {v!r}.") from exc
    if n < 1:
        raise ModelError(f"prithvi max_frames must be >= 1, got {n}.")
    return n


def _resolve_frame_stride_days() -> int:
    env = os.environ.get("RS_EMBED_PRITHVI_FRAME_STRIDE_DAYS", "").strip()
    if not env:
        return _DEFAULT_FRAME_STRIDE_DAYS
    try:
        n = int(env)
    except ValueError as exc:
        raise ModelError(
            f"RS_EMBED_PRITHVI_FRAME_STRIDE_DAYS must be an integer, got {env!r}."
        ) from exc
    return max(1, n)


def _auto_num_frames(temporal: TemporalSpec, *, max_frames: int, stride_days: int) -> int:
    """Pick a frame count from the window length: clamp(window_days // stride, 1, max_frames).

    Small windows (no room for month-spaced frames) collapse to T=1; large windows
    cap at ``max_frames`` so spacing stays within Prithvi's training regime.
    """
    from datetime import date as _date

    start = getattr(temporal, "start", None)
    end = getattr(temporal, "end", None)
    if not start or not end:
        return 1
    window_days = max(1, (_date.fromisoformat(str(end)) - _date.fromisoformat(str(start))).days)
    n = window_days // max(1, int(stride_days))
    return int(max(1, min(int(max_frames), n)))


def _resolve_max_frame_stride_days() -> int:
    env = os.environ.get("RS_EMBED_PRITHVI_MAX_STRIDE_DAYS", "").strip()
    if not env:
        return _DEFAULT_MAX_FRAME_STRIDE_DAYS
    try:
        n = int(env)
    except ValueError as exc:
        raise ModelError(
            f"RS_EMBED_PRITHVI_MAX_STRIDE_DAYS must be an integer, got {env!r}."
        ) from exc
    return max(1, n)


def _max_consecutive_gap_days(dates: list[str]) -> int:
    """Largest gap (in days) between consecutive frame dates; 0 for < 2 frames."""
    from datetime import date as _date

    if len(dates) < 2:
        return 0
    ds = sorted(_date.fromisoformat(str(d)) for d in dates)
    return max((ds[i + 1] - ds[i]).days for i in range(len(ds) - 1))


def _temporal_spacing_meta(
    dates: list[str], *, max_stride_days: int, label: str = ""
) -> dict[str, Any]:
    """Report the largest frame gap and flag/warn when it exceeds the training max.

    Prithvi-EO-2.0 saw ~1–6 month gaps in pretraining; gaps beyond ``max_stride_days``
    (~184 d ≈ 6 months) are out of distribution. This only *reports* it (meta flag +
    warning) and never truncates the window, so the embedding still represents the
    full requested period.
    """
    gap = _max_consecutive_gap_days(dates)
    meta: dict[str, Any] = {"max_frame_gap_days": int(gap)}
    if gap > int(max_stride_days):
        meta["temporal_spacing_out_of_range"] = True
        warnings.warn(
            f"Prithvi multi-frame spacing {gap}d exceeds the ~{int(max_stride_days)}d "
            f"(~6 month) maximum interval seen in pretraining"
            f"{(' for ' + label) if label else ''}; embeddings may be extrapolated. "
            "Shorten the temporal window to bring frame spacing back into range.",
            UserWarning,
            stacklevel=2,
        )
    return meta


def _normalize_prithvi_variant(variant: Any) -> str:
    raw = str(variant).strip().lower()
    resolved = _PRITHVI_VARIANT_TO_MODEL_KEY.get(raw)
    if resolved is None:
        raise ModelError(
            f"Unknown Prithvi variant='{variant}' "
            "(expected one of: prithvi_eo_v2_100_tl, prithvi_eo_v2_300_tl, prithvi_eo_v2_600_tl)."
        )
    return resolved


def _resolve_prithvi_model_key(
    *,
    model_config: dict[str, Any] | None,
    default_model_key: str,
) -> tuple[str, str]:
    variant_v = model_config_value(model_config, "variant")
    if variant_v is not None:
        model_key = _normalize_prithvi_variant(variant_v)
        return model_key, model_key

    model_key = (
        os.environ.get("RS_EMBED_PRITHVI_KEY", default_model_key).strip() or default_model_key
    )
    return str(model_key), str(model_key)


def _resolve_prithvi_hf_spec(model_key: str) -> dict[str, str]:
    spec = _PRITHVI_HF_SPECS.get(str(model_key).strip())
    if spec is None:
        raise ModelError(
            f"Unknown Prithvi model_key='{model_key}' "
            "(expected one of: prithvi_eo_v2_100_tl, prithvi_eo_v2_300_tl, prithvi_eo_v2_600_tl)."
        )
    return dict(spec)


def _prithvi_cache_dir() -> str | None:
    raw = str(os.environ.get("RS_EMBED_PRITHVI_CACHE_DIR", "")).strip()
    return raw or None


def _torch_load_checkpoint_compat(path: str):
    ensure_torch()
    import torch

    weights_only = str(os.environ.get("RS_EMBED_PRITHVI_WEIGHTS_ONLY", "1")).strip() not in (
        "0",
        "false",
        "False",
    )
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


@lru_cache(maxsize=16)
def _download_prithvi_file(repo_id: str, filename: str, cache_dir: str | None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise ModelError(
            "Prithvi checkpoint download requires huggingface_hub. "
            "Install: pip install huggingface_hub"
        ) from e
    return str(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir))


def _load_prithvi_module():
    from ._vendor.prithvi_mae import PrithviMAE

    return PrithviMAE


def _fetch_s2_prithvi6_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 30,
    cloudy_pct: int = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Returns CHW float32 [6,H,W] normalized to [0,1] from S2 SR (scaled by 1/10000).
    Uses provider.get_region_3857(spatial) to define the sampling rectangle.
    """
    # Use semantic aliases (BLUE/GREEN/...) so provider alias resolution stays centralized.
    x_chw = _fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(PRITHVI_S2_BANDS_DST),
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        fill_value=fill_value,
    )

    # S2 SR scaling: 0..10000
    x_chw = x_chw / 10000.0
    x_chw = np.clip(x_chw, 0.0, 1.0)
    x_chw = np.nan_to_num(x_chw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x_chw


def _fetch_s2_prithvi6_tchw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    n_frames: int,
    scale_m: int = 30,
    cloudy_pct: int = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch an S2 6-band time series as raw float32 [T,6,H,W] in [0,10000].

    Bins align with ``temporal_frame_midpoints`` (both use the shared
    ``split_date_range``), so frame ``i`` corresponds to midpoint date ``i``.
    """
    return _fetch_s2_multiframe_raw_tchw(
        provider,
        spatial=spatial,
        temporal=temporal,
        bands=PRITHVI_S2_BANDS_DST,
        n_frames=int(n_frames),
        collection="COPERNICUS/S2_SR_HARMONIZED",
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        fill_value=fill_value,
    )


def _raw_tchw_to_unit(x_tchw: np.ndarray) -> np.ndarray:
    """Scale raw S2 SR series [T,6,H,W] in 0..10000 to [0,1] float32."""
    x = np.asarray(x_tchw, dtype=np.float32) / 10000.0
    x = np.clip(x, 0.0, 1.0)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _drop_duplicate_frames(x_tchw: np.ndarray, dates: list[str]) -> tuple[np.ndarray, list[str]]:
    """Collapse bit-identical frames, keeping first occurrence and its date.

    The provider back-fills empty sub-windows with the whole-window composite,
    yielding identical frames; dropping them avoids feeding Prithvi duplicate
    timesteps and lets diversity-free windows degrade to T=1.
    """
    if x_tchw.ndim != 4:
        raise ModelError(f"Expected [T,C,H,W], got {getattr(x_tchw, 'shape', None)}")
    kept: list[np.ndarray] = []
    kept_dates: list[str] = []
    for i in range(int(x_tchw.shape[0])):
        frame = x_tchw[i]
        if any(np.array_equal(frame, k) for k in kept):
            continue
        kept.append(frame)
        kept_dates.append(dates[i] if i < len(dates) else dates[-1])
    return np.stack(kept, axis=0).astype(np.float32), kept_dates


def _prepare_prithvi_tchw(
    x_tchw: np.ndarray,
    *,
    fill_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the per-frame Prithvi prep (resize/pad) across a [T,C,H,W] stack."""
    if x_tchw.ndim != 4:
        raise ModelError(f"Expected [T,C,H,W], got {getattr(x_tchw, 'shape', None)}")
    frames: list[np.ndarray] = []
    prep_meta: dict[str, Any] = {}
    for i in range(int(x_tchw.shape[0])):
        y, prep_meta = _prepare_prithvi_chw(x_tchw[i], fill_value=fill_value)
        frames.append(y)
    return np.stack(frames, axis=0).astype(np.float32), prep_meta


def _pad_chw_to_multiple(x_chw: np.ndarray, mult: int = 16, value: float = 0.0) -> np.ndarray:
    """
    Pad CHW to make H and W divisible by mult.
    Pads on bottom and right only.
    """
    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW, got {x_chw.shape}")
    c, h, w = x_chw.shape
    nh = int(math.ceil(h / mult) * mult)
    nw = int(math.ceil(w / mult) * mult)
    if nh == h and nw == w:
        return x_chw
    out = np.full((c, nh, nw), float(value), dtype=np.float32)
    out[:, :h, :w] = x_chw.astype(np.float32)
    return out


def _resize_chw(x_chw: np.ndarray, *, size: int = 224) -> np.ndarray:
    """Make CHW square ``(size,size)`` without aspect-ratio distortion.

    A rectangular ROI is reflect-padded to square before resizing (see
    :mod:`rs_embed.tools.shape`) rather than stretched. This is the ``resize``
    prep mode; the separate ``pad`` mode (:func:`_pad_chw_to_multiple`) is
    unchanged.
    """
    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW array, got {x_chw.shape}")
    out, _ = prepare_square(x_chw, size=int(size), shape_adjust="pad")
    return out


def _prepare_prithvi_chw(
    x_chw: np.ndarray,
    *,
    fill_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Prepare CHW input before Prithvi forward.

    RS_EMBED_PRITHVI_PREP:
      - "resize": resize to RS_EMBED_PRITHVI_IMG (default 224)
      - "pad": pad H/W to RS_EMBED_PRITHVI_PATCH_MULT (default 16, legacy behavior)
    """
    prep = os.environ.get("RS_EMBED_PRITHVI_PREP", "resize").strip().lower()
    patch_mult = max(1, int(os.environ.get("RS_EMBED_PRITHVI_PATCH_MULT", "16")))
    target_size = max(16, int(os.environ.get("RS_EMBED_PRITHVI_IMG", "224")))

    if prep == "resize":
        y = _resize_chw(x_chw, size=target_size)
    elif prep == "pad":
        y = _pad_chw_to_multiple(x_chw, mult=patch_mult, value=float(fill_value))
    else:
        raise ModelError(f"Unknown RS_EMBED_PRITHVI_PREP='{prep}'. Use 'resize' or 'pad'.")

    return y, {
        "prep_mode": prep,
        "patch_mult": int(patch_mult),
        "target_image_size": int(target_size),
    }


def _spatial_center_lon_lat(spatial: SpatialSpec) -> tuple[float, float]:
    from ..core.specs import BBox, PointBuffer  # local import to avoid cycles

    if isinstance(spatial, BBox):
        spatial.validate()
        lon = (spatial.minlon + spatial.maxlon) / 2
        lat = (spatial.minlat + spatial.maxlat) / 2
        return float(lon), float(lat)
    if isinstance(spatial, PointBuffer):
        spatial.validate()
        return float(spatial.lon), float(spatial.lat)
    raise ModelError(f"Unsupported SpatialSpec: {type(spatial)}")


# -------------------------
# Prithvi model loading (vendored HF runtime)
# -------------------------


@lru_cache(maxsize=8)
def _load_prithvi_cached(
    model_key: str,
    pretrained: bool,
    bands: tuple[str, ...],
    num_frames: int,
    coords_encoding: tuple[str, ...],
    dev: str,
):
    ensure_torch()
    spec = _resolve_prithvi_hf_spec(model_key)
    cache_dir = _prithvi_cache_dir()
    cfg_path = _download_prithvi_file(spec["repo_id"], "config.json", cache_dir)
    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f).get("pretrained_cfg", {})

    if not isinstance(config, dict) or not config:
        raise ModelError(f"Invalid Prithvi config at {cfg_path!r}.")

    config = dict(config)
    config["num_frames"] = int(num_frames)
    config["in_chans"] = int(len(bands))
    config["coords_encoding"] = list(coords_encoding)
    if isinstance(config.get("patch_size"), list):
        config["patch_size"] = tuple(int(v) for v in config["patch_size"])

    try:
        PrithviMAE = _load_prithvi_module()
        m = PrithviMAE(**config)
    except Exception as e:
        raise ModelError(
            f"Failed to initialize vendored Prithvi runtime for '{model_key}': "
            f"{type(e).__name__}: {e}"
        ) from e

    ckpt_path = None
    if pretrained:
        ckpt_path = _download_prithvi_file(spec["repo_id"], spec["checkpoint"], cache_dir)
        state_dict = _torch_load_checkpoint_compat(ckpt_path)
        if not isinstance(state_dict, dict):
            raise ModelError(
                f"Unexpected Prithvi checkpoint object from {ckpt_path!r}: {type(state_dict)}"
            )
        state_dict = dict(state_dict)
        # HF inference.py replaces fixed positional embeddings with runtime-sized tensors.
        for key in list(state_dict.keys()):
            if key == "encoder.pos_embed":
                state_dict[key] = m.encoder.pos_embed
            elif key == "decoder.decoder_pos_embed":
                state_dict[key] = m.decoder.decoder_pos_embed
        try:
            m.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ModelError(
                f"Failed to load Prithvi checkpoint '{ckpt_path}': {type(e).__name__}: {e}"
            ) from e

    try:
        m = m.to(dev).eval()
    except Exception as _e:
        pass

    meta = {
        "model_key": model_key,
        "repo_id": spec["repo_id"],
        "checkpoint": spec["checkpoint"],
        "config_path": cfg_path,
        "checkpoint_path": ckpt_path,
        "pretrained": bool(pretrained),
        "bands": tuple(bands),
        "num_frames": int(num_frames),
        "coords_encoding": tuple(coords_encoding),
        "device": dev,
    }
    return m, meta


def _load_prithvi(
    model_key: str,
    *,
    pretrained: bool,
    bands: tuple[str, ...],
    num_frames: int,
    coords_encoding: tuple[str, ...],
    device: str = "auto",
):
    """Load (and cache) a vendored Prithvi backbone.

    Returns: (model, meta, resolved_device)
    """
    (loaded, dev) = _load_cached_with_device(
        _load_prithvi_cached,
        device=device,
        model_key=model_key,
        pretrained=bool(pretrained),
        bands=tuple(bands),
        num_frames=int(num_frames),
        coords_encoding=tuple(coords_encoding),
    )
    m, meta = loaded
    return m, meta, dev


def _prithvi_forward_tokens(
    model,
    x_chw: np.ndarray,
    *,
    lon: float,
    lat: float,
    date_str: str,
    device: str,
) -> np.ndarray:
    """
    Run Prithvi forward and return token sequence [N,D] (may include CLS).
    """
    ensure_torch()
    import pandas as pd
    import torch

    if x_chw.ndim != 3 or x_chw.shape[0] != 6:
        raise ModelError(f"Prithvi expects 6-band CHW, got {x_chw.shape}")

    x = torch.from_numpy(x_chw).unsqueeze(0).to(device)  # [1,6,H,W]

    d = pd.to_datetime(date_str)
    temporal_coords = torch.tensor(
        [[[float(d.year), float(d.dayofyear)]]], dtype=torch.float32, device=device
    )  # [1,1,2]
    # Vendored Prithvi runtime expects location_coords in (lat, lon) order.
    location_coords = torch.tensor(
        [[float(lat), float(lon)]], dtype=torch.float32, device=device
    )  # [1,2]

    with torch.no_grad():
        if hasattr(model, "forward_features"):
            out = model.forward_features(
                x,
                temporal_coords=temporal_coords,
                location_coords=location_coords,
            )
        else:
            out = model(x, temporal_coords=temporal_coords, location_coords=location_coords)

    # normalize output -> tokens
    tokens = None
    if isinstance(out, (tuple, list)):
        tokens = out[-1]
    elif hasattr(out, "last_hidden_state"):
        tokens = out.last_hidden_state
    elif isinstance(out, dict):
        tokens = out.get("tokens") or out.get("last_hidden_state") or out.get("hidden_states")
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[-1]
    else:
        tokens = out

    if tokens is None:
        raise ModelError("Prithvi forward did not return tokens.")

    if hasattr(tokens, "ndim") and tokens.ndim == 3:
        # [B,N,D]
        return tokens[0].detach().float().cpu().numpy().astype(np.float32)

    raise ModelError(
        f"Unexpected Prithvi tokens shape/type: {type(tokens)} {getattr(tokens, 'shape', None)}"
    )


def _prithvi_forward_tokens_batch(
    model,
    x_bchw: np.ndarray,
    *,
    lon_lat_batch: list[tuple[float, float]],
    date_str_batch: list[str],
    device: str,
) -> list[np.ndarray]:
    """Batch Prithvi forward for [B,6,H,W] inputs."""
    ensure_torch()
    import pandas as pd
    import torch

    if x_bchw.ndim != 4 or x_bchw.shape[1] != 6:
        raise ModelError(f"Prithvi expects BCHW with C=6, got {x_bchw.shape}")
    bsz = int(x_bchw.shape[0])
    if len(lon_lat_batch) != bsz or len(date_str_batch) != bsz:
        raise ModelError("lon_lat_batch/date_str_batch size mismatch with input batch.")

    xb = torch.from_numpy(x_bchw).to(device)
    tcoords = []
    lcoords = []
    for i in range(bsz):
        d = pd.to_datetime(date_str_batch[i])
        tcoords.append([float(d.year), float(d.dayofyear)])
        lon, lat = lon_lat_batch[i]
        lcoords.append([float(lat), float(lon)])
    temporal_coords = torch.tensor(tcoords, dtype=torch.float32, device=device).unsqueeze(
        1
    )  # [B,1,2]
    location_coords = torch.tensor(lcoords, dtype=torch.float32, device=device)  # [B,2]

    with torch.inference_mode():
        if hasattr(model, "forward_features"):
            out = model.forward_features(
                xb,
                temporal_coords=temporal_coords,
                location_coords=location_coords,
            )
        else:
            out = model(xb, temporal_coords=temporal_coords, location_coords=location_coords)

    tokens = None
    if isinstance(out, (tuple, list)):
        tokens = out[-1]
    elif hasattr(out, "last_hidden_state"):
        tokens = out.last_hidden_state
    elif isinstance(out, dict):
        tokens = out.get("tokens") or out.get("last_hidden_state") or out.get("hidden_states")
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[-1]
    else:
        tokens = out

    if tokens is None:
        raise ModelError("Prithvi forward did not return tokens.")
    if (not hasattr(tokens, "ndim")) or int(tokens.ndim) != 3:
        raise ModelError(
            f"Unexpected Prithvi batch tokens shape/type: {type(tokens)} {getattr(tokens, 'shape', None)}"
        )
    if int(tokens.shape[0]) != bsz:
        raise ModelError(f"Prithvi batch mismatch: got B={int(tokens.shape[0])}, expected {bsz}")

    toks_np = tokens.detach().float().cpu().numpy().astype(np.float32)  # [B,N,D]
    return [toks_np[i] for i in range(bsz)]


def _extract_tokens(out):
    """Normalize a Prithvi forward output into a token tensor."""
    if isinstance(out, (tuple, list)):
        return out[-1]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, dict):
        tok = out.get("tokens") or out.get("last_hidden_state") or out.get("hidden_states")
        return tok[-1] if isinstance(tok, (tuple, list)) else tok
    return out


def _temporal_coords_from_dates(date_strs: list[str]):
    """Build a [T,2] (year, day-of-year) list from ISO date strings."""
    import pandas as pd

    coords = []
    for ds in date_strs:
        d = pd.to_datetime(ds)
        coords.append([float(d.year), float(d.dayofyear)])
    return coords


def _prithvi_forward_tokens_multiframe(
    model,
    x_tchw: np.ndarray,
    *,
    lon: float,
    lat: float,
    date_strs: list[str],
    device: str,
) -> np.ndarray:
    """Run Prithvi over a [T,6,H,W] series; return token sequence [N,D]."""
    ensure_torch()
    import torch

    if x_tchw.ndim != 4 or x_tchw.shape[1] != 6:
        raise ModelError(f"Prithvi expects [T,6,H,W], got {getattr(x_tchw, 'shape', None)}")
    t = int(x_tchw.shape[0])
    if len(date_strs) != t:
        raise ModelError(f"date_strs length {len(date_strs)} != T={t}.")

    # [T,C,H,W] -> [1,C,T,H,W]
    x = torch.from_numpy(x_tchw).permute(1, 0, 2, 3).unsqueeze(0).to(device)
    temporal_coords = torch.tensor(
        [_temporal_coords_from_dates(date_strs)], dtype=torch.float32, device=device
    )  # [1,T,2]
    location_coords = torch.tensor(
        [[float(lat), float(lon)]], dtype=torch.float32, device=device
    )  # [1,2]

    with torch.no_grad():
        if hasattr(model, "forward_features"):
            out = model.forward_features(
                x, temporal_coords=temporal_coords, location_coords=location_coords
            )
        else:
            out = model(x, temporal_coords=temporal_coords, location_coords=location_coords)

    tokens = _extract_tokens(out)
    if tokens is None or not hasattr(tokens, "ndim") or tokens.ndim != 3:
        raise ModelError(
            f"Unexpected Prithvi tokens shape/type: {type(tokens)} {getattr(tokens, 'shape', None)}"
        )
    return tokens[0].detach().float().cpu().numpy().astype(np.float32)


def _prithvi_forward_tokens_batch_multiframe(
    model,
    x_btchw: np.ndarray,
    *,
    lon_lat_batch: list[tuple[float, float]],
    date_strs_batch: list[list[str]],
    device: str,
) -> list[np.ndarray]:
    """Batch Prithvi forward for [B,T,6,H,W] series (uniform T within a call)."""
    ensure_torch()
    import torch

    if x_btchw.ndim != 5 or x_btchw.shape[2] != 6:
        raise ModelError(f"Prithvi expects [B,T,6,H,W], got {getattr(x_btchw, 'shape', None)}")
    bsz = int(x_btchw.shape[0])
    t = int(x_btchw.shape[1])
    if len(lon_lat_batch) != bsz or len(date_strs_batch) != bsz:
        raise ModelError("lon_lat_batch/date_strs_batch size mismatch with input batch.")
    if any(len(ds) != t for ds in date_strs_batch):
        raise ModelError(f"every date_strs entry must have length T={t}.")

    # [B,T,C,H,W] -> [B,C,T,H,W]
    xb = torch.from_numpy(x_btchw).permute(0, 2, 1, 3, 4).to(device)
    temporal_coords = torch.tensor(
        [_temporal_coords_from_dates(ds) for ds in date_strs_batch],
        dtype=torch.float32,
        device=device,
    )  # [B,T,2]
    location_coords = torch.tensor(
        [[float(lat), float(lon)] for (lon, lat) in lon_lat_batch],
        dtype=torch.float32,
        device=device,
    )  # [B,2]

    with torch.inference_mode():
        if hasattr(model, "forward_features"):
            out = model.forward_features(
                xb, temporal_coords=temporal_coords, location_coords=location_coords
            )
        else:
            out = model(xb, temporal_coords=temporal_coords, location_coords=location_coords)

    tokens = _extract_tokens(out)
    if tokens is None or not hasattr(tokens, "ndim") or int(tokens.ndim) != 3:
        raise ModelError(
            f"Unexpected Prithvi batch tokens shape/type: {type(tokens)} {getattr(tokens, 'shape', None)}"
        )
    if int(tokens.shape[0]) != bsz:
        raise ModelError(f"Prithvi batch mismatch: got B={int(tokens.shape[0])}, expected {bsz}")
    toks_np = tokens.detach().float().cpu().numpy().astype(np.float32)
    return [toks_np[i] for i in range(bsz)]


# -------------------------
# Embedder
# -------------------------
@register("prithvi")
class PrithviEOV2S2_6B_Embedder(EmbedderBase):
    """
    Prithvi-EO v2 (vendored HF runtime) on-the-fly embeddings from Sentinel-2 6-band patch.

    Inputs:
      - spatial: BBox/PointBuffer (EPSG:4326)
      - temporal: range/year (year->full year)
      - sensor: controls provider fetch (scale/cloudy/composite)

    Outputs:
      - pooled: patch-token mean/max (exclude CLS if present)
      - grid: token map [D,H,W] (exclude CLS if present)
    """

    # Prithvi's grid reshape needs a square token grid → enlarge a rectangular
    # ROI to a square of real imagery (base.fetch_input for the single-frame
    # path; the multi-frame fetch_input below for multi) and crop back to the ROI.
    _requires_square_input = True
    DEFAULT_MODEL_KEY = "prithvi_eo_v2_100_tl"
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_IMAGE_SCALE_M = 30  # notebook used 30m
    DEFAULT_CLOUDY_PCT = 30
    DEFAULT_COMPOSITE = "median"
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 4
    DEFAULT_BATCH_CUDA = 16

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(PRITHVI_S2_BANDS_DST),
        scale_m=30,
        cloudy_pct=30,
        expected_channels=6,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["provider"],
            "model_key_default": self.DEFAULT_MODEL_KEY,
            "input_bands": PRITHVI_S2_BANDS_DST,
            "output": ["pooled", "grid"],
            "defaults": {
                "model_key": self.DEFAULT_MODEL_KEY,
                "variant": self.DEFAULT_MODEL_KEY,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "scale_m": self.input_spec.scale_m,
                "cloudy_pct": self.input_spec.cloudy_pct,
                "composite": self.input_spec.composite,
                "fill_value": self.input_spec.fill_value,
            },
            "model_config": {
                "variant": {
                    "type": "string",
                    "default": self.DEFAULT_MODEL_KEY,
                    "choices": [
                        "prithvi_eo_v2_100_tl",
                        "prithvi_eo_v2_300_tl",
                        "prithvi_eo_v2_600_tl",
                    ],
                },
                "temporal_mode": {
                    "type": "string",
                    "default": _DEFAULT_TEMPORAL_MODE,
                    "choices": ["auto", "single", "multi"],
                    "description": (
                        "auto (default): single when the window yields one frame, "
                        "else multi (≥2 frames). single: one median composite over the "
                        "whole window (T=1). multi: a true time series — T derived from "
                        "the window length (~28-day min spacing, capped at max_frames), "
                        "matching Prithvi's multi-temporal pretraining; provider-duplicated "
                        "frames are dropped so diversity-free windows degrade to T=1."
                    ),
                },
                "max_frames": {
                    "type": "int",
                    "default": _DEFAULT_MAX_FRAMES,
                    "description": (
                        "Max frames in multi mode (default 4, matching Prithvi-EO-2.0 "
                        "pretraining). Actual T = clamp(window_days // 30, 1, max_frames)."
                    ),
                },
            },
            "notes": [
                "Uses vendored PrithviMAE runtime with weights downloaded from Hugging Face.",
                "Requires temporal_coords (year, dayofyear) and location_coords (lat, lon).",
                "temporal_mode='multi' feeds a [B,6,T,H,W] series with per-frame dates; "
                "output dimensionality is identical to single mode (tokens pooled over time).",
            ],
        }

    def _default_sensor(self) -> SensorSpec:
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
        """Prefetch the raw S2 6-band input for API-side tiling / export.

        Overrides the generic base prefetch (which fetches a single composite)
        so the ``tile``/``auto``/``export_batch`` paths receive the **same
        window-adaptive multi-frame series** as the direct ``get_embedding``
        path — instead of silently degrading to ``T=1``. Returns raw DN
        ``[T,6,H,W]`` (0..10000) when the window resolves to multi-frame, or
        ``None`` for a single-frame window so the caller uses the generic
        single-composite fetch. Single/multi and the frame count follow the
        same ``temporal_mode`` / window logic as the direct path (``model_config``
        is unavailable at prefetch time, so an explicit ``temporal_mode`` arg,
        the env var, or the ``auto`` default decides — matching ``olmoearth`` /
        ``galileo``).
        """
        if sensor is None:
            sensor = self._default_sensor()
        t = temporal_to_range(temporal)
        cfg = {"temporal_mode": temporal_mode} if temporal_mode is not None else None
        if _effective_temporal_mode(cfg, t) != "multi":
            return None  # single-frame window: defer to the generic composite prefetch
        # Skip the whole-ROI square fetch when the API tiles the input (it squares
        # per tile); fetch the rectangular ROI directly instead.
        geo_roi = FULL_WINDOW
        if square_input:
            spatial, geo_roi = square_spatial(spatial)  # enlarge rectangle to square
        n_frames = _auto_num_frames(
            t,
            max_frames=_resolve_max_frames(cfg),
            stride_days=_resolve_frame_stride_days(),
        )
        raw_tchw = _fetch_s2_prithvi6_tchw(
            provider,
            spatial=spatial,
            temporal=t,
            n_frames=n_frames,
            scale_m=int(sensor.scale_m),
            cloudy_pct=int(sensor.cloudy_pct),
            composite=str(sensor.composite),
            fill_value=float(sensor.fill_value),
        )
        meta: dict[str, Any] = {
            "temporal_mode": "multi",
            "n_frames": int(n_frames),
            **(roi_fetch_meta(geo_roi) or {}),
        }
        return FetchResult(data=raw_tchw, meta=meta)

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_PRITHVI_FETCH_WORKERS",
                str(PrithviEOV2S2_6B_Embedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default_bs = (
            PrithviEOV2S2_6B_Embedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else PrithviEOV2S2_6B_Embedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_PRITHVI_BATCH_SIZE", str(default_bs)))
        return max(1, v)

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
            raise ModelError("prithvi_eo_v2_s2_6b expects a provider backend (or 'auto').")

        # Defaults for Prithvi inputs
        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)  # normalize to range

        if _effective_temporal_mode(model_config, t) == "multi":
            return self._get_embedding_multiframe(
                spatial=spatial,
                temporal=t,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
                input_chw=input_chw,
                model_config=model_config,
                fetch_meta=fetch_meta,
            )

        # Fetch-square ROI window: from the direct fetch, or carried in fetch_meta
        # when the API prefetched a square. The output is cropped back to it.
        geo_roi = geo_roi_from_meta(fetch_meta)

        # Load model
        model_key, variant = _resolve_prithvi_model_key(
            model_config=model_config,
            default_model_key=self.DEFAULT_MODEL_KEY,
        )
        pretrained = os.environ.get("RS_EMBED_PRITHVI_PRETRAINED", "1").strip() not in (
            "0",
            "false",
            "False",
        )
        coords_encoding = ("time", "location")
        num_frames = 1

        model, wmeta, dev = _load_prithvi(
            model_key,
            pretrained=pretrained,
            bands=tuple(PRITHVI_S2_BANDS_DST),
            num_frames=num_frames,
            coords_encoding=coords_encoding,
            device=device,
        )

        # Fetch S2 6-band patch from provider (optionally reuse pre-fetched raw
        # patch — a prefetched input must not require live provider auth).
        if input_chw is None:
            provider = self._get_provider(backend)
            spatial, geo_roi = square_spatial(spatial)  # enlarge rectangle to square
            x_chw = _fetch_s2_prithvi6_chw(
                provider,
                spatial=spatial,
                temporal=t,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
            )
        else:
            # input_chw expected to be raw S2 SR values (0..10000) in band order sensor.bands
            if input_chw.ndim != 3 or input_chw.shape[0] != 6:
                raise ModelError(
                    f"input_chw must be CHW with 6 bands for prithvi, got {getattr(input_chw, 'shape', None)}"
                )
            x_chw = input_chw.astype(np.float32) / 10000.0
            x_chw = np.clip(x_chw, 0.0, 1.0)
            x_chw = np.nan_to_num(x_chw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Optional: inspect on-the-fly provider input
        from ..tools.inspection import (
            checks_save_dir,
            checks_should_raise,
            maybe_inspect_chw,
            save_quicklook_rgb,
        )

        check_meta: dict[str, Any] = {}
        report = maybe_inspect_chw(
            x_chw,
            sensor=sensor,
            name="provider_s2_prithvi6_chw",
            expected_channels=6,
            value_range=(0.0, 1.0),
            fill_value=float(sensor.fill_value),
            meta=check_meta,
        )
        if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
            raise ModelError(
                "Provider input inspection failed: " + "; ".join(report.get("issues", []))
            )

        # Optional quicklook (RGB from RED/GREEN/BLUE)
        sd = checks_save_dir(sensor)
        if sd and report is not None:
            try:
                import uuid

                fn = f"prithvi_s2_rgb_{uuid.uuid4().hex[:8]}.png"
                save_quicklook_rgb(
                    x_chw,
                    path=os.path.join(sd, fn),
                    bands=(2, 1, 0),
                    vmin=0.0,
                    vmax=1.0,
                )
                check_meta.setdefault("input_checks_artifacts", []).append(
                    {"name": "quicklook_rgb", "path": os.path.join(sd, fn)}
                )
            except Exception as _e:
                check_meta.setdefault("input_checks_artifacts", []).append(
                    {"name": "quicklook_rgb", "error": repr(_e)}
                )
        x_chw, prep_meta = _prepare_prithvi_chw(
            x_chw,
            fill_value=float(sensor.fill_value),
        )

        # coords: use temporal mid-date and ROI center (EPSG:4326).
        lon, lat = _spatial_center_lon_lat(spatial)

        date_str = temporal_midpoint_str(t)

        tokens = _prithvi_forward_tokens(
            model,
            x_chw,
            lon=lon,
            lat=lat,
            date_str=date_str,
            device=dev,
        )  # [N,D] (maybe includes CLS)

        meta = base_meta(
            model_name=self.model_name,
            hf_id=str(wmeta.get("repo_id") or model_key),
            backend=str(backend).lower(),
            image_size=int(x_chw.shape[-1]),  # not fixed 224; depends on ROI/scale
            sensor=sensor,
            temporal=t,
            source=sensor.collection,
            extra={
                "temporal_range": (t.start, t.end),
                "coords_date": date_str,
                "coords_lonlat": (float(lon), float(lat)),
                "tokens_shape": tuple(tokens.shape),
                "model_key": model_key,
                "variant": variant,
                "pretrained": bool(pretrained),
                "coords_encoding": coords_encoding,
                "num_frames": num_frames,
                "input_hw": (int(x_chw.shape[1]), int(x_chw.shape[2])),
                **prep_meta,
                **check_meta,
            },
        )

        cropped_to_roi = not roi_is_full(geo_roi)

        if output.mode == "pooled":
            if cropped_to_roi:
                # Pool only the ROI's tokens (exclude the real-neighborhood context
                # fetched to make the input square).
                grid, _hw, cls_removed = tokens_to_grid_dhw(tokens)
                grid = crop_grid_to_roi(grid, geo_roi)
                vec = (
                    grid.max(axis=(1, 2)) if output.pooling == "max" else grid.mean(axis=(1, 2))
                ).astype("float32")
                pooling = f"roi_grid_{output.pooling}"
            else:
                vec, cls_removed = pool_from_tokens(tokens, output.pooling)
                pooling = f"patch_{output.pooling}"
            meta.update({"pooling": pooling, "cls_removed": bool(cls_removed)})
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            grid, (h, w), cls_removed = tokens_to_grid_dhw(tokens)
            if cropped_to_roi:
                grid = crop_grid_to_roi(grid, geo_roi)
                h, w = int(grid.shape[1]), int(grid.shape[2])
            meta.update(
                {
                    "grid_hw": (h, w),
                    "grid_kind": "patch_tokens",
                    "cls_removed": bool(cls_removed),
                }
            )

            try:
                import xarray as xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

            da = xr.DataArray(
                grid,
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(grid.shape[0]),
                    "y": np.arange(h),
                    "x": np.arange(w),
                },
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        raise ModelError(f"Unknown output mode: {output.mode}")

    def _get_embedding_multiframe(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec,
        sensor: SensorSpec,
        output: OutputSpec,
        backend: str,
        device: str,
        input_chw: np.ndarray | None,
        model_config: dict[str, Any] | None,
        fetch_meta: dict[str, Any] | None = None,
    ) -> Embedding:
        """Multi-frame Prithvi path: T derived from the window (≤ max_frames),
        provider-duplicated frames collapsed, then a true [B,6,T,H,W] forward."""
        t = temporal
        # Fetch-square ROI window: from the direct fetch, or carried in fetch_meta
        # when the API prefetched a square. Output is cropped back to it.
        geo_roi = geo_roi_from_meta(fetch_meta)
        model_key, variant = _resolve_prithvi_model_key(
            model_config=model_config,
            default_model_key=self.DEFAULT_MODEL_KEY,
        )
        pretrained = os.environ.get("RS_EMBED_PRITHVI_PRETRAINED", "1").strip() not in (
            "0",
            "false",
            "False",
        )
        coords_encoding = ("time", "location")
        max_frames = _resolve_max_frames(model_config)
        stride_days = _resolve_frame_stride_days()
        max_stride_days = _resolve_max_frame_stride_days()
        requested = _auto_num_frames(t, max_frames=max_frames, stride_days=stride_days)

        # Obtain a raw [T,6,H,W] series (provider fetch, or shape-driven override).
        if input_chw is None:
            provider = self._get_provider(backend)
            spatial, geo_roi = square_spatial(spatial)  # enlarge rectangle to square
            raw_tchw = _fetch_s2_prithvi6_tchw(
                provider,
                spatial=spatial,
                temporal=t,
                n_frames=requested,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
            )
            dates = list(_temporal_frame_midpoints(t, requested))
        else:
            arr = np.asarray(input_chw, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 6:
                raw_tchw = arr[np.newaxis]  # treat as a single frame, never duplicated
                dates = [temporal_midpoint_str(t) or "2022-07-01"]
            elif arr.ndim == 4 and arr.shape[1] == 6:
                raw_tchw = arr
                dates = list(_temporal_frame_midpoints(t, int(arr.shape[0])))
            else:
                raise ModelError(
                    f"input_chw must be CHW or TCHW with 6 bands for prithvi, "
                    f"got {getattr(input_chw, 'shape', None)}"
                )

        # Frame diversity of the fetched/provided series before duplicate frames are
        # collapsed below — captured on both paths (the tiled path feeds back-filled
        # tiles via input_chw), so it lands in meta everywhere.
        diversity_meta = frame_diversity_meta(
            n_requested=int(raw_tchw.shape[0]),
            n_distinct=count_distinct_frames(raw_tchw),
        )

        x_tchw = _raw_tchw_to_unit(raw_tchw)
        x_tchw, dates = _drop_duplicate_frames(x_tchw, dates)
        x_tchw, prep_meta = _prepare_prithvi_tchw(x_tchw, fill_value=float(sensor.fill_value))
        n_frames = int(x_tchw.shape[0])

        model, wmeta, dev = _load_prithvi(
            model_key,
            pretrained=pretrained,
            bands=tuple(PRITHVI_S2_BANDS_DST),
            num_frames=n_frames,
            coords_encoding=coords_encoding,
            device=device,
        )

        lon, lat = _spatial_center_lon_lat(spatial)
        tokens = _prithvi_forward_tokens_multiframe(
            model, x_tchw, lon=lon, lat=lat, date_strs=dates, device=dev
        )
        spacing_meta = _temporal_spacing_meta(dates, max_stride_days=max_stride_days)

        meta = base_meta(
            model_name=self.model_name,
            hf_id=str(wmeta.get("repo_id") or model_key),
            backend=str(backend).lower(),
            image_size=int(x_tchw.shape[-1]),
            sensor=sensor,
            temporal=t,
            source=sensor.collection,
            extra={
                "temporal_range": (t.start, t.end),
                "temporal_mode": "multi",
                "num_frames": n_frames,
                "requested_frames": int(requested),
                **diversity_meta,
                "frame_dates": tuple(dates),
                "frame_stride_days_min": int(stride_days),
                "frame_stride_days_max": int(max_stride_days),
                "coords_lonlat": (float(lon), float(lat)),
                "tokens_shape": tuple(tokens.shape),
                "model_key": model_key,
                "variant": variant,
                "pretrained": bool(pretrained),
                "coords_encoding": coords_encoding,
                "input_hw": (int(x_tchw.shape[-2]), int(x_tchw.shape[-1])),
                **spacing_meta,
                **prep_meta,
            },
        )

        cropped_to_roi = not roi_is_full(geo_roi)

        if output.mode == "pooled":
            if cropped_to_roi:
                # Pool only the ROI's tokens (time-collapsed grid, then ROI crop).
                grid, _hw, cls_removed = tokens_to_grid_dhw_tchw(tokens, n_frames, output.pooling)
                grid = crop_grid_to_roi(grid, geo_roi)
                vec = (
                    grid.max(axis=(1, 2)) if output.pooling == "max" else grid.mean(axis=(1, 2))
                ).astype("float32")
                pooling = f"roi_grid_temporal_{output.pooling}"
            else:
                vec, cls_removed = pool_from_tokens_tchw(tokens, n_frames, output.pooling)
                pooling = f"patch_temporal_{output.pooling}"
            meta.update(
                {
                    "pooling": pooling,
                    "temporal_pooling": output.pooling,
                    "cls_removed": bool(cls_removed),
                }
            )
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            grid, (h, w), cls_removed = tokens_to_grid_dhw_tchw(tokens, n_frames, output.pooling)
            if cropped_to_roi:
                grid = crop_grid_to_roi(grid, geo_roi)
                h, w = int(grid.shape[1]), int(grid.shape[2])
            meta.update(
                {
                    "grid_hw": (h, w),
                    "grid_kind": "patch_tokens",
                    "temporal_pooling": output.pooling,
                    "cls_removed": bool(cls_removed),
                }
            )
            try:
                import xarray as xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

            da = xr.DataArray(
                grid,
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(grid.shape[0]),
                    "y": np.arange(h),
                    "x": np.arange(w),
                },
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("prithvi_eo_v2_s2_6b expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)
        multi = _effective_temporal_mode(model_config, t) == "multi"
        n_frames_req = (
            _auto_num_frames(
                t,
                max_frames=_resolve_max_frames(model_config),
                stride_days=_resolve_frame_stride_days(),
            )
            if multi
            else 1
        )
        provider = self._get_provider(backend)

        def _fetch_raw(sq: SpatialSpec) -> np.ndarray:
            if multi:
                # Raw [T,6,H,W] in 0..10000; from_inputs detects the 4D shape.
                raw = _fetch_s2_prithvi6_tchw(
                    provider,
                    spatial=sq,
                    temporal=t,
                    n_frames=n_frames_req,
                    scale_m=int(sensor.scale_m),
                    cloudy_pct=int(sensor.cloudy_pct),
                    composite=str(sensor.composite),
                    fill_value=float(sensor.fill_value),
                )
                return np.clip(raw, 0.0, 10000.0).astype(np.float32)
            x_chw = _fetch_s2_prithvi6_chw(
                provider,
                spatial=sq,
                temporal=t,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
            )
            # get_embedding(input_chw=...) expects raw SR in [0..10000]
            return np.clip(x_chw * 10000.0, 0.0, 10000.0).astype(np.float32)

        # Square-fetch each ROI; the per-item ROI window rides in geo_rois and is
        # forwarded as _roi_windows_geo so each item's output is cropped back.
        raw_inputs, geo_rois = square_fetch_batch(
            spatials, _fetch_raw, max_workers=self._resolve_fetch_workers(len(spatials))
        )
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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("prithvi_eo_v2_s2_6b expects a provider backend (or 'auto').")
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []
        # Prefetched square inputs carry the ROI window in fetch_meta (the
        # export pipeline passes it via ``fetch_metas``); fold it into the
        # internal per-item ROI list so the output is cropped back to the ROI.
        if _roi_windows_geo is None and fetch_metas is not None:
            _roi_windows_geo = [(m or {}).get("roi_window_geo") for m in fetch_metas]

        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)

        def _roi_at(i: int) -> tuple[float, float, float, float]:
            return tuple((_roi_windows_geo[i] if _roi_windows_geo else None) or FULL_WINDOW)  # type: ignore[return-value]

        if _effective_temporal_mode(model_config, t) == "multi" or any(
            np.asarray(x).ndim == 4 for x in input_chws
        ):
            return self._batch_from_inputs_multiframe(
                spatials=spatials,
                input_chws=input_chws,
                temporal=t,
                sensor=sensor,
                model_config=model_config,
                output=output,
                backend=backend,
                device=device,
                _roi_windows_geo=_roi_windows_geo,
            )

        model_key, variant = _resolve_prithvi_model_key(
            model_config=model_config,
            default_model_key=self.DEFAULT_MODEL_KEY,
        )
        pretrained = os.environ.get("RS_EMBED_PRITHVI_PRETRAINED", "1").strip() not in (
            "0",
            "false",
            "False",
        )
        coords_encoding = ("time", "location")
        num_frames = 1
        prep_mode = os.environ.get("RS_EMBED_PRITHVI_PREP", "resize").strip().lower()
        patch_mult = max(1, int(os.environ.get("RS_EMBED_PRITHVI_PATCH_MULT", "16")))
        target_size = max(16, int(os.environ.get("RS_EMBED_PRITHVI_IMG", "224")))

        model, wmeta, dev = _load_prithvi(
            model_key,
            pretrained=pretrained,
            bands=tuple(PRITHVI_S2_BANDS_DST),
            num_frames=num_frames,
            coords_encoding=coords_encoding,
            device=device,
        )
        infer_bs = self._resolve_infer_batch(str(dev))

        x_prepared: list[np.ndarray] = []
        lon_lat: list[tuple[float, float]] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or input_chw.shape[0] != 6:
                raise ModelError(
                    f"input_chw must be CHW with 6 bands for prithvi, got {getattr(input_chw, 'shape', None)} at index={i}"
                )
            x_chw = input_chw.astype(np.float32) / 10000.0
            x_chw = np.clip(x_chw, 0.0, 1.0)
            x_chw = np.nan_to_num(x_chw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            x_chw, _ = _prepare_prithvi_chw(
                x_chw,
                fill_value=float(sensor.fill_value),
            )
            x_prepared.append(x_chw)
            lon_lat.append(_spatial_center_lon_lat(spatials[i]))

        date_str = temporal_midpoint_str(t) or "2022-07-01"
        shape_groups: dict[tuple[int, int, int], list[int]] = {}
        for i, x in enumerate(x_prepared):
            shape_groups.setdefault(tuple(x.shape), []).append(i)

        out: list[Embedding | None] = [None] * len(spatials)
        xr_mod = None
        if output.mode == "grid":
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

        for idxs in shape_groups.values():
            for s0 in range(0, len(idxs), infer_bs):
                chunk_ids = idxs[s0 : s0 + infer_bs]
                xb = np.stack([x_prepared[i] for i in chunk_ids], axis=0).astype(np.float32)
                toks_list = _prithvi_forward_tokens_batch(
                    model,
                    xb,
                    lon_lat_batch=[lon_lat[i] for i in chunk_ids],
                    date_str_batch=[date_str for _ in chunk_ids],
                    device=dev,
                )
                if len(toks_list) != len(chunk_ids):
                    raise ModelError(
                        f"Prithvi batch output mismatch: {len(toks_list)} != {len(chunk_ids)}"
                    )

                for j, i in enumerate(chunk_ids):
                    tokens = toks_list[j]
                    lon, lat = lon_lat[i]
                    x_chw = x_prepared[i]
                    meta = base_meta(
                        model_name=self.model_name,
                        hf_id=str(wmeta.get("repo_id") or model_key),
                        backend=str(backend).lower(),
                        image_size=int(x_chw.shape[-1]),
                        sensor=sensor,
                        temporal=t,
                        source=sensor.collection,
                        extra={
                            "temporal_range": (t.start, t.end),
                            "coords_date": date_str,
                            "coords_lonlat": (float(lon), float(lat)),
                            "tokens_shape": tuple(tokens.shape),
                            "model_key": model_key,
                            "variant": variant,
                            "pretrained": bool(pretrained),
                            "coords_encoding": coords_encoding,
                            "num_frames": num_frames,
                            "input_hw": (int(x_chw.shape[1]), int(x_chw.shape[2])),
                            "prep_mode": str(prep_mode),
                            "patch_mult": patch_mult,
                            "target_image_size": target_size,
                            "batch_infer": True,
                            "input_override": True,
                            **wmeta,
                        },
                    )

                    geo_roi_i = _roi_at(i)
                    cropped_i = not roi_is_full(geo_roi_i)

                    if output.mode == "pooled":
                        if cropped_i:
                            grid, _hw, cls_removed = tokens_to_grid_dhw(tokens)
                            grid = crop_grid_to_roi(grid, geo_roi_i)
                            vec = (
                                grid.max(axis=(1, 2))
                                if output.pooling == "max"
                                else grid.mean(axis=(1, 2))
                            ).astype("float32")
                            pooling = f"roi_grid_{output.pooling}"
                        else:
                            vec, cls_removed = pool_from_tokens(tokens, output.pooling)
                            pooling = f"patch_{output.pooling}"
                        meta.update({"pooling": pooling, "cls_removed": bool(cls_removed)})
                        out[i] = Embedding(data=vec, meta=meta)
                        continue

                    if output.mode == "grid":
                        grid, (h, w), cls_removed = tokens_to_grid_dhw(tokens)
                        if cropped_i:
                            grid = crop_grid_to_roi(grid, geo_roi_i)
                            h, w = int(grid.shape[1]), int(grid.shape[2])
                        meta.update(
                            {
                                "grid_hw": (h, w),
                                "grid_kind": "patch_tokens",
                                "cls_removed": bool(cls_removed),
                            }
                        )
                        assert xr_mod is not None
                        da = xr_mod.DataArray(
                            grid,
                            dims=("d", "y", "x"),
                            coords={
                                "d": np.arange(grid.shape[0]),
                                "y": np.arange(h),
                                "x": np.arange(w),
                            },
                            name="embedding",
                            attrs=meta,
                        )
                        out[i] = Embedding(data=da, meta=meta)
                        continue

                    raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError("prithvi_eo_v2_s2_6b batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

    def _batch_from_inputs_multiframe(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: TemporalSpec,
        sensor: SensorSpec,
        model_config: dict[str, Any] | None,
        output: OutputSpec,
        backend: str,
        device: str,
        _roi_windows_geo: list[tuple[float, float, float, float] | None] | None = None,
    ) -> list[Embedding]:
        """Batch multi-frame path. Per item: 3D→single frame, 4D→binned series;
        identical frames collapsed, then grouped by prepared (T,H,W) shape so a
        model with matching ``num_frames`` runs each group as [B,6,T,H,W]."""
        t = temporal

        def _roi_at(i: int) -> tuple[float, float, float, float]:
            return tuple((_roi_windows_geo[i] if _roi_windows_geo else None) or FULL_WINDOW)  # type: ignore[return-value]

        model_key, variant = _resolve_prithvi_model_key(
            model_config=model_config,
            default_model_key=self.DEFAULT_MODEL_KEY,
        )
        pretrained = os.environ.get("RS_EMBED_PRITHVI_PRETRAINED", "1").strip() not in (
            "0",
            "false",
            "False",
        )
        coords_encoding = ("time", "location")
        stride_days = _resolve_frame_stride_days()
        max_stride_days = _resolve_max_frame_stride_days()

        # Prepare each item to (T_eff, 6, H, W) + per-frame dates.
        prepared: list[np.ndarray] = []
        item_dates: list[list[str]] = []
        lon_lat: list[tuple[float, float]] = []
        for i, raw in enumerate(input_chws):
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == 6:
                raw_tchw = arr[np.newaxis]
                dates = [temporal_midpoint_str(t) or "2022-07-01"]
            elif arr.ndim == 4 and arr.shape[1] == 6:
                raw_tchw = arr
                dates = list(_temporal_frame_midpoints(t, int(arr.shape[0])))
            else:
                raise ModelError(
                    f"input_chw must be CHW or TCHW with 6 bands for prithvi, "
                    f"got {getattr(arr, 'shape', None)} at index={i}"
                )
            x_tchw = _raw_tchw_to_unit(raw_tchw)
            x_tchw, dates = _drop_duplicate_frames(x_tchw, dates)
            x_tchw, _ = _prepare_prithvi_tchw(x_tchw, fill_value=float(sensor.fill_value))
            prepared.append(x_tchw)
            item_dates.append(dates)
            lon_lat.append(_spatial_center_lon_lat(spatials[i]))

        # Group by full prepared shape (T,H,W) so each batch shares num_frames.
        shape_groups: dict[tuple[int, ...], list[int]] = {}
        for i, x in enumerate(prepared):
            shape_groups.setdefault(tuple(x.shape), []).append(i)

        xr_mod = None
        if output.mode == "grid":
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

        out: list[Embedding | None] = [None] * len(spatials)
        for idxs in shape_groups.values():
            n_frames = int(prepared[idxs[0]].shape[0])
            model, wmeta, dev = _load_prithvi(
                model_key,
                pretrained=pretrained,
                bands=tuple(PRITHVI_S2_BANDS_DST),
                num_frames=n_frames,
                coords_encoding=coords_encoding,
                device=device,
            )
            infer_bs = self._resolve_infer_batch(str(dev))
            for s0 in range(0, len(idxs), infer_bs):
                chunk_ids = idxs[s0 : s0 + infer_bs]
                xb = np.stack([prepared[i] for i in chunk_ids], axis=0).astype(np.float32)
                toks_list = _prithvi_forward_tokens_batch_multiframe(
                    model,
                    xb,
                    lon_lat_batch=[lon_lat[i] for i in chunk_ids],
                    date_strs_batch=[item_dates[i] for i in chunk_ids],
                    device=dev,
                )
                if len(toks_list) != len(chunk_ids):
                    raise ModelError(
                        f"Prithvi batch output mismatch: {len(toks_list)} != {len(chunk_ids)}"
                    )
                for j, i in enumerate(chunk_ids):
                    tokens = toks_list[j]
                    lon, lat = lon_lat[i]
                    x_item = prepared[i]
                    spacing_meta = _temporal_spacing_meta(
                        item_dates[i],
                        max_stride_days=max_stride_days,
                        label=f"spatial index {i}",
                    )
                    meta = base_meta(
                        model_name=self.model_name,
                        hf_id=str(wmeta.get("repo_id") or model_key),
                        backend=str(backend).lower(),
                        image_size=int(x_item.shape[-1]),
                        sensor=sensor,
                        temporal=t,
                        source=sensor.collection,
                        extra={
                            "temporal_range": (t.start, t.end),
                            "temporal_mode": "multi",
                            "num_frames": n_frames,
                            "frame_dates": tuple(item_dates[i]),
                            "frame_stride_days_min": int(stride_days),
                            "frame_stride_days_max": int(max_stride_days),
                            "coords_lonlat": (float(lon), float(lat)),
                            "tokens_shape": tuple(tokens.shape),
                            "model_key": model_key,
                            "variant": variant,
                            "pretrained": bool(pretrained),
                            "coords_encoding": coords_encoding,
                            "input_hw": (int(x_item.shape[-2]), int(x_item.shape[-1])),
                            "batch_infer": True,
                            **spacing_meta,
                        },
                    )

                    geo_roi_i = _roi_at(i)
                    cropped_i = not roi_is_full(geo_roi_i)

                    if output.mode == "pooled":
                        if cropped_i:
                            grid, _hw, cls_removed = tokens_to_grid_dhw_tchw(
                                tokens, n_frames, output.pooling
                            )
                            grid = crop_grid_to_roi(grid, geo_roi_i)
                            vec = (
                                grid.max(axis=(1, 2))
                                if output.pooling == "max"
                                else grid.mean(axis=(1, 2))
                            ).astype("float32")
                            pooling = f"roi_grid_temporal_{output.pooling}"
                        else:
                            vec, cls_removed = pool_from_tokens_tchw(
                                tokens, n_frames, output.pooling
                            )
                            pooling = f"patch_temporal_{output.pooling}"
                        meta.update(
                            {
                                "pooling": pooling,
                                "temporal_pooling": output.pooling,
                                "cls_removed": bool(cls_removed),
                            }
                        )
                        out[i] = Embedding(data=vec, meta=meta)
                        continue

                    if output.mode == "grid":
                        grid, (h, w), cls_removed = tokens_to_grid_dhw_tchw(
                            tokens, n_frames, output.pooling
                        )
                        if cropped_i:
                            grid = crop_grid_to_roi(grid, geo_roi_i)
                            h, w = int(grid.shape[1]), int(grid.shape[2])
                        meta.update(
                            {
                                "grid_hw": (h, w),
                                "grid_kind": "patch_tokens",
                                "temporal_pooling": output.pooling,
                                "cls_removed": bool(cls_removed),
                            }
                        )
                        assert xr_mod is not None
                        da = xr_mod.DataArray(
                            grid,
                            dims=("d", "y", "x"),
                            coords={
                                "d": np.arange(grid.shape[0]),
                                "y": np.arange(h),
                                "x": np.arange(w),
                            },
                            name="embedding",
                            attrs=meta,
                        )
                        out[i] = Embedding(data=da, meta=meta)
                        continue

                    raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError("prithvi_eo_v2_s2_6b batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

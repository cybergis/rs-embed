from __future__ import annotations

import importlib
import os
import warnings
from datetime import date
from functools import lru_cache
from pathlib import Path
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
from ..core.types import EmbedderCapabilities, FetchResult
from ..providers import ProviderBase
from ..providers.fetch import (
    count_distinct_frames,
    frame_diversity_meta,
)
from ..providers.fetch import (
    fetch_multiframe_patch_raw_tchw as _fetch_multiframe_patch_raw_tchw,
)
from ..providers.resolution import (
    is_provider_backend,
)
from ..tools.normalization import (
    coerce_input_to_tchw as _coerce_input_to_tchw,
)
from ..tools.runtime import (
    load_cached_with_device as _load_cached_with_device,
)
from ..tools.runtime import (
    move_model_to_device as _move_model_to_device,
)
from ..tools.runtime import (
    resolve_device_auto_torch as _resolve_device,
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
from ..tools.temporal import temporal_frame_midpoints
from .base import EmbedderBase
from .config import model_config_value
from .meta import build_meta, temporal_to_range
from .shared import grid_to_dataarray, normalize_s2, verify_loaded_params


def ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise ModelError("This embedder requires torch installed.") from e


_S2_10_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
_GALILEO_PRETRAIN_STATS = {
    "13": {
        "mean": [
            -11.728724389184965,
            -18.85558188024017,
            1395.3408730676722,
            1338.4026921784578,
            1343.09883810357,
            1543.8607982512297,
            2186.2022069512263,
            2525.0932853316694,
            2410.3377187373408,
            2750.2854646886753,
            2234.911100061487,
            1474.5311266077113,
            0.2892116502999044,
        ],
        "std": [
            4.887145774840316,
            5.730270320384293,
            917.7041440370853,
            913.2988423581528,
            1092.678723527555,
            1047.2206083460424,
            1048.0101611156767,
            1143.6903026819996,
            1098.979177731649,
            1204.472755085893,
            1145.9774063078878,
            980.2429840007796,
            0.2720939024500081,
        ],
    },
    "16": {
        "mean": [
            673.0152819503361,
            5.930092668915115,
            0.10470439140978786,
            0.23965913270066183,
            0.08158044385860364,
            0.04246976254259546,
            0.11304392863520317,
            0.17329647890362473,
            0.0698981691616277,
            0.12130267132802142,
            0.04671318615236216,
            10.973119802517362,
            1.0927069179958768,
            1.6991394232855903,
            0.03720594618055555,
            1.3671352688259548,
        ],
        "std": [
            983.0697298296237,
            8.167406789813247,
            0.18771647977504985,
            0.2368313455675914,
            0.08024268534756586,
            0.04045374496146404,
            0.11350342472061795,
            0.1279898111718168,
            0.12042341550438586,
            0.13602408145504347,
            0.043971116096060345,
            31.255340146970997,
            10.395974878206689,
            12.92380617159917,
            1.9285254295940466,
            11.612179775408928,
        ],
    },
    "6": {
        "mean": [
            271.5674963541667,
            0.08554303677156568,
            657.3181260091111,
            692.1291795806885,
            562.781331880633,
            1.5647115934036673,
        ],
        "std": [
            79.80828940314429,
            0.11669547098151486,
            704.0008695557707,
            925.0116126406431,
            453.2434022278578,
            7.513020170832818,
        ],
    },
    "18": {
        "mean": [
            188.20315880851746,
            0.2804946561574936,
            0.11371652073860168,
            0.058778801321983334,
            0.10474256777763366,
            0.2396918488264084,
            0.08152248692512512,
            0.04248040814399719,
            0.11303179881572724,
            0.17326324067115784,
            0.06998309404850006,
            0.12122812910079957,
            0.04671641788482666,
            10.98456594619751,
            1.0968475807189941,
            1.6947754135131836,
            0.03320046615600586,
            1.3602827312469483,
        ],
        "std": [
            1154.5919128300602,
            0.5276998078079327,
            0.7021637331734328,
            0.36528892213195063,
            0.17470213191865785,
            0.20411195416718833,
            0.0660782470089761,
            0.03380702424871257,
            0.09809195568521663,
            0.11292471052124119,
            0.09720748930233268,
            0.12912217763726777,
            0.0399973913151906,
            23.725471823867462,
            5.715238079725388,
            9.030481416228302,
            0.9950220242487364,
            7.754429123862099,
        ],
    },
}


def _is_galileo_official_stats_mode(mode: str) -> bool:
    return str(mode).lower().strip() in {
        "official_stats",
        "pretrain_stats",
        "pretraining_stats",
        "galileo_stats",
    }


def _resize_tchw(x_tchw: np.ndarray, *, out_hw: int) -> np.ndarray:
    """Make a [T,C,H,W] stack square ``out_hw`` without aspect-ratio distortion.

    A rectangular ROI is reflect-padded to square before resizing (see
    :mod:`rs_embed.tools.shape`) rather than stretched, which would smear a
    non-square ROI into distorted, striped embeddings.
    """
    if x_tchw.ndim != 4:
        raise ModelError(f"Expected [T,C,H,W], got {x_tchw.shape}")
    out, _ = prepare_square(x_tchw, size=int(out_hw), shape_adjust="pad")
    return out


def _normalize_s2(raw: np.ndarray, *, mode: str) -> np.ndarray:
    return normalize_s2(
        raw,
        mode=mode,
        model_name="Galileo",
        modes_hint="none, unit_scale, per_tile_minmax, official_stats",
        allow_tchw=True,
    )


def _apply_galileo_pretrain_stats(data: dict[str, Any]) -> None:
    ensure_torch()
    import torch

    def _norm_inplace(key: str) -> None:
        x = data[key]
        dim = int(x.shape[-1])
        stats = _GALILEO_PRETRAIN_STATS.get(str(dim))
        if stats is None:
            raise ModelError(f"Missing Galileo pretraining stats for last_dim={dim} on {key}.")
        mean = torch.tensor(stats["mean"], dtype=x.dtype, device=x.device)
        std = torch.tensor(stats["std"], dtype=x.dtype, device=x.device)
        std = torch.clamp(std, min=torch.finfo(x.dtype).eps)
        data[key] = (x - mean) / std

    for key in ("s_t_x", "sp_x", "t_x", "st_x"):
        _norm_inplace(key)


def _fetch_s2_10_raw_tchw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    n_frames: int = 8,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    raw = _fetch_multiframe_patch_raw_tchw(
        provider,
        spatial=spatial,
        temporal=temporal,
        bands=tuple(_S2_10_BANDS),
        n_frames=int(n_frames),
        collection="COPERNICUS/S2_SR_HARMONIZED",
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        fill_value=fill_value,
    )
    return np.clip(raw, 0.0, 10000.0).astype(np.float32)


def _resolve_model_folder(
    *,
    model_path: str | None,
    model_size: str,
    hf_repo: str,
    cache_dir: str | None,
    auto_download: bool,
) -> str:
    if model_path:
        p = os.path.expanduser(model_path)
    else:
        if not auto_download:
            raise ModelError(
                "Galileo model folder is required. Set RS_EMBED_GALILEO_MODEL_PATH "
                "or enable RS_EMBED_GALILEO_AUTO_DOWNLOAD=1."
            )
        p = _download_galileo_model_folder(
            model_size=model_size,
            hf_repo=hf_repo,
            cache_dir=cache_dir,
        )

    cfg = os.path.join(p, "config.json")
    enc = os.path.join(p, "encoder.pt")
    if not os.path.isfile(cfg) or not os.path.isfile(enc):
        raise ModelError(
            f"Galileo model folder is invalid: {p}. Expected config.json and encoder.pt."
        )
    return p


@lru_cache(maxsize=8)
def _load_galileo_module():
    try:
        mod = importlib.import_module("rs_embed.embedders._vendor.galileo_single_file")
        _ = mod.Encoder
    except Exception as e:
        raise ModelError(
            "Failed to import vendored Galileo runtime. "
            "Install missing dependencies: torch, einops."
        ) from e
    return mod


@lru_cache(maxsize=8)
def _download_galileo_model_folder(
    *,
    model_size: str,
    hf_repo: str,
    cache_dir: str | None,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ModelError(
            "Galileo auto-download requires huggingface_hub. Install: pip install huggingface_hub"
        ) from e

    snap = snapshot_download(
        repo_id=hf_repo,
        cache_dir=cache_dir,
        allow_patterns=[
            f"models/{model_size}/config.json",
            f"models/{model_size}/encoder.pt",
        ],
    )
    model_root = os.path.join(snap, "models", str(model_size))
    cfg = os.path.join(model_root, "config.json")
    enc = os.path.join(model_root, "encoder.pt")
    if not os.path.isfile(cfg) or not os.path.isfile(enc):
        raise ModelError(
            f"Downloaded Galileo snapshot for model_size={model_size!r} "
            f"but expected files were not found under {model_root}."
        )
    return model_root


def _month_from_iso(iso_date: str) -> int:
    d = date.fromisoformat(str(iso_date))
    # Vendored Galileo month embeddings are indexed from 0..11, not 1..12.
    return max(0, min(11, int(d.month) - 1))


def _frame_month_sequence(temporal: TemporalSpec, *, n_frames: int) -> np.ndarray:
    mids = temporal_frame_midpoints(temporal, max(1, int(n_frames)))
    if not mids:
        return np.full((max(1, int(n_frames)),), 5, dtype=np.int64)
    return np.array([_month_from_iso(v) for v in mids], dtype=np.int64)


def _month_override_sequence(month_value: int, *, n_frames: int) -> np.ndarray:
    month_value = max(1, min(12, int(month_value)))
    return np.full((max(1, int(n_frames)),), month_value - 1, dtype=np.int64)


# ---------------------------------------------------------------------------
# Temporal sampling (frame count is window-adaptive, mirroring OlmoEarth)
#
# Galileo pretrains on ~monthly composites and encodes month-of-year (0–11),
# capped at 12 frames per sample. Instead of always splitting the window into a
# fixed 8 frames, the frame count is derived from the window: ~30-day frames,
# at most 12. Windows longer than that capacity are equal-divided into 12 frames
# (so the whole window is covered, not just its first year) with a warning,
# since the wider spacing falls outside Galileo's monthly training cadence.
# ---------------------------------------------------------------------------

# "auto" (default) picks single/multi from the window: multi when the range spans
# ≥2 monthly frames, single otherwise. Use "single"/"multi" to force a mode.
_DEFAULT_TEMPORAL_MODE = "auto"
_FRAME_STRIDE_DAYS = 30
_MAX_FRAMES = 12


def _normalize_temporal_mode(mode: Any) -> str:
    m = str(mode or _DEFAULT_TEMPORAL_MODE).strip().lower()
    if m not in ("single", "multi", "auto"):
        raise ModelError(
            f"galileo temporal_mode must be 'single', 'multi', or 'auto', got {mode!r}."
        )
    return m


def _resolve_temporal_mode(model_config: dict[str, Any] | None) -> str:
    """Resolve the *configured* temporal mode (may be ``"auto"``)."""
    v = model_config_value(model_config, "temporal_mode")
    if v is not None:
        return _normalize_temporal_mode(v)
    env = os.environ.get("RS_EMBED_GALILEO_TEMPORAL_MODE", "").strip()
    if env:
        return _normalize_temporal_mode(env)
    return _DEFAULT_TEMPORAL_MODE


def _temporal_bins(t: TemporalSpec) -> tuple[tuple[tuple[str, str], ...], bool]:
    """Monthly frames for the window, returned as ``(bins, stretched)``.

    Up to ``_MAX_FRAMES`` (12) fixed ``_FRAME_STRIDE_DAYS`` (30-day) frames,
    mirroring Galileo's monthly pretraining cadence. Windows longer than that
    capacity are **equal-divided into 12 frames** instead of dropping the
    trailing time, so the whole window is covered (``stretched=True``); see
    :func:`rs_embed.tools.temporal.fixed_or_equal_bins`.
    """
    from ..tools.temporal import fixed_or_equal_bins  # noqa: PLC0415

    return fixed_or_equal_bins(
        str(t.start), str(t.end), stride_days=_FRAME_STRIDE_DAYS, max_bins=_MAX_FRAMES
    )


def _expand_auto_mode(mode: str, t: TemporalSpec) -> str:
    """Expand ``"auto"`` to ``"single"``/``"multi"`` from the window.

    Resolves to ``"multi"`` when the range spans ≥2 monthly frames (it genuinely
    covers multiple time steps), else ``"single"`` — a single frame where multi
    would add nothing but extra GEE fetches. ``"single"``/``"multi"`` pass through.
    """
    if mode != "auto":
        return mode
    bins, _ = _temporal_bins(t)
    return "multi" if len(bins) >= 2 else "single"


def _explicit_n_frames(model_config: dict[str, Any] | None) -> int | None:
    """Manual frame-count override (escape hatch): model_config > env > None.

    When set, the fixed count is used as-is and the adaptive monthly policy is
    bypassed — the caller takes responsibility for the spacing.
    """
    v = model_config_value(model_config, "n_frames")
    if v is not None:
        return max(1, int(v))
    env = os.environ.get("RS_EMBED_GALILEO_FRAMES", "").strip()
    if env:
        return max(1, int(env))
    return None


def _temporal_sampling_meta(bins: tuple[tuple[str, str], ...], stretched: bool) -> dict[str, Any]:
    """Pure metadata describing the temporal binning mode (no side effects)."""
    meta: dict[str, Any] = {
        "temporal_sampling": "equal_divided" if stretched else "fixed_stride",
        "temporal_spacing_stretched": bool(stretched),
    }
    if stretched and bins:
        span = (date.fromisoformat(bins[-1][1]) - date.fromisoformat(bins[0][0])).days
        meta["effective_stride_days"] = int(round(span / max(1, len(bins))))
    return meta


def _resolve_frame_plan(
    model_config: dict[str, Any] | None, t: TemporalSpec
) -> tuple[int, dict[str, Any]]:
    """Resolve ``(n_frames, sampling_meta)`` for the requested window.

    - ``temporal_mode`` ``single`` (or ``auto`` → single for a sub-month window)
      → ``T=1``.
    - explicit ``RS_EMBED_GALILEO_FRAMES`` / ``model_config['n_frames']`` → that
      fixed count (manual escape hatch; bypasses the adaptive monthly policy).
    - otherwise → window-adaptive count from :func:`_temporal_bins` (≤12), with a
      ``stretched`` flag for windows beyond the 12-month capacity.
    """
    mode = _expand_auto_mode(_resolve_temporal_mode(model_config), t)
    if mode == "single":
        return 1, {
            "temporal_mode": "single",
            "temporal_sampling": "single",
            "temporal_spacing_stretched": False,
        }
    explicit = _explicit_n_frames(model_config)
    if explicit is not None:
        return explicit, {
            "temporal_mode": "multi",
            "temporal_sampling": "manual",
            "temporal_spacing_stretched": False,
        }
    bins, stretched = _temporal_bins(t)
    meta = {"temporal_mode": "multi", **_temporal_sampling_meta(bins, stretched)}
    return len(bins), meta


def _warn_stretched_sampling(sampling_meta: dict[str, Any]) -> None:
    """Emit a single warning when equal-division (stretched) sampling was used."""
    if not sampling_meta.get("temporal_spacing_stretched"):
        return
    warnings.warn(
        f"Galileo window exceeds {_MAX_FRAMES} × {_FRAME_STRIDE_DAYS}-day frames; "
        f"switched to equal division (~{sampling_meta.get('effective_stride_days')}d apart) "
        "to cover the whole window instead of dropping the trailing time. Frame spacing is "
        "outside Galileo's monthly training cadence, so embeddings are extrapolated — "
        "narrow the temporal window to stay in-distribution.",
        UserWarning,
        stacklevel=2,
    )


@lru_cache(maxsize=6)
def _load_galileo_cached(
    *,
    model_size: str,
    model_path: str | None,
    hf_repo: str,
    cache_dir: str | None,
    auto_download: bool,
    dev: str,
) -> tuple[Any, dict[str, Any], Any]:
    ensure_torch()
    import torch

    model_root = _resolve_model_folder(
        model_path=model_path,
        model_size=model_size,
        hf_repo=hf_repo,
        cache_dir=cache_dir,
        auto_download=auto_download,
    )

    mod = _load_galileo_module()
    if not hasattr(mod, "Encoder"):
        raise ModelError("Vendored Galileo runtime does not expose Encoder class.")

    model_folder = Path(model_root)
    load_fn = getattr(mod.Encoder, "load_from_folder", None)
    if load_fn is None:
        raise ModelError("Galileo Encoder class has no load_from_folder method.")

    try:
        encoder = load_fn(model_folder, torch.device(dev))
    except TypeError:
        # compatibility with src.galileo signature without device
        encoder = load_fn(model_folder)

    encoder = _move_model_to_device(encoder, dev, model_name="Galileo")

    wstats = verify_loaded_params(
        encoder,
        model_name="Galileo",
        no_params_msg="Galileo encoder has no parameters; cannot verify load.",
    )

    meta = {
        "model_size": str(model_size),
        "model_root": model_root,
        "model_source": (model_root if model_path else f"hf://{hf_repo}/models/{model_size}"),
        "device": str(dev),
        **wstats,
    }
    return encoder, meta, mod


def _load_galileo(
    *,
    model_size: str,
    model_path: str | None,
    hf_repo: str,
    cache_dir: str | None,
    auto_download: bool,
    device: str,
) -> tuple[Any, dict[str, Any], Any, str]:
    (loaded, dev) = _load_cached_with_device(
        _load_galileo_cached,
        device=device,
        model_size=str(model_size),
        model_path=(os.path.expanduser(model_path) if model_path else None),
        hf_repo=str(hf_repo),
        cache_dir=(os.path.expanduser(cache_dir) if cache_dir else None),
        auto_download=bool(auto_download),
    )
    encoder, meta, mod = loaded
    return encoder, meta, mod, dev


def _prepare_galileo_encoder_inputs(
    raw_tchw: np.ndarray,
    *,
    image_size: int,
    patch_size: int,
    months_seq: np.ndarray,
    norm_mode: str,
    mod: Any,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_torch()
    import torch

    if raw_tchw.ndim != 4 or int(raw_tchw.shape[1]) != 10:
        raise ModelError(
            f"Galileo expects TCHW with C=10 S2 bands, got {getattr(raw_tchw, 'shape', None)}"
        )
    if image_size <= 0:
        raise ModelError(f"image_size must be > 0, got {image_size}")
    if patch_size <= 0:
        raise ModelError(f"patch_size must be > 0, got {patch_size}")
    if (image_size % patch_size) != 0:
        raise ModelError(
            f"Galileo requires image_size divisible by patch_size, got image_size={image_size}, patch_size={patch_size}"
        )

    mode_l = str(norm_mode).lower().strip()
    raw_norm_mode = "none" if _is_galileo_official_stats_mode(mode_l) else norm_mode

    x_tchw = raw_tchw.astype(np.float32, copy=False)
    if x_tchw.shape[-2] != image_size or x_tchw.shape[-1] != image_size:
        x_tchw = _resize_tchw(x_tchw, out_hw=image_size)
    x_tchw = _normalize_s2(x_tchw, mode=raw_norm_mode)  # [T,10,H,W]

    # [H,W,T,10]
    s2_hwtd = np.transpose(x_tchw, (2, 3, 0, 1)).astype(np.float32)
    t = int(s2_hwtd.shape[2])

    # create Galileo space_time tensor [B,H,W,T,len(SPACE_TIME_BANDS)]
    space_time_bands = list(mod.SPACE_TIME_BANDS)
    s2_bands = list(mod.S2_BANDS)
    s_t_groups = list(mod.SPACE_TIME_BANDS_GROUPS_IDX.keys())

    h, w = int(s2_hwtd.shape[0]), int(s2_hwtd.shape[1])
    s_t_x = np.zeros((1, h, w, t, len(space_time_bands)), dtype=np.float32)

    s2_map = [space_time_bands.index(b) for b in s2_bands]
    # Use a basic slice first so NumPy keeps [H,W,T,C] order during assignment.
    s_t_x[0][..., s2_map] = s2_hwtd

    # masks: 0 means seen by encoder, 1 means masked/ignored
    s_t_m = np.ones((1, h, w, t, len(s_t_groups)), dtype=np.float32)
    s2_group_indices = [i for i, key in enumerate(s_t_groups) if "S2" in str(key)]
    for idx in s2_group_indices:
        s_t_m[0, :, :, :, idx] = 0.0

    sp_len = len(mod.SPACE_BANDS)
    t_len = len(mod.TIME_BANDS)
    st_len = len(mod.STATIC_BANDS)
    sp_group_len = len(mod.SPACE_BAND_GROUPS_IDX)
    t_group_len = len(mod.TIME_BAND_GROUPS_IDX)
    st_group_len = len(mod.STATIC_BAND_GROUPS_IDX)

    sp_x = np.zeros((1, h, w, sp_len), dtype=np.float32)
    t_x = np.zeros((1, t, t_len), dtype=np.float32)
    st_x = np.zeros((1, st_len), dtype=np.float32)

    sp_m = np.ones((1, h, w, sp_group_len), dtype=np.float32)
    t_m = np.ones((1, t, t_group_len), dtype=np.float32)
    st_m = np.ones((1, st_group_len), dtype=np.float32)

    months_arr = np.asarray(months_seq, dtype=np.int64).reshape(-1)
    if months_arr.size == 0:
        months_arr = np.full((t,), 5, dtype=np.int64)
    if months_arr.size < t:
        months_arr = np.concatenate(
            [
                months_arr,
                np.full((t - months_arr.size,), int(months_arr[-1]), dtype=np.int64),
            ],
            axis=0,
        )
    elif months_arr.size > t:
        months_arr = months_arr[:t]
    months_arr = np.clip(months_arr, 0, 11).astype(np.int64)
    months = months_arr[None, :]

    data = {
        "s_t_x": torch.from_numpy(s_t_x).to(device),
        "sp_x": torch.from_numpy(sp_x).to(device),
        "t_x": torch.from_numpy(t_x).to(device),
        "st_x": torch.from_numpy(st_x).to(device),
        "s_t_m": torch.from_numpy(s_t_m).to(device),
        "sp_m": torch.from_numpy(sp_m).to(device),
        "t_m": torch.from_numpy(t_m).to(device),
        "st_m": torch.from_numpy(st_m).to(device),
        "months": torch.from_numpy(months).to(device),
    }
    if _is_galileo_official_stats_mode(mode_l):
        _apply_galileo_pretrain_stats(data)
    meta = {
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "n_frames": int(t),
        "months": tuple(int(v) + 1 for v in months_arr.tolist()),
        "month": int(months_arr[len(months_arr) // 2]) + 1,
        "month_indices": tuple(int(v) for v in months_arr.tolist()),
        "normalization": "official_stats"
        if _is_galileo_official_stats_mode(mode_l)
        else str(mode_l),
        "include_ndvi": False,
        "s2_group_indices": tuple(int(i) for i in s2_group_indices),
    }
    return data, meta


def _galileo_forward(
    encoder: Any,
    data: dict[str, Any],
    *,
    mod: Any,
    patch_size: int,
    add_layernorm_on_exit: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ensure_torch()
    import torch

    dev = _resolve_device(device)
    encoder = _move_model_to_device(encoder, dev, model_name="Galileo")

    with torch.no_grad():
        out = encoder(
            data["s_t_x"],
            data["sp_x"],
            data["t_x"],
            data["st_x"],
            data["s_t_m"],
            data["sp_m"],
            data["t_m"],
            data["st_m"],
            data["months"],
            patch_size=int(patch_size),
            add_layernorm_on_exit=bool(add_layernorm_on_exit),
        )

    if not isinstance(out, (tuple, list)) or len(out) < 8:
        raise ModelError(f"Unexpected Galileo encoder output type: {type(out)}")

    s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m = out[:8]

    # pooled features from all visible tokens
    vec_t = encoder.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
    if vec_t.ndim != 2 or int(vec_t.shape[0]) != 1:
        raise ModelError(f"Unexpected Galileo pooled output shape: {tuple(vec_t.shape)}")
    vec = vec_t[0].detach().float().cpu().numpy().astype(np.float32)

    patch_avg_fn = getattr(encoder, "apply_mask_and_average_tokens_per_patch", None)
    grid_kind = "patch_tokens"
    grid_source = "official_patch_mean"
    if callable(patch_avg_fn):
        # Match upstream patch-level aggregation as closely as possible:
        # average all visible tokens assigned to each spatial patch.
        patch_t = patch_avg_fn(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        if patch_t.ndim != 3 or int(patch_t.shape[0]) != 1:
            raise ModelError(f"Unexpected Galileo patch grid shape: {tuple(patch_t.shape)}")
        ph = int(s_t_x.shape[1])
        pw = int(s_t_x.shape[2])
        if int(patch_t.shape[1]) != (ph * pw):
            raise ModelError(
                "Galileo patch averaging returned inconsistent token count: "
                f"{int(patch_t.shape[1])} vs {ph}*{pw}"
            )
        grid_hwd = patch_t.reshape(1, ph, pw, int(patch_t.shape[-1]))[0]
    else:
        # Compatibility fallback for non-upstream runtimes.
        s_t_groups = list(mod.SPACE_TIME_BANDS_GROUPS_IDX.keys())
        s2_group_indices = [i for i, key in enumerate(s_t_groups) if "S2" in str(key)]
        if not s2_group_indices:
            raise ModelError(
                "Failed to locate Galileo S2 group indices in SPACE_TIME_BANDS_GROUPS_IDX"
            )
        s_t_sel = s_t_x[:, :, :, :, s2_group_indices, :]
        grid_hwd = s_t_sel.mean(dim=3).mean(dim=3)[0]
        grid_kind = "s2_group_patch_tokens"
        grid_source = "legacy_s2_group_mean"

    grid = grid_hwd.detach().float().cpu().numpy().transpose(2, 0, 1).astype(np.float32)  # [D,H,W]

    fmeta = {
        "feature_dim": int(vec.shape[0]),
        "grid_shape": tuple(grid.shape),
        "grid_hw": (int(grid.shape[1]), int(grid.shape[2])),
        "grid_kind": str(grid_kind),
        "grid_source": str(grid_source),
    }
    return vec, grid, fmeta


@register("galileo")
class GalileoEmbedder(EmbedderBase):
    # Square-input model: marks it for the API tiling path (its own fetch_input
    # squares the ROI for the single-input case and skips it when tiling).
    _requires_square_input = True
    DEFAULT_MODEL_SIZE = "nano"
    DEFAULT_PATCH = 8
    DEFAULT_IMAGE_SIZE = 64
    DEFAULT_FRAMES = 8
    DEFAULT_FETCH_WORKERS = 8
    _allow_auto_backend = True

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_S2_10_BANDS),
        scale_m=10,
        cloudy_pct=30,
        temporal_mode="multi",
        n_frames=8,
        image_size=64,
        expected_channels=10,
    )

    # Explicit pipeline-routing capabilities; the contract test asserts these
    # match the actual method signatures (tests/test_capabilities_contract.py).
    capabilities = EmbedderCapabilities(
        input_chw=True,
        fetch_meta=True,
        fetch_temporal_mode=True,
        batch_fetch_metas=True,
        model_config_single=True,
        model_config_batch=True,
        model_config_batch_inputs=True,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "collection": self.input_spec.collection,
                "bands": list(self.input_spec.bands),
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "model_size": self.DEFAULT_MODEL_SIZE,
                "patch_size": self.DEFAULT_PATCH,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "temporal_mode": _DEFAULT_TEMPORAL_MODE,
                "scale_m": self.input_spec.scale_m,
                "cloudy_pct": self.input_spec.cloudy_pct,
                "composite": self.input_spec.composite,
                "normalization": "none",
            },
            "model_config": {
                "temporal_mode": {
                    "type": "string",
                    "default": _DEFAULT_TEMPORAL_MODE,
                    "choices": ["auto", "single", "multi"],
                    "description": (
                        "auto (default): single when the range spans one monthly frame, "
                        "else multi (≥2 frames). single: one composite over the whole range "
                        f"(T=1). multi: ~{_FRAME_STRIDE_DAYS}-day frames (max {_MAX_FRAMES}, "
                        "mirroring Galileo's monthly cadence; longer windows are equal-divided "
                        f"into {_MAX_FRAMES} frames with a warning)."
                    ),
                },
                "n_frames": {
                    "type": "int",
                    "default": None,
                    "description": (
                        "Manual frame-count override (escape hatch). When set, this fixed "
                        "count is used as-is and the adaptive monthly policy is bypassed. "
                        "Also settable via RS_EMBED_GALILEO_FRAMES."
                    ),
                },
            },
            "notes": [
                "Loads Galileo Encoder from a vendored local runtime.",
                "Defaults to Hugging Face model snapshots under nasaharvest/galileo/models/<size>/.",
                "Frame count is window-adaptive (~30-day frames, max 12) mirroring Galileo's "
                "monthly pretraining cadence; sub-month windows collapse to a single frame.",
                "Uses Sentinel-2 10 bands; pooled output averages visible Galileo tokens.",
            ],
        }

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return GalileoEmbedder.input_spec.to_sensor_spec()

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_GALILEO_FETCH_WORKERS",
                str(GalileoEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

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
        """Fetch a window-adaptive S2 time series as ``[T,C,H,W]``.

        Overrides the generic base prefetch (which would fetch a fixed frame
        count) so the API/export prefetch path uses the same monthly-adaptive
        frame count as :meth:`get_embedding`. ``model_config`` is not available
        here, so the mode falls back to ``RS_EMBED_GALILEO_TEMPORAL_MODE`` /
        ``"auto"`` and ``RS_EMBED_GALILEO_FRAMES`` for a manual override.
        """
        ss = sensor or self._default_sensor()
        t = temporal_to_range(temporal)
        # Fetch-square: enlarge a rectangular ROI to a square of real imagery so
        # the encoder sees a square, in-distribution input; the ROI window rides
        # in meta['roi_window_geo'] and the output is cropped back to it (tiled
        # paths crop the stitched grid once). All pipeline callers keep the
        # square_input=True default; False is an escape hatch for callers that
        # manage ROI geometry themselves.
        geo_roi = FULL_WINDOW
        if square_input:
            spatial, geo_roi = square_spatial(spatial)
        if temporal_mode is not None:
            # Honor an explicit mode by routing it through the same resolver.
            cfg: dict[str, Any] | None = {"temporal_mode": _normalize_temporal_mode(temporal_mode)}
        else:
            cfg = None
        n_frames, sampling_meta = _resolve_frame_plan(cfg, t)
        _warn_stretched_sampling(sampling_meta)
        raw_tchw = _fetch_s2_10_raw_tchw(
            provider,
            spatial,
            t,
            n_frames=n_frames,
            scale_m=int(ss.scale_m),
            cloudy_pct=int(ss.cloudy_pct),
            composite=str(ss.composite),
            fill_value=float(ss.fill_value),
        )
        meta = {**dict(sampling_meta), **(roi_fetch_meta(geo_roi) or {})}
        return FetchResult(data=raw_tchw, meta=meta)

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
            raise ModelError("galileo expects a provider backend (or 'auto').")

        ss = sensor or self._default_sensor()
        t = temporal_to_range(temporal)
        # Fetch-square ROI window: from the fetch when we fetch here, or carried in
        # fetch_meta when the API prefetched a square. Output is cropped back to it.
        geo_roi = geo_roi_from_meta(fetch_meta)

        model_size = os.environ.get("RS_EMBED_GALILEO_MODEL_SIZE", self.DEFAULT_MODEL_SIZE).strip()
        model_path = os.environ.get("RS_EMBED_GALILEO_MODEL_PATH")
        hf_repo = os.environ.get("RS_EMBED_GALILEO_HF_REPO", "nasaharvest/galileo").strip()
        cache_dir = os.environ.get(
            "RS_EMBED_GALILEO_CACHE_DIR",
            os.path.join("~", ".cache", "rs_embed", "galileo"),
        )
        auto_download = os.environ.get("RS_EMBED_GALILEO_AUTO_DOWNLOAD", "1").strip() not in {
            "0",
            "false",
            "False",
        }

        image_size = int(os.environ.get("RS_EMBED_GALILEO_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        patch_size = int(os.environ.get("RS_EMBED_GALILEO_PATCH", str(self.DEFAULT_PATCH)))
        norm_mode = os.environ.get("RS_EMBED_GALILEO_NORM", "none").strip()
        add_layernorm = os.environ.get("RS_EMBED_GALILEO_ADD_LN", "1").strip() not in {
            "0",
            "false",
            "False",
        }
        month_override = os.environ.get("RS_EMBED_GALILEO_MONTH")

        # Window-adaptive frame count (~30-day frames, max 12), mirroring Galileo's
        # monthly pretraining cadence. See _resolve_frame_plan / _temporal_bins.
        n_frames, sampling_meta = _resolve_frame_plan(model_config, t)

        if input_chw is None:
            provider = self._get_provider(backend)
            _warn_stretched_sampling(sampling_meta)  # warn only when we actually fetch
            spatial, geo_roi = square_spatial(spatial)  # enlarge rectangle to square
            raw_tchw = _fetch_s2_10_raw_tchw(
                provider,
                spatial,
                t,
                n_frames=n_frames,
                scale_m=int(ss.scale_m),
                cloudy_pct=int(ss.cloudy_pct),
                composite=str(ss.composite),
                fill_value=float(ss.fill_value),
            )
        else:
            # Preserve a provided time series' own T; repeat a single CHW to the plan.
            raw = np.asarray(input_chw)
            coerce_frames = int(raw.shape[0]) if raw.ndim == 4 else n_frames
            raw_tchw = _coerce_input_to_tchw(
                input_chw,
                expected_channels=10,
                n_frames=coerce_frames,
                model_name="galileo",
            )

        # Frame diversity of the series actually fed to the encoder (tiles preserve
        # the back-filled duplicates), so it lands on the tiled path too.
        diversity_meta = frame_diversity_meta(
            n_requested=int(raw_tchw.shape[0]),
            n_distinct=count_distinct_frames(raw_tchw),
        )

        if month_override is not None:
            months_seq = _month_override_sequence(
                int(month_override),
                n_frames=int(raw_tchw.shape[0]),
            )
        else:
            months_seq = _frame_month_sequence(t, n_frames=int(raw_tchw.shape[0]))

        encoder, lmeta, mod, dev = _load_galileo(
            model_size=model_size,
            model_path=model_path,
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            auto_download=auto_download,
            device=device,
        )

        inputs, pmeta = _prepare_galileo_encoder_inputs(
            raw_tchw,
            image_size=image_size,
            patch_size=patch_size,
            months_seq=months_seq,
            norm_mode=norm_mode,
            mod=mod,
            device=dev,
        )

        vec, grid, fmeta = _galileo_forward(
            encoder,
            inputs,
            mod=mod,
            patch_size=patch_size,
            add_layernorm_on_exit=add_layernorm,
            device=dev,
        )

        # Crop the token grid back to the ROI when the fetch enlarged a rectangle
        # to a square (no-op when geo_roi is full). grid is (D, H', W').
        cropped_to_roi = not roi_is_full(geo_roi)
        if cropped_to_roi:
            grid = crop_grid_to_roi(grid, geo_roi)

        meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=str(backend).lower(),
            source=ss.collection,
            sensor={
                "collection": ss.collection,
                "bands": tuple(_S2_10_BANDS),
                "scale_m": int(ss.scale_m),
                "cloudy_pct": int(ss.cloudy_pct),
                "composite": str(ss.composite),
                "fill_value": float(ss.fill_value),
            },
            temporal=t,
            image_size=image_size,
            extra={
                "start": t.start,
                "end": t.end,
                "patch_size": int(patch_size),
                "n_frames": int(raw_tchw.shape[0]),
                "normalization": str(norm_mode),
                "include_ndvi": False,
                "device": dev,
                **diversity_meta,
                **sampling_meta,
                **lmeta,
                **pmeta,
                **fmeta,
            },
        )

        if output.mode == "pooled":
            if output.pooling == "max":
                # keep pooled mode semantics with optional max over grid
                vec_out = np.max(grid, axis=(1, 2)).astype(np.float32)
                pooling = "grid_max"
            elif cropped_to_roi:
                # Pool only the ROI's tokens (the global token mean would include
                # the real-neighborhood context fetched to make the input square).
                vec_out = np.mean(grid, axis=(1, 2)).astype(np.float32)
                pooling = "roi_grid_mean"
            else:
                vec_out = vec.astype(np.float32)
                pooling = "token_mean"
            ometa = {**meta, "pooling": pooling, "pooled_shape": tuple(vec_out.shape)}
            return Embedding(data=vec_out, meta=ometa)

        if output.mode == "grid":
            gmeta = {
                **meta,
                "grid_kind": str(meta.get("grid_kind", "patch_tokens")),
                "grid_shape": tuple(grid.shape),
                "grid_hw": (int(grid.shape[1]), int(grid.shape[2])),
            }
            da = grid_to_dataarray(grid.astype(np.float32), meta=gmeta)
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

        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("galileo expects a provider backend (or 'auto').")

        t = temporal_to_range(temporal)
        ss = sensor or self._default_sensor()
        provider = self._get_provider(backend)
        # Window-adaptive frame count shared by all points (same window); warn once.
        n_frames, sampling_meta = _resolve_frame_plan(model_config, t)
        _warn_stretched_sampling(sampling_meta)

        # Square-fetch each ROI, then re-feed get_embedding with the ROI window so
        # the output is cropped back. square_fetch_batch handles square + parallel.
        raws, geo_rois = square_fetch_batch(
            spatials,
            lambda sq: _fetch_s2_10_raw_tchw(
                provider,
                sq,
                t,
                n_frames=n_frames,
                scale_m=int(ss.scale_m),
                cloudy_pct=int(ss.cloudy_pct),
                composite=str(ss.composite),
                fill_value=float(ss.fill_value),
            ),
            max_workers=self._resolve_fetch_workers(len(spatials)),
        )
        return [
            self.get_embedding(
                spatial=sp,
                temporal=t,
                sensor=ss,
                output=output,
                backend=backend,
                device=device,
                input_chw=raw,
                model_config=model_config,
                fetch_meta=roi_fetch_meta(gr),
            )
            for sp, raw, gr in zip(spatials, raws, geo_rois, strict=True)
        ]

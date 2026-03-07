from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..providers import ProviderBase
from ._vit_mae_utils import (
    base_meta,
    ensure_torch,
    pool_from_tokens,
    temporal_to_range,
    tokens_to_grid_dhw,
)
from .base import EmbedderBase
from .runtime_utils import (
    fetch_sensor_patch_chw as _fetch_sensor_patch_chw,
    is_provider_backend,
    load_cached_with_device as _load_cached_with_device,
    resolve_device_auto_torch as _resolve_device,
)


# SatVision-TOA model defaults from published config/model card.
_DEFAULT_MODEL_ID = (
    "nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128"
)
_DEFAULT_IN_CHANS = 14
_DEFAULT_IMAGE_SIZE = 128
_DEFAULT_PATCH_SIZE = 4
_DEFAULT_MODIS_COLLECTION = "MODIS/061/MOD021KM"
_DEFAULT_MODIS_BANDS = (
    "1",
    "2",
    "3",
    "26",
    "6",
    "20",
    "7",
    "27",
    "28",
    "29",
    "31",
    "32",
    "33",
    "34",
)
_DEFAULT_REFLECTANCE_INDICES = (0, 1, 2, 3, 4, 6)
_DEFAULT_EMISSIVE_INDICES = (5, 7, 8, 9, 10, 11, 12, 13)
_DEFAULT_EMISSIVE_MINS = (
    223.1222,
    178.9174,
    204.3739,
    204.7677,
    194.8686,
    202.1759,
    201.3823,
    203.3537,
)
_DEFAULT_EMISSIVE_MAXS = (
    352.7182,
    261.2920,
    282.5529,
    319.0373,
    295.0209,
    324.0677,
    321.5254,
    285.9848,
)
_FALLBACK_MOD09_COLLECTION = "MODIS/061/MOD09GA"
_FALLBACK_MOD21_COLLECTION = "MODIS/061/MOD21A1D"


def _parse_int_list(s: str) -> Tuple[int, ...]:
    out: List[int] = []
    for x in str(s).split(","):
        t = x.strip()
        if not t:
            continue
        out.append(int(t))
    return tuple(out)


def _parse_float_list(s: str) -> Tuple[float, ...]:
    out: List[float] = []
    for x in str(s).split(","):
        t = x.strip()
        if not t:
            continue
        out.append(float(t))
    return tuple(out)


def _normalize_indices(indices: Sequence[int], n_channels: int) -> Tuple[int, ...]:
    out: List[int] = []
    for i in indices:
        ii = int(i)
        if ii < 0:
            ii = n_channels + ii
        if 0 <= ii < n_channels and ii not in out:
            out.append(ii)
    return tuple(out)


def _as_bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() not in {"0", "false", "no", "off"}


def _resize_chw(x_chw: np.ndarray, *, out_size: int) -> np.ndarray:
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW input, got {getattr(x_chw, 'shape', None)}")
    if int(out_size) <= 0:
        raise ModelError(f"image_size must be > 0, got {out_size}")

    c, h, w = int(x_chw.shape[0]), int(x_chw.shape[1]), int(x_chw.shape[2])
    if h == out_size and w == out_size:
        return x_chw.astype(np.float32, copy=False)

    xt = torch.from_numpy(x_chw.astype(np.float32, copy=False)).reshape(1, c, h, w)
    yt = F.interpolate(
        xt, size=(int(out_size), int(out_size)), mode="bilinear", align_corners=False
    )
    return yt[0].detach().cpu().numpy().astype(np.float32)


def _normalize_satvision_toa_input(
    raw_chw: np.ndarray,
    *,
    mode: str,
    reflectance_indices: Sequence[int],
    emissive_indices: Sequence[int],
    reflectance_divisor: float,
    emissive_mins: Sequence[float],
    emissive_maxs: Sequence[float],
) -> np.ndarray:
    """
    Normalize SatVision-TOA inputs to [0, 1].

    - Reflectance channels: value / reflectance_divisor
    - Emissive channels: (K - min) / (max - min)

    mode:
      - auto: if input already looks unit-scaled, keep as-is (clipped)
      - unit: clip to [0,1]
      - raw: apply channel-wise scaling
    """
    x = np.asarray(raw_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ModelError(
            f"SatVision-TOA input must be CHW, got {getattr(x, 'shape', None)}"
        )

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mode_l = str(mode).strip().lower()

    if mode_l not in {"auto", "raw", "unit"}:
        raise ModelError("RS_EMBED_SATVISION_TOA_NORM must be one of: auto, raw, unit")

    # Heuristic: treat as already normalized if values are near [0,1].
    if mode_l == "auto":
        vmin = float(np.nanmin(x)) if x.size else 0.0
        vmax = float(np.nanmax(x)) if x.size else 0.0
        if vmin >= -0.1 and vmax <= 1.5:
            return np.clip(x, 0.0, 1.0).astype(np.float32)
        mode_l = "raw"

    if mode_l == "unit":
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    # raw mode
    if reflectance_divisor <= 0:
        raise ModelError(f"reflectance_divisor must be > 0, got {reflectance_divisor}")
    n_channels = int(x.shape[0])
    r_indices = _normalize_indices(reflectance_indices, n_channels)
    e_indices = _normalize_indices(emissive_indices, n_channels)
    if len(e_indices) != len(emissive_mins) or len(e_indices) != len(emissive_maxs):
        raise ModelError(
            "SatVision emissive calibration mismatch: "
            f"len(emissive_indices)={len(e_indices)}, "
            f"len(emissive_mins)={len(emissive_mins)}, "
            f"len(emissive_maxs)={len(emissive_maxs)}"
        )

    e_min_map = {
        int(e_indices[i]): float(emissive_mins[i]) for i in range(len(e_indices))
    }
    e_max_map = {
        int(e_indices[i]): float(emissive_maxs[i]) for i in range(len(e_indices))
    }
    r_set = set(int(i) for i in r_indices)
    e_set = set(int(i) for i in e_indices)
    y = np.empty_like(x, dtype=np.float32)

    for c in range(n_channels):
        ch = x[c]
        if c in r_set:
            ch = ch / float(reflectance_divisor)
        elif c in e_set:
            lo = float(e_min_map[c])
            hi = float(e_max_map[c])
            if hi <= lo:
                raise ModelError(
                    f"Invalid emissive calibration range for channel index {c}: ({lo}, {hi})"
                )
            ch = (ch - lo) / (hi - lo)
        else:
            # Fallback for unspecified channels: use reflectance scaling.
            ch = ch / float(reflectance_divisor)
        y[c] = ch

    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _pick_first_tensor(obj: Any):
    """Best-effort pick first tensor-like value from nested output."""
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    if torch is not None and torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for it in obj:
            t = _pick_first_tensor(it)
            if t is not None:
                return t
        return None
    if isinstance(obj, dict):
        for k in ("x", "features", "tokens", "last_hidden_state", "out"):
            if k in obj:
                t = _pick_first_tensor(obj[k])
                if t is not None:
                    return t
        for _, v in obj.items():
            t = _pick_first_tensor(v)
            if t is not None:
                return t
    return None


def _decode_features_to_batch_arrays(
    out: Any, batch_size: int
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Convert model.forward_features output to per-item arrays.

    Returns arrays where each item is either:
      - tokens: [N, D]
      - pooled: [D]
    """
    ensure_torch()
    import torch

    t = _pick_first_tensor(out)
    if t is None or (not torch.is_tensor(t)):
        raise ModelError(
            "SatVision-TOA forward_features returned unsupported output type."
        )

    if int(t.shape[0]) != int(batch_size):
        raise ModelError(
            f"SatVision-TOA forward_features batch mismatch: got B={int(t.shape[0])}, expected {batch_size}"
        )

    # [B, N, D]
    if t.ndim == 3:
        arr = t.detach().float().cpu().numpy().astype(np.float32)
        return [arr[i] for i in range(arr.shape[0])], {
            "tokens_kind": "tokens",
            "tensor_shape": tuple(arr.shape),
        }

    # [B, H, W, D] or [B, D, H, W]
    if t.ndim == 4:
        s = tuple(int(v) for v in t.shape)
        # BHWC: spatial dims small, channel dim large
        if s[1] <= 64 and s[2] <= 64 and s[3] > 64:
            toks = t.reshape(s[0], s[1] * s[2], s[3])
            arr = toks.detach().float().cpu().numpy().astype(np.float32)
            return [arr[i] for i in range(arr.shape[0])], {
                "tokens_kind": "tokens_feature_map_bhwc",
                "feature_map_hw": (s[1], s[2]),
                "tensor_shape": s,
            }
        # BCHW: channel dim large, spatial dims small
        if s[1] > 64 and s[2] <= 64 and s[3] <= 64:
            toks = t.permute(0, 2, 3, 1).reshape(s[0], s[2] * s[3], s[1])
            arr = toks.detach().float().cpu().numpy().astype(np.float32)
            return [arr[i] for i in range(arr.shape[0])], {
                "tokens_kind": "tokens_feature_map_bchw",
                "feature_map_hw": (s[2], s[3]),
                "tensor_shape": s,
            }

    # [B, D] pooled vector
    if t.ndim == 2:
        arr = t.detach().float().cpu().numpy().astype(np.float32)
        return [arr[i] for i in range(arr.shape[0])], {
            "tokens_kind": "pooled",
            "tensor_shape": tuple(arr.shape),
        }

    raise ModelError(
        f"SatVision-TOA forward_features shape unsupported: {tuple(int(v) for v in t.shape)}"
    )


def _extract_state_dict(obj: Any) -> Dict[str, Any]:
    """Extract a tensor state_dict from common checkpoint layouts."""
    ensure_torch()
    import torch

    if isinstance(obj, dict):
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "module",
            "net",
            "weights",
        ):
            v = obj.get(k)
            if isinstance(v, dict) and v:
                # Accept dict if at least one tensor value.
                if any(torch.is_tensor(x) for x in v.values()):
                    return v
        if obj and any(torch.is_tensor(x) for x in obj.values()):
            return obj
    raise ModelError(
        "Unsupported SatVision checkpoint format: cannot locate state_dict"
    )


def _strip_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    plen = len(prefix)
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
        else:
            out[k] = v
    return out


def _select_best_state_dict(
    sd: Dict[str, Any], model_keys: Iterable[str]
) -> Dict[str, Any]:
    mkeys = set(model_keys)
    candidates = [
        sd,
        _strip_prefix(sd, "model."),
        _strip_prefix(sd, "module."),
        _strip_prefix(sd, "encoder."),
        _strip_prefix(_strip_prefix(sd, "model."), "module."),
    ]
    best = candidates[0]
    best_hit = -1
    for cand in candidates:
        hit = sum(1 for k in cand.keys() if k in mkeys)
        if hit > best_hit:
            best_hit = hit
            best = cand
    if best_hit <= 0:
        raise ModelError("SatVision checkpoint keys do not match model architecture.")
    return _align_swinv2_downsample_layout(best, mkeys)


def _downsample_indices(keys: Iterable[str]) -> Tuple[int, ...]:
    idx: set[int] = set()
    for k in keys:
        m = re.match(r"layers\.(\d+)\.downsample\.", str(k))
        if m:
            idx.add(int(m.group(1)))
    return tuple(sorted(idx))


def _shift_downsample_indices(sd: Dict[str, Any], delta: int) -> Dict[str, Any]:
    if delta == 0:
        return sd
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        m = re.match(r"layers\.(\d+)\.downsample\.(.*)", str(k))
        if m:
            i = int(m.group(1)) + int(delta)
            nk = f"layers.{i}.downsample.{m.group(2)}"
            out[nk] = v
        else:
            out[k] = v
    return out


def _align_swinv2_downsample_layout(
    sd: Dict[str, Any], model_keys: Iterable[str]
) -> Dict[str, Any]:
    """
    Align known SwinV2 naming layout mismatch between checkpoints and timm versions:
      - checkpoint: layers.{0,1,2}.downsample.*
      - model     : layers.{1,2,3}.downsample.*
    """
    state_idx = _downsample_indices(sd.keys())
    model_idx = _downsample_indices(model_keys)
    if not state_idx or not model_idx:
        return sd
    if state_idx == model_idx:
        return sd

    if len(state_idx) == len(model_idx):
        delta = model_idx[0] - state_idx[0]
        shifted = tuple(i + delta for i in state_idx)
        if shifted == model_idx:
            return _shift_downsample_indices(sd, delta)
    return sd


def _find_ckpt_file(path_or_dir: str) -> str:
    p = os.path.expanduser(path_or_dir)
    if os.path.isfile(p):
        return p
    if not os.path.isdir(p):
        raise ModelError(f"SatVision checkpoint path not found: {p}")

    preferred = (
        "model_best.pth",
        "pytorch_model.bin",
        "model.pth",
        "checkpoint.pth",
        "checkpoint.ckpt",
    )
    for name in preferred:
        q = os.path.join(p, name)
        if os.path.isfile(q):
            return q

    all_files: List[str] = []
    for fn in os.listdir(p):
        low = fn.lower()
        if low.endswith((".pth", ".pt", ".ckpt", ".bin")):
            all_files.append(os.path.join(p, fn))
    if not all_files:
        raise ModelError(f"No checkpoint file found in directory: {p}")

    # Pick the largest as a reasonable fallback.
    all_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return all_files[0]


def _resolve_ckpt(
    *,
    model_id: str,
    local_ckpt: Optional[str],
    auto_download: bool,
) -> Tuple[str, str]:
    """Return (ckpt_file, source)."""
    if local_ckpt:
        ckpt_file = _find_ckpt_file(local_ckpt)
        return ckpt_file, "local"

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ModelError(
            "SatVision-TOA requires huggingface_hub when RS_EMBED_SATVISION_TOA_CKPT is not set. "
            "Install: pip install huggingface_hub"
        ) from e

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        snap = snapshot_download(
            repo_id=model_id,
            token=token,
            local_files_only=not bool(auto_download),
            allow_patterns=[
                "*.pth",
                "*.pt",
                "*.ckpt",
                "*.bin",
                "*config*.yml",
                "*config*.yaml",
            ],
        )
    except Exception as e:
        msg = (
            "Failed to download/access SatVision-TOA weights from Hugging Face. "
            "If the model is gated/private, set HF_TOKEN or provide RS_EMBED_SATVISION_TOA_CKPT. "
            "Known public repo: "
            "nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128"
        )
        raise ModelError(msg) from e

    ckpt_file = _find_ckpt_file(snap)
    return ckpt_file, f"hf://{model_id}"


@lru_cache(maxsize=8)
def _load_satvision_toa_cached(
    *,
    ckpt_file: str,
    dev: str,
    in_chans: int,
) -> Tuple[Any, Dict[str, Any]]:
    ensure_torch()
    import torch

    try:
        from timm.models.swin_transformer_v2 import SwinTransformerV2
    except Exception as e:
        raise ModelError(
            "SatVision-TOA requires timm. Install: pip install timm"
        ) from e

    # Build architecture from SatVision published config.
    try:
        model = SwinTransformerV2(
            img_size=_DEFAULT_IMAGE_SIZE,
            patch_size=_DEFAULT_PATCH_SIZE,
            in_chans=int(in_chans),
            num_classes=0,
            embed_dim=512,
            depths=(2, 2, 42, 2),
            num_heads=(16, 32, 64, 128),
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True,
            pretrained_window_sizes=(12, 12, 12, 6),
        )
    except TypeError:
        # Older timm versions may not accept pretrained_window_sizes.
        model = SwinTransformerV2(
            img_size=_DEFAULT_IMAGE_SIZE,
            patch_size=_DEFAULT_PATCH_SIZE,
            in_chans=int(in_chans),
            num_classes=0,
            embed_dim=512,
            depths=(2, 2, 42, 2),
            num_heads=(16, 32, 64, 128),
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True,
        )

    obj = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)
    state = _select_best_state_dict(state, model.state_dict().keys())

    missing, unexpected = model.load_state_dict(state, strict=False)
    matched = len([k for k in model.state_dict().keys() if k in state])
    if matched <= 0:
        raise ModelError(
            "SatVision-TOA checkpoint load failed: no parameters matched model keys."
        )

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
        raise ModelError("SatVision-TOA model has no parameters after load.")
    if not torch.isfinite(p0).all():
        raise ModelError(
            "SatVision-TOA parameters contain NaN/Inf after checkpoint load."
        )

    p0f = p0.float()
    meta = {
        "checkpoint": ckpt_file,
        "device": dev,
        "matched_keys": int(matched),
        "missing_keys": int(len(missing)),
        "unexpected_keys": int(len(unexpected)),
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
    }
    return model, meta


def _load_satvision_toa(
    *,
    model_id: str,
    local_ckpt: Optional[str],
    auto_download: bool,
    in_chans: int,
    device: str,
) -> Tuple[Any, Dict[str, Any]]:
    ckpt_file, source = _resolve_ckpt(
        model_id=model_id, local_ckpt=local_ckpt, auto_download=auto_download
    )
    loaded, dev = _load_cached_with_device(
        _load_satvision_toa_cached,
        device=device,
        ckpt_file=ckpt_file,
        in_chans=int(in_chans),
    )
    model, meta = loaded
    return model, {**meta, "checkpoint_source": source, "model_id": model_id}


def _fetch_toa_raw_chw_from_gee(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    sensor: SensorSpec,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fetch generic CHW patch from provider according to SensorSpec.

    When MOD021KM is unavailable in the current EE project, automatically
    fallback to a surrogate built from MOD09GA (reflectance) + MOD21A1D (thermal).
    """
    try:
        arr = _fetch_sensor_patch_chw(
            provider,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
        )
        return arr, {
            "source_collection": str(sensor.collection),
            "fallback_used": False,
            "already_unit_scaled": False,
        }
    except Exception as e:
        if str(sensor.collection) != _DEFAULT_MODIS_COLLECTION:
            raise
        # Fallback path for environments where MOD021KM is not exposed in EE.
        # Build a 14-channel surrogate:
        #  - Reflectance proxies from MOD09GA
        #  - Thermal/emissive proxies from MOD21A1D
        # Output is already normalized to [0,1].
        ref_sensor = SensorSpec(
            collection=_FALLBACK_MOD09_COLLECTION,
            bands=(
                "sur_refl_b01",
                "sur_refl_b02",
                "sur_refl_b03",
                "sur_refl_b04",
                "sur_refl_b06",
                "sur_refl_b07",
            ),
            scale_m=int(sensor.scale_m),
            cloudy_pct=None,  # type: ignore[arg-type]
            fill_value=float(sensor.fill_value),
            composite=str(sensor.composite),
        )
        th_sensor = SensorSpec(
            collection=_FALLBACK_MOD21_COLLECTION,
            bands=("LST_1KM", "Emis_29", "Emis_31", "Emis_32"),
            scale_m=int(sensor.scale_m),
            cloudy_pct=None,  # type: ignore[arg-type]
            fill_value=float(sensor.fill_value),
            composite=str(sensor.composite),
        )
        ref = _fetch_sensor_patch_chw(
            provider,
            spatial=spatial,
            temporal=temporal,
            sensor=ref_sensor,
            to_float_image=True,
        )
        th = _fetch_sensor_patch_chw(
            provider,
            spatial=spatial,
            temporal=temporal,
            sensor=th_sensor,
            to_float_image=True,
        )

        # Normalize proxies to [0,1].
        r01, r02, r03, r04, r06, r07 = [np.clip(ch / 10000.0, 0.0, 1.0) for ch in ref]
        lst = np.clip(
            (th[0] - 200.0) / 130.0, 0.0, 1.0
        )  # LST_1KM in Kelvin (already scaled in MOD21A1D)
        e29 = np.clip((th[1] - 0.8) / 0.2, 0.0, 1.0)  # emissivity ~[0.8,1.0]
        e31 = np.clip((th[2] - 0.8) / 0.2, 0.0, 1.0)
        e32 = np.clip((th[3] - 0.8) / 0.2, 0.0, 1.0)

        # SatVision order: 1,2,3,26,6,20,7,27,28,29,31,32,33,34
        assembled = np.stack(
            [
                r01,  # 1
                r02,  # 2
                r03,  # 3
                r04,  # 26 proxy
                r06,  # 6
                lst,  # 20 proxy
                r07,  # 7
                lst,  # 27 proxy
                lst,  # 28 proxy
                e29,  # 29
                e31,  # 31
                e32,  # 32
                lst,  # 33 proxy
                lst,  # 34 proxy
            ],
            axis=0,
        ).astype(np.float32)
        return assembled, {
            "source_collection": str(sensor.collection),
            "fallback_used": True,
            "fallback_reason": repr(e),
            "fallback_collections": (
                _FALLBACK_MOD09_COLLECTION,
                _FALLBACK_MOD21_COLLECTION,
            ),
            "already_unit_scaled": True,
            "proxy_band_order": tuple(_DEFAULT_MODIS_BANDS),
        }


def _coerce_fetch_result(res: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        arr = np.asarray(res[0], dtype=np.float32)
        meta = dict(res[1])
        return arr, meta
    arr = np.asarray(res, dtype=np.float32)
    return arr, {
        "source_collection": None,
        "fallback_used": False,
        "already_unit_scaled": False,
    }


def _satvision_forward_batch(
    model: Any,
    x_chw_batch: List[np.ndarray],
    *,
    device: str,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    if not x_chw_batch:
        return [], {"tokens_kind": "empty"}

    ensure_torch()
    import torch

    dev = _resolve_device(device)
    xb = torch.from_numpy(
        np.stack([x.astype(np.float32, copy=False) for x in x_chw_batch], axis=0)
    ).to(dev)

    ff = getattr(model, "forward_features", None)
    if not callable(ff):
        raise ModelError("SatVision-TOA model does not expose forward_features().")

    with torch.no_grad():
        out = ff(xb)

    arrs, meta = _decode_features_to_batch_arrays(out, len(x_chw_batch))
    return arrs, meta


@register("satvision")
class SatVisionTOAEmbedder(EmbedderBase):
    """
    SatVision-TOA on-the-fly embedding from generic provider multi-band TOA inputs.

    Notes:
    - This model expects 14 channels in a specific training order.
    - Provide `sensor.bands` in the exact order expected by your checkpoint.
    """

    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 2
    DEFAULT_BATCH_CUDA = 8

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "collection": _DEFAULT_MODIS_COLLECTION,
                "bands": list(_DEFAULT_MODIS_BANDS),
            },
            "output": ["pooled", "grid"],
            "defaults": {
                "model_id": _DEFAULT_MODEL_ID,
                "in_chans": _DEFAULT_IN_CHANS,
                "image_size": _DEFAULT_IMAGE_SIZE,
                "norm": "auto",
                "reflectance_indices": _DEFAULT_REFLECTANCE_INDICES,
                "emissive_indices": _DEFAULT_EMISSIVE_INDICES,
                "reflectance_divisor": 100.0,
                "emissive_mins": _DEFAULT_EMISSIVE_MINS,
                "emissive_maxs": _DEFAULT_EMISSIVE_MAXS,
            },
            "notes": [
                "If sensor is omitted, rs-embed uses MODIS SatVision default band order.",
                "Use RS_EMBED_SATVISION_TOA_CKPT for local checkpoints, or HF token for gated repos.",
                "If MODIS L1B is unavailable in your Google Earth Engine project, set a custom SensorSpec collection.",
            ],
        }

    @staticmethod
    def _default_sensor() -> SensorSpec:
        bands_env = os.environ.get("RS_EMBED_SATVISION_TOA_BANDS")
        if bands_env:
            bands = tuple(b.strip() for b in bands_env.split(",") if b.strip())
        else:
            bands = _DEFAULT_MODIS_BANDS
        return SensorSpec(
            collection=os.environ.get(
                "RS_EMBED_SATVISION_TOA_COLLECTION", _DEFAULT_MODIS_COLLECTION
            ),
            bands=tuple(bands),
            scale_m=int(os.environ.get("RS_EMBED_SATVISION_TOA_SCALE_M", "1000")),
            cloudy_pct=int(os.environ.get("RS_EMBED_SATVISION_TOA_CLOUDY_PCT", "100")),
            fill_value=float(os.environ.get("RS_EMBED_SATVISION_TOA_FILL", "0")),
            composite=os.environ.get("RS_EMBED_SATVISION_TOA_COMPOSITE", "mosaic"),
        )

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_SATVISION_TOA_FETCH_WORKERS",
                str(SatVisionTOAEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default_bs = (
            SatVisionTOAEmbedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else SatVisionTOAEmbedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_SATVISION_TOA_BATCH_SIZE", str(default_bs)))
        return max(1, int(v))

    def _prepare_input(
        self,
        raw_chw: np.ndarray,
        *,
        in_chans: int,
        image_size: int,
        norm_mode: str,
        reflectance_indices: Sequence[int],
        emissive_indices: Sequence[int],
        reflectance_divisor: float,
        emissive_mins: Sequence[float],
        emissive_maxs: Sequence[float],
    ) -> np.ndarray:
        x = np.asarray(raw_chw, dtype=np.float32)
        if x.ndim != 3 or int(x.shape[0]) != int(in_chans):
            raise ModelError(
                f"SatVision-TOA expects input CHW with C={in_chans}, got {getattr(x, 'shape', None)}"
            )
        x = _normalize_satvision_toa_input(
            x,
            mode=norm_mode,
            reflectance_indices=reflectance_indices,
            emissive_indices=emissive_indices,
            reflectance_divisor=reflectance_divisor,
            emissive_mins=emissive_mins,
            emissive_maxs=emissive_maxs,
        )
        x = _resize_chw(x, out_size=int(image_size))
        return x.astype(np.float32)

    def _resolve_runtime(self, *, sensor: SensorSpec, device: str) -> Dict[str, Any]:
        model_id = os.environ.get(
            "RS_EMBED_SATVISION_TOA_ID", _DEFAULT_MODEL_ID
        ).strip()
        local_ckpt = os.environ.get("RS_EMBED_SATVISION_TOA_CKPT")
        auto_download = _as_bool_env("RS_EMBED_SATVISION_TOA_AUTO_DOWNLOAD", True)

        image_size = int(
            os.environ.get("RS_EMBED_SATVISION_TOA_IMG", str(_DEFAULT_IMAGE_SIZE))
        )
        in_chans = int(
            os.environ.get("RS_EMBED_SATVISION_TOA_IN_CHANS", str(_DEFAULT_IN_CHANS))
        )
        if len(sensor.bands) != int(in_chans):
            raise ModelError(
                f"satvision_toa requires exactly {in_chans} sensor bands, got {len(sensor.bands)}. "
                "Set RS_EMBED_SATVISION_TOA_IN_CHANS or adjust SensorSpec.bands."
            )

        norm_mode = (
            os.environ.get("RS_EMBED_SATVISION_TOA_NORM", "auto").strip().lower()
        )
        reflectance_indices = _parse_int_list(
            os.environ.get(
                "RS_EMBED_SATVISION_TOA_REFLECTANCE_IDXS",
                ",".join(str(i) for i in _DEFAULT_REFLECTANCE_INDICES),
            )
        )
        emissive_indices = _parse_int_list(
            os.environ.get(
                "RS_EMBED_SATVISION_TOA_EMISSIVE_IDXS",
                ",".join(str(i) for i in _DEFAULT_EMISSIVE_INDICES),
            )
        )
        reflectance_divisor = float(
            os.environ.get("RS_EMBED_SATVISION_TOA_REF_DIV", "100")
        )
        emissive_mins = _parse_float_list(
            os.environ.get(
                "RS_EMBED_SATVISION_TOA_EMISSIVE_MINS",
                ",".join(str(v) for v in _DEFAULT_EMISSIVE_MINS),
            )
        )
        emissive_maxs = _parse_float_list(
            os.environ.get(
                "RS_EMBED_SATVISION_TOA_EMISSIVE_MAXS",
                ",".join(str(v) for v in _DEFAULT_EMISSIVE_MAXS),
            )
        )

        model, lmeta = _load_satvision_toa(
            model_id=model_id,
            local_ckpt=local_ckpt,
            auto_download=auto_download,
            in_chans=in_chans,
            device=device,
        )
        dev = str(lmeta.get("device", _resolve_device(device)))

        return {
            "model": model,
            "model_meta": lmeta,
            "device": dev,
            "model_id": model_id,
            "image_size": image_size,
            "in_chans": in_chans,
            "norm_mode": norm_mode,
            "reflectance_indices": reflectance_indices,
            "emissive_indices": emissive_indices,
            "reflectance_divisor": reflectance_divisor,
            "emissive_mins": emissive_mins,
            "emissive_maxs": emissive_maxs,
        }

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satvision_toa expects a provider backend (or 'auto').")
        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)
        rt = self._resolve_runtime(sensor=sensor, device=device)

        if input_chw is None:
            raw, fetch_meta = _coerce_fetch_result(
                _fetch_toa_raw_chw_from_gee(
                    self._get_provider(backend),
                    spatial,
                    t,
                    sensor,
                )
            )
        else:
            raw = np.asarray(input_chw, dtype=np.float32)
            fetch_meta = {
                "source_collection": None,
                "fallback_used": False,
                "already_unit_scaled": False,
            }

        norm_mode_eff = str(rt["norm_mode"])
        if bool(fetch_meta.get("already_unit_scaled")):
            norm_mode_eff = "unit"

        x = self._prepare_input(
            raw,
            in_chans=rt["in_chans"],
            image_size=rt["image_size"],
            norm_mode=norm_mode_eff,
            reflectance_indices=rt["reflectance_indices"],
            emissive_indices=rt["emissive_indices"],
            reflectance_divisor=rt["reflectance_divisor"],
            emissive_mins=rt["emissive_mins"],
            emissive_maxs=rt["emissive_maxs"],
        )

        arrs, fmeta = _satvision_forward_batch(rt["model"], [x], device=rt["device"])
        out_arr = arrs[0]

        meta = base_meta(
            model_name=self.model_name,
            hf_id=rt["model_id"],
            backend=str(backend).lower(),
            image_size=int(rt["image_size"]),
            sensor=sensor,
            temporal=t,
            source=sensor.collection,
            extra={
                "in_chans": int(rt["in_chans"]),
                "norm_mode": rt["norm_mode"],
                "norm_mode_effective": norm_mode_eff,
                "reflectance_indices": tuple(
                    int(i)
                    for i in _normalize_indices(
                        rt["reflectance_indices"], rt["in_chans"]
                    )
                ),
                "emissive_indices": tuple(
                    int(i)
                    for i in _normalize_indices(rt["emissive_indices"], rt["in_chans"])
                ),
                "reflectance_divisor": float(rt["reflectance_divisor"]),
                "emissive_mins": tuple(float(v) for v in rt["emissive_mins"]),
                "emissive_maxs": tuple(float(v) for v in rt["emissive_maxs"]),
                "raw_input_shape": tuple(int(v) for v in raw.shape),
                "model_output_shape": tuple(int(v) for v in out_arr.shape),
                **fetch_meta,
                **rt["model_meta"],
                **fmeta,
            },
        )

        if output.mode == "pooled":
            if out_arr.ndim == 2:
                vec, cls_removed = pool_from_tokens(out_arr, output.pooling)
                meta.update(
                    {
                        "pooling": f"patch_{output.pooling}",
                        "cls_removed": bool(cls_removed),
                    }
                )
                return Embedding(data=vec, meta=meta)
            if out_arr.ndim == 1:
                meta.update({"pooling": "model_pooled", "cls_removed": False})
                return Embedding(data=out_arr.astype(np.float32), meta=meta)
            raise ModelError(
                f"Unexpected SatVision output shape for pooled mode: {out_arr.shape}"
            )

        if output.mode == "grid":
            if out_arr.ndim != 2:
                raise ModelError(
                    "grid output requires token sequence [N,D]. "
                    f"Got {out_arr.shape} (tokens_kind={meta.get('tokens_kind')})."
                )
            grid, (h, w), cls_removed = tokens_to_grid_dhw(out_arr)
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
                raise ModelError(
                    "grid output requires xarray. Install: pip install xarray"
                ) from e

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
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satvision_toa expects a provider backend (or 'auto').")
        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)
        rt = self._resolve_runtime(sensor=sensor, device=device)

        n = len(spatials)
        prefetched_raw: List[Optional[np.ndarray]] = [None] * n
        prefetched_meta: List[Optional[Dict[str, Any]]] = [None] * n

        def _fetch_one(
            i: int, sp: SpatialSpec
        ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
            raw, fetch_meta = _coerce_fetch_result(
                _fetch_toa_raw_chw_from_gee(self._get_provider(backend), sp, t, sensor)
            )
            return i, raw, fetch_meta

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                ii, raw, fetch_meta = _fetch_one(i, sp)
                prefetched_raw[ii] = raw
                prefetched_meta[ii] = fetch_meta
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, raw, fetch_meta = fut.result()
                    prefetched_raw[i] = raw
                    prefetched_meta[i] = fetch_meta

        infer_bs = self._resolve_infer_batch(rt["device"])
        out: List[Optional[Embedding]] = [None] * n

        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            raws = prefetched_raw[s0:s1]
            metas = prefetched_meta[s0:s1]
            if any(x is None for x in raws):
                raise ModelError(
                    f"Missing prefetched SatVision input in batch slice [{s0}:{s1}]."
                )
            if any(m is None for m in metas):
                raise ModelError(
                    f"Missing prefetched SatVision metadata in batch slice [{s0}:{s1}]."
                )

            x_batch: List[np.ndarray] = []
            norm_modes_eff: List[str] = []
            for x, m in zip(raws, metas):
                assert x is not None
                assert m is not None
                norm_mode_eff = str(rt["norm_mode"])
                if bool(m.get("already_unit_scaled")):
                    norm_mode_eff = "unit"
                x_batch.append(
                    self._prepare_input(
                        np.asarray(x, dtype=np.float32),
                        in_chans=rt["in_chans"],
                        image_size=rt["image_size"],
                        norm_mode=norm_mode_eff,
                        reflectance_indices=rt["reflectance_indices"],
                        emissive_indices=rt["emissive_indices"],
                        reflectance_divisor=rt["reflectance_divisor"],
                        emissive_mins=rt["emissive_mins"],
                        emissive_maxs=rt["emissive_maxs"],
                    )
                )
                norm_modes_eff.append(norm_mode_eff)

            arrs, fmeta = _satvision_forward_batch(
                rt["model"], x_batch, device=rt["device"]
            )

            for j, arr in enumerate(arrs):
                i = s0 + j
                raw = prefetched_raw[i]
                fetch_meta = prefetched_meta[i]
                assert raw is not None
                assert fetch_meta is not None
                meta = base_meta(
                    model_name=self.model_name,
                    hf_id=rt["model_id"],
                    backend=str(backend).lower(),
                    image_size=int(rt["image_size"]),
                    sensor=sensor,
                    temporal=t,
                    source=sensor.collection,
                    extra={
                        "in_chans": int(rt["in_chans"]),
                        "norm_mode": rt["norm_mode"],
                        "norm_mode_effective": norm_modes_eff[j],
                        "reflectance_indices": tuple(
                            int(k)
                            for k in _normalize_indices(
                                rt["reflectance_indices"], rt["in_chans"]
                            )
                        ),
                        "emissive_indices": tuple(
                            int(k)
                            for k in _normalize_indices(
                                rt["emissive_indices"], rt["in_chans"]
                            )
                        ),
                        "reflectance_divisor": float(rt["reflectance_divisor"]),
                        "emissive_mins": tuple(float(v) for v in rt["emissive_mins"]),
                        "emissive_maxs": tuple(float(v) for v in rt["emissive_maxs"]),
                        "raw_input_shape": tuple(int(v) for v in raw.shape),
                        "model_output_shape": tuple(int(v) for v in arr.shape),
                        **fetch_meta,
                        **rt["model_meta"],
                        **fmeta,
                    },
                )

                if output.mode == "pooled":
                    if arr.ndim == 2:
                        vec, cls_removed = pool_from_tokens(arr, output.pooling)
                        meta.update(
                            {
                                "pooling": f"patch_{output.pooling}",
                                "cls_removed": bool(cls_removed),
                            }
                        )
                        out[i] = Embedding(data=vec, meta=meta)
                    elif arr.ndim == 1:
                        meta.update({"pooling": "model_pooled", "cls_removed": False})
                        out[i] = Embedding(data=arr.astype(np.float32), meta=meta)
                    else:
                        raise ModelError(
                            f"Unexpected SatVision output shape for pooled mode: {arr.shape}"
                        )
                    continue

                if output.mode == "grid":
                    if arr.ndim != 2:
                        raise ModelError(
                            "grid output requires token sequence [N,D]. "
                            f"Got {arr.shape} (tokens_kind={meta.get('tokens_kind')})."
                        )
                    grid, (h, w), cls_removed = tokens_to_grid_dhw(arr)
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
                        raise ModelError(
                            "grid output requires xarray. Install: pip install xarray"
                        ) from e

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
                    out[i] = Embedding(data=da, meta=meta)
                    continue

                raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError(
                "satvision_toa batch inference produced incomplete outputs."
            )
        return [e for e in out if e is not None]

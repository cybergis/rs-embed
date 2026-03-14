from __future__ import annotations

import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..providers import ProviderBase
from ..tools.temporal import temporal_frame_midpoints
from ._vit_mae_utils import ensure_torch
from .base import EmbedderBase
from .meta_utils import build_meta, temporal_midpoint_str, temporal_to_range
from .runtime_utils import (
    coerce_input_to_tchw as _coerce_input_to_tchw,
)
from .runtime_utils import (
    fetch_s2_multiframe_raw_tchw as _fetch_s2_multiframe_raw_tchw,
)
from .runtime_utils import (
    is_provider_backend,
)
from .runtime_utils import (
    load_cached_with_device as _load_cached_with_device,
)
from .runtime_utils import (
    resolve_device_auto_torch as _resolve_device,
)

_S2_10_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

def _resize_tchw(x_tchw: np.ndarray, *, out_hw: int) -> np.ndarray:
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if x_tchw.ndim != 4:
        raise ModelError(f"Expected [T,C,H,W], got {x_tchw.shape}")
    x = torch.from_numpy(x_tchw.astype(np.float32, copy=False))
    y = F.interpolate(x, size=(int(out_hw), int(out_hw)), mode="bilinear", align_corners=False)
    return y.detach().cpu().numpy().astype(np.float32)

def _normalize_s2(raw: np.ndarray, *, mode: str) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim not in {3, 4}:
        raise ModelError(f"Galileo normalization expects CHW or TCHW, got {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 10000.0)

    m = str(mode).lower().strip()
    if m in {"unit", "unit_scale", "reflectance"}:
        x = x / 10000.0
    elif m in {"per_tile_minmax", "minmax", "tile_minmax"}:
        x = x / 10000.0
        if x.ndim == 3:
            lo = np.min(x, axis=(1, 2), keepdims=True)
            hi = np.max(x, axis=(1, 2), keepdims=True)
        else:
            lo = np.min(x, axis=(2, 3), keepdims=True)
            hi = np.max(x, axis=(2, 3), keepdims=True)
        den = np.maximum(hi - lo, 1e-6)
        x = (x - lo) / den
    elif m in {"none", "raw"}:
        pass
    else:
        raise ModelError(
            f"Unknown Galileo normalization mode '{mode}'. "
            "Use one of: unit_scale, per_tile_minmax, none."
        )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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
    raw = _fetch_s2_multiframe_raw_tchw(
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
    return max(1, min(12, int(d.month)))

def _frame_month_sequence(temporal: TemporalSpec, *, n_frames: int) -> np.ndarray:
    mids = temporal_frame_midpoints(temporal, max(1, int(n_frames)))
    if not mids:
        return np.full((max(1, int(n_frames)),), 6, dtype=np.int64)
    return np.array([_month_from_iso(v) for v in mids], dtype=np.int64)

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

    try:
        encoder = encoder.to(dev).eval()
    except Exception as _e:
        pass

    p0 = None
    for _, p in encoder.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError("Galileo encoder has no parameters; cannot verify load.")
    if not torch.isfinite(p0).all():
        raise ModelError("Galileo parameters contain NaN/Inf; load likely failed.")
    p0f = p0.float()

    meta = {
        "model_size": str(model_size),
        "model_root": model_root,
        "model_source": (model_root if model_path else f"hf://{hf_repo}/models/{model_size}"),
        "device": str(dev),
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
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
    include_ndvi: bool,
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

    x_tchw = raw_tchw.astype(np.float32, copy=False)
    if x_tchw.shape[-2] != image_size or x_tchw.shape[-1] != image_size:
        x_tchw = _resize_tchw(x_tchw, out_hw=image_size)
    x_tchw = _normalize_s2(x_tchw, mode=norm_mode)  # [T,10,H,W]

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

    ndvi_set = False
    if include_ndvi and ("NDVI" in space_time_bands):
        try:
            b8_idx = s2_bands.index("B8")
            b4_idx = s2_bands.index("B4")
            nir = s2_hwtd[:, :, :, b8_idx]
            red = s2_hwtd[:, :, :, b4_idx]
            ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
            s_t_x[0, :, :, :, space_time_bands.index("NDVI")] = ndvi.astype(np.float32)
            ndvi_set = True
        except Exception as _e:
            ndvi_set = False

    # masks: 0 means seen by encoder, 1 means masked/ignored
    s_t_m = np.ones((1, h, w, t, len(s_t_groups)), dtype=np.float32)
    s2_group_indices = [i for i, key in enumerate(s_t_groups) if "S2" in str(key)]
    for idx in s2_group_indices:
        s_t_m[0, :, :, :, idx] = 0.0

    if ndvi_set:
        for i, key in enumerate(s_t_groups):
            if str(key) == "NDVI":
                s_t_m[0, :, :, :, i] = 0.0

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
        months_arr = np.full((t,), 6, dtype=np.int64)
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
    months_arr = np.clip(months_arr, 1, 12).astype(np.int64)
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
    meta = {
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "n_frames": int(t),
        "months": tuple(int(v) for v in months_arr.tolist()),
        "month": int(months_arr[len(months_arr) // 2]),
        "normalization": str(norm_mode),
        "include_ndvi": bool(include_ndvi),
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
    try:
        encoder = encoder.to(dev).eval()
    except Exception as _e:
        pass

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

    # grid features from S2-related space-time groups only
    s_t_groups = list(mod.SPACE_TIME_BANDS_GROUPS_IDX.keys())
    s2_group_indices = [i for i, key in enumerate(s_t_groups) if "S2" in str(key)]
    if not s2_group_indices:
        raise ModelError("Failed to locate Galileo S2 group indices in SPACE_TIME_BANDS_GROUPS_IDX")

    # s_t_x shape: [B,H,W,T,Cg,D]
    s_t_sel = s_t_x[:, :, :, :, s2_group_indices, :]
    # average over time and channel-groups -> [B,H,W,D]
    grid_hwd = s_t_sel.mean(dim=3).mean(dim=3)[0]
    grid = grid_hwd.detach().float().cpu().numpy().transpose(2, 0, 1).astype(np.float32)  # [D,H,W]

    fmeta = {
        "feature_dim": int(vec.shape[0]),
        "grid_shape": tuple(grid.shape),
        "grid_hw": (int(grid.shape[1]), int(grid.shape[2])),
        "grid_kind": "s2_group_patch_tokens",
        "s2_group_indices": tuple(int(i) for i in s2_group_indices),
    }
    return vec, grid, fmeta

@register("galileo")
class GalileoEmbedder(EmbedderBase):
    DEFAULT_MODEL_SIZE = "nano"
    DEFAULT_PATCH = 8
    DEFAULT_IMAGE_SIZE = 64
    DEFAULT_FRAMES = 8
    DEFAULT_FETCH_WORKERS = 8
    _allow_auto_backend = False

    def describe(self) -> dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": _S2_10_BANDS,
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "model_size": self.DEFAULT_MODEL_SIZE,
                "patch_size": self.DEFAULT_PATCH,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "n_frames": self.DEFAULT_FRAMES,
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "normalization": "unit_scale",
            },
            "notes": [
                "Loads Galileo Encoder from a vendored local runtime.",
                "Defaults to Hugging Face model snapshots under nasaharvest/galileo/models/<size>/.",
                "Builds T-frame S2 series by splitting TemporalSpec.range into equal sub-windows.",
                "Uses Sentinel-2 10 bands; pooled output averages visible Galileo tokens.",
            ],
        }

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return SensorSpec(
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=tuple(_S2_10_BANDS),
            scale_m=10,
            cloudy_pct=30,
            composite="median",
            fill_value=0.0,
        )

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_GALILEO_FETCH_WORKERS",
                str(GalileoEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

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
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=False):
            raise ModelError("galileo expects a provider backend.")

        ss = sensor or self._default_sensor()
        t = temporal_to_range(temporal)

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
        n_frames = max(1, int(os.environ.get("RS_EMBED_GALILEO_FRAMES", str(self.DEFAULT_FRAMES))))
        norm_mode = os.environ.get("RS_EMBED_GALILEO_NORM", "unit_scale").strip()
        add_layernorm = os.environ.get("RS_EMBED_GALILEO_ADD_LN", "1").strip() not in {
            "0",
            "false",
            "False",
        }
        include_ndvi = os.environ.get("RS_EMBED_GALILEO_INCLUDE_NDVI", "1").strip() not in {
            "0",
            "false",
            "False",
        }
        month_override = os.environ.get("RS_EMBED_GALILEO_MONTH")

        if input_chw is None:
            provider = self._get_provider(backend)
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
            raw_tchw = _coerce_input_to_tchw(
                input_chw,
                expected_channels=10,
                n_frames=n_frames,
                model_name="galileo",
            )

        if month_override is not None:
            month_value = max(1, min(12, int(month_override)))
            months_seq = np.full((int(raw_tchw.shape[0]),), month_value, dtype=np.int64)
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
            include_ndvi=include_ndvi,
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
            input_time=temporal_midpoint_str(t),
            extra={
                "start": t.start,
                "end": t.end,
                "patch_size": int(patch_size),
                "n_frames": int(raw_tchw.shape[0]),
                "normalization": str(norm_mode),
                "include_ndvi": bool(include_ndvi),
                "device": dev,
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
            else:
                vec_out = vec.astype(np.float32)
                pooling = "token_mean"
            ometa = {**meta, "pooling": pooling, "pooled_shape": tuple(vec_out.shape)}
            return Embedding(data=vec_out, meta=ometa)

        if output.mode == "grid":
            gmeta = {
                **meta,
                "grid_kind": "s2_group_patch_tokens",
                "grid_shape": tuple(grid.shape),
                "grid_hw": (int(grid.shape[1]), int(grid.shape[2])),
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
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []

        if not is_provider_backend(backend, allow_auto=False):
            raise ModelError("galileo expects a provider backend.")

        t = temporal_to_range(temporal)
        ss = sensor or self._default_sensor()
        provider = self._get_provider(backend)
        n_frames = max(1, int(os.environ.get("RS_EMBED_GALILEO_FRAMES", str(self.DEFAULT_FRAMES))))

        n = len(spatials)
        prefetched_raw: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            raw = _fetch_s2_10_raw_tchw(
                provider,
                sp,
                t,
                n_frames=n_frames,
                scale_m=int(ss.scale_m),
                cloudy_pct=int(ss.cloudy_pct),
                composite=str(ss.composite),
                fill_value=float(ss.fill_value),
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
                raise ModelError(f"Missing prefetched input at index={i} for galileo.")
            out.append(
                self.get_embedding(
                    spatial=sp,
                    temporal=t,
                    sensor=ss,
                    output=output,
                    backend=backend,
                    device=device,
                    input_chw=raw,
                )
            )
        return out

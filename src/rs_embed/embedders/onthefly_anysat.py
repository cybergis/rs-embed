from __future__ import annotations

import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from functools import lru_cache
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

def _normalize_series(x_tchw: np.ndarray, *, mode: str) -> np.ndarray:
    mode_l = str(mode).lower().strip()
    x = x_tchw.astype(np.float32, copy=False)
    if mode_l in ("none", "off", "raw"):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if mode_l in ("unit", "unit_scale", "reflectance"):
        x = np.clip(x / 10000.0, 0.0, 1.0)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if mode_l in ("per_tile_zscore", "zscore", "tile_zscore"):
        mu = np.nanmean(x, axis=(0, 2, 3), keepdims=True)
        sigma = np.nanstd(x, axis=(0, 2, 3), keepdims=True)
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
        sigma = np.maximum(sigma, 1e-6)
        x = (x - mu) / sigma
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ModelError(
        f"Unknown AnySat normalization mode '{mode}'. "
        "Use one of: none, unit_scale, per_tile_zscore."
    )

def _doy0_from_iso(iso_date: str) -> int:
    d = date.fromisoformat(str(iso_date))
    # AnySat docs: 01/01 -> 0 ; 31/12 -> 364
    doy0 = int(d.timetuple().tm_yday) - 1
    return max(0, min(364, doy0))

def _frame_doy0_sequence(temporal: TemporalSpec, *, n_frames: int) -> np.ndarray:
    mids = temporal_frame_midpoints(temporal, max(1, int(n_frames)))
    if not mids:
        return np.full((max(1, int(n_frames)),), 182, dtype=np.int64)
    return np.array([_doy0_from_iso(v) for v in mids], dtype=np.int64)

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

@lru_cache(maxsize=8)
def _load_anysat_hub_module():
    try:
        mod = importlib.import_module("rs_embed.embedders._vendor.anysat.hubconf")
        _ = mod.AnySat
    except Exception as e:
        raise ModelError(
            "Failed to import vendored AnySat runtime. Install missing dependencies: torch, einops."
        ) from e
    return mod

@lru_cache(maxsize=4)
def _download_anysat_ckpt(
    *,
    hf_repo: str,
    filename: str,
    cache_dir: str | None,
    min_bytes: int,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise ModelError(
            "AnySat checkpoint download requires huggingface_hub. "
            "Install: pip install huggingface_hub"
        ) from e

    p = hf_hub_download(repo_id=hf_repo, filename=filename, cache_dir=cache_dir)
    if not os.path.exists(p):
        raise ModelError(f"Failed to download AnySat checkpoint: {hf_repo}/{filename}")
    sz = os.path.getsize(p)
    if sz < int(min_bytes):
        raise ModelError(f"Downloaded AnySat checkpoint looks too small ({sz} bytes): {p}")
    return p

def _load_ckpt_state_dict(ckpt_path: str) -> dict[str, Any]:
    ensure_torch()
    import torch

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ModelError(f"Unsupported checkpoint format at {ckpt_path}")

@lru_cache(maxsize=6)
def _load_anysat_cached(
    *,
    model_size: str,
    flash_attn: bool,
    pretrained: bool,
    ckpt_path: str | None,
    hf_repo: str,
    hf_filename: str,
    hf_cache_dir: str | None,
    hf_min_bytes: int,
    dev: str,
) -> tuple[Any, dict[str, Any]]:
    ensure_torch()
    import torch

    hub = _load_anysat_hub_module()
    if not hasattr(hub, "AnySat"):
        raise ModelError("Vendored AnySat runtime does not expose class AnySat.")

    if ckpt_path:
        ckpt_local = os.path.expanduser(ckpt_path)
        if not os.path.exists(ckpt_local):
            raise ModelError(f"AnySat checkpoint not found: {ckpt_local}")
        if os.path.getsize(ckpt_local) < 50 * 1024 * 1024:
            raise ModelError(f"AnySat checkpoint seems too small: {ckpt_local}")
        model = hub.AnySat(model_size=model_size, flash_attn=bool(flash_attn), device=dev)
        sd = _load_ckpt_state_dict(ckpt_local)
        model.model.load_state_dict(sd, strict=True)
        loaded_from = ckpt_local
    else:
        if pretrained:
            ckpt_local = _download_anysat_ckpt(
                hf_repo=hf_repo,
                filename=hf_filename,
                cache_dir=hf_cache_dir,
                min_bytes=hf_min_bytes,
            )
            model = hub.AnySat(model_size=model_size, flash_attn=bool(flash_attn), device=dev)
            sd = _load_ckpt_state_dict(ckpt_local)
            model.model.load_state_dict(sd, strict=True)
            loaded_from = f"hf://{hf_repo}/{hf_filename}"
        else:
            model = hub.AnySat(model_size=model_size, flash_attn=bool(flash_attn), device=dev)
            loaded_from = "random_init"

    try:
        model = model.to(dev).eval()
    except Exception as _e:
        pass

    p0 = None
    for _, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError("AnySat model has no parameters; cannot verify load.")
    if not torch.isfinite(p0).all():
        raise ModelError("AnySat parameters contain NaN/Inf; checkpoint load likely failed.")
    p0f = p0.float()

    meta = {
        "model_size": str(model_size),
        "flash_attn": bool(flash_attn),
        "pretrained": bool(pretrained),
        "loaded_from": loaded_from,
        "model_source": "vendored_rs_embed_runtime",
        "device": dev,
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
    }
    return model, meta

def _load_anysat(
    *,
    model_size: str,
    flash_attn: bool,
    pretrained: bool,
    ckpt_path: str | None,
    hf_repo: str,
    hf_filename: str,
    hf_cache_dir: str | None,
    hf_min_bytes: int,
    device: str,
) -> tuple[Any, dict[str, Any], str]:
    (loaded, dev) = _load_cached_with_device(
        _load_anysat_cached,
        device=device,
        model_size=str(model_size),
        flash_attn=bool(flash_attn),
        pretrained=bool(pretrained),
        ckpt_path=(os.path.expanduser(ckpt_path) if ckpt_path else None),
        hf_repo=str(hf_repo),
        hf_filename=str(hf_filename),
        hf_cache_dir=(os.path.expanduser(hf_cache_dir) if hf_cache_dir else None),
        hf_min_bytes=int(hf_min_bytes),
    )
    model, meta = loaded
    return model, meta, dev

def _prepare_anysat_s2_input(
    raw_tchw: np.ndarray,
    *,
    image_size: int,
    doy0_values: np.ndarray,
    norm_mode: str,
    device: str,
) -> dict[str, Any]:
    ensure_torch()
    import torch

    if raw_tchw.ndim != 4 or int(raw_tchw.shape[1]) != 10:
        raise ModelError(f"AnySat s2 expects [T,10,H,W], got shape={raw_tchw.shape}")
    x_tchw = raw_tchw.astype(np.float32, copy=False)

    if image_size > 0 and (x_tchw.shape[-1] != image_size or x_tchw.shape[-2] != image_size):
        x_tchw = _resize_tchw(x_tchw, out_hw=image_size)

    x_tchw = _normalize_series(x_tchw, mode=norm_mode)
    t = int(x_tchw.shape[0])
    doy = np.asarray(doy0_values, dtype=np.int64).reshape(-1)
    if doy.size == 0:
        doy = np.full((t,), 182, dtype=np.int64)
    if doy.size < t:
        doy = np.concatenate([doy, np.full((t - doy.size,), int(doy[-1]), dtype=np.int64)], axis=0)
    elif doy.size > t:
        doy = doy[:t]
    dates = doy[None, :]

    return {
        "s2": torch.from_numpy(x_tchw[None, ...]).to(device),  # [1,T,10,H,W]
        "s2_dates": torch.from_numpy(dates).to(device),  # [1,T]
    }

def _anysat_patch_features(
    model: Any,
    s2_input: dict[str, Any],
    *,
    patch_size_m: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_torch()
    import torch

    if patch_size_m <= 0 or (patch_size_m % 10) != 0:
        raise ModelError(
            f"AnySat patch_size must be a positive multiple of 10 (meters), got {patch_size_m}"
        )

    with torch.no_grad():
        out = model(s2_input, patch_size=int(patch_size_m), output="patch")

    if not hasattr(out, "ndim") or int(out.ndim) != 4:
        raise ModelError(
            f"AnySat output='patch' returned unexpected shape/type: {type(out)} {getattr(out, 'shape', None)}"
        )

    # AnySat patch output: [B,H,W,D]
    if int(out.shape[0]) != 1:
        raise ModelError(f"AnySat embedder expects B=1 per call, got {tuple(out.shape)}")
    arr = out[0].detach().float().cpu().numpy().astype(np.float32)  # [H,W,D]
    grid = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # [D,H,W]
    meta = {
        "patch_output_hw": (int(arr.shape[0]), int(arr.shape[1])),
        "feature_dim": int(arr.shape[2]),
        "patch_size_m": int(patch_size_m),
    }
    return grid, meta

@register("anysat")
class AnySatEmbedder(EmbedderBase):
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_FRAMES = 8
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
                "model_size": "base",
                "patch_size_m": 10,
                "image_size": 24,
                "n_frames": self.DEFAULT_FRAMES,
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "normalization": "per_tile_zscore",
            },
            "notes": [
                "AnySat expects S2 time-series + day-of-year dates.",
                "This adapter builds T frames by splitting TemporalSpec.range into equal sub-windows.",
                "Loads AnySat from a vendored local runtime and optional Hugging Face checkpoint.",
                "grid output maps AnySat output='patch' to [D,H,W].",
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
                "RS_EMBED_ANYSAT_FETCH_WORKERS",
                str(AnySatEmbedder.DEFAULT_FETCH_WORKERS),
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
            raise ModelError("anysat expects a provider backend.")

        ss = sensor or self._default_sensor()
        t = temporal_to_range(temporal)
        n_frames = max(1, int(os.environ.get("RS_EMBED_ANYSAT_FRAMES", str(self.DEFAULT_FRAMES))))

        model_size = os.environ.get("RS_EMBED_ANYSAT_MODEL_SIZE", "base").strip().lower()
        flash_attn = os.environ.get("RS_EMBED_ANYSAT_FLASH_ATTN", "0").strip() in {
            "1",
            "true",
            "True",
        }
        image_size = int(os.environ.get("RS_EMBED_ANYSAT_IMG", "24"))
        norm_mode = os.environ.get("RS_EMBED_ANYSAT_NORM", "per_tile_zscore").strip()
        patch_size_m = int(getattr(output, "scale_m", 10))

        ckpt_path = os.environ.get("RS_EMBED_ANYSAT_CKPT")
        pretrained = os.environ.get("RS_EMBED_ANYSAT_PRETRAINED", "1").strip() not in {
            "0",
            "false",
            "False",
        }
        hf_repo = os.environ.get("RS_EMBED_ANYSAT_HF_REPO", "g-astruc/AnySat").strip()
        hf_filename = os.environ.get("RS_EMBED_ANYSAT_HF_FILE", "models/AnySat.pth").strip()
        hf_cache_dir = os.environ.get(
            "RS_EMBED_ANYSAT_CACHE_DIR",
            os.path.join("~", ".cache", "rs_embed", "anysat"),
        )
        hf_min_bytes = int(os.environ.get("RS_EMBED_ANYSAT_CKPT_MIN_BYTES", str(50 * 1024 * 1024)))

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
                model_name="anysat",
            )

        doy0_values = _frame_doy0_sequence(t, n_frames=int(raw_tchw.shape[0]))

        model, lmeta, dev = _load_anysat(
            model_size=model_size,
            flash_attn=flash_attn,
            pretrained=pretrained,
            ckpt_path=ckpt_path,
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            hf_cache_dir=hf_cache_dir,
            hf_min_bytes=hf_min_bytes,
            device=device,
        )

        s2_input = _prepare_anysat_s2_input(
            raw_tchw,
            image_size=image_size,
            doy0_values=doy0_values,
            norm_mode=norm_mode,
            device=dev,
        )
        grid, fmeta = _anysat_patch_features(
            model,
            s2_input,
            patch_size_m=patch_size_m,
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
                "model_size": model_size,
                "flash_attn": bool(flash_attn),
                "normalization": norm_mode,
                "start": t.start,
                "end": t.end,
                "n_frames": int(raw_tchw.shape[0]),
                "doy0_values": tuple(int(v) for v in doy0_values.tolist()),
                "device": dev,
                **lmeta,
                **fmeta,
            },
        )

        if output.mode == "pooled":
            if output.pooling == "max":
                vec = np.max(grid, axis=(1, 2)).astype(np.float32)
            else:
                vec = np.mean(grid, axis=(1, 2)).astype(np.float32)
            ometa = {
                **meta,
                "pooling": f"patch_{output.pooling}",
                "pooled_shape": tuple(vec.shape),
            }
            return Embedding(data=vec, meta=ometa)

        if output.mode == "grid":
            gmeta = {
                **meta,
                "grid_kind": "patch_tokens",
                "grid_hw": (int(grid.shape[1]), int(grid.shape[2])),
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
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []

        if not is_provider_backend(backend, allow_auto=False):
            raise ModelError("anysat expects a provider backend.")

        t = temporal_to_range(temporal)
        ss = sensor or self._default_sensor()
        provider = self._get_provider(backend)
        n_frames = max(1, int(os.environ.get("RS_EMBED_ANYSAT_FRAMES", str(self.DEFAULT_FRAMES))))

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
                raise ModelError(f"Missing prefetched input at index={i} for anysat.")
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

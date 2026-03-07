from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.temporal_utils import temporal_frame_midpoints
from ..providers import ProviderBase
from .base import EmbedderBase
from .runtime_utils import (
    coerce_input_to_tchw as _coerce_input_to_tchw,
    fetch_s2_multiframe_raw_tchw as _fetch_s2_multiframe_raw_tchw,
    get_cached_provider,
    is_provider_backend,
    load_cached_with_device as _load_cached_with_device,
)
from .meta_utils import build_meta, temporal_midpoint_str, temporal_to_range
from ._vit_mae_utils import ensure_torch


_S2_10_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def _resize_tchw(x_tchw: np.ndarray, *, out_hw: int) -> np.ndarray:
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if x_tchw.ndim != 4:
        raise ModelError(f"Expected [T,C,H,W], got {x_tchw.shape}")
    x = torch.from_numpy(x_tchw.astype(np.float32, copy=False))
    y = F.interpolate(
        x, size=(int(out_hw), int(out_hw)), mode="bilinear", align_corners=False
    )
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


@lru_cache(maxsize=4)
def _ensure_anysat_repo(
    *,
    repo_url: str,
    cache_root: str,
) -> str:
    root = os.path.expanduser(cache_root)
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, "AnySat")

    if os.path.isdir(os.path.join(dst, "src")) and os.path.isfile(
        os.path.join(dst, "hubconf.py")
    ):
        return dst

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, dst],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        raise ModelError(
            "Failed to clone AnySat source code. "
            f"Tried: git clone --depth 1 {repo_url} {dst}"
        ) from e
    return dst


@lru_cache(maxsize=8)
def _load_anysat_hub_module(repo_root: str):
    hub_path = os.path.join(repo_root, "hubconf.py")
    if not os.path.exists(hub_path):
        raise ModelError(f"AnySat hubconf not found: {hub_path}")

    repo_abs = os.path.abspath(repo_root)
    if repo_abs not in sys.path:
        # AnySat hubconf imports use `from src...`; they require repo root on sys.path.
        sys.path.insert(0, repo_abs)

    spec = importlib.util.spec_from_file_location("anysat_hubconf", hub_path)
    if spec is None or spec.loader is None:
        raise ModelError("Failed to build import spec for AnySat hubconf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _resolve_anysat_repo(
    *,
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    auto_download: bool,
) -> str:
    if repo_path:
        p = os.path.expanduser(repo_path)
        if not os.path.isdir(p):
            raise ModelError(f"RS_EMBED_ANYSAT_REPO_PATH does not exist: {p}")
        return p
    if not auto_download:
        raise ModelError(
            "AnySat repository not provided. Set RS_EMBED_ANYSAT_REPO_PATH or enable auto download."
        )
    return _ensure_anysat_repo(repo_url=repo_url, cache_root=repo_cache_root)


def _load_ckpt_state_dict(ckpt_path: str) -> Dict[str, Any]:
    ensure_torch()
    import torch

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
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
    ckpt_path: Optional[str],
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    auto_download_repo: bool,
    dev: str,
) -> Tuple[Any, Dict[str, Any]]:
    ensure_torch()
    import torch

    repo_root = _resolve_anysat_repo(
        repo_path=repo_path,
        repo_url=repo_url,
        repo_cache_root=repo_cache_root,
        auto_download=auto_download_repo,
    )
    hub = _load_anysat_hub_module(repo_root)
    if not hasattr(hub, "AnySat"):
        raise ModelError("AnySat hubconf.py does not expose class AnySat.")

    if ckpt_path:
        ckpt_local = os.path.expanduser(ckpt_path)
        if not os.path.exists(ckpt_local):
            raise ModelError(f"AnySat checkpoint not found: {ckpt_local}")
        if os.path.getsize(ckpt_local) < 50 * 1024 * 1024:
            raise ModelError(f"AnySat checkpoint seems too small: {ckpt_local}")
        model = hub.AnySat(
            model_size=model_size, flash_attn=bool(flash_attn), device=dev
        )
        sd = _load_ckpt_state_dict(ckpt_local)
        model.model.load_state_dict(sd, strict=True)
        loaded_from = ckpt_local
    else:
        if pretrained:
            # AnySat hubconf handles download from HF.
            model = hub.AnySat.from_pretrained(
                model_size=model_size, flash_attn=bool(flash_attn), device=dev
            )
            loaded_from = "hf://g-astruc/AnySat/models/AnySat.pth"
        else:
            model = hub.AnySat(
                model_size=model_size, flash_attn=bool(flash_attn), device=dev
            )
            loaded_from = "random_init"

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
        raise ModelError("AnySat model has no parameters; cannot verify load.")
    if not torch.isfinite(p0).all():
        raise ModelError(
            "AnySat parameters contain NaN/Inf; checkpoint load likely failed."
        )
    p0f = p0.float()

    meta = {
        "model_size": str(model_size),
        "flash_attn": bool(flash_attn),
        "pretrained": bool(pretrained),
        "loaded_from": loaded_from,
        "repo_root": repo_root,
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
    ckpt_path: Optional[str],
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    auto_download_repo: bool,
    device: str,
) -> Tuple[Any, Dict[str, Any], str]:
    (loaded, dev) = _load_cached_with_device(
        _load_anysat_cached,
        device=device,
        model_size=str(model_size),
        flash_attn=bool(flash_attn),
        pretrained=bool(pretrained),
        ckpt_path=(os.path.expanduser(ckpt_path) if ckpt_path else None),
        repo_path=(os.path.expanduser(repo_path) if repo_path else None),
        repo_url=str(repo_url),
        repo_cache_root=str(repo_cache_root),
        auto_download_repo=bool(auto_download_repo),
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
) -> Dict[str, Any]:
    ensure_torch()
    import torch

    if raw_tchw.ndim != 4 or int(raw_tchw.shape[1]) != 10:
        raise ModelError(f"AnySat s2 expects [T,10,H,W], got shape={raw_tchw.shape}")
    x_tchw = raw_tchw.astype(np.float32, copy=False)

    if image_size > 0 and (
        x_tchw.shape[-1] != image_size or x_tchw.shape[-2] != image_size
    ):
        x_tchw = _resize_tchw(x_tchw, out_hw=image_size)

    x_tchw = _normalize_series(x_tchw, mode=norm_mode)
    t = int(x_tchw.shape[0])
    doy = np.asarray(doy0_values, dtype=np.int64).reshape(-1)
    if doy.size == 0:
        doy = np.full((t,), 182, dtype=np.int64)
    if doy.size < t:
        doy = np.concatenate(
            [doy, np.full((t - doy.size,), int(doy[-1]), dtype=np.int64)], axis=0
        )
    elif doy.size > t:
        doy = doy[:t]
    dates = doy[None, :]

    return {
        "s2": torch.from_numpy(x_tchw[None, ...]).to(device),  # [1,T,10,H,W]
        "s2_dates": torch.from_numpy(dates).to(device),  # [1,T]
    }


def _anysat_patch_features(
    model: Any,
    s2_input: Dict[str, Any],
    *,
    patch_size_m: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        raise ModelError(
            f"AnySat embedder expects B=1 per call, got {tuple(out.shape)}"
        )
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

    def describe(self) -> Dict[str, Any]:
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
                "grid output maps AnySat output='patch' to [D,H,W].",
            ],
        }

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderBase] = {}

    def _get_provider(self, backend: str) -> ProviderBase:
        return get_cached_provider(
            self._providers,
            backend=backend,
            allow_auto=False,
        )

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
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=False):
            raise ModelError("anysat expects a provider backend.")

        ss = sensor or self._default_sensor()
        t = temporal_to_range(temporal)
        n_frames = max(
            1, int(os.environ.get("RS_EMBED_ANYSAT_FRAMES", str(self.DEFAULT_FRAMES)))
        )

        model_size = (
            os.environ.get("RS_EMBED_ANYSAT_MODEL_SIZE", "base").strip().lower()
        )
        flash_attn = os.environ.get("RS_EMBED_ANYSAT_FLASH_ATTN", "0").strip() in {
            "1",
            "true",
            "True",
        }
        image_size = int(os.environ.get("RS_EMBED_ANYSAT_IMG", "24"))
        norm_mode = os.environ.get("RS_EMBED_ANYSAT_NORM", "per_tile_zscore").strip()
        patch_size_m = int(getattr(output, "scale_m", 10))

        repo_path = os.environ.get("RS_EMBED_ANYSAT_REPO_PATH")
        repo_url = os.environ.get(
            "RS_EMBED_ANYSAT_REPO_URL", "https://github.com/gastruc/AnySat.git"
        )
        repo_cache = os.environ.get(
            "RS_EMBED_ANYSAT_REPO_CACHE",
            os.path.join("~", ".cache", "rs_embed", "anysat"),
        )
        auto_download_repo = os.environ.get(
            "RS_EMBED_ANYSAT_AUTO_DOWNLOAD_REPO", "1"
        ).strip() not in {"0", "false", "False"}
        ckpt_path = os.environ.get("RS_EMBED_ANYSAT_CKPT")
        pretrained = os.environ.get("RS_EMBED_ANYSAT_PRETRAINED", "1").strip() not in {
            "0",
            "false",
            "False",
        }

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
            repo_path=repo_path,
            repo_url=repo_url,
            repo_cache_root=repo_cache,
            auto_download_repo=auto_download_repo,
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
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
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
        n_frames = max(
            1, int(os.environ.get("RS_EMBED_ANYSAT_FRAMES", str(self.DEFAULT_FRAMES)))
        )

        n = len(spatials)
        prefetched_raw: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
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

        out: List[Embedding] = []
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

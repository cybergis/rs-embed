# Implementation based on:
# FoMo-Bench: a multi-modal, multi-scale and multi-task benchmark for Forest Monitoring
#             Remote Sensing Foundation Models
# AAAI 2025
# https://arxiv.org/abs/2312.10114

from __future__ import annotations

import importlib.util
import os
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..providers import ProviderBase
from ._vit_mae_utils import ensure_torch
from .base import EmbedderBase
from .runtime_utils import (
    is_provider_backend,
    load_cached_with_device as _load_cached_with_device,
    resolve_device_auto_torch as _resolve_device,
)
from .meta_utils import build_meta, temporal_midpoint_str, temporal_to_range
from .onthefly_terramind import _fetch_s2_sr_12_raw_chw


_S2_SR_12_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
]

_DEFAULT_MODALITY_CHANNELS: Dict[int, str] = {
    0: "planet-r",
    1: "planet-g",
    2: "planet-b",
    3: "planet-nir",
    4: "sentinel-1-vv",
    5: "sentinel-1-vh",
    6: "sentinel-2-b1",
    7: "sentinel-2-b2",
    8: "sentinel-2-b3",
    9: "sentinel-2-b4",
    10: "sentinel-2-b5",
    11: "sentinel-2-b6",
    12: "sentinel-2-b7",
    13: "sentinel-2-b8",
    14: "sentinel-2-b8a",
    15: "sentinel-2-b9",
    16: "sentinel-2-b10",
    17: "sentinel-2-b11",
    18: "sentinel-2-b12",
    19: "landsat-r",
    20: "landsat-g",
    21: "landsat-b",
    22: "landsat-nir",
    23: "landsat-swir-1",
    24: "landsat-swir-2",
    25: "landsat-panchromatic",
    26: "landsat-aerosol",
    27: "landsat-cirrus",
    28: "aerial-r",
    29: "aerial-g",
    30: "aerial-b",
    31: "aerial-nir",
    32: "dem",
    33: "gaofen2-r",
    34: "gaofen2-g",
    35: "gaofen2-b",
    36: "gaofen2-nir",
}

_DEFAULT_S2_MODALITY_KEYS: Tuple[int, ...] = (
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    17,
    18,
)
_DEFAULT_CKPT_FILENAME = "fomo_single_embedding_layer_weights.pt"
_DEFAULT_CKPT_URL = (
    "https://www.dropbox.com/scl/fi/4ckmxlcbc0tcod8hknp7c/"
    "fomo_single_embedding_layer_weights.pt?rlkey=26tlf3yaz93vvcosr0qrvklub&dl=1"
)
_DEFAULT_REPO_URL = "https://github.com/RolnickLab/FoMo-Bench.git"
_DEFAULT_REPO_CACHE = "~/.cache/rs_embed/fomo"


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() not in {"0", "false", "no", "off", ""}


def _download_url_to_path(url: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=180) as r, open(tmp_path, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp_path, dst_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
    return dst_path


@lru_cache(maxsize=4)
def _download_fomo_ckpt(
    *,
    url: str,
    cache_dir: str,
    filename: str,
    min_bytes: int,
) -> str:
    root = os.path.expanduser(cache_dir)
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, filename)
    if os.path.exists(dst) and os.path.getsize(dst) >= int(min_bytes):
        return dst

    _download_url_to_path(url, dst)
    sz = os.path.getsize(dst) if os.path.exists(dst) else 0
    if sz < int(min_bytes):
        raise ModelError(
            f"Downloaded FoMo checkpoint is too small ({sz} bytes): {dst}. "
            "Set RS_EMBED_FOMO_CKPT to a valid local checkpoint path."
        )
    return dst


def _resolve_fomo_ckpt_path() -> str:
    local = str(os.environ.get("RS_EMBED_FOMO_CKPT") or "").strip()
    if local:
        p = os.path.expanduser(local)
        if not os.path.exists(p):
            raise ModelError(f"RS_EMBED_FOMO_CKPT does not exist: {p}")
        return p

    if not _env_flag("RS_EMBED_FOMO_AUTO_DOWNLOAD", True):
        raise ModelError(
            "FoMo checkpoint is required. Set RS_EMBED_FOMO_CKPT to a local path "
            "or enable RS_EMBED_FOMO_AUTO_DOWNLOAD=1."
        )

    cache_dir = os.environ.get("RS_EMBED_FOMO_CACHE_DIR", _DEFAULT_REPO_CACHE)
    filename = (
        os.environ.get("RS_EMBED_FOMO_CKPT_FILE", _DEFAULT_CKPT_FILENAME).strip()
        or _DEFAULT_CKPT_FILENAME
    )
    url = (
        os.environ.get("RS_EMBED_FOMO_CKPT_URL", _DEFAULT_CKPT_URL).strip()
        or _DEFAULT_CKPT_URL
    )
    min_bytes = int(
        os.environ.get("RS_EMBED_FOMO_CKPT_MIN_BYTES", str(50 * 1024 * 1024))
    )
    return _download_fomo_ckpt(
        url=url, cache_dir=cache_dir, filename=filename, min_bytes=min_bytes
    )


@lru_cache(maxsize=4)
def _ensure_fomo_repo(*, repo_url: str, cache_root: str) -> str:
    root = os.path.expanduser(cache_root)
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, "FoMo-Bench")
    mm = os.path.join(dst, "model_zoo", "multimodal_mae.py")
    if os.path.isfile(mm):
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
            "Failed to clone FoMo-Bench source code. "
            f"Tried: git clone --depth 1 {repo_url} {dst}"
        ) from e
    return dst


def _resolve_fomo_repo(
    *,
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    auto_download: bool,
) -> str:
    if repo_path:
        p = os.path.expanduser(repo_path)
        mm = os.path.join(p, "model_zoo", "multimodal_mae.py")
        if not os.path.isdir(p):
            raise ModelError(f"RS_EMBED_FOMO_REPO_PATH does not exist: {p}")
        if not os.path.isfile(mm):
            raise ModelError(
                f"FoMo repo path is missing model_zoo/multimodal_mae.py: {p}"
            )
        return p
    if not auto_download:
        raise ModelError(
            "FoMo-Bench repository is required. Set RS_EMBED_FOMO_REPO_PATH "
            "or enable RS_EMBED_FOMO_AUTO_DOWNLOAD_REPO=1."
        )
    return _ensure_fomo_repo(repo_url=repo_url, cache_root=repo_cache_root)


@lru_cache(maxsize=8)
def _load_fomo_module(repo_root: str):
    mm_path = os.path.join(repo_root, "model_zoo", "multimodal_mae.py")
    if not os.path.isfile(mm_path):
        raise ModelError(f"FoMo module not found: {mm_path}")

    spec = importlib.util.spec_from_file_location("fomo_multimodal_mae", mm_path)
    if spec is None or spec.loader is None:
        raise ModelError("Failed to create import spec for FoMo multimodal_mae.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _extract_state_dict(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ModelError(f"Unexpected FoMo checkpoint type: {type(obj)}")
    if isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ModelError("FoMo checkpoint has invalid state_dict format.")
    cleaned: Dict[str, Any] = {}
    for k, v in obj.items():
        nk = str(k)
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _build_fomo_model_config(
    *,
    image_size: int,
    patch_size: int,
    dim: int,
    depth: int,
    heads: int,
    mlp_dim: int,
    num_classes: int,
) -> Dict[str, Any]:
    return {
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "dim": int(dim),
        "depth": int(depth),
        "heads": int(heads),
        "mlp_dim": int(mlp_dim),
        "num_classes": int(num_classes),
        "single_embedding_layer": True,
        "modality_channels": dict(_DEFAULT_MODALITY_CHANNELS),
    }


@lru_cache(maxsize=8)
def _load_fomo_cached(
    *,
    ckpt_path: str,
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    auto_download_repo: bool,
    image_size: int,
    patch_size: int,
    dim: int,
    depth: int,
    heads: int,
    mlp_dim: int,
    num_classes: int,
    dev: str,
) -> Tuple[Any, Dict[str, Any]]:
    ensure_torch()
    import torch

    if importlib.util.find_spec("einops") is None:
        raise ModelError("FoMo requires einops. Install: pip install einops")

    repo_root = _resolve_fomo_repo(
        repo_path=repo_path,
        repo_url=repo_url,
        repo_cache_root=repo_cache_root,
        auto_download=auto_download_repo,
    )
    mod = _load_fomo_module(repo_root)
    if not hasattr(mod, "MultiSpectralViT"):
        raise ModelError("FoMo module does not expose MultiSpectralViT.")

    cfg = _build_fomo_model_config(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
    )

    model = mod.MultiSpectralViT(
        image_size=int(image_size),
        patch_size=int(patch_size),
        channels=1,
        num_classes=int(num_classes),
        dim=int(dim),
        depth=int(depth),
        heads=int(heads),
        mlp_dim=int(mlp_dim),
        configs=cfg,
    )
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(obj)
    msg = model.load_state_dict(sd, strict=False)

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
        raise ModelError(
            "FoMo model has no parameters; cannot verify loaded checkpoint."
        )
    if not torch.isfinite(p0).all():
        raise ModelError(
            "FoMo model parameters contain NaN/Inf; checkpoint load likely failed."
        )
    p0f = p0.float()

    meta = {
        "repo_root": repo_root,
        "ckpt_path": ckpt_path,
        "ckpt_size": int(os.path.getsize(ckpt_path)),
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "dim": int(dim),
        "depth": int(depth),
        "heads": int(heads),
        "mlp_dim": int(mlp_dim),
        "num_classes": int(num_classes),
        "device": str(dev),
        "missing_keys": int(len(getattr(msg, "missing_keys", []))),
        "unexpected_keys": int(len(getattr(msg, "unexpected_keys", []))),
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
    }
    return model, meta


def _load_fomo(
    *,
    ckpt_path: str,
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    auto_download_repo: bool,
    image_size: int,
    patch_size: int,
    dim: int,
    depth: int,
    heads: int,
    mlp_dim: int,
    num_classes: int,
    device: str,
) -> Tuple[Any, Dict[str, Any], str]:
    (loaded, dev) = _load_cached_with_device(
        _load_fomo_cached,
        device=device,
        ckpt_path=os.path.expanduser(ckpt_path),
        repo_path=(os.path.expanduser(repo_path) if repo_path else None),
        repo_url=str(repo_url),
        repo_cache_root=str(repo_cache_root),
        auto_download_repo=bool(auto_download_repo),
        image_size=int(image_size),
        patch_size=int(patch_size),
        dim=int(dim),
        depth=int(depth),
        heads=int(heads),
        mlp_dim=int(mlp_dim),
        num_classes=int(num_classes),
    )
    model, meta = loaded
    return model, meta, dev


def _resize_chw(x_chw: np.ndarray, *, out_hw: int) -> np.ndarray:
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW array, got {x_chw.shape}")
    x = torch.from_numpy(x_chw.astype(np.float32, copy=False)).unsqueeze(0)
    y = F.interpolate(
        x, size=(int(out_hw), int(out_hw)), mode="bilinear", align_corners=False
    )
    return y[0].detach().cpu().numpy().astype(np.float32)


def _normalize_s2(raw_chw: np.ndarray, *, mode: str) -> np.ndarray:
    x = np.asarray(raw_chw, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 10000.0)

    m = str(mode).lower().strip()
    if m in {"unit", "unit_scale", "reflectance"}:
        x = x / 10000.0
    elif m in {"per_tile_minmax", "minmax", "tile_minmax"}:
        x = x / 10000.0
        lo = np.min(x, axis=(1, 2), keepdims=True)
        hi = np.max(x, axis=(1, 2), keepdims=True)
        den = np.maximum(hi - lo, 1e-6)
        x = (x - lo) / den
    elif m in {"none", "raw"}:
        pass
    else:
        raise ModelError(
            f"Unknown FoMo normalization mode '{mode}'. "
            "Use one of: unit_scale, per_tile_minmax, none."
        )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _resolve_s2_modality_keys() -> Tuple[int, ...]:
    raw = str(os.environ.get("RS_EMBED_FOMO_S2_KEYS") or "").strip()
    if not raw:
        return _DEFAULT_S2_MODALITY_KEYS
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    keys = tuple(int(p) for p in parts)
    if len(keys) != len(_S2_SR_12_BANDS):
        raise ModelError(
            f"RS_EMBED_FOMO_S2_KEYS expects {len(_S2_SR_12_BANDS)} comma-separated integers, got {len(keys)}."
        )
    return keys


def _fomo_forward_tokens(
    model: Any,
    x_bchw: np.ndarray,
    *,
    spectral_keys: Tuple[int, ...],
    device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ensure_torch()
    import torch

    dev = _resolve_device(device)
    x = torch.from_numpy(x_bchw.astype(np.float32, copy=False)).to(dev)
    keys = [int(k) for k in spectral_keys]

    try:
        model = model.to(dev).eval()
    except Exception:
        pass

    with torch.no_grad():
        out = model((x, keys), pool=False)

    if not hasattr(out, "ndim") or int(out.ndim) != 3:
        raise ModelError(
            f"FoMo forward(pool=False) expected [B,N,D] tensor, got type={type(out)}"
        )
    if int(out.shape[0]) != 1:
        raise ModelError(f"FoMo embedder expects B=1 per call, got {tuple(out.shape)}")

    tokens = out[0].detach().float().cpu().numpy().astype(np.float32)  # [N,D]
    meta = {
        "token_count": int(tokens.shape[0]),
        "token_dim": int(tokens.shape[1]),
    }
    return tokens, meta


def _tokens_to_grid(
    tokens_nd: np.ndarray, *, n_modalities: int, patch_size: int, image_size: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_tokens, dim = int(tokens_nd.shape[0]), int(tokens_nd.shape[1])
    expected_gs = int(image_size) // int(patch_size) if int(patch_size) > 0 else 0
    expected_per_mod = expected_gs * expected_gs if expected_gs > 0 else 0
    expected_tokens = n_modalities * expected_per_mod

    if n_modalities <= 0 or n_tokens % n_modalities != 0:
        vec = np.mean(tokens_nd, axis=0).astype(np.float32)
        return vec[:, None, None], {
            "grid_kind": "vector_as_1x1",
            "grid_hw": (1, 1),
            "grid_shape": (int(vec.shape[0]), 1, 1),
            "grid_expected_tokens": int(expected_tokens),
        }

    per_mod = n_tokens // n_modalities
    gs = int(round(float(per_mod) ** 0.5))
    if gs * gs != per_mod:
        vec = np.mean(tokens_nd, axis=0).astype(np.float32)
        return vec[:, None, None], {
            "grid_kind": "vector_as_1x1",
            "grid_hw": (1, 1),
            "grid_shape": (int(vec.shape[0]), 1, 1),
            "grid_expected_tokens": int(expected_tokens),
        }

    toks = tokens_nd.reshape(n_modalities, gs, gs, dim)  # [K,H,W,D]
    grid = toks.mean(axis=0).transpose(2, 0, 1).astype(np.float32)  # [D,H,W]
    return grid, {
        "grid_kind": "spectral_mean_patch_tokens",
        "grid_hw": (int(gs), int(gs)),
        "grid_shape": tuple(grid.shape),
        "grid_modalities": int(n_modalities),
        "grid_expected_tokens": int(expected_tokens),
    }


@register("fomo")
class FoMoEmbedder(EmbedderBase):
    DEFAULT_IMAGE_SIZE = 64
    DEFAULT_PATCH = 16
    DEFAULT_DIM = 768
    DEFAULT_DEPTH = 12
    DEFAULT_HEADS = 12
    DEFAULT_MLP_DIM = 2048
    DEFAULT_NUM_CLASSES = 1000
    DEFAULT_FETCH_WORKERS = 8

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": _S2_SR_12_BANDS,
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "patch_size": self.DEFAULT_PATCH,
                "dim": self.DEFAULT_DIM,
                "depth": self.DEFAULT_DEPTH,
                "heads": self.DEFAULT_HEADS,
                "mlp_dim": self.DEFAULT_MLP_DIM,
                "num_classes": self.DEFAULT_NUM_CLASSES,
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "normalization": "unit_scale",
                "auto_download_ckpt": True,
            },
            "notes": [
                "Loads FoMo MultiSpectralViT from the official FoMo-Bench repository.",
                "Default checkpoint source is the FoMo-Net_1 weights link provided by FoMo-Bench README.",
                "Grid output averages patch tokens across the provided S2 spectral modalities.",
            ],
        }

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return SensorSpec(
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=tuple(_S2_SR_12_BANDS),
            scale_m=10,
            cloudy_pct=30,
            composite="median",
            fill_value=0.0,
        )

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_FOMO_FETCH_WORKERS", str(FoMoEmbedder.DEFAULT_FETCH_WORKERS)
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
        backend_l = backend.lower().strip()
        if not is_provider_backend(backend_l, allow_auto=True):
            raise ModelError("fomo expects a provider backend (or 'auto').")

        ss = sensor or self._default_sensor()
        t = temporal_to_range(temporal)

        image_size = int(
            os.environ.get("RS_EMBED_FOMO_IMG", str(self.DEFAULT_IMAGE_SIZE))
        )
        patch_size = int(os.environ.get("RS_EMBED_FOMO_PATCH", str(self.DEFAULT_PATCH)))
        dim = int(os.environ.get("RS_EMBED_FOMO_DIM", str(self.DEFAULT_DIM)))
        depth = int(os.environ.get("RS_EMBED_FOMO_DEPTH", str(self.DEFAULT_DEPTH)))
        heads = int(os.environ.get("RS_EMBED_FOMO_HEADS", str(self.DEFAULT_HEADS)))
        mlp_dim = int(
            os.environ.get("RS_EMBED_FOMO_MLP_DIM", str(self.DEFAULT_MLP_DIM))
        )
        num_classes = int(
            os.environ.get("RS_EMBED_FOMO_NUM_CLASSES", str(self.DEFAULT_NUM_CLASSES))
        )
        norm_mode = os.environ.get("RS_EMBED_FOMO_NORM", "unit_scale").strip()

        repo_path = os.environ.get("RS_EMBED_FOMO_REPO_PATH")
        repo_url = os.environ.get("RS_EMBED_FOMO_REPO_URL", _DEFAULT_REPO_URL).strip()
        repo_cache = os.environ.get("RS_EMBED_FOMO_REPO_CACHE", _DEFAULT_REPO_CACHE)
        auto_download_repo = _env_flag("RS_EMBED_FOMO_AUTO_DOWNLOAD_REPO", True)
        ckpt_path = _resolve_fomo_ckpt_path()
        spectral_keys = _resolve_s2_modality_keys()

        if input_chw is None:
            raw = _fetch_s2_sr_12_raw_chw(
                self._get_provider(backend_l),
                spatial,
                t,
                scale_m=int(ss.scale_m),
                cloudy_pct=int(ss.cloudy_pct),
                composite=str(ss.composite),
                fill_value=float(ss.fill_value),
            )
        else:
            raw = np.asarray(input_chw, dtype=np.float32)
            if raw.ndim != 3 or int(raw.shape[0]) != len(_S2_SR_12_BANDS):
                raise ModelError(
                    f"input_chw must be CHW with 12 bands for fomo, got {raw.shape}"
                )
            raw = np.clip(
                np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 10000.0
            ).astype(np.float32)

        x = _normalize_s2(raw, mode=norm_mode)
        if int(x.shape[-2]) != image_size or int(x.shape[-1]) != image_size:
            x = _resize_chw(x, out_hw=image_size)
        x_bchw = x[None, ...].astype(np.float32)

        if int(x_bchw.shape[1]) != len(spectral_keys):
            raise ModelError(
                f"FoMo spectral key count ({len(spectral_keys)}) does not match input channels ({int(x_bchw.shape[1])})."
            )

        model, lmeta, dev = _load_fomo(
            ckpt_path=ckpt_path,
            repo_path=repo_path,
            repo_url=repo_url,
            repo_cache_root=repo_cache,
            auto_download_repo=auto_download_repo,
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            device=device,
        )

        tokens, fmeta = _fomo_forward_tokens(
            model,
            x_bchw,
            spectral_keys=spectral_keys,
            device=dev,
        )
        grid, gmeta = _tokens_to_grid(
            tokens,
            n_modalities=len(spectral_keys),
            patch_size=patch_size,
            image_size=image_size,
        )

        if output.pooling == "max":
            vec = np.max(tokens, axis=0).astype(np.float32)
            pooling = "token_max"
        else:
            vec = np.mean(tokens, axis=0).astype(np.float32)
            pooling = "token_mean"

        meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=str(backend).lower(),
            source=ss.collection,
            sensor={
                "collection": ss.collection,
                "bands": tuple(_S2_SR_12_BANDS),
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
                "normalization": str(norm_mode),
                "spectral_keys": tuple(int(k) for k in spectral_keys),
                "device": dev,
                "pooling": pooling,
                "pooled_shape": tuple(vec.shape),
                **lmeta,
                **fmeta,
                **gmeta,
            },
        )

        if output.mode == "pooled":
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            gmeta_full = {
                **meta,
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
                attrs=gmeta_full,
            )
            return Embedding(data=da, meta=gmeta_full)

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

        backend_l = backend.lower().strip()
        if not is_provider_backend(backend_l, allow_auto=True):
            raise ModelError("fomo expects a provider backend (or 'auto').")

        t = temporal_to_range(temporal)
        ss = sensor or self._default_sensor()
        provider = self._get_provider(backend_l)

        n = len(spatials)
        prefetched_raw: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
            raw = _fetch_s2_sr_12_raw_chw(
                provider,
                sp,
                t,
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
                raise ModelError(f"Missing prefetched input at index={i} for fomo.")
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

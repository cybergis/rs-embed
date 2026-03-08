# Implementation based on:
# AgriFM: A multi-source temporal remote sensing foundation model for Agriculture mapping
# Remote Sensing of Environment 2026
# https://www.sciencedirect.com/science/article/pii/S0034425726000040

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
import sys
import types
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
    fetch_s2_multiframe_raw_tchw as _fetch_s2_multiframe_raw_tchw,
    is_provider_backend,
    load_cached_with_device as _load_cached_with_device,
)
from .meta_utils import build_meta, temporal_midpoint_str, temporal_to_range


_S2_10_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# AgriFM defaults from official config (S2 10-band statistics).
_AGRIFM_S2_MEAN = np.array(
    [
        4179.192015478227,
        4065.9106675194444,
        3957.274910960156,
        5207.452475253116,
        4327.12234687,
        4873.16102239,
        5049.1637925,
        5111.07806856,
        3056.86349163,
        2490.9675032,
    ],
    dtype=np.float32,
)
_AGRIFM_S2_STD = np.array(
    [
        4041.5212325268735,
        3691.003119315892,
        3629.331318356375,
        2973.5178530908756,
        3569.73343885,
        3085.9151435,
        2937.56005119,
        2806.04462314,
        1808.30013156,
        1694.20220774,
    ],
    dtype=np.float32,
)

_AGRIFM_DEFAULT_CKPT_URL = "https://glass.hku.hk/casual/AgriFM/AgriFM.pth"
_AGRIFM_DEFAULT_CKPT_FILENAME = "AgriFM.pth"
_AGRIFM_DEFAULT_CACHE_DIR = "~/.cache/rs_embed/agrifm"
_AGRIFM_DEFAULT_MIN_BYTES = 100 * 1024 * 1024


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() not in {"0", "false", "no", "off", ""}


def _download_url_to_path(url: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp_path, "wb") as f:
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
def _download_agrifm_ckpt(
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

    try:
        _download_url_to_path(url, dst)
    except Exception as e:
        raise ModelError(
            f"Failed to auto-download AgriFM checkpoint from: {url}\n"
            "Set RS_EMBED_AGRIFM_CKPT to a valid local .pth path if network/source is unavailable."
        ) from e

    sz = os.path.getsize(dst) if os.path.exists(dst) else 0
    if sz < int(min_bytes):
        raise ModelError(
            f"Downloaded AgriFM checkpoint is too small ({sz} bytes): {dst}. "
            "Set RS_EMBED_AGRIFM_CKPT to a valid local .pth checkpoint path."
        )
    return dst


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


def _normalize_s2_for_agrifm(raw_tchw: np.ndarray, *, mode: str) -> np.ndarray:
    if raw_tchw.ndim != 4 or int(raw_tchw.shape[1]) != len(_S2_10_BANDS):
        raise ModelError(
            f"AgriFM expects TCHW with C=10, got {getattr(raw_tchw, 'shape', None)}"
        )

    x = np.asarray(raw_tchw, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 10000.0)

    m = str(mode).lower().strip()
    if m in {"none", "raw", "off"}:
        return x.astype(np.float32)
    if m in {"unit", "unit_scale", "reflectance"}:
        return np.clip(x / 10000.0, 0.0, 1.0).astype(np.float32)
    if m in {"agrifm_stats", "zscore", "stats"}:
        std = np.maximum(_AGRIFM_S2_STD, 1e-6)
        x = (x - _AGRIFM_S2_MEAN[None, :, None, None]) / std[None, :, None, None]
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ModelError(
        f"Unknown AgriFM normalization mode '{mode}'. "
        "Use one of: agrifm_stats, unit_scale, none."
    )


@lru_cache(maxsize=4)
def _ensure_agrifm_repo(*, repo_url: str, cache_root: str) -> str:
    root = os.path.expanduser(cache_root)
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, "AgriFM")

    if os.path.isdir(os.path.join(dst, "AgriFM")) and os.path.isfile(
        os.path.join(dst, "README.md")
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
            "Failed to clone AgriFM source code. "
            f"Tried: git clone --depth 1 {repo_url} {dst}"
        ) from e
    return dst


def _resolve_agrifm_repo(
    *,
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    auto_download: bool,
) -> str:
    if repo_path:
        p = os.path.expanduser(repo_path)
        if not os.path.isdir(p):
            raise ModelError(f"RS_EMBED_AGRIFM_REPO_PATH does not exist: {p}")
        return p
    if not auto_download:
        raise ModelError(
            "AgriFM repository not provided. Set RS_EMBED_AGRIFM_REPO_PATH or enable auto download."
        )
    return _ensure_agrifm_repo(repo_url=repo_url, cache_root=repo_cache_root)


def _install_agrifm_lightweight_shims() -> None:
    """Install tiny mmseg/mmengine shims sufficient for AgriFM backbone import.

    This path avoids pulling heavy OpenMMLab runtime deps (mmcv/mmseg packages)
    when we only need PretrainingSwinTransformer3DEncoder for embeddings.
    """
    ensure_torch()
    import torch
    import torch.nn as nn

    class _MiniRegistry:
        def __init__(self, name: str):
            self.name = name
            self._items: Dict[str, Any] = {}

        def register_module(self, cls=None, name: Optional[str] = None):
            def _decorator(c):
                key = str(name or c.__name__)
                self._items[key] = c
                return c

            if cls is not None:
                return _decorator(cls)
            return _decorator

        def build(self, cfg: Dict[str, Any]):
            if not isinstance(cfg, dict):
                raise TypeError(f"{self.name}.build expects dict cfg, got {type(cfg)}")
            c = dict(cfg)
            typ = c.pop("type", None)
            if typ is None:
                raise KeyError(f"{self.name}.build requires cfg['type']")
            if isinstance(typ, str):
                if typ not in self._items:
                    raise KeyError(f"{self.name} cannot find class '{typ}'")
                cls = self._items[typ]
            elif callable(typ):
                cls = typ
            else:
                raise TypeError(f"Unsupported cfg['type']={type(typ)}")
            return cls(**c)

    def _new_module(name: str, *, is_package: bool = False) -> types.ModuleType:
        mod = types.ModuleType(name)
        # Some dependency loaders call importlib.util.find_spec("mmengine").
        # A module in sys.modules with __spec__=None can trigger ValueError.
        mod.__spec__ = importlib.util.spec_from_loader(
            name, loader=None, is_package=is_package
        )
        if is_package:
            # Mark as package so dotted imports like mmseg.models.decode_heads work.
            mod.__path__ = []  # type: ignore[attr-defined]
        return mod

    if "mmseg.models.builder" not in sys.modules:
        reg_backbones = _MiniRegistry("BACKBONES")
        mod_builder = _new_module("mmseg.models.builder")
        mod_builder.BACKBONES = reg_backbones

        mod_models = _new_module("mmseg.models", is_package=True)
        mod_models.builder = mod_builder

        # terratorch tries to import this optional module when mmseg is present.
        mod_decode_heads = _new_module("mmseg.models.decode_heads")
        mod_models.decode_heads = mod_decode_heads

        mod_mmseg = _new_module("mmseg", is_package=True)
        mod_mmseg.models = mod_models

        sys.modules["mmseg"] = mod_mmseg
        sys.modules["mmseg.models"] = mod_models
        sys.modules["mmseg.models.builder"] = mod_builder
        sys.modules["mmseg.models.decode_heads"] = mod_decode_heads
    else:
        reg_backbones = getattr(sys.modules["mmseg.models.builder"], "BACKBONES")

    if "mmseg.registry.registry" not in sys.modules:
        reg_models = _MiniRegistry("MODELS")
        reg_transforms = _MiniRegistry("TRANSFORMS")
        mod_registry = _new_module("mmseg.registry.registry")
        mod_registry.MODELS = reg_models
        mod_registry.TRANSFORMS = reg_transforms

        mod_registry_pkg = _new_module("mmseg.registry", is_package=True)
        mod_registry_pkg.registry = mod_registry
        mod_registry_pkg.MODELS = reg_models
        mod_registry_pkg.TRANSFORMS = reg_transforms

        sys.modules["mmseg.registry"] = mod_registry_pkg
        sys.modules["mmseg.registry.registry"] = mod_registry
    else:
        reg_models = getattr(sys.modules["mmseg.registry.registry"], "MODELS")

    # Keep registries shared so MODELS.build can resolve BACKBONES-registered classes.
    if hasattr(reg_models, "_items") and hasattr(reg_backbones, "_items"):
        try:
            reg_models._items = reg_backbones._items  # type: ignore[attr-defined]
        except Exception:
            pass

    # Rebind exported objects on modules.
    try:
        sys.modules["mmseg.models.builder"].BACKBONES = reg_models
        sys.modules["mmseg.registry.registry"].MODELS = reg_models
    except Exception:
        pass

    if "mmengine.model" not in sys.modules:
        mod_mmengine_model = _new_module("mmengine.model")

        class BaseModule(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class BaseModel(nn.Module):
            def __init__(self, data_preprocessor=None, init_cfg=None):
                super().__init__()
                self.data_preprocessor = data_preprocessor
                self.init_cfg = init_cfg

        mod_mmengine_model.BaseModule = BaseModule
        mod_mmengine_model.BaseModel = BaseModel
        sys.modules["mmengine.model"] = mod_mmengine_model

    if "mmengine.runner" not in sys.modules:
        mod_mmengine_runner = _new_module("mmengine.runner")

        def load_checkpoint(
            model,
            filename,
            strict=False,
            revise_keys=None,
            map_location="cpu",
            **kwargs,
        ):
            obj = torch.load(filename, map_location=map_location, weights_only=False)
            if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
                sd = obj["state_dict"]
            elif isinstance(obj, dict) and isinstance(obj.get("model"), dict):
                sd = obj["model"]
            elif isinstance(obj, dict):
                sd = obj
            else:
                raise RuntimeError(f"Unsupported checkpoint object type: {type(obj)}")

            if revise_keys:
                patched = {}
                for k, v in sd.items():
                    nk = str(k)
                    for pat, rep in revise_keys:
                        nk = re.sub(str(pat), str(rep), nk)
                    patched[nk] = v
                sd = patched

            incompatible = model.load_state_dict(sd, strict=bool(strict))
            return {
                "missing_keys": list(getattr(incompatible, "missing_keys", [])),
                "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
            }

        def load_state_dict(model, state_dict, strict=False):
            return model.load_state_dict(state_dict, strict=bool(strict))

        mod_mmengine_runner.load_checkpoint = load_checkpoint
        mod_mmengine_runner.load_state_dict = load_state_dict
        sys.modules["mmengine.runner"] = mod_mmengine_runner

    if "mmengine" not in sys.modules:
        mod_mmengine = _new_module("mmengine")
        mod_mmengine.model = sys.modules["mmengine.model"]
        mod_mmengine.runner = sys.modules["mmengine.runner"]
        sys.modules["mmengine"] = mod_mmengine
    else:
        mod_mmengine = sys.modules.get("mmengine")
        if mod_mmengine is not None and getattr(mod_mmengine, "__spec__", None) is None:
            mod_mmengine.__spec__ = importlib.util.spec_from_loader(
                "mmengine", loader=None
            )


@lru_cache(maxsize=8)
def _import_agrifm_swin(repo_root: str):
    repo_abs = os.path.abspath(os.path.expanduser(repo_root))
    if repo_abs not in sys.path:
        sys.path.insert(0, repo_abs)
    try:
        return importlib.import_module("AgriFM.models.video_swin_transformer")
    except Exception as first_error:
        # Fallback: import file directly with lightweight shims for mmseg/mmengine.
        mod_path = os.path.join(
            repo_abs, "AgriFM", "models", "video_swin_transformer.py"
        )
        if not os.path.isfile(mod_path):
            raise ModelError(
                f"Failed to locate AgriFM backbone source file: {mod_path}"
            ) from first_error
        try:
            _install_agrifm_lightweight_shims()
            spec = importlib.util.spec_from_file_location(
                "agrifm_video_swin_impl", mod_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    "Failed to create import spec for AgriFM video_swin_transformer.py"
                )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod
        except Exception as second_error:
            raise ModelError(
                "Failed to import AgriFM backbone even with lightweight shims. "
                "Install missing minimal deps: timm, einops (and torch). "
                f"Original import error: {type(first_error).__name__}: {first_error}; "
                f"fallback error: {type(second_error).__name__}: {second_error}"
            ) from second_error


def _resolve_ckpt_path() -> str:
    p = str(os.environ.get("RS_EMBED_AGRIFM_CKPT") or "").strip()
    if p:
        p = os.path.expanduser(p)
        if not os.path.exists(p):
            raise ModelError(f"RS_EMBED_AGRIFM_CKPT does not exist: {p}")
        return p

    if not _env_flag("RS_EMBED_AGRIFM_AUTO_DOWNLOAD", True):
        raise ModelError(
            "AgriFM checkpoint is required. Set RS_EMBED_AGRIFM_CKPT to a local path, "
            "or enable RS_EMBED_AGRIFM_AUTO_DOWNLOAD=1."
        )

    cache_dir = str(
        os.environ.get("RS_EMBED_AGRIFM_CACHE_DIR", _AGRIFM_DEFAULT_CACHE_DIR)
    ).strip()
    if not cache_dir:
        cache_dir = _AGRIFM_DEFAULT_CACHE_DIR
    filename = str(
        os.environ.get("RS_EMBED_AGRIFM_CKPT_FILE", _AGRIFM_DEFAULT_CKPT_FILENAME)
    ).strip()
    if not filename:
        filename = _AGRIFM_DEFAULT_CKPT_FILENAME
    url = str(
        os.environ.get("RS_EMBED_AGRIFM_CKPT_URL", _AGRIFM_DEFAULT_CKPT_URL)
    ).strip()
    if not url:
        url = _AGRIFM_DEFAULT_CKPT_URL
    min_bytes = int(
        os.environ.get("RS_EMBED_AGRIFM_CKPT_MIN_BYTES", str(_AGRIFM_DEFAULT_MIN_BYTES))
    )

    return _download_agrifm_ckpt(
        url=url,
        cache_dir=cache_dir,
        filename=filename,
        min_bytes=min_bytes,
    )


def _assert_weights_loaded(model) -> Dict[str, float]:
    ensure_torch()
    import torch

    p = None
    for _, param in model.named_parameters():
        if param is not None and param.numel() > 0:
            p = param.detach()
            break
    if p is None:
        raise ModelError("AgriFM model has no parameters; cannot verify weights.")
    if not torch.isfinite(p).all():
        raise ModelError(
            "AgriFM parameters contain NaN/Inf; checkpoint load likely failed."
        )

    p_f = p.float()
    std = float(p_f.std().cpu())
    mx = float(p_f.abs().max().cpu())
    mean = float(p_f.mean().cpu())
    if std < 1e-6 and mx < 1e-5:
        raise ModelError("AgriFM parameters look uninitialized (near-zero stats).")
    return {"param_mean": mean, "param_std": std, "param_absmax": mx}


@lru_cache(maxsize=6)
def _load_agrifm_cached(
    *,
    ckpt_path: str,
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    auto_download_repo: bool,
    dev: str,
) -> Tuple[Any, Dict[str, Any]]:
    ensure_torch()

    repo_root = _resolve_agrifm_repo(
        repo_path=repo_path,
        repo_url=repo_url,
        repo_cache_root=repo_cache_root,
        auto_download=auto_download_repo,
    )
    mod = _import_agrifm_swin(repo_root)

    cls = getattr(mod, "PretrainingSwinTransformer3DEncoder", None)
    if cls is None:
        raise ModelError(
            "AgriFM source does not expose PretrainingSwinTransformer3DEncoder."
        )

    patch_cfg = dict(
        type="SwinPatchEmbed3D",
        patch_size=(4, 2, 2),
        in_chans=10,
        embed_dim=128,
    )
    backbone_cfg = dict(
        type="SwinTransformer3D",
        pretrained=None,
        pretrained2d=False,
        patch_size=(4, 2, 2),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=False,
        frozen_stages=-1,
        use_checkpoint=False,
        downsample_steps=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        feature_fusion="cat",
        mean_frame_down=True,
    )

    model = cls(
        patch_emd_cfg=patch_cfg,
        backbone_cfg=backbone_cfg,
        init_cfg={"checkpoint": ckpt_path, "strict": False},
    )
    try:
        model = model.to(dev).eval()
    except Exception:
        pass

    stats = _assert_weights_loaded(model)
    meta = {
        "device": dev,
        "repo_root": repo_root,
        "repo_url": repo_url,
        "checkpoint": ckpt_path,
        "weights_verified": True,
        **stats,
    }
    return model, meta


def _load_agrifm(
    *,
    ckpt_path: str,
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    auto_download_repo: bool,
    device: str = "auto",
) -> Tuple[Any, Dict[str, Any], str]:
    (loaded, dev) = _load_cached_with_device(
        _load_agrifm_cached,
        device=device,
        ckpt_path=ckpt_path,
        repo_path=repo_path,
        repo_url=repo_url,
        repo_cache_root=repo_cache_root,
        auto_download_repo=auto_download_repo,
    )
    model, meta = loaded
    return model, meta, dev


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
    return _fetch_s2_multiframe_raw_tchw(
        provider,
        spatial=spatial,
        temporal=temporal,
        bands=tuple(_S2_10_BANDS),
        n_frames=int(n_frames),
        collection="COPERNICUS/S2_SR_HARMONIZED",
        scale_m=int(scale_m),
        cloudy_pct=int(cloudy_pct),
        composite=str(composite),
        fill_value=float(fill_value),
    )


def _agrifm_forward_grid(
    model, x_tchw: np.ndarray, *, device: str
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ensure_torch()
    import torch

    if x_tchw.ndim != 4 or int(x_tchw.shape[1]) != len(_S2_10_BANDS):
        raise ModelError(
            f"AgriFM expects TCHW with C=10, got {getattr(x_tchw, 'shape', None)}"
        )

    xb = (
        torch.from_numpy(x_tchw.astype(np.float32, copy=False)).unsqueeze(0).to(device)
    )  # [1,T,C,H,W]
    with torch.no_grad():
        out = model(xb, mode="tensor")

    feats = None
    feat_list_shapes: Optional[List[Tuple[int, ...]]] = None
    if isinstance(out, dict):
        feats = out.get("encoder_features", None)
        fl = out.get("features_list", None)
        if isinstance(fl, (tuple, list)):
            feat_list_shapes = [
                tuple(int(v) for v in x.shape) for x in fl if hasattr(x, "shape")
            ]
    else:
        feats = out

    if feats is None or not hasattr(feats, "ndim"):
        raise ModelError("AgriFM forward did not return encoder features.")

    if feats.ndim == 5:
        # [B,C,T,H,W] -> temporal mean for unified spatial grid output.
        feats = feats.mean(dim=2)
    if feats.ndim != 4:
        raise ModelError(
            f"Unexpected AgriFM feature shape: {getattr(feats, 'shape', None)}"
        )
    if int(feats.shape[0]) != 1:
        raise ModelError(
            f"AgriFM expects batch size 1 in single inference, got {tuple(feats.shape)}"
        )

    grid = feats[0].detach().float().cpu().numpy().astype(np.float32)  # [D,H,W]
    meta = {
        "feature_shape": tuple(int(v) for v in feats.shape),
        "feature_list_shapes": feat_list_shapes,
    }
    return grid, meta


@register("agrifm")
class AgriFMEmbedder(EmbedderBase):
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_FRAMES = 8
    DEFAULT_REPO_URL = "https://github.com/flyakon/AgriFM"
    DEFAULT_REPO_CACHE_ROOT = "~/.cache/rs_embed"
    DEFAULT_NORM = "agrifm_stats"

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": _S2_10_BANDS,
                "shape": "[T,10,H,W]",
            },
            "output": ["pooled", "grid"],
            "defaults": {
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "n_frames": self.DEFAULT_FRAMES,
                "norm_mode": self.DEFAULT_NORM,
                "auto_download_ckpt": True,
                "default_ckpt_url": _AGRIFM_DEFAULT_CKPT_URL,
            },
            "required_env": [],
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
                "RS_EMBED_AGRIFM_FETCH_WORKERS",
                str(AgriFMEmbedder.DEFAULT_FETCH_WORKERS),
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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("agrifm expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)
        n_frames = max(
            1, int(os.environ.get("RS_EMBED_AGRIFM_FRAMES", str(self.DEFAULT_FRAMES)))
        )
        image_size = max(
            16, int(os.environ.get("RS_EMBED_AGRIFM_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        )
        norm_mode = os.environ.get("RS_EMBED_AGRIFM_NORM", self.DEFAULT_NORM).strip()

        ckpt_path = _resolve_ckpt_path()
        repo_path = os.environ.get("RS_EMBED_AGRIFM_REPO_PATH", None)
        repo_url = os.environ.get(
            "RS_EMBED_AGRIFM_REPO_URL", self.DEFAULT_REPO_URL
        ).strip()
        repo_cache_root = os.environ.get(
            "RS_EMBED_AGRIFM_REPO_CACHE", self.DEFAULT_REPO_CACHE_ROOT
        ).strip()
        auto_download_repo = os.environ.get(
            "RS_EMBED_AGRIFM_AUTO_DOWNLOAD_REPO", "1"
        ).strip() not in ("0", "false", "False")

        if input_chw is None:
            raw_tchw = _fetch_s2_10_raw_tchw(
                self._get_provider(backend),
                spatial,
                t,
                n_frames=n_frames,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
            )
        else:
            raw = np.asarray(input_chw, dtype=np.float32)
            if raw.ndim == 3:
                if int(raw.shape[0]) != len(_S2_10_BANDS):
                    raise ModelError(
                        f"input_chw must be CHW with 10 bands for agrifm, got {raw.shape}"
                    )
                raw_tchw = np.repeat(raw[None, ...], repeats=n_frames, axis=0).astype(
                    np.float32
                )
            elif raw.ndim == 4:
                if int(raw.shape[1]) != len(_S2_10_BANDS):
                    raise ModelError(
                        f"input_chw must be TCHW with C=10 for agrifm, got {raw.shape}"
                    )
                raw_tchw = raw.astype(np.float32, copy=False)
                if raw_tchw.shape[0] < n_frames:
                    raw_tchw = np.concatenate(
                        [raw_tchw] + [raw_tchw[-1:]] * (n_frames - raw_tchw.shape[0]),
                        axis=0,
                    )
                elif raw_tchw.shape[0] > n_frames:
                    raw_tchw = raw_tchw[:n_frames]
            else:
                raise ModelError(
                    f"input_chw must be CHW (10 bands) or TCHW (T,10,H,W), got {getattr(raw, 'shape', None)}"
                )
            raw_tchw = np.nan_to_num(raw_tchw, nan=0.0, posinf=0.0, neginf=0.0)
            raw_tchw = np.clip(raw_tchw, 0.0, 10000.0).astype(np.float32)

        # Optional: inspect first frame on normalized [0,1] scale.
        from ..tools.inspection import maybe_inspect_chw, checks_should_raise

        check_meta: Dict[str, Any] = {"input_frames": int(raw_tchw.shape[0])}
        report = maybe_inspect_chw(
            np.clip(raw_tchw[0] / 10000.0, 0.0, 1.0).astype(np.float32),
            sensor=sensor,
            name="provider_s2_agrifm_frame0_chw",
            expected_channels=len(_S2_10_BANDS),
            value_range=(0.0, 1.0),
            fill_value=float(sensor.fill_value),
            meta=check_meta,
        )
        if (
            report is not None
            and (not report.get("ok", True))
            and checks_should_raise(sensor)
        ):
            raise ModelError(
                "Provider input inspection failed: "
                + "; ".join(report.get("issues", []))
            )

        x_tchw = _normalize_s2_for_agrifm(raw_tchw, mode=norm_mode)
        x_tchw = _resize_tchw(x_tchw, out_hw=image_size)

        model, wmeta, dev = _load_agrifm(
            ckpt_path=ckpt_path,
            repo_path=repo_path,
            repo_url=repo_url,
            repo_cache_root=repo_cache_root,
            auto_download_repo=auto_download_repo,
            device=device,
        )

        grid, fmeta = _agrifm_forward_grid(model, x_tchw, device=dev)
        meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=str(backend).lower(),
            source=sensor.collection,
            sensor=sensor,
            temporal=t,
            image_size=image_size,
            input_time=temporal_midpoint_str(t),
            extra={
                "bands": tuple(_S2_10_BANDS),
                "n_frames": int(raw_tchw.shape[0]),
                "norm_mode": norm_mode,
                "input_hw": (int(raw_tchw.shape[-2]), int(raw_tchw.shape[-1])),
                "grid_hw": (int(grid.shape[1]), int(grid.shape[2])),
                **check_meta,
                **wmeta,
                **fmeta,
            },
        )

        if output.mode == "pooled":
            if output.pooling == "mean":
                vec = grid.mean(axis=(1, 2)).astype(np.float32)
            elif output.pooling == "max":
                vec = grid.max(axis=(1, 2)).astype(np.float32)
            else:
                raise ModelError(f"Unknown pooling mode: {output.pooling}")
            meta = {**meta, "pooling": f"spatial_{output.pooling}"}
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            da = xr.DataArray(
                grid.astype(np.float32),
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(grid.shape[0]),
                    "y": np.arange(grid.shape[1]),
                    "x": np.arange(grid.shape[2]),
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
            raise ModelError("agrifm expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)
        n_frames = max(
            1, int(os.environ.get("RS_EMBED_AGRIFM_FRAMES", str(self.DEFAULT_FRAMES)))
        )
        provider = self._get_provider(backend)
        n = len(spatials)
        prefetched_raw: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
            raw_tchw = _fetch_s2_10_raw_tchw(
                provider,
                sp,
                t,
                n_frames=n_frames,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
            )
            return i, raw_tchw

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
                raise ModelError(f"Missing prefetched input at index={i} for agrifm.")
            out.append(
                self.get_embedding(
                    spatial=sp,
                    temporal=t,
                    sensor=sensor,
                    output=output,
                    backend=backend,
                    device=device,
                    input_chw=raw,
                )
            )
        return out

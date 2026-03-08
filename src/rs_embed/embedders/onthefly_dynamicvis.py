# Implementation based on:
# DynamicVis
# https://huggingface.co/KyanChen/DynamicVis

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..providers.base import ProviderBase
from .base import EmbedderBase
from .runtime_utils import (
    fetch_s2_rgb_chw as _fetch_s2_rgb_chw,
    is_provider_backend,
    load_cached_with_device as _load_cached_with_device,
    resolve_device_auto_torch as _resolve_device,
)
from .meta_utils import build_meta, temporal_midpoint_str, temporal_to_range
from ._vit_mae_utils import fetch_s2_rgb_u8_from_provider, resize_rgb_u8, ensure_torch


def _missing_runtime_modules() -> List[str]:
    # DynamicVis upstream relies on OpenMMLab runtime + mmcv.
    required = ("mmengine", "mmcv")
    missing: List[str] = []
    for name in required:
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None:
            missing.append(name)
    return missing


@lru_cache(maxsize=4)
def _ensure_dynamicvis_repo(
    *,
    repo_url: str,
    cache_root: str,
) -> str:
    root = os.path.expanduser(cache_root)
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, "DynamicVis")

    if os.path.isdir(os.path.join(dst, "dynamicvis")):
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
            "Failed to clone DynamicVis source code. "
            f"Tried: git clone --depth 1 {repo_url} {dst}"
        ) from e
    return dst


def _ensure_dynamicvis_importable(
    *,
    auto_download: bool,
    repo_path: Optional[str],
    repo_url: str,
    cache_root: str,
) -> str:
    # Prefer an existing installation/editable package.
    try:
        import dynamicvis  # noqa: F401

        mod = sys.modules.get("dynamicvis")
        return (
            os.path.dirname(os.path.abspath(getattr(mod, "__file__", "")))
            if mod is not None
            else "pythonpath"
        )
    except Exception:
        pass

    if repo_path:
        repo_root = os.path.expanduser(repo_path)
        if not os.path.isdir(repo_root):
            raise ModelError(
                f"RS_EMBED_DYNAMICVIS_REPO_PATH does not exist: {repo_root}"
            )
    elif auto_download:
        repo_root = _ensure_dynamicvis_repo(repo_url=repo_url, cache_root=cache_root)
    else:
        raise ModelError(
            "DynamicVis code not found. Install DynamicVis package or set "
            "RS_EMBED_DYNAMICVIS_REPO_PATH, or enable auto download."
        )

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import dynamicvis  # noqa: F401
    except Exception as e:
        raise ModelError(
            "DynamicVis source exists but import failed. "
            "Ensure dependencies are installed (mmengine, mmcv, transformers, etc.)."
        ) from e
    return repo_root


@lru_cache(maxsize=8)
def _download_dynamicvis_ckpt(
    *,
    hf_repo: str,
    ckpt_file: str,
    cache_dir: Optional[str],
    min_bytes: int = 50 * 1024 * 1024,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise ModelError(
            "DynamicVis requires huggingface_hub. Install: pip install huggingface_hub"
        ) from e

    p = hf_hub_download(repo_id=hf_repo, filename=ckpt_file, cache_dir=cache_dir)
    if not os.path.exists(p):
        raise ModelError(
            f"Failed to download DynamicVis checkpoint: {hf_repo}/{ckpt_file}"
        )
    sz = os.path.getsize(p)
    if sz < min_bytes:
        raise ModelError(
            f"Checkpoint looks too small: {p} ({sz} bytes). "
            "It may be an incomplete pointer/download."
        )
    return p


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, Any]:
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        return ckpt_obj
    raise ModelError(f"Unexpected checkpoint object type: {type(ckpt_obj)}")


def _load_weights_into_dynamicvis(model: Any, ckpt_path: str) -> Dict[str, Any]:
    ensure_torch()
    import torch

    # Best path: mmengine checkpoint loader with prefix rewrite.
    try:
        from mmengine.runner import load_checkpoint

        load_checkpoint(
            model,
            ckpt_path,
            map_location="cpu",
            strict=False,
            revise_keys=[(r"^module\.", ""), (r"^backbone\.", "")],
        )
        return {"checkpoint_loader": "mmengine.load_checkpoint"}
    except Exception:
        pass

    # Fallback: direct torch load + manual prefix stripping.
    ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = _extract_state_dict(ckpt_obj)
    sd = {}
    for k, v in raw_sd.items():
        nk = str(k)
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]
        sd[nk] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return {
        "checkpoint_loader": "torch.load",
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


@lru_cache(maxsize=4)
def _load_dynamicvis_backbone_cached(
    *,
    arch: str,
    image_size: int,
    path_type: str,
    sampling_scale: float,
    mamba2: bool,
    hf_repo: str,
    ckpt_file: str,
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    hf_cache_dir: Optional[str],
    auto_download_repo: bool,
    dev: str,
) -> Tuple[Any, Dict[str, Any]]:
    missing = _missing_runtime_modules()
    if missing:
        raise ModelError(
            "DynamicVis runtime deps missing: "
            + ", ".join(missing)
            + ". Follow DynamicVis README to install OpenMMLab runtime first."
        )

    repo_root = _ensure_dynamicvis_importable(
        auto_download=auto_download_repo,
        repo_path=repo_path,
        repo_url=repo_url,
        cache_root=repo_cache_root,
    )

    try:
        from dynamicvis.models.models import DynamicVisBackbone
    except Exception as e:
        raise ModelError(
            "Failed to import DynamicVisBackbone from dynamicvis.models.models. "
            "Check DynamicVis code and dependencies."
        ) from e

    ckpt_path = _download_dynamicvis_ckpt(
        hf_repo=hf_repo, ckpt_file=ckpt_file, cache_dir=hf_cache_dir
    )

    try:
        model = DynamicVisBackbone(
            mamba2=bool(mamba2),
            arch=str(arch),
            path_type=str(path_type),
            sampling_scale={"type": "fixed", "val": float(sampling_scale)},
            global_token_cfg={"pos": "head", "num": -1},
            img_size=int(image_size),
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            spatial_token_keep_ratios=[8, 4, 2, 1],
            out_indices=(3,),
            out_type="featmap",
        )
    except Exception as e:
        raise ModelError("Failed to construct DynamicVisBackbone.") from e

    load_meta = _load_weights_into_dynamicvis(model, ckpt_path)

    try:
        model = model.to(dev).eval()
    except Exception:
        pass

    # basic parameter sanity
    p0 = None
    for _, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError(
            "DynamicVis model has no parameters; cannot verify loaded weights."
        )

    ensure_torch()
    import torch

    if not torch.isfinite(p0).all():
        raise ModelError(
            "DynamicVis parameters contain NaN/Inf; checkpoint load likely failed."
        )

    p0f = p0.float()
    meta = {
        "arch": str(arch),
        "image_size": int(image_size),
        "path_type": str(path_type),
        "sampling_scale": float(sampling_scale),
        "mamba2": bool(mamba2),
        "hf_repo": str(hf_repo),
        "ckpt_file": str(ckpt_file),
        "ckpt_path": ckpt_path,
        "ckpt_size": int(os.path.getsize(ckpt_path)),
        "repo_root": repo_root,
        "device": str(dev),
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
        **load_meta,
    }
    return model, meta


def _load_dynamicvis_backbone(
    *,
    arch: str,
    image_size: int,
    path_type: str,
    sampling_scale: float,
    mamba2: bool,
    hf_repo: str,
    ckpt_file: str,
    repo_path: Optional[str],
    repo_url: str,
    repo_cache_root: str,
    hf_cache_dir: Optional[str],
    auto_download_repo: bool,
    device: str,
) -> Tuple[Any, Dict[str, Any], str]:
    (loaded, dev) = _load_cached_with_device(
        _load_dynamicvis_backbone_cached,
        device=device,
        arch=str(arch),
        image_size=int(image_size),
        path_type=str(path_type),
        sampling_scale=float(sampling_scale),
        mamba2=bool(mamba2),
        hf_repo=str(hf_repo),
        ckpt_file=str(ckpt_file),
        repo_path=(str(repo_path) if repo_path else None),
        repo_url=str(repo_url),
        repo_cache_root=str(repo_cache_root),
        hf_cache_dir=(str(hf_cache_dir) if hf_cache_dir else None),
        auto_download_repo=bool(auto_download_repo),
    )
    model, meta = loaded
    return model, meta, dev


def _rgb_u8_to_tensor_imagenet(rgb_u8: np.ndarray, *, image_size: int):
    ensure_torch()
    import torch

    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or int(rgb_u8.shape[2]) != 3:
        raise ModelError(
            f"Expected uint8 HWC RGB image, got dtype={rgb_u8.dtype}, shape={rgb_u8.shape}"
        )

    if rgb_u8.shape[0] != image_size or rgb_u8.shape[1] != image_size:
        rgb_u8 = resize_rgb_u8(rgb_u8, image_size)

    x = rgb_u8.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean[None, None, :]) / std[None, None, :]
    x = x.transpose(2, 0, 1)  # CHW
    t = torch.from_numpy(x).unsqueeze(0)  # [1,3,H,W]
    return t


def _dynamicvis_forward_last_fmap(
    model: Any,
    rgb_u8: np.ndarray,
    *,
    image_size: int,
    device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ensure_torch()
    import torch

    dev = _resolve_device(device)
    x = _rgb_u8_to_tensor_imagenet(rgb_u8, image_size=image_size).to(dev)

    try:
        model = model.to(dev).eval()
    except Exception:
        pass

    with torch.no_grad():
        out = model(x)

    t = None
    n_out = None
    if isinstance(out, (tuple, list)):
        n_out = len(out)
        if len(out) > 0 and hasattr(out[-1], "ndim"):
            t = out[-1]
    elif hasattr(out, "ndim"):
        n_out = 1
        t = out

    if t is None:
        raise ModelError(f"DynamicVis forward returned unsupported type: {type(out)}")

    if t.ndim == 4:
        fmap = t[0].detach().float().cpu().numpy().astype(np.float32)  # [C,H,W]
        meta = {
            "forward_out_count": int(n_out or 1),
            "feature_kind": "featmap",
            "feature_shape": tuple(fmap.shape),
        }
        return fmap, meta

    if t.ndim == 2:
        vec = t[0].detach().float().cpu().numpy().astype(np.float32)  # [D]
        fmap = vec[:, None, None]
        meta = {
            "forward_out_count": int(n_out or 1),
            "feature_kind": "vector_as_1x1",
            "feature_shape": tuple(fmap.shape),
        }
        return fmap, meta

    raise ModelError(
        f"DynamicVis forward output tensor has unsupported shape: {tuple(t.shape)}"
    )


@register("dynamicvis")
class DynamicVisEmbedder(EmbedderBase):
    DEFAULT_MODEL_REPO = "KyanChen/DynamicVis"
    DEFAULT_CKPT_FILE = (
        "pretrain_dynamicvis_b_bf16_mamba_best_single-label_f1-score_epoch_170.pth"
    )
    DEFAULT_ARCH = "b"
    DEFAULT_IMAGE_SIZE = 512
    DEFAULT_FETCH_WORKERS = 8

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": ["B4", "B3", "B2"],
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "hf_repo": self.DEFAULT_MODEL_REPO,
                "ckpt_file": self.DEFAULT_CKPT_FILE,
                "arch": self.DEFAULT_ARCH,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
            },
            "notes": [
                "Loads DynamicVis backbone from official repository + HuggingFace checkpoint.",
                "Requires OpenMMLab runtime dependencies (mmengine/mmcv).",
            ],
        }

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return SensorSpec(
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=("B4", "B3", "B2"),
            scale_m=10,
            cloudy_pct=30,
            composite="median",
        )

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_DYNAMICVIS_FETCH_WORKERS",
                str(DynamicVisEmbedder.DEFAULT_FETCH_WORKERS),
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
            raise ModelError("dynamicvis expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)
        image_size = int(
            os.environ.get("RS_EMBED_DYNAMICVIS_IMG", str(self.DEFAULT_IMAGE_SIZE))
        )
        hf_repo = os.environ.get(
            "RS_EMBED_DYNAMICVIS_HF_REPO", self.DEFAULT_MODEL_REPO
        ).strip()
        ckpt_file = os.environ.get(
            "RS_EMBED_DYNAMICVIS_CKPT_FILE", self.DEFAULT_CKPT_FILE
        ).strip()
        arch_env = os.environ.get("RS_EMBED_DYNAMICVIS_ARCH", "auto").strip().lower()
        arch = (
            arch_env
            if arch_env in ("b", "base", "l", "large")
            else ("l" if "_l_" in ckpt_file else "b")
        )
        path_type = os.environ.get(
            "RS_EMBED_DYNAMICVIS_PATH_TYPE", "forward_reverse_mean"
        ).strip()
        sampling_scale = float(
            os.environ.get("RS_EMBED_DYNAMICVIS_SAMPLING_SCALE", "0.1")
        )
        mamba2 = os.environ.get("RS_EMBED_DYNAMICVIS_MAMBA2", "0").strip() in {
            "1",
            "true",
            "True",
        }
        auto_download_repo = os.environ.get(
            "RS_EMBED_DYNAMICVIS_AUTO_DOWNLOAD_REPO", "1"
        ).strip() not in {
            "0",
            "false",
            "False",
        }
        repo_path = os.environ.get("RS_EMBED_DYNAMICVIS_REPO_PATH")
        repo_url = os.environ.get(
            "RS_EMBED_DYNAMICVIS_REPO_URL", "https://github.com/KyanChen/DynamicVis.git"
        ).strip()
        repo_cache_root = os.environ.get(
            "RS_EMBED_DYNAMICVIS_REPO_CACHE",
            os.path.join("~", ".cache", "rs_embed", "dynamicvis"),
        )
        hf_cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HOME")
        )

        if input_chw is None:
            rgb_u8 = fetch_s2_rgb_u8_from_provider(
                spatial=spatial,
                temporal=t,
                sensor=sensor,
                out_size=image_size,
                provider=self._get_provider(backend),
            )
        else:
            if input_chw.ndim != 3 or int(input_chw.shape[0]) != 3:
                raise ModelError(
                    f"input_chw must be CHW with 3 bands for dynamicvis, got {getattr(input_chw, 'shape', None)}"
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            rgb_u8 = resize_rgb_u8(rgb_u8, image_size)

        model, lmeta, dev = _load_dynamicvis_backbone(
            arch=arch,
            image_size=image_size,
            path_type=path_type,
            sampling_scale=sampling_scale,
            mamba2=mamba2,
            hf_repo=hf_repo,
            ckpt_file=ckpt_file,
            repo_path=repo_path,
            repo_url=repo_url,
            repo_cache_root=repo_cache_root,
            hf_cache_dir=hf_cache_dir,
            auto_download_repo=auto_download_repo,
            device=device,
        )
        fmap, fmeta = _dynamicvis_forward_last_fmap(
            model,
            rgb_u8,
            image_size=image_size,
            device=dev,
        )

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
                "pool_source": "last_stage_featmap",
                "start": t.start,
                "end": t.end,
                "device": dev,
                **lmeta,
                **fmeta,
            },
        )

        if output.mode == "pooled":
            if output.pooling == "max":
                vec = np.max(fmap, axis=(1, 2)).astype(np.float32)
            else:
                vec = np.mean(fmap, axis=(1, 2)).astype(np.float32)
            meta.update(
                {
                    "pooling": f"featmap_{output.pooling}",
                    "pooled_shape": tuple(vec.shape),
                }
            )
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            grid = fmap.astype(np.float32)  # [D,H,W]
            try:
                import xarray as xr
            except Exception as e:
                raise ModelError(
                    "grid output requires xarray. Install: pip install xarray"
                ) from e

            gmeta = {
                **meta,
                "grid_hw": (int(grid.shape[1]), int(grid.shape[2])),
                "grid_kind": "last_stage_featmap",
                "grid_shape": tuple(grid.shape),
            }
            da = xr.DataArray(
                grid,
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

        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("dynamicvis expects a provider backend (or 'auto').")

        t = temporal_to_range(temporal)
        ss = sensor or self._default_sensor()
        provider = self._get_provider(backend)
        n = len(spatials)

        scale_m = int(ss.scale_m)
        cloudy_pct = int(ss.cloudy_pct)
        composite = str(ss.composite)

        prefetched_raw: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
            s2_rgb_chw = _fetch_s2_rgb_chw(
                provider,
                spatial=sp,
                temporal=t,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
            )
            raw = np.clip(s2_rgb_chw * 10000.0, 0.0, 10000.0).astype(np.float32)
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
                raise ModelError(
                    f"Missing prefetched input at index={i} for dynamicvis."
                )
            out.append(
                self.get_embedding(
                    spatial=sp,
                    temporal=temporal,
                    sensor=ss,
                    output=output,
                    backend=backend,
                    device=device,
                    input_chw=raw,
                )
            )
        return out

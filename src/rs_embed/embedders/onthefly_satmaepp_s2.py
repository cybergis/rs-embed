from __future__ import annotations

import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..providers.base import ProviderBase
from ._vit_mae_utils import base_meta, ensure_torch, temporal_to_range
from .base import EmbedderBase
from .runtime_utils import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
    is_provider_backend,
    load_cached_with_device as _load_cached_with_device,
)

# SatMAE++ Sentinel branch in source repo drops [B1, B9, B10] and uses 10 channels.
# GEE S2 SR does not expose B10, so we directly fetch the final 10-channel subset.
_S2_SR_10_BANDS = (
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
)

# Source: techmn/satmae_pp util/datasets.py Sentinel mean/std after dropping [0,9,10].
_SENTINEL_MEAN_10 = np.array(
    [
        1184.3824625,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        1732.16362238,
        1247.91870117,
    ],
    dtype=np.float32,
)
_SENTINEL_STD_10 = np.array(
    [
        650.2842772,
        712.12507725,
        965.23119807,
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        1310.36996126,
        1087.6020813,
    ],
    dtype=np.float32,
)

# Source default grouped bands for dropped-band Sentinel setting.
_S2_10_CHANNEL_GROUPS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 6),
    (3, 4, 5, 7),
    (8, 9),
)

def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ModelError(f"{name} must be a boolean string (1/0, true/false), got {raw!r}.")

def _torch_load_checkpoint_compat(path: str):
    """
    Compatibility wrapper for PyTorch 2.6+ `weights_only` default change.

    SatMAE++ Sentinel source checkpoints are full training checkpoints and may
    include objects like argparse.Namespace, so we default to weights_only=False
    (matching source repo behavior).
    """
    ensure_torch()
    import torch

    weights_only = _env_bool("RS_EMBED_SATMAEPP_S2_WEIGHTS_ONLY", False)
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        # Older torch versions do not support `weights_only` kwarg.
        return torch.load(path, map_location="cpu")
    except Exception as e:
        if weights_only:
            raise ModelError(
                "SatMAE++ Sentinel checkpoint load failed with weights_only=True. "
                "This checkpoint likely contains non-tensor metadata. "
                "Set RS_EMBED_SATMAEPP_S2_WEIGHTS_ONLY=0 and retry if you trust the source."
            ) from e
        raise

def _resolve_hf_cache_dir() -> str | None:
    d = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HOME")
    )
    return str(d) if d else None

def _ensure_satmaepp_s2_assets(
    *,
    ckpt_repo: str,
    ckpt_file: str,
    cache_dir: str | None,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise ModelError(
            "SatMAE++ Sentinel checkpoint download requires huggingface_hub. Install: pip install huggingface_hub"
        ) from e

    ckpt_path = hf_hub_download(repo_id=ckpt_repo, filename=ckpt_file, cache_dir=cache_dir)
    if not os.path.exists(ckpt_path):
        raise ModelError(f"Failed to download checkpoint {ckpt_repo}/{ckpt_file}")

    return ckpt_path

@lru_cache(maxsize=1)
def _load_satmaepp_s2_module():
    try:
        return importlib.import_module(
            "rs_embed.embedders._vendor.satmaepp_s2.models_mae_group_channels"
        )
    except Exception as e:
        raise ModelError(
            f"Failed to import vendored SatMAE++ Sentinel runtime: {type(e).__name__}: {e}"
        ) from e

def _strip_module_prefix(sd: dict[str, Any]) -> dict[str, Any]:
    if not sd:
        return sd
    keys = list(sd.keys())
    if all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

@lru_cache(maxsize=4)
def _load_satmaepp_s2_cached(
    *,
    ckpt_repo: str,
    ckpt_file: str,
    model_fn: str,
    image_size: int,
    patch_size: int,
    dev: str,
    cache_dir: str | None,
):
    ensure_torch()
    import torch

    ckpt_path = _ensure_satmaepp_s2_assets(
        ckpt_repo=ckpt_repo,
        ckpt_file=ckpt_file,
        cache_dir=cache_dir,
    )
    mod = _load_satmaepp_s2_module()

    factory = getattr(mod, model_fn, None)
    if not callable(factory):
        raise ModelError(f"SatMAE++ source module has no callable model factory '{model_fn}'.")

    try:
        model = factory(
            img_size=int(image_size),
            patch_size=int(patch_size),
            in_chans=len(_S2_SR_10_BANDS),
            channel_groups=_S2_10_CHANNEL_GROUPS,
            spatial_mask=False,
            norm_pix_loss=False,
        )
    except Exception as e:
        raise ModelError(
            f"Failed to construct SatMAE++ Sentinel model via {model_fn}(): {e}"
        ) from e

    ckpt = _torch_load_checkpoint_compat(ckpt_path)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict) or not state:
        raise ModelError(
            f"Invalid checkpoint format for {ckpt_repo}/{ckpt_file}; expected non-empty state_dict."
        )
    state = _strip_module_prefix(state)

    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        raise ModelError(
            "Failed to load SatMAE++ Sentinel checkpoint with strict=True. "
            f"Check image/patch/channel config. Error: {type(e).__name__}: {e}"
        ) from e

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
        raise ModelError("SatMAE++ Sentinel model has no parameters; checkpoint load failed.")
    if not torch.isfinite(p0).all():
        raise ModelError("SatMAE++ Sentinel parameters contain NaN/Inf; checkpoint likely invalid.")

    meta = {
        "device": str(dev),
        "ckpt_repo": str(ckpt_repo),
        "ckpt_file": str(ckpt_file),
        "model_source": "vendored_rs_embed_runtime",
        "model_fn": str(model_fn),
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "in_chans": len(_S2_SR_10_BANDS),
        "channel_groups": tuple(tuple(int(i) for i in g) for g in _S2_10_CHANNEL_GROUPS),
        "param_mean": float(p0.float().mean().cpu()),
        "param_std": float(p0.float().std().cpu()),
        "param_absmax": float(p0.float().abs().max().cpu()),
        "torch_weights_only": bool(_env_bool("RS_EMBED_SATMAEPP_S2_WEIGHTS_ONLY", False)),
    }
    return model, meta

def _load_satmaepp_s2(
    *,
    ckpt_repo: str,
    ckpt_file: str,
    model_fn: str,
    image_size: int,
    patch_size: int,
    cache_dir: str | None,
    device: str,
):
    loaded, _dev = _load_cached_with_device(
        _load_satmaepp_s2_cached,
        device=device,
        ckpt_repo=ckpt_repo,
        ckpt_file=ckpt_file,
        model_fn=model_fn,
        image_size=int(image_size),
        patch_size=int(patch_size),
        cache_dir=cache_dir,
    )
    return loaded

def _satmaepp_s2_resize_short_side(image_size: int) -> int:
    crop_pct = (224.0 / 256.0) if int(image_size) <= 224 else 1.0
    return int(float(image_size) / crop_pct)

def _sentinel_to_uint8_hwc(raw_chw_10: np.ndarray) -> np.ndarray:
    if raw_chw_10.ndim != 3 or int(raw_chw_10.shape[0]) != len(_S2_SR_10_BANDS):
        raise ModelError(
            f"SatMAE++ Sentinel expects CHW with {len(_S2_SR_10_BANDS)} bands, got {getattr(raw_chw_10, 'shape', None)}"
        )
    x = raw_chw_10.astype(np.float32, copy=False).transpose(1, 2, 0)
    min_v = (_SENTINEL_MEAN_10 - 2.0 * _SENTINEL_STD_10).reshape(1, 1, -1)
    max_v = (_SENTINEL_MEAN_10 + 2.0 * _SENTINEL_STD_10).reshape(1, 1, -1)
    y = (x - min_v) / np.maximum(max_v - min_v, 1e-6)
    y = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)
    return y

def _satmaepp_s2_preprocess_tensor_batch(raw_chw_batch: list[np.ndarray], *, image_size: int):
    ensure_torch()
    import torch
    from torchvision import transforms

    resize_short = _satmaepp_s2_resize_short_side(image_size)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resize_short, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    )

    xs = []
    for i, raw_chw in enumerate(raw_chw_batch):
        hwc_u8 = _sentinel_to_uint8_hwc(raw_chw)
        x = preprocess(hwc_u8)
        if x.ndim != 3 or int(x.shape[0]) != len(_S2_SR_10_BANDS):
            raise ModelError(
                f"SatMAE++ Sentinel preprocess returned shape={tuple(x.shape)} at index={i}; "
                f"expected [{len(_S2_SR_10_BANDS)},H,W]."
            )
        xs.append(x)
    return torch.stack(xs, dim=0)

def _satmaepp_s2_forward_tokens_batch(
    model,
    raw_chw_batch: list[np.ndarray],
    *,
    image_size: int,
    device: str,
) -> list[np.ndarray]:
    if not raw_chw_batch:
        return []

    ensure_torch()
    import torch

    xb = _satmaepp_s2_preprocess_tensor_batch(raw_chw_batch, image_size=image_size).to(device)

    fe = getattr(model, "forward_encoder", None)
    if not callable(fe):
        raise ModelError("SatMAE++ Sentinel model does not expose forward_encoder().")

    with torch.no_grad():
        out = fe(xb, mask_ratio=0.0)
        toks = out[0] if isinstance(out, (tuple, list)) else out
        if toks.ndim != 3 or int(toks.shape[0]) != len(raw_chw_batch):
            raise ModelError(
                f"SatMAE++ Sentinel forward_encoder returned {tuple(toks.shape)}; "
                f"expected [B,N,D] with B={len(raw_chw_batch)}."
            )
        arr = toks.detach().float().cpu().numpy().astype(np.float32)
    return [arr[i] for i in range(arr.shape[0])]

def _satmaepp_s2_split_tokens(tokens_nd: np.ndarray) -> tuple[np.ndarray, bool, int]:
    if tokens_nd.ndim != 2:
        raise ModelError(f"Expected tokens [N,D], got {getattr(tokens_nd, 'shape', None)}")

    g = len(_S2_10_CHANNEL_GROUPS)
    n = int(tokens_nd.shape[0])

    has_cls = False
    patch = tokens_nd
    if n > 1 and ((n - 1) % g == 0):
        has_cls = True
        patch = tokens_nd[1:]

    if int(patch.shape[0]) % g != 0:
        raise ModelError(
            "SatMAE++ Sentinel token layout is incompatible with grouped-channel decoder assumptions. "
            f"token_count={n}, groups={g}"
        )
    l = int(patch.shape[0]) // g
    return patch, has_cls, l

def _satmaepp_s2_pool(tokens_nd: np.ndarray, pooling: str) -> tuple[np.ndarray, bool]:
    patch, has_cls, _ = _satmaepp_s2_split_tokens(tokens_nd)
    if int(patch.shape[0]) == 0:
        raise ModelError("SatMAE++ Sentinel has no patch tokens to pool.")

    if pooling == "mean":
        return patch.mean(axis=0).astype(np.float32), has_cls
    if pooling == "max":
        return patch.max(axis=0).astype(np.float32), has_cls
    raise ModelError(f"Unknown pooling='{pooling}' (expected 'mean' or 'max').")

def _satmaepp_s2_grid(
    tokens_nd: np.ndarray,
    *,
    group_reduce: str = "mean",
) -> tuple[np.ndarray, tuple[int, int], bool, dict[str, Any]]:
    patch, has_cls, l = _satmaepp_s2_split_tokens(tokens_nd)
    g = len(_S2_10_CHANNEL_GROUPS)
    d = int(patch.shape[1])

    p_gld = patch.reshape(g, l, d)
    reduce_l = str(group_reduce).strip().lower()
    if reduce_l == "mean":
        spatial_ld = p_gld.mean(axis=0)
    elif reduce_l == "max":
        spatial_ld = p_gld.max(axis=0)
    else:
        raise ModelError(f"Unknown SatMAE++ Sentinel group reduction: {group_reduce!r}")

    h = int(np.sqrt(l))
    w = h
    if h * w != l:
        raise ModelError(f"Spatial token count {l} is not square; cannot form grid.")

    grid = spatial_ld.reshape(h, w, d).transpose(2, 0, 1).astype(np.float32)
    emeta = {
        "group_count": int(g),
        "group_reduce": reduce_l,
        "tokens_per_group": int(l),
        "channel_groups": tuple(tuple(int(i) for i in grp) for grp in _S2_10_CHANNEL_GROUPS),
    }
    return grid, (h, w), has_cls, emeta

def _fetch_s2_sr_10_raw_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int,
    cloudy_pct: int,
    composite: str,
    fill_value: float,
) -> np.ndarray:
    raw = _fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_S2_SR_10_BANDS),
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        fill_value=fill_value,
    )
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != len(_S2_SR_10_BANDS):
        raise ModelError(
            f"Provider fetch returned shape={getattr(arr, 'shape', None)}; "
            f"expected [{len(_S2_SR_10_BANDS)},H,W]."
        )
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, 0.0, 10000.0).astype(np.float32)

@register("satmaepp_s2_10b")
class SatMAEPPSentinel10Embedder(EmbedderBase):
    """
    SatMAE++ Sentinel-2 10-band adapter reproducing source repo group-channel branch.

    Input bands (S2 SR): B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
    Source alignment:
      - grouped-channel encoder with groups ((0,1,2,6),(3,4,5,7),(8,9))
      - SentinelNormalize(mean/std) -> ToTensor -> Resize -> CenterCrop (eval-style)
      - forward_encoder(mask_ratio=0.0) for embedding extraction
    """

    DEFAULT_CKPT_REPO = "mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel"
    DEFAULT_CKPT_FILE = "checkpoint_ViT-L_pretrain_fmow_sentinel.pth"
    DEFAULT_MODEL_FN = "mae_vit_large_patch16"

    DEFAULT_IMAGE_SIZE = 96
    DEFAULT_PATCH_SIZE = 8
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 32

    def describe(self) -> dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["provider"],
            "model_id_default": self.DEFAULT_CKPT_REPO,
            "image_size": self.DEFAULT_IMAGE_SIZE,
            "inputs": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": list(_S2_SR_10_BANDS),
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "ckpt_repo": self.DEFAULT_CKPT_REPO,
                "ckpt_file": self.DEFAULT_CKPT_FILE,
                "model_fn": self.DEFAULT_MODEL_FN,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "patch_size": self.DEFAULT_PATCH_SIZE,
                "channel_groups": tuple(
                    tuple(int(i) for i in grp) for grp in _S2_10_CHANNEL_GROUPS
                ),
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "normalization": "sentinel_normalize_source",
            },
        }

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return SensorSpec(
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=tuple(_S2_SR_10_BANDS),
            scale_m=10,
            cloudy_pct=30,
            composite="median",
            fill_value=0.0,
        )

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_SATMAEPP_S2_FETCH_WORKERS",
                str(SatMAEPPSentinel10Embedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default_bs = (
            SatMAEPPSentinel10Embedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else SatMAEPPSentinel10Embedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_SATMAEPP_S2_BATCH_SIZE", str(default_bs)))
        return max(1, v)

    @staticmethod
    def _resolve_group_reduce() -> str:
        v = str(os.environ.get("RS_EMBED_SATMAEPP_S2_GRID_REDUCE", "mean")).strip().lower()
        if v not in {"mean", "max"}:
            raise ModelError("RS_EMBED_SATMAEPP_S2_GRID_REDUCE must be 'mean' or 'max'.")
        return v

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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satmaepp_s2_10b expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        if tuple(sensor.bands) != tuple(_S2_SR_10_BANDS):
            raise ModelError(
                "satmaepp_s2_10b requires exact band order "
                f"{_S2_SR_10_BANDS}; got {tuple(sensor.bands)}"
            )

        ckpt_repo = os.environ.get("RS_EMBED_SATMAEPP_S2_CKPT_REPO", self.DEFAULT_CKPT_REPO).strip()
        ckpt_file = os.environ.get("RS_EMBED_SATMAEPP_S2_CKPT_FILE", self.DEFAULT_CKPT_FILE).strip()
        model_fn = os.environ.get("RS_EMBED_SATMAEPP_S2_MODEL_FN", self.DEFAULT_MODEL_FN).strip()
        image_size = int(os.environ.get("RS_EMBED_SATMAEPP_S2_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        patch_size = int(os.environ.get("RS_EMBED_SATMAEPP_S2_PATCH", str(self.DEFAULT_PATCH_SIZE)))
        cache_dir = _resolve_hf_cache_dir()

        t = temporal_to_range(temporal)
        if input_chw is None:
            raw_chw = _fetch_s2_sr_10_raw_chw(
                provider=self._get_provider(backend),
                spatial=spatial,
                temporal=t,
                scale_m=int(getattr(sensor, "scale_m", 10)),
                cloudy_pct=int(getattr(sensor, "cloudy_pct", 30)),
                composite=str(getattr(sensor, "composite", "median")),
                fill_value=float(getattr(sensor, "fill_value", 0.0)),
            )
        else:
            if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(_S2_SR_10_BANDS):
                raise ModelError(
                    f"input_chw must be CHW with {len(_S2_SR_10_BANDS)} bands for satmaepp_s2_10b, "
                    f"got {getattr(input_chw, 'shape', None)}"
                )
            raw_chw = np.asarray(input_chw, dtype=np.float32)
            raw_chw = np.clip(np.nan_to_num(raw_chw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 10000.0)

        model, wmeta = _load_satmaepp_s2(
            ckpt_repo=ckpt_repo,
            ckpt_file=ckpt_file,
            model_fn=model_fn,
            image_size=image_size,
            patch_size=patch_size,
            cache_dir=cache_dir,
            device=device,
        )
        dev = wmeta.get("device", device)

        toks_batch = _satmaepp_s2_forward_tokens_batch(
            model,
            [raw_chw],
            image_size=image_size,
            device=dev,
        )
        tokens = toks_batch[0]

        meta = base_meta(
            model_name=self.model_name,
            hf_id=f"{ckpt_repo}/{ckpt_file}",
            backend=str(backend).lower(),
            image_size=image_size,
            sensor=sensor,
            temporal=t,
            source=sensor.collection,
            extra={
                "tokens_kind": "tokens_forward_encoder_group_channel",
                "tokens_shape": tuple(tokens.shape),
                "bands": tuple(_S2_SR_10_BANDS),
                "channel_groups": tuple(
                    tuple(int(i) for i in grp) for grp in _S2_10_CHANNEL_GROUPS
                ),
                "normalization": "sentinel_normalize_source",
                "patch_size": int(patch_size),
                "ckpt_repo": ckpt_repo,
                "ckpt_file": ckpt_file,
                "model_fn": model_fn,
                **wmeta,
            },
        )

        if output.mode == "pooled":
            vec, cls_removed = _satmaepp_s2_pool(tokens, output.pooling)
            meta.update(
                {
                    "pooling": f"group_tokens_{output.pooling}",
                    "cls_removed": bool(cls_removed),
                }
            )
            return Embedding(data=vec.astype(np.float32), meta=meta)

        if output.mode == "grid":
            group_reduce = self._resolve_group_reduce()
            grid, (h, w), cls_removed, gmeta = _satmaepp_s2_grid(tokens, group_reduce=group_reduce)
            meta.update(
                {
                    "grid_hw": (h, w),
                    "grid_kind": "spatial_tokens_aggregated_over_channel_groups",
                    "cls_removed": bool(cls_removed),
                    **gmeta,
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
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satmaepp_s2_10b expects a provider backend (or 'auto').")
        if not spatials:
            return []

        if sensor is None:
            sensor = self._default_sensor()
        if tuple(sensor.bands) != tuple(_S2_SR_10_BANDS):
            raise ModelError(
                "satmaepp_s2_10b requires exact band order "
                f"{_S2_SR_10_BANDS}; got {tuple(sensor.bands)}"
            )

        ckpt_repo = os.environ.get("RS_EMBED_SATMAEPP_S2_CKPT_REPO", self.DEFAULT_CKPT_REPO).strip()
        ckpt_file = os.environ.get("RS_EMBED_SATMAEPP_S2_CKPT_FILE", self.DEFAULT_CKPT_FILE).strip()
        model_fn = os.environ.get("RS_EMBED_SATMAEPP_S2_MODEL_FN", self.DEFAULT_MODEL_FN).strip()
        image_size = int(os.environ.get("RS_EMBED_SATMAEPP_S2_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        patch_size = int(os.environ.get("RS_EMBED_SATMAEPP_S2_PATCH", str(self.DEFAULT_PATCH_SIZE)))
        cache_dir = _resolve_hf_cache_dir()
        t = temporal_to_range(temporal)

        provider = self._get_provider(backend)
        n = len(spatials)
        raws: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            raw = _fetch_s2_sr_10_raw_chw(
                provider=provider,
                spatial=sp,
                temporal=t,
                scale_m=int(getattr(sensor, "scale_m", 10)),
                cloudy_pct=int(getattr(sensor, "cloudy_pct", 30)),
                composite=str(getattr(sensor, "composite", "median")),
                fill_value=float(getattr(sensor, "fill_value", 0.0)),
            )
            return i, raw

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                _, raw = _fetch_one(i, sp)
                raws[i] = raw
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, raw = fut.result()
                    raws[i] = raw

        for i, x in enumerate(raws):
            if x is None:
                raise ModelError(f"Missing fetched patch at index={i}; batch fetch failed.")

        model, wmeta = _load_satmaepp_s2(
            ckpt_repo=ckpt_repo,
            ckpt_file=ckpt_file,
            model_fn=model_fn,
            image_size=image_size,
            patch_size=patch_size,
            cache_dir=cache_dir,
            device=device,
        )
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))

        out: list[Embedding | None] = [None] * n
        want_grid = output.mode == "grid"
        xr_mod = None
        group_reduce = self._resolve_group_reduce() if want_grid else "mean"
        if want_grid:
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            raw_batch = [x for x in raws[s0:s1] if x is not None]
            toks_batch = _satmaepp_s2_forward_tokens_batch(
                model,
                raw_batch,
                image_size=image_size,
                device=dev,
            )
            for j, tokens in enumerate(toks_batch):
                i = s0 + j
                meta = base_meta(
                    model_name=self.model_name,
                    hf_id=f"{ckpt_repo}/{ckpt_file}",
                    backend=str(backend).lower(),
                    image_size=image_size,
                    sensor=sensor,
                    temporal=t,
                    source=sensor.collection,
                    extra={
                        "tokens_kind": "tokens_forward_encoder_group_channel",
                        "tokens_shape": tuple(tokens.shape),
                        "bands": tuple(_S2_SR_10_BANDS),
                        "channel_groups": tuple(
                            tuple(int(i) for i in grp) for grp in _S2_10_CHANNEL_GROUPS
                        ),
                        "normalization": "sentinel_normalize_source",
                        "patch_size": int(patch_size),
                        "ckpt_repo": ckpt_repo,
                        "ckpt_file": ckpt_file,
                        "model_fn": model_fn,
                        **wmeta,
                    },
                )

                if output.mode == "pooled":
                    vec, cls_removed = _satmaepp_s2_pool(tokens, output.pooling)
                    meta.update(
                        {
                            "pooling": f"group_tokens_{output.pooling}",
                            "cls_removed": bool(cls_removed),
                        }
                    )
                    out[i] = Embedding(data=vec.astype(np.float32), meta=meta)
                elif output.mode == "grid":
                    assert xr_mod is not None
                    grid, (h, w), cls_removed, gmeta = _satmaepp_s2_grid(
                        tokens, group_reduce=group_reduce
                    )
                    meta.update(
                        {
                            "grid_hw": (h, w),
                            "grid_kind": "spatial_tokens_aggregated_over_channel_groups",
                            "cls_removed": bool(cls_removed),
                            **gmeta,
                        }
                    )
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
                else:
                    raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError("satmaepp_s2_10b batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satmaepp_s2_10b expects a provider backend (or 'auto').")
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        if sensor is None:
            sensor = self._default_sensor()
        if tuple(sensor.bands) != tuple(_S2_SR_10_BANDS):
            raise ModelError(
                "satmaepp_s2_10b requires exact band order "
                f"{_S2_SR_10_BANDS}; got {tuple(sensor.bands)}"
            )

        ckpt_repo = os.environ.get("RS_EMBED_SATMAEPP_S2_CKPT_REPO", self.DEFAULT_CKPT_REPO).strip()
        ckpt_file = os.environ.get("RS_EMBED_SATMAEPP_S2_CKPT_FILE", self.DEFAULT_CKPT_FILE).strip()
        model_fn = os.environ.get("RS_EMBED_SATMAEPP_S2_MODEL_FN", self.DEFAULT_MODEL_FN).strip()
        image_size = int(os.environ.get("RS_EMBED_SATMAEPP_S2_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        patch_size = int(os.environ.get("RS_EMBED_SATMAEPP_S2_PATCH", str(self.DEFAULT_PATCH_SIZE)))
        cache_dir = _resolve_hf_cache_dir()
        t = temporal_to_range(temporal)

        raws: list[np.ndarray] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(_S2_SR_10_BANDS):
                raise ModelError(
                    f"input_chw must be CHW with {len(_S2_SR_10_BANDS)} bands for satmaepp_s2_10b, "
                    f"got {getattr(input_chw, 'shape', None)} at index={i}"
                )
            raw = np.asarray(input_chw, dtype=np.float32)
            raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 10000.0)
            raws.append(raw)

        model, wmeta = _load_satmaepp_s2(
            ckpt_repo=ckpt_repo,
            ckpt_file=ckpt_file,
            model_fn=model_fn,
            image_size=image_size,
            patch_size=patch_size,
            cache_dir=cache_dir,
            device=device,
        )
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))

        out: list[Embedding | None] = [None] * len(spatials)
        want_grid = output.mode == "grid"
        xr_mod = None
        group_reduce = self._resolve_group_reduce() if want_grid else "mean"
        if want_grid:
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            toks_batch = _satmaepp_s2_forward_tokens_batch(
                model,
                raws[s0:s1],
                image_size=image_size,
                device=dev,
            )
            for j, tokens in enumerate(toks_batch):
                i = s0 + j
                meta = base_meta(
                    model_name=self.model_name,
                    hf_id=f"{ckpt_repo}/{ckpt_file}",
                    backend=str(backend).lower(),
                    image_size=image_size,
                    sensor=sensor,
                    temporal=t,
                    source=sensor.collection,
                    extra={
                        "tokens_kind": "tokens_forward_encoder_group_channel",
                        "tokens_shape": tuple(tokens.shape),
                        "bands": tuple(_S2_SR_10_BANDS),
                        "channel_groups": tuple(
                            tuple(int(i) for i in grp) for grp in _S2_10_CHANNEL_GROUPS
                        ),
                        "normalization": "sentinel_normalize_source",
                        "patch_size": int(patch_size),
                        "batch_infer": True,
                        "input_override": True,
                        "ckpt_repo": ckpt_repo,
                        "ckpt_file": ckpt_file,
                        "model_fn": model_fn,
                        **wmeta,
                    },
                )

                if output.mode == "pooled":
                    vec, cls_removed = _satmaepp_s2_pool(tokens, output.pooling)
                    meta.update(
                        {
                            "pooling": f"group_tokens_{output.pooling}",
                            "cls_removed": bool(cls_removed),
                        }
                    )
                    out[i] = Embedding(data=vec.astype(np.float32), meta=meta)
                elif output.mode == "grid":
                    assert xr_mod is not None
                    grid, (h, w), cls_removed, gmeta = _satmaepp_s2_grid(
                        tokens, group_reduce=group_reduce
                    )
                    meta.update(
                        {
                            "grid_hw": (h, w),
                            "grid_kind": "spatial_tokens_aggregated_over_channel_groups",
                            "cls_removed": bool(cls_removed),
                            **gmeta,
                        }
                    )
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
                else:
                    raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError("satmaepp_s2_10b batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

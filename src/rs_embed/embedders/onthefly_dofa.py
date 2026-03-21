# src/rs_embed/embedders/onthefly_dofa.py
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import (
    ModelInputSpec,
    NormalizationSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from ..providers import ProviderBase
from ._vendor.dofa_vit import vit_base_patch16, vit_large_patch16
from .base import EmbedderBase
from .runtime_utils import (
    coerce_single_input_chw,
)
from .runtime_utils import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
)
from .runtime_utils import (
    load_cached_with_device as _load_cached_with_device,
)
from .runtime_utils import (
    resolve_device_auto_torch as _resolve_device_auto,
)

# -----------------------------
# Defaults: Sentinel-2 SR (12 bands)
# -----------------------------
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

# Sentinel-2 MSI band central wavelengths (µm)
_S2_WAVELENGTHS_UM = {
    "B1": 0.443,
    "B2": 0.490,
    "B3": 0.560,
    "B4": 0.665,
    "B5": 0.705,
    "B6": 0.740,
    "B7": 0.783,
    "B8": 0.842,
    "B8A": 0.865,
    "B9": 0.945,
    "B11": 1.610,
    "B12": 2.190,
}

def _infer_wavelengths_um(bands: list[str]) -> list[float] | None:
    wv = []
    for b in bands:
        if b not in _S2_WAVELENGTHS_UM:
            return None
        wv.append(float(_S2_WAVELENGTHS_UM[b]))
    return wv

def _resize_chw(
    x_chw: np.ndarray,
    *,
    size: int = 224,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    CHW float32 -> CHW float32 resized to (size,size) (bilinear), no crop/pad.
    """
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW, got shape={x_chw.shape}")
    c, h, w = x_chw.shape
    info = {
        "orig_hw": (int(h), int(w)),
        "target_hw": (int(size), int(size)),
        "mode": "bilinear",
    }

    x = torch.from_numpy(x_chw.astype(np.float32, copy=False)).unsqueeze(0)  # [1,C,H,W]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    y = x[0].cpu().numpy().astype(np.float32)
    return y, info

# -----------------------------
# Provider fetch (generic SR scaling /10000)
# -----------------------------
def _fetch_provider_multiband_sr_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    collection: str,
    bands: list[str],
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
    default_value: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = _fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection=str(collection),
        bands=tuple(str(b) for b in bands),
        scale_m=int(scale_m),
        cloudy_pct=int(cloudy_pct),
        composite=str(composite),
        fill_value=float(default_value),
    )
    x = np.clip(raw / 10000.0, 0.0, 1.0).astype(np.float32)

    meta: dict[str, Any] = {
        "provider_collection": collection,
        "provider_bands": list(bands),
        "provider_scale_m": int(scale_m),
        "provider_cloudy_pct": int(cloudy_pct),
        "provider_cloud_filter_applied": True,
        "provider_composite": str(composite),
        "provider_n_images": None,
        "provider_time_start_ms": None,
        "provider_time_end_ms": None,
        "raw_chw_shape": tuple(x.shape),
        "region_crs": "EPSG:3857",
    }
    return x, meta

def _fetch_gee_multiband_sr_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    collection: str,
    bands: list[str],
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
    default_value: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible alias for historical helper name."""
    return _fetch_provider_multiband_sr_chw(
        provider,
        spatial,
        temporal,
        collection=collection,
        bands=bands,
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        default_value=default_value,
    )

# -----------------------------
# DOFA model + forward adapters
# -----------------------------

_DOFA_HF_REPO_ID_DEFAULT = "earthflow/DOFA"
_DOFA_HF_REVISION_DEFAULT = "main"
_DOFA_WEIGHT_SPECS = {
    "base": {
        "filename": "DOFA_ViT_base_e100.pth",
        "env_var": "RS_EMBED_DOFA_BASE_WEIGHTS",
    },
    "large": {
        "filename": "DOFA_ViT_large_e100.pth",
        "env_var": "RS_EMBED_DOFA_LARGE_WEIGHTS",
    },
}


def _resolve_dofa_weight_spec(variant: str) -> dict[str, str]:
    variant_l = str(variant).lower().strip()
    if variant_l in ("b", "base"):
        variant_l = "base"
    elif variant_l in ("l", "large"):
        variant_l = "large"
    else:
        raise ModelError(f"Unknown DOFA variant='{variant}' (expected 'base' or 'large').")

    spec = dict(_DOFA_WEIGHT_SPECS[variant_l])
    spec["variant"] = variant_l
    spec["repo_id"] = os.environ.get("RS_EMBED_DOFA_HF_REPO_ID", _DOFA_HF_REPO_ID_DEFAULT)
    spec["revision"] = os.environ.get(
        "RS_EMBED_DOFA_HF_REVISION",
        _DOFA_HF_REVISION_DEFAULT,
    )
    return spec


def _model_config_value(
    model_config: dict[str, Any] | None,
    key: str,
) -> Any | None:
    if model_config is None:
        return None
    if isinstance(model_config, dict):
        return model_config.get(key)
    return getattr(model_config, key, None)


def _resolve_dofa_variant(
    *,
    model_config: dict[str, Any] | None,
) -> str:
    variant = _model_config_value(model_config, "variant")
    if variant is not None:
        return str(variant)
    return "base"


def _build_dofa_model(variant: str):
    variant_l = str(variant).lower().strip()
    if variant_l == "base":
        return vit_base_patch16(global_pool=True)
    if variant_l == "large":
        return vit_large_patch16(global_pool=True)
    raise ModelError(f"Unknown DOFA variant='{variant}' (expected 'base' or 'large').")


def _strip_state_dict_prefix_if_present(
    state_dict: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if keys and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _unwrap_dofa_state_dict(payload: Any) -> dict[str, Any]:
    obj = payload
    for _ in range(4):
        if not isinstance(obj, dict):
            break
        nested = None
        for key in ("state_dict", "model_state_dict", "model", "teacher", "student"):
            if key in obj and isinstance(obj[key], dict):
                nested = obj[key]
                break
        if nested is None:
            break
        obj = nested

    if not isinstance(obj, dict):
        raise ModelError("Unexpected DOFA checkpoint payload; expected a state dict.")

    state_dict = dict(obj)
    for prefix in ("module.", "backbone.", "encoder.", "vit_model.", "model."):
        state_dict = _strip_state_dict_prefix_if_present(state_dict, prefix)
    return state_dict


def _prepare_dofa_state_dict_for_model(
    model,
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    model_state = model.state_dict()
    prepared = dict(state_dict)

    fc_norm_weight_missing = (
        "fc_norm.weight" not in prepared
        or getattr(prepared.get("fc_norm.weight"), "shape", None)
        != getattr(model_state.get("fc_norm.weight"), "shape", None)
    )
    if "fc_norm.weight" in model_state and fc_norm_weight_missing and "norm.weight" in prepared:
        prepared["fc_norm.weight"] = prepared["norm.weight"].detach().clone()

    fc_norm_bias_missing = (
        "fc_norm.bias" not in prepared
        or getattr(prepared.get("fc_norm.bias"), "shape", None)
        != getattr(model_state.get("fc_norm.bias"), "shape", None)
    )
    if "fc_norm.bias" in model_state and fc_norm_bias_missing and "norm.bias" in prepared:
        prepared["fc_norm.bias"] = prepared["norm.bias"].detach().clone()

    filtered: dict[str, Any] = {}
    dropped_mismatched: list[str] = []
    for key, value in prepared.items():
        if key not in model_state:
            filtered[key] = value
            continue
        target = model_state[key]
        if getattr(value, "shape", None) == getattr(target, "shape", None):
            filtered[key] = value
        else:
            dropped_mismatched.append(key)

    load_result = model.load_state_dict(filtered, strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    return filtered, missing, unexpected + dropped_mismatched


def _resolve_dofa_weights_path(spec: dict[str, str]) -> tuple[str, str]:
    override = os.environ.get(spec["env_var"])
    if override:
        return override, override

    weights_dir = os.environ.get("RS_EMBED_DOFA_WEIGHTS_DIR")
    if weights_dir:
        local_path = os.path.join(weights_dir, spec["filename"])
        if os.path.exists(local_path):
            return local_path, local_path

    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
    except Exception as e:
        raise ModelError(
            "DOFA requires huggingface-hub to download weights, or set "
            f"{spec['env_var']} / RS_EMBED_DOFA_WEIGHTS_DIR to a local checkpoint."
        ) from e

    try:
        local_path = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            revision=spec["revision"],
        )
        remote_url = hf_hub_url(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            revision=spec["revision"],
        )
        return local_path, remote_url
    except Exception as e:
        raise ModelError(
            "Failed to fetch DOFA weights from Hugging Face. Set "
            f"{spec['env_var']} or RS_EMBED_DOFA_WEIGHTS_DIR to a local checkpoint."
        ) from e

@lru_cache(maxsize=4)
def _load_dofa_model_cached(variant: str, dev: str):
    import torch

    spec = _resolve_dofa_weight_spec(variant)
    variant_l = spec["variant"]
    model = _build_dofa_model(variant_l)
    weights_path, weights_url = _resolve_dofa_weights_path(spec)

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _unwrap_dofa_state_dict(checkpoint)
    _, missing_keys, unexpected_keys = _prepare_dofa_state_dict_for_model(model, state_dict)

    model = model.to(dev).eval()

    # sanity
    p0 = None
    for _, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError("DOFA model has no parameters; unexpected.")
    if not torch.isfinite(p0).all():
        raise ModelError("DOFA parameters contain NaN/Inf; weight load likely failed.")

    meta = {
        "variant": variant_l,
        "device": dev,
        "device_resolved": dev,
        "weights_url": str(weights_url),
        "weights_meta": {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "filename": spec["filename"],
            "path": weights_path,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "img_size": int(getattr(model, "img_size", 224)),
        "patch_size": int(getattr(model, "patch_size", 16)),
        "embed_dim": int(getattr(model, "embed_dim", -1)),
        "global_pool": bool(getattr(model, "global_pool", True)),
    }
    return model, meta

def _load_dofa_model(
    *,
    variant: str = "base",
    device: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    variant_l = str(variant).lower().strip()
    loaded, _dev = _load_cached_with_device(
        _load_dofa_model_cached,
        device=device,
        variant=variant_l,
    )
    return loaded

def _dofa_forward_tokens_and_pooled(
    model,
    x_bchw: np.ndarray,
    wavelengths_um: list[float],
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Returns:
      patch_tokens: [N, D] (no CLS)
      pooled:       [D]
    """
    import torch

    dev = _resolve_device_auto(device)
    x = torch.from_numpy(x_bchw).to(dev)
    if x.dtype != torch.float32:
        x = x.float()

    wavelist = torch.tensor(wavelengths_um, device=dev).float()

    with torch.no_grad():
        # Patch embedding
        xtok, _ = model.patch_embed(x, wavelist)  # [B, N, D]
        # Pos embed (skip cls position)
        xtok = xtok + model.pos_embed[:, 1:, :]
        # Prepend CLS
        cls_token = model.cls_token + model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(xtok.shape[0], -1, -1)
        xseq = torch.cat((cls_tokens, xtok), dim=1)  # [B, 1+N, D]

        # Transformer blocks
        for blk in model.blocks:
            xseq = blk(xseq)

        # pooled to match torchgeo logic
        if getattr(model, "global_pool", True):
            pooled_t = xseq[:, 1:, :].mean(dim=1)
            pooled_t = model.fc_norm(pooled_t)
            pooled = pooled_t[0].detach().float().cpu().numpy().astype(np.float32)
            norm_applied = "fc_norm(global_pool_mean)"
        else:
            xseq = model.norm(xseq)
            pooled = xseq[:, 0][0].detach().float().cpu().numpy().astype(np.float32)
            norm_applied = "norm(cls)"

        patch_tokens = xseq[:, 1:, :][0].detach().float().cpu().numpy().astype(np.float32)  # [N,D]

    n, d = patch_tokens.shape
    side = int(round(math.sqrt(n)))
    extra = {
        "token_count": int(n),
        "token_dim": int(d),
        "token_grid_side": int(side) if side * side == n else None,
        "tokens_include_cls": False,
        "pooled_norm": norm_applied,
    }
    return patch_tokens, pooled, extra

def _dofa_forward_tokens_and_pooled_batch(
    model,
    x_bchw: np.ndarray,
    wavelengths_um: list[float],
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Batch forward for DOFA.

    Returns:
      patch_tokens: [B, N, D] (no CLS)
      pooled:       [B, D]
    """
    import torch

    dev = _resolve_device_auto(device)
    x = torch.from_numpy(x_bchw).to(dev)
    if x.dtype != torch.float32:
        x = x.float()
    wavelist = torch.tensor(wavelengths_um, device=dev).float()

    with torch.inference_mode():
        xtok, _ = model.patch_embed(x, wavelist)  # [B,N,D]
        xtok = xtok + model.pos_embed[:, 1:, :]
        cls_token = model.cls_token + model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(xtok.shape[0], -1, -1)
        xseq = torch.cat((cls_tokens, xtok), dim=1)  # [B,1+N,D]

        for blk in model.blocks:
            xseq = blk(xseq)

        if getattr(model, "global_pool", True):
            pooled_t = xseq[:, 1:, :].mean(dim=1)
            pooled_t = model.fc_norm(pooled_t)
            pooled = pooled_t.detach().float().cpu().numpy().astype(np.float32)  # [B,D]
            norm_applied = "fc_norm(global_pool_mean)"
        else:
            xseq = model.norm(xseq)
            pooled = xseq[:, 0].detach().float().cpu().numpy().astype(np.float32)  # [B,D]
            norm_applied = "norm(cls)"

        patch_tokens = xseq[:, 1:, :].detach().float().cpu().numpy().astype(np.float32)  # [B,N,D]

    n = int(patch_tokens.shape[1])
    d = int(patch_tokens.shape[2])
    side = int(round(math.sqrt(n)))
    extra = {
        "token_count": int(n),
        "token_dim": int(d),
        "token_grid_side": int(side) if side * side == n else None,
        "tokens_include_cls": False,
        "pooled_norm": norm_applied,
        "batch_shape": tuple(patch_tokens.shape),
    }
    return patch_tokens, pooled, extra

# -----------------------------
# Embedder
# -----------------------------
@register("dofa")
class DOFAEmbedder(EmbedderBase):
    """
    DOFA embeddings.

    - backend="provider"/"auto": ROI -> S2 SR -> resize to 224 -> DOFA -> pooled/grid
    - backend="tensor": input_chw (CHW) -> resize to 224 -> DOFA

    Output:
      - OutputSpec.pooled(): (D,)
      - OutputSpec.grid():   (D, Ht, Wt) token grid, usually 14x14 for 224/patch16
    """

    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 64

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_S2_SR_12_BANDS),
        scale_m=10,
        cloudy_pct=30,
        normalization=NormalizationSpec(mode="s2_sr_clip"),
        image_size=224,
        expected_channels=12,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider", "tensor"],
            "inputs": {
                "provider_default": {
                    "collection": self.input_spec.collection,
                    "bands": list(self.input_spec.bands),
                    "wavelengths_um": "auto for S2 bands",
                }
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "variant": "base",
                "image_size": self.input_spec.image_size,
                "scale_m": self.input_spec.scale_m,
                "cloudy_pct": self.input_spec.cloudy_pct,
                "composite": self.input_spec.composite,
                "preprocess": "resize_to_224_bilinear",
            },
            "model_config": {
                "variant": {
                    "type": "string",
                    "default": "base",
                    "choices": ["base", "large"],
                }
            },
        }

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return DOFAEmbedder.input_spec.to_sensor_spec()

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get("RS_EMBED_DOFA_FETCH_WORKERS", str(DOFAEmbedder.DEFAULT_FETCH_WORKERS))
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default_bs = (
            DOFAEmbedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else DOFAEmbedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_DOFA_BATCH_SIZE", str(default_bs)))
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
    ) -> Embedding:
        backend_l = backend.lower().strip()
        variant = _resolve_dofa_variant(model_config=model_config)
        image_size = 224

        # For optional on-the-fly input inspection
        check_meta: dict[str, Any] = {}

        # -----------------
        # Build input + wavelengths
        # -----------------
        if backend_l == "tensor":
            if input_chw is None:
                raise ModelError(
                    "backend='tensor' requires input_chw as CHW. "
                    "Use get_embeddings_batch_from_inputs(...) for batches."
                )
            x_chw = coerce_single_input_chw(
                input_chw,
                expected_channels=None,
                model_name="DOFA",
            )
            x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
            x_bchw = x_chw[None, ...]

            wavelengths_um = getattr(sensor, "wavelengths", None)
            if wavelengths_um is None:
                bands = list(getattr(sensor, "bands", [])) if hasattr(sensor, "bands") else []
                if bands:
                    wavelengths_um = _infer_wavelengths_um(bands)
            if wavelengths_um is None:
                raise ModelError(
                    "DOFA requires wavelengths (µm) per channel. "
                    "Provide sensor.wavelengths=[...] or (for S2) provide sensor.bands to infer."
                )
            wavelengths_um = [float(v) for v in wavelengths_um]

            provider_meta = {"backend_tensor": True}

        else:
            provider = self._get_provider(backend)
            if temporal is None:
                raise ModelError("dofa provider backend requires TemporalSpec.range(start,end).")
            temporal.validate()
            if temporal.mode != "range":
                raise ModelError("dofa provider backend requires TemporalSpec.range in v0.1.")

            # overrides
            collection = (
                getattr(sensor, "collection", "COPERNICUS/S2_SR_HARMONIZED")
                if sensor
                else "COPERNICUS/S2_SR_HARMONIZED"
            )
            bands = (
                list(getattr(sensor, "bands", _S2_SR_12_BANDS)) if sensor else list(_S2_SR_12_BANDS)
            )
            scale_m = int(getattr(sensor, "scale_m", 10)) if sensor else 10
            cloudy_pct = int(getattr(sensor, "cloudy_pct", 30)) if sensor else 30
            composite = str(getattr(sensor, "composite", "median")) if sensor else "median"

            wavelengths_um = getattr(sensor, "wavelengths", None) if sensor else None
            if wavelengths_um is None:
                wavelengths_um = _infer_wavelengths_um(bands)
            if wavelengths_um is None:
                raise ModelError(
                    f"Cannot infer wavelengths for bands={bands}. Provide sensor.wavelengths explicitly (µm)."
                )
            wavelengths_um = [float(v) for v in wavelengths_um]

            if input_chw is None:
                x_chw, provider_meta = _fetch_gee_multiband_sr_chw(
                    provider,
                    spatial,
                    temporal,
                    collection=str(collection),
                    bands=bands,
                    scale_m=scale_m,
                    cloudy_pct=cloudy_pct,
                    composite=composite,
                    default_value=0.0,
                )
            else:
                # input_chw expected to be raw SR values (0..10000) in band order `bands`
                if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(bands):
                    raise ModelError(
                        f"input_chw must be CHW with {len(bands)} bands for DOFA, got {getattr(input_chw, 'shape', None)}"
                    )
                x_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0).astype(np.float32)
                provider_meta = {
                    "raw_chw_shape": tuple(x_chw.shape),
                    "input_override": True,
                }

            # Optional: inspect on-the-fly provider input
            from ..tools.inspection import checks_should_raise, maybe_inspect_chw

            check_meta.clear()
            report = maybe_inspect_chw(
                x_chw,
                sensor=sensor,
                name="provider_multiband_sr_chw",
                expected_channels=len(bands),
                value_range=(0.0, 1.0),
                fill_value=0.0,
                meta=check_meta,
            )
            if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
                raise ModelError(
                    "Provider input inspection failed: " + "; ".join(report.get("issues", []))
                )

            x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
            x_bchw = x_chw[None, ...].astype(np.float32)
        c = int(x_bchw.shape[1])
        if len(wavelengths_um) != c:
            raise ModelError(f"wavelengths length={len(wavelengths_um)} must equal channels C={c}.")

        # -----------------
        # Model + forward
        # -----------------
        model, mmeta = _load_dofa_model(variant=variant, device=device)
        dev = mmeta.get("device", device)
        tokens, pooled, tmeta = _dofa_forward_tokens_and_pooled(
            model, x_bchw, wavelengths_um=wavelengths_um, device=device
        )

        base_meta: dict[str, Any] = {
            "model": self.model_name,
            "type": "on_the_fly",
            "backend": backend_l,
            "variant": str(variant),
            "output_mode": output.mode,
            "device": str(device),
            "preprocess": {
                "strategy": "resize_to_224_bilinear",
                "resize_meta": resize_meta,
            },
            "input_channels": int(c),
            "wavelengths_um": list(map(float, wavelengths_um)),
            "input_size_hw": (int(x_bchw.shape[2]), int(x_bchw.shape[3])),
            "token_meta": tmeta,
            **check_meta,
            **mmeta,
            **provider_meta,
        }

        if output.mode == "pooled":
            base_meta["pooled_shape"] = tuple(pooled.shape)
            return Embedding(data=pooled.astype(np.float32), meta=base_meta)

        if output.mode == "grid":
            n, d = tokens.shape
            side = int(round(math.sqrt(n)))
            if side * side != n:
                raise ModelError(f"DOFA tokens N={n} not square; cannot reshape to grid.")
            grid = tokens.reshape(side, side, d).transpose(2, 0, 1).astype(np.float32)

            meta = {
                **base_meta,
                "grid_type": "vit_patch_tokens",
                "grid_shape": tuple(grid.shape),
                "grid_hw_tokens": (int(side), int(side)),
                "patch_size": int(getattr(model, "patch_size", 16)),
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

        backend_l = backend.lower().strip()
        if backend_l == "tensor":
            raise ModelError(
                "backend='tensor' batch inference requires get_embeddings_batch_from_inputs(...)."
            )
        provider = self._get_provider(backend)
        if temporal is None:
            raise ModelError("dofa provider backend requires TemporalSpec.range(start,end).")
        temporal.validate()
        if temporal.mode != "range":
            raise ModelError("dofa provider backend requires TemporalSpec.range in v0.1.")

        ss = sensor or self._default_sensor()
        collection = str(getattr(ss, "collection", "COPERNICUS/S2_SR_HARMONIZED"))
        bands = list(getattr(ss, "bands", _S2_SR_12_BANDS))
        scale_m = int(getattr(ss, "scale_m", 10))
        cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
        composite = str(getattr(ss, "composite", "median"))

        n = len(spatials)
        prefetched_raw: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            x_chw, _ = _fetch_gee_multiband_sr_chw(
                provider,
                sp,
                temporal,
                collection=collection,
                bands=bands,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
                default_value=0.0,
            )
            # get_embedding(input_chw=...) expects raw SR in [0..10000]
            raw = np.clip(x_chw * 10000.0, 0.0, 10000.0).astype(np.float32)
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

        raw_inputs: list[np.ndarray] = []
        for i, raw in enumerate(prefetched_raw):
            if raw is None:
                raise ModelError(f"Missing prefetched input at index={i} for dofa.")
            raw_inputs.append(raw)
        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            temporal=temporal,
            sensor=ss,
            model_config=model_config,
            output=output,
            backend=backend,
            device=device,
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
    ) -> list[Embedding]:
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        backend_l = backend.lower().strip()
        if backend_l == "tensor":
            return super().get_embeddings_batch_from_inputs(
                spatials=spatials,
                input_chws=input_chws,
                temporal=temporal,
                sensor=sensor,
                model_config=model_config,
                output=output,
                backend=backend,
                device=device,
            )
        self._get_provider(backend)
        if temporal is None:
            raise ModelError("dofa provider backend requires TemporalSpec.range(start,end).")
        temporal.validate()
        if temporal.mode != "range":
            raise ModelError("dofa provider backend requires TemporalSpec.range in v0.1.")

        ss = sensor or self._default_sensor()
        variant = _resolve_dofa_variant(model_config=model_config)
        bands = list(getattr(ss, "bands", _S2_SR_12_BANDS))
        wavelengths_um = getattr(ss, "wavelengths", None)
        if wavelengths_um is None:
            wavelengths_um = _infer_wavelengths_um(bands)
        if wavelengths_um is None:
            raise ModelError(
                f"Cannot infer wavelengths for bands={bands}. Provide sensor.wavelengths explicitly (µm)."
            )
        wavelengths_um = [float(v) for v in wavelengths_um]

        x_bchw_all: list[np.ndarray] = []
        resize_meta_all: list[dict[str, Any]] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(bands):
                raise ModelError(
                    f"input_chw must be CHW with {len(bands)} bands for DOFA, got "
                    f"{getattr(input_chw, 'shape', None)} at index={i}"
                )
            x_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0).astype(np.float32)
            x_chw, resize_meta = _resize_chw(x_chw, size=224)
            x_bchw_all.append(x_chw)
            resize_meta_all.append(resize_meta)

        model, mmeta = _load_dofa_model(variant=variant, device=device)
        dev = str(mmeta.get("device", device))
        infer_bs = self._resolve_infer_batch(dev)

        out: list[Embedding | None] = [None] * len(spatials)
        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            xb = np.stack(x_bchw_all[s0:s1], axis=0).astype(np.float32)
            tokens_bnd, pooled_bd, tmeta = _dofa_forward_tokens_and_pooled_batch(
                model,
                xb,
                wavelengths_um=wavelengths_um,
                device=dev,
            )
            for j in range(s1 - s0):
                i = s0 + j
                tokens = tokens_bnd[j]
                pooled = pooled_bd[j]
                base_meta: dict[str, Any] = {
                    "model": self.model_name,
                    "type": "on_the_fly",
                    "backend": backend_l,
                    "variant": str(variant),
                    "output_mode": output.mode,
                    "device": str(dev),
                    "preprocess": {
                        "strategy": "resize_to_224_bilinear",
                        "resize_meta": resize_meta_all[i],
                    },
                    "input_channels": int(x_bchw_all[i].shape[0]),
                    "wavelengths_um": list(map(float, wavelengths_um)),
                    "input_size_hw": (
                        int(x_bchw_all[i].shape[1]),
                        int(x_bchw_all[i].shape[2]),
                    ),
                    "token_meta": tmeta,
                    "batch_infer": True,
                    "input_override": True,
                    **mmeta,
                    "raw_chw_shape": tuple(input_chws[i].shape),
                }

                if output.mode == "pooled":
                    base_meta["pooled_shape"] = tuple(pooled.shape)
                    out[i] = Embedding(data=pooled.astype(np.float32), meta=base_meta)
                    continue

                if output.mode == "grid":
                    n_tok, d_tok = tokens.shape
                    side = int(round(math.sqrt(n_tok)))
                    if side * side != n_tok:
                        raise ModelError(
                            f"DOFA tokens N={n_tok} not square; cannot reshape to grid."
                        )
                    grid = tokens.reshape(side, side, d_tok).transpose(2, 0, 1).astype(np.float32)
                    meta = {
                        **base_meta,
                        "grid_type": "vit_patch_tokens",
                        "grid_shape": tuple(grid.shape),
                        "grid_hw_tokens": (int(side), int(side)),
                        "patch_size": int(getattr(model, "patch_size", 16)),
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
                        attrs=meta,
                    )
                    out[i] = Embedding(data=da, meta=meta)
                    continue

                raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError("dofa batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

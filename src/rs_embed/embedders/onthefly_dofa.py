# Implementation based on:
# Neural Plasticity-Inspired Foundation Model for Observing the Earth Across Optical and SAR Modalities
# arXiv 2024
# https://arxiv.org/abs/2403.15356

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import xarray as xr

from functools import lru_cache
from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers import ProviderBase
from .base import EmbedderBase
from .runtime_utils import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
    is_provider_backend,
    load_cached_with_device as _load_cached_with_device,
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


def _infer_wavelengths_um(bands: List[str]) -> Optional[List[float]]:
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
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    bands: List[str],
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
    default_value: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
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

    meta: Dict[str, Any] = {
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
    bands: List[str],
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
    default_value: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
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


@lru_cache(maxsize=4)
def _load_dofa_model_cached(variant: str, dev: str):
    try:
        import torch
        from torchgeo.models import (
            DOFABase16_Weights,
            DOFALarge16_Weights,
            dofa_base_patch16_224,
            dofa_large_patch16_224,
        )
    except Exception as e:
        raise ModelError("DOFA requires torchgeo. Install: pip install torchgeo") from e

    variant_l = str(variant).lower().strip()
    if variant_l in ("base", "b"):
        weights = DOFABase16_Weights.DOFA_MAE
        model = dofa_base_patch16_224(weights=weights)
    elif variant_l in ("large", "l"):
        weights = DOFALarge16_Weights.DOFA_MAE
        model = dofa_large_patch16_224(weights=weights)
    else:
        raise ModelError(
            f"Unknown DOFA variant='{variant}' (expected 'base' or 'large')."
        )

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
        "weights_url": str(weights.url),
        "weights_meta": (
            dict(weights.meta) if isinstance(weights.meta, dict) else str(weights.meta)
        ),
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
) -> Tuple[Any, Dict[str, Any]]:
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
    wavelengths_um: List[float],
    *,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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

        patch_tokens = (
            xseq[:, 1:, :][0].detach().float().cpu().numpy().astype(np.float32)
        )  # [N,D]

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
    wavelengths_um: List[float],
    *,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
            pooled = (
                xseq[:, 0].detach().float().cpu().numpy().astype(np.float32)
            )  # [B,D]
            norm_applied = "norm(cls)"

        patch_tokens = (
            xseq[:, 1:, :].detach().float().cpu().numpy().astype(np.float32)
        )  # [B,N,D]

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
    DOFA (TorchGeo) embeddings.

    - backend="provider"/"auto": ROI -> S2 SR -> resize to 224 -> DOFA -> pooled/grid
    - backend="tensor": sensor.data (CHW/BCHW) -> resize to 224 -> DOFA

    Output:
      - OutputSpec.pooled(): (D,)
      - OutputSpec.grid():   (D, Ht, Wt) token grid, usually 14x14 for 224/patch16
    """

    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 64

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider", "tensor"],
            "inputs": {
                "provider_default": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": _S2_SR_12_BANDS,
                    "wavelengths_um": "auto for S2 bands",
                }
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "variant": "base",
                "image_size": 224,
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "preprocess": "resize_to_224_bilinear",
            },
        }

    _allow_auto_backend = False

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
                "RS_EMBED_DOFA_FETCH_WORKERS", str(DOFAEmbedder.DEFAULT_FETCH_WORKERS)
            )
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
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        backend_l = backend.lower().strip()
        variant = getattr(sensor, "variant", "base") if sensor else "base"
        image_size = 224

        # For optional on-the-fly input inspection
        check_meta: Dict[str, Any] = {}

        # -----------------
        # Build input + wavelengths
        # -----------------
        if backend_l == "tensor":
            if sensor is None or not hasattr(sensor, "data"):
                raise ModelError(
                    "backend='tensor' requires sensor.data as CHW or BCHW."
                )
            x = sensor.data
            try:
                import torch

                if torch.is_tensor(x):
                    x = x.detach().cpu().numpy()
            except Exception:
                pass

            x = np.asarray(x)
            if x.ndim == 3:
                x_chw = x.astype(np.float32)
                x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
                x_bchw = x_chw[None, ...]
            elif x.ndim == 4:
                if x.shape[0] != 1:
                    raise ModelError("v0.1: tensor backend expects B=1.")
                x_chw = x[0].astype(np.float32)
                x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
                x_bchw = x_chw[None, ...]
            else:
                raise ModelError(f"Expected CHW or BCHW, got {x.shape}")

            wavelengths_um = getattr(sensor, "wavelengths", None)
            if wavelengths_um is None:
                bands = (
                    list(getattr(sensor, "bands", []))
                    if hasattr(sensor, "bands")
                    else []
                )
                if bands:
                    wavelengths_um = _infer_wavelengths_um(bands)
            if wavelengths_um is None:
                raise ModelError(
                    "DOFA requires wavelengths (µm) per channel. "
                    "Provide sensor.wavelengths=[...] or (for S2) provide sensor.bands to infer."
                )
            wavelengths_um = [float(v) for v in wavelengths_um]

            provider_meta = {"backend_tensor": True}

        elif is_provider_backend(backend_l, allow_auto=False):
            if temporal is None:
                raise ModelError(
                    "dofa provider backend requires TemporalSpec.range(start,end)."
                )
            temporal.validate()
            if temporal.mode != "range":
                raise ModelError(
                    "dofa provider backend requires TemporalSpec.range in v0.1."
                )

            # overrides
            collection = (
                getattr(sensor, "collection", "COPERNICUS/S2_SR_HARMONIZED")
                if sensor
                else "COPERNICUS/S2_SR_HARMONIZED"
            )
            bands = (
                list(getattr(sensor, "bands", _S2_SR_12_BANDS))
                if sensor
                else list(_S2_SR_12_BANDS)
            )
            scale_m = int(getattr(sensor, "scale_m", 10)) if sensor else 10
            cloudy_pct = int(getattr(sensor, "cloudy_pct", 30)) if sensor else 30
            composite = (
                str(getattr(sensor, "composite", "median")) if sensor else "median"
            )

            wavelengths_um = getattr(sensor, "wavelengths", None) if sensor else None
            if wavelengths_um is None:
                wavelengths_um = _infer_wavelengths_um(bands)
            if wavelengths_um is None:
                raise ModelError(
                    f"Cannot infer wavelengths for bands={bands}. Provide sensor.wavelengths explicitly (µm)."
                )
            wavelengths_um = [float(v) for v in wavelengths_um]

            if input_chw is None:
                provider = self._get_provider(backend_l)

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
                x_chw = np.clip(
                    input_chw.astype(np.float32) / 10000.0, 0.0, 1.0
                ).astype(np.float32)
                provider_meta = {
                    "raw_chw_shape": tuple(x_chw.shape),
                    "input_override": True,
                }

            # Optional: inspect on-the-fly provider input
            from ..tools.inspection import maybe_inspect_chw, checks_should_raise

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
            if (
                report is not None
                and (not report.get("ok", True))
                and checks_should_raise(sensor)
            ):
                raise ModelError(
                    "Provider input inspection failed: "
                    + "; ".join(report.get("issues", []))
                )

            x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
            x_bchw = x_chw[None, ...].astype(np.float32)

        else:
            raise ModelError("dofa supports a provider backend or 'tensor' only.")

        c = int(x_bchw.shape[1])
        if len(wavelengths_um) != c:
            raise ModelError(
                f"wavelengths length={len(wavelengths_um)} must equal channels C={c}."
            )

        # -----------------
        # Model + forward
        # -----------------
        model, mmeta = _load_dofa_model(variant=variant, device=device)
        dev = mmeta.get("device", device)
        tokens, pooled, tmeta = _dofa_forward_tokens_and_pooled(
            model, x_bchw, wavelengths_um=wavelengths_um, device=device
        )

        base_meta: Dict[str, Any] = {
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
                raise ModelError(
                    f"DOFA tokens N={n} not square; cannot reshape to grid."
                )
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
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []

        backend_l = backend.lower().strip()
        if not is_provider_backend(backend_l, allow_auto=False):
            # tensor path stays sequential in v0.1
            return super().get_embeddings_batch(
                spatials=spatials,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
            )
        if temporal is None:
            raise ModelError(
                "dofa provider backend requires TemporalSpec.range(start,end)."
            )
        temporal.validate()
        if temporal.mode != "range":
            raise ModelError(
                "dofa provider backend requires TemporalSpec.range in v0.1."
            )

        ss = sensor or self._default_sensor()
        collection = str(getattr(ss, "collection", "COPERNICUS/S2_SR_HARMONIZED"))
        bands = list(getattr(ss, "bands", _S2_SR_12_BANDS))
        scale_m = int(getattr(ss, "scale_m", 10))
        cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
        composite = str(getattr(ss, "composite", "median"))

        provider = self._get_provider(backend_l)
        n = len(spatials)
        prefetched_raw: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
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

        raw_inputs: List[np.ndarray] = []
        for i, raw in enumerate(prefetched_raw):
            if raw is None:
                raise ModelError(f"Missing prefetched input at index={i} for dofa.")
            raw_inputs.append(raw)
        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            temporal=temporal,
            sensor=ss,
            output=output,
            backend=backend,
            device=device,
        )

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
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
        if not is_provider_backend(backend_l, allow_auto=False):
            return super().get_embeddings_batch_from_inputs(
                spatials=spatials,
                input_chws=input_chws,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
            )
        if temporal is None:
            raise ModelError(
                "dofa provider backend requires TemporalSpec.range(start,end)."
            )
        temporal.validate()
        if temporal.mode != "range":
            raise ModelError(
                "dofa provider backend requires TemporalSpec.range in v0.1."
            )

        ss = sensor or self._default_sensor()
        variant = getattr(ss, "variant", "base")
        bands = list(getattr(ss, "bands", _S2_SR_12_BANDS))
        wavelengths_um = getattr(ss, "wavelengths", None)
        if wavelengths_um is None:
            wavelengths_um = _infer_wavelengths_um(bands)
        if wavelengths_um is None:
            raise ModelError(
                f"Cannot infer wavelengths for bands={bands}. Provide sensor.wavelengths explicitly (µm)."
            )
        wavelengths_um = [float(v) for v in wavelengths_um]

        x_bchw_all: List[np.ndarray] = []
        resize_meta_all: List[Dict[str, Any]] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(bands):
                raise ModelError(
                    f"input_chw must be CHW with {len(bands)} bands for DOFA, got "
                    f"{getattr(input_chw, 'shape', None)} at index={i}"
                )
            x_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0).astype(
                np.float32
            )
            x_chw, resize_meta = _resize_chw(x_chw, size=224)
            x_bchw_all.append(x_chw)
            resize_meta_all.append(resize_meta)

        model, mmeta = _load_dofa_model(variant=variant, device=device)
        dev = str(mmeta.get("device", device))
        infer_bs = self._resolve_infer_batch(dev)

        out: List[Optional[Embedding]] = [None] * len(spatials)
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
                base_meta: Dict[str, Any] = {
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
                    grid = (
                        tokens.reshape(side, side, d_tok)
                        .transpose(2, 0, 1)
                        .astype(np.float32)
                    )
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

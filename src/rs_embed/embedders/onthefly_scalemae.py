from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers.base import ProviderBase
from .base import EmbedderBase
from .runtime_utils import (
    get_cached_provider,
    is_provider_backend,
    load_cached_with_device as _load_cached_with_device,
)

from ._vit_mae_utils import (
    fetch_s2_rgb_u8_from_provider,
    resize_rgb_u8,
    temporal_to_range,
    pool_from_tokens,
    tokens_to_grid_dhw,
    base_meta,
    ensure_torch,
    rgb_u8_to_tensor_clipnorm,
)


@lru_cache(maxsize=8)
def _load_scalemae_cached(model_id: str, dev: str):
    ensure_torch()

    try:
        from rshf.scalemae import ScaleMAE  # type: ignore
    except Exception as e:
        raise ModelError(
            "ScaleMAE requires rshf with rshf.scalemae.ScaleMAE. Try: pip install -U rshf"
        ) from e

    model = ScaleMAE.from_pretrained(model_id)
    try:
        model = model.to(dev).eval()
    except Exception:
        pass

    meta = {"model_id": model_id, "device": dev}
    return model, meta


def _load_scalemae(model_id: str, device: str = "auto"):
    loaded, _dev = _load_cached_with_device(
        _load_scalemae_cached, device=device, model_id=model_id
    )
    return loaded


def _infer_patch_size(model) -> int:
    """
    Try best-effort to infer ViT patch size from common attributes.
    """
    # common: model.patch_size
    ps = getattr(model, "patch_size", None)
    if isinstance(ps, (int, float)):
        return int(ps)

    # common: model.patch_embed.patch_size (int or tuple)
    pe = getattr(model, "patch_embed", None)
    if pe is not None:
        ps2 = getattr(pe, "patch_size", None)
        if isinstance(ps2, (int, float)):
            return int(ps2)
        if isinstance(ps2, (tuple, list)) and len(ps2) >= 1:
            return int(ps2[0])

    # some timm variants: model.patch_embed.patch_size[0]
    # fallback: ViT-L/16 is common for ScaleMAE
    return 16


def _call_with_patch_size(fn, x, *, patch_size: int, input_res):
    """
    Call forward/forward_features with compatible signature across rshf versions.
    Tries kwargs then positional.
    """
    import inspect

    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None

    # Try kwargs if signature seems to accept them
    if sig is not None:
        params = sig.parameters
        kw = {}
        if "patch_size" in params:
            kw["patch_size"] = patch_size
        if "input_res" in params:
            kw["input_res"] = input_res
        try:
            return fn(x, **kw)
        except TypeError:
            pass

    # Positional fallbacks (your current error shows patch_size is positional)
    # Common patterns:
    #   fn(x, patch_size, input_res)
    #   fn(x, patch_size=..., input_res=...)
    #   fn(x, input_res, patch_size)  (rare)
    try:
        return fn(x, patch_size, input_res)
    except TypeError:
        try:
            return fn(x, patch_size=patch_size, input_res=input_res)
        except TypeError:
            try:
                return fn(x, input_res, patch_size)
            except TypeError as e:
                raise ModelError(
                    f"ScaleMAE call failed even with patch_size/input_res: {e}"
                ) from e


def _scalemae_forward_tokens_or_vec(
    model,
    rgb_u8: np.ndarray,
    *,
    image_size: int,
    device: str,
    input_res_m: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Your rshf ScaleMAE requires:
      - input_res: 1D tensor
      - patch_size: required positional argument in forward() (and sometimes in forward_features()).

    Prefer forward_features() to get tokens [N,D]; fallback to forward().
    """
    ensure_torch()
    import torch

    x = rgb_u8_to_tensor_clipnorm(rgb_u8, image_size).to(device)  # [B,3,H,W]
    input_res = torch.tensor(
        [float(input_res_m)], dtype=torch.float32, device=device
    )  # 1D
    patch_size = _infer_patch_size(model)

    with torch.no_grad():
        ff = getattr(model, "forward_features", None)
        if callable(ff):
            out = _call_with_patch_size(
                ff, x, patch_size=patch_size, input_res=input_res
            )
            out0 = out[0] if isinstance(out, (tuple, list)) else out

            if hasattr(out0, "ndim") and out0.ndim == 3:  # [B,N,D]
                toks = out0
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "tokens_forward_features",
                    "input_res_m": float(input_res_m),
                    "used_patch_size": int(patch_size),
                    "tokens_shape": tuple(toks.shape),
                }

            if hasattr(out0, "ndim") and out0.ndim == 2:  # [B,D]
                v = out0
                return v[0].detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "pooled_forward_features",
                    "input_res_m": float(input_res_m),
                    "used_patch_size": int(patch_size),
                    "vec_shape": tuple(v.shape),
                }

            if hasattr(out0, "ndim") and out0.ndim == 4:  # [B,C,H,W] -> tokens
                b, c, h, w = out0.shape
                toks = out0.permute(0, 2, 3, 1).reshape(b, h * w, c)
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "tokens_from_feature_map",
                    "input_res_m": float(input_res_m),
                    "used_patch_size": int(patch_size),
                    "feature_map_hw": (int(h), int(w)),
                    "tokens_shape": tuple(toks.shape),
                }

            raise ModelError(
                f"ScaleMAE forward_features returned unsupported: {type(out0)} {getattr(out0, 'shape', None)}"
            )

        # fallback: forward (must pass patch_size + input_res)
        out = _call_with_patch_size(
            model, x, patch_size=patch_size, input_res=input_res
        )
        out0 = out[0] if isinstance(out, (tuple, list)) else out

        if hasattr(out0, "ndim") and out0.ndim == 3:
            return out0[0].detach().float().cpu().numpy().astype(np.float32), {
                "tokens_kind": "tokens_forward",
                "input_res_m": float(input_res_m),
                "used_patch_size": int(patch_size),
                "tokens_shape": tuple(out0.shape),
            }
        if hasattr(out0, "ndim") and out0.ndim == 2:
            return out0[0].detach().float().cpu().numpy().astype(np.float32), {
                "tokens_kind": "pooled_forward",
                "input_res_m": float(input_res_m),
                "used_patch_size": int(patch_size),
                "vec_shape": tuple(out0.shape),
            }

        raise ModelError(
            "ScaleMAE: cannot obtain tokens or pooled vector from this model."
        )


def _scalemae_forward_tokens_or_vec_batch(
    model,
    rgb_u8_batch: List[np.ndarray],
    *,
    image_size: int,
    device: str,
    input_res_m: float,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Batch variant that returns one output array per input item."""
    if not rgb_u8_batch:
        return [], {"tokens_kind": "empty_batch"}

    ensure_torch()
    import torch

    xs = [rgb_u8_to_tensor_clipnorm(rgb_u8, image_size) for rgb_u8 in rgb_u8_batch]
    xb = torch.cat(xs, dim=0).to(device)  # [B,3,H,W]
    input_res = torch.full(
        (len(rgb_u8_batch),), float(input_res_m), dtype=torch.float32, device=device
    )
    patch_size = _infer_patch_size(model)

    def _to_list(arr: np.ndarray) -> List[np.ndarray]:
        return [arr[i].astype(np.float32, copy=False) for i in range(arr.shape[0])]

    with torch.inference_mode():
        ff = getattr(model, "forward_features", None)
        if callable(ff):
            out = _call_with_patch_size(
                ff, xb, patch_size=patch_size, input_res=input_res
            )
            out0 = out[0] if isinstance(out, (tuple, list)) else out

            if hasattr(out0, "ndim") and out0.ndim == 3:  # [B,N,D]
                toks = out0.detach().float().cpu().numpy().astype(np.float32)
                return _to_list(toks), {
                    "tokens_kind": "tokens_forward_features",
                    "input_res_m": float(input_res_m),
                    "used_patch_size": int(patch_size),
                    "batch_shape": tuple(toks.shape),
                }

            if hasattr(out0, "ndim") and out0.ndim == 2:  # [B,D]
                vec = out0.detach().float().cpu().numpy().astype(np.float32)
                return _to_list(vec), {
                    "tokens_kind": "pooled_forward_features",
                    "input_res_m": float(input_res_m),
                    "used_patch_size": int(patch_size),
                    "batch_shape": tuple(vec.shape),
                }

            if hasattr(out0, "ndim") and out0.ndim == 4:  # [B,C,H,W] -> tokens
                b, c, h, w = out0.shape
                toks = out0.permute(0, 2, 3, 1).reshape(b, h * w, c)
                toks_np = toks.detach().float().cpu().numpy().astype(np.float32)
                return _to_list(toks_np), {
                    "tokens_kind": "tokens_from_feature_map",
                    "input_res_m": float(input_res_m),
                    "used_patch_size": int(patch_size),
                    "feature_map_hw": (int(h), int(w)),
                    "batch_shape": tuple(toks_np.shape),
                }

            raise ModelError(
                f"ScaleMAE forward_features returned unsupported: {type(out0)} {getattr(out0, 'shape', None)}"
            )

        out = _call_with_patch_size(
            model, xb, patch_size=patch_size, input_res=input_res
        )
        out0 = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(out0, "ndim") and out0.ndim == 3:
            toks = out0.detach().float().cpu().numpy().astype(np.float32)
            return _to_list(toks), {
                "tokens_kind": "tokens_forward",
                "input_res_m": float(input_res_m),
                "used_patch_size": int(patch_size),
                "batch_shape": tuple(toks.shape),
            }
        if hasattr(out0, "ndim") and out0.ndim == 2:
            vec = out0.detach().float().cpu().numpy().astype(np.float32)
            return _to_list(vec), {
                "tokens_kind": "pooled_forward",
                "input_res_m": float(input_res_m),
                "used_patch_size": int(patch_size),
                "batch_shape": tuple(vec.shape),
            }

        raise ModelError(
            "ScaleMAE: cannot obtain tokens or pooled vector from this model."
        )


@register("scalemae")
class ScaleMAERGBEmbedder(EmbedderBase):
    """
    ScaleMAE on-the-fly embedding from Sentinel-2 RGB patch (provider backend).

    Strategy aligned via _vit_mae_utils:
      - pooled: pool patch tokens by OutputSpec.pooling (exclude CLS if present)
      - grid: patch token grid (exclude CLS if present)
      - scale: uses sensor.scale_m as input_res_m (required by your rshf)
    """

    DEFAULT_MODEL_ID = "MVRL/scalemae-vitlarge-800"
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 32

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["provider"],
            "model_id_default": self.DEFAULT_MODEL_ID,
            "image_size": self.DEFAULT_IMAGE_SIZE,
            "inputs": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": ["B4", "B3", "B2"],
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "model_id": self.DEFAULT_MODEL_ID,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
            },
        }

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderBase] = {}

    def _get_provider(self, backend: str) -> ProviderBase:
        return get_cached_provider(
            self._providers,
            backend=backend,
            allow_auto=True,
        )

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
                "RS_EMBED_SCALEMAE_FETCH_WORKERS",
                str(ScaleMAERGBEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default_bs = (
            ScaleMAERGBEmbedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else ScaleMAERGBEmbedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_SCALEMAE_BATCH_SIZE", str(default_bs)))
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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("scalemae_rgb expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        model_id = os.environ.get("RS_EMBED_SCALEMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(
            os.environ.get("RS_EMBED_SCALEMAE_IMG", str(self.DEFAULT_IMAGE_SIZE))
        )

        t = temporal_to_range(temporal)
        # Fetch RGB patch (optionally reuse pre-fetched raw patch)
        if input_chw is None:
            rgb_u8 = fetch_s2_rgb_u8_from_provider(
                spatial=spatial,
                temporal=t,
                sensor=sensor,
                out_size=image_size,
                provider=self._get_provider(backend),
            )
        else:
            # input_chw expected to be raw S2 SR values in band order (B4,B3,B2)
            if input_chw.ndim != 3 or input_chw.shape[0] != 3:
                raise ModelError(
                    "input_chw must be CHW with 3 bands for scalemae_rgb, got {shape}".format(
                        shape=getattr(input_chw, "shape", None),
                    )
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            rgb_u8 = resize_rgb_u8(rgb_u8, image_size)

        model, wmeta = _load_scalemae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        out, extra = _scalemae_forward_tokens_or_vec(
            model,
            rgb_u8,
            image_size=image_size,
            device=dev,
            input_res_m=float(sensor.scale_m),
        )

        meta = base_meta(
            model_name=self.model_name,
            hf_id=model_id,
            backend=str(backend).lower(),
            image_size=image_size,
            sensor=sensor,
            temporal=t,
            source=sensor.collection,
            extra={
                "used_scale_m": float(sensor.scale_m),
                **extra,
                "out_shape": tuple(out.shape),
            },
        )

        if output.mode == "pooled":
            if out.ndim == 2:
                vec, cls_removed = pool_from_tokens(out, output.pooling)
                meta.update(
                    {
                        "pooling": f"patch_{output.pooling}",
                        "cls_removed": bool(cls_removed),
                    }
                )
                return Embedding(data=vec, meta=meta)

            if out.ndim == 1:
                meta.update({"pooling": "model_pooled", "cls_removed": False})
                return Embedding(data=out.astype(np.float32), meta=meta)

            raise ModelError(f"Unexpected shape for pooled: {out.shape}")

        if output.mode == "grid":
            if out.ndim != 2:
                raise ModelError(
                    "grid output requires token sequence [N,D]. "
                    f"Got {out.shape} (tokens_kind={meta.get('tokens_kind')})."
                )

            grid, (h, w), cls_removed = tokens_to_grid_dhw(out)
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
            raise ModelError("scalemae_rgb expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        model_id = os.environ.get("RS_EMBED_SCALEMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(
            os.environ.get("RS_EMBED_SCALEMAE_IMG", str(self.DEFAULT_IMAGE_SIZE))
        )
        t = temporal_to_range(temporal)

        provider = self._get_provider(backend)
        n = len(spatials)
        rgb_u8_all: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
            rgb_u8 = fetch_s2_rgb_u8_from_provider(
                spatial=sp,
                temporal=t,
                sensor=sensor,
                out_size=image_size,
                provider=provider,
            )
            return i, rgb_u8

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                ii, rgb = _fetch_one(i, sp)
                rgb_u8_all[ii] = rgb
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, rgb = fut.result()
                    rgb_u8_all[i] = rgb

        model, wmeta = _load_scalemae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))

        out: List[Optional[Embedding]] = [None] * len(spatials)
        xr_mod = None
        if output.mode == "grid":
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError(
                    "grid output requires xarray. Install: pip install xarray"
                ) from e

        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            chunk_rgb = []
            chunk_idx = []
            for i in range(s0, s1):
                rgb_u8 = rgb_u8_all[i]
                if rgb_u8 is None:
                    raise ModelError(
                        f"Missing prefetched patch at index={i} for scalemae_rgb."
                    )
                chunk_idx.append(i)
                chunk_rgb.append(rgb_u8)

            try:
                chunk_outs, chunk_extra = _scalemae_forward_tokens_or_vec_batch(
                    model,
                    chunk_rgb,
                    image_size=image_size,
                    device=dev,
                    input_res_m=float(sensor.scale_m),
                )
            except Exception:
                chunk_outs = []
                chunk_extra = {}
                for rgb_u8 in chunk_rgb:
                    o1, e1 = _scalemae_forward_tokens_or_vec(
                        model,
                        rgb_u8,
                        image_size=image_size,
                        device=dev,
                        input_res_m=float(sensor.scale_m),
                    )
                    chunk_outs.append(o1)
                    chunk_extra = e1

            if len(chunk_outs) != len(chunk_idx):
                raise ModelError(
                    f"scalemae_rgb batch output mismatch: {len(chunk_outs)} != {len(chunk_idx)}"
                )

            for j, o in enumerate(chunk_outs):
                i = chunk_idx[j]
                meta = base_meta(
                    model_name=self.model_name,
                    hf_id=model_id,
                    backend=str(backend).lower(),
                    image_size=image_size,
                    sensor=sensor,
                    temporal=t,
                    source=sensor.collection,
                    extra={
                        "used_scale_m": float(sensor.scale_m),
                        **chunk_extra,
                        "out_shape": tuple(o.shape),
                        "batch_infer": True,
                    },
                )

                if output.mode == "pooled":
                    if o.ndim == 2:
                        vec, cls_removed = pool_from_tokens(o, output.pooling)
                        meta.update(
                            {
                                "pooling": f"patch_{output.pooling}",
                                "cls_removed": bool(cls_removed),
                            }
                        )
                        out[i] = Embedding(data=vec, meta=meta)
                    elif o.ndim == 1:
                        meta.update({"pooling": "model_pooled", "cls_removed": False})
                        out[i] = Embedding(data=o.astype(np.float32), meta=meta)
                    else:
                        raise ModelError(f"Unexpected shape for pooled: {o.shape}")
                    continue

                if output.mode == "grid":
                    if o.ndim != 2:
                        raise ModelError(
                            "grid output requires token sequence [N,D]. "
                            f"Got {o.shape} (tokens_kind={meta.get('tokens_kind')})."
                        )
                    grid, (h, w), cls_removed = tokens_to_grid_dhw(o)
                    meta.update(
                        {
                            "grid_hw": (h, w),
                            "grid_kind": "patch_tokens",
                            "cls_removed": bool(cls_removed),
                        }
                    )
                    assert xr_mod is not None
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
                    continue

                raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError(
                "scalemae_rgb batch inference produced incomplete outputs."
            )
        return [e for e in out if e is not None]

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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("scalemae_rgb expects a provider backend (or 'auto').")
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        if sensor is None:
            sensor = self._default_sensor()

        model_id = os.environ.get("RS_EMBED_SCALEMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(
            os.environ.get("RS_EMBED_SCALEMAE_IMG", str(self.DEFAULT_IMAGE_SIZE))
        )
        t = temporal_to_range(temporal)
        model, wmeta = _load_scalemae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))

        rgb_u8_all: List[np.ndarray] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or input_chw.shape[0] != 3:
                raise ModelError(
                    "input_chw must be CHW with 3 bands for scalemae_rgb, got "
                    f"{getattr(input_chw, 'shape', None)} at index={i}"
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            rgb_u8_all.append(resize_rgb_u8(rgb_u8, image_size))

        out: List[Optional[Embedding]] = [None] * len(spatials)
        xr_mod = None
        if output.mode == "grid":
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError(
                    "grid output requires xarray. Install: pip install xarray"
                ) from e

        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            chunk_idx = list(range(s0, s1))
            chunk_rgb = [rgb_u8_all[i] for i in chunk_idx]
            chunk_outs, chunk_extra = _scalemae_forward_tokens_or_vec_batch(
                model,
                chunk_rgb,
                image_size=image_size,
                device=dev,
                input_res_m=float(sensor.scale_m),
            )
            if len(chunk_outs) != len(chunk_idx):
                raise ModelError(
                    f"scalemae_rgb prefetched batch output mismatch: {len(chunk_outs)} != {len(chunk_idx)}"
                )

            for j, o in enumerate(chunk_outs):
                i = chunk_idx[j]
                meta = base_meta(
                    model_name=self.model_name,
                    hf_id=model_id,
                    backend=str(backend).lower(),
                    image_size=image_size,
                    sensor=sensor,
                    temporal=t,
                    source=sensor.collection,
                    extra={
                        "used_scale_m": float(sensor.scale_m),
                        **chunk_extra,
                        "out_shape": tuple(o.shape),
                        "batch_infer": True,
                        "input_override": True,
                    },
                )

                if output.mode == "pooled":
                    if o.ndim == 2:
                        vec, cls_removed = pool_from_tokens(o, output.pooling)
                        meta.update(
                            {
                                "pooling": f"patch_{output.pooling}",
                                "cls_removed": bool(cls_removed),
                            }
                        )
                        out[i] = Embedding(data=vec, meta=meta)
                    elif o.ndim == 1:
                        meta.update({"pooling": "model_pooled", "cls_removed": False})
                        out[i] = Embedding(data=o.astype(np.float32), meta=meta)
                    else:
                        raise ModelError(f"Unexpected shape for pooled: {o.shape}")
                    continue

                if output.mode == "grid":
                    if o.ndim != 2:
                        raise ModelError(
                            "grid output requires token sequence [N,D]. "
                            f"Got {o.shape} (tokens_kind={meta.get('tokens_kind')})."
                        )
                    grid, (h, w), cls_removed = tokens_to_grid_dhw(o)
                    meta.update(
                        {
                            "grid_hw": (h, w),
                            "grid_kind": "patch_tokens",
                            "cls_removed": bool(cls_removed),
                        }
                    )
                    assert xr_mod is not None
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
                    continue

                raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError(
                "scalemae_rgb prefetched batch inference produced incomplete outputs."
            )
        return [e for e in out if e is not None]

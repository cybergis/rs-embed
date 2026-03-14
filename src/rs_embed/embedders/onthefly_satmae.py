from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ._vit_mae_utils import (
    base_meta,
    ensure_torch,
    fetch_s2_rgb_u8_from_provider,
    maybe_use_model_transform,
    pool_from_tokens,
    resize_rgb_u8,
    rgb_u8_to_tensor_clipnorm,
    temporal_to_range,
    tokens_to_grid_dhw,
)
from .base import EmbedderBase
from .runtime_utils import (
    is_provider_backend,
)
from .runtime_utils import (
    load_cached_with_device as _load_cached_with_device,
)


@lru_cache(maxsize=8)
def _load_satmae_cached(model_id: str, dev: str):
    ensure_torch()

    try:
        from rshf.satmae import SatMAE
    except Exception as e:
        raise ModelError("SatMAE requires rshf. Install: pip install rshf") from e

    model = SatMAE.from_pretrained(model_id)
    try:
        model = model.to(dev).eval()
    except Exception as _e:
        pass

    meta = {"model_id": model_id, "device": dev}
    return model, meta

def _load_satmae(model_id: str, device: str = "auto"):
    loaded, _dev = _load_cached_with_device(_load_satmae_cached, device=device, model_id=model_id)
    return loaded

def _satmae_forward_tokens(
    model, rgb_u8: np.ndarray, *, image_size: int, device: str
) -> np.ndarray:
    """
    Return tokens [N,D] via forward_encoder(mask_ratio=0.0).
    """
    return _satmae_forward_tokens_batch(
        model,
        [rgb_u8],
        image_size=image_size,
        device=device,
    )[0]

def _satmae_forward_tokens_batch(
    model,
    rgb_u8_batch: list[np.ndarray],
    *,
    image_size: int,
    device: str,
) -> list[np.ndarray]:
    """
    Batch version of forward_encoder.
    Returns one [N,D] float32 token array per input image.
    """
    if not rgb_u8_batch:
        return []

    ensure_torch()
    import torch

    xs = []
    for rgb_u8 in rgb_u8_batch:
        # prefer wrapper transform()
        x = maybe_use_model_transform(model, rgb_u8, image_size)
        if x is None:
            # fallback: generic preprocessing (CLIP norm)
            x = rgb_u8_to_tensor_clipnorm(rgb_u8, image_size)
        if x.ndim != 4 or x.shape[0] != 1:
            raise ModelError(
                f"SatMAE transform returned shape={tuple(x.shape)}; expected [1,C,H,W]."
            )
        xs.append(x[0])

    xb = torch.stack(xs, dim=0).to(device)

    fe = getattr(model, "forward_encoder", None)
    if not callable(fe):
        raise ModelError("SatMAE wrapper does not expose forward_encoder(). Update rshf.")

    with torch.no_grad():
        out = fe(xb, mask_ratio=0.0)
        toks = out[0] if isinstance(out, (tuple, list)) else out  # [B,N,D]
        if toks.ndim != 3 or int(toks.shape[0]) != len(rgb_u8_batch):
            raise ModelError(
                f"SatMAE forward_encoder returned {tuple(toks.shape)}; "
                f"expected [B,N,D] with B={len(rgb_u8_batch)}."
            )
        out_np = toks.detach().float().cpu().numpy().astype(np.float32)
        return [out_np[i] for i in range(out_np.shape[0])]

@register("satmae")
class SatMAERGBEmbedder(EmbedderBase):
    """
    SatMAE (ViT/MAE) on-the-fly embeddings from Sentinel-2 RGB patch (provider backend).

    Strategy aligned via _vit_mae_utils:
      - pooled: pool patch tokens by OutputSpec.pooling (exclude CLS if present)
      - grid: patch token grid (exclude CLS if present)
    """

    DEFAULT_MODEL_ID = "MVRL/satmae-vitlarge-fmow-pretrain-800"
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 32

    def describe(self) -> dict[str, Any]:
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

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return SensorSpec(
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=("B4", "B3", "B2"),
            scale_m=10,
            cloudy_pct=30,
            composite="median",
        )

    def _resolve_fetch_workers(self, n_items: int) -> int:
        v = int(os.environ.get("RS_EMBED_SATMAE_FETCH_WORKERS", str(self.DEFAULT_FETCH_WORKERS)))
        return max(1, min(int(n_items), v))

    def _resolve_infer_batch(self, dev: str) -> int:
        default_bs = (
            self.DEFAULT_BATCH_CUDA if str(dev).startswith("cuda") else self.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_SATMAE_BATCH_SIZE", str(default_bs)))
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
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satmae_rgb expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        model_id = os.environ.get("RS_EMBED_SATMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SATMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))

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
                    "input_chw must be CHW with 3 bands for satmae_rgb, got {shape}".format(
                        shape=getattr(input_chw, "shape", None),
                    )
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            rgb_u8 = resize_rgb_u8(rgb_u8, image_size)

        model, wmeta = _load_satmae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        tokens = _satmae_forward_tokens(model, rgb_u8, image_size=image_size, device=dev)  # [N,D]

        meta = base_meta(
            model_name=self.model_name,
            hf_id=model_id,
            backend=str(backend).lower(),
            image_size=image_size,
            sensor=sensor,
            temporal=t,
            source=sensor.collection,
            extra={
                "tokens_kind": "tokens_forward_encoder",
                "tokens_shape": tuple(tokens.shape),
            },
        )

        if output.mode == "pooled":
            vec, cls_removed = pool_from_tokens(tokens, output.pooling)
            meta.update({"pooling": f"patch_{output.pooling}", "cls_removed": bool(cls_removed)})
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            grid, (h, w), cls_removed = tokens_to_grid_dhw(tokens)
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
            raise ModelError("satmae_rgb expects a provider backend (or 'auto').")
        if not spatials:
            return []

        if sensor is None:
            sensor = self._default_sensor()

        model_id = os.environ.get("RS_EMBED_SATMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SATMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        t = temporal_to_range(temporal)

        provider = self._get_provider(backend)
        n = len(spatials)
        rgb_u8_all: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            rgb = fetch_s2_rgb_u8_from_provider(
                spatial=sp,
                temporal=t,
                sensor=sensor,
                out_size=image_size,
                provider=provider,
            )
            return i, rgb

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                _, rgb = _fetch_one(i, sp)
                rgb_u8_all[i] = rgb
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, rgb = fut.result()
                    rgb_u8_all[i] = rgb

        for i, x in enumerate(rgb_u8_all):
            if x is None:
                raise ModelError(f"Missing fetched patch at index={i}; batch fetch failed.")

        model, wmeta = _load_satmae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))

        out: list[Embedding | None] = [None] * n
        want_grid = output.mode == "grid"
        xr_mod = None
        if want_grid:
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            rgb_batch = [x for x in rgb_u8_all[s0:s1] if x is not None]
            toks_batch = _satmae_forward_tokens_batch(
                model,
                rgb_batch,
                image_size=image_size,
                device=dev,
            )
            for j, tokens in enumerate(toks_batch):
                i = s0 + j
                meta = base_meta(
                    model_name=self.model_name,
                    hf_id=model_id,
                    backend=str(backend).lower(),
                    image_size=image_size,
                    sensor=sensor,
                    temporal=t,
                    source=sensor.collection,
                    extra={
                        "tokens_kind": "tokens_forward_encoder",
                        "tokens_shape": tuple(tokens.shape),
                    },
                )
                if output.mode == "pooled":
                    vec, cls_removed = pool_from_tokens(tokens, output.pooling)
                    meta.update(
                        {
                            "pooling": f"patch_{output.pooling}",
                            "cls_removed": bool(cls_removed),
                        }
                    )
                    out[i] = Embedding(data=vec, meta=meta)
                elif output.mode == "grid":
                    assert xr_mod is not None
                    grid, (h, w), cls_removed = tokens_to_grid_dhw(tokens)
                    meta.update(
                        {
                            "grid_hw": (h, w),
                            "grid_kind": "patch_tokens",
                            "cls_removed": bool(cls_removed),
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
            raise ModelError("satmae_rgb batch inference produced incomplete outputs.")
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
            raise ModelError("satmae_rgb expects a provider backend (or 'auto').")
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        if sensor is None:
            sensor = self._default_sensor()

        model_id = os.environ.get("RS_EMBED_SATMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SATMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        t = temporal_to_range(temporal)

        rgb_u8_all: list[np.ndarray] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or input_chw.shape[0] != 3:
                raise ModelError(
                    "input_chw must be CHW with 3 bands for satmae_rgb, got "
                    f"{getattr(input_chw, 'shape', None)} at index={i}"
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            rgb_u8_all.append(resize_rgb_u8(rgb_u8, image_size))

        model, wmeta = _load_satmae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))

        out: list[Embedding | None] = [None] * len(spatials)
        want_grid = output.mode == "grid"
        xr_mod = None
        if want_grid:
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            toks_batch = _satmae_forward_tokens_batch(
                model,
                rgb_u8_all[s0:s1],
                image_size=image_size,
                device=dev,
            )
            for j, tokens in enumerate(toks_batch):
                i = s0 + j
                meta = base_meta(
                    model_name=self.model_name,
                    hf_id=model_id,
                    backend=str(backend).lower(),
                    image_size=image_size,
                    sensor=sensor,
                    temporal=t,
                    source=sensor.collection,
                    extra={
                        "tokens_kind": "tokens_forward_encoder",
                        "tokens_shape": tuple(tokens.shape),
                        "batch_infer": True,
                        "input_override": True,
                    },
                )
                if output.mode == "pooled":
                    vec, cls_removed = pool_from_tokens(tokens, output.pooling)
                    meta.update(
                        {
                            "pooling": f"patch_{output.pooling}",
                            "cls_removed": bool(cls_removed),
                        }
                    )
                    out[i] = Embedding(data=vec, meta=meta)
                elif output.mode == "grid":
                    assert xr_mod is not None
                    grid, (h, w), cls_removed = tokens_to_grid_dhw(tokens)
                    meta.update(
                        {
                            "grid_hw": (h, w),
                            "grid_kind": "patch_tokens",
                            "cls_removed": bool(cls_removed),
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
            raise ModelError("satmae_rgb batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

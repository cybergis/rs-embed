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
    pool_from_tokens,
    tokens_to_grid_dhw,
    base_meta,
    temporal_to_range,
    ensure_torch,
)


_SATMAEPP_RGB_MEAN = (0.4182007312774658, 0.4214799106121063, 0.3991275727748871)
_SATMAEPP_RGB_STD = (0.28774282336235046, 0.27541765570640564, 0.2764017581939697)


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _falsy(v: str) -> bool:
    return str(v).strip().lower() in {"0", "false", "no", "n", "off"}


def _resolve_satmaepp_channel_order(model_id: str) -> str:
    """
    Resolve channel order for SatMAE++ preprocessing.

    Priority:
      1) RS_EMBED_SATMAEPP_CHANNEL_ORDER in {"rgb","bgr"}
      2) RS_EMBED_SATMAEPP_BGR boolean (legacy knob)
      3) auto heuristic: default BGR for known fmow_rgb checkpoint
    """
    order = str(os.environ.get("RS_EMBED_SATMAEPP_CHANNEL_ORDER", "")).strip().lower()
    if order in {"rgb", "bgr"}:
        return order

    legacy_bgr = os.environ.get("RS_EMBED_SATMAEPP_BGR", "")
    if legacy_bgr:
        if _truthy(legacy_bgr):
            return "bgr"
        if _falsy(legacy_bgr):
            return "rgb"

    mid = str(model_id).strip().lower()
    if "fmow_rgb" in mid:
        return "bgr"
    return "rgb"


def _satmaepp_resize_short_side(image_size: int) -> int:
    crop_pct = (224.0 / 256.0) if int(image_size) <= 224 else 1.0
    return int(float(image_size) / crop_pct)


def _satmaepp_preprocess_info(model_id: str, image_size: int) -> Dict[str, Any]:
    channel_order = _resolve_satmaepp_channel_order(model_id)
    resize_short = _satmaepp_resize_short_side(image_size)
    return {
        "preprocess_name": "satmaepp_fmow_rgb_eval",
        "channel_order": channel_order,
        "norm_mean": tuple(float(x) for x in _SATMAEPP_RGB_MEAN),
        "norm_std": tuple(float(x) for x in _SATMAEPP_RGB_STD),
        "resize_short_side": int(resize_short),
        "center_crop": int(image_size),
    }


def _satmaepp_preprocess_tensor_batch(
    rgb_u8_batch: List[np.ndarray],
    *,
    image_size: int,
    channel_order: str,
):
    """
    SatMAE++ preprocessing aligned to official fmow-rgb eval transform:
      ToTensor -> Normalize(fmow mean/std) -> Resize(short side) -> CenterCrop
    """
    ensure_torch()
    import torch
    from PIL import Image
    from torchvision import transforms

    if channel_order not in {"rgb", "bgr"}:
        raise ModelError(
            f"Invalid SatMAE++ channel_order={channel_order!r}; expected 'rgb' or 'bgr'."
        )

    resize_short = _satmaepp_resize_short_side(image_size)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_SATMAEPP_RGB_MEAN, std=_SATMAEPP_RGB_STD),
            transforms.Resize(
                resize_short, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
        ]
    )

    xs = []
    for i, rgb_u8 in enumerate(rgb_u8_batch):
        if (
            not isinstance(rgb_u8, np.ndarray)
            or rgb_u8.ndim != 3
            or int(rgb_u8.shape[2]) != 3
        ):
            raise ModelError(
                f"SatMAE++ preprocessing expects uint8 HWC RGB arrays; got shape={getattr(rgb_u8, 'shape', None)} at index={i}."
            )
        if rgb_u8.dtype != np.uint8:
            raise ModelError(
                f"SatMAE++ preprocessing expects dtype=uint8; got dtype={getattr(rgb_u8, 'dtype', None)} at index={i}."
            )
        x_hwc = rgb_u8[..., ::-1] if channel_order == "bgr" else rgb_u8
        x = preprocess(Image.fromarray(x_hwc, mode="RGB"))
        if x.ndim != 3:
            raise ModelError(
                f"SatMAE++ preprocess returned shape={tuple(x.shape)} at index={i}; expected [C,H,W]."
            )
        xs.append(x)

    return torch.stack(xs, dim=0)


@lru_cache(maxsize=8)
def _load_satmaepp_cached(model_id: str, dev: str):
    ensure_torch()

    try:
        from rshf.satmaepp import SatMAEPP
    except Exception as e:
        raise ModelError("SatMAE++ requires rshf. Install: pip install rshf") from e

    model = SatMAEPP.from_pretrained(model_id)
    cfg = getattr(model, "config", None)
    in_chans = int(getattr(cfg, "in_chans", 3)) if cfg is not None else 3
    if in_chans != 3:
        raise ModelError(
            f"SatMAE++ RGB adapter expects a 3-channel checkpoint, but {model_id!r} has in_chans={in_chans}."
        )
    try:
        model = model.to(dev).eval()
    except Exception:
        pass

    meta = {"model_id": model_id, "device": dev, "in_chans": in_chans}
    return model, meta


def _load_satmaepp(model_id: str, device: str = "auto"):
    loaded, _dev = _load_cached_with_device(
        _load_satmaepp_cached, device=device, model_id=model_id
    )
    return loaded


def _satmaepp_forward_tokens(
    model,
    rgb_u8: np.ndarray,
    *,
    image_size: int,
    device: str,
    model_id: str,
) -> np.ndarray:
    """
    Return tokens [N,D] via forward_encoder(mask_ratio=0.0).
    """
    return _satmaepp_forward_tokens_batch(
        model,
        [rgb_u8],
        image_size=image_size,
        device=device,
        model_id=model_id,
    )[0]


def _satmaepp_forward_tokens_batch(
    model,
    rgb_u8_batch: List[np.ndarray],
    *,
    image_size: int,
    device: str,
    model_id: str,
) -> List[np.ndarray]:
    """
    Batch version of forward_encoder.
    Returns one [N,D] float32 token array per input image.
    """
    if not rgb_u8_batch:
        return []

    ensure_torch()
    import torch

    channel_order = _resolve_satmaepp_channel_order(model_id)
    xb = _satmaepp_preprocess_tensor_batch(
        rgb_u8_batch,
        image_size=image_size,
        channel_order=channel_order,
    ).to(device)

    fe = getattr(model, "forward_encoder", None)
    if not callable(fe):
        raise ModelError(
            "SatMAE++ wrapper does not expose forward_encoder(). Update rshf."
        )

    with torch.no_grad():
        out = fe(xb, mask_ratio=0.0)
        toks = out[0] if isinstance(out, (tuple, list)) else out  # [B,N,D]
        if toks.ndim != 3 or int(toks.shape[0]) != len(rgb_u8_batch):
            raise ModelError(
                f"SatMAE++ forward_encoder returned {tuple(toks.shape)}; "
                f"expected [B,N,D] with B={len(rgb_u8_batch)}."
            )
        out_np = toks.detach().float().cpu().numpy().astype(np.float32)
        return [out_np[i] for i in range(out_np.shape[0])]


@register("satmaepp")
class SatMAEPPEmbedder(EmbedderBase):
    """
    SatMAE++ (ViT/MAE) on-the-fly embeddings from Sentinel-2 RGB patch (provider backend).

    Strategy aligned via _vit_mae_utils:
      - pooled: pool patch tokens by OutputSpec.pooling (exclude CLS if present)
      - grid: patch token grid (exclude CLS if present)
    """

    DEFAULT_MODEL_ID = "MVRL/satmaepp_ViT-L_pretrain_fmow_rgb"
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
                "channel_order": "bgr",
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

    def _resolve_fetch_workers(self, n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_SATMAEPP_FETCH_WORKERS", str(self.DEFAULT_FETCH_WORKERS)
            )
        )
        return max(1, min(int(n_items), v))

    def _resolve_infer_batch(self, dev: str) -> int:
        default_bs = (
            self.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else self.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_SATMAEPP_BATCH_SIZE", str(default_bs)))
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
            raise ModelError("satmaepp_rgb expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        model_id = os.environ.get("RS_EMBED_SATMAEPP_ID", self.DEFAULT_MODEL_ID)
        image_size = int(
            os.environ.get("RS_EMBED_SATMAEPP_IMG", str(self.DEFAULT_IMAGE_SIZE))
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
                    "input_chw must be CHW with 3 bands for satmaepp_rgb, got {shape}".format(
                        shape=getattr(input_chw, "shape", None),
                    )
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            rgb_u8 = resize_rgb_u8(rgb_u8, image_size)

        model, wmeta = _load_satmaepp(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        pp_info = _satmaepp_preprocess_info(model_id=model_id, image_size=image_size)
        tokens = _satmaepp_forward_tokens(
            model,
            rgb_u8,
            image_size=image_size,
            device=dev,
            model_id=model_id,
        )  # [N,D]

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
                **pp_info,
            },
        )

        if output.mode == "pooled":
            vec, cls_removed = pool_from_tokens(tokens, output.pooling)
            meta.update(
                {"pooling": f"patch_{output.pooling}", "cls_removed": bool(cls_removed)}
            )
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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satmaepp_rgb expects a provider backend (or 'auto').")
        if not spatials:
            return []

        if sensor is None:
            sensor = self._default_sensor()

        model_id = os.environ.get("RS_EMBED_SATMAEPP_ID", self.DEFAULT_MODEL_ID)
        image_size = int(
            os.environ.get("RS_EMBED_SATMAEPP_IMG", str(self.DEFAULT_IMAGE_SIZE))
        )
        t = temporal_to_range(temporal)

        provider = self._get_provider(backend)
        n = len(spatials)
        rgb_u8_all: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
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
                raise ModelError(
                    f"Missing fetched patch at index={i}; batch fetch failed."
                )

        model, wmeta = _load_satmaepp(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))
        pp_info = _satmaepp_preprocess_info(model_id=model_id, image_size=image_size)

        out: List[Optional[Embedding]] = [None] * n
        want_grid = output.mode == "grid"
        xr_mod = None
        if want_grid:
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError(
                    "grid output requires xarray. Install: pip install xarray"
                ) from e

        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            rgb_batch = [x for x in rgb_u8_all[s0:s1] if x is not None]
            toks_batch = _satmaepp_forward_tokens_batch(
                model,
                rgb_batch,
                image_size=image_size,
                device=dev,
                model_id=model_id,
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
                        **pp_info,
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
            raise ModelError(
                "satmaepp_rgb batch inference produced incomplete outputs."
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
            raise ModelError("satmaepp_rgb expects a provider backend (or 'auto').")
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        if sensor is None:
            sensor = self._default_sensor()

        model_id = os.environ.get("RS_EMBED_SATMAEPP_ID", self.DEFAULT_MODEL_ID)
        image_size = int(
            os.environ.get("RS_EMBED_SATMAEPP_IMG", str(self.DEFAULT_IMAGE_SIZE))
        )
        t = temporal_to_range(temporal)

        rgb_u8_all: List[np.ndarray] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or input_chw.shape[0] != 3:
                raise ModelError(
                    "input_chw must be CHW with 3 bands for satmaepp_rgb, got "
                    f"{getattr(input_chw, 'shape', None)} at index={i}"
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            rgb_u8_all.append(resize_rgb_u8(rgb_u8, image_size))

        model, wmeta = _load_satmaepp(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))
        pp_info = _satmaepp_preprocess_info(model_id=model_id, image_size=image_size)

        out: List[Optional[Embedding]] = [None] * len(spatials)
        want_grid = output.mode == "grid"
        xr_mod = None
        if want_grid:
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
            toks_batch = _satmaepp_forward_tokens_batch(
                model,
                rgb_u8_all[s0:s1],
                image_size=image_size,
                device=dev,
                model_id=model_id,
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
                        **pp_info,
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
            raise ModelError(
                "satmaepp_rgb batch inference produced incomplete outputs."
            )
        return [e for e in out if e is not None]

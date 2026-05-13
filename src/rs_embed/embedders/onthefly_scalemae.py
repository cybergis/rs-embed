from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import (
    ModelInputSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from ..providers.resolution import (
    is_provider_backend,
)
from ..tools.runtime import (
    load_cached_with_device as _load_cached_with_device,
)
from .base import EmbedderBase
from .meta import build_meta, temporal_to_range


def ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise ModelError("This embedder requires torch installed.") from e


def fetch_s2_rgb_u8_from_provider(provider, *, spatial, temporal, sensor, out_size):
    from ..providers.fetch import fetch_s2_rgb_chw

    s2_chw = fetch_s2_rgb_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        scale_m=sensor.scale_m,
        cloudy_pct=sensor.cloudy_pct,
        composite=sensor.composite,
    )
    rgb_u8 = (np.clip(s2_chw / 10000.0, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)
    if out_size is None:
        return rgb_u8
    from PIL import Image

    im = Image.fromarray(rgb_u8, mode="RGB")
    return np.array(im.resize((out_size, out_size), resample=Image.BICUBIC), dtype=np.uint8)


def base_meta(
    *,
    model_name,
    hf_id,
    backend,
    image_size,
    sensor,
    temporal=None,
    source=None,
    embed_type="on_the_fly",
    extra=None,
):
    m = build_meta(
        model=model_name,
        kind=embed_type,
        backend=backend,
        source=source or getattr(sensor, "collection", None),
        sensor=sensor,
        temporal=temporal,
        image_size=image_size,
    )
    m["hf_id"] = hf_id
    if extra:
        m.update(extra)
    return m


def pool_from_tokens(tokens, pooling):
    n = len(tokens)
    h2 = int((n - 1) ** 0.5)
    has_cls = n > 1 and h2 * h2 == n - 1
    patch = tokens[1:] if has_cls else tokens
    if len(patch) == 0:
        return tokens[0].astype("float32"), has_cls
    if pooling == "mean":
        return patch.mean(axis=0).astype("float32"), has_cls
    if pooling == "max":
        return patch.max(axis=0).astype("float32"), has_cls
    raise ModelError(f"Unknown pooling={pooling!r} (expected 'mean' or 'max').")


def tokens_to_grid_dhw(tokens):
    n = len(tokens)
    h2 = int((n - 1) ** 0.5)
    has_cls = n > 1 and h2 * h2 == n - 1
    patch = tokens[1:] if has_cls else tokens
    p, d = patch.shape
    hw = int(p**0.5)
    if hw * hw != p:
        raise ModelError(f"Patch token count {p} is not a perfect square.")
    return patch.reshape(hw, hw, d).transpose(2, 0, 1).astype("float32"), (hw, hw), has_cls


_SCALEMAE_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_SCALEMAE_IMAGENET_STD = (0.229, 0.224, 0.225)


def _scalemae_resize_short_side(image_size: int) -> int:
    crop_pct = (224.0 / 256.0) if int(image_size) <= 224 else 1.0
    return int(float(image_size) / crop_pct)


def _scalemae_preprocess_info(image_size: int) -> dict[str, Any]:
    resize_short = _scalemae_resize_short_side(image_size)
    return {
        "preprocess_name": "imagenet_eval",
        "norm_mean": tuple(float(x) for x in _SCALEMAE_IMAGENET_MEAN),
        "norm_std": tuple(float(x) for x in _SCALEMAE_IMAGENET_STD),
        "resize_short_side": int(resize_short),
        "center_crop": int(image_size),
    }


def _scalemae_effective_input_res_m(
    rgb_u8: np.ndarray,
    *,
    image_size: int,
    source_res_m: float,
) -> float:
    if not isinstance(rgb_u8, np.ndarray) or rgb_u8.ndim != 3 or int(rgb_u8.shape[2]) != 3:
        raise ModelError(
            "ScaleMAE effective resolution expects uint8 HWC RGB arrays; "
            f"got shape={getattr(rgb_u8, 'shape', None)}."
        )
    h, w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    if h <= 0 or w <= 0:
        raise ModelError(f"ScaleMAE effective resolution got invalid shape={rgb_u8.shape}.")
    resize_short = _scalemae_resize_short_side(image_size)
    short_side = min(h, w)
    return float(source_res_m) * (float(short_side) / float(resize_short))


def _scalemae_preprocess_tensor_batch(
    rgb_u8_batch: list[np.ndarray],
    *,
    image_size: int,
):
    ensure_torch()
    import torch
    from PIL import Image
    from torchvision import transforms

    resize_short = _scalemae_resize_short_side(image_size)
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_short, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_SCALEMAE_IMAGENET_MEAN,
                std=_SCALEMAE_IMAGENET_STD,
            ),
        ]
    )

    xs = []
    for i, rgb_u8 in enumerate(rgb_u8_batch):
        if not isinstance(rgb_u8, np.ndarray) or rgb_u8.ndim != 3 or int(rgb_u8.shape[2]) != 3:
            raise ModelError(
                "ScaleMAE preprocessing expects uint8 HWC RGB arrays; "
                f"got shape={getattr(rgb_u8, 'shape', None)} at index={i}."
            )
        if rgb_u8.dtype != np.uint8:
            raise ModelError(
                "ScaleMAE preprocessing expects dtype=uint8; "
                f"got dtype={getattr(rgb_u8, 'dtype', None)} at index={i}."
            )
        x = preprocess(Image.fromarray(rgb_u8, mode="RGB"))
        if x.ndim != 3:
            raise ModelError(
                f"ScaleMAE preprocess returned shape={tuple(x.shape)} at index={i}; expected [C,H,W]."
            )
        xs.append(x.unsqueeze(0))

    if not xs:
        return torch.empty((0, 3, int(image_size), int(image_size)))
    return torch.cat(xs, dim=0)


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
    except Exception as _e:
        pass

    meta = {"model_id": model_id, "device": dev}
    return model, meta


def _load_scalemae(model_id: str, device: str = "auto"):
    loaded, _dev = _load_cached_with_device(_load_scalemae_cached, device=device, model_id=model_id)
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


def _resolve_scalemae_forward_features(model) -> tuple[Any, Any, str]:
    """
    Resolve the real ScaleMAE backbone that exposes forward_features().

    rshf's ScaleMAE wrapper is a thin nn.Module shell whose actual ViT backbone
    lives at ``wrapper.model``. The official scale-aware path is implemented on
    that nested module, not on the outer wrapper.
    """
    candidates = [
        ("self", model),
        ("model", getattr(model, "model", None)),
        ("backbone", getattr(model, "backbone", None)),
        ("encoder", getattr(model, "encoder", None)),
    ]
    for owner, candidate in candidates:
        if candidate is None:
            continue
        ff = getattr(candidate, "forward_features", None)
        if callable(ff):
            return candidate, ff, owner
    raise ModelError(
        "ScaleMAE wrapper does not expose forward_features(), including common nested "
        "backbone attributes such as .model. Update rshf or use a wrapper that keeps "
        "the official ScaleMAE feature-extraction path."
    )


def _call_with_patch_size(fn, x, *, patch_size: int, input_res):
    """
    Call forward/forward_features with compatible signature across rshf versions.
    Tries kwargs then positional.
    """
    import inspect

    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception as _e:
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
                raise ModelError(f"ScaleMAE call failed even with patch_size/input_res: {e}") from e


def _scalemae_forward_tokens_or_vec(
    model,
    rgb_u8: np.ndarray,
    *,
    image_size: int,
    device: str,
    input_res_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Your rshf ScaleMAE requires:
      - input_res: 1D tensor
      - patch_size: required positional argument in forward() (and sometimes in forward_features()).

    Always use the official ScaleMAE feature-extraction path. For rshf wrappers,
    this may require unwrapping the nested backbone at ``model.model``.
    """
    core_model, ff, ff_owner = _resolve_scalemae_forward_features(model)

    ensure_torch()
    import torch

    x = _scalemae_preprocess_tensor_batch([rgb_u8], image_size=image_size).to(device)  # [B,3,H,W]
    input_res = torch.tensor([float(input_res_m)], dtype=torch.float32, device=device)  # 1D
    patch_size = _infer_patch_size(core_model)
    prep = _scalemae_preprocess_info(image_size)

    with torch.no_grad():
        out = _call_with_patch_size(ff, x, patch_size=patch_size, input_res=input_res)
        out0 = out[0] if isinstance(out, (tuple, list)) else out

        if hasattr(out0, "ndim") and out0.ndim == 3:  # [B,N,D]
            toks = out0
            return toks[0].detach().float().cpu().numpy().astype(np.float32), {
                "tokens_kind": "tokens_forward_features",
                "forward_features_owner": ff_owner,
                "input_res_m": float(input_res_m),
                "used_patch_size": int(patch_size),
                "tokens_shape": tuple(toks.shape),
                **prep,
            }

        if hasattr(out0, "ndim") and out0.ndim == 2:  # [B,D]
            v = out0
            return v[0].detach().float().cpu().numpy().astype(np.float32), {
                "tokens_kind": "pooled_forward_features",
                "forward_features_owner": ff_owner,
                "input_res_m": float(input_res_m),
                "used_patch_size": int(patch_size),
                "vec_shape": tuple(v.shape),
                **prep,
            }

        if hasattr(out0, "ndim") and out0.ndim == 4:  # [B,C,H,W] -> tokens
            b, c, h, w = out0.shape
            toks = out0.permute(0, 2, 3, 1).reshape(b, h * w, c)
            return toks[0].detach().float().cpu().numpy().astype(np.float32), {
                "tokens_kind": "tokens_from_feature_map",
                "forward_features_owner": ff_owner,
                "input_res_m": float(input_res_m),
                "used_patch_size": int(patch_size),
                "feature_map_hw": (int(h), int(w)),
                "tokens_shape": tuple(toks.shape),
                **prep,
            }

        raise ModelError(
            f"ScaleMAE forward_features returned unsupported: {type(out0)} {getattr(out0, 'shape', None)}"
        )


def _scalemae_forward_tokens_or_vec_batch(
    model,
    rgb_u8_batch: list[np.ndarray],
    *,
    image_size: int,
    device: str,
    input_res_m: float | list[float] | np.ndarray,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Batch variant that returns one output array per input item."""
    if not rgb_u8_batch:
        return [], {"tokens_kind": "empty_batch"}

    core_model, ff, ff_owner = _resolve_scalemae_forward_features(model)

    ensure_torch()
    import torch

    xb = _scalemae_preprocess_tensor_batch(rgb_u8_batch, image_size=image_size).to(
        device
    )  # [B,3,H,W]
    if np.isscalar(input_res_m):
        input_res_values = [float(input_res_m)] * len(rgb_u8_batch)
    else:
        input_res_values = [float(x) for x in input_res_m]
        if len(input_res_values) != len(rgb_u8_batch):
            raise ModelError(
                "ScaleMAE batch input_res_m length mismatch: "
                f"{len(input_res_values)} != {len(rgb_u8_batch)}"
            )
    input_res = torch.tensor(input_res_values, dtype=torch.float32, device=device)
    patch_size = _infer_patch_size(core_model)
    prep = _scalemae_preprocess_info(image_size)

    def _to_list(arr: np.ndarray) -> list[np.ndarray]:
        return [arr[i].astype(np.float32, copy=False) for i in range(arr.shape[0])]

    common_extra: dict[str, Any] = {
        "forward_features_owner": ff_owner,
        "used_patch_size": int(patch_size),
        **prep,
    }
    if input_res_values:
        if all(abs(x - input_res_values[0]) < 1e-6 for x in input_res_values[1:]):
            common_extra["input_res_m"] = float(input_res_values[0])
        else:
            common_extra["input_res_m_kind"] = "per_item"

    with torch.inference_mode():
        out = _call_with_patch_size(ff, xb, patch_size=patch_size, input_res=input_res)
        out0 = out[0] if isinstance(out, (tuple, list)) else out

        if hasattr(out0, "ndim") and out0.ndim == 3:  # [B,N,D]
            toks = out0.detach().float().cpu().numpy().astype(np.float32)
            return _to_list(toks), {
                "tokens_kind": "tokens_forward_features",
                "batch_shape": tuple(toks.shape),
                **common_extra,
            }

        if hasattr(out0, "ndim") and out0.ndim == 2:
            vec = out0.detach().float().cpu().numpy().astype(np.float32)
            return _to_list(vec), {
                "tokens_kind": "pooled_forward_features",
                "batch_shape": tuple(vec.shape),
                **common_extra,
            }

        if hasattr(out0, "ndim") and out0.ndim == 4:  # [B,C,H,W] -> tokens
            b, c, h, w = out0.shape
            toks = out0.permute(0, 2, 3, 1).reshape(b, h * w, c)
            toks_np = toks.detach().float().cpu().numpy().astype(np.float32)
            return _to_list(toks_np), {
                "tokens_kind": "tokens_from_feature_map",
                "feature_map_hw": (int(h), int(w)),
                "batch_shape": tuple(toks_np.shape),
                **common_extra,
            }

        raise ModelError(
            f"ScaleMAE forward_features returned unsupported: {type(out0)} {getattr(out0, 'shape', None)}"
        )


@register("scalemae")
class ScaleMAERGBEmbedder(EmbedderBase):
    """
    ScaleMAE on-the-fly embedding from Sentinel-2 RGB patch (provider backend).

    Strategy aligned:
      - pooled: pool patch tokens by OutputSpec.pooling (exclude CLS if present)
      - grid: patch token grid (exclude CLS if present)
      - scale: derives effective input_res_m after preprocessing from sensor.scale_m
    """

    DEFAULT_MODEL_ID = "MVRL/scalemae-vitlarge-800"
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 32

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=10,
        cloudy_pct=30,
        image_size=224,
        expected_channels=3,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["provider"],
            "model_id_default": self.DEFAULT_MODEL_ID,
            "image_size": self.DEFAULT_IMAGE_SIZE,
            "inputs": {
                "collection": self.input_spec.collection,
                "bands": list(self.input_spec.bands),
                "model_preprocess": "s2_sr_raw_then_imagenet_eval",
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "model_id": self.DEFAULT_MODEL_ID,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "scale_m": self.input_spec.scale_m,
                "cloudy_pct": self.input_spec.cloudy_pct,
                "composite": self.input_spec.composite,
            },
        }

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
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: np.ndarray | None = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("scalemae_rgb expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self.input_spec.to_sensor_spec()

        model_id = os.environ.get("RS_EMBED_SCALEMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SCALEMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))

        t = temporal_to_range(temporal)
        # Fetch RGB patch (optionally reuse pre-fetched raw patch)
        if input_chw is None:
            rgb_u8 = fetch_s2_rgb_u8_from_provider(
                spatial=spatial,
                temporal=t,
                sensor=sensor,
                out_size=None,
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
        input_res_m = _scalemae_effective_input_res_m(
            rgb_u8,
            image_size=image_size,
            source_res_m=float(sensor.scale_m),
        )

        model, wmeta = _load_scalemae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        out, extra = _scalemae_forward_tokens_or_vec(
            model,
            rgb_u8,
            image_size=image_size,
            device=dev,
            input_res_m=float(input_res_m),
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
        if not spatials:
            return []
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("scalemae_rgb expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self.input_spec.to_sensor_spec()

        model_id = os.environ.get("RS_EMBED_SCALEMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SCALEMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        t = temporal_to_range(temporal)

        provider = self._get_provider(backend)
        n = len(spatials)
        rgb_u8_all: list[np.ndarray | None] = [None] * n
        input_res_all: list[float | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            rgb_u8 = fetch_s2_rgb_u8_from_provider(
                spatial=sp,
                temporal=t,
                sensor=sensor,
                out_size=None,
                provider=provider,
            )
            return i, rgb_u8

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                ii, rgb = _fetch_one(i, sp)
                rgb_u8_all[ii] = rgb
                input_res_all[ii] = _scalemae_effective_input_res_m(
                    rgb,
                    image_size=image_size,
                    source_res_m=float(sensor.scale_m),
                )
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, rgb = fut.result()
                    rgb_u8_all[i] = rgb
                    input_res_all[i] = _scalemae_effective_input_res_m(
                        rgb,
                        image_size=image_size,
                        source_res_m=float(sensor.scale_m),
                    )

        model, wmeta = _load_scalemae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))

        out: list[Embedding | None] = [None] * len(spatials)
        xr_mod = None
        if output.mode == "grid":
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            chunk_rgb = []
            chunk_idx = []
            chunk_res = []
            for i in range(s0, s1):
                rgb_u8 = rgb_u8_all[i]
                if rgb_u8 is None:
                    raise ModelError(f"Missing prefetched patch at index={i} for scalemae_rgb.")
                input_res_m = input_res_all[i]
                if input_res_m is None:
                    raise ModelError(
                        f"Missing prefetched effective input resolution at index={i} for scalemae_rgb."
                    )
                chunk_idx.append(i)
                chunk_rgb.append(rgb_u8)
                chunk_res.append(float(input_res_m))

            try:
                chunk_outs, chunk_extra = _scalemae_forward_tokens_or_vec_batch(
                    model,
                    chunk_rgb,
                    image_size=image_size,
                    device=dev,
                    input_res_m=chunk_res,
                )
            except Exception as _e:
                chunk_outs = []
                chunk_extra = {}
                for rgb_u8, input_res_m in zip(chunk_rgb, chunk_res, strict=True):
                    o1, e1 = _scalemae_forward_tokens_or_vec(
                        model,
                        rgb_u8,
                        image_size=image_size,
                        device=dev,
                        input_res_m=float(input_res_m),
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
                        "input_res_m": float(chunk_res[j]),
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
            raise ModelError("scalemae_rgb batch inference produced incomplete outputs.")
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
            raise ModelError("scalemae_rgb expects a provider backend (or 'auto').")
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        if sensor is None:
            sensor = self.input_spec.to_sensor_spec()

        model_id = os.environ.get("RS_EMBED_SCALEMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SCALEMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        t = temporal_to_range(temporal)
        model, wmeta = _load_scalemae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))

        rgb_u8_all: list[np.ndarray] = []
        input_res_all: list[float] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or input_chw.shape[0] != 3:
                raise ModelError(
                    "input_chw must be CHW with 3 bands for scalemae_rgb, got "
                    f"{getattr(input_chw, 'shape', None)} at index={i}"
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            rgb_u8_all.append(rgb_u8)
            input_res_all.append(
                _scalemae_effective_input_res_m(
                    rgb_u8,
                    image_size=image_size,
                    source_res_m=float(sensor.scale_m),
                )
            )

        out: list[Embedding | None] = [None] * len(spatials)
        xr_mod = None
        if output.mode == "grid":
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            chunk_idx = list(range(s0, s1))
            chunk_rgb = [rgb_u8_all[i] for i in chunk_idx]
            chunk_res = [float(input_res_all[i]) for i in chunk_idx]
            chunk_outs, chunk_extra = _scalemae_forward_tokens_or_vec_batch(
                model,
                chunk_rgb,
                image_size=image_size,
                device=dev,
                input_res_m=chunk_res,
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
                        "input_res_m": float(chunk_res[j]),
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
            raise ModelError("scalemae_rgb prefetched batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

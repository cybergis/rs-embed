from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
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
      3) auto heuristic: default RGB for the official fmow_rgb checkpoint
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
        return "rgb"
    return "rgb"


def _satmaepp_resize_short_side(image_size: int) -> int:
    crop_pct = (224.0 / 256.0) if int(image_size) <= 224 else 1.0
    return int(float(image_size) / crop_pct)


def _satmaepp_preprocess_info(model_id: str, image_size: int) -> dict[str, Any]:
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
    rgb_u8_batch: list[np.ndarray],
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
            transforms.Resize(resize_short, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    )

    xs = []
    for i, rgb_u8 in enumerate(rgb_u8_batch):
        if not isinstance(rgb_u8, np.ndarray) or rgb_u8.ndim != 3 or int(rgb_u8.shape[2]) != 3:
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


def _canonicalize_satmaepp_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    aliases: dict[str, tuple[str, ...]] = {
        "in_chans": ("in_chans", "num_channels", "in_channels", "channels"),
        "img_size": ("img_size", "image_size"),
        "patch_size": ("patch_size",),
        "embed_dim": ("embed_dim", "hidden_size"),
        "depth": ("depth", "num_hidden_layers"),
        "num_heads": ("num_heads", "num_attention_heads"),
        "decoder_embed_dim": ("decoder_embed_dim",),
        "decoder_depth": ("decoder_depth",),
        "decoder_num_heads": ("decoder_num_heads",),
        "mlp_ratio": ("mlp_ratio",),
        "proj_ratio": ("proj_ratio",),
        "norm_pix_loss": ("norm_pix_loss",),
    }
    defaults: dict[str, Any] = {
        "in_chans": 3,
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
        "proj_ratio": 4,
        "norm_pix_loss": False,
    }
    for key, names in aliases.items():
        value = None
        for name in names:
            if name in cfg:
                value = cfg[name]
                break
        cfg[key] = defaults[key] if value is None else value
    return cfg


def _unwrap_satmaepp_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return nested
    if isinstance(payload, dict):
        return payload
    raise ModelError(f"Unsupported SatMAE++ checkpoint payload type: {type(payload).__name__}")


def _load_satmaepp_from_snapshot(*, SatMAEPP: Any, model_id: str, dev: str):
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ModelError(
            "SatMAE++ fallback loading requires huggingface_hub. Install: pip install huggingface_hub"
        ) from e

    ensure_torch()
    import torch

    snap_dir = Path(
        snapshot_download(
            repo_id=model_id,
            allow_patterns=["config.json", "pytorch_model.bin", "model.safetensors"],
        )
    )
    config_path = snap_dir / "config.json"
    if not config_path.is_file():
        raise ModelError(f"SatMAE++ snapshot is missing config.json: {snap_dir}")
    cfg = _canonicalize_satmaepp_config(json.loads(config_path.read_text(encoding="utf-8")))
    model = SatMAEPP(cfg)

    ckpt_path = snap_dir / "pytorch_model.bin"
    if ckpt_path.is_file():
        payload = torch.load(str(ckpt_path), map_location="cpu")
    else:
        st_path = snap_dir / "model.safetensors"
        if not st_path.is_file():
            raise ModelError(f"SatMAE++ snapshot has no supported weights file: {snap_dir}")
        try:
            from safetensors.torch import load_file as load_safetensors
        except Exception as e:
            raise ModelError(
                "SatMAE++ safetensors checkpoint requires safetensors. Install: pip install safetensors"
            ) from e
        payload = load_safetensors(str(st_path), device="cpu")

    state_dict = _unwrap_satmaepp_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    try:
        model = model.to(dev).eval()
    except Exception as _e:
        pass

    meta = {
        "model_id": model_id,
        "device": dev,
        "in_chans": int(cfg["in_chans"]),
        "load_path": str(snap_dir),
        "load_mode": "manual_snapshot",
    }
    return model, meta


@lru_cache(maxsize=8)
def _load_satmaepp_cached(model_id: str, dev: str):
    ensure_torch()

    try:
        from rshf.satmaepp import SatMAEPP
    except Exception as e:
        raise ModelError("SatMAE++ requires rshf. Install: pip install rshf") from e

    try:
        model = SatMAEPP.from_pretrained(model_id)
        load_mode = "from_pretrained"
    except AttributeError as e:
        # Some rshf / transformers combinations instantiate a generic
        # PreTrainedConfig that drops SatMAE++-specific fields such as in_chans.
        if "in_chans" not in str(e):
            raise
        model, meta = _load_satmaepp_from_snapshot(SatMAEPP=SatMAEPP, model_id=model_id, dev=dev)
        return model, meta

    cfg = getattr(model, "config", None)
    in_chans = int(getattr(cfg, "in_chans", 3)) if cfg is not None else 3
    if in_chans != 3:
        raise ModelError(
            f"SatMAE++ RGB adapter expects a 3-channel checkpoint, but {model_id!r} has in_chans={in_chans}."
        )
    try:
        model = model.to(dev).eval()
    except Exception as _e:
        pass

    meta = {"model_id": model_id, "device": dev, "in_chans": in_chans, "load_mode": load_mode}
    return model, meta


def _load_satmaepp(model_id: str, device: str = "auto"):
    loaded, _dev = _load_cached_with_device(_load_satmaepp_cached, device=device, model_id=model_id)
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
    rgb_u8_batch: list[np.ndarray],
    *,
    image_size: int,
    device: str,
    model_id: str,
) -> list[np.ndarray]:
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
        raise ModelError("SatMAE++ wrapper does not expose forward_encoder(). Update rshf.")

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

    Strategy aligned:
      - pooled: pool patch tokens by OutputSpec.pooling (exclude CLS if present)
      - grid: patch token grid (exclude CLS if present)
    """

    DEFAULT_MODEL_ID = "MVRL/satmaepp_ViT-L_pretrain_fmow_rgb"
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
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "model_id": self.DEFAULT_MODEL_ID,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "scale_m": self.input_spec.scale_m,
                "cloudy_pct": self.input_spec.cloudy_pct,
                "composite": self.input_spec.composite,
                "channel_order": "rgb",
            },
        }

    def _resolve_fetch_workers(self, n_items: int) -> int:
        v = int(os.environ.get("RS_EMBED_SATMAEPP_FETCH_WORKERS", str(self.DEFAULT_FETCH_WORKERS)))
        return max(1, min(int(n_items), v))

    def _resolve_infer_batch(self, dev: str) -> int:
        default_bs = (
            self.DEFAULT_BATCH_CUDA if str(dev).startswith("cuda") else self.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_SATMAEPP_BATCH_SIZE", str(default_bs)))
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
            raise ModelError("satmaepp_rgb expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self.input_spec.to_sensor_spec()

        model_id = os.environ.get("RS_EMBED_SATMAEPP_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SATMAEPP_IMG", str(self.DEFAULT_IMAGE_SIZE)))

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
                    "input_chw must be CHW with 3 bands for satmaepp_rgb, got {shape}".format(
                        shape=getattr(input_chw, "shape", None),
                    )
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)

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
            raise ModelError("satmaepp_rgb expects a provider backend (or 'auto').")
        if not spatials:
            return []

        if sensor is None:
            sensor = self.input_spec.to_sensor_spec()

        model_id = os.environ.get("RS_EMBED_SATMAEPP_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SATMAEPP_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        t = temporal_to_range(temporal)

        provider = self._get_provider(backend)
        n = len(spatials)
        rgb_u8_all: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            rgb = fetch_s2_rgb_u8_from_provider(
                spatial=sp,
                temporal=t,
                sensor=sensor,
                out_size=None,
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

        model, wmeta = _load_satmaepp(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))
        pp_info = _satmaepp_preprocess_info(model_id=model_id, image_size=image_size)

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
            raise ModelError("satmaepp_rgb batch inference produced incomplete outputs.")
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
            raise ModelError("satmaepp_rgb expects a provider backend (or 'auto').")
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        if sensor is None:
            sensor = self.input_spec.to_sensor_spec()

        model_id = os.environ.get("RS_EMBED_SATMAEPP_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SATMAEPP_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        t = temporal_to_range(temporal)

        rgb_u8_all: list[np.ndarray] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or input_chw.shape[0] != 3:
                raise ModelError(
                    "input_chw must be CHW with 3 bands for satmaepp_rgb, got "
                    f"{getattr(input_chw, 'shape', None)} at index={i}"
                )
            s2_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            rgb_u8 = (s2_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            rgb_u8_all.append(rgb_u8)

        model, wmeta = _load_satmaepp(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))
        pp_info = _satmaepp_preprocess_info(model_id=model_id, image_size=image_size)

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
            raise ModelError("satmaepp_rgb batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

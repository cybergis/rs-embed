from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import (
    ModelInputSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from ..core.types import EmbedderCapabilities
from ..providers.base import ProviderBase
from ..providers.fetch import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
)
from ..providers.resolution import (
    is_provider_backend,
)
from ..tools.runtime import (
    load_cached_with_device as _load_cached_with_device,
)
from ..tools.runtime import (
    move_model_to_device as _move_model_to_device,
)
from ..tools.shape import (
    crop_grid_and_pool,
    crop_grid_to_roi,
    geo_roi_from_meta,
    roi_is_full,
    square_fetch_batch,
)
from ..tools.spatial import square_spatial
from .base import EmbedderBase
from .config import model_config_value
from .meta import base_meta, temporal_to_range
from .shared import grid_to_dataarray, import_xarray, resolve_hf_cache_dir, verify_loaded_params


def ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise ModelError("This embedder requires torch installed.") from e


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

_SATMAEPP_S2_MODEL_FN_BY_VARIANT = {
    "large": "mae_vit_large_patch16",
}


def _normalize_satmaepp_s2_variant(variant: Any) -> str:
    raw = str(variant).strip().lower()
    aliases = {
        "l": "large",
        "large": "large",
    }
    resolved = aliases.get(raw)
    if resolved is None:
        raise ModelError(
            f"Unknown satmaepp_s2_10b variant='{variant}'. rs-embed currently exposes "
            "only variant='large' because this adapter is wired to the published "
            "ViT-Large Sentinel checkpoint."
        )
    return resolved


def _variant_from_satmaepp_s2_model_fn(model_fn: str) -> str | None:
    for variant, candidate in _SATMAEPP_S2_MODEL_FN_BY_VARIANT.items():
        if candidate == model_fn:
            return variant
    return None


def _resolve_satmaepp_s2_runtime_config(
    *,
    model_config: dict[str, Any] | None,
    default_ckpt_repo: str,
    default_ckpt_file: str,
    default_model_fn: str,
    default_image_size: int,
    default_patch_size: int,
) -> dict[str, Any]:
    variant_v = model_config_value(model_config, "variant")
    if variant_v is not None:
        variant = _normalize_satmaepp_s2_variant(variant_v)
        model_fn = _SATMAEPP_S2_MODEL_FN_BY_VARIANT[variant]
    else:
        model_fn = os.environ.get("RS_EMBED_SATMAEPP_S2_MODEL_FN", default_model_fn).strip()
        variant = _variant_from_satmaepp_s2_model_fn(model_fn)
        if variant is None:
            raise ModelError(
                f"Unknown satmaepp_s2_10b model_fn='{model_fn}' from "
                "RS_EMBED_SATMAEPP_S2_MODEL_FN. rs-embed currently exposes only the "
                "large Sentinel checkpoint ('mae_vit_large_patch16')."
            )

    ckpt_repo = os.environ.get("RS_EMBED_SATMAEPP_S2_CKPT_REPO", default_ckpt_repo).strip()

    ckpt_file = os.environ.get("RS_EMBED_SATMAEPP_S2_CKPT_FILE", default_ckpt_file).strip()

    image_size = int(os.environ.get("RS_EMBED_SATMAEPP_S2_IMG", str(default_image_size)))

    patch_size = int(os.environ.get("RS_EMBED_SATMAEPP_S2_PATCH", str(default_patch_size)))

    return {
        "variant": variant,
        "ckpt_repo": ckpt_repo,
        "ckpt_file": ckpt_file,
        "model_fn": model_fn,
        "image_size": int(image_size),
        "patch_size": int(patch_size),
    }


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
    d = resolve_hf_cache_dir()
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

    model = _move_model_to_device(model, dev, model_name="SatMAE++ Sentinel-2")

    wstats = verify_loaded_params(
        model,
        model_name="SatMAE++ Sentinel",
        no_params_msg="SatMAE++ Sentinel model has no parameters; checkpoint load failed.",
        nonfinite_msg="SatMAE++ Sentinel parameters contain NaN/Inf; checkpoint likely invalid.",
    )

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
        **wstats,
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

    # No center crop: resize the (square) input straight to image_size so the ViT
    # token grid covers the whole fetched square, keeping the ROI crop-back and
    # tile stitcher aligned. The source repo's eval used Resize(short→256/0.875)+
    # CenterCrop, which makes the grid cover only the central ~87.5% of the input
    # — dropping FOV and misaligning per-tile grids. Matches the RGB SatMAE++ /
    # ScaleMAE direct-resize preprocessing. (SentinelNormalize is already folded
    # into the uint8 conversion above, so the transform only resizes + scales.)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (int(image_size), int(image_size)),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
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
    return arr


class SatMAEPPSentinel10Embedder(EmbedderBase):
    """
    SatMAE++ Sentinel-2 10-band adapter reproducing source repo group-channel branch.

    This is no longer a standalone registered model: it backs the ``"s2_10b"``
    modality of the ``satmaepp`` model, which delegates to it when a 10-band
    Sentinel-2 sensor is selected (see :mod:`onthefly_satmaepp`). It reports
    ``model_name="satmaepp"`` so emitted metadata is labelled under the single
    public model name.

    Input bands (S2 SR): B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
    Source alignment:
      - grouped-channel encoder with groups ((0,1,2,6),(3,4,5,7),(8,9))
      - SentinelNormalize(mean/std) -> ToTensor -> Resize -> CenterCrop (eval-style)
      - forward_encoder(mask_ratio=0.0) for embedding extraction
    """

    # Backs the satmaepp "s2_10b" modality; metadata is labelled "satmaepp".
    model_name = "satmaepp"

    # ViT-grid model needs a square input → base.fetch_input (and the direct/batch
    # fetch paths) enlarge a rectangular ROI to a square of real imagery; the token
    # grid is then cropped back to the ROI window.
    _requires_square_input = True

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_S2_SR_10_BANDS),
        scale_m=10,
        cloudy_pct=30,
        image_size=96,
        expected_channels=10,
    )

    DEFAULT_CKPT_REPO = "mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel"
    DEFAULT_CKPT_FILE = "checkpoint_ViT-L_pretrain_fmow_sentinel.pth"
    DEFAULT_MODEL_FN = "mae_vit_large_patch16"

    DEFAULT_IMAGE_SIZE = 96
    DEFAULT_PATCH_SIZE = 8
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 32

    # Explicit pipeline-routing capabilities; the contract test asserts these
    # match the actual method signatures (tests/test_capabilities_contract.py).
    capabilities = EmbedderCapabilities(
        input_chw=True,
        fetch_meta=True,
        batch_fetch_metas=True,
        model_config_single=True,
        model_config_batch=True,
        model_config_batch_inputs=True,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["provider"],
            "model_id_default": self.DEFAULT_CKPT_REPO,
            "image_size": self.DEFAULT_IMAGE_SIZE,
            "inputs": {
                "collection": self.input_spec.collection,
                "bands": list(self.input_spec.bands),
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "ckpt_repo": self.DEFAULT_CKPT_REPO,
                "ckpt_file": self.DEFAULT_CKPT_FILE,
                "model_fn": self.DEFAULT_MODEL_FN,
                "variant": "large",
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
            "model_config": {
                "variant": {
                    "type": "string",
                    "default": "large",
                    "choices": ["large"],
                }
            },
        }

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return SatMAEPPSentinel10Embedder.input_spec.to_sensor_spec()

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
        model_config: dict[str, Any] | None = None,
        fetch_meta: dict[str, Any] | None = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satmaepp modality='s2_10b' expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        # Fetch-square ROI window: set on the direct fetch below, or carried in
        # fetch_meta when the API/batch prefetched a square. The token grid is
        # cropped back to it (full frame → no-op, behavior unchanged).
        geo_roi = geo_roi_from_meta(fetch_meta)

        if tuple(sensor.bands) != tuple(_S2_SR_10_BANDS):
            raise ModelError(
                "satmaepp modality='s2_10b' requires exact band order "
                f"{_S2_SR_10_BANDS}; got {tuple(sensor.bands)}"
            )

        runtime_cfg = _resolve_satmaepp_s2_runtime_config(
            model_config=model_config,
            default_ckpt_repo=self.DEFAULT_CKPT_REPO,
            default_ckpt_file=self.DEFAULT_CKPT_FILE,
            default_model_fn=self.DEFAULT_MODEL_FN,
            default_image_size=self.DEFAULT_IMAGE_SIZE,
            default_patch_size=self.DEFAULT_PATCH_SIZE,
        )
        ckpt_repo = str(runtime_cfg["ckpt_repo"])
        ckpt_file = str(runtime_cfg["ckpt_file"])
        model_fn = str(runtime_cfg["model_fn"])
        image_size = int(runtime_cfg["image_size"])
        patch_size = int(runtime_cfg["patch_size"])
        cache_dir = _resolve_hf_cache_dir()

        t = temporal_to_range(temporal)
        if input_chw is None:
            spatial, geo_roi = square_spatial(spatial)  # enlarge rectangle to square
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
                "variant": runtime_cfg["variant"],
                **wmeta,
            },
        )

        # Build the spatial grid for grid output, or to crop+pool the ROI when the
        # input was enlarged to a square at fetch time.
        cropped_to_roi = not roi_is_full(geo_roi)

        if output.mode == "pooled":
            if cropped_to_roi:
                # Pool only the ROI tokens (the model's group-token pool would
                # include the real-neighborhood context fetched only to square the
                # input). Build the grid, crop to ROI, then pool.
                group_reduce = self._resolve_group_reduce()
                grid, _, cls_removed, _ = _satmaepp_s2_grid(tokens, group_reduce=group_reduce)
                _, vec = crop_grid_and_pool(grid, geo_roi, pooling=output.pooling)
                assert vec is not None
                meta.update(
                    {
                        "pooling": f"roi_grid_{output.pooling}",
                        "cls_removed": bool(cls_removed),
                    }
                )
            else:
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
            if cropped_to_roi:
                grid = crop_grid_to_roi(grid, geo_roi)
                h, w = int(grid.shape[1]), int(grid.shape[2])
            meta.update(
                {
                    "grid_hw": (h, w),
                    "grid_kind": "spatial_tokens_aggregated_over_channel_groups",
                    "cls_removed": bool(cls_removed),
                    **gmeta,
                }
            )

            da = grid_to_dataarray(grid, meta=meta)
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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satmaepp modality='s2_10b' expects a provider backend (or 'auto').")
        if not spatials:
            return []

        if sensor is None:
            sensor = self._default_sensor()
        if tuple(sensor.bands) != tuple(_S2_SR_10_BANDS):
            raise ModelError(
                "satmaepp modality='s2_10b' requires exact band order "
                f"{_S2_SR_10_BANDS}; got {tuple(sensor.bands)}"
            )

        runtime_cfg = _resolve_satmaepp_s2_runtime_config(
            model_config=model_config,
            default_ckpt_repo=self.DEFAULT_CKPT_REPO,
            default_ckpt_file=self.DEFAULT_CKPT_FILE,
            default_model_fn=self.DEFAULT_MODEL_FN,
            default_image_size=self.DEFAULT_IMAGE_SIZE,
            default_patch_size=self.DEFAULT_PATCH_SIZE,
        )
        ckpt_repo = str(runtime_cfg["ckpt_repo"])
        ckpt_file = str(runtime_cfg["ckpt_file"])
        model_fn = str(runtime_cfg["model_fn"])
        image_size = int(runtime_cfg["image_size"])
        patch_size = int(runtime_cfg["patch_size"])
        cache_dir = _resolve_hf_cache_dir()
        t = temporal_to_range(temporal)

        provider = self._get_provider(backend)
        n = len(spatials)

        # Square-fetch each ROI (enlarge rectangle to a square of real imagery),
        # collecting the ROI window per item so each token grid is cropped back.
        raws, geo_rois = square_fetch_batch(
            spatials,
            lambda sq: _fetch_s2_sr_10_raw_chw(
                provider=provider,
                spatial=sq,
                temporal=t,
                scale_m=int(getattr(sensor, "scale_m", 10)),
                cloudy_pct=int(getattr(sensor, "cloudy_pct", 30)),
                composite=str(getattr(sensor, "composite", "median")),
                fill_value=float(getattr(sensor, "fill_value", 0.0)),
            ),
            max_workers=self._resolve_fetch_workers(n),
        )

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
        any_cropped = any(not roi_is_full(gr) for gr in geo_rois)
        # group_reduce is needed for grid output, and also for pooled output when an
        # ROI was enlarged to a square (pooled then derives from the cropped grid).
        group_reduce = self._resolve_group_reduce() if (want_grid or any_cropped) else "mean"
        if want_grid:
            import_xarray()  # fail fast before batch inference

        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            raw_batch = list(raws[s0:s1])
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
                        "variant": runtime_cfg["variant"],
                        **wmeta,
                    },
                )

                cropped_to_roi = not roi_is_full(geo_rois[i])

                if output.mode == "pooled":
                    if cropped_to_roi:
                        grid, _, cls_removed, _ = _satmaepp_s2_grid(
                            tokens, group_reduce=group_reduce
                        )
                        _, vec = crop_grid_and_pool(grid, geo_rois[i], pooling=output.pooling)
                        assert vec is not None
                        meta.update(
                            {
                                "pooling": f"roi_grid_{output.pooling}",
                                "cls_removed": bool(cls_removed),
                            }
                        )
                    else:
                        vec, cls_removed = _satmaepp_s2_pool(tokens, output.pooling)
                        meta.update(
                            {
                                "pooling": f"group_tokens_{output.pooling}",
                                "cls_removed": bool(cls_removed),
                            }
                        )
                    out[i] = Embedding(data=vec.astype(np.float32), meta=meta)
                elif output.mode == "grid":
                    grid, (h, w), cls_removed, gmeta = _satmaepp_s2_grid(
                        tokens, group_reduce=group_reduce
                    )
                    if cropped_to_roi:
                        grid = crop_grid_to_roi(grid, geo_rois[i])
                        h, w = int(grid.shape[1]), int(grid.shape[2])
                    meta.update(
                        {
                            "grid_hw": (h, w),
                            "grid_kind": "spatial_tokens_aggregated_over_channel_groups",
                            "cls_removed": bool(cls_removed),
                            **gmeta,
                        }
                    )
                    da = grid_to_dataarray(grid, meta=meta)
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
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
        fetch_metas: list[dict[str, Any] | None] | None = None,
    ) -> list[Embedding]:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satmaepp modality='s2_10b' expects a provider backend (or 'auto').")
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
                "satmaepp modality='s2_10b' requires exact band order "
                f"{_S2_SR_10_BANDS}; got {tuple(sensor.bands)}"
            )

        runtime_cfg = _resolve_satmaepp_s2_runtime_config(
            model_config=model_config,
            default_ckpt_repo=self.DEFAULT_CKPT_REPO,
            default_ckpt_file=self.DEFAULT_CKPT_FILE,
            default_model_fn=self.DEFAULT_MODEL_FN,
            default_image_size=self.DEFAULT_IMAGE_SIZE,
            default_patch_size=self.DEFAULT_PATCH_SIZE,
        )
        ckpt_repo = str(runtime_cfg["ckpt_repo"])
        ckpt_file = str(runtime_cfg["ckpt_file"])
        model_fn = str(runtime_cfg["model_fn"])
        image_size = int(runtime_cfg["image_size"])
        patch_size = int(runtime_cfg["patch_size"])
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
            raws.append(raw)

        # Fetch-square ROI window per item, carried in fetch_metas when the caller
        # prefetched a square input. The token grid is cropped back to it
        # (full frame / missing meta → no-op, behavior unchanged).
        geo_rois = [
            geo_roi_from_meta(
                fetch_metas[k] if (fetch_metas is not None and k < len(fetch_metas)) else None
            )
            for k in range(len(input_chws))
        ]

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
        any_cropped = any(not roi_is_full(gr) for gr in geo_rois)
        # group_reduce is needed for grid output, and also for pooled output when an
        # ROI was enlarged to a square (pooled then derives from the cropped grid).
        group_reduce = self._resolve_group_reduce() if (want_grid or any_cropped) else "mean"
        if want_grid:
            import_xarray()  # fail fast before batch inference

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
                        "variant": runtime_cfg["variant"],
                        **wmeta,
                    },
                )

                cropped_to_roi = not roi_is_full(geo_rois[i])

                if output.mode == "pooled":
                    if cropped_to_roi:
                        grid, _, cls_removed, _ = _satmaepp_s2_grid(
                            tokens, group_reduce=group_reduce
                        )
                        _, vec = crop_grid_and_pool(grid, geo_rois[i], pooling=output.pooling)
                        assert vec is not None
                        meta.update(
                            {
                                "pooling": f"roi_grid_{output.pooling}",
                                "cls_removed": bool(cls_removed),
                            }
                        )
                    else:
                        vec, cls_removed = _satmaepp_s2_pool(tokens, output.pooling)
                        meta.update(
                            {
                                "pooling": f"group_tokens_{output.pooling}",
                                "cls_removed": bool(cls_removed),
                            }
                        )
                    out[i] = Embedding(data=vec.astype(np.float32), meta=meta)
                elif output.mode == "grid":
                    grid, (h, w), cls_removed, gmeta = _satmaepp_s2_grid(
                        tokens, group_reduce=group_reduce
                    )
                    if cropped_to_roi:
                        grid = crop_grid_to_roi(grid, geo_rois[i])
                        h, w = int(grid.shape[1]), int(grid.shape[2])
                    meta.update(
                        {
                            "grid_hw": (h, w),
                            "grid_kind": "spatial_tokens_aggregated_over_channel_groups",
                            "cls_removed": bool(cls_removed),
                            **gmeta,
                        }
                    )
                    da = grid_to_dataarray(grid, meta=meta)
                    out[i] = Embedding(data=da, meta=meta)
                else:
                    raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError("satmaepp_s2_10b batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

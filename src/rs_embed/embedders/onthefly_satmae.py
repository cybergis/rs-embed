from __future__ import annotations

import os
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
from ..tools.shape import (
    crop_grid_to_roi,
    geo_roi_from_meta,
    roi_is_full,
    square_fetch_batch,
)
from ..tools.spatial import square_spatial
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


def resize_rgb_u8(rgb_u8, out_size):
    from PIL import Image

    if rgb_u8.shape[0] == out_size and rgb_u8.shape[1] == out_size:
        return rgb_u8
    im = Image.fromarray(rgb_u8, mode="RGB")
    return np.array(im.resize((out_size, out_size), resample=Image.BICUBIC), dtype=np.uint8)


# fMoW RGB normalization — the rshf SatMAE config defaults. Used as a fallback
# when a checkpoint does not carry img_mean/img_std on its config.
_SATMAE_RGB_MEAN = (0.4182007312774658, 0.4214799106121063, 0.3991275727748871)
_SATMAE_RGB_STD = (0.28774282336235046, 0.27541765570640564, 0.2764017581939697)


def _satmae_norm_from_model(model) -> tuple[tuple[float, ...], tuple[float, ...]]:
    cfg = getattr(model, "config", None)
    mean = getattr(cfg, "img_mean", None)
    std = getattr(cfg, "img_std", None)
    if mean is None or std is None:
        return _SATMAE_RGB_MEAN, _SATMAE_RGB_STD
    return tuple(float(x) for x in mean), tuple(float(x) for x in std)


def _satmae_preprocess_info(model, image_size: int) -> dict[str, Any]:
    mean, std = _satmae_norm_from_model(model)
    # Resize the (square) input straight to image_size with NO center crop, so the
    # token grid covers the whole fetched square and the ROI crop / tile stitch stay
    # aligned. This intentionally diverges from the wrapper's transform(), which does
    # Resize(256)+CenterCrop(224) and would drop ~12.5% of the FOV per tile/ROI.
    return {
        "preprocess_name": "direct",
        "norm_mean": mean,
        "norm_std": std,
        "resize_to": int(image_size),
        "center_crop": None,
    }


def _satmae_preprocess_tensor_batch(model, rgb_u8_batch: list[np.ndarray], *, image_size: int):
    ensure_torch()
    import torch
    from PIL import Image
    from torchvision import transforms

    mean, std = _satmae_norm_from_model(model)
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (int(image_size), int(image_size)),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    xs = []
    for i, rgb_u8 in enumerate(rgb_u8_batch):
        if not isinstance(rgb_u8, np.ndarray) or rgb_u8.ndim != 3 or int(rgb_u8.shape[2]) != 3:
            raise ModelError(
                "SatMAE preprocessing expects uint8 HWC RGB arrays; "
                f"got shape={getattr(rgb_u8, 'shape', None)} at index={i}."
            )
        if rgb_u8.dtype != np.uint8:
            raise ModelError(
                "SatMAE preprocessing expects dtype=uint8; "
                f"got dtype={getattr(rgb_u8, 'dtype', None)} at index={i}."
            )
        xs.append(preprocess(Image.fromarray(rgb_u8, mode="RGB")))
    if not xs:
        return torch.empty((0, 3, int(image_size), int(image_size)))
    return torch.stack(xs, dim=0)


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


def build_token_embedding(tokens, *, geo_roi, output, meta):
    """Turn ``[N,D]`` tokens into an Embedding, cropping the patch grid to the ROI.

    When the input was enlarged to a square at fetch time (``geo_roi`` is a
    sub-window) the patch-token grid is cropped back to the ROI before pooling /
    emitting, so neighbourhood context fetched only to square the input does not
    leak into the result. When ``geo_roi`` is the full frame this reproduces the
    legacy token-pool / full-grid behaviour exactly. ``meta`` is mutated in place
    with the chosen pooling / grid provenance.
    """
    cropped = not roi_is_full(geo_roi)

    if output.mode == "pooled":
        if cropped:
            grid, _hw, cls_removed = tokens_to_grid_dhw(tokens)
            g = crop_grid_to_roi(grid, geo_roi)
            reduce = g.max if output.pooling == "max" else g.mean
            vec = reduce(axis=(1, 2)).astype("float32")
            meta.update({"pooling": f"roi_grid_{output.pooling}", "cls_removed": bool(cls_removed)})
        else:
            vec, cls_removed = pool_from_tokens(tokens, output.pooling)
            meta.update({"pooling": f"patch_{output.pooling}", "cls_removed": bool(cls_removed)})
        return Embedding(data=vec, meta=meta)

    if output.mode == "grid":
        grid, (h, w), cls_removed = tokens_to_grid_dhw(tokens)
        if cropped:
            grid = crop_grid_to_roi(grid, geo_roi)
            h, w = int(grid.shape[1]), int(grid.shape[2])
        meta.update(
            {"grid_hw": (h, w), "grid_kind": "patch_tokens", "cls_removed": bool(cls_removed)}
        )
        try:
            import xarray as xr
        except Exception as e:
            raise ModelError("grid output requires xarray. Install: pip install xarray") from e
        da = xr.DataArray(
            grid,
            dims=("d", "y", "x"),
            coords={"d": np.arange(grid.shape[0]), "y": np.arange(h), "x": np.arange(w)},
            name="embedding",
            attrs=meta,
        )
        return Embedding(data=da, meta=meta)

    raise ModelError(f"Unknown output mode: {output.mode}")


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

    xb = _satmae_preprocess_tensor_batch(model, rgb_u8_batch, image_size=image_size).to(device)

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

    Strategy aligned:
      - pooled: pool patch tokens by OutputSpec.pooling (exclude CLS if present)
      - grid: patch token grid (exclude CLS if present)
    """

    # SatMAE needs a square token grid → base.fetch_input enlarges a rectangular
    # ROI to a square of real imagery; the output is cropped back to the ROI.
    _requires_square_input = True
    DEFAULT_MODEL_ID = "MVRL/satmae-vitlarge-fmow-pretrain-800"
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
            },
        }

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
        fetch_meta: dict[str, Any] | None = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("satmae_rgb expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self.input_spec.to_sensor_spec()

        model_id = os.environ.get("RS_EMBED_SATMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SATMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))

        t = temporal_to_range(temporal)
        # Fetch-square ROI window: from the direct fetch, or carried in fetch_meta
        # when the API prefetched a square. The output grid is cropped back to it.
        geo_roi = geo_roi_from_meta(fetch_meta)
        # Fetch RGB patch (optionally reuse pre-fetched raw patch)
        if input_chw is None:
            spatial, geo_roi = square_spatial(spatial)  # enlarge rectangle to square
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
        pp_info = _satmae_preprocess_info(model, image_size)
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
                **pp_info,
            },
        )

        return build_token_embedding(tokens, geo_roi=geo_roi, output=output, meta=meta)

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
            sensor = self.input_spec.to_sensor_spec()

        model_id = os.environ.get("RS_EMBED_SATMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SATMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        t = temporal_to_range(temporal)

        provider = self._get_provider(backend)
        n = len(spatials)
        # Square-fetch each ROI; geo_rois[i] is the ROI window within the square so
        # each item's token grid is cropped back to its ROI after the forward.
        rgb_u8_all, geo_rois = square_fetch_batch(
            spatials,
            lambda sq: fetch_s2_rgb_u8_from_provider(
                spatial=sq,
                temporal=t,
                sensor=sensor,
                out_size=image_size,
                provider=provider,
            ),
            max_workers=self._resolve_fetch_workers(n),
        )

        model, wmeta = _load_satmae(model_id=model_id, device=device)
        dev = wmeta.get("device", device)
        infer_bs = self._resolve_infer_batch(str(dev))
        pp_info = _satmae_preprocess_info(model, image_size)

        out: list[Embedding | None] = [None] * n
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
                        **pp_info,
                    },
                )
                out[i] = build_token_embedding(
                    tokens, geo_roi=geo_rois[i], output=output, meta=meta
                )

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
            sensor = self.input_spec.to_sensor_spec()

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
        pp_info = _satmae_preprocess_info(model, image_size)

        out: list[Embedding | None] = [None] * len(spatials)
        # User-supplied inputs carry no fetch-square ROI, so each output covers the
        # whole frame (build_token_embedding reproduces the legacy token path).
        geo_roi = geo_roi_from_meta(None)

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
                        **pp_info,
                    },
                )
                out[i] = build_token_embedding(tokens, geo_roi=geo_roi, output=output, meta=meta)

        if any(e is None for e in out):
            raise ModelError("satmae_rgb batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

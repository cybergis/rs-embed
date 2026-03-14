from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import ModelError
from ..core.specs import TemporalSpec, SensorSpec, SpatialSpec
from ..providers import ProviderBase
from .meta_utils import temporal_to_range, build_meta
from .runtime_utils import create_provider_for_backend, fetch_collection_patch_chw

# -------------------------
# Image resize / provider-backed fetch
# -------------------------
def resize_rgb_u8(rgb_u8: np.ndarray, out_size: int) -> np.ndarray:
    """
    rgb_u8: uint8 HxWx3 -> resized uint8 out_size x out_size x 3
    """
    from PIL import Image

    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ModelError(f"Expected uint8 HxWx3, got {rgb_u8.dtype} {rgb_u8.shape}")
    if rgb_u8.shape[0] == out_size and rgb_u8.shape[1] == out_size:
        return rgb_u8
    im = Image.fromarray(rgb_u8, mode="RGB")
    im = im.resize((out_size, out_size), resample=Image.BICUBIC)
    return np.array(im, dtype=np.uint8)

def _s2_rgb_u8_from_chw(s2_chw: np.ndarray) -> np.ndarray:
    """s2_chw: [3,H,W] float in [0,1] -> uint8 [H,W,3]."""
    if s2_chw.ndim != 3 or s2_chw.shape[0] != 3:
        raise ModelError(f"Expected S2 RGB CHW with 3 bands, got shape={s2_chw.shape}")
    x = np.clip(s2_chw, 0.0, 1.0)
    return (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)

def fetch_s2_rgb_u8_from_provider(
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    sensor: SensorSpec,
    out_size: int,
    provider: ProviderBase | None = None,
    backend: str = "auto",
    default_temporal: tuple[str, str] = ("2022-06-01", "2022-09-01"),
) -> np.ndarray:
    """
    Single source of truth for "ROI+time -> uint8 RGB patch".

    Uses shared runtime helpers to fetch S2 RGB CHW, converts to uint8 HWC,
    then resizes to out_size.
    """
    t = temporal_to_range(temporal, default_temporal)

    if provider is None:
        p = create_provider_for_backend(backend, allow_auto=True)
    else:
        p = provider
    p.ensure_ready()

    s2_raw = fetch_collection_patch_chw(
        p,
        spatial=spatial,
        temporal=t,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=sensor.scale_m,
        cloudy_pct=sensor.cloudy_pct,
        composite=sensor.composite,
        fill_value=0.0,
    )
    s2_chw = np.clip(s2_raw / 10000.0, 0.0, 1.0).astype(np.float32)

    # Optional: inspect on-the-fly provider input (shared by multiple embedders)
    from ..tools.inspection import maybe_inspect_chw, checks_should_raise

    report = maybe_inspect_chw(
        s2_chw,
        sensor=sensor,
        name="provider_s2_rgb_chw",
        expected_channels=3,
        value_range=(0.0, 1.0),
        fill_value=0.0,
        meta=None,
    )
    if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
        raise ModelError("Provider input inspection failed: " + "; ".join(report.get("issues", [])))

    rgb_u8 = _s2_rgb_u8_from_chw(s2_chw)
    return resize_rgb_u8(rgb_u8, out_size)

# -------------------------
# Tokens semantics (CLS, pooling, grid)
# -------------------------
def infer_has_cls(n_tokens: int) -> bool:
    """
    Heuristic: if (N-1) is perfect square, treat token[0] as CLS.
    """
    if n_tokens < 2:
        return False
    p = n_tokens - 1
    h = int(np.sqrt(p))
    return h * h == p

def split_cls_patch(
    tokens: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    """
    tokens: [N,D]
    Returns: (cls_token [D] or None, patch_tokens [P,D], has_cls)
    """
    if tokens.ndim != 2:
        raise ModelError(f"Expected tokens [N,D], got {tokens.shape}")
    n, _ = tokens.shape
    has = infer_has_cls(n)
    if has:
        return tokens[0], tokens[1:], True
    return None, tokens, False

def tokens_to_grid_dhw(tokens: np.ndarray) -> tuple[np.ndarray, tuple[int, int], bool]:
    """
    tokens: [N,D]
    Returns:
      - grid [D,H,W] (patch tokens reshaped)
      - (H,W)
      - cls_removed flag
    """
    cls, patch, has_cls = split_cls_patch(tokens)
    p, d = patch.shape
    h = int(np.sqrt(p))
    w = h
    if h * w != p:
        raise ModelError(f"Patch token count {p} is not a square; cannot form grid.")
    grid = patch.reshape(h, w, d).transpose(2, 0, 1).astype(np.float32)
    return grid, (h, w), has_cls

def pool_from_tokens(tokens: np.ndarray, pooling: str) -> tuple[np.ndarray, bool]:
    """
    Unified pooling: always pool over patch tokens (exclude CLS if present).
    pooling: 'mean' | 'max'
    Returns pooled vec [D] and cls_removed flag.
    """
    _, patch, has_cls = split_cls_patch(tokens)
    if patch.shape[0] == 0:
        # only CLS existed; fallback to first token
        return tokens[0].astype(np.float32), has_cls
    if pooling == "mean":
        return patch.mean(axis=0).astype(np.float32), has_cls
    if pooling == "max":
        return patch.max(axis=0).astype(np.float32), has_cls
    raise ModelError(f"Unknown pooling='{pooling}' (expected 'mean' or 'max').")

# -------------------------
# Torch preprocessing
# -------------------------
def ensure_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise ModelError("This embedder requires torch installed.") from e

def maybe_use_model_transform(model: Any, rgb_u8: np.ndarray, image_size: int):
    """
    If model exposes `transform(img_float32, image_size)->Tensor[C,H,W]`, use it.
    Returns torch.Tensor [B,3,H,W] on CPU.
    """
    ensure_torch()
    import torch

    if hasattr(model, "transform") and callable(getattr(model, "transform")):
        x = model.transform(rgb_u8.astype(np.float32), image_size)
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            return x.unsqueeze(0)
    return None

def rgb_u8_to_tensor_clipnorm(rgb_u8: np.ndarray, image_size: int):
    """
    Standard preprocessing (CLIP mean/std). Works well for open_clip-like models,
    and is a reasonable default for ViT encoders when wrapper lacks transform().
    Returns torch.Tensor [B,3,H,W] on CPU.
    """
    ensure_torch()
    from torchvision import transforms
    from PIL import Image

    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    img = Image.fromarray(rgb_u8, mode="RGB")
    x = preprocess(img).unsqueeze(0)
    return x

# -------------------------
# Meta helpers
# -------------------------
def base_meta(
    *,
    model_name: str,
    hf_id: str,
    backend: str,
    image_size: int,
    sensor: SensorSpec,
    temporal: TemporalSpec | None = None,
    source: str | None = None,
    input_time: str | None = None,
    embed_type: str = "on_the_fly",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = build_meta(
        model=model_name,
        kind=embed_type,
        backend=backend,
        source=source or getattr(sensor, "collection", None),
        sensor=sensor,
        temporal=temporal,
        image_size=image_size,
        input_time=input_time,
        extra=None,
    )
    extra_fields = {"hf_id": hf_id}
    if extra:
        extra_fields.update(extra)
    base.update(extra_fields)
    return base

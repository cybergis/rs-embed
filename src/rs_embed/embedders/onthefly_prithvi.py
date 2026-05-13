from __future__ import annotations

import json
import math
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
from ..providers import ProviderBase
from ..providers.fetch import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
)
from ..providers.resolution import (
    is_provider_backend,
)
from ..tools.runtime import (
    load_cached_with_device as _load_cached_with_device,
)
from .base import EmbedderBase
from .config import model_config_value
from .meta import build_meta, temporal_midpoint_str, temporal_to_range


def ensure_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise ModelError("This embedder requires torch installed.") from e


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


# -------------------------
# Provider: Sentinel-2 -> Prithvi 6-band (CHW float32 in [0,1])
# -------------------------
PRITHVI_S2_BANDS_SRC = ["B2", "B3", "B4", "B8", "B11", "B12"]
PRITHVI_S2_BANDS_DST = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
_PRITHVI_VARIANT_TO_MODEL_KEY = {
    "prithvi_eo_v2_100_tl": "prithvi_eo_v2_100_tl",
    "100_tl": "prithvi_eo_v2_100_tl",
    "100m_tl": "prithvi_eo_v2_100_tl",
    "prithvi_eo_v2_300_tl": "prithvi_eo_v2_300_tl",
    "300_tl": "prithvi_eo_v2_300_tl",
    "300m_tl": "prithvi_eo_v2_300_tl",
    "prithvi_eo_v2_600_tl": "prithvi_eo_v2_600_tl",
    "600_tl": "prithvi_eo_v2_600_tl",
    "600m_tl": "prithvi_eo_v2_600_tl",
}
_PRITHVI_HF_SPECS = {
    "prithvi_eo_v2_100_tl": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL",
        "checkpoint": "Prithvi_EO_V2_100M_TL.pt",
    },
    "prithvi_eo_v2_300_tl": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
        "checkpoint": "Prithvi_EO_V2_300M_TL.pt",
    },
    "prithvi_eo_v2_600_tl": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
        "checkpoint": "Prithvi_EO_V2_600M_TL.pt",
    },
}


def _normalize_prithvi_variant(variant: Any) -> str:
    raw = str(variant).strip().lower()
    resolved = _PRITHVI_VARIANT_TO_MODEL_KEY.get(raw)
    if resolved is None:
        raise ModelError(
            f"Unknown Prithvi variant='{variant}' "
            "(expected one of: prithvi_eo_v2_100_tl, prithvi_eo_v2_300_tl, prithvi_eo_v2_600_tl)."
        )
    return resolved


def _resolve_prithvi_model_key(
    *,
    model_config: dict[str, Any] | None,
    default_model_key: str,
) -> tuple[str, str]:
    variant_v = model_config_value(model_config, "variant")
    if variant_v is not None:
        model_key = _normalize_prithvi_variant(variant_v)
        return model_key, model_key

    model_key = (
        os.environ.get("RS_EMBED_PRITHVI_KEY", default_model_key).strip() or default_model_key
    )
    return str(model_key), str(model_key)


def _resolve_prithvi_hf_spec(model_key: str) -> dict[str, str]:
    spec = _PRITHVI_HF_SPECS.get(str(model_key).strip())
    if spec is None:
        raise ModelError(
            f"Unknown Prithvi model_key='{model_key}' "
            "(expected one of: prithvi_eo_v2_100_tl, prithvi_eo_v2_300_tl, prithvi_eo_v2_600_tl)."
        )
    return dict(spec)


def _prithvi_cache_dir() -> str | None:
    raw = str(os.environ.get("RS_EMBED_PRITHVI_CACHE_DIR", "")).strip()
    return raw or None


def _torch_load_checkpoint_compat(path: str):
    ensure_torch()
    import torch

    weights_only = str(os.environ.get("RS_EMBED_PRITHVI_WEIGHTS_ONLY", "1")).strip() not in (
        "0",
        "false",
        "False",
    )
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


@lru_cache(maxsize=16)
def _download_prithvi_file(repo_id: str, filename: str, cache_dir: str | None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise ModelError(
            "Prithvi checkpoint download requires huggingface_hub. "
            "Install: pip install huggingface_hub"
        ) from e
    return str(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir))


def _load_prithvi_module():
    from ._vendor.prithvi_mae import PrithviMAE

    return PrithviMAE


def _fetch_s2_prithvi6_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 30,
    cloudy_pct: int = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Returns CHW float32 [6,H,W] normalized to [0,1] from S2 SR (scaled by 1/10000).
    Uses provider.get_region_3857(spatial) to define the sampling rectangle.
    """
    # Use semantic aliases (BLUE/GREEN/...) so provider alias resolution stays centralized.
    x_chw = _fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(PRITHVI_S2_BANDS_DST),
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        fill_value=fill_value,
    )

    # S2 SR scaling: 0..10000
    x_chw = x_chw / 10000.0
    x_chw = np.clip(x_chw, 0.0, 1.0)
    x_chw = np.nan_to_num(x_chw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x_chw


def _pad_chw_to_multiple(x_chw: np.ndarray, mult: int = 16, value: float = 0.0) -> np.ndarray:
    """
    Pad CHW to make H and W divisible by mult.
    Pads on bottom and right only.
    """
    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW, got {x_chw.shape}")
    c, h, w = x_chw.shape
    nh = int(math.ceil(h / mult) * mult)
    nw = int(math.ceil(w / mult) * mult)
    if nh == h and nw == w:
        return x_chw
    out = np.full((c, nh, nw), float(value), dtype=np.float32)
    out[:, :h, :w] = x_chw.astype(np.float32)
    return out


def _resize_chw(x_chw: np.ndarray, *, size: int = 224) -> np.ndarray:
    """Resize CHW to square (size,size) using bilinear interpolation."""
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW array, got {x_chw.shape}")
    x = torch.from_numpy(x_chw.astype(np.float32, copy=False)).unsqueeze(0)
    y = F.interpolate(x, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    return y[0].detach().cpu().numpy().astype(np.float32)


def _prepare_prithvi_chw(
    x_chw: np.ndarray,
    *,
    fill_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Prepare CHW input before Prithvi forward.

    RS_EMBED_PRITHVI_PREP:
      - "resize": resize to RS_EMBED_PRITHVI_IMG (default 224)
      - "pad": pad H/W to RS_EMBED_PRITHVI_PATCH_MULT (default 16, legacy behavior)
    """
    prep = os.environ.get("RS_EMBED_PRITHVI_PREP", "resize").strip().lower()
    patch_mult = max(1, int(os.environ.get("RS_EMBED_PRITHVI_PATCH_MULT", "16")))
    target_size = max(16, int(os.environ.get("RS_EMBED_PRITHVI_IMG", "224")))

    if prep == "resize":
        y = _resize_chw(x_chw, size=target_size)
    elif prep == "pad":
        y = _pad_chw_to_multiple(x_chw, mult=patch_mult, value=float(fill_value))
    else:
        raise ModelError(f"Unknown RS_EMBED_PRITHVI_PREP='{prep}'. Use 'resize' or 'pad'.")

    return y, {
        "prep_mode": prep,
        "patch_mult": int(patch_mult),
        "target_image_size": int(target_size),
    }


def _spatial_center_lon_lat(spatial: SpatialSpec) -> tuple[float, float]:
    from ..core.specs import BBox, PointBuffer  # local import to avoid cycles

    if isinstance(spatial, BBox):
        spatial.validate()
        lon = (spatial.minlon + spatial.maxlon) / 2
        lat = (spatial.minlat + spatial.maxlat) / 2
        return float(lon), float(lat)
    if isinstance(spatial, PointBuffer):
        spatial.validate()
        return float(spatial.lon), float(spatial.lat)
    raise ModelError(f"Unsupported SpatialSpec: {type(spatial)}")


# -------------------------
# Prithvi model loading (vendored HF runtime)
# -------------------------


@lru_cache(maxsize=8)
def _load_prithvi_cached(
    model_key: str,
    pretrained: bool,
    bands: tuple[str, ...],
    num_frames: int,
    coords_encoding: tuple[str, ...],
    dev: str,
):
    ensure_torch()
    spec = _resolve_prithvi_hf_spec(model_key)
    cache_dir = _prithvi_cache_dir()
    cfg_path = _download_prithvi_file(spec["repo_id"], "config.json", cache_dir)
    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f).get("pretrained_cfg", {})

    if not isinstance(config, dict) or not config:
        raise ModelError(f"Invalid Prithvi config at {cfg_path!r}.")

    config = dict(config)
    config["num_frames"] = int(num_frames)
    config["in_chans"] = int(len(bands))
    config["coords_encoding"] = list(coords_encoding)
    if isinstance(config.get("patch_size"), list):
        config["patch_size"] = tuple(int(v) for v in config["patch_size"])

    try:
        PrithviMAE = _load_prithvi_module()
        m = PrithviMAE(**config)
    except Exception as e:
        raise ModelError(
            f"Failed to initialize vendored Prithvi runtime for '{model_key}': "
            f"{type(e).__name__}: {e}"
        ) from e

    ckpt_path = None
    if pretrained:
        ckpt_path = _download_prithvi_file(spec["repo_id"], spec["checkpoint"], cache_dir)
        state_dict = _torch_load_checkpoint_compat(ckpt_path)
        if not isinstance(state_dict, dict):
            raise ModelError(
                f"Unexpected Prithvi checkpoint object from {ckpt_path!r}: {type(state_dict)}"
            )
        state_dict = dict(state_dict)
        # HF inference.py replaces fixed positional embeddings with runtime-sized tensors.
        for key in list(state_dict.keys()):
            if key == "encoder.pos_embed":
                state_dict[key] = m.encoder.pos_embed
            elif key == "decoder.decoder_pos_embed":
                state_dict[key] = m.decoder.decoder_pos_embed
        try:
            m.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ModelError(
                f"Failed to load Prithvi checkpoint '{ckpt_path}': {type(e).__name__}: {e}"
            ) from e

    try:
        m = m.to(dev).eval()
    except Exception as _e:
        pass

    meta = {
        "model_key": model_key,
        "repo_id": spec["repo_id"],
        "checkpoint": spec["checkpoint"],
        "config_path": cfg_path,
        "checkpoint_path": ckpt_path,
        "pretrained": bool(pretrained),
        "bands": tuple(bands),
        "num_frames": int(num_frames),
        "coords_encoding": tuple(coords_encoding),
        "device": dev,
    }
    return m, meta


def _load_prithvi(
    model_key: str,
    *,
    pretrained: bool,
    bands: tuple[str, ...],
    num_frames: int,
    coords_encoding: tuple[str, ...],
    device: str = "auto",
):
    """Load (and cache) a vendored Prithvi backbone.

    Returns: (model, meta, resolved_device)
    """
    (loaded, dev) = _load_cached_with_device(
        _load_prithvi_cached,
        device=device,
        model_key=model_key,
        pretrained=bool(pretrained),
        bands=tuple(bands),
        num_frames=int(num_frames),
        coords_encoding=tuple(coords_encoding),
    )
    m, meta = loaded
    return m, meta, dev


def _prithvi_forward_tokens(
    model,
    x_chw: np.ndarray,
    *,
    lon: float,
    lat: float,
    date_str: str,
    device: str,
) -> np.ndarray:
    """
    Run Prithvi forward and return token sequence [N,D] (may include CLS).
    """
    ensure_torch()
    import pandas as pd
    import torch

    if x_chw.ndim != 3 or x_chw.shape[0] != 6:
        raise ModelError(f"Prithvi expects 6-band CHW, got {x_chw.shape}")

    x = torch.from_numpy(x_chw).unsqueeze(0).to(device)  # [1,6,H,W]

    d = pd.to_datetime(date_str)
    temporal_coords = torch.tensor(
        [[[float(d.year), float(d.dayofyear)]]], dtype=torch.float32, device=device
    )  # [1,1,2]
    # Vendored Prithvi runtime expects location_coords in (lat, lon) order.
    location_coords = torch.tensor(
        [[float(lat), float(lon)]], dtype=torch.float32, device=device
    )  # [1,2]

    with torch.no_grad():
        if hasattr(model, "forward_features"):
            out = model.forward_features(
                x,
                temporal_coords=temporal_coords,
                location_coords=location_coords,
            )
        else:
            out = model(x, temporal_coords=temporal_coords, location_coords=location_coords)

    # normalize output -> tokens
    tokens = None
    if isinstance(out, (tuple, list)):
        tokens = out[-1]
    elif hasattr(out, "last_hidden_state"):
        tokens = out.last_hidden_state
    elif isinstance(out, dict):
        tokens = out.get("tokens") or out.get("last_hidden_state") or out.get("hidden_states")
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[-1]
    else:
        tokens = out

    if tokens is None:
        raise ModelError("Prithvi forward did not return tokens.")

    if hasattr(tokens, "ndim") and tokens.ndim == 3:
        # [B,N,D]
        return tokens[0].detach().float().cpu().numpy().astype(np.float32)

    raise ModelError(
        f"Unexpected Prithvi tokens shape/type: {type(tokens)} {getattr(tokens, 'shape', None)}"
    )


def _prithvi_forward_tokens_batch(
    model,
    x_bchw: np.ndarray,
    *,
    lon_lat_batch: list[tuple[float, float]],
    date_str_batch: list[str],
    device: str,
) -> list[np.ndarray]:
    """Batch Prithvi forward for [B,6,H,W] inputs."""
    ensure_torch()
    import pandas as pd
    import torch

    if x_bchw.ndim != 4 or x_bchw.shape[1] != 6:
        raise ModelError(f"Prithvi expects BCHW with C=6, got {x_bchw.shape}")
    bsz = int(x_bchw.shape[0])
    if len(lon_lat_batch) != bsz or len(date_str_batch) != bsz:
        raise ModelError("lon_lat_batch/date_str_batch size mismatch with input batch.")

    xb = torch.from_numpy(x_bchw).to(device)
    tcoords = []
    lcoords = []
    for i in range(bsz):
        d = pd.to_datetime(date_str_batch[i])
        tcoords.append([float(d.year), float(d.dayofyear)])
        lon, lat = lon_lat_batch[i]
        lcoords.append([float(lat), float(lon)])
    temporal_coords = torch.tensor(tcoords, dtype=torch.float32, device=device).unsqueeze(
        1
    )  # [B,1,2]
    location_coords = torch.tensor(lcoords, dtype=torch.float32, device=device)  # [B,2]

    with torch.inference_mode():
        if hasattr(model, "forward_features"):
            out = model.forward_features(
                xb,
                temporal_coords=temporal_coords,
                location_coords=location_coords,
            )
        else:
            out = model(xb, temporal_coords=temporal_coords, location_coords=location_coords)

    tokens = None
    if isinstance(out, (tuple, list)):
        tokens = out[-1]
    elif hasattr(out, "last_hidden_state"):
        tokens = out.last_hidden_state
    elif isinstance(out, dict):
        tokens = out.get("tokens") or out.get("last_hidden_state") or out.get("hidden_states")
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[-1]
    else:
        tokens = out

    if tokens is None:
        raise ModelError("Prithvi forward did not return tokens.")
    if (not hasattr(tokens, "ndim")) or int(tokens.ndim) != 3:
        raise ModelError(
            f"Unexpected Prithvi batch tokens shape/type: {type(tokens)} {getattr(tokens, 'shape', None)}"
        )
    if int(tokens.shape[0]) != bsz:
        raise ModelError(f"Prithvi batch mismatch: got B={int(tokens.shape[0])}, expected {bsz}")

    toks_np = tokens.detach().float().cpu().numpy().astype(np.float32)  # [B,N,D]
    return [toks_np[i] for i in range(bsz)]


# -------------------------
# Embedder
# -------------------------
@register("prithvi")
class PrithviEOV2S2_6B_Embedder(EmbedderBase):
    """
    Prithvi-EO v2 (vendored HF runtime) on-the-fly embeddings from Sentinel-2 6-band patch.

    Inputs:
      - spatial: BBox/PointBuffer (EPSG:4326)
      - temporal: range/year (year->full year)
      - sensor: controls provider fetch (scale/cloudy/composite)

    Outputs:
      - pooled: patch-token mean/max (exclude CLS if present)
      - grid: token map [D,H,W] (exclude CLS if present)
    """

    DEFAULT_MODEL_KEY = "prithvi_eo_v2_100_tl"
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_IMAGE_SCALE_M = 30  # notebook used 30m
    DEFAULT_CLOUDY_PCT = 30
    DEFAULT_COMPOSITE = "median"
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 4
    DEFAULT_BATCH_CUDA = 16

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(PRITHVI_S2_BANDS_DST),
        scale_m=30,
        cloudy_pct=30,
        expected_channels=6,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["provider"],
            "model_key_default": self.DEFAULT_MODEL_KEY,
            "input_bands": PRITHVI_S2_BANDS_DST,
            "output": ["pooled", "grid"],
            "defaults": {
                "model_key": self.DEFAULT_MODEL_KEY,
                "variant": self.DEFAULT_MODEL_KEY,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "scale_m": self.input_spec.scale_m,
                "cloudy_pct": self.input_spec.cloudy_pct,
                "composite": self.input_spec.composite,
                "fill_value": self.input_spec.fill_value,
            },
            "model_config": {
                "variant": {
                    "type": "string",
                    "default": self.DEFAULT_MODEL_KEY,
                    "choices": [
                        "prithvi_eo_v2_100_tl",
                        "prithvi_eo_v2_300_tl",
                        "prithvi_eo_v2_600_tl",
                    ],
                }
            },
            "notes": [
                "Uses vendored PrithviMAE runtime with weights downloaded from Hugging Face.",
                "Requires temporal_coords (year, dayofyear) and location_coords (lat, lon).",
            ],
        }

    def _default_sensor(self) -> SensorSpec:
        return self.input_spec.to_sensor_spec()

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_PRITHVI_FETCH_WORKERS",
                str(PrithviEOV2S2_6B_Embedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default_bs = (
            PrithviEOV2S2_6B_Embedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else PrithviEOV2S2_6B_Embedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_PRITHVI_BATCH_SIZE", str(default_bs)))
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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("prithvi_eo_v2_s2_6b expects a provider backend (or 'auto').")

        # Defaults for Prithvi inputs
        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)  # normalize to range

        # Load model
        model_key, variant = _resolve_prithvi_model_key(
            model_config=model_config,
            default_model_key=self.DEFAULT_MODEL_KEY,
        )
        pretrained = os.environ.get("RS_EMBED_PRITHVI_PRETRAINED", "1").strip() not in (
            "0",
            "false",
            "False",
        )
        coords_encoding = ("time", "location")
        num_frames = 1

        model, wmeta, dev = _load_prithvi(
            model_key,
            pretrained=pretrained,
            bands=tuple(PRITHVI_S2_BANDS_DST),
            num_frames=num_frames,
            coords_encoding=coords_encoding,
            device=device,
        )

        # Fetch S2 6-band patch from provider
        provider = self._get_provider(backend)

        # Fetch S2 6-band patch from provider (optionally reuse pre-fetched raw patch)
        if input_chw is None:
            x_chw = _fetch_s2_prithvi6_chw(
                provider,
                spatial=spatial,
                temporal=t,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
            )
        else:
            # input_chw expected to be raw S2 SR values (0..10000) in band order sensor.bands
            if input_chw.ndim != 3 or input_chw.shape[0] != 6:
                raise ModelError(
                    f"input_chw must be CHW with 6 bands for prithvi, got {getattr(input_chw, 'shape', None)}"
                )
            x_chw = input_chw.astype(np.float32) / 10000.0
            x_chw = np.clip(x_chw, 0.0, 1.0)
            x_chw = np.nan_to_num(x_chw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Optional: inspect on-the-fly provider input
        from ..tools.inspection import (
            checks_save_dir,
            checks_should_raise,
            maybe_inspect_chw,
            save_quicklook_rgb,
        )

        check_meta: dict[str, Any] = {}
        report = maybe_inspect_chw(
            x_chw,
            sensor=sensor,
            name="provider_s2_prithvi6_chw",
            expected_channels=6,
            value_range=(0.0, 1.0),
            fill_value=float(sensor.fill_value),
            meta=check_meta,
        )
        if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
            raise ModelError(
                "Provider input inspection failed: " + "; ".join(report.get("issues", []))
            )

        # Optional quicklook (RGB from RED/GREEN/BLUE)
        sd = checks_save_dir(sensor)
        if sd and report is not None:
            try:
                import uuid

                fn = f"prithvi_s2_rgb_{uuid.uuid4().hex[:8]}.png"
                save_quicklook_rgb(
                    x_chw,
                    path=os.path.join(sd, fn),
                    bands=(2, 1, 0),
                    vmin=0.0,
                    vmax=1.0,
                )
                check_meta.setdefault("input_checks_artifacts", []).append(
                    {"name": "quicklook_rgb", "path": os.path.join(sd, fn)}
                )
            except Exception as _e:
                check_meta.setdefault("input_checks_artifacts", []).append(
                    {"name": "quicklook_rgb", "error": repr(_e)}
                )
        x_chw, prep_meta = _prepare_prithvi_chw(
            x_chw,
            fill_value=float(sensor.fill_value),
        )

        # coords: use temporal mid-date and ROI center (EPSG:4326).
        lon, lat = _spatial_center_lon_lat(spatial)

        date_str = temporal_midpoint_str(t)

        tokens = _prithvi_forward_tokens(
            model,
            x_chw,
            lon=lon,
            lat=lat,
            date_str=date_str,
            device=dev,
        )  # [N,D] (maybe includes CLS)

        meta = base_meta(
            model_name=self.model_name,
            hf_id=str(wmeta.get("repo_id") or model_key),
            backend=str(backend).lower(),
            image_size=int(x_chw.shape[-1]),  # not fixed 224; depends on ROI/scale
            sensor=sensor,
            temporal=t,
            source=sensor.collection,
            extra={
                "temporal_range": (t.start, t.end),
                "coords_date": date_str,
                "coords_lonlat": (float(lon), float(lat)),
                "tokens_shape": tuple(tokens.shape),
                "model_key": model_key,
                "variant": variant,
                "pretrained": bool(pretrained),
                "coords_encoding": coords_encoding,
                "num_frames": num_frames,
                "input_hw": (int(x_chw.shape[1]), int(x_chw.shape[2])),
                **prep_meta,
                **check_meta,
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
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("prithvi_eo_v2_s2_6b expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)
        provider = self._get_provider(backend)
        n = len(spatials)
        prefetched_raw: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            x_chw = _fetch_s2_prithvi6_chw(
                provider,
                spatial=sp,
                temporal=t,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
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
                raise ModelError(f"Missing prefetched input at index={i} for prithvi_eo_v2_s2_6b.")
            raw_inputs.append(raw)
        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            temporal=temporal,
            sensor=sensor,
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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("prithvi_eo_v2_s2_6b expects a provider backend (or 'auto').")
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        if sensor is None:
            sensor = self._default_sensor()

        t = temporal_to_range(temporal)
        model_key, variant = _resolve_prithvi_model_key(
            model_config=model_config,
            default_model_key=self.DEFAULT_MODEL_KEY,
        )
        pretrained = os.environ.get("RS_EMBED_PRITHVI_PRETRAINED", "1").strip() not in (
            "0",
            "false",
            "False",
        )
        coords_encoding = ("time", "location")
        num_frames = 1
        prep_mode = os.environ.get("RS_EMBED_PRITHVI_PREP", "resize").strip().lower()
        patch_mult = max(1, int(os.environ.get("RS_EMBED_PRITHVI_PATCH_MULT", "16")))
        target_size = max(16, int(os.environ.get("RS_EMBED_PRITHVI_IMG", "224")))

        model, wmeta, dev = _load_prithvi(
            model_key,
            pretrained=pretrained,
            bands=tuple(PRITHVI_S2_BANDS_DST),
            num_frames=num_frames,
            coords_encoding=coords_encoding,
            device=device,
        )
        infer_bs = self._resolve_infer_batch(str(dev))

        x_prepared: list[np.ndarray] = []
        lon_lat: list[tuple[float, float]] = []
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or input_chw.shape[0] != 6:
                raise ModelError(
                    f"input_chw must be CHW with 6 bands for prithvi, got {getattr(input_chw, 'shape', None)} at index={i}"
                )
            x_chw = input_chw.astype(np.float32) / 10000.0
            x_chw = np.clip(x_chw, 0.0, 1.0)
            x_chw = np.nan_to_num(x_chw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            x_chw, _ = _prepare_prithvi_chw(
                x_chw,
                fill_value=float(sensor.fill_value),
            )
            x_prepared.append(x_chw)
            lon_lat.append(_spatial_center_lon_lat(spatials[i]))

        date_str = temporal_midpoint_str(t) or "2022-07-01"
        shape_groups: dict[tuple[int, int, int], list[int]] = {}
        for i, x in enumerate(x_prepared):
            shape_groups.setdefault(tuple(x.shape), []).append(i)

        out: list[Embedding | None] = [None] * len(spatials)
        xr_mod = None
        if output.mode == "grid":
            try:
                import xarray as xr  # type: ignore

                xr_mod = xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

        for idxs in shape_groups.values():
            for s0 in range(0, len(idxs), infer_bs):
                chunk_ids = idxs[s0 : s0 + infer_bs]
                xb = np.stack([x_prepared[i] for i in chunk_ids], axis=0).astype(np.float32)
                toks_list = _prithvi_forward_tokens_batch(
                    model,
                    xb,
                    lon_lat_batch=[lon_lat[i] for i in chunk_ids],
                    date_str_batch=[date_str for _ in chunk_ids],
                    device=dev,
                )
                if len(toks_list) != len(chunk_ids):
                    raise ModelError(
                        f"Prithvi batch output mismatch: {len(toks_list)} != {len(chunk_ids)}"
                    )

                for j, i in enumerate(chunk_ids):
                    tokens = toks_list[j]
                    lon, lat = lon_lat[i]
                    x_chw = x_prepared[i]
                    meta = base_meta(
                        model_name=self.model_name,
                        hf_id=str(wmeta.get("repo_id") or model_key),
                        backend=str(backend).lower(),
                        image_size=int(x_chw.shape[-1]),
                        sensor=sensor,
                        temporal=t,
                        source=sensor.collection,
                        extra={
                            "temporal_range": (t.start, t.end),
                            "coords_date": date_str,
                            "coords_lonlat": (float(lon), float(lat)),
                            "tokens_shape": tuple(tokens.shape),
                            "model_key": model_key,
                            "variant": variant,
                            "pretrained": bool(pretrained),
                            "coords_encoding": coords_encoding,
                            "num_frames": num_frames,
                            "input_hw": (int(x_chw.shape[1]), int(x_chw.shape[2])),
                            "prep_mode": str(prep_mode),
                            "patch_mult": patch_mult,
                            "target_image_size": target_size,
                            "batch_infer": True,
                            "input_override": True,
                            **wmeta,
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
                        continue

                    if output.mode == "grid":
                        grid, (h, w), cls_removed = tokens_to_grid_dhw(tokens)
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
            raise ModelError("prithvi_eo_v2_s2_6b batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

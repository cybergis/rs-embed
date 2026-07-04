# src/rs_embed/embedders/onthefly_clay.py
from __future__ import annotations

import math
import os
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import xarray as xr

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
from ..tools.normalization import (
    coerce_single_input_chw,
)
from ..tools.runtime import (
    load_cached_with_device as _load_cached_with_device,
)
from ..tools.runtime import (
    resolve_device_auto_torch as _resolve_device_auto,
)
from ..tools.shape import (
    crop_grid_to_roi,
    geo_roi_from_meta,
    prepare_square,
    roi_is_full,
    square_fetch_batch,
)
from ..tools.spatial import FULL_WINDOW, square_spatial
from .base import EmbedderBase
from .meta import build_meta, temporal_midpoint_str, temporal_to_range

# -----------------------------
# Defaults: Sentinel-2 L2A (official Clay v1.5 metadata.yaml, sentinel-2-l2a)
# Band order blue..swir22 mapped to GEE S2_SR_HARMONIZED band names.
# -----------------------------
_CLAY_S2_BANDS = [
    "B2",  # blue
    "B3",  # green
    "B4",  # red
    "B5",  # rededge1
    "B6",  # rededge2
    "B7",  # rededge3
    "B8",  # nir
    "B8A",  # nir08
    "B11",  # swir16
    "B12",  # swir22
]

_CLAY_S2_MEAN = np.array(
    [1105.0, 1355.0, 1552.0, 1887.0, 2422.0, 2630.0, 2743.0, 2785.0, 2388.0, 1835.0],
    dtype=np.float32,
)

_CLAY_S2_STD = np.array(
    [1809.0, 1757.0, 1888.0, 1870.0, 1732.0, 1697.0, 1742.0, 1648.0, 1470.0, 1379.0],
    dtype=np.float32,
)

# Central wavelengths (µm) from Clay's metadata.yaml (NOT the generic S2 table:
# Clay conditions its dynamic patch embedding on exactly these values).
_CLAY_S2_WAVELENGTHS_UM = np.array(
    [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19],
    dtype=np.float32,
)

_CLAY_S2_STATS_BY_BAND = {
    band: (float(mean), float(std), float(wave))
    for band, mean, std, wave in zip(
        _CLAY_S2_BANDS, _CLAY_S2_MEAN, _CLAY_S2_STD, _CLAY_S2_WAVELENGTHS_UM, strict=True
    )
}

_CLAY_S2_GSD_M = 10.0
_CLAY_IMAGE_SIZE = 256
_CLAY_PATCH_SIZE = 8


def _resolve_clay_s2_stats(
    bands: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-band (mean, std, wavelength_um) for a subset of Clay's S2 bands."""
    means: list[float] = []
    stds: list[float] = []
    waves: list[float] = []
    for band in bands:
        stats = _CLAY_S2_STATS_BY_BAND.get(str(band))
        if stats is None:
            return None
        mean, std, wave = stats
        means.append(mean)
        stds.append(std)
        waves.append(wave)
    return (
        np.asarray(means, dtype=np.float32),
        np.asarray(stds, dtype=np.float32),
        np.asarray(waves, dtype=np.float32),
    )


def _normalize_clay_input_chw(
    raw_chw: np.ndarray,
    *,
    bands: list[str],
    input_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Official Clay preprocessing: (raw_DN - mean) / std per band.

    Clay normalizes raw surface-reflectance DN values directly with the
    per-band statistics from its metadata.yaml (no 0..1 rescale, no clip),
    mirroring ``torchvision v2.Normalize(mean, std)`` in the upstream
    embedding tutorials. Returns (x, wavelengths_um, norm_meta).
    """
    x = np.asarray(raw_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ModelError(f"{input_name} must be CHW, got shape={getattr(raw_chw, 'shape', None)}")
    if int(x.shape[0]) != len(bands):
        raise ModelError(
            f"{input_name} channel mismatch: got C={int(x.shape[0])}, expected {len(bands)} for bands={bands}"
        )

    stats = _resolve_clay_s2_stats(bands)
    if stats is None:
        raise ModelError(
            "Clay official preprocessing is only defined for Sentinel-2 subsets of "
            f"{_CLAY_S2_BANDS}. Got bands={bands}."
        )

    max_v = float(np.nanmax(x)) if x.size else 0.0
    min_v = float(np.nanmin(x)) if x.size else 0.0
    if min_v >= 0.0 and max_v <= 2.0:
        raise ModelError(
            f"{input_name} appears already normalized (min={min_v:.4f}, max={max_v:.4f}). "
            "Clay expects raw Sentinel-2 SR values in approximately [0,10000]."
        )

    mean, std, waves = stats
    std = np.maximum(std, 1e-6)
    x = (x - mean[:, None, None]) / std[:, None, None]
    meta = {
        "normalization": "official_clay_metadata_s2_stats",
        "normalization_source": "clay_foundation_metadata_yaml_sentinel_2_l2a",
        "normalization_input_scale": "raw_sr_dn",
    }
    return x.astype(np.float32, copy=False), waves, meta


def _resize_chw(
    x_chw: np.ndarray,
    *,
    size: int = _CLAY_IMAGE_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """CHW float32 -> square (size,size), padding a rectangular ROI before resize."""
    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW, got shape={x_chw.shape}")
    _, h, w = x_chw.shape
    y, shape_meta = prepare_square(x_chw, size=int(size), shape_adjust="pad")
    info = {
        "orig_hw": (int(h), int(w)),
        "target_hw": (int(size), int(size)),
        "mode": "bilinear",
        **shape_meta,
    }
    return y, info


# -----------------------------
# Metadata encodings (official Clay recipe)
# -----------------------------
def _normalize_clay_timestamp(dt: datetime) -> np.ndarray:
    """[sin(week), cos(week), sin(hour), cos(hour)] as in Clay's tutorials."""
    week = dt.isocalendar()[1] * 2 * math.pi / 52
    hour = dt.hour * 2 * math.pi / 24
    return np.array(
        [math.sin(week), math.cos(week), math.sin(hour), math.cos(hour)],
        dtype=np.float32,
    )


def _normalize_clay_latlon(lat: float, lon: float) -> np.ndarray:
    """[sin(lat), cos(lat), sin(lon), cos(lon)] in radians, as in Clay's tutorials."""
    lat_r = float(lat) * math.pi / 180.0
    lon_r = float(lon) * math.pi / 180.0
    return np.array(
        [math.sin(lat_r), math.cos(lat_r), math.sin(lon_r), math.cos(lon_r)],
        dtype=np.float32,
    )


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


def _clay_time_vec(temporal: TemporalSpec | None) -> tuple[np.ndarray, str | None]:
    """Time encoding from the midpoint of the (normalized) temporal window."""
    t = temporal_to_range(temporal)
    mid = temporal_midpoint_str(t)
    if mid is None:
        return np.zeros(4, dtype=np.float32), None
    return _normalize_clay_timestamp(datetime.fromisoformat(mid)), mid


# -----------------------------
# Provider fetch (raw SR values)
# -----------------------------
def _fetch_provider_multiband_sr_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    collection: str,
    bands: list[str],
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
    default_value: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    x = np.asarray(raw, dtype=np.float32)

    meta: dict[str, Any] = {
        "provider_collection": collection,
        "provider_bands": list(bands),
        "provider_scale_m": int(scale_m),
        "provider_cloudy_pct": int(cloudy_pct),
        "provider_cloud_filter_applied": True,
        "provider_composite": str(composite),
        "raw_chw_shape": tuple(x.shape),
        "region_crs": "EPSG:3857",
    }
    return x, meta


# -----------------------------
# Clay model loading
# -----------------------------
_CLAY_HF_REPO_ID_DEFAULT = "made-with-clay/Clay"
_CLAY_HF_FILENAME_DEFAULT = "v1.5/clay-v1.5.ckpt"
_CLAY_HF_REVISION_DEFAULT = "main"
_CLAY_MODEL_SIZE_DEFAULT = "large"
_CLAY_VERSION_LABEL = "v1.5"


def _model_config_value(
    model_config: dict[str, Any] | None,
    key: str,
) -> Any | None:
    if model_config is None:
        return None
    if isinstance(model_config, dict):
        return model_config.get(key)
    return getattr(model_config, key, None)


def _resolve_clay_model_size(
    *,
    model_config: dict[str, Any] | None,
) -> str:
    size = _model_config_value(model_config, "model_size")
    if size is None:
        size = os.environ.get("RS_EMBED_CLAY_MODEL_SIZE", _CLAY_MODEL_SIZE_DEFAULT)
    return str(size).lower().strip()


def _resolve_clay_weights_path() -> tuple[str, str]:
    override = os.environ.get("RS_EMBED_CLAY_WEIGHTS")
    if override:
        return override, override

    filename = os.environ.get("RS_EMBED_CLAY_HF_FILENAME", _CLAY_HF_FILENAME_DEFAULT)
    weights_dir = os.environ.get("RS_EMBED_CLAY_WEIGHTS_DIR")
    if weights_dir:
        local_path = os.path.join(weights_dir, os.path.basename(filename))
        if os.path.exists(local_path):
            return local_path, local_path

    repo_id = os.environ.get("RS_EMBED_CLAY_HF_REPO_ID", _CLAY_HF_REPO_ID_DEFAULT)
    revision = os.environ.get("RS_EMBED_CLAY_HF_REVISION", _CLAY_HF_REVISION_DEFAULT)
    try:
        from huggingface_hub import hf_hub_download, hf_hub_url
    except Exception as e:
        raise ModelError(
            "Clay requires huggingface-hub to download weights, or set "
            "RS_EMBED_CLAY_WEIGHTS / RS_EMBED_CLAY_WEIGHTS_DIR to a local checkpoint."
        ) from e

    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        remote_url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
        return local_path, remote_url
    except Exception as e:
        raise ModelError(
            "Failed to fetch Clay weights from Hugging Face. Set "
            "RS_EMBED_CLAY_WEIGHTS or RS_EMBED_CLAY_WEIGHTS_DIR to a local checkpoint."
        ) from e


def _extract_clay_encoder_state_dict(payload: Any) -> dict[str, Any]:
    """Extract encoder weights from a Clay Lightning checkpoint.

    The published ckpt stores the full ClayMAE (encoder + decoder + frozen
    teacher) under ``state_dict`` with a ``model.`` prefix; only
    ``model.encoder.*`` is needed for embeddings.
    """
    obj = payload
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ModelError("Unexpected Clay checkpoint payload; expected a state dict.")

    for prefix in ("model.encoder.", "encoder."):
        enc = {
            k[len(prefix) :]: v
            for k, v in obj.items()
            if isinstance(k, str) and k.startswith(prefix)
        }
        if enc:
            return enc
    raise ModelError(
        "Clay checkpoint contains no encoder weights (expected 'model.encoder.*' keys)."
    )


@lru_cache(maxsize=2)
def _load_clay_model_cached(model_size: str, dev: str):
    import torch

    from ._vendor.clay.model import ENCODER_SIZE_ARGS, clay_encoder

    size_l = str(model_size).lower().strip()
    if size_l not in ENCODER_SIZE_ARGS:
        raise ModelError(
            f"Unknown Clay model_size='{model_size}' (expected one of {list(ENCODER_SIZE_ARGS)})."
        )
    model = clay_encoder(size_l, patch_size=_CLAY_PATCH_SIZE, mask_ratio=0.0, shuffle=False)

    weights_path, weights_url = _resolve_clay_weights_path()
    # Lightning checkpoints pickle non-tensor hyper-parameters, so
    # weights_only=True cannot load them (PyTorch 2.6+ default change).
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = _extract_clay_encoder_state_dict(checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing:
        raise ModelError(
            f"Clay encoder weights incomplete: missing keys {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}. Checkpoint/model_size mismatch?"
        )

    model = model.to(dev).eval()

    # sanity
    p0 = None
    for _, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError("Clay model has no parameters; unexpected.")
    if not torch.isfinite(p0).all():
        raise ModelError("Clay parameters contain NaN/Inf; weight load likely failed.")

    meta = {
        "model_size": size_l,
        "clay_version": _CLAY_VERSION_LABEL,
        "device": dev,
        "device_resolved": dev,
        "weights_url": str(weights_url),
        "weights_meta": {
            "path": weights_path,
            "unexpected_keys": unexpected,
        },
        "patch_size": _CLAY_PATCH_SIZE,
        "embed_dim": int(ENCODER_SIZE_ARGS[size_l]["dim"]),
    }
    return model, meta


def _load_clay_model(
    *,
    model_size: str = _CLAY_MODEL_SIZE_DEFAULT,
    device: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    loaded, _dev = _load_cached_with_device(
        _load_clay_model_cached,
        device=device,
        model_size=str(model_size).lower().strip(),
    )
    return loaded


# -----------------------------
# Forward adapters
# -----------------------------
def _clay_forward_tokens_and_cls_batch(
    model,
    x_bchw: np.ndarray,
    *,
    wavelengths_um: np.ndarray,
    gsd_m: float,
    times_b4: np.ndarray,
    latlons_b4: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the Clay encoder on a normalized batch.

    Builds the official datacube dict (pixels, time, latlon, gsd, waves) and
    returns (patch_tokens [B,N,D] without CLS, cls [B,D], extra).
    """
    import torch

    dev = _resolve_device_auto(device)
    x = torch.from_numpy(x_bchw).to(dev)
    if x.dtype != torch.float32:
        x = x.float()

    datacube = {
        "pixels": x,
        "time": torch.from_numpy(np.asarray(times_b4, dtype=np.float32)).to(dev),
        "latlon": torch.from_numpy(np.asarray(latlons_b4, dtype=np.float32)).to(dev),
        # gsd/waves stay on CPU: the encoder moves its position encodings to
        # the pixel device itself (as in the upstream tutorials).
        "gsd": torch.tensor(float(gsd_m)),
        "waves": torch.tensor([float(v) for v in wavelengths_um]),
    }

    with torch.inference_mode():
        encoded, _unmsk_idx, _msk_idx, _msk_matrix = model(datacube)  # [B, 1+N, D]
        cls = encoded[:, 0, :].detach().float().cpu().numpy().astype(np.float32)  # [B,D]
        patch_tokens = (
            encoded[:, 1:, :].detach().float().cpu().numpy().astype(np.float32)
        )  # [B,N,D]

    n = int(patch_tokens.shape[1])
    d = int(patch_tokens.shape[2])
    side = int(round(math.sqrt(n)))
    extra = {
        "token_count": n,
        "token_dim": d,
        "token_grid_side": int(side) if side * side == n else None,
        "tokens_include_cls": False,
        "pooled_source": "cls_token",
        "batch_shape": tuple(patch_tokens.shape),
    }
    return patch_tokens, cls, extra


def build_clay_embedding(
    tokens_nd: np.ndarray,
    pooled_d: np.ndarray,
    *,
    geo_roi: tuple[float, float, float, float] | None,
    output: OutputSpec,
    base_meta: dict[str, Any],
    patch_size: int = _CLAY_PATCH_SIZE,
) -> Embedding:
    """Turn Clay patch tokens [N,D] + CLS vector [D] into an Embedding.

    When ``geo_roi`` is a sub-window (fetch-square enlargement), the token grid
    is cropped back to the ROI; pooled output then means the ROI's tokens
    (``roi_grid_mean``) instead of the model's CLS vector.
    """
    geo_roi = tuple(geo_roi or FULL_WINDOW)
    cropped_to_roi = not roi_is_full(geo_roi)

    grid = None
    if output.mode == "grid" or cropped_to_roi:
        n, d = tokens_nd.shape
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ModelError(f"Clay tokens N={n} not square; cannot reshape to grid.")
        grid = tokens_nd.reshape(side, side, d).transpose(2, 0, 1).astype(np.float32)
        if cropped_to_roi:
            grid = crop_grid_to_roi(grid, geo_roi)

    if output.mode == "pooled":
        vec = grid.mean(axis=(1, 2)).astype(np.float32) if cropped_to_roi else pooled_d
        vec = vec.astype(np.float32)
        meta = dict(base_meta)
        meta["pooled_shape"] = tuple(vec.shape)
        meta["pooling"] = "roi_grid_mean" if cropped_to_roi else "model_cls"
        return Embedding(data=vec, meta=meta)

    if output.mode == "grid":
        assert grid is not None
        meta = {
            **base_meta,
            "grid_type": "vit_patch_tokens",
            "grid_shape": tuple(grid.shape),
            "grid_hw_tokens": (int(grid.shape[1]), int(grid.shape[2])),
            "patch_size": int(patch_size),
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


# -----------------------------
# Embedder
# -----------------------------
@register("clay")
class ClayEmbedder(EmbedderBase):
    """
    Clay v1.5 embeddings (made-with-clay/Clay).

    - backend="provider"/"auto": ROI -> S2 L2A 10-band SR -> normalize (Clay
      metadata stats) -> resize to 256 -> Clay encoder conditioned on
      lat/lon + acquisition time + gsd + band wavelengths -> pooled/grid
    - backend="tensor": input_chw (CHW raw SR values) -> same pipeline

    Output:
      - OutputSpec.pooled(): (D,) CLS token (official embedding), or ROI token
        mean when a rectangular ROI was fetch-square enlarged
      - OutputSpec.grid():   (D, Ht, Wt) patch-token grid, 32x32 for 256/patch8
    """

    # Clay needs a square token grid → base.fetch_input enlarges a rectangular
    # ROI to a square of real imagery; the output is cropped back to the ROI.
    _requires_square_input = True
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 4
    DEFAULT_BATCH_CUDA = 32

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_CLAY_S2_BANDS),
        scale_m=10,
        cloudy_pct=30,
        image_size=_CLAY_IMAGE_SIZE,
        expected_channels=10,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider", "tensor"],
            "inputs": {
                "provider_default": {
                    "collection": self.input_spec.collection,
                    "bands": list(self.input_spec.bands),
                    "wavelengths_um": "Clay metadata.yaml sentinel-2-l2a values",
                }
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "model_size": _CLAY_MODEL_SIZE_DEFAULT,
                "clay_version": _CLAY_VERSION_LABEL,
                "image_size": self.input_spec.image_size,
                "patch_size": _CLAY_PATCH_SIZE,
                "scale_m": self.input_spec.scale_m,
                "cloudy_pct": self.input_spec.cloudy_pct,
                "composite": self.input_spec.composite,
                "preprocess": "official_clay_metadata_s2_stats_then_resize_to_256_bilinear",
                "metadata_conditioning": ["latlon", "time", "gsd", "wavelengths"],
            },
            "model_config": {
                "model_size": {
                    "type": "string",
                    "default": _CLAY_MODEL_SIZE_DEFAULT,
                    "choices": ["tiny", "small", "base", "large"],
                    "note": "must match the checkpoint; the published v1.5 ckpt is 'large'",
                }
            },
        }

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return ClayEmbedder.input_spec.to_sensor_spec()

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get("RS_EMBED_CLAY_FETCH_WORKERS", str(ClayEmbedder.DEFAULT_FETCH_WORKERS))
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default_bs = (
            ClayEmbedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else ClayEmbedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_CLAY_BATCH_SIZE", str(default_bs)))
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
        fetch_meta: dict[str, Any] | None = None,
    ) -> Embedding:
        backend_l = backend.lower().strip()
        model_size = _resolve_clay_model_size(model_config=model_config)
        image_size = _CLAY_IMAGE_SIZE
        # Fetch-square ROI window: from the direct provider fetch, or carried in
        # fetch_meta when the API prefetched a square. Output cropped back to it.
        geo_roi = geo_roi_from_meta(fetch_meta)

        t = temporal_to_range(temporal)
        check_meta: dict[str, Any] = {}

        # -----------------
        # Build input
        # -----------------
        if backend_l == "tensor":
            if input_chw is None:
                raise ModelError(
                    "backend='tensor' requires input_chw as CHW. "
                    "Use get_embeddings_batch_from_inputs(...) for batches."
                )
            x_chw = coerce_single_input_chw(
                input_chw,
                expected_channels=None,
                model_name="Clay",
            )
            bands = list(getattr(sensor, "bands", [])) if hasattr(sensor, "bands") else []
            if not bands:
                bands = list(_CLAY_S2_BANDS) if int(x_chw.shape[0]) == len(_CLAY_S2_BANDS) else []
            if not bands:
                raise ModelError(
                    "Clay tensor backend requires sensor.bands so official preprocessing "
                    "and wavelength conditioning can be applied."
                )
            x_chw, wavelengths_um, norm_meta = _normalize_clay_input_chw(
                x_chw,
                bands=bands,
                input_name="input_chw",
            )
            x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
            x_bchw = x_chw[None, ...]
            scale_m = int(getattr(sensor, "scale_m", 10)) if sensor else 10
            provider_meta = {"backend_tensor": True, **norm_meta}

        else:
            provider = self._get_provider(backend)

            # overrides
            collection = (
                getattr(sensor, "collection", "COPERNICUS/S2_SR_HARMONIZED")
                if sensor
                else "COPERNICUS/S2_SR_HARMONIZED"
            )
            bands = (
                list(getattr(sensor, "bands", _CLAY_S2_BANDS)) if sensor else list(_CLAY_S2_BANDS)
            )
            scale_m = int(getattr(sensor, "scale_m", 10)) if sensor else 10
            cloudy_pct = int(getattr(sensor, "cloudy_pct", 30)) if sensor else 30
            composite = str(getattr(sensor, "composite", "median")) if sensor else "median"

            if input_chw is None:
                spatial, geo_roi = square_spatial(spatial)  # enlarge rectangle to square
                raw_chw, provider_meta = _fetch_provider_multiband_sr_chw(
                    provider,
                    spatial,
                    t,
                    collection=str(collection),
                    bands=bands,
                    scale_m=scale_m,
                    cloudy_pct=cloudy_pct,
                    composite=composite,
                    default_value=0.0,
                )
            else:
                # input_chw expected to be raw SR values in band order `bands`
                if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(bands):
                    raise ModelError(
                        f"input_chw must be CHW with {len(bands)} bands for Clay, got {getattr(input_chw, 'shape', None)}"
                    )
                raw_chw = np.asarray(input_chw, dtype=np.float32)
                provider_meta = {
                    "raw_chw_shape": tuple(raw_chw.shape),
                    "input_override": True,
                }

            # Optional: inspect on-the-fly provider input
            from ..tools.inspection import checks_should_raise, maybe_inspect_chw

            check_meta.clear()
            report = maybe_inspect_chw(
                raw_chw,
                sensor=sensor,
                name="provider_multiband_sr_raw_chw",
                expected_channels=len(bands),
                value_range=(0.0, 10000.0),
                fill_value=0.0,
                meta=check_meta,
            )
            if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
                raise ModelError(
                    "Provider input inspection failed: " + "; ".join(report.get("issues", []))
                )

            x_chw, wavelengths_um, norm_meta = _normalize_clay_input_chw(
                raw_chw,
                bands=bands,
                input_name="provider_raw_chw",
            )
            provider_meta.update(norm_meta)
            x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
            x_bchw = x_chw[None, ...].astype(np.float32)

        # -----------------
        # Metadata encodings (latlon / time / gsd)
        # -----------------
        lon, lat = _spatial_center_lon_lat(spatial)
        latlons_b4 = _normalize_clay_latlon(lat, lon)[None, :]
        time_vec, time_mid = _clay_time_vec(t)
        times_b4 = time_vec[None, :]
        # Nominal platform gsd, as in the upstream ClayMAE forward
        # (metadata[platform].gsd) and embedding tutorials.
        gsd_m = float(scale_m)

        # -----------------
        # Model + forward
        # -----------------
        model, mmeta = _load_clay_model(model_size=model_size, device=device)
        tokens_bnd, cls_bd, tmeta = _clay_forward_tokens_and_cls_batch(
            model,
            x_bchw,
            wavelengths_um=np.asarray(wavelengths_um, dtype=np.float32),
            gsd_m=gsd_m,
            times_b4=times_b4,
            latlons_b4=latlons_b4,
            device=device,
        )

        source = (
            getattr(sensor, "collection", None)
            if sensor is not None
            else ("tensor_input" if backend_l == "tensor" else self.input_spec.collection)
        )
        base_meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=backend_l,
            source=source,
            sensor=sensor,
            temporal=temporal,
            image_size=image_size,
            extra={
                "output_mode": output.mode,
                "device": str(device),
                "preprocess": {
                    "strategy": "official_clay_metadata_s2_stats_then_resize_to_256_bilinear",
                    "resize_meta": resize_meta,
                },
                "input_channels": int(x_bchw.shape[1]),
                "wavelengths_um": [float(v) for v in wavelengths_um],
                "input_size_hw": (int(x_bchw.shape[2]), int(x_bchw.shape[3])),
                "metadata_conditioning": {
                    "latlon_center": (float(lat), float(lon)),
                    "time_midpoint": time_mid,
                    "gsd_m": gsd_m,
                },
                "token_meta": tmeta,
                **check_meta,
                **mmeta,
                **provider_meta,
            },
        )

        return build_clay_embedding(
            tokens_bnd[0],
            cls_bd[0],
            geo_roi=geo_roi,
            output=output,
            base_meta=base_meta,
            patch_size=int(mmeta.get("patch_size", _CLAY_PATCH_SIZE)),
        )

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

        backend_l = backend.lower().strip()
        if backend_l == "tensor":
            raise ModelError(
                "backend='tensor' batch inference requires get_embeddings_batch_from_inputs(...)."
            )
        provider = self._get_provider(backend)
        t = temporal_to_range(temporal)

        ss = sensor or self._default_sensor()
        collection = str(getattr(ss, "collection", "COPERNICUS/S2_SR_HARMONIZED"))
        bands = list(getattr(ss, "bands", _CLAY_S2_BANDS))
        scale_m = int(getattr(ss, "scale_m", 10))
        cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
        composite = str(getattr(ss, "composite", "median"))

        # Square-fetch each ROI; the per-item ROI window rides in geo_rois and is
        # forwarded as _roi_windows_geo so the per-item output is cropped back.
        raw_inputs, geo_rois = square_fetch_batch(
            spatials,
            lambda sq: _fetch_provider_multiband_sr_chw(
                provider,
                sq,
                t,
                collection=collection,
                bands=bands,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
                default_value=0.0,
            )[0].astype(np.float32),
            max_workers=self._resolve_fetch_workers(len(spatials)),
        )
        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            temporal=temporal,
            sensor=ss,
            model_config=model_config,
            output=output,
            _roi_windows_geo=geo_rois,
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
        _roi_windows_geo: list[tuple[float, float, float, float] | None] | None = None,
        fetch_metas: list[dict[str, Any] | None] | None = None,
    ) -> list[Embedding]:
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []
        # Prefetched square inputs carry the ROI window in fetch_meta (the
        # export pipeline passes it via ``fetch_metas``); fold it into the
        # internal per-item ROI list so the output is cropped back to the ROI.
        if _roi_windows_geo is None and fetch_metas is not None:
            _roi_windows_geo = [(m or {}).get("roi_window_geo") for m in fetch_metas]

        backend_l = backend.lower().strip()
        if backend_l == "tensor":
            return super().get_embeddings_batch_from_inputs(
                spatials=spatials,
                input_chws=input_chws,
                temporal=temporal,
                sensor=sensor,
                model_config=model_config,
                output=output,
                backend=backend,
                device=device,
            )
        self._get_provider(backend)
        t = temporal_to_range(temporal)

        ss = sensor or self._default_sensor()
        model_size = _resolve_clay_model_size(model_config=model_config)
        bands = list(getattr(ss, "bands", _CLAY_S2_BANDS))
        scale_m = int(getattr(ss, "scale_m", 10))
        gsd_m = float(scale_m)

        x_bchw_all: list[np.ndarray] = []
        resize_meta_all: list[dict[str, Any]] = []
        wavelengths_um: np.ndarray | None = None
        for i, input_chw in enumerate(input_chws):
            if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(bands):
                raise ModelError(
                    f"input_chw must be CHW with {len(bands)} bands for Clay, got "
                    f"{getattr(input_chw, 'shape', None)} at index={i}"
                )
            x_chw, wavelengths_um, _ = _normalize_clay_input_chw(
                input_chw,
                bands=bands,
                input_name=f"input_chw[{i}]",
            )
            x_chw, resize_meta = _resize_chw(x_chw, size=_CLAY_IMAGE_SIZE)
            x_bchw_all.append(x_chw)
            resize_meta_all.append(resize_meta)
        assert wavelengths_um is not None

        # Per-item metadata encodings (shared time window, per-item latlon)
        time_vec, time_mid = _clay_time_vec(t)
        lon_lat = [_spatial_center_lon_lat(s) for s in spatials]
        latlons_all = np.stack(
            [_normalize_clay_latlon(lat, lon) for lon, lat in lon_lat],
            axis=0,
        ).astype(np.float32)

        model, mmeta = _load_clay_model(model_size=model_size, device=device)
        dev = str(mmeta.get("device", device))
        infer_bs = self._resolve_infer_batch(dev)

        out: list[Embedding | None] = [None] * len(spatials)
        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            xb = np.stack(x_bchw_all[s0:s1], axis=0).astype(np.float32)
            times_b4 = np.repeat(time_vec[None, :], s1 - s0, axis=0)
            tokens_bnd, cls_bd, tmeta = _clay_forward_tokens_and_cls_batch(
                model,
                xb,
                wavelengths_um=np.asarray(wavelengths_um, dtype=np.float32),
                gsd_m=gsd_m,
                times_b4=times_b4,
                latlons_b4=latlons_all[s0:s1],
                device=dev,
            )
            for j in range(s1 - s0):
                i = s0 + j
                lon, lat = _spatial_center_lon_lat(spatials[i])
                base_meta = build_meta(
                    model=self.model_name,
                    kind="on_the_fly",
                    backend=backend_l,
                    source="tensor_input" if backend_l == "tensor" else self.input_spec.collection,
                    sensor=sensor,
                    temporal=t,
                    image_size=_CLAY_IMAGE_SIZE,
                    extra={
                        "output_mode": output.mode,
                        "device": str(dev),
                        "preprocess": {
                            "strategy": "official_clay_metadata_s2_stats_then_resize_to_256_bilinear",
                            "resize_meta": resize_meta_all[i],
                        },
                        "input_channels": int(x_bchw_all[i].shape[0]),
                        "wavelengths_um": [float(v) for v in wavelengths_um],
                        "input_size_hw": (
                            int(x_bchw_all[i].shape[1]),
                            int(x_bchw_all[i].shape[2]),
                        ),
                        "metadata_conditioning": {
                            "latlon_center": (float(lat), float(lon)),
                            "time_midpoint": time_mid,
                            "gsd_m": gsd_m,
                        },
                        "token_meta": tmeta,
                        "batch_infer": True,
                        "input_override": True,
                        "normalization": "official_clay_metadata_s2_stats",
                        "normalization_source": "clay_foundation_metadata_yaml_sentinel_2_l2a",
                        "normalization_input_scale": "raw_sr_dn",
                        **mmeta,
                        "raw_chw_shape": tuple(input_chws[i].shape),
                    },
                )

                geo_roi = tuple((_roi_windows_geo[i] if _roi_windows_geo else None) or FULL_WINDOW)
                out[i] = build_clay_embedding(
                    tokens_bnd[j],
                    cls_bd[j],
                    geo_roi=geo_roi,
                    output=output,
                    base_meta=base_meta,
                    patch_size=int(mmeta.get("patch_size", _CLAY_PATCH_SIZE)),
                )

        if any(e is None for e in out):
            raise ModelError("clay batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

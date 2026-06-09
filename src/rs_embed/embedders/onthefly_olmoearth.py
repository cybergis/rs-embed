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
from ..providers.fetch import fetch_collection_patch_chw as _fetch_collection_patch_chw
from ..providers.resolution import is_provider_backend
from ..tools.runtime import load_cached_with_device as _load_cached_with_device
from .base import EmbedderBase
from .config import model_config_value
from .meta import build_meta, temporal_midpoint_str, temporal_to_range

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# S2 L2A band order expected by OlmoEarth (matches Modality.SENTINEL2_L2A):
#   BandSet-0 (10 m): B02, B03, B04, B08
#   BandSet-1 (20 m): B05, B06, B07, B8A, B11, B12
#   BandSet-2 (60 m): B01, B09
# GEE COPERNICUS/S2_SR_HARMONIZED uses names without leading zeros.
_S2_BANDS_GEE: tuple[str, ...] = (
    "B2",
    "B3",
    "B4",
    "B8",  # 10 m
    "B5",
    "B6",
    "B7",
    "B8A",
    "B11",
    "B12",  # 20 m
    "B1",
    "B9",  # 60 m
)
_N_BANDS = len(_S2_BANDS_GEE)  # 12
_N_BAND_SETS = 3  # matches OlmoEarth S2 L2A

# Map canonical variant names to (ModelID enum string, size, version)
_VARIANT_SPECS: dict[str, tuple[str, str, str]] = {
    "nano": ("OlmoEarth-v1-Nano", "nano", "v1"),
    "tiny": ("OlmoEarth-v1-Tiny", "tiny", "v1"),
    "base": ("OlmoEarth-v1-Base", "base", "v1"),
    "large": ("OlmoEarth-v1-Large", "large", "v1"),
    "nano_v1_1": ("OlmoEarth-v1_1-Nano", "nano", "v1.1"),
    "tiny_v1_1": ("OlmoEarth-v1_1-Tiny", "tiny", "v1.1"),
    "base_v1_1": ("OlmoEarth-v1_1-Base", "base", "v1.1"),
}

_VARIANT_ALIASES: dict[str, str] = {
    "nano_v1": "nano",
    "tiny_v1": "tiny",
    "base_v1": "base",
    "large_v1": "large",
    "nano_11": "nano_v1_1",
    "tiny_11": "tiny_v1_1",
    "base_11": "base_v1_1",
}

_DEFAULT_VARIANT = "nano"
_DEFAULT_IMAGE_SIZE = 256  # training tile size; model accepts any size divisible by patch_size
_DEFAULT_PATCH_SIZE = 4
_DEFAULT_SCALE_M = 10
_DEFAULT_CLOUDY_PCT = 30


# ---------------------------------------------------------------------------
# Package guard
# ---------------------------------------------------------------------------


def _ensure_olmoearth() -> Any:
    """Import and return olmoearth_pretrain_minimal, raising a clear error if absent."""
    try:
        import olmoearth_pretrain_minimal as om  # type: ignore[import-untyped]

        return om
    except ImportError as exc:
        raise ModelError(
            "OlmoEarth requires olmoearth-pretrain-minimal. "
            "Install: uv pip install olmoearth-pretrain-minimal"
        ) from exc


def _ensure_torch() -> Any:
    try:
        import torch  # noqa: F401

        return torch
    except ImportError as exc:
        raise ModelError("OlmoEarth requires torch. Install: uv pip install torch") from exc


# ---------------------------------------------------------------------------
# Variant helpers
# ---------------------------------------------------------------------------


def _normalize_variant(variant: Any) -> str:
    raw = str(variant).strip().lower().replace("-", "_").replace(".", "_")
    if raw in _VARIANT_SPECS:
        return raw
    if raw in _VARIANT_ALIASES:
        return _VARIANT_ALIASES[raw]
    raise ModelError(
        f"Unknown OlmoEarth variant='{variant}'. Valid choices: {sorted(_VARIANT_SPECS)}."
    )


def _resolve_variant(model_config: dict[str, Any] | None) -> str:
    v = model_config_value(model_config, "variant")
    if v is not None:
        return _normalize_variant(v)
    env = os.environ.get("RS_EMBED_OLMOEARTH_VARIANT", "").strip()
    if env and env.lower() not in ("", "auto"):
        return _normalize_variant(env)
    return _DEFAULT_VARIANT


def _resolve_patch_size(model_config: dict[str, Any] | None) -> int:
    v = model_config_value(model_config, "patch_size")
    if v is not None:
        ps = int(v)
        if ps < 1 or ps > 8:
            raise ModelError(f"OlmoEarth patch_size must be 1–8, got {ps}.")
        return ps
    env = os.environ.get("RS_EMBED_OLMOEARTH_PATCH_SIZE", "").strip()
    if env:
        return int(env)
    return _DEFAULT_PATCH_SIZE


def _resolve_image_size(model_config: dict[str, Any] | None) -> int:
    v = model_config_value(model_config, "image_size")
    if v is not None:
        return int(v)
    env = os.environ.get("RS_EMBED_OLMOEARTH_IMAGE_SIZE", "").strip()
    if env:
        return int(env)
    return _DEFAULT_IMAGE_SIZE


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_olmoearth_cached(model_id_str: str, dev: str):
    om = _ensure_olmoearth()
    model_id = om.ModelID(model_id_str)
    try:
        model = om.load_model_from_id(model_id, load_weights=True)
    except Exception as exc:
        raise ModelError(
            f"Failed to load OlmoEarth model '{model_id_str}': {type(exc).__name__}: {exc}"
        ) from exc
    model = model.to(dev).eval()
    meta = {
        "model_id": model_id_str,
        "hf_repo": model_id.repo_id(),
        "device": dev,
    }
    return model, meta


def _load_olmoearth(variant: str, *, device: str = "auto"):
    model_id_str, _, _ = _VARIANT_SPECS[variant]
    (model, meta), dev = _load_cached_with_device(
        _load_olmoearth_cached,
        device=device,
        model_id_str=model_id_str,
    )
    return model, meta, dev


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _normalize_s2_chw(x_chw: np.ndarray) -> np.ndarray:
    """Normalize CHW S2 raw DN to OlmoEarth's expected range via mean±2σ clipping."""
    om = _ensure_olmoearth()
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import (
        Modality,  # type: ignore
    )

    # Normalizer expects shape (..., C) — transpose to HWC, normalize, back to CHW
    hwc = np.moveaxis(x_chw, 0, -1).astype(np.float32)  # [H, W, 12]
    norm = om.Normalizer(std_multiplier=2.0)
    hwc = norm.normalize(Modality.SENTINEL2_L2A, hwc).astype(np.float32)
    hwc = np.nan_to_num(hwc, nan=0.0, posinf=1.0, neginf=0.0)
    return np.moveaxis(hwc, -1, 0)  # [12, H, W]


def _resize_chw(x_chw: np.ndarray, *, size: int) -> np.ndarray:
    """Resize a CHW float32 array to (size, size) using bilinear interpolation."""
    torch = _ensure_torch()
    import torch.nn.functional as F  # noqa: PLC0415

    t = torch.from_numpy(x_chw).unsqueeze(0)  # [1, C, H, W]
    out = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return out[0].cpu().numpy().astype(np.float32)


def _prepare_chw(
    x_chw: np.ndarray,
    *,
    image_size: int,
    patch_size: int,
) -> np.ndarray:
    """Normalize and resize CHW patch for OlmoEarth input."""
    if x_chw.ndim != 3 or x_chw.shape[0] != _N_BANDS:
        raise ModelError(f"OlmoEarth expects {_N_BANDS}-band CHW input, got {x_chw.shape}.")
    x = _normalize_s2_chw(x_chw)

    # Ensure H and W are divisible by patch_size
    target = max(patch_size, (image_size // patch_size) * patch_size)
    if x.shape[1] != target or x.shape[2] != target:
        x = _resize_chw(x, size=target)
    return x


def _date_to_timestamp(date_str: str | None) -> tuple[int, int, int]:
    """Convert ISO date string to OlmoEarth timestamp tuple (day, month_0idx, year)."""
    if not date_str:
        return (1, 0, 2022)
    from datetime import date as _date  # noqa: PLC0415

    d = _date.fromisoformat(date_str)
    return (d.day, d.month - 1, d.year)  # month is 0-indexed in OlmoEarth


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _build_sample(x_chw: np.ndarray, *, timestamp: tuple[int, int, int]):
    """Wrap a single CHW patch as a batched MaskedOlmoEarthSample (B=1, T=1)."""
    torch = _ensure_torch()
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (  # type: ignore
        MaskedOlmoEarthSample,
    )

    _, h, w = x_chw.shape
    # Model expects (B, H, W, T, C)
    s2 = torch.from_numpy(x_chw).permute(1, 2, 0).unsqueeze(0).unsqueeze(3)  # [1,H,W,1,12]
    # Mask: all ONLINE_ENCODER (0)
    mask = torch.zeros(1, h, w, 1, _N_BAND_SETS, dtype=torch.float32)
    ts = torch.tensor(
        [[[*timestamp]]], dtype=torch.long
    )  # [1, T=1, 3]; must be int for month embedding
    return MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=mask,
        timestamps=ts,
    )


def _build_batch_sample(x_bchw: np.ndarray, *, timestamps: list[tuple[int, int, int]]):
    """Wrap a BCHW batch as a batched MaskedOlmoEarthSample (B, T=1)."""
    torch = _ensure_torch()
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (  # type: ignore
        MaskedOlmoEarthSample,
    )

    b, _, h, w = x_bchw.shape
    # (B, H, W, T=1, C)
    s2 = torch.from_numpy(x_bchw).permute(0, 2, 3, 1).unsqueeze(3)
    mask = torch.zeros(b, h, w, 1, _N_BAND_SETS, dtype=torch.float32)
    ts = torch.tensor(
        [[list(t)] for t in timestamps], dtype=torch.long
    )  # [B, 1, 3]; int for month embedding
    return MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=mask,
        timestamps=ts,
    )


def _encoder_forward(model, sample, *, patch_size: int, device: str) -> Any:
    """Run the encoder forward pass. Returns the tokens_and_masks output."""
    torch = _ensure_torch()

    # Move sample tensors to device
    kwargs: dict[str, Any] = {}
    for field in sample._fields:
        val = getattr(sample, field)
        if val is not None:
            kwargs[field] = val.to(device)
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (  # type: ignore
        MaskedOlmoEarthSample,
    )

    sample_dev = MaskedOlmoEarthSample(**kwargs)
    with torch.no_grad():
        # load_model_from_id returns LatentMIM directly (.encoder is a direct attr).
        # OlmoEarthPretrain_v1 delegates via __getattr__, so .encoder works for both.
        encoder = model.encoder if hasattr(model, "encoder") else model.model.encoder
        output = encoder(sample_dev, fast_pass=True, patch_size=patch_size)
    return output["tokens_and_masks"]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _pool_tokens(tokens_and_masks, pooling: str) -> np.ndarray:
    """Pool spatial + temporal + band-set dims → (B, D) numpy array."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import (
        PoolingType,  # type: ignore
    )

    pt = PoolingType.MEAN if pooling == "mean" else PoolingType.MAX
    pooled = tokens_and_masks.pool_unmasked_tokens(pooling_type=pt)  # (B, D)
    return pooled.detach().float().cpu().numpy().astype(np.float32)


def _tokens_to_grid(tokens_and_masks, pooling: str) -> np.ndarray:
    """Return spatial grid (D, H', W') by averaging over T and band-set dims."""
    s2 = tokens_and_masks.sentinel2_l2a  # (B, H', W', T, S, D)
    if s2 is None:
        raise ModelError("No S2 L2A tokens in OlmoEarth encoder output.")
    # pool over T (dim 3) and band-sets (dim 4)
    if pooling == "mean":
        spatial = s2.mean(dim=[3, 4])  # (B, H', W', D)
    else:
        spatial = s2.max(dim=4).values.max(dim=3).values  # (B, H', W', D)
    # Take first batch item and move dim to (D, H', W')
    grid = spatial[0].permute(2, 0, 1).detach().float().cpu().numpy().astype(np.float32)
    return grid


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


@register("olmoearth")
class OlmoEarthEmbedder(EmbedderBase):
    """OlmoEarth v1/v1.1 on-the-fly embeddings from Sentinel-2 L2A 12-band patches.

    Inputs:
      - spatial : BBox / PointBuffer (EPSG:4326)
      - temporal: range / year (year → full year composite)
      - sensor  : controls provider fetch (scale/cloudy/composite)

    Outputs:
      - pooled: global mean/max over spatial, temporal and band-set token dims
      - grid  : spatial token map [D, H', W'] averaged over temporal/band-set dims

    Model variants (via ``model_config={"variant": "..."}``):
      v1  : nano (128-d), tiny (192-d), base (768-d), large (1024-d)
      v1.1: nano_v1_1 (128-d), tiny_v1_1 (192-d), base_v1_1 (768-d)
    """

    DEFAULT_VARIANT = _DEFAULT_VARIANT
    DEFAULT_IMAGE_SIZE = _DEFAULT_IMAGE_SIZE
    DEFAULT_PATCH_SIZE = _DEFAULT_PATCH_SIZE
    DEFAULT_SCALE_M = _DEFAULT_SCALE_M
    DEFAULT_CLOUDY_PCT = _DEFAULT_CLOUDY_PCT
    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 4
    DEFAULT_BATCH_CUDA = 16

    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=_S2_BANDS_GEE,
        scale_m=_DEFAULT_SCALE_M,
        cloudy_pct=_DEFAULT_CLOUDY_PCT,
        expected_channels=_N_BANDS,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["provider"],
            "input_bands": list(_S2_BANDS_GEE),
            "output": ["pooled", "grid"],
            "defaults": {
                "variant": self.DEFAULT_VARIANT,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "patch_size": self.DEFAULT_PATCH_SIZE,
                "scale_m": self.DEFAULT_SCALE_M,
                "cloudy_pct": self.DEFAULT_CLOUDY_PCT,
            },
            "model_config": {
                "variant": {
                    "type": "string",
                    "default": self.DEFAULT_VARIANT,
                    "choices": sorted(_VARIANT_SPECS),
                    "description": (
                        "Model size/version. v1: nano, tiny, base, large. "
                        "v1.1: nano_v1_1, tiny_v1_1, base_v1_1."
                    ),
                },
                "patch_size": {
                    "type": "int",
                    "default": self.DEFAULT_PATCH_SIZE,
                    "description": "Patch size for FlexiViT encoder (1–8). Smaller = more tokens.",
                },
                "image_size": {
                    "type": "int",
                    "default": self.DEFAULT_IMAGE_SIZE,
                    "description": "Image resize target before encoding (pixels). Must be divisible by patch_size.",
                },
            },
            "notes": [
                "OlmoEarth is a Vision Transformer trained on the Major TOM dataset.",
                "Requires olmoearth-pretrain-minimal (pip install olmoearth-pretrain-minimal).",
                "Weights are downloaded automatically from Hugging Face on first use.",
                "Normalization: per-band mean±2σ clipping (OlmoEarth COMPUTED strategy).",
            ],
        }

    def _default_sensor(self) -> SensorSpec:
        assert self.input_spec is not None
        return self.input_spec.to_sensor_spec()

    @staticmethod
    def _resolve_fetch_workers(n: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_OLMOEARTH_FETCH_WORKERS",
                str(OlmoEarthEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(n, v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default = (
            OlmoEarthEmbedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else OlmoEarthEmbedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_OLMOEARTH_BATCH_SIZE", str(default)))
        return max(1, v)

    # ------------------------------------------------------------------
    # Single embedding
    # ------------------------------------------------------------------

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
            raise ModelError("olmoearth expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()
        t = temporal_to_range(temporal)

        variant = _resolve_variant(model_config)
        patch_size = _resolve_patch_size(model_config)
        image_size = _resolve_image_size(model_config)
        _, size, version = _VARIANT_SPECS[variant]

        model, wmeta, dev = _load_olmoearth(variant, device=device)

        if input_chw is None:
            provider = self._get_provider(backend)
            x_chw = _fetch_collection_patch_chw(
                provider,
                spatial=spatial,
                temporal=t,
                collection=sensor.collection,
                bands=_S2_BANDS_GEE,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
            )
        else:
            if input_chw.ndim != 3 or input_chw.shape[0] != _N_BANDS:
                raise ModelError(
                    f"input_chw must be CHW with {_N_BANDS} bands for olmoearth, "
                    f"got {getattr(input_chw, 'shape', None)}."
                )
            x_chw = input_chw.astype(np.float32)

        x_chw = _prepare_chw(x_chw, image_size=image_size, patch_size=patch_size)

        date_str = temporal_midpoint_str(t)
        timestamp = _date_to_timestamp(date_str)
        sample = _build_sample(x_chw, timestamp=timestamp)
        tokens_and_masks = _encoder_forward(model, sample, patch_size=patch_size, device=dev)

        meta = _build_embedding_meta(
            model_name=self.model_name,
            wmeta=wmeta,
            variant=variant,
            size=size,
            version=version,
            backend=backend,
            sensor=sensor,
            temporal=t,
            image_size=int(x_chw.shape[-1]),
            patch_size=patch_size,
            date_str=date_str,
        )

        if output.mode == "pooled":
            vec = _pool_tokens(tokens_and_masks, output.pooling)[0]  # (D,)
            meta["pooling"] = f"spatial_temporal_bandset_{output.pooling}"
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            return _make_grid_embedding(tokens_and_masks, output, meta)

        raise ModelError(f"Unknown output mode: {output.mode!r}.")

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

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
            raise ModelError("olmoearth expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()
        t = temporal_to_range(temporal)
        provider = self._get_provider(backend)
        n = len(spatials)
        prefetched: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            raw = _fetch_collection_patch_chw(
                provider,
                spatial=sp,
                temporal=t,
                collection=sensor.collection,
                bands=_S2_BANDS_GEE,
                scale_m=int(sensor.scale_m),
                cloudy_pct=int(sensor.cloudy_pct),
                composite=str(sensor.composite),
                fill_value=float(sensor.fill_value),
            )
            return i, raw.astype(np.float32)

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                ii, raw = _fetch_one(i, sp)
                prefetched[ii] = raw
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    ii, raw = fut.result()
                    prefetched[ii] = raw

        raw_inputs: list[np.ndarray] = []
        for i, raw in enumerate(prefetched):
            if raw is None:
                raise ModelError(f"Missing prefetched input at index {i} for olmoearth.")
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
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}."
            )
        if not spatials:
            return []
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("olmoearth expects a provider backend (or 'auto').")

        if sensor is None:
            sensor = self._default_sensor()
        t = temporal_to_range(temporal)

        variant = _resolve_variant(model_config)
        patch_size = _resolve_patch_size(model_config)
        image_size = _resolve_image_size(model_config)
        _, size, version = _VARIANT_SPECS[variant]

        model, wmeta, dev = _load_olmoearth(variant, device=device)
        infer_bs = self._resolve_infer_batch(dev)

        date_str = temporal_midpoint_str(t)
        timestamp = _date_to_timestamp(date_str)

        # Prepare all inputs (normalize + resize) and group by shape
        prepared: list[np.ndarray] = []
        for i, x_chw in enumerate(input_chws):
            if x_chw.ndim != 3 or x_chw.shape[0] != _N_BANDS:
                raise ModelError(
                    f"input_chw at index {i} must be CHW with {_N_BANDS} bands, "
                    f"got {getattr(x_chw, 'shape', None)}."
                )
            prepared.append(
                _prepare_chw(x_chw.astype(np.float32), image_size=image_size, patch_size=patch_size)
            )

        shape_groups: dict[tuple[int, int, int], list[int]] = {}
        for i, x in enumerate(prepared):
            shape_groups.setdefault(x.shape, []).append(i)

        xr_mod = None
        if output.mode == "grid":
            try:
                import xarray as xr  # noqa: PLC0415

                xr_mod = xr
            except ImportError as exc:
                raise ModelError(
                    "grid output requires xarray. Install: pip install xarray"
                ) from exc

        out: list[Embedding | None] = [None] * len(spatials)

        for idxs in shape_groups.values():
            for s0 in range(0, len(idxs), infer_bs):
                chunk = idxs[s0 : s0 + infer_bs]
                xb = np.stack([prepared[i] for i in chunk], axis=0)
                sample = _build_batch_sample(xb, timestamps=[timestamp] * len(chunk))
                tokens_and_masks = _encoder_forward(
                    model, sample, patch_size=patch_size, device=dev
                )

                if output.mode == "pooled":
                    pooled = _pool_tokens(tokens_and_masks, output.pooling)  # (B, D)
                    for j, i in enumerate(chunk):
                        meta = _build_embedding_meta(
                            model_name=self.model_name,
                            wmeta=wmeta,
                            variant=variant,
                            size=size,
                            version=version,
                            backend=backend,
                            sensor=sensor,
                            temporal=t,
                            image_size=int(prepared[i].shape[-1]),
                            patch_size=patch_size,
                            date_str=date_str,
                            extra={"batch_infer": True},
                        )
                        meta["pooling"] = f"spatial_temporal_bandset_{output.pooling}"
                        out[i] = Embedding(data=pooled[j], meta=meta)
                    continue

                if output.mode == "grid":
                    s2 = tokens_and_masks.sentinel2_l2a  # (B, H', W', T, S, D)
                    if s2 is None:
                        raise ModelError("No S2 L2A tokens in OlmoEarth encoder output.")
                    if output.pooling == "mean":
                        spatial_b = s2.mean(dim=[3, 4])  # (B, H', W', D)
                    else:
                        spatial_b = s2.max(dim=4).values.max(dim=3).values
                    spatial_b = spatial_b.detach().float().cpu()
                    for j, i in enumerate(chunk):
                        grid_np = spatial_b[j].permute(2, 0, 1).numpy().astype(np.float32)
                        meta = _build_embedding_meta(
                            model_name=self.model_name,
                            wmeta=wmeta,
                            variant=variant,
                            size=size,
                            version=version,
                            backend=backend,
                            sensor=sensor,
                            temporal=t,
                            image_size=int(prepared[i].shape[-1]),
                            patch_size=patch_size,
                            date_str=date_str,
                            extra={"batch_infer": True},
                        )
                        d, h, w = grid_np.shape
                        meta.update({"grid_hw": (h, w), "grid_kind": "spatial_tokens"})
                        assert xr_mod is not None
                        da = xr_mod.DataArray(
                            grid_np,
                            dims=("d", "y", "x"),
                            coords={"d": np.arange(d), "y": np.arange(h), "x": np.arange(w)},
                            name="embedding",
                            attrs=meta,
                        )
                        out[i] = Embedding(data=da, meta=meta)
                    continue

                raise ModelError(f"Unknown output mode: {output.mode!r}.")

        if any(e is None for e in out):
            raise ModelError("OlmoEarth batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]


# ---------------------------------------------------------------------------
# Internal helpers shared between single and batch paths
# ---------------------------------------------------------------------------


def _build_embedding_meta(
    *,
    model_name: str,
    wmeta: dict[str, Any],
    variant: str,
    size: str,
    version: str,
    backend: str,
    sensor: SensorSpec,
    temporal: TemporalSpec,
    image_size: int,
    patch_size: int,
    date_str: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = build_meta(
        model=model_name,
        kind="on_the_fly",
        backend=str(backend).lower(),
        source=sensor.collection,
        sensor=sensor,
        temporal=temporal,
        image_size=image_size,
    )
    meta.update(
        {
            "hf_repo": wmeta.get("hf_repo"),
            "variant": variant,
            "model_size": size,
            "model_version": version,
            "patch_size": patch_size,
            "temporal_range": (temporal.start, temporal.end),
            "date_str": date_str,
        }
    )
    if extra:
        meta.update(extra)
    return meta


def _make_grid_embedding(
    tokens_and_masks: Any, output: OutputSpec, meta: dict[str, Any]
) -> Embedding:
    try:
        import xarray as xr  # noqa: PLC0415
    except ImportError as exc:
        raise ModelError("grid output requires xarray. Install: pip install xarray") from exc

    grid = _tokens_to_grid(tokens_and_masks, output.pooling)  # (D, H', W')
    d, h, w = grid.shape
    meta.update({"grid_hw": (h, w), "grid_kind": "spatial_tokens"})
    da = xr.DataArray(
        grid,
        dims=("d", "y", "x"),
        coords={"d": np.arange(d), "y": np.arange(h), "x": np.arange(w)},
        name="embedding",
        attrs=meta,
    )
    return Embedding(data=da, meta=meta)

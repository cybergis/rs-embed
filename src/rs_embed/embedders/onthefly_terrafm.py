# src/rs_embed/embedders/onthefly_terrafm.py
from __future__ import annotations

import importlib
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
from ..core.types import EmbedderCapabilities, FetchResult
from ..providers import ProviderBase
from ..providers.fetch import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
)
from ..providers.fetch import (
    fetch_s1_vvvh_raw_chw as _fetch_s1_vvvh_raw_chw_shared,
)
from ..providers.fetch import (
    fetch_s1_vvvh_raw_chw_with_meta as _fetch_s1_vvvh_raw_chw_with_meta_shared,
)
from ..providers.fetch import (
    normalize_s1_vvvh_chw as _normalize_s1_vvvh_chw,
)
from ..tools.normalization import (
    coerce_single_input_chw,
)
from ..tools.runtime import (
    resolve_device_auto_torch as _auto_device,
)
from ..tools.shape import (
    crop_grid_and_pool,
    crop_grid_to_roi,
    geo_roi_from_meta,
    roi_fetch_meta,
    roi_is_full,
)
from ..tools.spatial import FULL_WINDOW, square_spatial
from .base import EmbedderBase
from .config import model_config_value
from .meta import build_meta, temporal_to_range
from .shared import grid_to_dataarray, resolve_hf_cache_dir, verify_loaded_params

HF_REPO_ID = "MBZUAI/TerraFM"
HF_WEIGHT_FILE_B = "TerraFM-B.pth"


def _resolve_terrafm_cache_dir(model_config: dict[str, Any] | None) -> str | None:
    """Resolve the TerraFM weight cache directory.

    ``model_config['cache_dir']`` wins; the Hugging Face cache env chain
    (HUGGINGFACE_HUB_CACHE > HF_HOME > HUGGINGFACE_HOME) is the fallback;
    ``None`` uses the huggingface_hub default cache.
    """
    v = model_config_value(model_config, "cache_dir")
    if v is not None and str(v).strip():
        return str(v).strip()
    return resolve_hf_cache_dir()


# -----------------------------
# Small utils
# -----------------------------
def _resize_chw_to_224(x_chw: np.ndarray, *, size: int = 224) -> np.ndarray:
    """Resize CHW float32 -> CHW float32 (bilinear)."""
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW array, got {x_chw.shape}")
    x = torch.from_numpy(x_chw).unsqueeze(0)  # [1,C,H,W]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x[0].cpu().numpy().astype(np.float32)


# -----------------------------
# Provider: Fetch S2 (12 bands, SR)
# -----------------------------
_S2_SR_12_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
]


def _fetch_s2_sr_12_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
) -> np.ndarray:
    """Returns CHW float32 in [0,1], resized later to 224."""
    raw = _fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_S2_SR_12_BANDS),
        scale_m=int(scale_m),
        cloudy_pct=int(cloudy_pct),
        composite=str(composite),
        fill_value=0.0,
    )
    return np.clip(raw / 10000.0, 0.0, 1.0).astype(np.float32)


# -----------------------------
# Provider: Fetch S1 (VV/VH)
# -----------------------------
def _fetch_s1_vvvh_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    use_float_linear: bool = True,
    composite: str = "median",
    require_iw: bool = True,
    relax_iw_on_empty: bool = True,
) -> np.ndarray:
    """Returns normalized S1 VV/VH CHW float32 [2,H,W] in [0,1]."""
    raw = _fetch_s1_vvvh_raw_chw_shared(
        provider,
        spatial=spatial,
        temporal=temporal,
        scale_m=int(scale_m),
        use_float_linear=bool(use_float_linear),
        composite=str(composite),
        fill_value=0.0,
        require_iw=bool(require_iw),
        relax_iw_on_empty=bool(relax_iw_on_empty),
    )
    return _normalize_s1_vvvh_chw(raw)


def _fetch_s1_vvvh_raw_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    use_float_linear: bool = True,
    composite: str = "median",
    require_iw: bool = True,
    relax_iw_on_empty: bool = True,
) -> np.ndarray:
    """Returns raw VV/VH CHW without log/normalization."""
    return _fetch_s1_vvvh_raw_chw_shared(
        provider,
        spatial=spatial,
        temporal=temporal,
        scale_m=int(scale_m),
        use_float_linear=bool(use_float_linear),
        composite=str(composite),
        fill_value=0.0,
        require_iw=bool(require_iw),
        relax_iw_on_empty=bool(relax_iw_on_empty),
    )


def _fetch_s1_vvvh_raw_chw_with_meta(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    use_float_linear: bool = True,
    composite: str = "median",
    require_iw: bool = True,
    relax_iw_on_empty: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _fetch_s1_vvvh_raw_chw_with_meta_shared(
        provider,
        spatial=spatial,
        temporal=temporal,
        scale_m=int(scale_m),
        use_float_linear=bool(use_float_linear),
        composite=str(composite),
        fill_value=0.0,
        require_iw=bool(require_iw),
        relax_iw_on_empty=bool(relax_iw_on_empty),
    )


def _prepare_tensor_input_chw(
    input_chw: Any,
    *,
    modality: str,
    image_size: int,
) -> np.ndarray:
    modality_l = str(modality).lower().strip()
    if modality_l == "s2":
        x_chw = coerce_single_input_chw(
            input_chw,
            expected_channels=12,
            model_name="TerraFM S2",
        )
        x_chw = np.clip(x_chw / 10000.0, 0.0, 1.0).astype(np.float32)
    elif modality_l == "s1":
        x_chw = coerce_single_input_chw(
            input_chw,
            expected_channels=2,
            model_name="TerraFM S1",
        )
        x_chw = _normalize_s1_vvvh_chw(x_chw)
    else:
        raise ModelError("modality must be 's2' or 's1'.")
    return _resize_chw_to_224(x_chw, size=image_size)


# -----------------------------
# HF asset management (strict)
# -----------------------------
@lru_cache(maxsize=8)
def _ensure_hf_terrafm_weights(
    repo_id: str,
    *,
    auto_download: bool = True,
    cache_dir: str | None = None,
    min_bytes: int = 50 * 1024 * 1024,
) -> str:
    """Returns local TerraFM weight path."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise ModelError("Install huggingface_hub: pip install huggingface_hub") from e

    wt_path = hf_hub_download(repo_id=repo_id, filename=HF_WEIGHT_FILE_B, cache_dir=cache_dir)

    if not os.path.exists(wt_path):
        raise ModelError(f"Failed to download '{HF_WEIGHT_FILE_B}' from {repo_id}.")

    # size sanity: avoid pointer/placeholder
    sz = os.path.getsize(wt_path)
    if sz < min_bytes:
        raise ModelError(
            f"Found '{wt_path}' but it's only {sz} bytes — likely not real weights.\n"
            "Fix (if using LFS/xet):\n"
            "  pip install -U hf_xet\n"
            '  (optional) pip install -U "huggingface_hub[hf_transfer]"\n'
            "Then delete the cached snapshot and re-run.\n"
        )

    return wt_path


@lru_cache(maxsize=1)
def _load_terrafm_module():
    try:
        return importlib.import_module("rs_embed.embedders._vendor.terrafm")
    except Exception as e:
        raise ModelError(
            f"Failed to import vendored TerraFM runtime. Import error: {type(e).__name__}: {e}"
        ) from e


def _assert_weights_loaded(model) -> dict[str, float]:
    """Same philosophy as your RemoteCLIP: param stats should not be near-zero."""
    return verify_loaded_params(model, model_name="TerraFM", check_near_zero=True)


@lru_cache(maxsize=4)
def _load_terrafm_b(
    *,
    dev: str,
    auto_download: bool = True,
    cache_dir: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Returns (model_on_dev, weight_meta).

    Keyed by device like every other embedder's loader: a device-less cache
    shared one nn.Module across per-device instance locks, so concurrent
    cpu/cuda calls raced .to(dev) against a running forward and sequential
    mixed-device calls bounced the weights back and forth.
    """
    import torch

    wt_path = _ensure_hf_terrafm_weights(
        HF_REPO_ID, auto_download=auto_download, cache_dir=cache_dir
    )
    mod = _load_terrafm_module()

    if not hasattr(mod, "terrafm_base"):
        raise ModelError("Vendored TerraFM runtime has no 'terrafm_base()' factory.")

    model = mod.terrafm_base()
    state = torch.load(wt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    stats = _assert_weights_loaded(model)
    model = model.to(dev).eval()

    meta = {
        "hf_repo": HF_REPO_ID,
        "model_source": "vendored_rs_embed_runtime",
        "weight_file": wt_path,
        "weight_file_size": os.path.getsize(wt_path),
        "weights_verified": True,
        "device": str(dev),
        **stats,
    }
    return model, meta


# -----------------------------
# TerraFM forward adapters
# -----------------------------
def _terrafm_pooled_and_grid(
    model,
    x_bchw: np.ndarray,
    *,
    device: str,
    want_grid: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Returns (pooled_vec[D], grid_dhw[D,Ht,Wt] or None)
    """
    import torch

    dev = _auto_device(device)

    x = torch.from_numpy(x_bchw).to(dev)  # [B,C,H,W]
    with torch.no_grad():
        pooled = model(x)  # TerraFM forward returns CLS embedding (B,D)
        pooled_np = pooled[0].detach().float().cpu().numpy().astype(np.float32)

        if not want_grid:
            return pooled_np, None

        # extract_feature returns list of feature maps; we grab last layer by default
        depth = len(getattr(model, "blocks", []))
        if depth <= 0 or not hasattr(model, "extract_feature"):
            raise ModelError(
                "TerraFM model does not expose extract_feature/blocks for grid output."
            )

        last_idx = depth - 1
        feats = model.extract_feature(x, return_h_w=True, out_indices=[last_idx])
        # feats[-1] is (B, C, H, W) for the requested index
        fmap = feats[-1]
        grid = fmap[0].detach().float().cpu().numpy().astype(np.float32)  # [D,Ht,Wt]
        return pooled_np, grid


def _terrafm_pooled_and_grid_batch(
    model,
    x_bchw: np.ndarray,
    *,
    device: str,
    want_grid: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Batch forward for TerraFM.

    Returns:
      pooled [B,D], grid [B,D,Ht,Wt] or None
    """
    import torch

    dev = _auto_device(device)
    x = torch.from_numpy(x_bchw).to(dev)  # [B,C,H,W]

    with torch.inference_mode():
        pooled = model(x)  # [B,D]
        pooled_np = pooled.detach().float().cpu().numpy().astype(np.float32)

        if not want_grid:
            return pooled_np, None

        depth = len(getattr(model, "blocks", []))
        if depth <= 0 or not hasattr(model, "extract_feature"):
            raise ModelError(
                "TerraFM model does not expose extract_feature/blocks for grid output."
            )
        last_idx = depth - 1
        feats = model.extract_feature(x, return_h_w=True, out_indices=[last_idx])
        fmap = feats[-1]  # [B,D,H,W]
        grid = fmap.detach().float().cpu().numpy().astype(np.float32)
        return pooled_np, grid


# -----------------------------
# Embedder
# -----------------------------
@register("terrafm")
class TerraFMBEmbedder(EmbedderBase):
    """
    ROI -> (provider S2 SR 12-band OR S1 VV/VH) -> TerraFM-B -> pooled or grid embedding

    - OutputSpec.pooled(): vec [D]
    - OutputSpec.grid():  grid [D, Ht, Wt] (model-native feature map grid)
    """

    # Square-input model: marks it for the API tiling path (its own fetch_input
    # squares the ROI for the single-input case and skips it when tiling).
    _requires_square_input = True

    # Primary modality spec (S2 12-band). TerraFM also supports S1 VV/VH via
    # its custom fetch_input() override; this spec documents the default path.
    input_spec = ModelInputSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_S2_SR_12_BANDS),
        scale_m=10,
        cloudy_pct=30,
        image_size=224,
        expected_channels=12,
    )

    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 64

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
            "type": "on_the_fly",
            "backend": ["provider", "tensor"],
            "inputs": {
                "s2_sr": {
                    "collection": self.input_spec.collection,
                    "bands": list(self.input_spec.bands),
                },
                "s1": {
                    "collection": "COPERNICUS/S1_GRD_FLOAT (default) or COPERNICUS/S1_GRD",
                    "bands": ["VV", "VH"],
                },
            },
            "modalities": {
                "s2": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": _S2_SR_12_BANDS,
                },
                "s1": {
                    "collection": "COPERNICUS/S1_GRD_FLOAT",
                    "bands": ["VV", "VH"],
                    "defaults": {
                        "use_float_linear": True,
                        "s1_require_iw": True,
                        "s1_relax_iw_on_empty": True,
                    },
                },
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "modality": "s2",  # or "s1"
                "use_float_linear": True,
                "s1_require_iw": True,
                "s1_relax_iw_on_empty": True,
                "image_size": 224,
            },
            "model_config": {
                "cache_dir": {
                    "type": "string",
                    "default": None,
                    "description": (
                        "Weight cache directory for the TerraFM checkpoint download. "
                        "Precedence: model_config > HUGGINGFACE_HUB_CACHE/HF_HOME/"
                        "HUGGINGFACE_HOME env vars > huggingface_hub default cache."
                    ),
                },
            },
            "notes": "grid output is model feature-map grid (not pixel grid).",
        }

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_TERRAFM_FETCH_WORKERS",
                str(TerraFMBEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _resolve_infer_batch(dev: str) -> int:
        default_bs = (
            TerraFMBEmbedder.DEFAULT_BATCH_CUDA
            if str(dev).startswith("cuda")
            else TerraFMBEmbedder.DEFAULT_BATCH_CPU
        )
        v = int(os.environ.get("RS_EMBED_TERRAFM_BATCH_SIZE", str(default_bs)))
        return max(1, v)

    def fetch_input(
        self,
        provider: ProviderBase,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec,
        square_input: bool = True,
    ) -> FetchResult | None:
        """Fetch raw TerraFM input with model-specific logic.

        Returns raw provider values (before normalization).  For S1,
        this includes IW-mode decisions and fallback metadata.

        Parameters
        ----------
        provider : ProviderBase
            Ready provider instance.
        spatial : SpatialSpec
            Spatial request definition.
        temporal : TemporalSpec or None
            Temporal filter (required for TerraFM).
        sensor : SensorSpec
            Sensor/source definition carrying modality and fetch params.

        Returns
        -------
        FetchResult
            Raw CHW array and fetch-time metadata.
        """
        modality = str(getattr(sensor, "modality", "s2") or "s2").lower()
        t = temporal_to_range(temporal)

        # Fetch-square: enlarge a rectangular ROI to a square of real imagery so the
        # encoder sees a square, in-distribution input; the ROI window rides in
        # meta['roi_window_geo'] and the output is cropped back to it (tiled
        # paths crop the stitched grid once). All pipeline callers keep the
        # square_input=True default; False is an escape hatch for callers that
        # manage ROI geometry themselves.
        geo_roi = FULL_WINDOW
        if square_input:
            spatial, geo_roi = square_spatial(spatial)

        if modality == "s2":
            raw = _fetch_collection_patch_chw(
                provider,
                spatial=spatial,
                temporal=t,
                collection="COPERNICUS/S2_SR_HARMONIZED",
                bands=tuple(_S2_SR_12_BANDS),
                scale_m=int(getattr(sensor, "scale_m", 10)),
                cloudy_pct=int(getattr(sensor, "cloudy_pct", 30)),
                composite=str(getattr(sensor, "composite", "median")),
                fill_value=0.0,
            )
            return FetchResult(data=raw, meta=roi_fetch_meta(geo_roi) or {})
        elif modality == "s1":
            raw, meta = _fetch_s1_vvvh_raw_chw_with_meta(
                provider,
                spatial,
                t,
                scale_m=int(getattr(sensor, "scale_m", 10)),
                use_float_linear=bool(getattr(sensor, "use_float_linear", True)),
                composite=str(getattr(sensor, "composite", "median")),
                require_iw=bool(getattr(sensor, "s1_require_iw", True)),
                relax_iw_on_empty=bool(getattr(sensor, "s1_relax_iw_on_empty", True)),
            )
            return FetchResult(data=raw, meta={**meta, **(roi_fetch_meta(geo_roi) or {})})
        else:
            raise ModelError("modality must be 's2' or 's1'.")

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
        uses_provider = backend_l != "tensor"
        # Fetch-square ROI window: carried in fetch_meta when the API prefetched a
        # square, or read from our own fetch_input below. Output cropped back to it.
        geo_roi = geo_roi_from_meta(fetch_meta)

        # defaults / overrides (match your style: sensor carries overrides)
        modality = getattr(sensor, "modality", "s2") if sensor else "s2"
        modality = str(modality).lower()

        # Extract sensor-level fetch params (used in metadata below)
        scale_m = int(getattr(sensor, "scale_m", 10)) if sensor else 10
        cloudy_pct = int(getattr(sensor, "cloudy_pct", 30)) if sensor else 30
        composite = str(getattr(sensor, "composite", "median")) if sensor else "median"
        use_float_linear = bool(getattr(sensor, "use_float_linear", True)) if sensor else True
        s1_require_iw = bool(getattr(sensor, "s1_require_iw", True)) if sensor else True
        s1_relax_iw_on_empty = (
            bool(getattr(sensor, "s1_relax_iw_on_empty", True)) if sensor else True
        )

        image_size = 224
        cache_dir = _resolve_terrafm_cache_dir(model_config)

        # For optional on-the-fly input inspection
        check_meta: dict[str, Any] = {}
        # Provenance spread into output meta (e.g. S1 IW-mode decisions). Seeded
        # from any prefetch fetch_meta, minus the transport-only ROI window.
        fetch_provenance: dict[str, Any] = {
            k: v for k, v in (fetch_meta or {}).items() if k != "roi_window_geo"
        }

        # -----------------
        # Build input tensor
        # -----------------
        if backend_l == "tensor":
            if input_chw is None:
                raise ModelError(
                    "backend='tensor' requires input_chw as CHW. "
                    "Use get_embeddings_batch_from_inputs(...) for batches."
                )
            x_bchw = _prepare_tensor_input_chw(
                input_chw,
                modality=modality,
                image_size=image_size,
            )[None, ...].astype(np.float32)

        else:
            provider = self._get_provider(backend)
            if input_chw is None:
                result = self.fetch_input(
                    provider,
                    spatial=spatial,
                    temporal=temporal,
                    sensor=sensor
                    or SensorSpec(
                        collection="",
                        bands=(),
                        modality=modality,
                    ),
                )
                assert result is not None
                input_chw = result.data
                # fetch_input squared the ROI: take its window, keep the rest as
                # provenance for the output meta.
                geo_roi = geo_roi_from_meta(result.meta)
                fetch_provenance = {
                    k: v for k, v in (result.meta or {}).items() if k != "roi_window_geo"
                }

            # input_chw is raw provider values in the order implied by `sensor.bands`
            if modality == "s2":
                if input_chw.ndim != 3 or int(input_chw.shape[0]) != 12:
                    raise ModelError(
                        f"input_chw must be CHW with 12 bands for TerraFM S2, got {getattr(input_chw, 'shape', None)}"
                    )
                x_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
            elif modality == "s1":
                if input_chw.ndim != 3 or int(input_chw.shape[0]) != 2:
                    raise ModelError(
                        f"input_chw must be CHW with 2 bands (VV,VH) for TerraFM S1, got {getattr(input_chw, 'shape', None)}"
                    )
                x_chw = _normalize_s1_vvvh_chw(input_chw)
            else:
                raise ModelError("modality must be 's2' or 's1'.")

            # Optional: inspect on-the-fly provider input
            from ..tools.inspection import checks_should_raise, maybe_inspect_chw

            check_meta.clear()
            exp_c = 12 if modality == "s2" else 2
            report = maybe_inspect_chw(
                x_chw,
                sensor=sensor,
                name=f"provider_{modality}_chw",
                expected_channels=exp_c,
                value_range=(0.0, 1.0),
                fill_value=0.0,
                meta=check_meta,
            )
            if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
                raise ModelError(
                    "Provider input inspection failed: " + "; ".join(report.get("issues", []))
                )

            # resize to 224
            x_chw = _resize_chw_to_224(x_chw, size=image_size)
            x_bchw = x_chw[None, ...].astype(np.float32)
        # channel sanity: TerraFM HF terrafm.py routes by C==2 (S1) else (S2). Keep it strict.
        c = int(x_bchw.shape[1])
        if c not in (2, 12):
            raise ModelError(f"TerraFM expects C=2 (S1 VV/VH) or C=12 (S2 SR bands). Got C={c}")

        # -----------------
        # Load model (strict weights)
        # -----------------
        model, wmeta = _load_terrafm_b(
            dev=_auto_device(device), auto_download=True, cache_dir=cache_dir
        )

        # Need the feature-map grid for grid output, or to crop+pool the ROI when
        # the input was enlarged to a square at fetch time.
        cropped_to_roi = not roi_is_full(geo_roi)
        pooled, grid = _terrafm_pooled_and_grid(
            model,
            x_bchw.astype(np.float32),
            device=device,
            want_grid=(output.mode == "grid" or cropped_to_roi),
        )

        # Provider fetches always record the resolved window. Tensor inputs
        # record it only when the caller supplied a temporal (it describes
        # their data); otherwise the window is genuinely unknown -> None.
        temporal_used = (
            temporal_to_range(temporal) if (uses_provider or temporal is not None) else None
        )
        sensor_meta = None
        source = None
        if uses_provider:
            if modality == "s2":
                sensor_meta = {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": tuple(_S2_SR_12_BANDS),
                    "scale_m": scale_m,
                    "cloudy_pct": cloudy_pct,
                    "composite": composite,
                }
                source = sensor_meta["collection"]
            elif modality == "s1":
                sensor_meta = {
                    "collection": (
                        "COPERNICUS/S1_GRD_FLOAT" if use_float_linear else "COPERNICUS/S1_GRD"
                    ),
                    "bands": ("VV", "VH"),
                    "scale_m": scale_m,
                    "cloudy_pct": cloudy_pct,
                    "composite": composite,
                    "use_float_linear": use_float_linear,
                    "s1_require_iw": s1_require_iw,
                    "s1_relax_iw_on_empty": s1_relax_iw_on_empty,
                }
                source = sensor_meta["collection"]

        base_meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=backend_l,
            source=source,
            sensor=sensor_meta,
            temporal=temporal_used,
            image_size=image_size,
            extra={
                "modality": modality,
                "scale_m": scale_m if uses_provider else None,
                "cloudy_pct": cloudy_pct if uses_provider else None,
                "composite": composite if uses_provider else None,
                "use_float_linear": (
                    use_float_linear if (uses_provider and modality == "s1") else None
                ),
                "s1_require_iw": s1_require_iw if (uses_provider and modality == "s1") else None,
                "s1_relax_iw_on_empty": (
                    s1_relax_iw_on_empty if (uses_provider and modality == "s1") else None
                ),
                "start": getattr(temporal_used, "start", None),
                "end": getattr(temporal_used, "end", None),
                "image_size": image_size,
                "device": device,
                "hf_cache_dir": cache_dir,
                **fetch_provenance,
                **check_meta,
                **wmeta,
            },
        )

        # ---- pooled output ----
        if output.mode == "pooled":
            if cropped_to_roi and grid is not None:
                # Pool only the ROI tokens (the model's global vector would include
                # the real-neighborhood context fetched only to square the input).
                _, pooled = crop_grid_and_pool(
                    grid, geo_roi, pooling=output.pooling, pooled_fallback=pooled
                )
                base_meta["pooling"] = f"roi_grid_{output.pooling}"
            return Embedding(data=pooled.astype(np.float32), meta=base_meta)

        # ---- grid output ----
        if output.mode == "grid":
            if grid is None:
                raise ModelError("Grid output requested but TerraFM grid extraction returned None.")
            if cropped_to_roi:
                grid = crop_grid_to_roi(grid, geo_roi)

            meta = {
                **base_meta,
                "grid_type": "feature_map",
                "grid_shape": tuple(grid.shape),
            }
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
        if not spatials:
            return []

        backend_l = backend.lower().strip()
        if backend_l == "tensor":
            raise ModelError(
                "backend='tensor' batch inference requires get_embeddings_batch_from_inputs(...)."
            )
        provider = self._get_provider(backend)

        t = temporal_to_range(temporal)

        modality = str(getattr(sensor, "modality", "s2") if sensor else "s2").lower()
        scale_m = int(getattr(sensor, "scale_m", 10) if sensor else 10)
        cloudy_pct = int(getattr(sensor, "cloudy_pct", 30) if sensor else 30)
        composite = str(getattr(sensor, "composite", "median") if sensor else "median")
        use_float_linear = bool(getattr(sensor, "use_float_linear", True)) if sensor else True
        s1_require_iw = bool(getattr(sensor, "s1_require_iw", True)) if sensor else True
        s1_relax_iw_on_empty = (
            bool(getattr(sensor, "s1_relax_iw_on_empty", True)) if sensor else True
        )

        n = len(spatials)
        prefetched_raw: list[np.ndarray | None] = [None] * n
        prefetched_meta: list[dict[str, Any] | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray, dict[str, Any] | None]:
            # Fetch-square each ROI; the ROI window rides in the per-item meta
            # (roi_window_geo) so from_inputs crops each output back to it.
            sq, geo_roi = square_spatial(sp)
            if modality == "s2":
                x_chw = _fetch_s2_sr_12_chw(
                    provider,
                    sq,
                    t,
                    scale_m=scale_m,
                    cloudy_pct=cloudy_pct,
                    composite=composite,
                )
                # get_embedding(input_chw=...) expects raw S2 SR in [0..10000]
                raw = np.clip(x_chw * 10000.0, 0.0, 10000.0).astype(np.float32)
                return i, raw, roi_fetch_meta(geo_roi)
            if modality == "s1":
                raw, fetch_meta = _fetch_s1_vvvh_raw_chw_with_meta(
                    provider,
                    sq,
                    t,
                    scale_m=scale_m,
                    use_float_linear=use_float_linear,
                    composite=composite,
                    require_iw=s1_require_iw,
                    relax_iw_on_empty=s1_relax_iw_on_empty,
                )
                return i, raw, {**fetch_meta, **(roi_fetch_meta(geo_roi) or {})}
            raise ModelError("modality must be 's2' or 's1'.")

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                ii, raw, fetch_meta = _fetch_one(i, sp)
                prefetched_raw[ii] = raw
                prefetched_meta[ii] = fetch_meta
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, raw, fetch_meta = fut.result()
                    prefetched_raw[i] = raw
                    prefetched_meta[i] = fetch_meta

        raw_inputs: list[np.ndarray] = []
        for i, raw in enumerate(prefetched_raw):
            if raw is None:
                raise ModelError(f"Missing prefetched input at index={i} for terrafm_b.")
            raw_inputs.append(raw)

        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            fetch_metas=prefetched_meta,
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
        fetch_metas: list[dict[str, Any] | None] | None = None,
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if fetch_metas is not None and len(fetch_metas) != len(input_chws):
            raise ModelError(
                f"fetch_metas/input_chws length mismatch: {len(fetch_metas)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        backend_l = backend.lower().strip()
        uses_provider = backend_l != "tensor"
        if uses_provider:
            self._get_provider(backend)

        modality = str(getattr(sensor, "modality", "s2") if sensor else "s2").lower()
        scale_m = int(getattr(sensor, "scale_m", 10) if sensor else 10)
        cloudy_pct = int(getattr(sensor, "cloudy_pct", 30) if sensor else 30)
        composite = str(getattr(sensor, "composite", "median") if sensor else "median")
        use_float_linear = bool(getattr(sensor, "use_float_linear", True)) if sensor else True
        s1_require_iw = bool(getattr(sensor, "s1_require_iw", True)) if sensor else True
        s1_relax_iw_on_empty = (
            bool(getattr(sensor, "s1_relax_iw_on_empty", True)) if sensor else True
        )

        image_size = 224
        cache_dir = _resolve_terrafm_cache_dir(model_config)

        x_bchw_all: list[np.ndarray] = []
        for i, input_chw in enumerate(input_chws):
            try:
                x_bchw_all.append(
                    _prepare_tensor_input_chw(
                        input_chw,
                        modality=modality,
                        image_size=image_size,
                    )
                )
            except ModelError as exc:
                raise ModelError(f"{exc} at index={i}") from exc

        model, wmeta = _load_terrafm_b(
            dev=_auto_device(device), auto_download=True, cache_dir=cache_dir
        )
        dev = str(wmeta.get("device", _auto_device(device)))
        infer_bs = self._resolve_infer_batch(dev)

        # Provider fetches always record the resolved window. Tensor inputs
        # record it only when the caller supplied a temporal (it describes
        # their data); otherwise the window is genuinely unknown -> None.
        temporal_used = (
            temporal_to_range(temporal) if (uses_provider or temporal is not None) else None
        )
        sensor_meta = None
        source = None
        if uses_provider and modality == "s2":
            sensor_meta = {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": tuple(_S2_SR_12_BANDS),
                "scale_m": scale_m,
                "cloudy_pct": cloudy_pct,
                "composite": composite,
            }
            source = sensor_meta["collection"]
        elif uses_provider and modality == "s1":
            sensor_meta = {
                "collection": (
                    "COPERNICUS/S1_GRD_FLOAT" if use_float_linear else "COPERNICUS/S1_GRD"
                ),
                "bands": ("VV", "VH"),
                "scale_m": scale_m,
                "cloudy_pct": cloudy_pct,
                "composite": composite,
                "use_float_linear": use_float_linear,
                "s1_require_iw": s1_require_iw,
                "s1_relax_iw_on_empty": s1_relax_iw_on_empty,
            }
            source = sensor_meta["collection"]

        # Per-item fetch-square ROI windows (full frame when fetch_metas is
        # absent — direct user inputs). The grid is needed whenever any item was
        # enlarged to a square, so it can be cropped back to its ROI.
        geo_rois = [
            geo_roi_from_meta(
                fetch_metas[i] if (fetch_metas is not None and i < len(fetch_metas)) else None
            )
            for i in range(len(spatials))
        ]
        any_cropped = any(not roi_is_full(g) for g in geo_rois)

        out: list[Embedding | None] = [None] * len(spatials)
        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            xb = np.stack(x_bchw_all[s0:s1], axis=0).astype(np.float32)
            pooled_bd, grid_bdhw = _terrafm_pooled_and_grid_batch(
                model,
                xb,
                device=dev,
                want_grid=(output.mode == "grid" or any_cropped),
            )
            for j in range(s1 - s0):
                i = s0 + j
                geo_roi = geo_rois[i]
                cropped_to_roi = not roi_is_full(geo_roi)
                fetch_meta = (
                    {k: v for k, v in (fetch_metas[i] or {}).items() if k != "roi_window_geo"}
                    if fetch_metas is not None and i < len(fetch_metas)
                    else {}
                )
                base_meta = build_meta(
                    model=self.model_name,
                    kind="on_the_fly",
                    backend=backend_l,
                    source=source,
                    sensor=sensor_meta,
                    temporal=temporal_used,
                    image_size=image_size,
                    extra={
                        "modality": modality,
                        "scale_m": scale_m if uses_provider else None,
                        "cloudy_pct": cloudy_pct if uses_provider else None,
                        "composite": composite if uses_provider else None,
                        "use_float_linear": (
                            use_float_linear if (uses_provider and modality == "s1") else None
                        ),
                        "s1_require_iw": (
                            s1_require_iw if (uses_provider and modality == "s1") else None
                        ),
                        "s1_relax_iw_on_empty": (
                            s1_relax_iw_on_empty if (uses_provider and modality == "s1") else None
                        ),
                        "start": getattr(temporal_used, "start", None),
                        "end": getattr(temporal_used, "end", None),
                        "image_size": image_size,
                        "device": dev,
                        "hf_cache_dir": cache_dir,
                        "batch_infer": True,
                        "input_override": True,
                        **fetch_meta,
                        **wmeta,
                    },
                )

                if output.mode == "pooled":
                    pooled_vec = pooled_bd[j]
                    if cropped_to_roi and grid_bdhw is not None:
                        _, pooled_vec = crop_grid_and_pool(
                            grid_bdhw[j],
                            geo_roi,
                            pooling=output.pooling,
                            pooled_fallback=pooled_vec,
                        )
                        base_meta["pooling"] = f"roi_grid_{output.pooling}"
                    out[i] = Embedding(data=pooled_vec.astype(np.float32), meta=base_meta)
                    continue

                if output.mode == "grid":
                    if grid_bdhw is None:
                        raise ModelError(
                            "Grid output requested but TerraFM grid extraction returned None."
                        )
                    grid = grid_bdhw[j]
                    if cropped_to_roi:
                        grid = crop_grid_to_roi(grid, geo_roi)
                    meta = {
                        **base_meta,
                        "grid_type": "feature_map",
                        "grid_shape": tuple(grid.shape),
                    }
                    da = grid_to_dataarray(grid, meta=meta)
                    out[i] = Embedding(data=da, meta=meta)
                    continue

                raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError("terrafm_b batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

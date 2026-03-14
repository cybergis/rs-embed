# src/rs_embed/embedders/onthefly_terrafm.py
from __future__ import annotations

import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np
import xarray as xr

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers import ProviderBase
from .base import EmbedderBase
from .meta_utils import build_meta, temporal_midpoint_str
from .runtime_utils import (
    coerce_single_input_chw,
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
    fetch_s1_vvvh_raw_chw as _fetch_s1_vvvh_raw_chw_shared,
    normalize_s1_vvvh_chw as _normalize_s1_vvvh_chw,
    resolve_device_auto_torch as _auto_device,
)

HF_REPO_ID = "MBZUAI/TerraFM"
HF_WEIGHT_FILE_B = "TerraFM-B.pth"

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
    orbit: str | None = None,  # "ASCENDING" | "DESCENDING"
    use_float_linear: bool = True,
    composite: str = "median",
) -> np.ndarray:
    """Returns normalized S1 VV/VH CHW float32 [2,H,W] in [0,1]."""
    raw = _fetch_s1_vvvh_raw_chw_shared(
        provider,
        spatial=spatial,
        temporal=temporal,
        scale_m=int(scale_m),
        orbit=orbit,
        use_float_linear=bool(use_float_linear),
        composite=str(composite),
        fill_value=0.0,
    )
    return _normalize_s1_vvvh_chw(raw)

def _fetch_s1_vvvh_raw_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    orbit: str | None = None,
    use_float_linear: bool = True,
    composite: str = "median",
) -> np.ndarray:
    """Returns raw VV/VH CHW without log/normalization."""
    return _fetch_s1_vvvh_raw_chw_shared(
        provider,
        spatial=spatial,
        temporal=temporal,
        scale_m=int(scale_m),
        orbit=orbit,
        use_float_linear=bool(use_float_linear),
        composite=str(composite),
        fill_value=0.0,
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
    import torch

    p = None
    for _, param in model.named_parameters():
        if param is not None and param.numel() > 0:
            p = param.detach()
            break
    if p is None:
        raise ModelError("TerraFM model has no parameters; cannot verify weights.")
    if not torch.isfinite(p).all():
        raise ModelError("TerraFM parameters contain NaN/Inf; load likely failed.")

    p_f = p.float()
    std = float(p_f.std().cpu())
    mx = float(p_f.abs().max().cpu())
    mean = float(p_f.mean().cpu())
    if std < 1e-6 and mx < 1e-5:
        raise ModelError("TerraFM parameters look uninitialized (near-zero stats).")
    return {"param_mean": mean, "param_std": std, "param_absmax": mx}

@lru_cache(maxsize=4)
def _load_terrafm_b(
    *,
    auto_download: bool = True,
    cache_dir: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Returns (model, weight_meta).
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

    meta = {
        "hf_repo": HF_REPO_ID,
        "model_source": "vendored_rs_embed_runtime",
        "weight_file": wt_path,
        "weight_file_size": os.path.getsize(wt_path),
        "weights_verified": True,
        **stats,
    }
    return model, meta

# -----------------------------
# TerraFM forward adapters
# -----------------------------
def _terrafm_pooled_and_grid(
    model,
    x_bchw: "np.ndarray",
    *,
    device: str,
    want_grid: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Returns (pooled_vec[D], grid_dhw[D,Ht,Wt] or None)
    """
    import torch

    dev = _auto_device(device)
    model = model.to(dev).eval()

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
    x_bchw: "np.ndarray",
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
    model = model.to(dev).eval()
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

    DEFAULT_FETCH_WORKERS = 8
    DEFAULT_BATCH_CPU = 8
    DEFAULT_BATCH_CUDA = 64

    def describe(self) -> dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider", "tensor"],
            "inputs": {
                "s2_sr": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": _S2_SR_12_BANDS,
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
                        "orbit": None,
                        "use_float_linear": True,
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
                "orbit": None,
                "use_float_linear": True,
                "image_size": 224,
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
        backend_l = backend.lower().strip()
        uses_provider = backend_l != "tensor"

        # defaults / overrides (match your style: sensor carries overrides)
        modality = getattr(sensor, "modality", "s2") if sensor else "s2"
        modality = str(modality).lower()

        scale_m = getattr(sensor, "scale_m", 10) if sensor else 10
        cloudy_pct = getattr(sensor, "cloudy_pct", 30) if sensor else 30
        composite = getattr(sensor, "composite", "median") if sensor else "median"
        orbit = getattr(sensor, "orbit", None) if sensor else None
        use_float_linear = bool(getattr(sensor, "use_float_linear", True)) if sensor else True

        image_size = 224
        cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HOME")
        )

        # For optional on-the-fly input inspection
        check_meta: dict[str, Any] = {}

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
            if temporal is None:
                raise ModelError("terrafm_b_gee requires TemporalSpec.range(start,end).")
            temporal.validate()
            if temporal.mode != "range":
                raise ModelError("terrafm_b_gee requires TemporalSpec.range in v0.1.")
            if input_chw is None:
                if modality == "s2":
                    x_chw = _fetch_s2_sr_12_chw(
                        provider,
                        spatial,
                        temporal,
                        scale_m=scale_m,
                        cloudy_pct=cloudy_pct,
                        composite=composite,
                    )  # [12,H,W]
                elif modality == "s1":
                    x_chw = _fetch_s1_vvvh_chw(
                        provider,
                        spatial,
                        temporal,
                        scale_m=scale_m,
                        orbit=orbit,
                        use_float_linear=use_float_linear,
                        composite=composite,
                    )  # [2,H,W]
                else:
                    raise ModelError("modality must be 's2' or 's1'.")
            else:
                # input_chw is expected to be raw provider values in the order implied by `sensor.bands`
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
                    x = input_chw.astype(np.float32)
                    x = np.log1p(np.maximum(x, 0.0))
                    denom = np.percentile(x, 99) if np.isfinite(x).all() else 1.0
                    denom = float(denom) if denom > 0 else 1.0
                    x_chw = np.clip(x / denom, 0.0, 1.0).astype(np.float32)
                else:
                    raise ModelError("modality must be 's2' or 's1'.")

            # Optional: inspect on-the-fly provider input
            from ..tools.inspection import maybe_inspect_chw, checks_should_raise

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
        model, wmeta = _load_terrafm_b(auto_download=True, cache_dir=cache_dir)

        pooled, grid = _terrafm_pooled_and_grid(
            model,
            x_bchw.astype(np.float32),
            device=device,
            want_grid=(output.mode == "grid"),
        )

        temporal_used = temporal if uses_provider else None
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
                    "orbit": orbit,
                    "use_float_linear": use_float_linear,
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
            input_time=temporal_midpoint_str(temporal_used),
            extra={
                "modality": modality,
                "scale_m": scale_m if uses_provider else None,
                "cloudy_pct": cloudy_pct if uses_provider else None,
                "composite": composite if uses_provider else None,
                "orbit": orbit if (uses_provider and modality == "s1") else None,
                "use_float_linear": (
                    use_float_linear if (uses_provider and modality == "s1") else None
                ),
                "start": getattr(temporal_used, "start", None),
                "end": getattr(temporal_used, "end", None),
                "image_size": image_size,
                "device": device,
                "hf_cache_dir": cache_dir,
                **check_meta,
                **wmeta,
            },
        )

        # ---- pooled output ----
        if output.mode == "pooled":
            return Embedding(data=pooled.astype(np.float32), meta=base_meta)

        # ---- grid output ----
        if output.mode == "grid":
            if grid is None:
                raise ModelError("Grid output requested but TerraFM grid extraction returned None.")

            meta = {
                **base_meta,
                "grid_type": "feature_map",
                "grid_shape": tuple(grid.shape),
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

        backend_l = backend.lower().strip()
        if backend_l == "tensor":
            raise ModelError(
                "backend='tensor' batch inference requires get_embeddings_batch_from_inputs(...)."
            )
        provider = self._get_provider(backend)

        if temporal is None:
            raise ModelError("terrafm_b_gee requires TemporalSpec.range(start,end).")
        temporal.validate()
        if temporal.mode != "range":
            raise ModelError("terrafm_b_gee requires TemporalSpec.range in v0.1.")

        modality = str(getattr(sensor, "modality", "s2") if sensor else "s2").lower()
        scale_m = int(getattr(sensor, "scale_m", 10) if sensor else 10)
        cloudy_pct = int(getattr(sensor, "cloudy_pct", 30) if sensor else 30)
        composite = str(getattr(sensor, "composite", "median") if sensor else "median")
        orbit = getattr(sensor, "orbit", None) if sensor else None
        use_float_linear = bool(getattr(sensor, "use_float_linear", True)) if sensor else True

        n = len(spatials)
        prefetched_raw: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            if modality == "s2":
                x_chw = _fetch_s2_sr_12_chw(
                    provider,
                    sp,
                    temporal,
                    scale_m=scale_m,
                    cloudy_pct=cloudy_pct,
                    composite=composite,
                )
                # get_embedding(input_chw=...) expects raw S2 SR in [0..10000]
                raw = np.clip(x_chw * 10000.0, 0.0, 10000.0).astype(np.float32)
                return i, raw
            if modality == "s1":
                raw = _fetch_s1_vvvh_raw_chw(
                    provider,
                    sp,
                    temporal,
                    scale_m=scale_m,
                    orbit=orbit,
                    use_float_linear=use_float_linear,
                    composite=composite,
                )
                return i, raw
            raise ModelError("modality must be 's2' or 's1'.")

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
                raise ModelError(f"Missing prefetched input at index={i} for terrafm_b.")
            raw_inputs.append(raw)

        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            temporal=temporal,
            sensor=sensor,
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
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if len(spatials) != len(input_chws):
            raise ModelError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if not spatials:
            return []

        backend_l = backend.lower().strip()
        uses_provider = backend_l != "tensor"
        if uses_provider:
            self._get_provider(backend)
            if temporal is None:
                raise ModelError("terrafm_b_gee requires TemporalSpec.range(start,end).")
            temporal.validate()
            if temporal.mode != "range":
                raise ModelError("terrafm_b_gee requires TemporalSpec.range in v0.1.")

        modality = str(getattr(sensor, "modality", "s2") if sensor else "s2").lower()
        scale_m = int(getattr(sensor, "scale_m", 10) if sensor else 10)
        cloudy_pct = int(getattr(sensor, "cloudy_pct", 30) if sensor else 30)
        composite = str(getattr(sensor, "composite", "median") if sensor else "median")
        orbit = getattr(sensor, "orbit", None) if sensor else None
        use_float_linear = bool(getattr(sensor, "use_float_linear", True)) if sensor else True

        image_size = 224
        cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HOME")
        )

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

        model, wmeta = _load_terrafm_b(auto_download=True, cache_dir=cache_dir)
        dev = str(wmeta.get("device", _auto_device(device)))
        infer_bs = self._resolve_infer_batch(dev)

        temporal_used = temporal if uses_provider else None
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
                "orbit": orbit,
                "use_float_linear": use_float_linear,
            }
            source = sensor_meta["collection"]

        out: list[Embedding | None] = [None] * len(spatials)
        n = len(spatials)
        for s0 in range(0, n, infer_bs):
            s1 = min(n, s0 + infer_bs)
            xb = np.stack(x_bchw_all[s0:s1], axis=0).astype(np.float32)
            pooled_bd, grid_bdhw = _terrafm_pooled_and_grid_batch(
                model,
                xb,
                device=dev,
                want_grid=(output.mode == "grid"),
            )
            for j in range(s1 - s0):
                i = s0 + j
                base_meta = build_meta(
                    model=self.model_name,
                    kind="on_the_fly",
                    backend=backend_l,
                    source=source,
                    sensor=sensor_meta,
                    temporal=temporal_used,
                    image_size=image_size,
                    input_time=temporal_midpoint_str(temporal_used),
                    extra={
                        "modality": modality,
                        "scale_m": scale_m if uses_provider else None,
                        "cloudy_pct": cloudy_pct if uses_provider else None,
                        "composite": composite if uses_provider else None,
                        "orbit": orbit if (uses_provider and modality == "s1") else None,
                        "use_float_linear": (
                            use_float_linear if (uses_provider and modality == "s1") else None
                        ),
                        "start": getattr(temporal_used, "start", None),
                        "end": getattr(temporal_used, "end", None),
                        "image_size": image_size,
                        "device": dev,
                        "hf_cache_dir": cache_dir,
                        "batch_infer": True,
                        "input_override": True,
                        **wmeta,
                    },
                )

                if output.mode == "pooled":
                    out[i] = Embedding(data=pooled_bd[j].astype(np.float32), meta=base_meta)
                    continue

                if output.mode == "grid":
                    if grid_bdhw is None:
                        raise ModelError(
                            "Grid output requested but TerraFM grid extraction returned None."
                        )
                    grid = grid_bdhw[j]
                    meta = {
                        **base_meta,
                        "grid_type": "feature_map",
                        "grid_shape": tuple(grid.shape),
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
                    out[i] = Embedding(data=da, meta=meta)
                    continue

                raise ModelError(f"Unknown output mode: {output.mode}")

        if any(e is None for e in out):
            raise ModelError("terrafm_b batch inference produced incomplete outputs.")
        return [e for e in out if e is not None]

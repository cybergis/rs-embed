from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..providers import ProviderBase
from .base import EmbedderBase
from .runtime_utils import (
    coerce_single_input_chw,
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
    load_cached_with_device as _load_cached_with_device,
    resolve_device_auto_torch as _resolve_device,
)
from .meta_utils import build_meta, temporal_midpoint_str, temporal_to_range
from ._vit_mae_utils import ensure_torch, pool_from_tokens, tokens_to_grid_dhw

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

_TERRAMIND_S2L2A_BANDS = [
    "COASTAL_AEROSOL",
    "BLUE",
    "GREEN",
    "RED",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
    "NIR_BROAD",
    "NIR_NARROW",
    "WATER_VAPOR",
    "SWIR_1",
    "SWIR_2",
]

# From TerraTorch terramind_register.py (v1_pretraining_mean/std for untok_sen2l2a@224)
_V1_MEAN = np.array(
    [
        1390.458,
        1503.317,
        1718.197,
        1853.91,
        2199.1,
        2779.975,
        2987.011,
        3083.234,
        3132.22,
        3162.988,
        2424.884,
        1857.648,
    ],
    dtype=np.float32,
)
_V1_STD = np.array(
    [
        2106.761,
        2141.107,
        2038.973,
        2134.138,
        2085.321,
        1889.926,
        1820.257,
        1871.918,
        1753.829,
        1797.379,
        1434.261,
        1334.311,
    ],
    dtype=np.float32,
)

# From TerraTorch terramind_register.py (v01_pretraining_mean/std for untok_sen2l2a@224)
_V01_MEAN = np.array(
    [
        794.311,
        925.161,
        1183.128,
        1338.041,
        1667.254,
        2233.633,
        2460.96,
        2555.569,
        2619.542,
        2703.298,
        2406.497,
        1841.645,
    ],
    dtype=np.float32,
)
_V01_STD = np.array(
    [
        1164.883,
        1205.586,
        1223.713,
        1399.638,
        1403.298,
        1378.513,
        1434.924,
        1491.141,
        1454.089,
        1660.395,
        1473.248,
        1365.08,
    ],
    dtype=np.float32,
)

def _resize_chw(x_chw: np.ndarray, *, size: int = 224) -> np.ndarray:
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW array, got {x_chw.shape}")
    x = torch.from_numpy(x_chw.astype(np.float32, copy=False)).unsqueeze(0)
    y = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return y[0].detach().cpu().numpy().astype(np.float32)

def _fetch_s2_sr_12_raw_chw(
    provider: ProviderBase,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    raw = _fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=tuple(_S2_SR_12_BANDS),
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        fill_value=fill_value,
    )
    return np.clip(raw, 0.0, 10000.0).astype(np.float32)

def _terramind_zscore_s2(raw_chw: np.ndarray, *, model_key: str, mode: str) -> np.ndarray:
    if raw_chw.ndim != 3 or int(raw_chw.shape[0]) != len(_S2_SR_12_BANDS):
        raise ModelError(
            f"TerraMind expects CHW with 12 S2 bands, got {getattr(raw_chw, 'shape', None)}"
        )

    mode_l = str(mode).lower().strip()
    if mode_l in ("none", "off", "raw"):
        x = raw_chw.astype(np.float32, copy=False)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    use_v01 = str(model_key).lower().startswith("terramind_v01")
    mean = _V01_MEAN if use_v01 else _V1_MEAN
    std = _V01_STD if use_v01 else _V1_STD
    std = np.maximum(std, 1e-6)

    x = raw_chw.astype(np.float32, copy=False)
    x = (x - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

@lru_cache(maxsize=8)
def _load_terramind_cached(
    model_key: str,
    pretrained: bool,
    modality: str,
    dev: str,
) -> tuple[Any, dict[str, Any]]:
    ensure_torch()
    import torch

    try:
        from terratorch.registry import BACKBONE_REGISTRY
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")).split(".")[0] == "terratorch":
            raise ModelError(
                "TerraMind requires terratorch. Install: pip install terratorch"
            ) from e
        raise ModelError(
            "Failed to import terratorch registry while loading TerraMind. "
            f"Missing dependency: {getattr(e, 'name', None) or e}. "
            "Check optional mmseg/mmengine deps or process-level shim/module conflicts."
        ) from e
    except Exception as e:
        raise ModelError(
            f"Failed to import terratorch registry while loading TerraMind: {type(e).__name__}: {e}"
        ) from e

    try:
        model = BACKBONE_REGISTRY.build(
            str(model_key),
            pretrained=bool(pretrained),
            modalities=[str(modality)],
        )
    except Exception as e:
        raise ModelError(
            f"Failed to build TerraMind backbone '{model_key}'. "
            "Check terratorch install and model_key (e.g. terramind_v1_small)."
        ) from e

    try:
        model = model.to(dev).eval()
    except Exception as _e:
        pass

    p0 = None
    for _, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError("TerraMind model has no parameters; cannot verify weights.")
    if not torch.isfinite(p0).all():
        raise ModelError("TerraMind parameters contain NaN/Inf; load likely failed.")

    p0f = p0.float()
    meta = {
        "model_key": str(model_key),
        "pretrained": bool(pretrained),
        "modality": str(modality),
        "device": str(dev),
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
    }
    return model, meta

def _load_terramind(
    *,
    model_key: str,
    pretrained: bool,
    modality: str,
    device: str,
) -> tuple[Any, dict[str, Any], str]:
    (loaded, dev) = _load_cached_with_device(
        _load_terramind_cached,
        device=device,
        model_key=str(model_key),
        pretrained=bool(pretrained),
        modality=str(modality),
    )
    model, meta = loaded
    return model, meta, dev

def _terramind_forward_tokens(
    model: Any,
    x_bchw: np.ndarray,
    *,
    modality: str,
    layer_index: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_torch()
    import torch

    dev = _resolve_device(device)
    model = model.to(dev).eval()
    x = torch.from_numpy(x_bchw.astype(np.float32, copy=False)).to(dev)

    with torch.no_grad():
        out = None
        # Preferred path: explicit modality dict
        try:
            out = model({str(modality): x})
        except Exception as _e:
            # Fallback for wrappers that accept plain tensor
            out = model(x)

    def _pick_from_sequence(seq: Any, idx: int) -> torch.Tensor | None:
        if not isinstance(seq, (list, tuple)) or len(seq) == 0:
            return None
        cand = None
        try:
            cand = seq[idx]
        except Exception as _e:
            cand = None
        if torch.is_tensor(cand) and cand.ndim == 3:
            return cand
        for v in reversed(seq):
            if torch.is_tensor(v) and v.ndim == 3:
                return v
        return None

    toks_t = None
    if isinstance(out, (list, tuple)):
        toks_t = _pick_from_sequence(out, layer_index)
    elif isinstance(out, dict):
        vals = list(out.values())
        toks_t = _pick_from_sequence(vals, layer_index)
        if toks_t is None:
            for v in vals:
                if torch.is_tensor(v) and v.ndim == 3:
                    toks_t = v
                    break
    elif hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
        if out.last_hidden_state.ndim == 3:
            toks_t = out.last_hidden_state
    elif torch.is_tensor(out) and out.ndim == 3:
        toks_t = out

    if toks_t is None:
        raise ModelError(
            f"TerraMind forward did not return token tensor [B,N,D]. Got type={type(out)}."
        )

    tokens = toks_t[0].detach().float().cpu().numpy().astype(np.float32)
    meta = {
        "tokens_shape": tuple(tokens.shape),
        "layer_index": int(layer_index),
        "tokens_include_cls": False,
    }
    return tokens, meta

def _terramind_forward_tokens_batch(
    model: Any,
    x_bchw: np.ndarray,
    *,
    modality: str,
    layer_index: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_torch()
    import torch

    dev = _resolve_device(device)
    model = model.to(dev).eval()
    x = torch.from_numpy(x_bchw.astype(np.float32, copy=False)).to(dev)

    with torch.no_grad():
        out = None
        try:
            out = model({str(modality): x})
        except Exception as _e:
            out = model(x)

    def _pick_from_sequence(seq: Any, idx: int) -> torch.Tensor | None:
        if not isinstance(seq, (list, tuple)) or len(seq) == 0:
            return None
        cand = None
        try:
            cand = seq[idx]
        except Exception as _e:
            cand = None
        if torch.is_tensor(cand) and cand.ndim == 3:
            return cand
        for v in reversed(seq):
            if torch.is_tensor(v) and v.ndim == 3:
                return v
        return None

    toks_t = None
    if isinstance(out, (list, tuple)):
        toks_t = _pick_from_sequence(out, layer_index)
    elif isinstance(out, dict):
        vals = list(out.values())
        toks_t = _pick_from_sequence(vals, layer_index)
        if toks_t is None:
            for v in vals:
                if torch.is_tensor(v) and v.ndim == 3:
                    toks_t = v
                    break
    elif hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
        if out.last_hidden_state.ndim == 3:
            toks_t = out.last_hidden_state
    elif torch.is_tensor(out) and out.ndim == 3:
        toks_t = out

    if toks_t is None:
        raise ModelError(
            f"TerraMind forward did not return token tensor [B,N,D]. Got type={type(out)}."
        )

    tokens = toks_t.detach().float().cpu().numpy().astype(np.float32)
    meta = {
        "tokens_shape": tuple(tokens.shape[1:]),
        "batch_tokens_shape": tuple(tokens.shape),
        "layer_index": int(layer_index),
        "tokens_include_cls": False,
    }
    return tokens, meta

def _prepare_terramind_input_chw(
    input_chw: Any,
    *,
    image_size: int,
    model_key: str,
    normalize_mode: str,
) -> np.ndarray:
    raw_chw = coerce_single_input_chw(
        input_chw,
        expected_channels=len(_S2_SR_12_BANDS),
        model_name="TerraMind",
    )
    raw_chw = np.clip(raw_chw, 0.0, 10000.0).astype(np.float32)
    raw_chw = _resize_chw(raw_chw, size=image_size)
    return _terramind_zscore_s2(raw_chw, model_key=model_key, mode=normalize_mode)

@register("terramind")
class TerraMindEmbedder(EmbedderBase):
    DEFAULT_MODEL_KEY = "terramind_v1_small"
    DEFAULT_MODALITY = "S2L2A"
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_FETCH_WORKERS = 8

    def describe(self) -> dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider", "tensor"],
            "inputs": {
                "s2_sr": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": _S2_SR_12_BANDS,
                }
            },
            "modalities": {
                "s2_l2a": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": _S2_SR_12_BANDS,
                    "defaults": {"modality": self.DEFAULT_MODALITY},
                }
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "model_key": self.DEFAULT_MODEL_KEY,
                "modality": self.DEFAULT_MODALITY,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "normalization": "zscore_v1_or_v01",
            },
            "notes": [
                "Loads TerraMind backbone via terratorch BACKBONE_REGISTRY.",
                "grid output is ViT patch-token grid (typically 14x14 for 224/16).",
            ],
        }

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_TERRAMIND_FETCH_WORKERS",
                str(TerraMindEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return SensorSpec(
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=tuple(_S2_SR_12_BANDS),
            scale_m=10,
            cloudy_pct=30,
            composite="median",
            fill_value=0.0,
        )

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

        model_key = os.environ.get("RS_EMBED_TERRAMIND_MODEL_KEY", self.DEFAULT_MODEL_KEY).strip()
        modality = str(
            getattr(sensor, "modality", None)
            or os.environ.get("RS_EMBED_TERRAMIND_MODALITY", self.DEFAULT_MODALITY).strip()
            or self.DEFAULT_MODALITY
        )
        if modality.strip().lower().replace("-", "_") == "s2_l2a":
            modality = self.DEFAULT_MODALITY
        normalize_mode = os.environ.get("RS_EMBED_TERRAMIND_NORMALIZE", "zscore").strip()
        layer_index = int(os.environ.get("RS_EMBED_TERRAMIND_LAYER_INDEX", "-1"))
        pretrained = os.environ.get("RS_EMBED_TERRAMIND_PRETRAINED", "1").strip() not in {
            "0",
            "false",
            "False",
        }
        image_size = self.DEFAULT_IMAGE_SIZE

        check_meta: dict[str, Any] = {}
        source = None
        sensor_meta = None
        temporal_used: TemporalSpec | None = None

        if backend_l == "tensor":
            if input_chw is None:
                raise ModelError(
                    "backend='tensor' requires input_chw as CHW. "
                    "Use get_embeddings_batch_from_inputs(...) for batches."
                )
            x_bchw = _prepare_terramind_input_chw(
                input_chw,
                image_size=image_size,
                model_key=model_key,
                normalize_mode=normalize_mode,
            )[None, ...].astype(np.float32)

        else:
            provider = self._get_provider(backend)
            t = temporal_to_range(temporal)
            temporal_used = t
            ss = sensor or self._default_sensor()

            scale_m = int(getattr(ss, "scale_m", 10))
            cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
            composite = str(getattr(ss, "composite", "median"))
            fill_value = float(getattr(ss, "fill_value", 0.0))

            if input_chw is None:
                raw_chw = _fetch_s2_sr_12_raw_chw(
                    provider,
                    spatial,
                    t,
                    scale_m=scale_m,
                    cloudy_pct=cloudy_pct,
                    composite=composite,
                    fill_value=fill_value,
                )
            else:
                if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(_S2_SR_12_BANDS):
                    raise ModelError(
                        f"input_chw must be CHW with 12 bands for TerraMind, got {getattr(input_chw, 'shape', None)}"
                    )
                raw_chw = np.asarray(input_chw, dtype=np.float32)
                raw_chw = np.clip(
                    np.nan_to_num(raw_chw, nan=0.0, posinf=0.0, neginf=0.0),
                    0.0,
                    10000.0,
                ).astype(np.float32)

            from ..tools.inspection import maybe_inspect_chw, checks_should_raise

            check_meta.clear()
            report = maybe_inspect_chw(
                raw_chw,
                sensor=sensor,
                name="provider_s2_sr_12_raw_chw",
                expected_channels=len(_S2_SR_12_BANDS),
                value_range=(0.0, 10000.0),
                fill_value=fill_value,
                meta=check_meta,
            )
            if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
                raise ModelError(
                    "Provider input inspection failed: " + "; ".join(report.get("issues", []))
                )

            raw_chw = _resize_chw(raw_chw, size=image_size)
            x_chw = _terramind_zscore_s2(raw_chw, model_key=model_key, mode=normalize_mode)
            x_bchw = x_chw[None, ...].astype(np.float32)

            sensor_meta = {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": tuple(_S2_SR_12_BANDS),
                "bands_terramind": tuple(_TERRAMIND_S2L2A_BANDS),
                "scale_m": scale_m,
                "cloudy_pct": cloudy_pct,
                "composite": composite,
                "fill_value": fill_value,
            }
            source = sensor_meta["collection"]
        model, wmeta, dev = _load_terramind(
            model_key=model_key,
            pretrained=pretrained,
            modality=modality,
            device=device,
        )
        tokens, tmeta = _terramind_forward_tokens(
            model,
            x_bchw,
            modality=modality,
            layer_index=layer_index,
            device=dev,
        )

        meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=backend_l,
            source=source,
            sensor=sensor_meta,
            temporal=temporal_used,
            image_size=image_size,
            input_time=temporal_midpoint_str(temporal_used),
            extra={
                "model_key": model_key,
                "modality": modality,
                "bands": tuple(_S2_SR_12_BANDS),
                "bands_terramind": tuple(_TERRAMIND_S2L2A_BANDS),
                "normalization": str(normalize_mode),
                "device": dev,
                "pretrained": bool(pretrained),
                **check_meta,
                **wmeta,
                **tmeta,
            },
        )

        if output.mode == "pooled":
            vec, cls_removed = pool_from_tokens(tokens, output.pooling)
            ometa = {
                **meta,
                "pooling": output.pooling,
                "cls_removed": bool(cls_removed),
            }
            return Embedding(data=vec.astype(np.float32), meta=ometa)

        if output.mode == "grid":
            grid, (gh, gw), cls_removed = tokens_to_grid_dhw(tokens)
            gmeta = {
                **meta,
                "grid_type": "vit_patch_tokens",
                "grid_hw": (int(gh), int(gw)),
                "grid_shape": tuple(grid.shape),
                "cls_removed": bool(cls_removed),
            }
            da = xr.DataArray(
                grid.astype(np.float32),
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(grid.shape[0]),
                    "y": np.arange(grid.shape[1]),
                    "x": np.arange(grid.shape[2]),
                },
                name="embedding",
                attrs=gmeta,
            )
            return Embedding(data=da, meta=gmeta)

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
        t = temporal_to_range(temporal)
        ss = sensor or self._default_sensor()

        scale_m = int(getattr(ss, "scale_m", 10))
        cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
        composite = str(getattr(ss, "composite", "median"))
        fill_value = float(getattr(ss, "fill_value", 0.0))

        n = len(spatials)
        prefetched_raw: list[np.ndarray | None] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray]:
            raw = _fetch_s2_sr_12_raw_chw(
                provider,
                sp,
                t,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
                fill_value=fill_value,
            )
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
                raise ModelError(f"Missing prefetched input at index={i} for terramind.")
            raw_inputs.append(raw)

        return self.get_embeddings_batch_from_inputs(
            spatials=spatials,
            input_chws=raw_inputs,
            temporal=temporal,
            sensor=ss,
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
        t = None
        fill_value = 0.0
        source = None
        sensor_meta = None
        if uses_provider:
            self._get_provider(backend)
            t = temporal_to_range(temporal)
            ss = sensor or self._default_sensor()
            scale_m = int(getattr(ss, "scale_m", 10))
            cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
            composite = str(getattr(ss, "composite", "median"))
            fill_value = float(getattr(ss, "fill_value", 0.0))
            sensor_meta = {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": tuple(_S2_SR_12_BANDS),
                "bands_terramind": tuple(_TERRAMIND_S2L2A_BANDS),
                "scale_m": scale_m,
                "cloudy_pct": cloudy_pct,
                "composite": composite,
                "fill_value": fill_value,
            }
            source = sensor_meta["collection"]
        else:
            scale_m = None
            cloudy_pct = None
            composite = None

        model_key = os.environ.get("RS_EMBED_TERRAMIND_MODEL_KEY", self.DEFAULT_MODEL_KEY).strip()
        modality = str(
            getattr(sensor, "modality", None)
            or os.environ.get("RS_EMBED_TERRAMIND_MODALITY", self.DEFAULT_MODALITY).strip()
            or self.DEFAULT_MODALITY
        )
        if modality.strip().lower().replace("-", "_") == "s2_l2a":
            modality = self.DEFAULT_MODALITY
        normalize_mode = os.environ.get("RS_EMBED_TERRAMIND_NORMALIZE", "zscore").strip()
        layer_index = int(os.environ.get("RS_EMBED_TERRAMIND_LAYER_INDEX", "-1"))
        pretrained = os.environ.get("RS_EMBED_TERRAMIND_PRETRAINED", "1").strip() not in {
            "0",
            "false",
            "False",
        }
        image_size = self.DEFAULT_IMAGE_SIZE

        x_bchw = np.stack(
            [
                _prepare_terramind_input_chw(
                    input_chw,
                    image_size=image_size,
                    model_key=model_key,
                    normalize_mode=normalize_mode,
                )
                for input_chw in input_chws
            ],
            axis=0,
        ).astype(np.float32)

        model, wmeta, dev = _load_terramind(
            model_key=model_key,
            pretrained=pretrained,
            modality=modality,
            device=device,
        )
        tokens_bnd, tmeta = _terramind_forward_tokens_batch(
            model,
            x_bchw,
            modality=modality,
            layer_index=layer_index,
            device=dev,
        )

        out: list[Embedding] = []
        for i, _spatial in enumerate(spatials):
            meta = build_meta(
                model=self.model_name,
                kind="on_the_fly",
                backend=backend_l,
                source=source,
                sensor=sensor_meta,
                temporal=t,
                image_size=image_size,
                input_time=temporal_midpoint_str(t),
                extra={
                    "model_key": model_key,
                    "modality": modality,
                    "bands": tuple(_S2_SR_12_BANDS),
                    "bands_terramind": tuple(_TERRAMIND_S2L2A_BANDS),
                    "normalization": str(normalize_mode),
                    "device": dev,
                    "pretrained": bool(pretrained),
                    "scale_m": scale_m,
                    "cloudy_pct": cloudy_pct,
                    "composite": composite,
                    "fill_value": fill_value if uses_provider else None,
                    "batch_infer": True,
                    "input_override": True,
                    **wmeta,
                    **tmeta,
                },
            )
            tokens = tokens_bnd[i]
            if output.mode == "pooled":
                vec, cls_removed = pool_from_tokens(tokens, output.pooling)
                out.append(
                    Embedding(
                        data=vec.astype(np.float32),
                        meta={
                            **meta,
                            "pooling": output.pooling,
                            "cls_removed": bool(cls_removed),
                        },
                    )
                )
                continue

            if output.mode == "grid":
                grid, (gh, gw), cls_removed = tokens_to_grid_dhw(tokens)
                gmeta = {
                    **meta,
                    "grid_type": "vit_patch_tokens",
                    "grid_hw": (int(gh), int(gw)),
                    "grid_shape": tuple(grid.shape),
                    "cls_removed": bool(cls_removed),
                }
                da = xr.DataArray(
                    grid.astype(np.float32),
                    dims=("d", "y", "x"),
                    coords={
                        "d": np.arange(grid.shape[0]),
                        "y": np.arange(grid.shape[1]),
                        "x": np.arange(grid.shape[2]),
                    },
                    name="embedding",
                    attrs=gmeta,
                )
                out.append(Embedding(data=da, meta=gmeta))
                continue

            raise ModelError(f"Unknown output mode: {output.mode}")

        return out

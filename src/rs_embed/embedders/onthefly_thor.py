# Implementation based on:
# THOR
# arXiv 2026
# https://arxiv.org/abs/2601.16011

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..providers import ProviderBase
from ._vit_mae_utils import ensure_torch, pool_from_tokens, tokens_to_grid_dhw
from .base import EmbedderBase
from .runtime_utils import (
    fetch_collection_patch_chw as _fetch_collection_patch_chw,
    is_provider_backend,
    load_cached_with_device as _load_cached_with_device,
    resolve_device_auto_torch as _resolve_device,
)
from .meta_utils import build_meta, temporal_midpoint_str, temporal_to_range


_S2_SR_10_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
_THOR_MODEL_BANDS = [
    "BLUE",
    "GREEN",
    "RED",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
    "NIR_BROAD",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]

_THOR_S2_MEAN = np.array(
    [
        0.176620,
        0.195923,
        0.213948,
        0.263378,
        0.300818,
        0.313144,
        0.308133,
        0.320993,
        0.221550,
        0.175772,
    ],
    dtype=np.float32,
)
_THOR_S2_STD = np.array(
    [
        0.264520,
        0.252949,
        0.259180,
        0.272771,
        0.248175,
        0.235432,
        0.226434,
        0.223274,
        0.171606,
        0.156223,
    ],
    dtype=np.float32,
)


def _resize_chw(x_chw: np.ndarray, *, out_hw: int) -> np.ndarray:
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW array, got {x_chw.shape}")
    x = torch.from_numpy(x_chw.astype(np.float32, copy=False)).unsqueeze(0)
    y = F.interpolate(
        x, size=(int(out_hw), int(out_hw)), mode="bilinear", align_corners=False
    )
    return y[0].detach().cpu().numpy().astype(np.float32)


def _normalize_s2_for_thor(raw_chw: np.ndarray, *, mode: str) -> np.ndarray:
    if raw_chw.ndim != 3 or int(raw_chw.shape[0]) != len(_S2_SR_10_BANDS):
        raise ModelError(
            f"Expected CHW with 10 S2 bands, got {getattr(raw_chw, 'shape', None)}"
        )

    x = np.asarray(raw_chw, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 10000.0)

    m = str(mode).lower().strip()
    if m in {"raw", "none", "off"}:
        return x.astype(np.float32)

    x = x / 10000.0
    if m in {"unit", "unit_scale", "reflectance"}:
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    if m in {"thor_stats", "zscore", "thor_zscore"}:
        std = np.maximum(_THOR_S2_STD, 1e-6)
        x = (x - _THOR_S2_MEAN[:, None, None]) / std[:, None, None]
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    raise ModelError(
        f"Unknown THOR normalization mode '{mode}'. "
        "Use one of: thor_stats, unit_scale, none."
    )


def _fetch_s2_sr_10_raw_chw(
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
        bands=tuple(_S2_SR_10_BANDS),
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        fill_value=fill_value,
    )
    return np.clip(raw, 0.0, 10000.0).astype(np.float32)


def _extract_feature_and_channel_params(
    out: Any,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    channel_params = None
    features = out
    if isinstance(out, tuple) and len(out) >= 2:
        features = out[0]
        if isinstance(out[1], dict):
            channel_params = out[1]
    if not isinstance(features, (list, tuple)) or len(features) == 0:
        raise ModelError(
            f"THOR forward expected list/tuple of features, got type={type(features)}"
        )
    feat_t = features[-1]
    return feat_t, channel_params


def _group_patch_sizes(
    *,
    channel_params: Dict[str, Any],
    groups: Dict[str, List[str]],
) -> Tuple[List[int], List[str]]:
    patch_sizes: List[int] = []
    used_groups: List[str] = []
    for gname, members in groups.items():
        member = next(
            (
                m
                for m in members
                if m in channel_params
                and isinstance(channel_params[m], dict)
                and channel_params[m].get("num_patch") is not None
            ),
            None,
        )
        if member is None:
            continue
        p = int(channel_params[member]["num_patch"])
        if p <= 0:
            continue
        patch_sizes.append(p)
        used_groups.append(str(gname))
    return patch_sizes, used_groups


def _thor_group_grid_from_tokens(
    tokens_bnd,
    *,
    channel_params: Dict[str, Any],
    groups: Dict[str, List[str]],
    merge: str,
):
    ensure_torch()
    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(tokens_bnd) or tokens_bnd.ndim != 3:
        raise ModelError(
            f"Expected THOR tokens [B,N,D], got {getattr(tokens_bnd, 'shape', None)}"
        )

    patch_sizes, used_groups = _group_patch_sizes(
        channel_params=channel_params, groups=groups
    )
    if not patch_sizes:
        raise ModelError("THOR returned no usable group patch sizes in channel_params.")

    expected_patch_tokens = int(sum(p * p for p in patch_sizes))
    n_tok = int(tokens_bnd.shape[1])
    start = 1 if n_tok == expected_patch_tokens + 1 else 0
    cls_removed = bool(start == 1)
    if n_tok < expected_patch_tokens + start:
        raise ModelError(
            f"THOR token count mismatch. got N={n_tok}, expected at least {expected_patch_tokens + start}"
        )

    patch_tokens = tokens_bnd[:, start : start + expected_patch_tokens, :]
    b, _, d = patch_tokens.shape
    max_p = max(patch_sizes)
    maps = []
    idx = 0
    for p in patch_sizes:
        pp = int(p * p)
        t = patch_tokens[:, idx : idx + pp, :]
        idx += pp
        t = t.reshape(b, p, p, d).permute(0, 3, 1, 2)  # [B,D,H,W]
        if p != max_p:
            t = F.interpolate(
                t, size=(max_p, max_p), mode="bilinear", align_corners=False
            )
        maps.append(t)

    merge_l = str(merge).lower().strip()
    if merge_l == "concat":
        grid = torch.cat(maps, dim=1)
    elif merge_l == "sum":
        grid = torch.stack(maps, dim=0).sum(dim=0)
    else:
        if merge_l not in {"mean", "avg", "average"}:
            raise ModelError(
                f"Unknown THOR group merge '{merge}'. Use mean/sum/concat."
            )
        grid = torch.stack(maps, dim=0).mean(dim=0)

    meta = {
        "expected_patch_tokens": expected_patch_tokens,
        "group_patch_sizes": tuple(int(p) for p in patch_sizes),
        "groups_used": tuple(used_groups),
        "cls_removed": cls_removed,
        "group_merge": "mean" if merge_l in {"mean", "avg", "average"} else merge_l,
    }
    return grid, meta


def _pool_thor_tokens(
    tokens: np.ndarray,
    *,
    pooling: str,
    expected_patch_tokens: Optional[int],
) -> Tuple[np.ndarray, bool]:
    if (
        expected_patch_tokens is not None
        and tokens.ndim == 2
        and int(tokens.shape[0])
        in {int(expected_patch_tokens), int(expected_patch_tokens) + 1}
    ):
        cls_removed = int(tokens.shape[0]) == int(expected_patch_tokens) + 1
        patch_tokens = tokens[1:] if cls_removed else tokens
        if pooling == "mean":
            return patch_tokens.mean(axis=0).astype(np.float32), bool(cls_removed)
        if pooling == "max":
            return patch_tokens.max(axis=0).astype(np.float32), bool(cls_removed)
        raise ModelError(f"Unknown pooling='{pooling}' (expected 'mean' or 'max').")
    return pool_from_tokens(tokens, pooling)


@lru_cache(maxsize=8)
def _load_thor_cached(
    model_key: str,
    model_bands: Tuple[str, ...],
    pretrained: bool,
    ckpt_path: Optional[str],
    ground_cover: int,
    patch_size: int,
    dev: str,
) -> Tuple[Any, Dict[str, Any]]:
    ensure_torch()
    import torch

    try:
        from terratorch.registry import BACKBONE_REGISTRY
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")).split(".")[0] == "terratorch":
            raise ModelError(
                "THOR requires terratorch. Install: pip install terratorch"
            ) from e
        raise ModelError(
            "Failed to import terratorch registry while loading THOR. "
            f"Missing dependency: {getattr(e, 'name', None) or e}. "
            "Check optional mmseg/mmengine deps or process-level shim/module conflicts."
        ) from e
    except Exception as e:
        raise ModelError(
            "Failed to import terratorch registry while loading THOR: "
            f"{type(e).__name__}: {e}"
        ) from e

    try:
        import thor_terratorch_ext  # noqa: F401
    except Exception as e:
        raise ModelError(
            "THOR extension not found. Install: pip install git+https://github.com/FM4CS/thor_terratorch_ext.git"
        ) from e

    build_kwargs: Dict[str, Any] = {
        "pretrained": bool(pretrained),
        "model_bands": list(model_bands),
        "out_indices": [-1],
        "return_channel_params": True,
        "input_params": {
            "ground_covers": [int(ground_cover)],
            "flexivit_patch_size_seqs": [int(patch_size)],
        },
    }
    if ckpt_path:
        build_kwargs["ckpt"] = os.path.expanduser(str(ckpt_path))

    try:
        model = BACKBONE_REGISTRY.build(str(model_key), **build_kwargs)
    except Exception as e:
        raise ModelError(
            f"Failed to build THOR backbone '{model_key}'. "
            "Check terratorch/thor_terratorch_ext installation and model key."
        ) from e

    try:
        model = model.to(dev).eval()
    except Exception:
        pass

    p0 = None
    for _, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError("THOR model has no parameters; cannot verify weights.")
    if not torch.isfinite(p0).all():
        raise ModelError("THOR parameters contain NaN/Inf; load likely failed.")

    p0f = p0.float()
    out_channels = getattr(model, "out_channels", None)
    embed_dim = None
    if isinstance(out_channels, (list, tuple)) and len(out_channels) > 0:
        try:
            embed_dim = int(out_channels[-1])
        except Exception:
            embed_dim = None

    meta = {
        "device": str(dev),
        "model_key": str(model_key),
        "model_bands": tuple(model_bands),
        "ground_cover_m": int(ground_cover),
        "patch_size": int(patch_size),
        "pretrained": bool(pretrained),
        "ckpt_path": os.path.expanduser(str(ckpt_path)) if ckpt_path else None,
        "embed_dim": embed_dim,
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
    }
    return model, meta


def _load_thor(
    *,
    model_key: str,
    model_bands: Tuple[str, ...],
    pretrained: bool,
    ckpt_path: Optional[str],
    ground_cover: int,
    patch_size: int,
    device: str,
) -> Tuple[Any, Dict[str, Any], str]:
    (loaded, dev) = _load_cached_with_device(
        _load_thor_cached,
        device=device,
        model_key=str(model_key),
        model_bands=tuple(model_bands),
        pretrained=bool(pretrained),
        ckpt_path=(os.path.expanduser(ckpt_path) if ckpt_path else None),
        ground_cover=int(ground_cover),
        patch_size=int(patch_size),
    )
    model, meta = loaded
    return model, meta, dev


def _thor_forward_single(
    model: Any,
    x_chw: np.ndarray,
    *,
    device: str,
    group_merge: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    ensure_torch()
    import torch

    if x_chw.ndim != 3:
        raise ModelError(
            f"Expected CHW input for THOR, got {getattr(x_chw, 'shape', None)}"
        )

    dev = _resolve_device(device)
    model = model.to(dev).eval()
    x = torch.from_numpy(x_chw.astype(np.float32, copy=False)).unsqueeze(0).to(dev)

    with torch.no_grad():
        out = model(x)

    feat_t, channel_params = _extract_feature_and_channel_params(out)
    if not torch.is_tensor(feat_t) or feat_t.ndim != 3:
        raise ModelError(
            f"THOR feature tensor must be [B,N,D], got {getattr(feat_t, 'shape', None)}"
        )
    if int(feat_t.shape[0]) != 1:
        raise ModelError(
            f"THOR embedder expects B=1 in single inference, got B={int(feat_t.shape[0])}"
        )

    tokens = feat_t[0].detach().float().cpu().numpy().astype(np.float32)
    grid: Optional[np.ndarray] = None
    expected_patch_tokens: Optional[int] = None
    grid_meta: Dict[str, Any] = {}

    groups = getattr(model, "groups", None)
    if isinstance(channel_params, dict) and isinstance(groups, dict):
        try:
            grid_bdhw, gmeta = _thor_group_grid_from_tokens(
                feat_t,
                channel_params=channel_params,
                groups=groups,
                merge=group_merge,
            )
            grid = grid_bdhw[0].detach().float().cpu().numpy().astype(np.float32)
            expected_patch_tokens = int(gmeta["expected_patch_tokens"])
            grid_meta = {
                "grid_kind": "thor_group_grid",
                "grid_group_merge": gmeta["group_merge"],
                "group_patch_sizes": gmeta["group_patch_sizes"],
                "groups_used": gmeta["groups_used"],
                "cls_removed": bool(gmeta["cls_removed"]),
                "expected_patch_tokens": expected_patch_tokens,
            }
        except Exception:
            # fall back to square-token reshape (works for simple ViT-style outputs)
            grid = None

    if grid is None:
        try:
            g, (gh, gw), cls_removed = tokens_to_grid_dhw(tokens)
            grid = g.astype(np.float32)
            grid_meta = {
                "grid_kind": "patch_tokens",
                "grid_hw": (int(gh), int(gw)),
                "cls_removed": bool(cls_removed),
            }
        except Exception:
            grid = None

    meta = {
        "tokens_shape": tuple(tokens.shape),
        "expected_patch_tokens": expected_patch_tokens,
        **grid_meta,
    }
    return tokens, grid, meta


@register("thor")
class THORBaseEmbedder(EmbedderBase):
    DEFAULT_MODEL_KEY = "thor_v1_base"
    DEFAULT_IMAGE_SIZE = 288
    DEFAULT_FETCH_WORKERS = 8

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "inputs": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": _S2_SR_10_BANDS,
            },
            "output": ["pooled", "grid"],
            "defaults": {
                "model_key": self.DEFAULT_MODEL_KEY,
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "normalization": "thor_stats",
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "group_merge": "mean",
            },
            "notes": [
                "Loads THOR backbone via terratorch + thor_terratorch_ext.",
                "Default weights come from Hugging Face FM4CS/THOR-1.0-base when pretrained=true.",
            ],
        }

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_THOR_FETCH_WORKERS",
                str(THORBaseEmbedder.DEFAULT_FETCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return SensorSpec(
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=tuple(_S2_SR_10_BANDS),
            scale_m=10,
            cloudy_pct=30,
            composite="median",
            fill_value=0.0,
        )

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("thor_1_0_base expects a provider backend (or 'auto').")

        ss = sensor or self._default_sensor()
        t = temporal_to_range(temporal)

        image_size = int(
            os.environ.get("RS_EMBED_THOR_IMG", str(self.DEFAULT_IMAGE_SIZE))
        )
        model_key = (
            os.environ.get("RS_EMBED_THOR_MODEL_KEY", self.DEFAULT_MODEL_KEY).strip()
            or self.DEFAULT_MODEL_KEY
        )
        ckpt_path = os.environ.get("RS_EMBED_THOR_CKPT")
        pretrained = os.environ.get("RS_EMBED_THOR_PRETRAINED", "1").strip() not in {
            "0",
            "false",
            "False",
        }
        normalize_mode = os.environ.get("RS_EMBED_THOR_NORMALIZE", "thor_stats").strip()
        group_merge = (
            os.environ.get("RS_EMBED_THOR_GROUP_MERGE", "mean").strip().lower()
        )
        patch_size = int(os.environ.get("RS_EMBED_THOR_PATCH_SIZE", "16"))
        ground_cover = int(round(float(getattr(ss, "scale_m", 10)) * float(image_size)))

        source = str(getattr(ss, "collection", "COPERNICUS/S2_SR_HARMONIZED"))
        scale_m = int(getattr(ss, "scale_m", 10))
        cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
        composite = str(getattr(ss, "composite", "median"))
        fill_value = float(getattr(ss, "fill_value", 0.0))

        if input_chw is None:
            raw_chw = _fetch_s2_sr_10_raw_chw(
                self._get_provider(backend),
                spatial,
                t,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
                fill_value=fill_value,
            )
        else:
            if input_chw.ndim != 3 or int(input_chw.shape[0]) != len(_S2_SR_10_BANDS):
                raise ModelError(
                    f"input_chw must be CHW with 10 bands for THOR, got {getattr(input_chw, 'shape', None)}"
                )
            raw_chw = np.asarray(input_chw, dtype=np.float32)
            raw_chw = np.clip(
                np.nan_to_num(raw_chw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 10000.0
            ).astype(np.float32)

        from ..tools.inspection import checks_should_raise, maybe_inspect_chw

        check_meta: Dict[str, Any] = {}
        report = maybe_inspect_chw(
            raw_chw,
            sensor=ss,
            name="provider_s2_sr_10_raw_chw",
            expected_channels=len(_S2_SR_10_BANDS),
            value_range=(0.0, 10000.0),
            fill_value=fill_value,
            meta=check_meta,
        )
        if (
            report is not None
            and (not report.get("ok", True))
            and checks_should_raise(ss)
        ):
            raise ModelError(
                "Provider input inspection failed: "
                + "; ".join(report.get("issues", []))
            )

        x_chw = _normalize_s2_for_thor(raw_chw, mode=normalize_mode)
        if x_chw.shape[-1] != image_size or x_chw.shape[-2] != image_size:
            x_chw = _resize_chw(x_chw, out_hw=image_size)

        model, wmeta, dev = _load_thor(
            model_key=model_key,
            model_bands=tuple(_THOR_MODEL_BANDS),
            pretrained=pretrained,
            ckpt_path=ckpt_path,
            ground_cover=ground_cover,
            patch_size=patch_size,
            device=device,
        )
        tokens, grid, fmeta = _thor_forward_single(
            model,
            x_chw,
            device=dev,
            group_merge=group_merge,
        )

        meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=str(backend).lower(),
            source=source,
            sensor={
                "collection": source,
                "bands": tuple(_S2_SR_10_BANDS),
                "bands_thor": tuple(_THOR_MODEL_BANDS),
                "scale_m": scale_m,
                "cloudy_pct": cloudy_pct,
                "composite": composite,
                "fill_value": fill_value,
            },
            temporal=t,
            image_size=image_size,
            input_time=temporal_midpoint_str(t),
            extra={
                "hf_id": "FM4CS/THOR-1.0-base",
                "normalization": normalize_mode,
                "group_merge": group_merge,
                "ground_cover_m": ground_cover,
                "patch_size": patch_size,
                **check_meta,
                **wmeta,
                **fmeta,
            },
        )

        if output.mode == "pooled":
            vec, cls_removed = _pool_thor_tokens(
                tokens,
                pooling=output.pooling,
                expected_patch_tokens=fmeta.get("expected_patch_tokens"),
            )
            out_meta = {
                **meta,
                "pooling": output.pooling,
                "cls_removed": bool(cls_removed),
            }
            return Embedding(data=vec.astype(np.float32), meta=out_meta)

        if output.mode == "grid":
            if grid is None:
                raise ModelError(
                    "THOR grid output is unavailable for this configuration. "
                    "Try pooled output, or use default model/input settings."
                )
            gmeta = {
                **meta,
                "grid_shape": tuple(grid.shape),
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
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []

        backend_l = backend.lower().strip()
        if not is_provider_backend(backend_l, allow_auto=True):
            raise ModelError("thor_1_0_base expects a provider backend (or 'auto').")

        t = temporal_to_range(temporal)
        ss = sensor or self._default_sensor()
        provider = self._get_provider(backend_l)

        scale_m = int(getattr(ss, "scale_m", 10))
        cloudy_pct = int(getattr(ss, "cloudy_pct", 30))
        composite = str(getattr(ss, "composite", "median"))
        fill_value = float(getattr(ss, "fill_value", 0.0))

        n = len(spatials)
        prefetched_raw: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
            raw = _fetch_s2_sr_10_raw_chw(
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

        out: List[Embedding] = []
        for i, sp in enumerate(spatials):
            raw = prefetched_raw[i]
            if raw is None:
                raise ModelError(
                    f"Missing prefetched input at index={i} for thor_1_0_base."
                )
            out.append(
                self.get_embedding(
                    spatial=sp,
                    temporal=temporal,
                    sensor=ss,
                    output=output,
                    backend=backend,
                    device=device,
                    input_chw=raw,
                )
            )
        return out

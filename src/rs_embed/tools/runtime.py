"""Embedder runtime primitives.

This module owns embedder lifecycle concerns: instance caching, capability
introspection, and request dispatch for single/batch embedding calls.
Provider selection/management lives in ``providers.resolution``.
Provider fetch helpers live in ``providers.fetch``.
"""

from __future__ import annotations

import inspect
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any, TypeVar

import numpy as np

from ..core import registry as _runtime_registry
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import get_embedder_cls
from ..core.specs import InputPrepSpec, OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import FetchResult
from ..core.validation import assert_supported
from ..providers import ProviderBase, get_provider, has_provider
from ..providers.fetch import fetch_sensor_patch_chw as _fetch_sensor_patch_chw
from ..providers.resolution import default_provider_backend_name
from .model_defaults import default_sensor_for_model
from .normalization import (
    _resolve_embedding_api_backend,
    normalize_backend_name,
    normalize_device_name,
    normalize_model_name,
)
from .output import normalize_embedding_output

_T = TypeVar("_T")


def resolve_device_auto_torch(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception as _e:
        return "cpu"


def load_cached_with_device(
    cached_loader: Callable[..., _T],
    *,
    device: str,
    **kwargs: Any,
) -> tuple[_T, str]:
    """Resolve device once and call a cached loader that accepts `dev=...`."""
    dev = resolve_device_auto_torch(device)
    loaded = cached_loader(dev=dev, **kwargs)
    return loaded, dev


@dataclass(frozen=True)
class _EmbeddingRequestContext:
    model_n: str
    backend_n: str
    device: str
    sensor_eff: SensorSpec | None
    model_config: dict[str, Any] | None
    input_prep: Any | None
    input_prep_resolved: Any
    input_prep_requested_mode: str
    input_prep_model_policy: str | None
    embedder: Any
    lock: Any


_IMAGE_LEVEL_VIT_GRID_MODELS = frozenset(
    {
        "satmae",
        "satmaepp",
        "scalemae",
    }
)


def _is_image_level_vit_grid_model(model_n: str) -> bool:
    return str(model_n).strip().lower() in _IMAGE_LEVEL_VIT_GRID_MODELS


def _warn_image_level_vit_tiled_grid_seam(model_l: str) -> None:
    warnings.warn(
        f"{model_l} grid output stitches image-level ViT patch tokens into a tiled "
        "mosaic, which can show seams at tile boundaries. Pass input_prep='resize' for "
        "a seamless (downsampled) grid.",
        UserWarning,
        stacklevel=4,
    )


def resolve_model_aware_input_prep(
    *,
    model_n: str,
    input_prep: Any | None,
    output: OutputSpec,
) -> tuple[Any | None, Any, str, str | None]:
    """Resolve input prep with a uniform tile default.

    Image-level ViT adapters expose patch-token grids. Like every other model,
    they tile by default so on-the-fly inputs keep native resolution. Tiled
    grids can show stitching seams at tile boundaries, but that is an inherent
    property of mosaicking independent patch-token tiles rather than a bug in a
    particular adapter, so we surface a lightweight heads-up (pointing at
    ``resize`` for a seamless, downsampled grid) instead of overriding the
    caller's prep. ``resize`` is the only seam-free path and stays silent.
    """
    from .tiling import _resolve_input_prep_spec

    resolved = _resolve_input_prep_spec(input_prep)
    requested_mode = "default" if input_prep is None else str(resolved.mode)
    model_l = str(model_n).strip().lower()
    if not _is_image_level_vit_grid_model(model_l):
        return input_prep, resolved, requested_mode, None

    if input_prep is None or str(resolved.mode) == "auto":
        # Resolve the unset/auto default to an explicit tile so this model
        # behaves like the package-wide default rather than running auto's
        # long-axis-tile/short-axis-resize heuristic.
        effective = InputPrepSpec.tile()
        effective_resolved = _resolve_input_prep_spec(effective)
        policy = "tile_default_for_image_level_vit_patch_grid"
        if output.mode == "grid":
            _warn_image_level_vit_tiled_grid_seam(model_l)
        return effective, effective_resolved, requested_mode, policy

    if str(resolved.mode) == "tile" and output.mode == "grid":
        _warn_image_level_vit_tiled_grid_seam(model_l)
        return input_prep, resolved, requested_mode, "explicit_tile_for_image_level_vit_patch_grid"

    return input_prep, resolved, requested_mode, None


@lru_cache(maxsize=64)
def describe_model_cached(model_n: str) -> dict[str, Any]:
    """Return cached ``describe()`` output for a model class (no weights loaded)."""
    cls = get_embedder_cls(model_n)
    return cls().describe()


@lru_cache(maxsize=32)
def get_embedder_bundle_cached(model: str, backend: str, device: str, sensor_k: tuple):
    """Return (embedder instance, instance lock)."""
    cls = get_embedder_cls(model)
    emb = cls()
    return emb, RLock()


def _clear_loaded_embedder_module_caches() -> int:
    """Clear ``lru_cache`` wrappers found on already-imported embedder modules."""
    cleared = 0
    seen: set[int] = set()
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith("rs_embed.embedders."):
            continue
        for obj in vars(module).values():
            cache_clear = getattr(obj, "cache_clear", None)
            if not callable(cache_clear):
                continue
            obj_id = id(obj)
            if obj_id in seen:
                continue
            cache_clear()
            seen.add(obj_id)
            cleared += 1
    return cleared


def reset_runtime() -> dict[str, int]:
    """Clear embedder runtime caches without dropping registered classes.

    This is a notebook-friendly escape hatch for recovering from stale runtime
    state after failed lazy imports or cached loader state.  It preserves any
    custom classes already registered in ``core.registry`` while clearing
    instance caches and lazy-loader bookkeeping.
    """
    import_errors_cleared = len(_runtime_registry._REGISTRY_IMPORT_ERRORS)
    _runtime_registry._REGISTRY_IMPORT_ERRORS.clear()

    get_embedder_bundle_cached.cache_clear()
    describe_model_cached.cache_clear()
    _embedder_method_accepts_parameter.cache_clear()
    embedder_accepts_input_chw.cache_clear()
    embedder_accepts_model_config.cache_clear()
    embedder_module_caches_cleared = _clear_loaded_embedder_module_caches()

    return {
        "import_errors_cleared": int(import_errors_cleared),
        "runtime_caches_cleared": 5,
        "embedder_module_caches_cleared": int(embedder_module_caches_cleared),
    }


def sensor_key(sensor: SensorSpec | None) -> tuple:
    """Build a hashable cache key from a :class:`SensorSpec`.

    Parameters
    ----------
    sensor : SensorSpec or None
        Sensor to hash. Returns a sentinel tuple when ``None``.

    Returns
    -------
    tuple
        Hashable key representing all sensor fields relevant to caching.
    """
    if sensor is None:
        return ("__none__",)
    return (
        sensor.collection,
        sensor.bands,
        sensor.scale_m,
        sensor.cloudy_pct,
        float(sensor.fill_value),
        str(sensor.composite),
        getattr(sensor, "modality", None),
        getattr(sensor, "orbit", None),
        bool(getattr(sensor, "use_float_linear", True)),
        bool(getattr(sensor, "s1_require_iw", True)),
        bool(getattr(sensor, "s1_relax_iw_on_empty", True)),
        bool(getattr(sensor, "check_input", False)),
        bool(getattr(sensor, "check_raise", True)),
        getattr(sensor, "check_save_dir", None),
    )


def _overrides_base_method(embedder: Any, method_name: str) -> bool:
    """Return True when *embedder* overrides *method_name* from EmbedderBase."""
    fn = getattr(type(embedder), method_name, None)
    if fn is None:
        return False
    from ..embedders.base import EmbedderBase

    return fn is not getattr(EmbedderBase, method_name, None)


def supports_batch_api(embedder: Any) -> bool:
    """Return True when embedder overrides EmbedderBase.get_embeddings_batch."""
    return _overrides_base_method(embedder, "get_embeddings_batch")


def supports_prefetched_batch_api(embedder: Any) -> bool:
    """Return True when embedder overrides batch-from-inputs fast path."""
    return _overrides_base_method(embedder, "get_embeddings_batch_from_inputs")


@lru_cache(maxsize=128)
def _embedder_method_accepts_parameter(
    embedder_cls: type,
    method_name: str,
    param_name: str,
) -> bool:
    fn = getattr(embedder_cls, method_name, None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except Exception as _e:
        return False
    if param_name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


@lru_cache(maxsize=128)
def embedder_accepts_input_chw(embedder_cls: type) -> bool:
    return _embedder_method_accepts_parameter(embedder_cls, "get_embedding", "input_chw")


@lru_cache(maxsize=256)
def embedder_accepts_model_config(
    embedder_cls: type,
    method_name: str = "get_embedding",
) -> bool:
    return _embedder_method_accepts_parameter(embedder_cls, method_name, "model_config")


def _display_model_name(embedder: Any) -> str:
    return str(getattr(embedder, "model_name", type(embedder).__name__))


def require_model_config_support(
    *,
    embedder: Any,
    model_config: dict[str, Any] | None,
    method_name: str = "get_embedding",
) -> None:
    if model_config is None:
        return
    if embedder_accepts_model_config(type(embedder), method_name):
        return
    keys = sorted(str(k) for k in model_config.keys())
    raise ModelError(
        f"Model {_display_model_name(embedder)} does not accept model-specific"
        f" keyword arguments for {method_name}(); got keys {keys}."
    )


def call_embedder_get_embedding(
    *,
    embedder: Any,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    sensor: SensorSpec | None,
    output: OutputSpec,
    backend: str,
    device: str,
    input_chw: np.ndarray | None = None,
    model_config: dict[str, Any] | None = None,
    fetch_meta: dict[str, Any] | None = None,
) -> Embedding:
    require_model_config_support(
        embedder=embedder,
        model_config=model_config,
        method_name="get_embedding",
    )
    kwargs: dict[str, Any] = {
        "spatial": spatial,
        "temporal": temporal,
        "sensor": sensor,
        "model_config": model_config,
        "output": output,
        "backend": backend,
        "device": device,
    }
    if model_config is None:
        kwargs.pop("model_config", None)
    if input_chw is not None and embedder_accepts_input_chw(type(embedder)):
        kwargs["input_chw"] = input_chw
    if fetch_meta is not None and _embedder_method_accepts_parameter(
        type(embedder),
        "get_embedding",
        "fetch_meta",
    ):
        kwargs["fetch_meta"] = fetch_meta
    out = embedder.get_embedding(**kwargs)
    return normalize_embedding_output(emb=out, output=output)


def run_with_retry(
    fn: Callable[[], _T],
    *,
    retries: int = 0,
    backoff_s: float = 0.0,
) -> _T:
    """Run a callable with bounded retries and optional exponential backoff."""
    tries = max(0, int(retries))
    backoff = max(0.0, float(backoff_s))
    for attempt in range(tries + 1):
        try:
            return fn()
        except Exception:  # pragma: no cover - exercised by call-sites
            if attempt >= tries:
                raise
            if backoff > 0:
                time.sleep(backoff * (2**attempt))
    # Loop always returns on success or raises on last attempt; this is unreachable.
    raise AssertionError("unreachable")


def _create_default_gee_provider() -> ProviderBase:
    return get_provider("gee", auto_auth=True)


def provider_factory_for_backend(
    backend: str,
) -> Callable[[], ProviderBase] | None:
    b = normalize_backend_name(backend)
    if b == "auto":
        b = default_provider_backend_name() or "gee"
    if not has_provider(b):
        return None
    if b == "gee":
        return _create_default_gee_provider
    return lambda: get_provider(b)


def _prepare_embedding_request_context(
    *,
    model: str,
    temporal: TemporalSpec | None,
    sensor: SensorSpec | None,
    model_config: dict[str, Any] | None,
    output: OutputSpec,
    backend: str,
    device: str,
    input_prep: Any | None,
) -> _EmbeddingRequestContext:
    model_n = normalize_model_name(model)
    backend_n = _resolve_embedding_api_backend(model_n, normalize_backend_name(backend))
    device_n = normalize_device_name(device)
    (
        input_prep_eff,
        input_prep_resolved,
        input_prep_requested_mode,
        input_prep_model_policy,
    ) = resolve_model_aware_input_prep(
        model_n=model_n,
        input_prep=input_prep,
        output=output,
    )

    sensor_eff = sensor
    if input_prep_resolved.mode == "tile" and sensor_eff is None:
        sensor_eff = default_sensor_for_model(model_n)

    sensor_k = sensor_key(sensor_eff)
    embedder, lock = get_embedder_bundle_cached(model_n, backend_n, device_n, sensor_k)
    assert_supported(embedder, backend=backend_n, output=output, temporal=temporal)

    return _EmbeddingRequestContext(
        model_n=model_n,
        backend_n=backend_n,
        device=device_n,
        sensor_eff=sensor_eff,
        model_config=model_config,
        input_prep=input_prep_eff,
        input_prep_resolved=input_prep_resolved,
        input_prep_requested_mode=input_prep_requested_mode,
        input_prep_model_policy=input_prep_model_policy,
        embedder=embedder,
        lock=lock,
    )


def _annotate_image_level_vit_grid_embedding(
    *,
    emb: Embedding,
    ctx: _EmbeddingRequestContext,
    output: OutputSpec,
) -> Embedding:
    if not _is_image_level_vit_grid_model(ctx.model_n):
        return emb
    meta = getattr(emb, "meta", None)
    if not isinstance(meta, dict):
        return emb
    from .tiling import INPUT_PREP_VERSION

    prep = meta.setdefault(
        "input_prep",
        {
            "prep_version": INPUT_PREP_VERSION,
            "requested_mode": str(ctx.input_prep_requested_mode),
            "resolved_mode": str(getattr(ctx.input_prep_resolved, "mode", "tile")),
        },
    )
    if isinstance(prep, dict):
        prep.setdefault("prep_version", INPUT_PREP_VERSION)
        prep.setdefault("requested_mode", str(ctx.input_prep_requested_mode))
        prep.setdefault("resolved_mode", str(getattr(ctx.input_prep_resolved, "mode", "tile")))
        if ctx.input_prep_model_policy is not None:
            prep["model_policy"] = str(ctx.input_prep_model_policy)
            prep["resolved_by_model_policy"] = (
                str(ctx.input_prep_model_policy) == "tile_default_for_image_level_vit_patch_grid"
            )
        if output.mode == "grid":
            prep["tiled_grid_seam_risk"] = "high"
            prep["tiled_grid_recommended"] = False

    if output.mode == "grid":
        meta.setdefault("grid_semantics", "vit_patch_tokens")
        meta.setdefault("grid_tile_recommended", False)
        meta.setdefault("preferred_output", "pooled")
    return emb


def _annotate_embedding_list(
    *,
    embs: list[Embedding],
    ctx: _EmbeddingRequestContext,
    output: OutputSpec,
) -> list[Embedding]:
    return [
        _annotate_image_level_vit_grid_embedding(emb=emb, ctx=ctx, output=output) for emb in embs
    ]


def fetch_api_side_inputs(
    *,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None,
    backend_n: str,
    sensor_eff: SensorSpec | None,
    input_prep_resolved: Any,
    embedder: Any,
    model_n: str,
    model_config: dict[str, Any] | None = None,
) -> list[FetchResult] | None:
    """Prefetch provider inputs for API-side input prep.

    Returns a list of ``FetchResult`` (one per spatial) when API-side
    input prep is active and a provider backend is available, or ``None``
    to let the embedder fetch internally.
    """
    mode = str(getattr(input_prep_resolved, "mode", "resize")).strip().lower()
    use_api_side_input_prep = mode in {"tile", "auto"}
    if not use_api_side_input_prep:
        return None

    # Precomputed models manage their own fetch/tiling internally.
    if getattr(type(embedder), "_is_precomputed", False):
        return None

    # API-side tiling needs a provider backend, a sensor, and an embedder that
    # accepts prefetched CHW input. When any prerequisite is missing we degrade
    # gracefully (let the embedder fetch internally and resize) rather than
    # raising — ``tile`` is the package default, so it must never hard-fail basic
    # usage. The embedding's meta["input_prep"] records the resolved mode so the
    # fallback stays auditable. This applies to both ``tile`` and ``auto``.
    factory = provider_factory_for_backend(backend_n)
    if factory is None:
        return None
    if sensor_eff is None:
        return None
    if not embedder_accepts_input_chw(type(embedder)):
        return None

    provider = factory()
    ensure_ready = getattr(provider, "ensure_ready", None)
    if callable(ensure_ready):
        run_with_retry(lambda: ensure_ready(), retries=0, backoff_s=0.0)

    # Forward fetch-affecting model_config (e.g. temporal_mode) so the prefetch
    # path fetches the SAME input as the direct get_embedding path. Without this
    # the prefetch used fetch_input's defaults and silently ignored the user's
    # config (e.g. temporal_mode="single" became multi via the env/auto default).
    fetch_extra: dict[str, Any] = {}
    if model_config and _embedder_method_accepts_parameter(
        type(embedder), "fetch_input", "temporal_mode"
    ):
        tm = model_config.get("temporal_mode")
        if tm is not None:
            fetch_extra["temporal_mode"] = tm

    # Use the embedder's fetch_input() when available; fall back to generic.
    results: list[FetchResult] = []
    for idx, spatial in enumerate(spatials):
        try:
            fr = embedder.fetch_input(
                provider,
                spatial=spatial,
                temporal=temporal,
                sensor=sensor_eff,
                **fetch_extra,
            )
            if fr is not None:
                results.append(fr)
            else:
                raw = _fetch_sensor_patch_chw(
                    provider,
                    spatial=spatial,
                    temporal=temporal,
                    sensor=sensor_eff,
                )
                results.append(FetchResult(data=raw, meta={}))
        except Exception as exc:
            raise ModelError(
                f"Failed to fetch API-side input for spatial[{idx}] ({spatial}): {exc}"
            ) from exc
    return results


# Transport-only fetch_meta keys consumed by the embedder (a crop target etc.),
# not meant to surface in the output embedding's metadata.
_INTERNAL_FETCH_META_KEYS = frozenset({"roi_window_geo"})


def stamp_prefetch_fetch_meta(emb: Embedding, fetch_meta: dict[str, Any] | None) -> Embedding:
    """Preserve fetch_input meta on the prefetch path so it is never silently lost.

    On the direct ``get_embedding`` path the embedder keeps its own fetch meta; on
    the API prefetch path the input is handed back as ``input_chw`` and that meta
    would otherwise vanish. This stamps it (diagnostic only) under
    ``meta['prefetch_fetch_meta']`` without overwriting anything the embedder set,
    so every model behaves the same regardless of which path ran. Transport-only
    keys (e.g. the fetch-square crop window) are excluded.
    """
    if not fetch_meta or not isinstance(getattr(emb, "meta", None), dict):
        return emb
    diag = {k: v for k, v in fetch_meta.items() if k not in _INTERNAL_FETCH_META_KEYS}
    if diag:
        emb.meta.setdefault("prefetch_fetch_meta", diag)
    return emb


def run_embedding_request(
    *,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None,
    sensor: SensorSpec | None,
    output: OutputSpec,
    ctx: _EmbeddingRequestContext,
) -> list[Embedding]:
    from .tiling import _call_embedder_get_embedding_with_input_prep

    prefetched_inputs = fetch_api_side_inputs(
        spatials=spatials,
        temporal=temporal,
        backend_n=ctx.backend_n,
        sensor_eff=ctx.sensor_eff,
        input_prep_resolved=ctx.input_prep_resolved,
        embedder=ctx.embedder,
        model_n=ctx.model_n,
        model_config=ctx.model_config,
    )
    if prefetched_inputs is not None:
        out: list[Embedding] = []
        for spatial, fr in zip(spatials, prefetched_inputs, strict=True):
            with ctx.lock:
                emb = _call_embedder_get_embedding_with_input_prep(
                    embedder=ctx.embedder,
                    spatial=spatial,
                    temporal=temporal,
                    sensor=ctx.sensor_eff,
                    model_config=ctx.model_config,
                    output=output,
                    backend=ctx.backend_n,
                    device=ctx.device,
                    input_chw=fr.data,
                    input_prep=ctx.input_prep,
                    fetch_meta=fr.meta if fr.meta else None,
                )
            out.append(stamp_prefetch_fetch_meta(emb, fr.meta))
        return _annotate_embedding_list(embs=out, ctx=ctx, output=output)

    if len(spatials) == 1:
        with ctx.lock:
            emb = call_embedder_get_embedding(
                embedder=ctx.embedder,
                spatial=spatials[0],
                temporal=temporal,
                sensor=sensor,
                model_config=ctx.model_config,
                output=output,
                backend=ctx.backend_n,
                device=ctx.device,
            )
        return _annotate_embedding_list(embs=[emb], ctx=ctx, output=output)

    if ctx.model_config is not None and not embedder_accepts_model_config(
        type(ctx.embedder),
        "get_embeddings_batch",
    ):
        out: list[Embedding] = []
        for spatial in spatials:
            with ctx.lock:
                emb = call_embedder_get_embedding(
                    embedder=ctx.embedder,
                    spatial=spatial,
                    temporal=temporal,
                    sensor=sensor,
                    model_config=ctx.model_config,
                    output=output,
                    backend=ctx.backend_n,
                    device=ctx.device,
                )
            out.append(emb)
        return _annotate_embedding_list(embs=out, ctx=ctx, output=output)

    with ctx.lock:
        kwargs: dict[str, Any] = {
            "spatials": spatials,
            "temporal": temporal,
            "sensor": sensor,
            "output": output,
            "backend": ctx.backend_n,
            "device": ctx.device,
        }
        if ctx.model_config is not None and embedder_accepts_model_config(
            type(ctx.embedder),
            "get_embeddings_batch",
        ):
            kwargs["model_config"] = ctx.model_config
        embs = ctx.embedder.get_embeddings_batch(**kwargs)
    return _annotate_embedding_list(
        embs=[normalize_embedding_output(emb=emb, output=output) for emb in embs],
        ctx=ctx,
        output=output,
    )

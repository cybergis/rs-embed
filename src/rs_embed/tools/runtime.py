"""Embedder runtime primitives.

This module owns embedder lifecycle concerns: instance caching, capability
introspection, and request dispatch for single/batch embedding calls.
Provider selection/fetch helpers live in ``embedders.runtime_utils``.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import get_embedder_cls
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.validation import assert_supported
from ..embedders.runtime_utils import default_provider_backend_name
from ..providers import ProviderBase, get_provider, has_provider
from ..providers.gee_utils import fetch_gee_patch_raw
from .model_defaults import default_sensor_for_model
from .normalization import (
    _resolve_embedding_api_backend,
    normalize_backend_name,
    normalize_device_name,
    normalize_model_name,
)
from .output import normalize_embedding_output

_T = TypeVar("_T")


@dataclass(frozen=True)
class _EmbeddingRequestContext:
    model_n: str
    backend_n: str
    device: str
    sensor_eff: Optional[SensorSpec]
    input_prep: Optional[Any]
    input_prep_resolved: Any
    embedder: Any
    lock: Any


@lru_cache(maxsize=32)
def get_embedder_bundle_cached(model: str, backend: str, device: str, sensor_k: Tuple):
    """Return (embedder instance, instance lock)."""
    cls = get_embedder_cls(model)
    emb = cls()
    return emb, RLock()


def sensor_key(sensor: Optional[SensorSpec]) -> Tuple:
    if sensor is None:
        return ("__none__",)
    return (
        sensor.collection,
        sensor.bands,
        int(sensor.scale_m),
        int(sensor.cloudy_pct),
        float(sensor.fill_value),
        str(sensor.composite),
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
def embedder_accepts_input_chw(embedder_cls: type) -> bool:
    fn = getattr(embedder_cls, "get_embedding", None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    if "input_chw" in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def call_embedder_get_embedding(
    *,
    embedder: Any,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    input_chw: Optional[np.ndarray] = None,
) -> Embedding:
    kwargs: Dict[str, Any] = {
        "spatial": spatial,
        "temporal": temporal,
        "sensor": sensor,
        "output": output,
        "backend": backend,
        "device": device,
    }
    if input_chw is not None and embedder_accepts_input_chw(type(embedder)):
        kwargs["input_chw"] = input_chw
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
    last_err: Optional[Exception] = None
    for attempt in range(tries + 1):
        try:
            return fn()
        except Exception as e:  # pragma: no cover - exercised by call-sites
            last_err = e
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
) -> Optional[Callable[[], ProviderBase]]:
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
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    input_prep: Optional[Any],
) -> _EmbeddingRequestContext:
    from .tiling import _resolve_input_prep_spec

    model_n = normalize_model_name(model)
    backend_n = _resolve_embedding_api_backend(
        model_n, normalize_backend_name(backend)
    )
    device_n = normalize_device_name(device)
    input_prep_resolved = _resolve_input_prep_spec(input_prep)

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
        input_prep=input_prep,
        input_prep_resolved=input_prep_resolved,
        embedder=embedder,
        lock=lock,
    )


def fetch_api_side_inputs(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    backend_n: str,
    sensor_eff: Optional[SensorSpec],
    input_prep_resolved: Any,
    embedder: Any,
    model_n: str,
) -> Optional[List[np.ndarray]]:
    mode = str(getattr(input_prep_resolved, "mode", "resize")).strip().lower()
    use_api_side_input_prep = mode in {"tile", "auto"}
    if not use_api_side_input_prep:
        return None

    factory = provider_factory_for_backend(backend_n)
    if factory is None:
        if mode == "tile":
            raise ModelError(
                "input_prep.mode='tile' currently requires a provider backend (e.g. gee)."
            )
        return None
    if sensor_eff is None:
        if mode == "tile":
            raise ModelError(
                "input_prep.mode='tile' requires a sensor for provider-backed on-the-fly models."
            )
        return None
    if not embedder_accepts_input_chw(type(embedder)):
        if mode == "tile":
            raise ModelError(
                f"Model {model_n} does not accept input_chw; cannot apply input_prep.mode='tile'."
            )
        return None

    provider = factory()
    ensure_ready = getattr(provider, "ensure_ready", None)
    if callable(ensure_ready):
        run_with_retry(lambda: ensure_ready(), retries=0, backoff_s=0.0)
    return [
        fetch_gee_patch_raw(
            provider,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor_eff,
        )
        for spatial in spatials
    ]


def run_embedding_request(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    ctx: _EmbeddingRequestContext,
) -> List[Embedding]:
    from .tiling import _call_embedder_get_embedding_with_input_prep

    prefetched_inputs = fetch_api_side_inputs(
        spatials=spatials,
        temporal=temporal,
        backend_n=ctx.backend_n,
        sensor_eff=ctx.sensor_eff,
        input_prep_resolved=ctx.input_prep_resolved,
        embedder=ctx.embedder,
        model_n=ctx.model_n,
    )
    if prefetched_inputs is not None:
        out: List[Embedding] = []
        for spatial, raw in zip(spatials, prefetched_inputs):
            with ctx.lock:
                emb = _call_embedder_get_embedding_with_input_prep(
                    embedder=ctx.embedder,
                    spatial=spatial,
                    temporal=temporal,
                    sensor=ctx.sensor_eff,
                    output=output,
                    backend=ctx.backend_n,
                    device=ctx.device,
                    input_chw=raw,
                    input_prep=ctx.input_prep,
                )
            out.append(emb)
        return out

    if len(spatials) == 1:
        with ctx.lock:
            emb = call_embedder_get_embedding(
                embedder=ctx.embedder,
                spatial=spatials[0],
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=ctx.backend_n,
                device=ctx.device,
            )
        return [emb]

    with ctx.lock:
        embs = ctx.embedder.get_embeddings_batch(
            spatials=spatials,
            temporal=temporal,
            sensor=sensor,
            output=output,
            backend=ctx.backend_n,
            device=ctx.device,
        )
    return [normalize_embedding_output(emb=emb, output=output) for emb in embs]

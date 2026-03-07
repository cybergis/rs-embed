"""Public API for rs-embed: get_embedding, get_embeddings_batch, export_batch.

Orchestration flow
------------------
1. **Validation** – _validate_spatials checks spatial/temporal/output specs.
2. **Context**    – _prepare_embedding_request_context resolves model, backend,
   device, sensor, input-prep and returns a frozen _EmbeddingRequestContext.
3. **Execution**  – _run_embedding_request (single/batch) or BatchExporter.run()
   (chunked export with prefetch → infer → write pipeline).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .providers.gee_utils import (
    fetch_gee_patch_raw as _fetch_gee_patch_raw,
)
from .tools.normalization import (
    normalize_backend_name as _normalize_backend_name,
    normalize_device_name as _normalize_device_name,
    normalize_model_name as _normalize_model_name,
)
from .tools.checkpoint_utils import (
    is_incomplete_combined_manifest as _is_incomplete_combined_manifest,
)
from .tools.runtime import (
    embedder_accepts_input_chw as _embedder_accepts_input_chw,
    get_embedder_bundle_cached as _get_embedder_bundle_cached,
    run_with_retry as _run_with_retry,
    sensor_key as _sensor_key,
)
from .core.validation import (
    assert_supported as _assert_supported,
    validate_specs as _validate_specs,
)
from .tools.manifest import (
    combined_resume_manifest as _combined_resume_manifest,
    load_json_dict as _load_json_dict,
)
from .tools.model_defaults import (
    default_sensor_for_model as _default_sensor_for_model,
)
from .tools.output import (
    normalize_embedding_output as _normalize_embedding_output,
)
from .core.embedding import Embedding
from .core.errors import ModelError
from .core.registry import get_embedder_cls
from .core.specs import (
    InputPrepSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from .core.types import ExportConfig, ExportLayout, ExportTarget, ModelConfig
from .embedders.catalog import MODEL_ALIASES, MODEL_SPECS
from .providers import ProviderBase, get_provider, has_provider, list_providers

# Backward-compatibility hook: tests/downstream may monkeypatch api.GEEProvider.
GEEProvider: Optional[Callable[..., ProviderBase]] = None


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Internal: provider helpers
# -----------------------------------------------------------------------------


def _create_default_gee_provider() -> ProviderBase:
    # If tests/downstream code set api.GEEProvider, use it directly.
    cls = GEEProvider
    if cls is None:
        try:
            from .providers import GEEProvider as _GEEProvider  # type: ignore

            cls = _GEEProvider
        except Exception:
            pass

    if cls is not None:
        try:
            return cls(auto_auth=True)
        except TypeError:
            return cls()

    return get_provider("gee", auto_auth=True)


def _provider_factory_for_backend(backend: str) -> Optional[Callable[[], ProviderBase]]:
    b = _normalize_backend_name(backend)
    if b == "auto":
        b = _default_provider_backend_for_api()
    if not has_provider(b):
        return None
    if b == "gee":
        return _create_default_gee_provider
    return lambda: get_provider(b)


def _probe_model_describe(model_n: str) -> Dict[str, Any]:
    """Best-effort model describe() probe used for API-level routing decisions."""
    try:
        cls = get_embedder_cls(model_n)
        emb = cls()
        desc = emb.describe() or {}
        return desc if isinstance(desc, dict) else {}
    except Exception:
        return {}


def _default_provider_backend_for_api() -> str:
    providers = [str(p).strip().lower() for p in list_providers()]
    if "gee" in providers:
        return "gee"
    if providers:
        return providers[0]
    return "gee"


def _resolve_embedding_api_backend(model_n: str, backend_n: str) -> str:
    """Normalize backend semantics for precomputed models."""
    desc = _probe_model_describe(model_n)
    if str(desc.get("type", "")).strip().lower() != "precomputed":
        return backend_n

    backends = desc.get("backend")
    if not isinstance(backends, list):
        return backend_n
    allowed = [str(b).strip().lower() for b in backends if str(b).strip()]
    if not allowed:
        return backend_n

    provider_allowed = ("provider" in allowed) or ("gee" in allowed)
    if backend_n in allowed:
        if backend_n == "auto" and provider_allowed:
            return _default_provider_backend_for_api()
        return backend_n
    if backend_n == "local" and "auto" in allowed and not provider_allowed:
        return "auto"
    if has_provider(backend_n) and provider_allowed:
        return backend_n

    if backend_n in {"gee", "auto"}:
        if "auto" in allowed:
            return "auto"
        if "local" in allowed:
            return "local"
        if provider_allowed:
            return _default_provider_backend_for_api()

    return backend_n


# ---------------------------------------------------------------------------
# Internal: export target resolution
# ---------------------------------------------------------------------------


def _normalize_export_layout(layout: str) -> ExportLayout:
    layout_n = str(layout).strip().lower().replace("-", "_")
    if layout_n in {"combined", "single_file", "file"}:
        return ExportLayout.COMBINED
    if layout_n in {"per_item", "dir", "directory"}:
        return ExportLayout.PER_ITEM
    raise ModelError(
        f"Unsupported export layout: {layout!r}. Supported: 'combined', 'per_item'."
    )


def _resolve_export_batch_target(
    *,
    n_spatials: int,
    ext: str,
    out: Optional[str],
    layout: Optional[str],
    out_dir: Optional[str],
    out_path: Optional[str],
    names: Optional[List[str]],
) -> ExportTarget:
    if (out is not None) or (layout is not None):
        if out is None or layout is None:
            raise ModelError(
                "Provide both out and layout when using the decoupled output API."
            )
        if out_dir is not None or out_path is not None:
            raise ModelError("Use either out+layout or out_dir/out_path, not both.")
        layout_enum = _normalize_export_layout(layout)
        if layout_enum == ExportLayout.COMBINED:
            out_path = out
        else:
            out_dir = out

    if out_dir is None and out_path is None:
        raise ModelError("export_batch requires out_dir or out_path.")
    if out_dir is not None and out_path is not None:
        raise ModelError("Provide only one of out_dir or out_path.")

    if out_path is not None:
        out_file = out_path if out_path.endswith(ext) else (out_path + ext)
        return ExportTarget(layout=ExportLayout.COMBINED, out_file=out_file)

    assert out_dir is not None
    point_names = (
        names if names is not None else [f"p{i:05d}" for i in range(n_spatials)]
    )
    if len(point_names) != n_spatials:
        raise ModelError("names must have the same length as spatials.")
    return ExportTarget(layout=ExportLayout.PER_ITEM, out_dir=out_dir, names=point_names)


# ---------------------------------------------------------------------------
# Tiling / input-prep
# ---------------------------------------------------------------------------
from .tools.tiling import (  # noqa: E402
    _ResolvedInputPrepSpec,
    _resolve_input_prep_spec,
    _call_embedder_get_embedding_with_input_prep,
    _tile_yx_starts,
)


# ---------------------------------------------------------------------------
# Internal: embedding request helpers (for get_embedding / get_embeddings_batch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EmbeddingRequestContext:
    model_n: str
    backend_n: str
    device: str
    sensor_eff: Optional[SensorSpec]
    input_prep: Optional[InputPrepSpec | str]
    input_prep_resolved: _ResolvedInputPrepSpec
    embedder: Any
    lock: Any


def _validate_spatials(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
) -> None:
    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty List[SpatialSpec].")
    for spatial in spatials:
        _validate_specs(spatial=spatial, temporal=temporal, output=output)


# Contract: normalizes all user-facing strings (model, backend, device, input_prep,
# sensor) and returns a fully-resolved, frozen _EmbeddingRequestContext.  After this
# call, all downstream code can assume canonical names and a ready embedder instance.
def _prepare_embedding_request_context(
    *,
    model: str,
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    input_prep: Optional[InputPrepSpec | str],
) -> _EmbeddingRequestContext:
    model_n = _normalize_model_name(model)
    backend_n = _resolve_embedding_api_backend(
        model_n, _normalize_backend_name(backend)
    )
    device_n = _normalize_device_name(device)
    input_prep_resolved = _resolve_input_prep_spec(input_prep)

    sensor_eff = sensor
    if input_prep_resolved.mode == "tile" and sensor_eff is None:
        sensor_eff = _default_sensor_for_model(model_n)

    sensor_k = _sensor_key(sensor_eff)
    embedder, lock = _get_embedder_bundle_cached(model_n, backend_n, device_n, sensor_k)
    _assert_supported(embedder, backend=backend_n, output=output, temporal=temporal)

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


def _maybe_fetch_api_side_inputs(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    ctx: _EmbeddingRequestContext,
) -> Optional[List[np.ndarray]]:
    use_api_side_input_prep = (ctx.input_prep is not None) and (
        ctx.input_prep_resolved.mode in {"tile", "auto"}
    )
    if not use_api_side_input_prep:
        return None

    provider_factory = _provider_factory_for_backend(ctx.backend_n)
    if provider_factory is None:
        if ctx.input_prep_resolved.mode == "tile":
            raise ModelError(
                "input_prep.mode='tile' currently requires a provider backend (e.g. gee)."
            )
        return None
    if ctx.sensor_eff is None:
        if ctx.input_prep_resolved.mode == "tile":
            raise ModelError(
                "input_prep.mode='tile' requires a sensor for provider-backed on-the-fly models."
            )
        return None
    if not _embedder_accepts_input_chw(type(ctx.embedder)):
        if ctx.input_prep_resolved.mode == "tile":
            raise ModelError(
                f"Model {ctx.model_n} does not accept input_chw; cannot apply input_prep.mode='tile'."
            )
        return None

    provider = provider_factory()
    ensure_ready = getattr(provider, "ensure_ready", None)
    if callable(ensure_ready):
        _run_with_retry(lambda: ensure_ready(), retries=0, backoff_s=0.0)
    return [
        _fetch_gee_patch_raw(
            provider, spatial=spatial, temporal=temporal, sensor=ctx.sensor_eff
        )
        for spatial in spatials
    ]


# Contract: given a resolved context + spatials, returns one Embedding per spatial.
# May prefetch inputs API-side (tile/auto) or delegate directly to the embedder.
# Output embeddings are already normalize_embedding_output'd — do NOT normalize again.
def _run_embedding_request(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    ctx: _EmbeddingRequestContext,
) -> List[Embedding]:
    prefetched_inputs = _maybe_fetch_api_side_inputs(
        spatials=spatials, temporal=temporal, ctx=ctx
    )
    if prefetched_inputs is not None:
        out: List[Embedding] = []
        for spatial, raw in zip(spatials, prefetched_inputs):
            with ctx.lock:
                # _call_embedder_get_embedding_with_input_prep already applies
                # normalize_embedding_output internally (via _call_embedder_get_embedding).
                # Do NOT normalize again here — double-normalization corrupts
                # grid_orientation_applied metadata for south-to-north models.
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
            emb = ctx.embedder.get_embedding(
                spatial=spatials[0],
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=ctx.backend_n,
                device=ctx.device,
            )
        return [_normalize_embedding_output(emb=emb, output=output)]

    with ctx.lock:
        embs = ctx.embedder.get_embeddings_batch(
            spatials=spatials,
            temporal=temporal,
            sensor=sensor,
            output=output,
            backend=ctx.backend_n,
            device=ctx.device,
        )
    return [_normalize_embedding_output(emb=emb, output=output) for emb in embs]


# -----------------------------------------------------------------------------
# Public: embeddings
# -----------------------------------------------------------------------------


def list_models(*, include_aliases: bool = False) -> List[str]:
    """Return the stable model catalog, independent of runtime lazy-load state."""
    model_ids = set(MODEL_SPECS.keys())
    if include_aliases:
        model_ids.update(MODEL_ALIASES.keys())
    return sorted(model_ids)


def get_embedding(
    model: str,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: Optional[InputPrepSpec | str] = "resize",
) -> Embedding:
    """Compute a single embedding.

    Notes
    -----
    This function reuses a cached embedder instance when possible to avoid
    repeatedly loading model weights / initializing providers.
    """
    _validate_spatials(spatials=[spatial], temporal=temporal, output=output)
    ctx = _prepare_embedding_request_context(
        model=model,
        temporal=temporal,
        sensor=sensor,
        output=output,
        backend=backend,
        device=device,
        input_prep=input_prep,
    )
    return _run_embedding_request(
        spatials=[spatial],
        temporal=temporal,
        sensor=sensor,
        output=output,
        ctx=ctx,
    )[0]


def get_embeddings_batch(
    model: str,
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: Optional[InputPrepSpec | str] = "resize",
) -> List[Embedding]:
    """Compute embeddings for multiple SpatialSpecs using a shared embedder instance."""
    _validate_spatials(spatials=spatials, temporal=temporal, output=output)
    ctx = _prepare_embedding_request_context(
        model=model,
        temporal=temporal,
        sensor=sensor,
        output=output,
        backend=backend,
        device=device,
        input_prep=input_prep,
    )
    return _run_embedding_request(
        spatials=spatials,
        temporal=temporal,
        sensor=sensor,
        output=output,
        ctx=ctx,
    )


# -----------------------------------------------------------------------------
# Public: batch export (core)
# -----------------------------------------------------------------------------


def export_batch(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    models: List[str],
    out: Optional[str] = None,
    layout: Optional[str] = None,
    out_dir: Optional[str] = None,
    out_path: Optional[str] = None,
    names: Optional[List[str]] = None,
    backend: str = "auto",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: Optional[SensorSpec] = None,
    per_model_sensors: Optional[Dict[str, SensorSpec]] = None,
    format: str = "npz",
    save_inputs: bool = True,
    save_embeddings: bool = True,
    save_manifest: bool = True,
    fail_on_bad_input: bool = False,
    chunk_size: int = 16,
    infer_batch_size: Optional[int] = None,
    num_workers: int = 8,
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    async_write: bool = True,
    writer_workers: int = 2,
    resume: bool = False,
    show_progress: bool = True,
    input_prep: Optional[InputPrepSpec | str] = "resize",
) -> Any:
    """Export inputs + embeddings for many spatials and many models.

    This is the recommended high-level entrypoint for batch export.
    Delegates to :class:`~rs_embed.pipelines.exporter.BatchExporter`.
    """
    from .pipelines.exporter import BatchExporter

    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty List[SpatialSpec].")
    if not isinstance(models, list) or len(models) == 0:
        raise ModelError("models must be a non-empty List[str].")

    backend_n = _normalize_backend_name(backend)
    device_n = _normalize_device_name(device)
    fmt = format.lower().strip()

    from .writers import SUPPORTED_FORMATS, get_extension

    if fmt not in SUPPORTED_FORMATS:
        raise ModelError(
            f"Unsupported export format: {format!r}. Supported: {SUPPORTED_FORMATS}."
        )
    ext = get_extension(fmt)

    target = _resolve_export_batch_target(
        n_spatials=len(spatials),
        ext=ext,
        out=out,
        layout=layout,
        out_dir=out_dir,
        out_path=out_path,
        names=names,
    )

    _validate_spatials(spatials=spatials, temporal=temporal, output=output)

    # Early resume check for combined layout
    if target.layout == ExportLayout.COMBINED and bool(resume) and target.out_file:
        if os.path.exists(target.out_file):
            json_path = os.path.splitext(target.out_file)[0] + ".json"
            resume_manifest = _load_json_dict(json_path)
            if not _is_incomplete_combined_manifest(resume_manifest):
                return _combined_resume_manifest(
                    spatials=spatials,
                    temporal=temporal,
                    output=output,
                    backend=backend_n,
                    device=device_n,
                    out_file=target.out_file,
                )

    per_model_sensors = per_model_sensors or {}

    # Resolve per-model config
    model_configs: List[ModelConfig] = []
    resolved_backend: Dict[str, str] = {}
    for m in models:
        m_n = _normalize_model_name(m)
        eff_backend = _resolve_embedding_api_backend(m_n, backend_n)
        resolved_backend[m] = eff_backend
        cls = get_embedder_cls(m_n)
        try:
            emb_check = cls()
            _assert_supported(
                emb_check, backend=eff_backend, output=output, temporal=temporal
            )
            desc = emb_check.describe() or {}
        except ModelError:
            raise
        except Exception:
            desc = {}
        m_type = str(desc.get("type", "")).lower()
        if m in per_model_sensors:
            s = per_model_sensors[m]
        elif sensor is not None:
            s = sensor
        else:
            s = _default_sensor_for_model(m_n)
        model_configs.append(
            ModelConfig(name=m, backend=eff_backend, sensor=s, model_type=m_type)
        )

    config = ExportConfig(
        format=fmt,
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        save_manifest=save_manifest,
        fail_on_bad_input=fail_on_bad_input,
        chunk_size=chunk_size,
        infer_batch_size=infer_batch_size,
        num_workers=num_workers,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        async_write=async_write,
        writer_workers=writer_workers,
        resume=resume,
        show_progress=show_progress,
        input_prep=input_prep,
    )

    exporter = BatchExporter(
        spatials=spatials,
        temporal=temporal,
        models=model_configs,
        target=target,
        output=output,
        config=config,
        backend=backend_n,
        resolved_backend=resolved_backend,
        device=device_n,
        provider_factory=_provider_factory_for_backend(backend_n),
    )
    return exporter.run()

"""Public API for rs-embed: get_embedding, get_embeddings_batch, export_batch.

Orchestration flow
------------------
1. **Validation** – _validate_spatials checks spatial/temporal/output specs.
2. **Context**    – _prepare_embedding_request_context resolves model, backend,
   device, sensor, input-prep and returns a frozen _EmbeddingRequestContext.
3. **Execution**  – _run_embedding_request (single/batch) or _export_batch_per_item
   (chunked export with prefetch → infer → write pipeline).

Tiling/input-prep logic lives in ``internal.api.tiling_helpers``.  All other
delegate helpers (validation, runtime, output normalization, manifests, etc.)
live under ``internal.api.*``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .core.export_helpers import embedding_to_numpy as _embedding_to_numpy
from .core.export_helpers import jsonable as _jsonable
from .core.export_helpers import sanitize_key as _sanitize_key
from .core.export_helpers import sensor_cache_key as _sensor_cache_key
from .core.export_helpers import sha1 as _sha1
from .internal.api.api_helpers import (
    fetch_gee_patch_raw as _fetch_gee_patch_raw,
    inspect_input_raw as _inspect_input_raw,
    normalize_backend_name as _normalize_backend_name,
    normalize_device_name as _normalize_device_name,
    normalize_input_chw as _normalize_input_chw,
    normalize_model_name as _normalize_model_name,
)
from .internal.api.checkpoint_helpers import (
    is_incomplete_combined_manifest as _is_incomplete_combined_manifest,
)
from .internal.api.prefetch_helpers import (
    build_gee_prefetch_plan as _build_gee_prefetch_plan,
    select_prefetched_channels as _select_prefetched_channels,
)
from .internal.api.export_flow_helpers import (
    build_one_point_payload as _build_one_point_payload,
    export_combined as _export_combined_npz,
    write_one_payload as _write_one_payload,
)
from .internal.api.runtime_helpers import (
    call_embedder_get_embedding as _call_embedder_get_embedding,
    embedder_accepts_input_chw as _embedder_accepts_input_chw,
    get_embedder_bundle_cached as _get_embedder_bundle_cached,
    run_with_retry as _run_with_retry,
    sensor_key as _sensor_key,
    supports_batch_api as _supports_batch_api,
    supports_prefetched_batch_api as _supports_prefetched_batch_api,
)
from .internal.api.validation_helpers import (
    assert_supported as _assert_supported,
    validate_specs as _validate_specs,
)
from .internal.api.manifest_helpers import (
    combined_resume_manifest as _combined_resume_manifest,
    load_json_dict as _load_json_dict,
    point_failure_manifest as _point_failure_manifest,
    point_resume_manifest as _point_resume_manifest,
)
from .internal.api.progress_helpers import create_progress as _create_progress
from .internal.api.model_defaults_helpers import (
    default_sensor_for_model as _default_sensor_for_model,
)
from .internal.api.output_helpers import (
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
    BBox,
)
from .embedders.catalog import MODEL_ALIASES, MODEL_SPECS
from .providers import ProviderBase, get_provider, has_provider, list_providers

# Backward-compatibility hook: tests/downstream may monkeypatch api.GEEProvider.
GEEProvider: Optional[Callable[..., ProviderBase]] = None


# -----------------------------------------------------------------------------
# Internal: export flow wrappers (kept in api for monkeypatch-friendly tests)
# -----------------------------------------------------------------------------


def _create_default_gee_provider() -> ProviderBase:
    # If tests/downstream code set api.GEEProvider, use it directly.
    cls = GEEProvider
    if cls is None:
        # Attempt import without caching in the module global — keeps the
        # provider registry as the canonical source after first creation.
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
        # Keep monkeypatch-friendly behavior for existing tests.
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
    # Keep legacy behavior as a fallback; downstream provider creation will
    # still raise a clear error if no provider backend is available.
    return "gee"


def _resolve_embedding_api_backend(model_n: str, backend_n: str) -> str:
    """Normalize backend semantics for precomputed models.

    For precomputed products, the data source is fixed by the model. This helper
    makes the public API less coupled to provider/backend naming details by
    auto-selecting a compatible access backend when the generic default
    (historically `gee`) is passed in.
    """
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
    # Legacy compatibility: some precomputed users still pass backend="local".
    if backend_n == "local" and "auto" in allowed and not provider_allowed:
        return "auto"
    if has_provider(backend_n) and provider_allowed:
        return backend_n

    # Public API default is historically backend="gee". For precomputed models,
    # treat that as "use the model's fixed source via its supported access path".
    if backend_n in {"gee", "auto"}:
        if "auto" in allowed:
            return "auto"
        if "local" in allowed:
            return "local"
        if provider_allowed:
            return _default_provider_backend_for_api()

    return backend_n


# ---------------------------------------------------------------------------
# Tiling / input-prep (delegated to internal.api.tiling_helpers)
# ---------------------------------------------------------------------------
from .internal.api.tiling_helpers import (  # noqa: E402
    _ResolvedInputPrepSpec,
    _resolve_input_prep_spec,
    _call_embedder_get_embedding_with_input_prep,
    _tile_yx_starts,
)


@dataclass(frozen=True)
class _ExportFlowOverrides:
    provider_factory: Optional[Callable[[], ProviderBase]]
    call_embedder_get_embedding_fn: Callable[..., Embedding]
    supports_prefetched_batch_api_fn: Callable[[Any], bool]
    supports_batch_api_fn: Callable[[Any], bool]


def _call_embedder_with_prep(
    input_prep: Optional[InputPrepSpec | str], **kwargs: Any
) -> Embedding:
    return _call_embedder_get_embedding_with_input_prep(input_prep=input_prep, **kwargs)


def _check_prefetched_batch(embedder: Any, *, disable: bool) -> bool:
    if disable:
        return False
    return _supports_prefetched_batch_api(embedder)


def _check_batch(embedder: Any, *, disable: bool) -> bool:
    if disable:
        return False
    return _supports_batch_api(embedder)


# Contract: wraps embedder call + batch-support queries with input_prep awareness.
# When input_prep is tile/auto, batch APIs are disabled (each point needs its own
# tiling pass).  The returned overrides are consumed by _export_batch_per_item and
# _export_combined_npz.
def _make_export_flow_overrides(
    *,
    backend: str,
    input_prep: Optional[InputPrepSpec | str],
) -> _ExportFlowOverrides:
    from functools import partial

    provider_factory = _provider_factory_for_backend(backend)
    resolved_input_prep = _resolve_input_prep_spec(input_prep)
    disable_batch = (input_prep is not None) and (
        resolved_input_prep.mode in {"tile", "auto"}
    )

    return _ExportFlowOverrides(
        provider_factory=provider_factory,
        call_embedder_get_embedding_fn=partial(_call_embedder_with_prep, input_prep),
        supports_prefetched_batch_api_fn=partial(
            _check_prefetched_batch, disable=disable_batch
        ),
        supports_batch_api_fn=partial(_check_batch, disable=disable_batch),
    )


@dataclass(frozen=True)
class _ExportBatchTarget:
    mode: str  # "combined" | "per_item"
    out_file: Optional[str] = None
    out_dir: Optional[str] = None
    names: Optional[List[str]] = None


def _normalize_export_layout(layout: str) -> str:
    layout_n = str(layout).strip().lower().replace("-", "_")
    if layout_n in {"combined", "single_file", "file"}:
        return "combined"
    if layout_n in {"per_item", "dir", "directory"}:
        return "per_item"
    raise ModelError(
        f"Unsupported export layout: {layout!r}. Supported: 'combined', 'per_item'."
    )


def _device_has_gpu(device: str) -> bool:
    dev = str(device or "").strip().lower()
    if dev and dev not in {"auto", "cpu"}:
        return True
    if dev == "cpu":
        return False
    try:
        import torch  # type: ignore

        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return True
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(getattr(mps, "is_available", lambda: False)()):
            return True
    except Exception:
        return False
    return False


def _should_prefer_batch_inference(*, device: str) -> bool:
    return _device_has_gpu(device)


def _recompute_point_manifest_summary(manifest: Dict[str, Any]) -> None:
    models = manifest.get("models") or []
    n_failed = sum(
        1 for x in models if isinstance(x, dict) and x.get("status") == "failed"
    )
    if not models:
        manifest["status"] = "ok"
        manifest["summary"] = {"total_models": 0, "failed_models": 0, "ok_models": 0}
        return
    if n_failed == 0:
        status = "ok"
    elif n_failed < len(models):
        status = "partial"
    else:
        status = "failed"
    manifest["status"] = status
    manifest["summary"] = {
        "total_models": len(models),
        "failed_models": n_failed,
        "ok_models": len(models) - n_failed,
    }


def _resolve_export_batch_target(
    *,
    n_spatials: int,
    ext: str,
    out: Optional[str],
    layout: Optional[str],
    out_dir: Optional[str],
    out_path: Optional[str],
    names: Optional[List[str]],
) -> _ExportBatchTarget:
    # New decoupled API: `out` + `layout`.
    if (out is not None) or (layout is not None):
        if out is None or layout is None:
            raise ModelError(
                "Provide both out and layout when using the decoupled output API."
            )
        if out_dir is not None or out_path is not None:
            raise ModelError("Use either out+layout or out_dir/out_path, not both.")
        layout_n = _normalize_export_layout(layout)
        if layout_n == "combined":
            out_path = out
        else:
            out_dir = out

    # Backward-compatible API: `out_dir` xor `out_path`.
    if out_dir is None and out_path is None:
        raise ModelError("export_batch requires out_dir or out_path.")
    if out_dir is not None and out_path is not None:
        raise ModelError("Provide only one of out_dir or out_path.")

    if out_path is not None:
        out_file = out_path if out_path.endswith(ext) else (out_path + ext)
        return _ExportBatchTarget(mode="combined", out_file=out_file)

    assert out_dir is not None
    point_names = (
        names if names is not None else [f"p{i:05d}" for i in range(n_spatials)]
    )
    if len(point_names) != n_spatials:
        raise ModelError("names must have the same length as spatials.")
    return _ExportBatchTarget(mode="per_item", out_dir=out_dir, names=point_names)


# Contract: for a chunk of point indices, run inference across all requested models.
# Dispatches to single-point or batched inference based on device / model capability.
# Returns {(point_idx, model_name): {"embedding": ..., "meta": ..., "status": ...}}.
def _infer_chunk_embeddings_for_per_item(
    *,
    idxs: List[int],
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    models: List[str],
    backend_n: str,
    resolved_backend: Dict[str, str],
    device: str,
    output: OutputSpec,
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
    provider_enabled: bool,
    inputs_cache: Dict[Tuple[int, str], np.ndarray],
    prefetch_errors: Dict[Tuple[int, str], str],
    continue_on_error: bool,
    max_retries: int,
    retry_backoff_s: float,
    infer_batch_size: int,
    model_progress_cb: Optional[Callable[[str], None]] = None,
    input_prep: Optional[InputPrepSpec | str] = None,
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    prefer_batch = _should_prefer_batch_inference(device=device)
    infer_bs = max(1, int(infer_batch_size))
    input_prep_resolved = _resolve_input_prep_spec(input_prep)
    explicit_nonresize_input_prep = (input_prep is not None) and (
        input_prep_resolved.mode in {"tile", "auto"}
    )

    def _mark_done(model_name: str) -> None:
        if model_progress_cb is not None:
            try:
                model_progress_cb(model_name)
            except Exception:
                pass

    for m in models:
        sspec = resolved_sensor.get(m)
        needs_provider_input = (
            provider_enabled
            and sspec is not None
            and "precomputed" not in (model_type.get(m) or "")
        )
        skey = (
            _sensor_cache_key(sspec)
            if needs_provider_input and sspec is not None
            else None
        )
        sensor_k = _sensor_key(sspec)
        m_backend = resolved_backend.get(m, backend_n)
        embedder, lock = _get_embedder_bundle_cached(
            _normalize_model_name(m), m_backend, device, sensor_k
        )

        def _record_ok(i: int, emb: Embedding) -> None:
            e_np = _embedding_to_numpy(emb)
            out[(i, m)] = {
                "status": "ok",
                "embedding": e_np,
                "meta": _jsonable(getattr(emb, "meta", None)),
            }
            _mark_done(m)

        def _record_err(i: int, e: Exception) -> None:
            out[(i, m)] = {"status": "failed", "error": repr(e)}
            _mark_done(m)

        def _get_input_for_idx(i: int) -> Optional[np.ndarray]:
            if not needs_provider_input or skey is None:
                return None
            hit = inputs_cache.get((i, skey))
            if hit is not None:
                return hit
            pref_err = prefetch_errors.get((i, skey))
            if pref_err:
                raise RuntimeError(
                    f"Prefetch previously failed for model={m}, index={i}, sensor={skey}: {pref_err}"
                )
            raise RuntimeError(
                f"Missing prefetched input for model={m}, index={i}, sensor={skey}"
            )

        def _infer_single(i: int) -> Embedding:
            inp = _get_input_for_idx(i)
            return _run_with_retry(
                lambda: _call_embedder_get_embedding_with_input_prep(
                    embedder=embedder,
                    spatial=spatials[i],
                    temporal=temporal,
                    sensor=sspec,
                    output=output,
                    backend=m_backend,
                    device=device,
                    input_chw=inp,
                    input_prep=input_prep,
                ),
                retries=max_retries,
                backoff_s=retry_backoff_s,
            )

        batch_attempted = False
        batch_succeeded = False

        can_batch_prefetched = (
            prefer_batch
            and (not explicit_nonresize_input_prep)
            and _supports_prefetched_batch_api(embedder)
            and needs_provider_input
            and sspec is not None
            and skey is not None
        )
        can_batch_no_input = (
            prefer_batch
            and (not explicit_nonresize_input_prep)
            and _supports_batch_api(embedder)
            and (not needs_provider_input)
        )

        if can_batch_prefetched:
            batch_attempted = True
            try:
                ready: List[Tuple[int, SpatialSpec, np.ndarray]] = []
                for i in idxs:
                    try:
                        inp = _get_input_for_idx(i)
                        assert inp is not None
                        ready.append(
                            (i, spatials[i], np.asarray(inp, dtype=np.float32))
                        )
                    except Exception as e:
                        if not continue_on_error:
                            raise
                        _record_err(
                            i, e if isinstance(e, Exception) else RuntimeError(str(e))
                        )
                for start in range(0, len(ready), infer_bs):
                    sub = ready[start : start + infer_bs]
                    if not sub:
                        continue
                    sub_indices = [t[0] for t in sub]
                    sub_spatials = [t[1] for t in sub]
                    sub_inputs = [t[2] for t in sub]

                    def _infer_batch_prefetched():
                        with lock:
                            return embedder.get_embeddings_batch_from_inputs(
                                spatials=sub_spatials,
                                input_chws=sub_inputs,
                                temporal=temporal,
                                sensor=sspec,
                                output=output,
                                backend=m_backend,
                                device=device,
                            )

                    batch_out = _run_with_retry(
                        _infer_batch_prefetched,
                        retries=max_retries,
                        backoff_s=retry_backoff_s,
                    )
                    if len(batch_out) != len(sub_indices):
                        raise RuntimeError(
                            f"Model {m} returned {len(batch_out)} embeddings for {len(sub_indices)} prefetched inputs."
                        )
                    for j, emb in enumerate(batch_out):
                        emb_n = _normalize_embedding_output(emb=emb, output=output)
                        _record_ok(sub_indices[j], emb_n)
                batch_succeeded = True
            except Exception:
                batch_succeeded = False

        if (not batch_attempted) and can_batch_no_input:
            batch_attempted = True
            try:
                for start in range(0, len(idxs), infer_bs):
                    sub_indices = idxs[start : start + infer_bs]
                    sub_spatials = [spatials[i] for i in sub_indices]

                    def _infer_batch():
                        with lock:
                            return embedder.get_embeddings_batch(
                                spatials=sub_spatials,
                                temporal=temporal,
                                sensor=sspec,
                                output=output,
                                backend=m_backend,
                                device=device,
                            )

                    batch_out = _run_with_retry(
                        _infer_batch,
                        retries=max_retries,
                        backoff_s=retry_backoff_s,
                    )
                    if len(batch_out) != len(sub_indices):
                        raise RuntimeError(
                            f"Model {m} returned {len(batch_out)} embeddings for {len(sub_indices)} inputs."
                        )
                    for j, emb in enumerate(batch_out):
                        emb_n = _normalize_embedding_output(emb=emb, output=output)
                        _record_ok(sub_indices[j], emb_n)
                batch_succeeded = True
            except Exception:
                batch_succeeded = False

        if not batch_succeeded:
            for i in idxs:
                if (i, m) in out:
                    continue
                try:
                    emb = _infer_single(i)
                    _record_ok(i, emb)
                except Exception as e:
                    if not continue_on_error:
                        raise
                    _record_err(i, e)

    return out


def _inject_precomputed_embeddings_into_point_payload(
    *,
    point_index: int,
    models: List[str],
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    embed_results: Dict[Tuple[int, str], Dict[str, Any]],
) -> None:
    model_entries = manifest.get("models") or []
    entry_by_model = {
        str(entry.get("model")): entry
        for entry in model_entries
        if isinstance(entry, dict)
    }
    for m in models:
        entry = entry_by_model.get(m)
        if entry is None:
            continue
        rec = embed_results.get((point_index, m))
        if rec is None:
            continue
        if rec.get("status") == "ok":
            e_np = np.asarray(rec["embedding"])
            emb_key = f"embedding__{_sanitize_key(m)}"
            arrays[emb_key] = e_np
            entry["embedding"] = {
                "npz_key": emb_key,
                "dtype": str(e_np.dtype),
                "shape": list(e_np.shape),
                "sha1": _sha1(e_np),
            }
            entry["meta"] = rec.get("meta")
        else:
            if entry.get("status") != "failed":
                entry["status"] = "failed"
                entry["error"] = rec.get("error")
            entry["embedding"] = None
            entry["meta"] = None
    _recompute_point_manifest_summary(manifest)


# Contract: orchestrates the full per-item export pipeline in chunks:
#   prefetch inputs (threaded) → infer embeddings (per model) → write payloads.
# Handles resume, progress, and error collection.  Returns one manifest dict per point.
def _export_batch_per_item(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    models: List[str],
    out_dir: str,
    names: List[str],
    ext: str,
    backend_n: str,
    resolved_backend: Dict[str, str],
    device: str,
    output: OutputSpec,
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
    fmt: str,
    save_inputs: bool,
    save_embeddings: bool,
    save_manifest: bool,
    fail_on_bad_input: bool,
    chunk_size: int,
    num_workers: int,
    continue_on_error: bool,
    max_retries: int,
    retry_backoff_s: float,
    async_write: bool,
    writer_workers: int,
    infer_batch_size: int,
    resume: bool,
    show_progress: bool,
    input_prep: Optional[InputPrepSpec | str] = None,
) -> List[Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)

    n = len(spatials)
    progress = _create_progress(
        enabled=bool(show_progress),
        total=n,
        desc="export_batch",
        unit="point",
    )
    model_progress: Dict[str, Any] = {}

    manifests: List[Dict[str, Any]] = []
    pending_idxs: List[int] = []
    try:
        for i in range(n):
            out_file = os.path.join(out_dir, f"{names[i]}{ext}")
            if bool(resume) and os.path.exists(out_file):
                manifests.append(
                    _point_resume_manifest(
                        point_index=i,
                        spatial=spatials[i],
                        temporal=temporal,
                        output=output,
                        backend=backend_n,
                        device=device,
                        out_file=out_file,
                    )
                )
                progress.update(1)
            else:
                pending_idxs.append(i)

        if not pending_idxs:
            manifests.sort(key=lambda x: int(x.get("point_index", -1)))
            return manifests

        if save_embeddings:
            model_progress = {
                m: _create_progress(
                    enabled=bool(show_progress),
                    total=len(pending_idxs),
                    desc=f"infer[{m}]",
                    unit="point",
                )
                for m in models
            }

        def _on_model_done(model_name: str) -> None:
            bar = model_progress.get(model_name)
            if bar is not None:
                bar.update(1)

        export_flow = _make_export_flow_overrides(
            backend=backend_n, input_prep=input_prep
        )
        provider_factory = export_flow.provider_factory
        provider_enabled = provider_factory is not None

        need_prefetch = (
            provider_enabled
            and bool(save_inputs or save_embeddings)
            and bool(pending_idxs)
        )
        pass_input_into_embedder = provider_enabled and bool(save_embeddings)
        provider: Optional[ProviderBase] = None
        band_resolver = None
        if need_prefetch:
            assert provider_factory is not None
            provider = provider_factory()
            band_resolver = getattr(provider, "normalize_bands", None)
        (
            sensor_by_key,
            fetch_sensor_by_key,
            sensor_to_fetch,
            sensor_models,
            fetch_members,
        ) = _build_gee_prefetch_plan(
            models=models,
            resolved_sensor=resolved_sensor,
            model_type=model_type,
            resolve_bands_fn=band_resolver,
        )

        if need_prefetch:
            assert provider is not None
            _run_with_retry(
                lambda: provider.ensure_ready(),
                retries=max_retries,
                backoff_s=retry_backoff_s,
            )

        csize = max(1, int(chunk_size))

        def _prefetch_chunk_inputs(
            idxs: List[int],
        ) -> Tuple[
            Dict[Tuple[int, str], np.ndarray],
            Dict[Tuple[int, str], Dict[str, Any]],
            Dict[Tuple[int, str], str],
        ]:
            inputs_cache: Dict[Tuple[int, str], np.ndarray] = {}
            input_reports: Dict[Tuple[int, str], Dict[str, Any]] = {}
            prefetch_errors: Dict[Tuple[int, str], str] = {}

            if (not idxs) or (not need_prefetch) or provider is None:
                return inputs_cache, input_reports, prefetch_errors

            tasks = [
                (i, fetch_key, fetch_sensor)
                for i in idxs
                for fetch_key, fetch_sensor in fetch_sensor_by_key.items()
            ]
            if not tasks:
                return inputs_cache, input_reports, prefetch_errors

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _fetch_one(ii: int, sk: str, ss: SensorSpec):
                assert provider is not None
                x = _run_with_retry(
                    lambda: _fetch_gee_patch_raw(
                        provider, spatial=spatials[ii], temporal=temporal, sensor=ss
                    ),
                    retries=max_retries,
                    backoff_s=retry_backoff_s,
                )
                return ii, sk, x

            mw = max(1, int(num_workers))
            with ThreadPoolExecutor(max_workers=mw) as ex:
                fut_map = {
                    ex.submit(_fetch_one, ii, sk, ss): (ii, sk)
                    for (ii, sk, ss) in tasks
                }
                for fut in as_completed(fut_map):
                    ii, sk = fut_map[fut]
                    try:
                        ii, sk, x = fut.result()
                    except Exception as e:
                        if not continue_on_error:
                            raise
                        err_s = repr(e)
                        for member_skey in fetch_members.get(sk, []):
                            prefetch_errors[(ii, member_skey)] = err_s
                        continue
                    for member_skey in fetch_members.get(sk, []):
                        member_idx = sensor_to_fetch[member_skey][1]
                        x_member = _normalize_input_chw(
                            _select_prefetched_channels(x, member_idx),
                            expected_channels=len(member_idx),
                            name=f"gee_input_{member_skey}",
                        )
                        if fail_on_bad_input:
                            sspec_member = sensor_by_key[member_skey]
                            rep = _inspect_input_raw(
                                x_member,
                                sensor=sspec_member,
                                name=f"gee_input_{member_skey}",
                            )
                            if not bool(rep.get("ok", True)):
                                issues = (rep.get("report", {}) or {}).get("issues", [])
                                mlist = sorted(set(sensor_models.get(member_skey, [])))
                                err = RuntimeError(
                                    f"Input inspection failed for index={ii}, sensor={member_skey}, models={mlist}: {issues}"
                                )
                                if not continue_on_error:
                                    raise err
                                prefetch_errors[(ii, member_skey)] = repr(err)
                                continue
                            input_reports[(ii, member_skey)] = rep
                        inputs_cache[(ii, member_skey)] = x_member

            return inputs_cache, input_reports, prefetch_errors

        chunk_groups = [
            pending_idxs[chunk_start : chunk_start + csize]
            for chunk_start in range(0, len(pending_idxs), csize)
        ]
        prefetch_pipeline_ex = None
        prefetched_chunk_fut = None
        try:
            if need_prefetch and provider is not None and chunk_groups:
                from concurrent.futures import ThreadPoolExecutor

                # A one-slot pipeline overlaps prefetch(chunk k+1) with infer/write(chunk k)
                # while keeping memory bounded to roughly two chunks of cached inputs.
                prefetch_pipeline_ex = ThreadPoolExecutor(max_workers=1)
                prefetched_chunk_fut = prefetch_pipeline_ex.submit(
                    _prefetch_chunk_inputs, chunk_groups[0]
                )

            for chunk_idx, idxs in enumerate(chunk_groups):
                if prefetched_chunk_fut is not None:
                    inputs_cache, input_reports, prefetch_errors = (
                        prefetched_chunk_fut.result()
                    )
                else:
                    inputs_cache, input_reports, prefetch_errors = (
                        _prefetch_chunk_inputs(idxs)
                    )
                prefetched_chunk_fut = None

                if prefetch_pipeline_ex is not None and (chunk_idx + 1) < len(
                    chunk_groups
                ):
                    prefetched_chunk_fut = prefetch_pipeline_ex.submit(
                        _prefetch_chunk_inputs, chunk_groups[chunk_idx + 1]
                    )

                chunk_embed_results: Dict[Tuple[int, str], Dict[str, Any]] = {}
                use_chunk_batch_infer = bool(
                    save_embeddings and _should_prefer_batch_inference(device=device)
                )
                if use_chunk_batch_infer:
                    chunk_embed_results = _infer_chunk_embeddings_for_per_item(
                        idxs=idxs,
                        spatials=spatials,
                        temporal=temporal,
                        models=models,
                        backend_n=backend_n,
                        resolved_backend=resolved_backend,
                        device=device,
                        output=output,
                        resolved_sensor=resolved_sensor,
                        model_type=model_type,
                        provider_enabled=provider_enabled,
                        inputs_cache=inputs_cache,
                        prefetch_errors=prefetch_errors,
                        continue_on_error=continue_on_error,
                        max_retries=max_retries,
                        retry_backoff_s=retry_backoff_s,
                        infer_batch_size=infer_batch_size,
                        model_progress_cb=_on_model_done,
                        input_prep=input_prep,
                    )

                # export each point in chunk
                writer_async = bool(async_write)
                writer_mw = max(1, int(writer_workers))
                write_futs = []
                writer_ex = None
                if writer_async:
                    from concurrent.futures import ThreadPoolExecutor

                    writer_ex = ThreadPoolExecutor(max_workers=writer_mw)
                for i in idxs:
                    out_file = os.path.join(out_dir, f"{names[i]}{ext}")
                    try:
                        arrays, manifest = _build_one_point_payload(
                            point_index=i,
                            spatial=spatials[i],
                            temporal=temporal,
                            models=models,
                            backend=backend_n,
                            resolved_backend=resolved_backend,
                            device=device,
                            output=output,
                            resolved_sensor=resolved_sensor,
                            model_type=model_type,
                            inputs_cache=inputs_cache,
                            input_reports=input_reports,
                            prefetch_errors=prefetch_errors,
                            pass_input_into_embedder=pass_input_into_embedder,
                            save_inputs=save_inputs,
                            save_embeddings=(
                                False if use_chunk_batch_infer else save_embeddings
                            ),
                            fail_on_bad_input=fail_on_bad_input,
                            continue_on_error=continue_on_error,
                            max_retries=max_retries,
                            retry_backoff_s=retry_backoff_s,
                            model_progress_cb=(
                                None
                                if use_chunk_batch_infer
                                else (_on_model_done if save_embeddings else None)
                            ),
                            normalize_model_name_fn=_normalize_model_name,
                            sensor_key_fn=_sensor_key,
                            get_embedder_bundle_cached_fn=_get_embedder_bundle_cached,
                            run_with_retry_fn=_run_with_retry,
                            fetch_gee_patch_raw_fn=_fetch_gee_patch_raw,
                            inspect_input_raw_fn=_inspect_input_raw,
                            call_embedder_get_embedding_fn=export_flow.call_embedder_get_embedding_fn,
                            provider_factory=provider_factory,
                        )
                        if use_chunk_batch_infer:
                            _inject_precomputed_embeddings_into_point_payload(
                                point_index=i,
                                models=models,
                                arrays=arrays,
                                manifest=manifest,
                                embed_results=chunk_embed_results,
                            )
                    except Exception as e:
                        if not continue_on_error:
                            if writer_ex is not None:
                                writer_ex.shutdown(wait=False)
                            raise
                        manifests.append(
                            _point_failure_manifest(
                                point_index=i,
                                spatial=spatials[i],
                                temporal=temporal,
                                output=output,
                                backend=backend_n,
                                device=device,
                                stage="build",
                                error=e,
                            )
                        )
                        progress.update(1)
                        continue

                    if writer_ex is not None:
                        fut = writer_ex.submit(
                            _write_one_payload,
                            out_path=out_file,
                            arrays=arrays,
                            manifest=manifest,
                            save_manifest=save_manifest,
                            fmt=fmt,
                            max_retries=max_retries,
                            retry_backoff_s=retry_backoff_s,
                            run_with_retry_fn=_run_with_retry,
                            jsonable_fn=_jsonable,
                        )
                        write_futs.append((i, fut))
                    else:
                        try:
                            mani = _write_one_payload(
                                out_path=out_file,
                                arrays=arrays,
                                manifest=manifest,
                                save_manifest=save_manifest,
                                fmt=fmt,
                                max_retries=max_retries,
                                retry_backoff_s=retry_backoff_s,
                                run_with_retry_fn=_run_with_retry,
                                jsonable_fn=_jsonable,
                            )
                        except Exception as e:
                            if not continue_on_error:
                                raise
                            mani = _point_failure_manifest(
                                point_index=i,
                                spatial=spatials[i],
                                temporal=temporal,
                                output=output,
                                backend=backend_n,
                                device=device,
                                stage="write",
                                error=e,
                            )
                        manifests.append(mani)
                        progress.update(1)

                if writer_ex is not None:
                    from concurrent.futures import as_completed

                    try:
                        fut_map = {fut: i for (i, fut) in write_futs}
                        for fut in as_completed(fut_map):
                            i = fut_map[fut]
                            try:
                                manifests.append(fut.result())
                            except Exception as e:
                                if not continue_on_error:
                                    raise
                                manifests.append(
                                    _point_failure_manifest(
                                        point_index=i,
                                        spatial=spatials[i],
                                        temporal=temporal,
                                        output=output,
                                        backend=backend_n,
                                        device=device,
                                        stage="write",
                                        error=e,
                                    )
                                )
                            finally:
                                progress.update(1)
                    finally:
                        writer_ex.shutdown(wait=True)
        finally:
            if prefetch_pipeline_ex is not None:
                prefetch_pipeline_ex.shutdown(wait=True)

        manifests.sort(key=lambda x: int(x.get("point_index", -1)))
        return manifests
    finally:
        for bar in model_progress.values():
            bar.close()
        progress.close()


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

    - Accept any SpatialSpec list (like get_embeddings_batch).
    - Reuse cached embedder instances to avoid re-loading models/providers.
    - For GEE backends, prefetch raw inputs once per (point, sensor) and reuse
      them for input export and embedding inference.
    - Output target can be specified via legacy `out_dir`/`out_path`, or via
      decoupled `out` + `layout` (`"per_item"` / `"combined"`).
    - Inference batching is auto-selected internally: CPU defaults to per-item
      inference; GPU/accelerators prefer batched inference when supported.
    """
    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty List[SpatialSpec].")
    if not isinstance(models, list) or len(models) == 0:
        raise ModelError("models must be a non-empty List[str].")

    backend_n = _normalize_backend_name(backend)
    device = _normalize_device_name(device)
    fmt = format.lower().strip()
    infer_batch_size_n = max(
        1, int(chunk_size if infer_batch_size is None else infer_batch_size)
    )
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

    # validate specs early — use helper to avoid double-validating spatials[0]
    _validate_spatials(spatials=spatials, temporal=temporal, output=output)

    per_model_sensors = per_model_sensors or {}

    # resolve sensors + type + backend per model; validate capabilities upfront
    resolved_sensor: Dict[str, Optional[SensorSpec]] = {}
    resolved_backend: Dict[str, str] = {}
    model_type: Dict[str, str] = {}
    for m in models:
        m_n = _normalize_model_name(m)
        eff_backend = _resolve_embedding_api_backend(m_n, backend_n)
        resolved_backend[m] = eff_backend
        cls = get_embedder_cls(m_n)
        try:
            emb_check = cls()
            # Capability mismatch (wrong backend/output/temporal) is always a
            # fatal configuration error — raise before the export starts.
            _assert_supported(
                emb_check, backend=eff_backend, output=output, temporal=temporal
            )
            desc = emb_check.describe() or {}
        except ModelError:
            raise
        except Exception:
            # __init__ or describe() failure: let the export flow handle it
            # per-model, respecting continue_on_error.
            desc = {}
        model_type[m] = str(desc.get("type", "")).lower()
        if m in per_model_sensors:
            resolved_sensor[m] = per_model_sensors[m]
        elif sensor is not None:
            resolved_sensor[m] = sensor
        else:
            resolved_sensor[m] = _default_sensor_for_model(m_n)

    # combined mode
    if target.mode == "combined":
        out_file = target.out_file
        assert out_file is not None
        export_flow = _make_export_flow_overrides(
            backend=backend_n, input_prep=input_prep
        )
        if bool(resume) and os.path.exists(out_file):
            json_path = os.path.splitext(out_file)[0] + ".json"
            resume_manifest = _load_json_dict(json_path)
            if not _is_incomplete_combined_manifest(resume_manifest):
                return _combined_resume_manifest(
                    spatials=spatials,
                    temporal=temporal,
                    output=output,
                    backend=backend_n,
                    device=device,
                    out_file=out_file,
                )
        return _export_combined_npz(
            spatials=spatials,
            temporal=temporal,
            models=models,
            out_path=out_file,
            backend=backend_n,
            resolved_backend=resolved_backend,
            device=device,
            output=output,
            resolved_sensor=resolved_sensor,
            model_type=model_type,
            save_inputs=save_inputs,
            save_embeddings=save_embeddings,
            save_manifest=save_manifest,
            fail_on_bad_input=fail_on_bad_input,
            chunk_size=chunk_size,
            num_workers=num_workers,
            fmt=fmt,
            continue_on_error=continue_on_error,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            # Preserve historical combined-export behavior: auto still attempts
            # batched model APIs (with fallback to per-item inference on failure).
            inference_strategy="auto",
            infer_batch_size=infer_batch_size_n,
            resume=resume,
            show_progress=show_progress,
            provider_factory=export_flow.provider_factory,
            run_with_retry_fn=_run_with_retry,
            fetch_gee_patch_raw_fn=_fetch_gee_patch_raw,
            inspect_input_raw_fn=_inspect_input_raw,
            normalize_input_chw_fn=_normalize_input_chw,
            select_prefetched_channels_fn=_select_prefetched_channels,
            create_progress_fn=_create_progress,
            get_embedder_bundle_cached_fn=_get_embedder_bundle_cached,
            sensor_key_fn=_sensor_key,
            normalize_model_name_fn=_normalize_model_name,
            call_embedder_get_embedding_fn=export_flow.call_embedder_get_embedding_fn,
            supports_prefetched_batch_api_fn=export_flow.supports_prefetched_batch_api_fn,
            supports_batch_api_fn=export_flow.supports_batch_api_fn,
            sensor_cache_key_fn=_sensor_cache_key,
            load_json_dict_fn=_load_json_dict,
            is_incomplete_combined_manifest_fn=_is_incomplete_combined_manifest,
            write_one_payload_fn=_write_one_payload,
        )

    # per-item mode (directory layout)
    assert target.out_dir is not None
    assert target.names is not None
    return _export_batch_per_item(
        spatials=spatials,
        temporal=temporal,
        models=models,
        out_dir=target.out_dir,
        names=target.names,
        ext=ext,
        backend_n=backend_n,
        resolved_backend=resolved_backend,
        device=device,
        output=output,
        resolved_sensor=resolved_sensor,
        model_type=model_type,
        fmt=fmt,
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        save_manifest=save_manifest,
        fail_on_bad_input=fail_on_bad_input,
        chunk_size=chunk_size,
        num_workers=num_workers,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        async_write=async_write,
        writer_workers=writer_workers,
        infer_batch_size=infer_batch_size_n,
        resume=resume,
        show_progress=show_progress,
        input_prep=input_prep,
    )

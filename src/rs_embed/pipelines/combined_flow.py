"""Combined-layout helper flow for :class:`BatchExporter`.

This module keeps combined-export model execution helpers separate from the
``BatchExporter`` class body for readability. It is intentionally internal and
called from ``BatchExporter._run_combined``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig, Status, TaskResult
from ..tools.runtime import (
    get_embedder_bundle_cached,
    sensor_key,
)
from ..tools.serialization import (
    jsonable,
    sanitize_key,
    sensor_cache_key,
)
from ..tools.normalization import normalize_model_name
from ..tools.checkpoint_utils import drop_model_arrays
from ..tools.progress import create_progress


def run_pending_models(
    *,
    pending_models: List[str],
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
    backend: str,
    resolved_backend: Optional[Dict[str, str]] = None,
    provider_enabled: bool,
    device: str,
    save_inputs: bool,
    save_embeddings: bool,
    continue_on_error: bool,
    chunk_size: int,
    inference_strategy: str,
    infer_batch_size: int,
    max_retries: int,
    retry_backoff_s: float,
    show_progress: bool,
    input_refs_by_sensor: Dict[str, Dict[str, Any]],
    get_or_fetch_input_fn: Callable[[int, str, SensorSpec], np.ndarray],
    write_checkpoint_fn: Callable[..., Dict[str, Any]],
    progress: Any,
    inference_engine: Optional[Any] = None,
    progress_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run inference for each pending model, delegating to *inference_engine*.

    If *inference_engine* is ``None`` a temporary one is built (keeps older
    call-sites and tests working without changes).

    Callback contracts
    ------------------
    ``get_or_fetch_input_fn`` must return a CHW ndarray for
    ``(point_index, sensor_key, sensor_spec)``.
    ``write_checkpoint_fn`` must accept ``stage=...`` and return an updated
    combined manifest dict.
    """
    if inference_engine is None:
        from .inference import InferenceEngine

        inference_engine = InferenceEngine(
            device=device,
            output=output,
            config=ExportConfig(
                save_inputs=save_inputs,
                save_embeddings=save_embeddings,
                continue_on_error=continue_on_error,
                chunk_size=chunk_size,
                infer_batch_size=infer_batch_size,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                show_progress=show_progress,
            ),
        )

    _resolved_backend = resolved_backend or {}
    create_progress_fn = progress_factory or create_progress
    for m in pending_models:
        drop_model_arrays(arrays, m, sanitize_key=sanitize_key)
        infer_progress = create_progress_fn(
            enabled=bool(show_progress and save_embeddings),
            total=len(spatials),
            desc=f"infer[{m}]",
            unit="point",
        )
        infer_progress_done = 0
        m_entry: Dict[str, Any] = {
            "model": m,
            "sensor": jsonable(resolved_sensor.get(m)),
            "status": "ok",
        }
        sspec = resolved_sensor.get(m)
        try:
            m_backend = _resolved_backend.get(m, backend)
            is_precomputed = "precomputed" in (model_type.get(m) or "")

            # Resolve embedder just for describe()
            sensor_k = sensor_key(sspec)
            embedder, _lock = get_embedder_bundle_cached(
                normalize_model_name(m), m_backend, device, sensor_k
            )
            try:
                m_entry["describe"] = jsonable(embedder.describe())
            except Exception as e:
                m_entry["describe"] = {"error": repr(e)}

            needs_provider_input = (
                provider_enabled and sspec is not None and not is_precomputed
            )
            skey = (
                sensor_cache_key(sspec)
                if needs_provider_input and sspec is not None
                else None
            )

            # ── Save inputs ─────────────────────────────────────
            _gather_inputs(
                m_entry=m_entry,
                m=m,
                sspec=sspec,
                skey=skey,
                spatials=spatials,
                arrays=arrays,
                save_inputs=save_inputs,
                needs_provider_input=needs_provider_input,
                continue_on_error=continue_on_error,
                input_refs_by_sensor=input_refs_by_sensor,
                get_or_fetch_input_fn=get_or_fetch_input_fn,
            )

            # ── Inference via engine ────────────────────────────
            if save_embeddings:

                def _progress_cb(i: int) -> None:
                    nonlocal infer_progress_done
                    infer_progress_done += 1
                    infer_progress.update(1)

                results = inference_engine.infer_model(
                    model_name=m,
                    model_backend=m_backend,
                    sensor=sspec,
                    is_precomputed=is_precomputed,
                    provider_enabled=provider_enabled,
                    spatials=spatials,
                    temporal=temporal,
                    inference_strategy=inference_strategy,
                    get_input_fn=get_or_fetch_input_fn,
                    progress_cb=_progress_cb,
                )

                _pack_embedding_results(
                    results=results,
                    m_entry=m_entry,
                    m=m,
                    n=len(spatials),
                    arrays=arrays,
                )
            else:
                m_entry["embeddings"] = None
                m_entry["metas"] = None
        except Exception as e:
            if not continue_on_error:
                raise
            m_entry["status"] = "failed"
            m_entry["error"] = repr(e)
            m_entry["embeddings"] = None
            m_entry["metas"] = None
        finally:
            remaining = max(0, len(spatials) - infer_progress_done)
            if remaining > 0:
                try:
                    infer_progress.update(remaining)
                except Exception:
                    pass
            infer_progress.close()
            progress.update(1)

        manifest["models"].append(m_entry)
        manifest = write_checkpoint_fn(stage=f"model:{sanitize_key(m)}", final=False)

    return manifest


# ── Private helpers ────────────────────────────────────────────────


def _gather_inputs(
    *,
    m_entry: Dict[str, Any],
    m: str,
    sspec: Optional[SensorSpec],
    skey: Optional[str],
    spatials: List[SpatialSpec],
    arrays: Dict[str, np.ndarray],
    save_inputs: bool,
    needs_provider_input: bool,
    continue_on_error: bool,
    input_refs_by_sensor: Dict[str, Dict[str, Any]],
    get_or_fetch_input_fn: Callable[[int, str, SensorSpec], np.ndarray],
) -> None:
    """Collect and store provider inputs for *m*, updating *m_entry* in place."""
    if not (save_inputs and needs_provider_input and skey is not None):
        m_entry["inputs"] = None
        return

    if skey in input_refs_by_sensor:
        m_entry["inputs"] = {**input_refs_by_sensor[skey], "dedup_reused": True}
        return

    xs: List[np.ndarray] = []
    xs_indices: List[int] = []
    missing: List[tuple] = []
    for i in range(len(spatials)):
        try:
            x = get_or_fetch_input_fn(i, skey, sspec)
        except Exception as e:
            missing.append((i, repr(e)))
            continue
        xs.append(np.asarray(x, dtype=np.float32))
        xs_indices.append(int(i))

    if missing and not continue_on_error:
        raise RuntimeError(f"Missing prefetched inputs for model={m}: {missing}")

    if not xs:
        m_entry["inputs"] = None
        return

    try:
        arr = np.stack(xs, axis=0)
        in_key = f"inputs_bchw__{sanitize_key(m)}"
        arrays[in_key] = arr
        ref: Dict[str, Any] = {
            "npz_key": in_key,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        if len(xs_indices) != len(spatials):
            ref["indices"] = list(xs_indices)
        input_refs_by_sensor[skey] = dict(ref)
        m_entry["inputs"] = ref
    except Exception:
        keys = []
        for i in range(len(spatials)):
            try:
                x = get_or_fetch_input_fn(i, skey, sspec)
            except Exception:
                continue
            k = f"input_chw__{sanitize_key(m)}__{i:05d}"
            arrays[k] = np.asarray(x, dtype=np.float32)
            keys.append(k)
        ref = {"npz_keys": keys}
        input_refs_by_sensor[skey] = dict(ref)
        m_entry["inputs"] = ref


def _pack_embedding_results(
    *,
    results: Dict[int, TaskResult],
    m_entry: Dict[str, Any],
    m: str,
    n: int,
    arrays: Dict[str, np.ndarray],
) -> None:
    """Convert ``{idx: TaskResult}`` into stacked arrays and manifest entries."""
    ok_indices = [
        i for i in range(n) if i in results and results[i].status == Status.OK
    ]
    errors_by_idx = {
        i: results[i].error
        for i in range(n)
        if i in results and results[i].status == Status.FAILED
    }
    metas_by_idx = [results[i].meta if i in results else None for i in range(n)]

    if ok_indices:
        try:
            e_arr = np.stack(
                [results[i].embedding for i in ok_indices],
                axis=0,  # type: ignore[arg-type]
            )
            if len(ok_indices) == n:
                e_key = f"embeddings__{sanitize_key(m)}"
                arrays[e_key] = e_arr
                m_entry["embeddings"] = {
                    "npz_key": e_key,
                    "shape": list(e_arr.shape),
                    "dtype": str(e_arr.dtype),
                }
            else:
                keys = []
                index_map = []
                for j, i in enumerate(ok_indices):
                    k = f"embedding__{sanitize_key(m)}__{i:05d}"
                    arrays[k] = e_arr[j]
                    keys.append(k)
                    index_map.append(i)
                m_entry["embeddings"] = {"npz_keys": keys, "indices": index_map}
        except Exception:
            keys = []
            index_map = []
            for i in ok_indices:
                k = f"embedding__{sanitize_key(m)}__{i:05d}"
                arrays[k] = results[i].embedding  # type: ignore[assignment]
                keys.append(k)
                index_map.append(i)
            m_entry["embeddings"] = {"npz_keys": keys, "indices": index_map}
    else:
        m_entry["embeddings"] = None

    m_entry["metas"] = metas_by_idx
    if errors_by_idx:
        m_entry["failed_indices"] = sorted(errors_by_idx.keys())
        m_entry["errors_by_index"] = errors_by_idx
        if ok_indices:
            m_entry["status"] = "partial"
        else:
            m_entry["status"] = "failed"

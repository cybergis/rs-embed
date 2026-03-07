from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.embedding import Embedding
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..tools.output import normalize_embedding_output


@dataclass(frozen=True)
class CombinedPrefetchDeps:
    run_with_retry: Callable[..., Any]
    fetch_gee_patch_raw: Callable[..., np.ndarray]
    normalize_input_chw: Callable[..., np.ndarray]
    select_prefetched_channels: Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]
    inspect_input_raw: Callable[..., Dict[str, Any]]


def run_combined_prefetch_tasks(
    *,
    provider: Any,
    tasks: List[Tuple[int, str, SensorSpec]],
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    max_retries: int,
    retry_backoff_s: float,
    num_workers: int,
    continue_on_error: bool,
    fail_on_bad_input: bool,
    fetch_members: Dict[str, List[str]],
    sensor_to_fetch: Dict[str, Tuple[str, Tuple[int, ...]]],
    sensor_by_key: Dict[str, SensorSpec],
    sensor_models: Dict[str, List[str]],
    inputs_cache: Dict[Tuple[int, str], np.ndarray],
    input_reports: Dict[Tuple[int, str], Dict[str, Any]],
    prefetch_errors: Dict[Tuple[int, str], str],
    progress: Any,
    deps: CombinedPrefetchDeps,
) -> None:
    if not tasks:
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(i: int, skey: str, sspec: SensorSpec):
        x = deps.run_with_retry(
            lambda: deps.fetch_gee_patch_raw(
                provider, spatial=spatials[i], temporal=temporal, sensor=sspec
            ),
            retries=max_retries,
            backoff_s=retry_backoff_s,
        )
        return i, skey, x

    mw = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=mw) as ex:
        fut_map = {ex.submit(_fetch_one, i, sk, ss): (i, sk) for (i, sk, ss) in tasks}
        for fut in as_completed(fut_map):
            i, skey = fut_map[fut]
            try:
                i, skey, x = fut.result()
            except Exception as e:
                if not continue_on_error:
                    raise
                err_s = repr(e)
                for member_skey in fetch_members.get(skey, []):
                    prefetch_errors[(i, member_skey)] = err_s
            else:
                for member_skey in fetch_members.get(skey, []):
                    member_idx = sensor_to_fetch[member_skey][1]
                    x_member = deps.normalize_input_chw(
                        deps.select_prefetched_channels(x, member_idx),
                        expected_channels=len(member_idx),
                        name=f"gee_input_{member_skey}",
                    )
                    if fail_on_bad_input:
                        sspec_member = sensor_by_key[member_skey]
                        rep = deps.inspect_input_raw(
                            x_member,
                            sensor=sspec_member,
                            name=f"gee_input_{member_skey}",
                        )
                        if not bool(rep.get("ok", True)):
                            issues = (rep.get("report", {}) or {}).get("issues", [])
                            mlist = sorted(set(sensor_models.get(member_skey, [])))
                            err = RuntimeError(
                                f"Input inspection failed for index={i}, models={mlist}, sensor={member_skey}: {issues}"
                            )
                            if not continue_on_error:
                                raise err
                            prefetch_errors[(i, member_skey)] = repr(err)
                            continue
                        input_reports[(i, member_skey)] = rep
                    inputs_cache[(i, member_skey)] = x_member
            finally:
                progress.update(1)


def get_or_fetch_input(
    *,
    i: int,
    skey: str,
    sspec: SensorSpec,
    provider: Any,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    max_retries: int,
    retry_backoff_s: float,
    fail_on_bad_input: bool,
    inputs_cache: Dict[Tuple[int, str], np.ndarray],
    input_reports: Dict[Tuple[int, str], Dict[str, Any]],
    prefetch_errors: Dict[Tuple[int, str], str],
    deps: CombinedPrefetchDeps,
) -> np.ndarray:
    hit = inputs_cache.get((i, skey))
    if hit is not None:
        return hit
    pref_err = prefetch_errors.get((i, skey))
    if pref_err:
        raise RuntimeError(
            f"Prefetch previously failed for index={i}, sensor={skey}: {pref_err}"
        )
    if provider is None:
        raise RuntimeError(
            f"Missing provider for input fetch: index={i}, sensor={skey}"
        )
    x = deps.run_with_retry(
        lambda: deps.fetch_gee_patch_raw(
            provider, spatial=spatials[i], temporal=temporal, sensor=sspec
        ),
        retries=max_retries,
        backoff_s=retry_backoff_s,
    )
    rep = deps.inspect_input_raw(x, sensor=sspec, name=f"gee_input_{skey}")
    if fail_on_bad_input and (not bool(rep.get("ok", True))):
        issues = (rep.get("report", {}) or {}).get("issues", [])
        raise RuntimeError(
            f"Input inspection failed for index={i}, sensor={skey}: {issues}"
        )
    inputs_cache[(i, skey)] = x
    input_reports[(i, skey)] = rep
    return x


@dataclass(frozen=True)
class CombinedModelDeps:
    create_progress: Callable[..., Any]
    drop_model_arrays: Callable[[Dict[str, np.ndarray], str], None]
    jsonable: Callable[[Any], Any]
    sensor_key: Callable[[Optional[SensorSpec]], Tuple]
    normalize_model_name: Callable[[str], str]
    get_embedder_bundle_cached: Callable[[str, str, str, Tuple], Tuple[Any, Any]]
    sensor_cache_key: Callable[[SensorSpec], str]
    sanitize_key: Callable[[str], str]
    run_with_retry: Callable[..., Any]
    call_embedder_get_embedding: Callable[..., Embedding]
    supports_prefetched_batch_api: Callable[[Any], bool]
    supports_batch_api: Callable[[Any], bool]
    embedding_to_numpy: Callable[[Embedding], np.ndarray]


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
    deps: CombinedModelDeps,
) -> Dict[str, Any]:
    _resolved_backend = resolved_backend or {}
    for m in pending_models:
        deps.drop_model_arrays(arrays, m)
        infer_progress = deps.create_progress(
            enabled=bool(show_progress and save_embeddings),
            total=len(spatials),
            desc=f"infer[{m}]",
            unit="point",
        )
        infer_done: set[int] = set()
        infer_progress_done = 0
        m_entry: Dict[str, Any] = {
            "model": m,
            "sensor": deps.jsonable(resolved_sensor.get(m)),
            "status": "ok",
        }
        sspec = resolved_sensor.get(m)
        try:
            sensor_k = deps.sensor_key(sspec)
            m_backend = _resolved_backend.get(m, backend)
            embedder, lock = deps.get_embedder_bundle_cached(
                deps.normalize_model_name(m), m_backend, device, sensor_k
            )
            try:
                m_entry["describe"] = deps.jsonable(embedder.describe())
            except Exception as e:
                m_entry["describe"] = {"error": repr(e)}

            needs_provider_input = (
                provider_enabled
                and sspec is not None
                and "precomputed" not in (model_type.get(m) or "")
            )
            skey = (
                deps.sensor_cache_key(sspec)
                if needs_provider_input and sspec is not None
                else None
            )

            if save_inputs and needs_provider_input and skey is not None:
                if skey in input_refs_by_sensor:
                    m_entry["inputs"] = {
                        **input_refs_by_sensor[skey],
                        "dedup_reused": True,
                    }
                else:
                    xs = []
                    xs_indices: List[int] = []
                    missing = []
                    for i in range(len(spatials)):
                        try:
                            x = get_or_fetch_input_fn(i, skey, sspec)
                        except Exception as e:
                            missing.append((i, repr(e)))
                            continue
                        xs.append(np.asarray(x, dtype=np.float32))
                        xs_indices.append(int(i))
                    if missing and not continue_on_error:
                        raise RuntimeError(
                            f"Missing prefetched inputs for model={m}: {missing}"
                        )
                    if not xs:
                        m_entry["inputs"] = None
                    else:
                        try:
                            arr = np.stack(xs, axis=0)
                            in_key = f"inputs_bchw__{deps.sanitize_key(m)}"
                            arrays[in_key] = arr
                            ref = {
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
                                k = f"input_chw__{deps.sanitize_key(m)}__{i:05d}"
                                arrays[k] = np.asarray(x, dtype=np.float32)
                                keys.append(k)
                            ref = {"npz_keys": keys}
                            input_refs_by_sensor[skey] = dict(ref)
                            m_entry["inputs"] = ref
            else:
                m_entry["inputs"] = None

            if save_embeddings:
                n = len(spatials)
                infer_chunk = max(1, int(infer_batch_size))
                embs_by_idx: List[Optional[np.ndarray]] = [None] * n
                metas_by_idx: List[Optional[Dict[str, Any]]] = [None] * n
                errors_by_idx: Dict[int, str] = {}
                strategy = str(inference_strategy).strip().lower()
                # Keep historical combined-export behavior: auto still attempts batched paths.
                # Per-item layout now uses a stricter GPU-only auto preference in api.py.
                prefer_batch = (strategy == "batch") or (strategy == "auto")
                allow_batch = strategy != "single"

                def _mark_infer_done(i: int) -> None:
                    nonlocal infer_progress_done
                    if i in infer_done:
                        return
                    infer_done.add(i)
                    infer_progress_done += 1
                    infer_progress.update(1)

                def _infer_one(i: int) -> Embedding:
                    inp = None
                    if needs_provider_input and skey is not None and sspec is not None:
                        inp = get_or_fetch_input_fn(i, skey, sspec)
                    with lock:
                        return deps.call_embedder_get_embedding(
                            embedder=embedder,
                            spatial=spatials[i],
                            temporal=temporal,
                            sensor=sspec,
                            output=output,
                            backend=m_backend,
                            device=device,
                            input_chw=inp,
                        )

                can_batch_prefetched = (
                    allow_batch
                    and prefer_batch
                    and deps.supports_prefetched_batch_api(embedder)
                    and needs_provider_input
                    and skey is not None
                    and sspec is not None
                )
                can_batch = (
                    allow_batch
                    and prefer_batch
                    and deps.supports_batch_api(embedder)
                    and not needs_provider_input
                )
                batch_attempted = False
                batch_succeeded = False

                if can_batch_prefetched:
                    batch_attempted = True
                    try:
                        batch_indices: List[int] = []
                        batch_spatials: List[SpatialSpec] = []
                        batch_inputs: List[np.ndarray] = []
                        for i in range(n):
                            try:
                                inp = get_or_fetch_input_fn(i, skey, sspec)
                            except Exception as e:
                                if not continue_on_error:
                                    raise
                                errors_by_idx[i] = repr(e)
                                continue
                            batch_indices.append(i)
                            batch_spatials.append(spatials[i])
                            batch_inputs.append(np.asarray(inp, dtype=np.float32))

                        if batch_spatials:
                            for start in range(0, len(batch_spatials), infer_chunk):
                                sub_spatials = batch_spatials[
                                    start : start + infer_chunk
                                ]
                                sub_inputs = batch_inputs[start : start + infer_chunk]
                                sub_indices = batch_indices[start : start + infer_chunk]

                                def _infer_batch_prefetched():
                                    with lock:
                                        return (
                                            embedder.get_embeddings_batch_from_inputs(
                                                spatials=sub_spatials,
                                                input_chws=sub_inputs,
                                                temporal=temporal,
                                                sensor=sspec,
                                                output=output,
                                                backend=m_backend,
                                                device=device,
                                            )
                                        )

                                batch_out = deps.run_with_retry(
                                    _infer_batch_prefetched,
                                    retries=max_retries,
                                    backoff_s=retry_backoff_s,
                                )
                                if len(batch_out) != len(sub_spatials):
                                    raise RuntimeError(
                                        f"Model {m} returned {len(batch_out)} embeddings for "
                                        f"{len(sub_spatials)} prefetched inputs."
                                    )
                                for j, emb in enumerate(batch_out):
                                    emb = normalize_embedding_output(
                                        emb=emb, output=output
                                    )
                                    i = sub_indices[j]
                                    embs_by_idx[i] = deps.embedding_to_numpy(emb)
                                    metas_by_idx[i] = deps.jsonable(emb.meta)
                                    errors_by_idx.pop(i, None)
                                    _mark_infer_done(i)

                        batch_succeeded = True
                        if len(batch_indices) < n:
                            batch_set = set(batch_indices)
                            for i in range(n):
                                if i not in batch_set:
                                    _mark_infer_done(i)
                    except Exception:
                        # Fall back to per-item inference when batched path fails.
                        # This keeps combined export robust to model-specific batch quirks.
                        batch_succeeded = False

                if (not batch_attempted) and can_batch:
                    batch_attempted = True
                    try:
                        for start in range(0, n, infer_chunk):
                            sub_spatials = spatials[start : start + infer_chunk]

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

                            batch_out = deps.run_with_retry(
                                _infer_batch,
                                retries=max_retries,
                                backoff_s=retry_backoff_s,
                            )
                            if len(batch_out) != len(sub_spatials):
                                raise RuntimeError(
                                    f"Model {m} returned {len(batch_out)} embeddings for {len(sub_spatials)} inputs."
                                )
                            for j, emb in enumerate(batch_out):
                                emb = normalize_embedding_output(emb=emb, output=output)
                                i = start + j
                                embs_by_idx[i] = deps.embedding_to_numpy(emb)
                                metas_by_idx[i] = deps.jsonable(emb.meta)
                                errors_by_idx.pop(i, None)
                                _mark_infer_done(i)
                        batch_succeeded = True
                    except Exception:
                        # Fall back to per-item inference when batched path fails.
                        batch_succeeded = False

                if not batch_succeeded:
                    for i in range(n):
                        if i in infer_done:
                            continue
                        try:
                            emb = deps.run_with_retry(
                                lambda i=i: _infer_one(i),
                                retries=max_retries,
                                backoff_s=retry_backoff_s,
                            )
                            embs_by_idx[i] = deps.embedding_to_numpy(emb)
                            metas_by_idx[i] = deps.jsonable(emb.meta)
                            errors_by_idx.pop(i, None)
                        except Exception as e:
                            if not continue_on_error:
                                raise
                            errors_by_idx[i] = repr(e)
                        finally:
                            _mark_infer_done(i)

                ok_indices = [i for i, e in enumerate(embs_by_idx) if e is not None]
                if ok_indices:
                    try:
                        e_arr = np.stack([embs_by_idx[i] for i in ok_indices], axis=0)  # type: ignore[list-item]
                        if len(ok_indices) == n:
                            e_key = f"embeddings__{deps.sanitize_key(m)}"
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
                                k = f"embedding__{deps.sanitize_key(m)}__{i:05d}"
                                arrays[k] = e_arr[j]
                                keys.append(k)
                                index_map.append(i)
                            m_entry["embeddings"] = {
                                "npz_keys": keys,
                                "indices": index_map,
                            }
                    except Exception:
                        keys = []
                        index_map = []
                        for i in ok_indices:
                            k = f"embedding__{deps.sanitize_key(m)}__{i:05d}"
                            arrays[k] = embs_by_idx[i]  # type: ignore[index]
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
        manifest = write_checkpoint_fn(
            stage=f"model:{deps.sanitize_key(m)}", final=False
        )

    return manifest

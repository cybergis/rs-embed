from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig
from ..tools.serialization import (
    embedding_to_numpy,
    jsonable,
    sanitize_key,
    sensor_cache_key,
    sha1,
    utc_ts,
)
from ..tools.normalization import normalize_model_name
from ..tools.runtime import (
    call_embedder_get_embedding,
    get_embedder_bundle_cached,
    run_with_retry,
    sensor_key,
)
from ..providers.gee_utils import fetch_gee_patch_raw, inspect_input_raw


def build_one_point_payload(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    models: List[str],
    backend: str,
    resolved_backend: Optional[Dict[str, str]] = None,
    device: str,
    output: OutputSpec,
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
    inputs_cache: Dict[Tuple[int, str], np.ndarray],
    input_reports: Dict[Tuple[int, str], Dict[str, Any]],
    prefetch_errors: Dict[Tuple[int, str], str],
    pass_input_into_embedder: bool,
    config: ExportConfig,
    provider_factory: Optional[Callable[[], Any]] = None,
    model_progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    save_inputs = config.save_inputs
    save_embeddings = config.save_embeddings
    fail_on_bad_input = config.fail_on_bad_input
    continue_on_error = config.continue_on_error
    max_retries = config.max_retries
    retry_backoff_s = config.retry_backoff_s

    arrays: Dict[str, np.ndarray] = {}
    manifest: Dict[str, Any] = {
        "created_at": utc_ts(),
        "point_index": int(point_index),
        "status": "ok",
        "backend": backend,
        "device": device,
        "models": [],
        "spatial": jsonable(spatial),
        "temporal": jsonable(temporal),
        "output": jsonable(output),
    }

    try:
        from importlib.metadata import version

        manifest["package_version"] = version("rs-embed")
    except Exception:
        manifest["package_version"] = None

    local_inp: Dict[str, np.ndarray] = {}
    local_input_meta: Dict[str, Dict[str, Any]] = {}

    _resolved_backend = resolved_backend or {}

    for m in models:
        m_entry: Dict[str, Any] = {"model": m, "status": "ok"}
        sspec = resolved_sensor.get(m)
        m_entry["sensor"] = jsonable(sspec)

        try:
            sensor_k = sensor_key(sspec)
            m_backend = _resolved_backend.get(m, backend)
            embedder, lock = get_embedder_bundle_cached(
                normalize_model_name(m), m_backend, device, sensor_k
            )

            try:
                m_entry["describe"] = jsonable(embedder.describe())
            except Exception as e:
                m_entry["describe"] = {"error": repr(e)}

            input_chw: Optional[np.ndarray] = None
            report: Optional[Dict[str, Any]] = None
            provider_enabled = provider_factory is not None
            needs_provider_input = (
                provider_enabled
                and sspec is not None
                and "precomputed" not in (model_type.get(m) or "")
            )
            needs_input_for_embed = bool(
                pass_input_into_embedder and save_embeddings and needs_provider_input
            )
            needs_input_for_export = bool(save_inputs and needs_provider_input)
            if needs_input_for_embed or needs_input_for_export:
                skey = sensor_cache_key(sspec)
                if skey in local_inp:
                    input_chw = local_inp[skey]
                else:
                    cached = inputs_cache.get((point_index, skey))
                    if cached is not None:
                        input_chw = cached
                        local_inp[skey] = input_chw
                    else:
                        pref_err = prefetch_errors.get((point_index, skey))
                        if pref_err and continue_on_error:
                            raise RuntimeError(
                                f"Prefetch previously failed for model={m}, "
                                f"index={point_index}, sensor={skey}: {pref_err}"
                            )
                        if provider_factory is None:
                            raise RuntimeError(
                                f"Missing provider factory for model={m}, index={point_index}, sensor={skey}"
                            )
                        prov = provider_factory()
                        run_with_retry(
                            lambda: prov.ensure_ready(),
                            retries=max_retries,
                            backoff_s=retry_backoff_s,
                        )
                        input_chw = run_with_retry(
                            lambda: fetch_gee_patch_raw(
                                prov,
                                spatial=spatial,
                                temporal=temporal,
                                sensor=sspec,
                            ),
                            retries=max_retries,
                            backoff_s=retry_backoff_s,
                        )
                        local_inp[skey] = input_chw

                report = input_reports.get((point_index, skey))
                if report is None and input_chw is not None:
                    report = inspect_input_raw(
                        input_chw, sensor=sspec, name=f"gee_input_{skey}"
                    )

                if (
                    fail_on_bad_input
                    and report is not None
                    and (not bool(report.get("ok", True)))
                ):
                    issues = (report.get("report", {}) or {}).get("issues", [])
                    raise RuntimeError(
                        f"Input inspection failed for model={m}: {issues}"
                    )

                if save_inputs and input_chw is not None:
                    if skey in local_input_meta:
                        input_meta = dict(local_input_meta[skey])
                        input_meta["dedup_reused"] = True
                    else:
                        input_key = f"input_chw__{sanitize_key(m)}"
                        arrays[input_key] = np.asarray(input_chw, dtype=np.float32)
                        input_meta = {
                            "npz_key": input_key,
                            "dtype": str(arrays[input_key].dtype),
                            "shape": list(arrays[input_key].shape),
                            "sha1": sha1(arrays[input_key]),
                            "inspection": jsonable(report),
                        }
                        local_input_meta[skey] = dict(input_meta)
                    m_entry["input"] = input_meta
                else:
                    m_entry["input"] = None
            else:
                m_entry["input"] = None

            if save_embeddings:

                def _infer_once():
                    with lock:
                        return call_embedder_get_embedding(
                            embedder=embedder,
                            spatial=spatial,
                            temporal=temporal,
                            sensor=sspec,
                            output=output,
                            backend=m_backend,
                            device=device,
                            input_chw=(input_chw if pass_input_into_embedder else None),
                        )

                emb = run_with_retry(
                    _infer_once,
                    retries=max_retries,
                    backoff_s=retry_backoff_s,
                )
                e_np = embedding_to_numpy(emb)
                emb_key = f"embedding__{sanitize_key(m)}"
                arrays[emb_key] = e_np
                m_entry["embedding"] = {
                    "npz_key": emb_key,
                    "dtype": str(e_np.dtype),
                    "shape": list(e_np.shape),
                    "sha1": sha1(e_np),
                }
                m_entry["meta"] = jsonable(emb.meta)
            else:
                m_entry["embedding"] = None
                m_entry["meta"] = None
        except Exception as e:
            if not continue_on_error:
                raise
            m_entry["status"] = "failed"
            m_entry["error"] = repr(e)
            m_entry["input"] = m_entry.get("input")
            m_entry["embedding"] = None
            m_entry["meta"] = None
        finally:
            if model_progress_cb is not None:
                try:
                    model_progress_cb(m)
                except Exception:
                    pass

        manifest["models"].append(m_entry)

    n_failed = sum(1 for x in manifest["models"] if x.get("status") == "failed")
    if n_failed == 0:
        manifest["status"] = "ok"
    elif n_failed < len(manifest["models"]):
        manifest["status"] = "partial"
    else:
        manifest["status"] = "failed"
    manifest["summary"] = {
        "total_models": len(manifest["models"]),
        "failed_models": n_failed,
        "ok_models": len(manifest["models"]) - n_failed,
    }
    return arrays, manifest

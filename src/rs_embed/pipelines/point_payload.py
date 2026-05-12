"""Per-point payload assembly helpers.

This module builds arrays and manifest entries for one spatial point in
per-item export layout. It resolves provider input through a cache-first
fallback chain and records model-level success/failure metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig
from ..providers import fetch as _providers_fetch
from ..tools.manifest import summarize_status
from ..tools.normalization import normalize_model_name
from ..tools.runtime import (
    get_embedder_bundle_cached,
    run_with_retry,
    sensor_key,
)
from ..tools.serialization import (
    embedding_to_numpy,
    jsonable,
    sanitize_key,
    sensor_cache_key,
    sha1,
    utc_ts,
)
from ..tools.tiling import _call_embedder_get_embedding_with_input_prep


def build_one_point_payload(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    models: list[str],
    backend: str,
    resolved_backend: dict[str, str] | None = None,
    device: str,
    output: OutputSpec,
    resolved_sensor: dict[str, SensorSpec | None],
    resolved_model_config: dict[str, dict[str, Any] | None],
    model_type: dict[str, str],
    inputs_cache: dict[tuple[int, str], np.ndarray],
    input_reports: dict[tuple[int, str], dict[str, Any]],
    prefetch_errors: dict[tuple[int, str], str],
    pass_input_into_embedder: bool,
    config: ExportConfig,
    provider_factory: Callable[[], Any] | None = None,
    model_progress_cb: Callable[[str], None] | None = None,
    fetch_fn: Callable[..., np.ndarray] | None = None,
    inspect_fn: Callable[..., dict[str, Any]] | None = None,
    fetch_meta_cache: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build arrays + manifest payload for one point across all models.

    Input resolution order for provider-backed models:
    1. local per-point cache,
    2. prefetch cache,
    3. prefetch error map (if continuing on error),
    4. synchronous fetch via ``provider_factory`` fallback.
    """
    save_inputs = config.save_inputs
    save_embeddings = config.save_embeddings
    fail_on_bad_input = config.fail_on_bad_input
    continue_on_error = config.continue_on_error
    max_retries = config.max_retries
    retry_backoff_s = config.retry_backoff_s
    fetch = fetch_fn or _providers_fetch.fetch_sensor_patch_chw
    inspect = inspect_fn or _providers_fetch.inspect_fetch_result

    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
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
    except Exception as _e:
        manifest["package_version"] = None

    local_inp: dict[str, np.ndarray] = {}
    local_input_meta: dict[str, dict[str, Any]] = {}
    local_fetch_meta: dict[str, dict[str, Any]] = {}

    _resolved_backend = resolved_backend or {}

    for m in models:
        m_entry: dict[str, Any] = {"model": m, "status": "ok"}
        sspec = resolved_sensor.get(m)
        m_entry["sensor"] = jsonable(sspec)

        try:
            sensor_k = sensor_key(sspec)
            m_backend = _resolved_backend.get(m, backend)
            model_config = resolved_model_config.get(m)
            embedder, lock = get_embedder_bundle_cached(
                normalize_model_name(m), m_backend, device, sensor_k
            )

            try:
                m_entry["describe"] = jsonable(embedder.describe())
            except Exception as e:
                m_entry["describe"] = {"error": repr(e)}

            input_chw: np.ndarray | None = None
            report: dict[str, Any] | None = None
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
                            lambda _p=prov: _p.ensure_ready(),
                            retries=max_retries,
                            backoff_s=retry_backoff_s,
                        )
                        fetch_result = run_with_retry(
                            lambda _p=prov, _ss=sspec, _embedder=embedder: _embedder.fetch_input(
                                _p,
                                spatial=spatial,
                                temporal=temporal,
                                sensor=_ss,
                            ),
                            retries=max_retries,
                            backoff_s=retry_backoff_s,
                        )
                        if fetch_result is not None:
                            input_chw = np.asarray(fetch_result.data, dtype=np.float32)
                            if fetch_result.meta:
                                local_fetch_meta[skey] = jsonable(fetch_result.meta)
                        else:
                            input_chw = run_with_retry(
                                lambda _p=prov, _ss=sspec: fetch(
                                    _p,
                                    spatial=spatial,
                                    temporal=temporal,
                                    sensor=_ss,
                                ),
                                retries=max_retries,
                                backoff_s=retry_backoff_s,
                            )
                        local_inp[skey] = input_chw

                report = input_reports.get((point_index, skey))
                if report is None and input_chw is not None:
                    report = inspect(input_chw, sensor=sspec, name=f"gee_input_{skey}")

                if fail_on_bad_input and report is not None and (not bool(report.get("ok", True))):
                    issues = (report.get("report", {}) or {}).get("issues", [])
                    raise RuntimeError(f"Input inspection failed for model={m}: {issues}")

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

            # Resolve fetch-time metadata from the prefetch cache.
            _fmeta: dict[str, Any] | None = None
            if fetch_meta_cache and sspec is not None:
                _fmeta = fetch_meta_cache.get((point_index, sensor_cache_key(sspec))) or None
            if _fmeta is None and sspec is not None:
                _fmeta = local_fetch_meta.get(sensor_cache_key(sspec)) or None
            if _fmeta:
                m_entry["fetch_meta"] = jsonable(_fmeta)

            if save_embeddings:
                input_prep = getattr(config, "input_prep", "resize")

                def _infer_once(
                    _m_backend=m_backend,
                    _input_chw=input_chw,
                    _input_prep=input_prep,
                    _model_config=model_config,
                    _fmeta_cap=_fmeta,
                    _lock=lock,
                    _embedder=embedder,
                    _sspec=sspec,
                ):
                    with _lock:
                        return _call_embedder_get_embedding_with_input_prep(
                            embedder=_embedder,
                            spatial=spatial,
                            temporal=temporal,
                            sensor=_sspec,
                            output=output,
                            backend=_m_backend,
                            device=device,
                            input_chw=(_input_chw if pass_input_into_embedder else None),
                            input_prep=_input_prep,
                            model_config=_model_config,
                            fetch_meta=_fmeta_cap,
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
                except Exception as _e:
                    pass

        manifest["models"].append(m_entry)

    n_failed = sum(1 for x in manifest["models"] if x.get("status") == "failed")
    manifest["status"] = summarize_status(manifest["models"])
    manifest["summary"] = {
        "total_models": len(manifest["models"]),
        "failed_models": n_failed,
        "ok_models": len(manifest["models"]) - n_failed,
    }
    return arrays, manifest


def write_one_payload(
    *,
    out_path: str,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    save_manifest: bool,
    fmt: str,
    max_retries: int,
    retry_backoff_s: float,
) -> dict[str, Any]:
    """Persist one payload with retry and return writer metadata."""
    from ..writers import write_arrays

    return run_with_retry(
        lambda: write_arrays(
            fmt=fmt,
            out_path=out_path,
            arrays=arrays,
            manifest=jsonable(manifest),
            save_manifest=save_manifest,
        ),
        retries=max_retries,
        backoff_s=retry_backoff_s,
    )

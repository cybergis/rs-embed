from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec


def init_combined_export_state(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    models: List[str],
    out_path: str,
    fmt: str,
    resume: bool,
    load_json_dict: Callable[[str], Optional[Dict[str, Any]]],
    is_incomplete_combined_manifest: Callable[[Optional[Dict[str, Any]]], bool],
    load_saved_arrays: Callable[..., Dict[str, np.ndarray]],
    jsonable: Callable[[Any], Any],
    utc_ts: Callable[[], str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[str], str]:
    arrays: Dict[str, np.ndarray] = {}
    manifest: Dict[str, Any] = {
        "created_at": utc_ts(),
        "status": "running",
        "stage": "init",
        "resume_incomplete": True,
        "backend": backend,
        "device": device,
        "models": [],
        "n_items": len(spatials),
        "temporal": jsonable(temporal),
        "output": jsonable(output),
        "spatials": [jsonable(s) for s in spatials],
    }
    json_path = os.path.splitext(out_path)[0] + ".json"
    completed_models: Set[str] = set()

    if bool(resume) and os.path.exists(out_path):
        resume_manifest = load_json_dict(json_path)
        if is_incomplete_combined_manifest(resume_manifest):
            try:
                arrays = load_saved_arrays(fmt=fmt, out_path=out_path)
            except Exception:
                arrays = {}
            if resume_manifest is not None:
                manifest = dict(resume_manifest)
                old_models = manifest.get("models")
                kept_models: List[Dict[str, Any]] = []
                if isinstance(old_models, list):
                    for m in models:
                        for item in old_models:
                            if not isinstance(item, dict) or item.get("model") != m:
                                continue
                            status_s = str(item.get("status", "")).lower()
                            if status_s in ("ok", "partial"):
                                kept_models.append(item)
                                completed_models.add(m)
                            break
                manifest["models"] = kept_models

    manifest.setdefault("created_at", utc_ts())
    manifest["status"] = "running"
    manifest["stage"] = str(manifest.get("stage", "init"))
    manifest["resume_incomplete"] = True
    manifest["backend"] = backend
    manifest["device"] = device
    manifest["n_items"] = len(spatials)
    manifest["temporal"] = jsonable(temporal)
    manifest["output"] = jsonable(output)
    manifest["spatials"] = [jsonable(s) for s in spatials]
    if not isinstance(manifest.get("models"), list):
        manifest["models"] = []

    pending_models = [m for m in models if m not in completed_models]
    return arrays, manifest, pending_models, json_path


def collect_input_refs_by_sensor(
    *,
    manifest: Dict[str, Any],
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    sensor_cache_key: Callable[[SensorSpec], str],
) -> Dict[str, Dict[str, Any]]:
    input_refs_by_sensor: Dict[str, Dict[str, Any]] = {}
    for prev in manifest.get("models", []):
        if not isinstance(prev, dict):
            continue
        pm = prev.get("model")
        if not isinstance(pm, str):
            continue
        ps = resolved_sensor.get(pm)
        if ps is None:
            continue
        pref = prev.get("inputs")
        if isinstance(pref, dict):
            sk = sensor_cache_key(ps)
            clean_pref = dict(pref)
            clean_pref.pop("dedup_reused", None)
            input_refs_by_sensor.setdefault(sk, clean_pref)
    return input_refs_by_sensor


def summarize_combined_models(
    models_entries: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, int]]:
    n_failed = sum(1 for x in models_entries if x.get("status") == "failed")
    n_partial = sum(1 for x in models_entries if x.get("status") == "partial")
    if n_failed == 0 and n_partial == 0:
        status = "ok"
    elif n_failed < len(models_entries):
        status = "partial"
    else:
        status = "failed"
    summary = {
        "total_models": len(models_entries),
        "failed_models": n_failed,
        "partial_models": n_partial,
        "ok_models": len(models_entries) - n_failed - n_partial,
    }
    return status, summary

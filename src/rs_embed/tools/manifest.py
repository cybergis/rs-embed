from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from ..core.specs import OutputSpec, SpatialSpec, TemporalSpec
from .serialization import jsonable as _jsonable
from .serialization import utc_ts as _utc_ts


def export_request_fingerprint(
    *,
    models: list[Any],
    temporal: TemporalSpec | None,
    output: OutputSpec,
    config: Any,
    spatials: list[SpatialSpec] | None = None,
    spatial: SpatialSpec | None = None,
) -> str:
    """Hash of the request fields that determine an export file's contents.

    Stored in every export manifest as ``request_fingerprint`` and compared on
    resume: an existing output only counts as "already done" for the same
    request. Covers resolved per-model configuration (name, backend, sensor,
    model_config), temporal, output, and the content-affecting config flags —
    not runtime knobs (device, workers, retries, progress) that leave the
    output unchanged. Pass ``spatials`` for a combined export or the single
    item's ``spatial`` for a per-item file, so per-item resume survives
    appending points but not reordering them.
    """
    payload: dict[str, Any] = {
        "models": [
            {
                "name": mc.name,
                "backend": mc.backend,
                "sensor": _jsonable(mc.sensor),
                "model_config": _jsonable(mc.model_config),
            }
            for mc in models
        ],
        "temporal": _jsonable(temporal),
        "output": _jsonable(output),
        "format": str(config.format),
        "save_inputs": bool(config.save_inputs),
        "save_embeddings": bool(config.save_embeddings),
        "input_prep": _jsonable(config.input_prep),
    }
    if spatials is not None:
        payload["spatials"] = [_jsonable(s) for s in spatials]
    if spatial is not None:
        payload["spatial"] = _jsonable(spatial)
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def load_json_dict(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception as _e:
        return None
    return None


def _resume_manifest(
    *,
    out_file: str,
    backend: str,
    device: str,
    temporal: TemporalSpec | None,
    output: OutputSpec,
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    """Build a resume-skipped manifest, loading existing JSON if available."""
    json_path = os.path.splitext(out_file)[0] + ".json"
    manifest = load_json_dict(json_path)
    if manifest is None:
        manifest = {
            "created_at": _utc_ts(),
            "status": "skipped",
            "stage": "resume",
            "reason": "output_exists",
            "backend": backend,
            "device": device,
            "models": [],
            "temporal": _jsonable(temporal),
            "output": _jsonable(output),
            **extra_fields,
        }
    manifest["resume_skipped"] = True
    manifest["resume_output_path"] = out_file
    manifest.setdefault("status", "ok")
    return manifest


def point_resume_manifest(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    output: OutputSpec,
    backend: str,
    device: str,
    out_file: str,
) -> dict[str, Any]:
    manifest = _resume_manifest(
        out_file=out_file,
        backend=backend,
        device=device,
        temporal=temporal,
        output=output,
        extra_fields={
            "point_index": int(point_index),
            "spatial": _jsonable(spatial),
        },
    )
    manifest.setdefault("point_index", int(point_index))
    return manifest


def combined_resume_manifest(
    *,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None,
    output: OutputSpec,
    backend: str,
    device: str,
    out_file: str,
) -> dict[str, Any]:
    return _resume_manifest(
        out_file=out_file,
        backend=backend,
        device=device,
        temporal=temporal,
        output=output,
        extra_fields={
            "n_items": len(spatials),
            "spatials": [_jsonable(s) for s in spatials],
        },
    )


def point_failure_manifest(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    output: OutputSpec,
    backend: str,
    device: str,
    stage: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "created_at": _utc_ts(),
        "point_index": int(point_index),
        "status": "failed",
        "stage": stage,
        "error": repr(error),
        "backend": backend,
        "device": device,
        "models": [],
        "spatial": _jsonable(spatial),
        "temporal": _jsonable(temporal),
        "output": _jsonable(output),
    }


def summarize_status(entries: list[dict[str, Any]]) -> str:
    """Summarize a list of model/status entries into ok/partial/failed."""
    if not entries:
        return "ok"
    n_failed = sum(
        1
        for item in entries
        if isinstance(item, dict) and str(item.get("status", "")).lower() == "failed"
    )
    has_partial = any(
        isinstance(item, dict) and str(item.get("status", "")).lower() == "partial"
        for item in entries
    )
    if n_failed == 0 and not has_partial:
        return "ok"
    if n_failed < len(entries):
        return "partial"
    return "failed"

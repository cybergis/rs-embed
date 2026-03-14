from __future__ import annotations

import json
import os
from typing import Any

from .serialization import jsonable as _jsonable
from .serialization import utc_ts as _utc_ts
from ..core.specs import OutputSpec, SpatialSpec, TemporalSpec

def load_json_dict(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
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

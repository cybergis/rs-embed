"""Checkpoint and manifest management for batch exports.

This module encapsulates resume detection, manifest shaping, and checkpoint
write/load operations for per-item and combined layouts.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..core.specs import OutputSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig, ExportTarget
from ..tools.checkpoint_utils import (
    is_incomplete_combined_manifest,
    load_saved_arrays,
)
from ..tools.manifest import load_json_dict as _load_json_dict
from ..tools.serialization import jsonable, utc_ts
from ..writers import write_arrays
from .runner import run_with_retry


class CheckpointManager:
    """Manages manifests, resume detection, and checkpoint writes.

    Parameters
    ----------
    target : ExportTarget
        Output location and layout.
    config : ExportConfig
        Behavioral flags.
    """

    def __init__(self, target: ExportTarget, config: ExportConfig) -> None:
        self.target = target
        self.config = config

    # ── per-item resume ────────────────────────────────────────────

    def per_item_should_skip(self, out_file: str) -> bool:
        return bool(self.config.resume) and os.path.exists(out_file)

    # ── combined resume ────────────────────────────────────────────

    def combined_init_state(
        self,
        *,
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None,
        output: OutputSpec,
        backend: str,
        device: str,
        models: list[str],
        out_path: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str], str]:
        """Initialize combined export state, handling resume if applicable."""
        fmt = self.config.format
        resume = self.config.resume

        arrays: dict[str, np.ndarray] = {}
        manifest: dict[str, Any] = {
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
        completed_models: set[str] = set()

        if bool(resume) and os.path.exists(out_path):
            resume_manifest = _load_json_dict(json_path)
            if is_incomplete_combined_manifest(resume_manifest):
                try:
                    arrays = load_saved_arrays(fmt=fmt, out_path=out_path)
                except Exception as _e:
                    arrays = {}
                if resume_manifest is not None:
                    manifest = dict(resume_manifest)
                    old_models = manifest.get("models")
                    kept: list[dict[str, Any]] = []
                    if isinstance(old_models, list):
                        for m in models:
                            for item in old_models:
                                if not isinstance(item, dict) or item.get("model") != m:
                                    continue
                                if str(item.get("status", "")).lower() in (
                                    "ok",
                                    "partial",
                                ):
                                    kept.append(item)
                                    completed_models.add(m)
                                break
                    manifest["models"] = kept

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

        pending = [m for m in models if m not in completed_models]
        return arrays, manifest, pending, json_path

    def combined_write_checkpoint(
        self,
        *,
        manifest: dict[str, Any],
        arrays: dict[str, np.ndarray],
        stage: str,
        final: bool,
        out_path: str,
        json_path: str,
    ) -> dict[str, Any]:
        """Write a combined checkpoint to disk."""
        cfg = self.config
        manifest["stage"] = stage
        manifest["resume_incomplete"] = not final
        if not final:
            manifest["status"] = "running"
        if final and not cfg.save_manifest:
            manifest.pop("manifest_path", None)

        written = run_with_retry(
            lambda: write_arrays(
                fmt=cfg.format,
                out_path=out_path,
                arrays=arrays,
                manifest=jsonable(manifest),
                save_manifest=cfg.save_manifest if final else True,
            ),
            retries=cfg.max_retries,
            backoff_s=cfg.retry_backoff_s,
        )
        if final and not cfg.save_manifest:
            try:
                if os.path.exists(json_path):
                    os.remove(json_path)
            except Exception as _e:
                pass
        return written


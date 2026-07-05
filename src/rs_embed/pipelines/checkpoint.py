"""Checkpoint and manifest management for batch exports.

This module encapsulates resume detection, manifest shaping, and checkpoint
write/load operations for per-item and combined layouts.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig, ExportTarget
from ..tools.checkpoint_utils import (
    drop_model_arrays,
    drop_prefetch_checkpoint_arrays,
    is_incomplete_combined_manifest,
    load_saved_arrays,
    restore_prefetch_checkpoint_cache,
    store_prefetch_checkpoint_arrays,
)
from ..tools.manifest import (
    load_json_dict as _load_json_dict,
)
from ..tools.manifest import (
    summarize_status,
)
from ..tools.serialization import jsonable, sanitize_key, sensor_cache_key, utc_ts
from ..writers import write_arrays
from .runner import run_with_retry


def _array_refs_present(ref: Any, arrays: dict[str, np.ndarray]) -> bool:
    """Whether every array a manifest entry references actually exists.

    Entries reference arrays as ``{"npz_key": ...}`` or ``{"npz_keys": [...]}``;
    ``None`` (nothing saved) references nothing and is trivially present.
    """
    if not isinstance(ref, dict):
        return True
    if "npz_key" in ref:
        return str(ref["npz_key"]) in arrays
    if "npz_keys" in ref:
        keys = ref["npz_keys"]
        return isinstance(keys, list) and all(str(k) in arrays for k in keys)
    return True


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

    def per_item_should_skip(self, out_file: str, *, fingerprint: str | None = None) -> bool:
        """Skip a point only when its existing output matches the current request.

        The sidecar manifest's ``request_fingerprint`` is compared against
        *fingerprint*; an output from a different request (or with a missing/
        unreadable sidecar) is recomputed rather than trusted. When manifests
        are disabled (``save_manifest=False``) there is nothing to verify
        against, so existence alone decides — the caller opted out.
        """
        if not (bool(self.config.resume) and os.path.exists(out_file)):
            return False
        if fingerprint is None or not self.config.save_manifest:
            return True
        sidecar = _load_json_dict(os.path.splitext(out_file)[0] + ".json")
        return bool(sidecar) and sidecar.get("request_fingerprint") == fingerprint

    def per_item_resume_manifest(
        self,
        *,
        point_index: int,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        output: OutputSpec,
        backend: str,
        device: str,
        out_file: str,
    ) -> dict[str, Any]:
        from ..tools.manifest import point_resume_manifest

        return point_resume_manifest(
            point_index=point_index,
            spatial=spatial,
            temporal=temporal,
            output=output,
            backend=backend,
            device=device,
            out_file=out_file,
        )

    def per_item_failure_manifest(
        self,
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
        from ..tools.manifest import point_failure_manifest

        return point_failure_manifest(
            point_index=point_index,
            spatial=spatial,
            temporal=temporal,
            output=output,
            backend=backend,
            device=device,
            stage=stage,
            error=error,
        )

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
        fingerprint: str | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str], str]:
        """Initialize combined export state, handling resume if applicable.

        A prior checkpoint is only resumed when its ``request_fingerprint``
        matches *fingerprint* — otherwise its results belong to a different
        request and splicing them in would mislabel old embeddings as the new
        spatials. A model entry is only kept as completed when its status is
        ``ok`` (``partial`` re-runs so failed points get retried) and every
        array it references actually loaded; stale arrays of models no longer
        in the request are dropped.
        """
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
            fingerprint_ok = (
                resume_manifest is not None
                and (fingerprint is None or resume_manifest.get("request_fingerprint") == fingerprint)
            )
            if is_incomplete_combined_manifest(resume_manifest) and not fingerprint_ok:
                warnings.warn(
                    f"Checkpoint at '{out_path}' was written by a different export request; "
                    "ignoring it and starting fresh.",
                    stacklevel=4,
                )
            if is_incomplete_combined_manifest(resume_manifest) and fingerprint_ok:
                try:
                    arrays = load_saved_arrays(fmt=fmt, out_path=out_path)
                except Exception as _e:
                    warnings.warn(
                        f"Could not load checkpoint arrays from '{out_path}': {_e}. "
                        "Starting with empty array cache; all models will be re-run.",
                        stacklevel=4,
                    )
                    arrays = {}
                manifest = dict(resume_manifest)
                old_models = manifest.get("models")
                kept: list[dict[str, Any]] = []
                if arrays and isinstance(old_models, list):
                    current = set(models)
                    for item in old_models:
                        name = item.get("model") if isinstance(item, dict) else None
                        if isinstance(name, str) and name not in current:
                            drop_model_arrays(arrays, name, sanitize_key=sanitize_key)
                    for m in models:
                        for item in old_models:
                            if not isinstance(item, dict) or item.get("model") != m:
                                continue
                            if (
                                str(item.get("status", "")).lower() == "ok"
                                and _array_refs_present(item.get("embeddings"), arrays)
                                and _array_refs_present(item.get("inputs"), arrays)
                            ):
                                kept.append(item)
                                completed_models.add(m)
                            break
                manifest["models"] = kept

        if fingerprint is not None:
            manifest["request_fingerprint"] = fingerprint
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
        # write_arrays operates on a jsonable copy; merge its path keys back
        # and return the original dict so callers that keep mutating the
        # manifest (e.g. appending model entries between checkpoints) and
        # closures holding a reference to it stay in sync.
        for key in ("manifest_path", "npz_path", "npz_keys", "nc_path", "nc_variables"):
            if key in written:
                manifest[key] = written[key]
        if final and not cfg.save_manifest:
            try:
                if os.path.exists(json_path):
                    os.remove(json_path)
            except Exception as _e:
                pass
        return manifest

    # ── combined helpers ───────────────────────────────────────────

    @staticmethod
    def restore_prefetch_cache(
        manifest: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> dict[tuple[int, str], np.ndarray]:
        cache: dict[tuple[int, str], np.ndarray] = {}
        prefetch_meta = manifest.get("prefetch")
        if isinstance(prefetch_meta, dict):
            cache.update(
                restore_prefetch_checkpoint_cache(arrays=arrays, prefetch_meta=prefetch_meta)
            )
        return cache

    @staticmethod
    def store_prefetch_arrays(
        *,
        arrays: dict[str, np.ndarray],
        manifest: dict[str, Any],
        sensor_by_key: dict[str, SensorSpec],
        inputs_cache: dict[tuple[int, str], np.ndarray],
        n_items: int,
    ) -> None:
        store_prefetch_checkpoint_arrays(
            arrays=arrays,
            manifest=manifest,
            sensor_by_key=sensor_by_key,
            inputs_cache=inputs_cache,
            n_items=n_items,
        )

    @staticmethod
    def drop_prefetch_arrays(arrays: dict[str, np.ndarray]) -> None:
        drop_prefetch_checkpoint_arrays(arrays)

    @staticmethod
    def collect_input_refs(
        manifest: dict[str, Any],
        resolved_sensor: dict[str, SensorSpec | None],
    ) -> dict[str, dict[str, Any]]:
        refs: dict[str, dict[str, Any]] = {}
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
                clean = dict(pref)
                clean.pop("dedup_reused", None)
                refs.setdefault(sk, clean)
        return refs

    @staticmethod
    def summarize_models(
        entries: list[dict[str, Any]],
    ) -> tuple[str, dict[str, int]]:
        n_failed = sum(1 for x in entries if x.get("status") == "failed")
        n_partial = sum(1 for x in entries if x.get("status") == "partial")
        status = summarize_status(entries)
        return status, {
            "total_models": len(entries),
            "failed_models": n_failed,
            "partial_models": n_partial,
            "ok_models": len(entries) - n_failed - n_partial,
        }

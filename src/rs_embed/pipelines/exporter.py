"""Batch export orchestrator.

This module owns the top-level export lifecycle for both layouts:
prefetch -> payload/inference -> write -> checkpoint/resume bookkeeping.
It composes :class:`PrefetchManager`, :class:`InferenceEngine`, and
:class:`CheckpointManager`, while keeping runtime state on ``self``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..tools.serialization import (
    sanitize_key,
    sha1,
    utc_ts,
)
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig, ExportLayout, ExportTarget, ModelConfig
from .point_payload import build_one_point_payload, write_one_payload
from ..tools.manifest import (
    point_failure_manifest,
    point_resume_manifest,
    summarize_status,
)
from ..tools.progress import create_progress as _default_create_progress
from ..providers import gee_utils as _gee_utils
from ..writers import get_extension
from .checkpoint import CheckpointManager
from .inference import InferenceEngine
from .prefetch import PrefetchManager
from .runner import run_with_retry


class BatchExporter:
    """Orchestrates batch embedding export (per-item or combined).

    The exporter delegates focused responsibilities to pipeline managers,
    while retaining layout-level control flow and writer coordination.
    All run-scoped state is stored on ``self``.

    Parameters
    ----------
    spatials : list of SpatialSpec
        Spatial locations to export.
    temporal : TemporalSpec or None
        Temporal window.
    models : list of ModelConfig
        Resolved model configurations.
    target : ExportTarget
        Output target (dir or file).
    output : OutputSpec
        Embedding output spec (pooled/grid).
    config : ExportConfig
        Behavioral flags.
    backend : str
        Canonical backend name.
    resolved_backend : dict
        Per-model resolved backends.
    device : str
        Target device.
    provider_factory : callable or None
        Factory for the data provider (e.g. GEEProvider).
    """

    def __init__(
        self,
        *,
        spatials: List[SpatialSpec],
        temporal: Optional[TemporalSpec],
        models: List[ModelConfig],
        target: ExportTarget,
        output: OutputSpec,
        config: ExportConfig,
        backend: str,
        resolved_backend: Dict[str, str],
        device: str,
        provider_factory: Optional[Callable[[], Any]] = None,
        fetch_fn: Optional[Callable[..., np.ndarray]] = None,
        inspect_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        progress_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.spatials = spatials
        self.temporal = temporal
        self.models = models
        self.target = target
        self.output = output
        self.config = config
        self.backend = backend
        self.resolved_backend = resolved_backend
        self.device = device
        self.provider_factory = provider_factory
        self.fetch_fn = fetch_fn or _gee_utils.fetch_gee_patch_raw
        self.inspect_fn = inspect_fn or _gee_utils.inspect_input_raw
        self.create_progress = progress_factory or _default_create_progress

        # Model name lists for convenience
        self.model_names = [mc.name for mc in models]
        self.resolved_sensor: Dict[str, Optional[SensorSpec]] = {
            mc.name: mc.sensor for mc in models
        }
        self.model_type: Dict[str, str] = {mc.name: mc.model_type for mc in models}

        # Derived extension
        self.ext = get_extension(config.format)

        # Compose engine sub-systems
        self.inference = InferenceEngine(device=device, output=output, config=config)
        self.checkpoint = CheckpointManager(target=target, config=config)

    # ── public entry point ─────────────────────────────────────────

    def run(self) -> Any:
        """Execute the export, dispatching by layout."""
        if self.target.layout == ExportLayout.COMBINED:
            return self._run_combined()
        return self._run_per_item()

    # ── per-item export ────────────────────────────────────────────

    def _run_per_item(self) -> List[Dict[str, Any]]:
        """Per-item export: one file per spatial location.

        Chunked pipeline: ``prefetch → infer → write`` with a 1-slot
        pipelined prefetch (overlap fetch of chunk N+1 with
        processing of chunk N).
        """
        cfg = self.config
        out_dir = self.target.out_dir
        names = self.target.names
        assert out_dir is not None and names is not None

        os.makedirs(out_dir, exist_ok=True)

        n = len(self.spatials)
        progress = self.create_progress(
            enabled=bool(cfg.show_progress), total=n, desc="export_batch", unit="point"
        )

        prefetch, provider_enabled = self._setup_prefetch()
        need_prefetch = prefetch.enabled and bool(cfg.save_inputs or cfg.save_embeddings)
        pending_idxs, manifests = self._build_pending_queue(progress=progress)

        if not pending_idxs:
            manifests.sort(key=lambda x: int(x.get("point_index", -1)))
            progress.close()
            return manifests

        model_progress, on_model_done = self._create_model_progress(total=len(pending_idxs))

        # Chunk pipeline
        csize = cfg.effective_chunk_size
        chunk_groups = [
            pending_idxs[s : s + csize] for s in range(0, len(pending_idxs), csize)
        ]

        prefetch_pipeline_ex: Optional[ThreadPoolExecutor] = None
        prefetched_chunk_fut = None
        try:
            if need_prefetch and chunk_groups:
                prefetch_pipeline_ex = ThreadPoolExecutor(max_workers=1)
                prefetched_chunk_fut = prefetch_pipeline_ex.submit(
                    self._prefetch_chunk, prefetch, chunk_groups[0]
                )

            for chunk_idx, idxs in enumerate(chunk_groups):
                # Wait for prefetched data
                if prefetched_chunk_fut is not None:
                    prefetched_chunk_fut.result()
                    prefetched_chunk_fut = None

                # Kick off next chunk prefetch
                if prefetch_pipeline_ex is not None and (chunk_idx + 1) < len(
                    chunk_groups
                ):
                    # Clone a fresh prefetch manager for next chunk to avoid
                    # cache collisions — the current chunk's data is still in use.
                    next_prefetch = self._clone_prefetch(prefetch)
                    prefetched_chunk_fut = prefetch_pipeline_ex.submit(
                        self._prefetch_chunk, next_prefetch, chunk_groups[chunk_idx + 1]
                    )

                # Batch inference (GPU path)
                chunk_embed_results: Dict[Tuple[int, str], Any] = {}
                use_batch = bool(cfg.save_embeddings and self.inference.prefer_batch)
                if use_batch:
                    chunk_embed_results = self.inference.infer_chunk(
                        idxs=idxs,
                        spatials=self.spatials,
                        temporal=self.temporal,
                        models=self.models,
                        prefetch_cache=prefetch.cache,
                        prefetch_errors=prefetch.errors,
                        model_progress_cb=on_model_done,
                    )

                # Build + write each point
                self._write_per_item_chunk(
                    idxs=idxs,
                    prefetch=prefetch,
                    provider_enabled=provider_enabled,
                    chunk_embed_results=chunk_embed_results,
                    use_batch=use_batch,
                    manifests=manifests,
                    progress=progress,
                    model_progress_cb=on_model_done,
                )

                # Free chunk memory
                prefetch.clear_chunk()

        finally:
            if prefetch_pipeline_ex is not None:
                prefetch_pipeline_ex.shutdown(wait=True)

        manifests.sort(key=lambda x: int(x.get("point_index", -1)))
        for bar in model_progress.values():
            bar.close()
        progress.close()
        return manifests

    # ── combined export ────────────────────────────────────────────

    def _run_combined(self) -> Dict[str, Any]:
        """Combined export: all points in a single file.

        Prefetches all inputs → checkpoints → runs inference per-model →
        writes final combined file.
        """
        cfg = self.config
        out_file = self.target.out_file
        assert out_file is not None

        # Initialize combined state (with resume support)
        arrays, manifest, pending_models, json_path = (
            self.checkpoint.combined_init_state(
                spatials=self.spatials,
                temporal=self.temporal,
                output=self.output,
                backend=self.backend,
                device=self.device,
                models=self.model_names,
                out_path=out_file,
            )
        )

        prefetch, _provider_enabled = self._setup_prefetch()

        # Restore prefetch cache from checkpoint
        restored = self.checkpoint.restore_prefetch_cache(manifest, arrays)
        prefetch.cache.update(restored)

        # Build prefetch tasks
        all_idxs = list(range(len(self.spatials)))
        tasks = prefetch.build_tasks(all_idxs, self.spatials)

        progress = self.create_progress(
            enabled=bool(cfg.show_progress),
            total=(len(tasks) + len(pending_models)),
            desc="export_batch[combined]",
            unit="step",
        )

        # Prefetch all inputs
        if prefetch.provider is not None and tasks:
            prefetch.fetch_chunk(
                all_idxs, self.spatials, self.temporal, progress=progress
            )

        # Store prefetch checkpoint
        if prefetch.provider is not None:
            self.checkpoint.store_prefetch_arrays(
                arrays=arrays,
                manifest=manifest,
                sensor_by_key=prefetch.sensor_by_key,
                inputs_cache=prefetch.cache,
                n_items=len(self.spatials),
            )
            manifest = self.checkpoint.combined_write_checkpoint(
                manifest=manifest,
                arrays=arrays,
                stage="prefetched",
                final=False,
                out_path=out_file,
                json_path=json_path,
            )

        # Collect input refs from previously completed models
        input_refs_by_sensor = self.checkpoint.collect_input_refs(
            manifest, self.resolved_sensor
        )

        # Run pending models
        from .combined_flow import run_pending_models

        def _get_or_fetch(i: int, skey: str, sspec: SensorSpec) -> np.ndarray:
            return prefetch.get_or_fetch(
                idx=i,
                skey=skey,
                sspec=sspec,
                spatial=self.spatials[i],
                temporal=self.temporal,
            )

        def _write_ckpt(*, stage: str, final: bool = False) -> Dict[str, Any]:
            return self.checkpoint.combined_write_checkpoint(
                manifest=manifest,
                arrays=arrays,
                stage=stage,
                final=final,
                out_path=out_file,
                json_path=json_path,
            )

        manifest = run_pending_models(
            pending_models=pending_models,
            arrays=arrays,
            manifest=manifest,
            spatials=self.spatials,
            temporal=self.temporal,
            output=self.output,
            resolved_sensor=self.resolved_sensor,
            model_type=self.model_type,
            backend=self.backend,
            resolved_backend=self.resolved_backend,
            provider_enabled=prefetch.enabled,
            device=self.device,
            save_inputs=cfg.save_inputs,
            save_embeddings=cfg.save_embeddings,
            continue_on_error=cfg.continue_on_error,
            chunk_size=cfg.effective_chunk_size,
            inference_strategy="auto",
            infer_batch_size=cfg.effective_infer_batch_size,
            max_retries=cfg.max_retries,
            retry_backoff_s=cfg.retry_backoff_s,
            show_progress=cfg.show_progress,
            input_refs_by_sensor=input_refs_by_sensor,
            get_or_fetch_input_fn=_get_or_fetch,
            write_checkpoint_fn=_write_ckpt,
            progress=progress,
            inference_engine=self.inference,
            progress_factory=self.create_progress,
        )

        # Drop prefetch checkpoint arrays before final write
        self.checkpoint.drop_prefetch_arrays(arrays)

        # Summarize
        model_entries = manifest.get("models", [])
        status_str, summary = self.checkpoint.summarize_models(model_entries)
        manifest["status"] = status_str
        manifest["summary"] = summary
        manifest["completed_at"] = utc_ts()

        # Final write
        manifest = _write_ckpt(stage="done", final=True)
        progress.close()
        return manifest

    # ── internal helpers ───────────────────────────────────────────

    def _init_provider(self) -> Any:
        """Create and ready the provider, if applicable."""
        cfg = self.config
        if self.provider_factory is None:
            return None
        if not (cfg.save_inputs or cfg.save_embeddings):
            return None
        provider = self.provider_factory()
        run_with_retry(
            lambda: provider.ensure_ready(),
            retries=cfg.max_retries,
            backoff_s=cfg.retry_backoff_s,
        )
        return provider

    def _setup_prefetch(self) -> Tuple[PrefetchManager, bool]:
        """Create and plan a PrefetchManager plus provider-enabled flag."""
        provider = self._init_provider()
        prefetch = PrefetchManager(
            provider=provider,
            models=self.model_names,
            resolved_sensor=self.resolved_sensor,
            model_type=self.model_type,
            config=self.config,
            fetch_fn=self.fetch_fn,
            inspect_fn=self.inspect_fn,
        )
        band_resolver = (
            getattr(provider, "normalize_bands", None) if provider is not None else None
        )
        prefetch.plan(resolve_bands_fn=band_resolver)
        return prefetch, prefetch.enabled

    def _build_pending_queue(self, *, progress: Any) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Split per-item indices into pending work and resume manifests."""
        out_dir = self.target.out_dir
        names = self.target.names
        assert out_dir is not None and names is not None

        pending_idxs: List[int] = []
        manifests: List[Dict[str, Any]] = []
        for i in range(len(self.spatials)):
            out_file = os.path.join(out_dir, f"{names[i]}{self.ext}")
            if self.checkpoint.per_item_should_skip(out_file):
                manifests.append(
                    point_resume_manifest(
                        point_index=i,
                        spatial=self.spatials[i],
                        temporal=self.temporal,
                        output=self.output,
                        backend=self.backend,
                        device=self.device,
                        out_file=out_file,
                    )
                )
                progress.update(1)
            else:
                pending_idxs.append(i)
        return pending_idxs, manifests

    def _create_model_progress(
        self, total: int
    ) -> Tuple[Dict[str, Any], Optional[Callable[[str], None]]]:
        """Create per-model progress bars and callback if embedding export is enabled."""
        if not self.config.save_embeddings:
            return {}, None

        model_progress = {
            m: self.create_progress(
                enabled=bool(self.config.show_progress),
                total=total,
                desc=f"infer[{m}]",
                unit="point",
            )
            for m in self.model_names
        }

        def _on_model_done(model_name: str) -> None:
            bar = model_progress.get(model_name)
            if bar is not None:
                bar.update(1)

        return model_progress, _on_model_done

    def _prefetch_chunk(self, prefetch: PrefetchManager, idxs: List[int]) -> None:
        """Prefetch a chunk of inputs (for use in pipelined prefetch)."""
        prefetch.fetch_chunk(idxs, self.spatials, self.temporal)

    def _clone_prefetch(self, src: PrefetchManager) -> PrefetchManager:
        """Clone a PrefetchManager preserving the plan but with the same cache refs."""
        clone = PrefetchManager(
            provider=src.provider,
            models=self.model_names,
            resolved_sensor=self.resolved_sensor,
            model_type=self.model_type,
            config=self.config,
            fetch_fn=self.fetch_fn,
            inspect_fn=self.inspect_fn,
        )
        clone.sensor_by_key = src.sensor_by_key
        clone.fetch_sensor_by_key = src.fetch_sensor_by_key
        clone.sensor_to_fetch = src.sensor_to_fetch
        clone.sensor_models = src.sensor_models
        clone.fetch_members = src.fetch_members
        # Share cache references so piped prefetch populates the same dict
        clone.cache = src.cache
        clone.errors = src.errors
        clone.input_reports = src.input_reports
        return clone

    def _write_per_item_chunk(
        self,
        *,
        idxs: List[int],
        prefetch: PrefetchManager,
        provider_enabled: bool,
        chunk_embed_results: Dict[Tuple[int, str], Any],
        use_batch: bool,
        manifests: List[Dict[str, Any]],
        progress: Any,
        model_progress_cb: Optional[Callable[[str], None]],
    ) -> None:
        """Build payloads and write each point in a chunk."""
        cfg = self.config
        out_dir = self.target.out_dir
        names = self.target.names
        assert out_dir is not None and names is not None

        writer_async = bool(cfg.async_write)
        writer_mw = max(1, int(cfg.writer_workers))
        write_futs: List[Tuple[int, Any]] = []
        writer_ex: Optional[ThreadPoolExecutor] = None
        if writer_async:
            writer_ex = ThreadPoolExecutor(max_workers=writer_mw)

        try:
            for i in idxs:
                out_file = os.path.join(out_dir, f"{names[i]}{self.ext}")
                try:
                    per_item_cfg = (
                        replace(cfg, save_embeddings=False) if use_batch else cfg
                    )
                    arrays, manifest = build_one_point_payload(
                        point_index=i,
                        spatial=self.spatials[i],
                        temporal=self.temporal,
                        models=self.model_names,
                        backend=self.backend,
                        resolved_backend=self.resolved_backend,
                        device=self.device,
                        output=self.output,
                        resolved_sensor=self.resolved_sensor,
                        model_type=self.model_type,
                        inputs_cache=prefetch.cache,
                        input_reports=prefetch.input_reports,
                        prefetch_errors=prefetch.errors,
                        pass_input_into_embedder=provider_enabled
                        and bool(cfg.save_embeddings),
                        config=per_item_cfg,
                        provider_factory=self.provider_factory,
                        model_progress_cb=(None if use_batch else model_progress_cb),
                        fetch_fn=self.fetch_fn,
                        inspect_fn=self.inspect_fn,
                    )
                    if use_batch:
                        self._inject_precomputed_embeddings(
                            point_index=i,
                            models=self.model_names,
                            arrays=arrays,
                            manifest=manifest,
                            embed_results=chunk_embed_results,
                        )
                except Exception as e:
                    if not cfg.continue_on_error:
                        if writer_ex is not None:
                            writer_ex.shutdown(wait=False)
                        raise
                    manifests.append(
                        point_failure_manifest(
                            point_index=i,
                            spatial=self.spatials[i],
                            temporal=self.temporal,
                            output=self.output,
                            backend=self.backend,
                            device=self.device,
                            stage="build",
                            error=e,
                        )
                    )
                    progress.update(1)
                    continue

                if writer_ex is not None:
                    fut = writer_ex.submit(
                        write_one_payload,
                        out_path=out_file,
                        arrays=arrays,
                        manifest=manifest,
                        save_manifest=cfg.save_manifest,
                        fmt=cfg.format,
                        max_retries=cfg.max_retries,
                        retry_backoff_s=cfg.retry_backoff_s,
                    )
                    write_futs.append((i, fut))
                else:
                    mani = self._write_payload_sync(
                        point_index=i,
                        out_path=out_file,
                        arrays=arrays,
                        manifest=manifest,
                    )
                    manifests.append(mani)
                    progress.update(1)

            # Collect async write results
            if writer_ex is not None:
                try:
                    self._collect_async_results(
                        write_futs=write_futs,
                        manifests=manifests,
                        progress=progress,
                    )
                finally:
                    writer_ex.shutdown(wait=True)
        finally:
            if writer_ex is not None and not writer_ex._shutdown:
                writer_ex.shutdown(wait=True)

    def _write_payload_sync(
        self,
        *,
        point_index: int,
        out_path: str,
        arrays: Dict[str, np.ndarray],
        manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Write one payload synchronously, converting write errors as configured."""
        cfg = self.config
        try:
            return write_one_payload(
                out_path=out_path,
                arrays=arrays,
                manifest=manifest,
                save_manifest=cfg.save_manifest,
                fmt=cfg.format,
                max_retries=cfg.max_retries,
                retry_backoff_s=cfg.retry_backoff_s,
            )
        except Exception as e:
            if not cfg.continue_on_error:
                raise
            return point_failure_manifest(
                point_index=point_index,
                spatial=self.spatials[point_index],
                temporal=self.temporal,
                output=self.output,
                backend=self.backend,
                device=self.device,
                stage="write",
                error=e,
            )

    def _collect_async_results(
        self,
        *,
        write_futs: List[Tuple[int, Any]],
        manifests: List[Dict[str, Any]],
        progress: Any,
    ) -> None:
        """Collect async write results and translate failures into manifests."""
        cfg = self.config
        fut_map = {fut: i for (i, fut) in write_futs}
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            try:
                manifests.append(fut.result())
            except Exception as e:
                if not cfg.continue_on_error:
                    raise
                manifests.append(
                    point_failure_manifest(
                        point_index=i,
                        spatial=self.spatials[i],
                        temporal=self.temporal,
                        output=self.output,
                        backend=self.backend,
                        device=self.device,
                        stage="write",
                        error=e,
                    )
                )
            finally:
                progress.update(1)

    # ── static helpers ─────────────────────────────────────────────

    @staticmethod
    def _inject_precomputed_embeddings(
        *,
        point_index: int,
        models: List[str],
        arrays: Dict[str, np.ndarray],
        manifest: Dict[str, Any],
        embed_results: Dict[Tuple[int, str], Any],
    ) -> None:
        """Inject pre-computed batch embeddings into a per-item payload."""
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
            # TaskResult objects
            from ..core.types import Status, TaskResult

            if isinstance(rec, TaskResult):
                if rec.status == Status.OK and rec.embedding is not None:
                    e_np = np.asarray(rec.embedding)
                    emb_key = f"embedding__{sanitize_key(m)}"
                    arrays[emb_key] = e_np
                    entry["embedding"] = {
                        "npz_key": emb_key,
                        "dtype": str(e_np.dtype),
                        "shape": list(e_np.shape),
                        "sha1": sha1(e_np),
                    }
                    entry["meta"] = rec.meta
                else:
                    if entry.get("status") != "failed":
                        entry["status"] = "failed"
                        entry["error"] = rec.error
                    entry["embedding"] = None
                    entry["meta"] = None
            # Legacy dict format
            elif isinstance(rec, dict):
                if rec.get("status") == "ok":
                    e_np = np.asarray(rec["embedding"])
                    emb_key = f"embedding__{sanitize_key(m)}"
                    arrays[emb_key] = e_np
                    entry["embedding"] = {
                        "npz_key": emb_key,
                        "dtype": str(e_np.dtype),
                        "shape": list(e_np.shape),
                        "sha1": sha1(e_np),
                    }
                    entry["meta"] = rec.get("meta")
                else:
                    if entry.get("status") != "failed":
                        entry["status"] = "failed"
                        entry["error"] = rec.get("error")
                    entry["embedding"] = None
                    entry["meta"] = None

        # Recompute summary
        all_models = manifest.get("models") or []
        n_failed = sum(
            1
            for x in all_models
            if isinstance(x, dict) and str(x.get("status", "")).lower() == "failed"
        )
        manifest["status"] = summarize_status(all_models)
        manifest["summary"] = {
            "total_models": len(all_models),
            "failed_models": n_failed,
            "ok_models": len(all_models) - n_failed,
        }

"""Inference engine: encapsulates embedder resolution, batch/single dispatch.

Absorbs ``_infer_chunk_embeddings_for_per_item`` (api.py, 200 lines),
``runtime_helpers.py`` (embedder caching, batch detection), and the
``_ExportFlowOverrides`` pattern into a single stateful object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.embedding import Embedding
from ..core.export_helpers import embedding_to_numpy, jsonable, sensor_cache_key
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig, ModelConfig, Status, TaskResult
from ..internal.api.output_helpers import normalize_embedding_output
from ..internal.api.runtime_helpers import (
    call_embedder_get_embedding,
    get_embedder_bundle_cached,
    sensor_key,
    supports_batch_api,
    supports_prefetched_batch_api,
)
from ..internal.api.tiling_helpers import (
    _call_embedder_get_embedding_with_input_prep,
    _resolve_input_prep_spec,
)
from .runner import run_with_retry


class InferenceEngine:
    """Manages embedder lifecycle and dispatches single/batch inference.

    Replaces the 200-line ``_infer_chunk_embeddings_for_per_item`` function
    and the ``_ExportFlowOverrides`` dataclass.  All config lives on ``self``.

    Parameters
    ----------
    device : str
        Target device (``"auto"``, ``"cpu"``, ``"cuda"``, …).
    output : OutputSpec
        Embedding output spec (pooled/grid).
    config : ExportConfig
        Behavioral flags.
    """

    def __init__(
        self,
        *,
        device: str,
        output: OutputSpec,
        config: ExportConfig,
    ) -> None:
        self.device = device
        self.output = output
        self.config = config
        self.input_prep_resolved = _resolve_input_prep_spec(config.input_prep)
        self.prefer_batch = _device_has_gpu(device)
        self._explicit_nonresize = (config.input_prep is not None) and (
            self.input_prep_resolved.mode in {"tile", "auto"}
        )

    # ── single-point inference ─────────────────────────────────────

    def infer_single(
        self,
        *,
        embedder: Any,
        lock: Any,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        backend: str,
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        """Run a single embedding with retry + optional tiling."""
        cfg = self.config
        return run_with_retry(
            lambda: _call_embedder_get_embedding_with_input_prep(
                embedder=embedder,
                spatial=spatial,
                temporal=temporal,
                sensor=sensor,
                output=self.output,
                backend=backend,
                device=self.device,
                input_chw=input_chw,
                input_prep=cfg.input_prep,
            ),
            retries=cfg.max_retries,
            backoff_s=cfg.retry_backoff_s,
        )

    # ── chunk inference (multi-point × multi-model) ────────────────

    def infer_chunk(
        self,
        *,
        idxs: List[int],
        spatials: List[SpatialSpec],
        temporal: Optional[TemporalSpec],
        models: List[ModelConfig],
        prefetch_cache: Dict[Tuple[int, str], np.ndarray],
        prefetch_errors: Dict[Tuple[int, str], str],
        model_progress_cb: Optional[Any] = None,
    ) -> Dict[Tuple[int, str], TaskResult]:
        """Infer embeddings for a chunk of spatial indices across all models.

        Returns ``{(point_idx, model_name): TaskResult}``.
        """
        out: Dict[Tuple[int, str], TaskResult] = {}
        cfg = self.config
        infer_bs = cfg.effective_infer_batch_size

        for mc in models:
            sensor_k = sensor_key(mc.sensor)
            skey = (
                sensor_cache_key(mc.sensor)
                if (not mc.is_precomputed) and mc.sensor is not None
                else None
            )
            needs_provider_input = skey is not None

            from ..internal.api.api_helpers import normalize_model_name
            embedder, lock = get_embedder_bundle_cached(
                normalize_model_name(mc.name), mc.backend, self.device, sensor_k
            )

            def _get_input(i: int) -> Optional[np.ndarray]:
                if not needs_provider_input or skey is None:
                    return None
                hit = prefetch_cache.get((i, skey))
                if hit is not None:
                    return hit
                err = prefetch_errors.get((i, skey))
                if err:
                    raise RuntimeError(
                        f"Prefetch failed for model={mc.name}, index={i}: {err}"
                    )
                raise RuntimeError(
                    f"Missing prefetched input for model={mc.name}, index={i}"
                )

            def _single(i: int) -> Embedding:
                inp = _get_input(i)
                return self.infer_single(
                    embedder=embedder,
                    lock=lock,
                    spatial=spatials[i],
                    temporal=temporal,
                    sensor=mc.sensor,
                    backend=mc.backend,
                    input_chw=inp,
                )

            def _mark_done(model_name: str) -> None:
                if model_progress_cb is not None:
                    try:
                        model_progress_cb(model_name)
                    except Exception:
                        pass

            def _record_ok(i: int) -> None:
                pass  # placeholder for progress; actual recording below

            # -- try batch paths first --
            batch_succeeded = False

            can_batch_prefetched = (
                self.prefer_batch
                and not self._explicit_nonresize
                and supports_prefetched_batch_api(embedder)
                and needs_provider_input
                and mc.sensor is not None
                and skey is not None
            )
            can_batch_no_input = (
                self.prefer_batch
                and not self._explicit_nonresize
                and supports_batch_api(embedder)
                and not needs_provider_input
            )

            if can_batch_prefetched:
                try:
                    ready: List[Tuple[int, SpatialSpec, np.ndarray]] = []
                    for i in idxs:
                        try:
                            inp = _get_input(i)
                            assert inp is not None
                            ready.append((i, spatials[i], np.asarray(inp, dtype=np.float32)))
                        except Exception as e:
                            if not cfg.continue_on_error:
                                raise
                            out[(i, mc.name)] = TaskResult.failed(e)
                            _mark_done(mc.name)

                    for start in range(0, len(ready), infer_bs):
                        sub = ready[start : start + infer_bs]
                        if not sub:
                            continue
                        sub_idx = [t[0] for t in sub]
                        sub_sp = [t[1] for t in sub]
                        sub_inp = [t[2] for t in sub]

                        batch_out = run_with_retry(
                            lambda: embedder.get_embeddings_batch_from_inputs(
                                spatials=sub_sp,
                                input_chws=sub_inp,
                                temporal=temporal,
                                sensor=mc.sensor,
                                output=self.output,
                                backend=mc.backend,
                                device=self.device,
                            ),
                            retries=cfg.max_retries,
                            backoff_s=cfg.retry_backoff_s,
                        )
                        if len(batch_out) != len(sub_idx):
                            raise RuntimeError(
                                f"Model {mc.name} returned {len(batch_out)} embeddings "
                                f"for {len(sub_idx)} prefetched inputs."
                            )
                        for j, emb in enumerate(batch_out):
                            emb_n = normalize_embedding_output(emb=emb, output=self.output)
                            out[(sub_idx[j], mc.name)] = TaskResult.ok(
                                embedding_to_numpy(emb_n), jsonable(getattr(emb_n, "meta", None))
                            )
                            _mark_done(mc.name)
                    batch_succeeded = True
                except Exception:
                    batch_succeeded = False

            if not batch_succeeded and can_batch_no_input:
                try:
                    for start in range(0, len(idxs), infer_bs):
                        sub_idx = idxs[start : start + infer_bs]
                        sub_sp = [spatials[i] for i in sub_idx]

                        batch_out = run_with_retry(
                            lambda: embedder.get_embeddings_batch(
                                spatials=sub_sp,
                                temporal=temporal,
                                sensor=mc.sensor,
                                output=self.output,
                                backend=mc.backend,
                                device=self.device,
                            ),
                            retries=cfg.max_retries,
                            backoff_s=cfg.retry_backoff_s,
                        )
                        if len(batch_out) != len(sub_idx):
                            raise RuntimeError(
                                f"Model {mc.name} returned {len(batch_out)} embeddings "
                                f"for {len(sub_idx)} inputs."
                            )
                        for j, emb in enumerate(batch_out):
                            emb_n = normalize_embedding_output(emb=emb, output=self.output)
                            out[(sub_idx[j], mc.name)] = TaskResult.ok(
                                embedding_to_numpy(emb_n), jsonable(getattr(emb_n, "meta", None))
                            )
                            _mark_done(mc.name)
                    batch_succeeded = True
                except Exception:
                    batch_succeeded = False

            # -- fallback: single-item --
            if not batch_succeeded:
                for i in idxs:
                    if (i, mc.name) in out:
                        continue
                    try:
                        emb = _single(i)
                        out[(i, mc.name)] = TaskResult.ok(
                            embedding_to_numpy(emb), jsonable(getattr(emb, "meta", None))
                        )
                    except Exception as e:
                        if not cfg.continue_on_error:
                            raise
                        out[(i, mc.name)] = TaskResult.failed(e)
                    _mark_done(mc.name)

        return out

    # ── embedder helpers ───────────────────────────────────────────

    @staticmethod
    def resolve_embedder(model_config: ModelConfig, device: str) -> Tuple[Any, Any]:
        """Return ``(embedder, lock)`` for the given model config."""
        from ..internal.api.api_helpers import normalize_model_name
        sensor_k = sensor_key(model_config.sensor)
        return get_embedder_bundle_cached(
            normalize_model_name(model_config.name),
            model_config.backend,
            device,
            sensor_k,
        )


# ── module-level helpers ───────────────────────────────────────────


def _device_has_gpu(device: str) -> bool:
    dev = str(device or "").strip().lower()
    if dev and dev not in {"auto", "cpu"}:
        return True
    if dev == "cpu":
        return False
    try:
        import torch

        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return True
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(getattr(mps, "is_available", lambda: False)()):
            return True
    except Exception:
        return False
    return False

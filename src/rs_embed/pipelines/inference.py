"""Inference engine for export pipelines.

This module resolves embedders and executes model inference in either
single-point or batch mode, with consistent retry/error shaping.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np

from ..core.embedding import Embedding
from ..tools.serialization import embedding_to_numpy, jsonable, sensor_cache_key
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig, ModelConfig, TaskResult
from ..tools.output import normalize_embedding_output
from ..tools.runtime import (
    call_embedder_get_embedding,
    get_embedder_bundle_cached,
    sensor_key,
    supports_batch_api,
    supports_prefetched_batch_api,
)
from ..tools.tiling import (
    _call_embedder_get_embedding_with_input_prep,
    _resolve_input_prep_spec,
)
from .runner import run_with_retry


class _ModelContext(NamedTuple):
    """Resolved embedder/runtime context for a single model inference pass."""

    embedder: Any
    lock: Any
    sensor_k: str
    skey: Optional[str]
    needs_provider_input: bool


class InferenceEngine:
    """Manages embedder lifecycle and dispatches single/batch inference.

    It centralizes inference policy (batch vs. single fallback), embedder
    bundle reuse, and result normalization into :class:`TaskResult` records.

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

    def _embedding_to_result(self, emb: Embedding) -> TaskResult:
        """Normalize an embedding and convert it into a successful TaskResult."""
        emb_n = normalize_embedding_output(emb=emb, output=self.output)
        return TaskResult.ok(
            embedding_to_numpy(emb_n), jsonable(getattr(emb_n, "meta", None))
        )

    def _resolve_model_context(
        self,
        *,
        name: str,
        backend: str,
        sensor: Optional[SensorSpec],
        is_precomputed: bool,
        provider_enabled: bool,
    ) -> _ModelContext:
        """Resolve embedder bundle and provider-input requirements for one model."""
        from ..tools.normalization import normalize_model_name

        sensor_k = sensor_key(sensor)
        skey = (
            sensor_cache_key(sensor)
            if provider_enabled and sensor is not None and not is_precomputed
            else None
        )
        embedder, lock = get_embedder_bundle_cached(
            normalize_model_name(name), backend, self.device, sensor_k
        )
        return _ModelContext(
            embedder=embedder,
            lock=lock,
            sensor_k=sensor_k,
            skey=skey,
            needs_provider_input=(skey is not None),
        )

    @staticmethod
    def _evaluate_batch_capability(
        *,
        embedder: Any,
        needs_provider_input: bool,
        sensor: Optional[SensorSpec],
        skey: Optional[str],
        prefer_batch: bool,
        allow_nonresize: bool,
    ) -> Tuple[bool, bool]:
        """Return tier-1/2 batch eligibility booleans for current model context."""
        can_batch_prefetched = (
            prefer_batch
            and allow_nonresize
            and supports_prefetched_batch_api(embedder)
            and needs_provider_input
            and sensor is not None
            and skey is not None
        )
        can_batch_no_input = (
            prefer_batch
            and allow_nonresize
            and supports_batch_api(embedder)
            and not needs_provider_input
        )
        return can_batch_prefetched, can_batch_no_input

    def _run_batch_prefetched(
        self,
        *,
        idxs: List[int],
        spatials: List[SpatialSpec],
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        embedder: Any,
        lock: Any,
        backend: str,
        get_input_fn: Callable[[int], np.ndarray],
        batch_size: int,
        continue_on_error: bool,
        on_done: Callable[[int], None],
        use_lock: bool,
        model_name: str,
    ) -> Tuple[Dict[int, TaskResult], bool]:
        """Run tier-1 batch inference using prefetched provider inputs."""
        cfg = self.config
        out: Dict[int, TaskResult] = {}
        try:
            ready: List[Tuple[int, SpatialSpec, np.ndarray]] = []
            for i in idxs:
                try:
                    inp = get_input_fn(i)
                    ready.append((i, spatials[i], np.asarray(inp, dtype=np.float32)))
                except Exception as e:
                    if not continue_on_error:
                        raise
                    out[i] = TaskResult.failed(e)
                    on_done(i)

            for start in range(0, len(ready), batch_size):
                sub = ready[start : start + batch_size]
                if not sub:
                    continue
                sub_idx = [t[0] for t in sub]
                sub_sp = [t[1] for t in sub]
                sub_inp = [t[2] for t in sub]

                def _infer_prefetched(_sp=sub_sp, _inp=sub_inp):
                    if use_lock:
                        with lock:
                            return embedder.get_embeddings_batch_from_inputs(
                                spatials=_sp,
                                input_chws=_inp,
                                temporal=temporal,
                                sensor=sensor,
                                output=self.output,
                                backend=backend,
                                device=self.device,
                            )
                    return embedder.get_embeddings_batch_from_inputs(
                        spatials=_sp,
                        input_chws=_inp,
                        temporal=temporal,
                        sensor=sensor,
                        output=self.output,
                        backend=backend,
                        device=self.device,
                    )

                batch_out = run_with_retry(
                    _infer_prefetched,
                    retries=cfg.max_retries,
                    backoff_s=cfg.retry_backoff_s,
                )
                if len(batch_out) != len(sub_idx):
                    raise RuntimeError(
                        f"Model {model_name} returned {len(batch_out)} embeddings "
                        f"for {len(sub_idx)} prefetched inputs."
                    )
                for j, emb in enumerate(batch_out):
                    out[sub_idx[j]] = self._embedding_to_result(emb)
                    on_done(sub_idx[j])
            return out, True
        except Exception:
            return out, False

    def _run_batch_no_input(
        self,
        *,
        idxs: List[int],
        spatials: List[SpatialSpec],
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        embedder: Any,
        lock: Any,
        backend: str,
        batch_size: int,
        on_done: Callable[[int], None],
        use_lock: bool,
        model_name: str,
    ) -> Tuple[Dict[int, TaskResult], bool]:
        """Run tier-2 batch inference that does not require provider inputs."""
        cfg = self.config
        out: Dict[int, TaskResult] = {}
        try:
            for start in range(0, len(idxs), batch_size):
                sub_idx = idxs[start : start + batch_size]
                sub_sp = [spatials[i] for i in sub_idx]

                def _infer_batch(_sp=sub_sp):
                    if use_lock:
                        with lock:
                            return embedder.get_embeddings_batch(
                                spatials=_sp,
                                temporal=temporal,
                                sensor=sensor,
                                output=self.output,
                                backend=backend,
                                device=self.device,
                            )
                    return embedder.get_embeddings_batch(
                        spatials=_sp,
                        temporal=temporal,
                        sensor=sensor,
                        output=self.output,
                        backend=backend,
                        device=self.device,
                    )

                batch_out = run_with_retry(
                    _infer_batch,
                    retries=cfg.max_retries,
                    backoff_s=cfg.retry_backoff_s,
                )
                if len(batch_out) != len(sub_idx):
                    raise RuntimeError(
                        f"Model {model_name} returned {len(batch_out)} embeddings "
                        f"for {len(sub_idx)} inputs."
                    )
                for j, emb in enumerate(batch_out):
                    out[sub_idx[j]] = self._embedding_to_result(emb)
                    on_done(sub_idx[j])
            return out, True
        except Exception:
            return out, False

    def _run_single_fallback(
        self,
        *,
        idxs: List[int],
        already_done: Set[int],
        infer_one_fn: Callable[[int], Embedding],
        continue_on_error: bool,
        on_done: Callable[[int], None],
    ) -> Dict[int, TaskResult]:
        """Run tier-3 single-item fallback for unfinished indices."""
        out: Dict[int, TaskResult] = {}
        for i in idxs:
            if i in already_done:
                continue
            try:
                emb = infer_one_fn(i)
                out[i] = self._embedding_to_result(emb)
            except Exception as e:
                if not continue_on_error:
                    raise
                out[i] = TaskResult.failed(e)
            on_done(i)
        return out

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
            ctx = self._resolve_model_context(
                name=mc.name,
                backend=mc.backend,
                sensor=mc.sensor,
                is_precomputed=mc.is_precomputed,
                provider_enabled=True,
            )

            def _get_input(i: int) -> np.ndarray:
                if not ctx.needs_provider_input or ctx.skey is None:
                    raise RuntimeError(
                        f"Missing prefetched input for model={mc.name}, index={i}"
                    )
                hit = prefetch_cache.get((i, ctx.skey))
                if hit is not None:
                    return hit
                err = prefetch_errors.get((i, ctx.skey))
                if err:
                    raise RuntimeError(
                        f"Prefetch failed for model={mc.name}, index={i}: {err}"
                    )
                raise RuntimeError(
                    f"Missing prefetched input for model={mc.name}, index={i}"
                )

            def _single(i: int) -> Embedding:
                inp = _get_input(i) if ctx.needs_provider_input else None
                return self.infer_single(
                    embedder=ctx.embedder,
                    lock=ctx.lock,
                    spatial=spatials[i],
                    temporal=temporal,
                    sensor=mc.sensor,
                    backend=mc.backend,
                    input_chw=inp,
                )

            def _mark_done(_: int) -> None:
                if model_progress_cb is None:
                    return
                try:
                    model_progress_cb(mc.name)
                except Exception:
                    pass

            can_batch_prefetched, can_batch_no_input = self._evaluate_batch_capability(
                embedder=ctx.embedder,
                needs_provider_input=ctx.needs_provider_input,
                sensor=mc.sensor,
                skey=ctx.skey,
                prefer_batch=self.prefer_batch,
                allow_nonresize=not self._explicit_nonresize,
            )

            batch_succeeded = False
            if can_batch_prefetched:
                prefetched_out, batch_succeeded = self._run_batch_prefetched(
                    idxs=idxs,
                    spatials=spatials,
                    temporal=temporal,
                    sensor=mc.sensor,
                    embedder=ctx.embedder,
                    lock=ctx.lock,
                    backend=mc.backend,
                    get_input_fn=_get_input,
                    batch_size=infer_bs,
                    continue_on_error=cfg.continue_on_error,
                    on_done=_mark_done,
                    use_lock=False,
                    model_name=mc.name,
                )
                for i, rec in prefetched_out.items():
                    out[(i, mc.name)] = rec

            if not batch_succeeded and can_batch_no_input:
                batch_out, batch_succeeded = self._run_batch_no_input(
                    idxs=idxs,
                    spatials=spatials,
                    temporal=temporal,
                    sensor=mc.sensor,
                    embedder=ctx.embedder,
                    lock=ctx.lock,
                    backend=mc.backend,
                    batch_size=infer_bs,
                    on_done=_mark_done,
                    use_lock=False,
                    model_name=mc.name,
                )
                for i, rec in batch_out.items():
                    out[(i, mc.name)] = rec

            if not batch_succeeded:
                already_done = {i for i in idxs if (i, mc.name) in out}
                fallback_out = self._run_single_fallback(
                    idxs=idxs,
                    already_done=already_done,
                    infer_one_fn=_single,
                    continue_on_error=cfg.continue_on_error,
                    on_done=_mark_done,
                )
                for i, rec in fallback_out.items():
                    out[(i, mc.name)] = rec

        return out

    # ── embedder helpers ───────────────────────────────────────────

    @staticmethod
    def resolve_embedder(model_config: ModelConfig, device: str) -> Tuple[Any, Any]:
        """Return ``(embedder, lock)`` for the given model config."""
        from ..tools.normalization import normalize_model_name

        sensor_k = sensor_key(model_config.sensor)
        return get_embedder_bundle_cached(
            normalize_model_name(model_config.name),
            model_config.backend,
            device,
            sensor_k,
        )

    # ── combined-export: infer all spatials for one model ──────────

    def infer_model(
        self,
        *,
        model_name: str,
        model_backend: str,
        sensor: Optional[SensorSpec],
        is_precomputed: bool,
        provider_enabled: bool,
        spatials: List[SpatialSpec],
        temporal: Optional[TemporalSpec],
        inference_strategy: str,
        get_input_fn: Callable[[int, str, SensorSpec], np.ndarray],
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Dict[int, TaskResult]:
        """Infer embeddings for ALL spatial indices for a single model.

        Used by combined export.  Returns ``{spatial_idx: TaskResult}``.

        Parameters
        ----------
        get_input_fn
            ``fn(idx, sensor_cache_key, sensor_spec) -> np.ndarray`` that
            retrieves the (possibly cached) provider input for a point.
        progress_cb
            Called with spatial index after each point finishes inference.
        """
        cfg = self.config
        n = len(spatials)
        infer_bs = cfg.effective_infer_batch_size
        out: Dict[int, TaskResult] = {}
        done: set[int] = set()

        ctx = self._resolve_model_context(
            name=model_name,
            backend=model_backend,
            sensor=sensor,
            is_precomputed=is_precomputed,
            provider_enabled=provider_enabled,
        )

        strategy = str(inference_strategy).strip().lower()
        prefer_batch = (strategy == "batch") or (strategy == "auto")
        allow_batch = strategy != "single"

        def _mark_done(i: int) -> None:
            if i in done:
                return
            done.add(i)
            if progress_cb is not None:
                progress_cb(i)

        def _infer_one(i: int) -> Embedding:
            inp = None
            if ctx.needs_provider_input and ctx.skey is not None and sensor is not None:
                inp = get_input_fn(i, ctx.skey, sensor)
            with ctx.lock:
                return call_embedder_get_embedding(
                    embedder=ctx.embedder,
                    spatial=spatials[i],
                    temporal=temporal,
                    sensor=sensor,
                    output=self.output,
                    backend=model_backend,
                    device=self.device,
                    input_chw=inp,
                )

        def _infer_one_with_retry(i: int) -> Embedding:
            return run_with_retry(
                lambda i=i: _infer_one(i),
                retries=cfg.max_retries,
                backoff_s=cfg.retry_backoff_s,
            )

        can_batch_prefetched, can_batch = self._evaluate_batch_capability(
            embedder=ctx.embedder,
            needs_provider_input=ctx.needs_provider_input,
            sensor=sensor,
            skey=ctx.skey,
            prefer_batch=(allow_batch and prefer_batch),
            allow_nonresize=True,
        )

        batch_attempted = False
        batch_succeeded = False
        all_idxs = list(range(n))

        # Tier 1: batch with prefetched inputs
        if can_batch_prefetched:
            batch_attempted = True
            assert ctx.skey is not None and sensor is not None
            prefetched_out, batch_succeeded = self._run_batch_prefetched(
                idxs=all_idxs,
                spatials=spatials,
                temporal=temporal,
                sensor=sensor,
                embedder=ctx.embedder,
                lock=ctx.lock,
                backend=model_backend,
                get_input_fn=lambda i: get_input_fn(i, ctx.skey, sensor),
                batch_size=infer_bs,
                continue_on_error=cfg.continue_on_error,
                on_done=_mark_done,
                use_lock=True,
                model_name=model_name,
            )
            out.update(prefetched_out)

        # Tier 2: batch without inputs
        if not batch_attempted and can_batch:
            batch_attempted = True
            batch_out, batch_succeeded = self._run_batch_no_input(
                idxs=all_idxs,
                spatials=spatials,
                temporal=temporal,
                sensor=sensor,
                embedder=ctx.embedder,
                lock=ctx.lock,
                backend=model_backend,
                batch_size=infer_bs,
                on_done=_mark_done,
                use_lock=True,
                model_name=model_name,
            )
            out.update(batch_out)

        # Tier 3: single-item fallback
        if not batch_succeeded:
            fallback_out = self._run_single_fallback(
                idxs=all_idxs,
                already_done=done,
                infer_one_fn=_infer_one_with_retry,
                continue_on_error=cfg.continue_on_error,
                on_done=_mark_done,
            )
            out.update(fallback_out)

        return out


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

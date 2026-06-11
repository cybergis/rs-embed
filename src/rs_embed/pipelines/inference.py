"""Inference engine for export pipelines.

This module resolves embedders and executes model inference in either
single-point or batch mode, with consistent retry/error shaping.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig, ModelConfig, TaskResult
from ..tools.output import normalize_embedding_output
from ..tools.runtime import call_embedder_get_embedding as _runtime_call_embedder_get_embedding
from ..tools.runtime import (
    embedder_accepts_model_config,
    get_embedder_bundle_cached,
    require_model_config_support,
    sensor_key,
    supports_batch_api,
    supports_prefetched_batch_api,
)
from ..tools.serialization import embedding_to_numpy, jsonable, sensor_cache_key
from ..tools.tiling import (
    _aggregate_tiled_embeddings,
    _augment_model_config_for_tiled_dispatch,
    _call_embedder_get_embedding_with_input_prep,
    _embedder_default_image_size,
    _estimate_tile_count,
    _resolve_input_prep_spec,
    _slice_and_pad_tile,
    _tile_subspatial,
    _tile_yx_starts,
)
from .runner import run_with_retry

# Backward-compatible module attribute for tests/monkeypatches.
call_embedder_get_embedding = _runtime_call_embedder_get_embedding


class _ModelContext(NamedTuple):
    """Resolved embedder/runtime context for a single model inference pass."""

    embedder: Any
    lock: Any
    sensor_k: str
    skey: str | None
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
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        backend: str,
        input_chw: np.ndarray | None = None,
        model_config: dict[str, Any] | None = None,
        fetch_meta: dict[str, Any] | None = None,
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
                model_config=model_config,
                fetch_meta=fetch_meta,
            ),
            retries=cfg.max_retries,
            backoff_s=cfg.retry_backoff_s,
        )

    def _embedding_to_result(self, emb: Embedding) -> TaskResult:
        """Normalize an embedding and convert it into a successful TaskResult."""
        emb_n = normalize_embedding_output(emb=emb, output=self.output)
        return TaskResult.ok(embedding_to_numpy(emb_n), jsonable(getattr(emb_n, "meta", None)))

    def _resolve_model_context(
        self,
        *,
        name: str,
        backend: str,
        sensor: SensorSpec | None,
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
        sensor: SensorSpec | None,
        skey: str | None,
        prefer_batch: bool,
        allow_nonresize: bool,
        input_prep_mode: str = "resize",
    ) -> tuple[bool, bool, bool]:
        """Return (can_batch_prefetched, can_batch_no_input, can_batch_tiled)."""
        can_batch_prefetched = (
            prefer_batch
            and allow_nonresize
            and supports_prefetched_batch_api(embedder)
            and needs_provider_input
            and sensor is not None
            and skey is not None
        )
        can_batch_no_input = (
            prefer_batch and supports_batch_api(embedder) and not needs_provider_input
        )
        # Tier 1.5: batch across multiple spatial points, each tiled internally.
        # Only for explicit tile mode — auto mode requires per-image size inspection
        # which can't be batched without first fetching every image.
        can_batch_tiled = (
            prefer_batch
            and input_prep_mode == "tile"
            and supports_prefetched_batch_api(embedder)
            and needs_provider_input
            and sensor is not None
            and skey is not None
        )
        return can_batch_prefetched, can_batch_no_input, can_batch_tiled

    def _run_batch_prefetched(
        self,
        *,
        idxs: list[int],
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        embedder: Any,
        lock: Any,
        backend: str,
        get_input_fn: Callable[[int], np.ndarray],
        batch_size: int,
        continue_on_error: bool,
        on_done: Callable[[int], None],
        use_lock: bool,
        model_name: str,
        model_config: dict[str, Any] | None = None,
    ) -> tuple[dict[int, TaskResult], bool]:
        """Run tier-1 batch inference using prefetched provider inputs."""
        cfg = self.config
        out: dict[int, TaskResult] = {}
        try:
            ready: list[tuple[int, SpatialSpec, np.ndarray]] = []
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
                    if model_config is not None:
                        require_model_config_support(
                            embedder=embedder,
                            model_config=model_config,
                            method_name="get_embeddings_batch_from_inputs",
                        )
                    batch_kwargs: dict[str, Any] = {
                        "spatials": _sp,
                        "input_chws": _inp,
                        "temporal": temporal,
                        "sensor": sensor,
                        "output": self.output,
                        "backend": backend,
                        "device": self.device,
                    }
                    if model_config is not None and embedder_accepts_model_config(
                        type(embedder),
                        "get_embeddings_batch_from_inputs",
                    ):
                        batch_kwargs["model_config"] = model_config
                    if use_lock:
                        with lock:
                            return embedder.get_embeddings_batch_from_inputs(**batch_kwargs)
                    return embedder.get_embeddings_batch_from_inputs(**batch_kwargs)

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
        except Exception as _e:
            return out, False

    def _run_batch_no_input(
        self,
        *,
        idxs: list[int],
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        embedder: Any,
        lock: Any,
        backend: str,
        batch_size: int,
        on_done: Callable[[int], None],
        use_lock: bool,
        model_name: str,
        model_config: dict[str, Any] | None = None,
    ) -> tuple[dict[int, TaskResult], bool]:
        """Run tier-2 batch inference that does not require provider inputs."""
        cfg = self.config
        out: dict[int, TaskResult] = {}
        try:
            for start in range(0, len(idxs), batch_size):
                sub_idx = idxs[start : start + batch_size]
                sub_sp = [spatials[i] for i in sub_idx]

                def _infer_batch(_sp=sub_sp):
                    if model_config is not None:
                        require_model_config_support(
                            embedder=embedder,
                            model_config=model_config,
                            method_name="get_embeddings_batch",
                        )
                    batch_kwargs: dict[str, Any] = {
                        "spatials": _sp,
                        "temporal": temporal,
                        "sensor": sensor,
                        "output": self.output,
                        "backend": backend,
                        "device": self.device,
                    }
                    if model_config is not None and embedder_accepts_model_config(
                        type(embedder),
                        "get_embeddings_batch",
                    ):
                        batch_kwargs["model_config"] = model_config
                    if use_lock:
                        with lock:
                            return embedder.get_embeddings_batch(**batch_kwargs)
                    return embedder.get_embeddings_batch(**batch_kwargs)

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
        except Exception as _e:
            return out, False

    def _run_batch_tiled(
        self,
        *,
        idxs: list[int],
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        embedder: Any,
        lock: Any,
        backend: str,
        get_input_fn: Callable[[int], np.ndarray],
        batch_size: int,
        continue_on_error: bool,
        on_done: Callable[[int], None],
        use_lock: bool,
        model_name: str,
        model_config: dict[str, Any] | None = None,
    ) -> tuple[dict[int, TaskResult], bool]:
        """Run tile-mode batch inference across multiple spatial points.

        Each input image is tiled, all tiles from all spatial points are batched
        together for model inference, then stitched back into per-point results.
        """
        cfg = self.config
        spec = self.input_prep_resolved
        out: dict[int, TaskResult] = {}
        try:
            model_img = _embedder_default_image_size(embedder)
            tile_size = int(spec.tile_size or model_img or 0)
            if tile_size <= 0:
                return out, False
            stride = int(spec.tile_stride or tile_size)
            if stride <= 0 or stride != tile_size:
                return out, False
            tiled_model_config = _augment_model_config_for_tiled_dispatch(
                embedder, model_config, tile_size=tile_size
            )
            if tiled_model_config is not None and not embedder_accepts_model_config(
                type(embedder), "get_embeddings_batch_from_inputs"
            ):
                return out, False

            # Step 1: fetch inputs and slice each image into tiles.
            ready: list[tuple[int, SpatialSpec, np.ndarray]] = []
            for i in idxs:
                try:
                    inp = np.asarray(get_input_fn(i), dtype=np.float32)
                    ready.append((i, spatials[i], inp))
                except Exception as e:
                    if not continue_on_error:
                        raise
                    out[i] = TaskResult.failed(e)
                    on_done(i)

            all_tiles: list[np.ndarray] = []
            all_tile_spatials: list[SpatialSpec] = []
            # {spatial_idx: (flat_start, tile_count, tile_metas, h, w)}
            tile_map: dict[int, tuple[int, int, list[dict[str, Any]], int, int]] = {}

            for i, spatial, inp in ready:
                h, w = int(inp.shape[-2]), int(inp.shape[-1])
                num_tiles = _estimate_tile_count(h=h, w=w, tile_size=tile_size, stride=stride)
                if num_tiles > spec.max_tiles:
                    err = ModelError(
                        f"input_prep tile would create {num_tiles} tiles "
                        f"(> max_tiles={spec.max_tiles}); increase max_tiles or use resize/auto."
                    )
                    if not continue_on_error:
                        raise err
                    out[i] = TaskResult.failed(err)
                    on_done(i)
                    continue
                fill_value = float(sensor.fill_value) if sensor is not None else 0.0
                ys, xs = _tile_yx_starts(h=h, w=w, tile_size=tile_size, stride=stride)
                tiles: list[np.ndarray] = []
                tile_metas: list[dict[str, Any]] = []
                tile_spatials_pt: list[SpatialSpec] = []
                for r, y0 in enumerate(ys):
                    for c, x0 in enumerate(xs):
                        tile, meta = _slice_and_pad_tile(
                            inp,
                            y0=int(y0),
                            x0=int(x0),
                            tile_size=tile_size,
                            pad_edges=bool(spec.pad_edges),
                            fill_value=fill_value,
                        )
                        meta["row"] = int(r)
                        meta["col"] = int(c)
                        tiles.append(tile)
                        tile_metas.append(meta)
                        tile_spatials_pt.append(
                            _tile_subspatial(
                                spatial,
                                full_h=h,
                                full_w=w,
                                y0=meta["y0"],
                                y1=meta["y1"],
                                x0=meta["x0"],
                                x1=meta["x1"],
                            )
                        )
                flat_start = len(all_tiles)
                all_tiles.extend(tiles)
                all_tile_spatials.extend(tile_spatials_pt)
                tile_map[i] = (flat_start, len(tiles), tile_metas, h, w)

            if not all_tiles:
                return out, True

            # Step 2: run batch inference on the flat tile list.
            all_tile_embs: list[Embedding] = []
            for start in range(0, len(all_tiles), batch_size):
                sub_tiles = all_tiles[start : start + batch_size]
                sub_spatials = all_tile_spatials[start : start + batch_size]
                batch_kwargs: dict[str, Any] = {
                    "spatials": sub_spatials,
                    "input_chws": sub_tiles,
                    "temporal": temporal,
                    "sensor": sensor,
                    "output": self.output,
                    "backend": backend,
                    "device": self.device,
                }
                if tiled_model_config is not None:
                    batch_kwargs["model_config"] = tiled_model_config

                def _infer_tiles(_kw: dict[str, Any] = batch_kwargs) -> list[Embedding]:
                    if use_lock:
                        with lock:
                            return embedder.get_embeddings_batch_from_inputs(**_kw)
                    return embedder.get_embeddings_batch_from_inputs(**_kw)

                sub_embs = run_with_retry(
                    _infer_tiles,
                    retries=cfg.max_retries,
                    backoff_s=cfg.retry_backoff_s,
                )
                if len(sub_embs) != len(sub_tiles):
                    raise RuntimeError(
                        f"Model {model_name} returned {len(sub_embs)} embeddings "
                        f"for {len(sub_tiles)} tiles."
                    )
                all_tile_embs.extend(sub_embs)

            # Step 3: stitch tiles back into per-point embeddings. Iterate
            # tile_map (not ready): points that exceeded max_tiles under
            # continue_on_error were already marked failed and never tiled.
            for i, (flat_start, tile_count, tile_metas, h, w) in tile_map.items():
                tile_embs = all_tile_embs[flat_start : flat_start + tile_count]
                tile_embs_n = [
                    normalize_embedding_output(emb=e, output=self.output) for e in tile_embs
                ]
                if tile_count == 1:
                    result = self._embedding_to_result(tile_embs_n[0])
                else:
                    prep_meta: dict[str, Any] = {
                        "requested_mode": spec.mode,
                        "resolved_mode": "tile",
                        "tile_layout": "cover_shift",
                        "tile_size": int(tile_size),
                        "tile_stride": int(stride),
                        "tile_count": int(tile_count),
                        "pad_edges": bool(spec.pad_edges),
                        "max_tiles": int(spec.max_tiles),
                        "input_hw": (int(h), int(w)),
                    }
                    stitched = _aggregate_tiled_embeddings(
                        embs=tile_embs_n,
                        tile_meta=tile_metas,
                        output=self.output,
                        tile_size=tile_size,
                        stride=stride,
                        prep_meta=prep_meta,
                    )
                    result = self._embedding_to_result(stitched)
                out[i] = result
                on_done(i)

            return out, True
        except Exception:
            return out, False

    def _run_single_fallback(
        self,
        *,
        idxs: list[int],
        already_done: set[int],
        infer_one_fn: Callable[[int], Embedding],
        continue_on_error: bool,
        on_done: Callable[[int], None],
    ) -> dict[int, TaskResult]:
        """Run tier-3 single-item fallback for unfinished indices."""
        out: dict[int, TaskResult] = {}
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
        idxs: list[int],
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None,
        models: list[ModelConfig],
        prefetch_cache: dict[tuple[int, str], np.ndarray],
        prefetch_errors: dict[tuple[int, str], str],
        prefetch_meta: dict[tuple[int, str], dict[str, Any]] | None = None,
        model_progress_cb: Any | None = None,
    ) -> dict[tuple[int, str], TaskResult]:
        """Infer embeddings for a chunk of spatial indices across all models.

        Returns ``{(point_idx, model_name): TaskResult}``.
        """
        out: dict[tuple[int, str], TaskResult] = {}
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

            def _get_input(i: int, _ctx=ctx, _mc=mc) -> np.ndarray:
                if not _ctx.needs_provider_input or _ctx.skey is None:
                    raise RuntimeError(f"Missing prefetched input for model={_mc.name}, index={i}")
                hit = prefetch_cache.get((i, _ctx.skey))
                if hit is not None:
                    return hit
                err = prefetch_errors.get((i, _ctx.skey))
                if err:
                    raise RuntimeError(f"Prefetch failed for model={_mc.name}, index={i}: {err}")
                raise RuntimeError(f"Missing prefetched input for model={_mc.name}, index={i}")

            def _single(i: int, _ctx=ctx, _mc=mc) -> Embedding:
                inp = _get_input(i) if _ctx.needs_provider_input else None
                fmeta = None
                if (
                    _ctx.needs_provider_input
                    and _ctx.skey is not None
                    and prefetch_meta is not None
                ):
                    fmeta = prefetch_meta.get((i, _ctx.skey)) or None
                return self.infer_single(
                    embedder=_ctx.embedder,
                    lock=_ctx.lock,
                    spatial=spatials[i],
                    temporal=temporal,
                    sensor=_mc.sensor,
                    backend=_mc.backend,
                    input_chw=inp,
                    fetch_meta=fmeta,
                    model_config=_mc.model_config,
                )

            def _mark_done(_: int, _mc=mc) -> None:
                if model_progress_cb is None:
                    return
                try:
                    model_progress_cb(_mc.name)
                except Exception as _e:
                    pass

            can_batch_prefetched, can_batch_no_input, can_batch_tiled = (
                self._evaluate_batch_capability(
                    embedder=ctx.embedder,
                    needs_provider_input=ctx.needs_provider_input,
                    sensor=mc.sensor,
                    skey=ctx.skey,
                    prefer_batch=(self.prefer_batch or mc.is_precomputed),
                    allow_nonresize=not self._explicit_nonresize,
                    input_prep_mode=self.input_prep_resolved.mode,
                )
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
                    model_config=mc.model_config,
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
                    model_config=mc.model_config,
                )
                for i, rec in batch_out.items():
                    out[(i, mc.name)] = rec

            if not batch_succeeded and can_batch_tiled:
                tiled_out, batch_succeeded = self._run_batch_tiled(
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
                    model_config=mc.model_config,
                )
                for i, rec in tiled_out.items():
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
    def resolve_embedder(model_config: ModelConfig, device: str) -> tuple[Any, Any]:
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
        sensor: SensorSpec | None,
        is_precomputed: bool,
        provider_enabled: bool,
        model_config: dict[str, Any] | None,
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None,
        inference_strategy: str,
        get_input_fn: Callable[[int, str, SensorSpec], np.ndarray],
        get_fetch_meta_fn: Callable[[int, str], dict[str, Any]] | None = None,
        progress_cb: Callable[[int], None] | None = None,
    ) -> dict[int, TaskResult]:
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
        out: dict[int, TaskResult] = {}
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
            fmeta = None
            if ctx.needs_provider_input and ctx.skey is not None and sensor is not None:
                inp = get_input_fn(i, ctx.skey, sensor)
                if get_fetch_meta_fn is not None:
                    fmeta = get_fetch_meta_fn(i, ctx.skey) or None
            with ctx.lock:
                kwargs: dict[str, Any] = {
                    "embedder": ctx.embedder,
                    "spatial": spatials[i],
                    "temporal": temporal,
                    "sensor": sensor,
                    "output": self.output,
                    "backend": model_backend,
                    "device": self.device,
                    "input_chw": inp,
                    "input_prep": self.config.input_prep,
                }
                if fmeta is not None:
                    kwargs["fetch_meta"] = fmeta
                if model_config is not None:
                    kwargs["model_config"] = model_config
                return _call_embedder_get_embedding_with_input_prep(
                    **kwargs,
                )

        def _infer_one_with_retry(i: int) -> Embedding:
            return run_with_retry(
                lambda i=i: _infer_one(i),
                retries=cfg.max_retries,
                backoff_s=cfg.retry_backoff_s,
            )

        can_batch_prefetched, can_batch, can_batch_tiled = self._evaluate_batch_capability(
            embedder=ctx.embedder,
            needs_provider_input=ctx.needs_provider_input,
            sensor=sensor,
            skey=ctx.skey,
            prefer_batch=(allow_batch and prefer_batch),
            allow_nonresize=not self._explicit_nonresize,
            input_prep_mode=self.input_prep_resolved.mode,
        )

        batch_attempted = False
        batch_succeeded = False
        all_idxs = list(range(n))

        # Tier 1: batch with prefetched inputs (resize/pass-through mode)
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
                model_config=model_config,
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
                model_config=model_config,
            )
            out.update(batch_out)

        # Tier 1.5: tiled batch — tile each image then batch all tiles together
        if not batch_attempted and can_batch_tiled:
            batch_attempted = True
            assert ctx.skey is not None and sensor is not None
            tiled_out, batch_succeeded = self._run_batch_tiled(
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
                model_config=model_config,
            )
            out.update(tiled_out)

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
    except Exception as _e:
        return False
    return False

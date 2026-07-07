"""Provider input prefetch manager.

This module owns prefetch planning, threaded fetch execution, and cache/error
tracking for provider-backed input tensors used during export.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..core.specs import SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig, FetchResult, ModelConfig
from ..providers import fetch as _providers_fetch
from ..providers.prefetch_plan import (
    build_prefetch_plan,
    select_prefetched_channels,
)
from ..tools.normalization import normalize_input_array
from ..tools.progress import FetchStats
from ..tools.shape import square_fetch_request
from .runner import run_with_retry


class PrefetchManager:
    """Manages provider input prefetching, caching, and error tracking.

    After construction, call :meth:`plan` to build the fetch plan, then
    :meth:`fetch_chunk` for each batch of spatial indices.  Cached inputs
    are available via :meth:`get_input`.

    Parameters
    ----------
    provider : optional
        A ready provider instance (e.g. GEEProvider).  ``None`` disables
        prefetching.
    models : list of str
        Model names in the export.
    resolved_sensor : dict
        ``{model_name: SensorSpec | None}`` resolved per model.
    model_type : dict
        ``{model_name: type_str}`` from ``describe()``.
    config : ExportConfig
        Behavioral flags (retries, workers, fail_on_bad_input, …).
    """

    def __init__(
        self,
        *,
        provider: Any | None,
        models: list[str],
        resolved_sensor: dict[str, SensorSpec | None],
        model_type: dict[str, str],
        config: ExportConfig,
        fetch_fn: Callable[..., np.ndarray] | None = None,
        inspect_fn: Callable[..., dict[str, Any]] | None = None,
        fetcher_by_key: dict[str, Any] | None = None,
        model_configs: list[ModelConfig] | None = None,
        resolved_backend: dict[str, str] | None = None,
        backend: str = "auto",
        device: str = "auto",
    ) -> None:
        self.provider = provider
        self.models = models
        self.resolved_sensor = resolved_sensor
        self.model_type = model_type
        self.config = config
        self.fetch_fn = fetch_fn or _providers_fetch.fetch_sensor_patch_chw
        self.inspect_fn = inspect_fn or _providers_fetch.inspect_fetch_result
        self.model_configs = model_configs
        self.resolved_backend = resolved_backend or {}
        self.backend = backend
        self.device = device
        self.fetcher_by_key: dict[str, Any] = fetcher_by_key or {}
        # Fetch keys whose GENERIC fetches must enlarge a rectangular ROI to a
        # square (with the crop-back window in fetch meta), mirroring the
        # fetch-square behavior every model-specific ``fetch_input`` applies by
        # default: merged band-union groups and models without a custom
        # fetch_input, whenever every member model honors ``fetch_meta``.
        # Resolved by plan() when ``model_configs`` is provided.
        self.square_fetch_keys: set[str] = set()
        self._plan_resolve_bands_fn: Callable[..., Any] | None = None

        # Caches populated by fetch_chunk / restored from checkpoint
        self.cache: dict[tuple[int, str], np.ndarray] = {}
        self.errors: dict[tuple[int, str], str] = {}
        self.input_reports: dict[tuple[int, str], dict[str, Any]] = {}
        self.fetch_meta: dict[tuple[int, str], dict[str, Any]] = {}

        # Populated by plan()
        self.sensor_by_key: dict[str, SensorSpec] = {}
        self.fetch_sensor_by_key: dict[str, SensorSpec] = {}
        self.sensor_to_fetch: dict[str, tuple[str, tuple[int, ...]]] = {}
        self.sensor_models: dict[str, list[str]] = {}
        self.fetch_members: dict[str, list[str]] = {}

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    def _generic_fetch_spatial(
        self, spatial: SpatialSpec, fetch_key: str
    ) -> tuple[SpatialSpec, dict[str, Any]]:
        """Apply fetch-square to a generic fetch when the plan calls for it.

        Returns ``(spatial_to_fetch, fetch_meta)`` — the ROI enlarged to a
        square with its ``roi_window_geo`` crop-back window, or the spatial
        unchanged with empty meta.
        """
        if fetch_key not in self.square_fetch_keys:
            return spatial, {}
        return square_fetch_request(spatial)

    # ── plan ───────────────────────────────────────────────────────

    def plan(self, resolve_bands_fn: Callable[..., Any] | None = None) -> None:
        """Build the prefetch plan (sensor dedup + band unions + fetch policy).

        Owns the whole plan: which fetches use a model-specific ``fetch_input``
        and which generic fetches must fetch-square. Keeping this inside
        ``plan()`` makes :meth:`clone` trivially safe — the policy can never be
        forgotten when copying a manager (that omission once silently disabled
        fetch-square for every chunk after the first).
        """
        self._plan_resolve_bands_fn = resolve_bands_fn
        (
            self.sensor_by_key,
            self.fetch_sensor_by_key,
            self.sensor_to_fetch,
            self.sensor_models,
            self.fetch_members,
        ) = build_prefetch_plan(
            models=self.models,
            resolved_sensor=self.resolved_sensor,
            model_type=self.model_type,
            resolve_bands_fn=resolve_bands_fn,
        )
        if self.model_configs is not None:
            self.fetcher_by_key, self.square_fetch_keys = self._resolve_fetchers()

    def _resolve_fetchers(self) -> tuple[dict[str, Any], set[str]]:
        """fetch_key → embedder for model-specific fetches, and the generic
        fetch keys that must fetch-square.

        The canonical single-embedding path always enlarges a rectangular ROI
        to a square fetch (``fetch_input``'s ``square_input=True`` default) and
        crops the output back via ``roi_window_geo``. Model-specific
        ``fetch_input`` fetches inherit that for free; generic fetches (merged
        multi-model band-union groups, models without a custom ``fetch_input``)
        are squared by the prefetch itself whenever every member model's
        ``get_embedding`` accepts ``fetch_meta`` and thus honors the crop-back
        window.
        """
        from ..tools.normalization import normalize_model_name
        from ..tools.runtime import (
            _overrides_base_method,
            embedder_honors_fetch_meta,
            get_embedder_bundle_cached,
        )
        from ..tools.serialization import sensor_cache_key

        fetcher_by_key: dict[str, Any] = {}
        # fetch_key → True while every member seen so far honors fetch_meta.
        meta_safe_by_key: dict[str, bool] = {}
        for mc in self.model_configs or []:
            if mc.sensor is None or "precomputed" in mc.model_type.lower():
                continue
            member_skey = sensor_cache_key(mc.sensor)
            mapping = self.sensor_to_fetch.get(member_skey)
            if mapping is None:
                continue
            fetch_key = mapping[0]
            member_keys = self.fetch_members.get(fetch_key, [])
            embedder, _lock = get_embedder_bundle_cached(
                normalize_model_name(mc.name),
                self.resolved_backend.get(mc.name, self.backend),
                self.device,
            )
            meta_safe_by_key[fetch_key] = meta_safe_by_key.get(
                fetch_key, True
            ) and embedder_honors_fetch_meta(type(embedder))
            # A merged fetch group may represent a union of channels needed by
            # multiple models. Model-specific fetch_input() implementations
            # generally return only that model's own contract, so using one of
            # them for a shared union fetch can truncate the prefetched tensor.
            if len(member_keys) != 1:
                continue
            if fetch_key in fetcher_by_key:
                continue
            if getattr(embedder, "has_custom_fetch", False) or _overrides_base_method(
                embedder, "fetch_input"
            ):
                fetcher_by_key[fetch_key] = embedder
        square_fetch_keys = {
            k for k, safe in meta_safe_by_key.items() if safe and k not in fetcher_by_key
        }
        return fetcher_by_key, square_fetch_keys

    def clone(self) -> PrefetchManager:
        """A sibling manager with the same plan and fresh per-chunk caches."""
        twin = PrefetchManager(
            provider=self.provider,
            models=self.models,
            resolved_sensor=self.resolved_sensor,
            model_type=self.model_type,
            config=self.config,
            fetch_fn=self.fetch_fn,
            inspect_fn=self.inspect_fn,
            fetcher_by_key=dict(self.fetcher_by_key),
            model_configs=self.model_configs,
            resolved_backend=self.resolved_backend,
            backend=self.backend,
            device=self.device,
        )
        twin.plan(self._plan_resolve_bands_fn)
        return twin

    # ── fetch ──────────────────────────────────────────────────────

    def build_tasks(
        self, idxs: list[int], spatials: list[SpatialSpec]
    ) -> list[tuple[int, str, SensorSpec]]:
        """Return list of ``(spatial_idx, fetch_key, fetch_sensor)`` needing fetch."""
        tasks: list[tuple[int, str, SensorSpec]] = []
        if not self.enabled:
            return tasks
        for i in idxs:
            for fetch_key, fetch_sensor in self.fetch_sensor_by_key.items():
                member_keys = self.fetch_members.get(fetch_key, [])
                if member_keys and all((i, mk) in self.cache for mk in member_keys):
                    continue
                tasks.append((i, fetch_key, fetch_sensor))
        return tasks

    def fetch_chunk(
        self,
        idxs: list[int],
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None,
        *,
        progress: Any = None,
        fetch_stats: FetchStats | None = None,
    ) -> None:
        """Prefetch provider inputs for *idxs* into ``self.cache``."""
        if not self.enabled or self.provider is None:
            return

        tasks = self.build_tasks(idxs, spatials)

        if fetch_stats is not None:
            possible = len(idxs) * len(self.fetch_sensor_by_key)
            cache_hits = possible - len(tasks)
            if cache_hits > 0:
                fetch_stats.record_cache_hits(cache_hits)
            fetch_stats.record_planned(len(tasks))

        if not tasks:
            return

        cfg = self.config
        provider = self.provider

        def _fetch_one(
            i: int,
            skey: str,
            sspec: SensorSpec,
        ) -> tuple[int, str, np.ndarray, dict[str, Any]]:
            fetcher = self.fetcher_by_key.get(skey)
            if fetcher is not None:
                fr: FetchResult = run_with_retry(
                    lambda: fetcher.fetch_input(
                        provider,
                        spatial=spatials[i],
                        temporal=temporal,
                        sensor=sspec,
                    ),
                    retries=cfg.max_retries,
                    backoff_s=cfg.retry_backoff_s,
                )
                return i, skey, fr.data, fr.meta
            sq_spatial, fmeta = self._generic_fetch_spatial(spatials[i], skey)
            x = run_with_retry(
                lambda: self.fetch_fn(
                    provider, spatial=sq_spatial, temporal=temporal, sensor=sspec
                ),
                retries=cfg.max_retries,
                backoff_s=cfg.retry_backoff_s,
            )
            return i, skey, x, fmeta

        mw = max(1, cfg.num_workers)
        with ThreadPoolExecutor(max_workers=mw) as ex:
            fut_map = {ex.submit(_fetch_one, i, sk, ss): (i, sk) for (i, sk, ss) in tasks}
            for fut in as_completed(fut_map):
                i, skey = fut_map[fut]
                try:
                    i, skey, x, fmeta = fut.result()
                except Exception as e:
                    if not cfg.continue_on_error:
                        raise
                    err_s = repr(e)
                    for member_skey in self.fetch_members.get(skey, []):
                        self.errors[(i, member_skey)] = err_s
                    if fetch_stats is not None:
                        fetch_stats.record_failure()
                    if progress is not None:
                        progress.update(1)
                    continue

                member_failed = False
                for member_skey in self.fetch_members.get(skey, []):
                    member_idx = self.sensor_to_fetch[member_skey][1]
                    try:
                        x_member = normalize_input_array(
                            select_prefetched_channels(x, member_idx),
                            expected_channels=len(member_idx),
                            name=f"gee_input_{member_skey}",
                        )
                    except Exception as e:
                        # Post-fetch processing failures are per-member errors,
                        # not batch aborts, under continue_on_error.
                        if not cfg.continue_on_error:
                            raise
                        self.errors[(i, member_skey)] = repr(e)
                        member_failed = True
                        continue
                    if cfg.fail_on_bad_input:
                        sspec_member = self.sensor_by_key[member_skey]
                        rep = self.inspect_fn(
                            x_member,
                            sensor=sspec_member,
                            name=f"gee_input_{member_skey}",
                        )
                        if not bool(rep.get("ok", True)):
                            issues = (rep.get("report", {}) or {}).get("issues", [])
                            mlist = sorted(set(self.sensor_models.get(member_skey, [])))
                            err = RuntimeError(
                                f"Input inspection failed for index={i}, models={mlist}, sensor={member_skey}: {issues}"
                            )
                            if not cfg.continue_on_error:
                                raise err
                            self.errors[(i, member_skey)] = repr(err)
                            member_failed = True
                            continue
                        self.input_reports[(i, member_skey)] = rep
                    self.cache[(i, member_skey)] = x_member
                    if fmeta:
                        # Independent copy per member: consumers may mutate the
                        # dict they receive (or stamp it into Embedding.meta),
                        # which must not leak into sibling models' crop windows.
                        self.fetch_meta[(i, member_skey)] = dict(fmeta)

                # One stats record per fetch task: failed if any member failed
                # inspection (previously counted as both a failure and a success).
                if fetch_stats is not None:
                    if member_failed:
                        fetch_stats.record_failure()
                    else:
                        fsensor = self.fetch_sensor_by_key.get(skey)
                        fetch_stats.record_success(
                            point=i,
                            sensor=fsensor.collection if fsensor is not None else None,
                        )
                if progress is not None:
                    progress.update(1)

    # ── lookup ─────────────────────────────────────────────────────

    def get_input(self, idx: int, sensor_key: str) -> np.ndarray:
        """Return cached input, raising on miss or prior error."""
        hit = self.cache.get((idx, sensor_key))
        if hit is not None:
            return hit
        err = self.errors.get((idx, sensor_key))
        if err:
            raise RuntimeError(f"Prefetch failed for index={idx}, sensor={sensor_key}: {err}")
        raise RuntimeError(f"Missing prefetched input for index={idx}, sensor={sensor_key}")

    def get_or_fetch(
        self,
        idx: int,
        skey: str,
        sspec: SensorSpec,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
    ) -> np.ndarray:
        """Return cached input, or fetch on demand if not cached."""
        hit = self.cache.get((idx, skey))
        if hit is not None:
            return hit
        err = self.errors.get((idx, skey))
        if err:
            raise RuntimeError(f"Prefetch previously failed for index={idx}, sensor={skey}: {err}")
        if self.provider is None:
            raise RuntimeError(f"Missing provider for input fetch: index={idx}, sensor={skey}")
        cfg = self.config
        fetcher = self.fetcher_by_key.get(skey)
        if fetcher is not None:
            fr: FetchResult = run_with_retry(
                lambda: fetcher.fetch_input(
                    self.provider,
                    spatial=spatial,
                    temporal=temporal,
                    sensor=sspec,
                ),
                retries=cfg.max_retries,
                backoff_s=cfg.retry_backoff_s,
            )
            x = fr.data
            if fr.meta:
                self.fetch_meta[(idx, skey)] = fr.meta
        else:
            fetch_key = self.sensor_to_fetch.get(skey, (skey, ()))[0]
            sq_spatial, fmeta = self._generic_fetch_spatial(spatial, fetch_key)
            x = run_with_retry(
                lambda: self.fetch_fn(
                    self.provider, spatial=sq_spatial, temporal=temporal, sensor=sspec
                ),
                retries=cfg.max_retries,
                backoff_s=cfg.retry_backoff_s,
            )
            if fmeta:
                self.fetch_meta[(idx, skey)] = fmeta
        rep = self.inspect_fn(x, sensor=sspec, name=f"gee_input_{skey}")
        if cfg.fail_on_bad_input and (not bool(rep.get("ok", True))):
            issues = (rep.get("report", {}) or {}).get("issues", [])
            raise RuntimeError(f"Input inspection failed for index={idx}, sensor={skey}: {issues}")
        self.cache[(idx, skey)] = x
        self.input_reports[(idx, skey)] = rep
        return x

    def get_fetch_meta(self, idx: int, sensor_key: str) -> dict[str, Any]:
        """Return a copy of the fetch metadata for a cached input, or empty dict.

        A copy so a caller that mutates the returned dict cannot corrupt the
        cached entry other consumers (retries, sibling models) still read.
        """
        return dict(self.fetch_meta.get((idx, sensor_key), {}))

    def has_error(self, idx: int, sensor_key: str) -> str | None:
        return self.errors.get((idx, sensor_key))

    def clear_chunk(self) -> None:
        """Clear caches between chunks to bound memory."""
        self.cache.clear()
        self.errors.clear()
        self.input_reports.clear()
        self.fetch_meta.clear()

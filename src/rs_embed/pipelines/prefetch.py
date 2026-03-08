"""GEE input prefetch manager.

This module owns prefetch planning, threaded fetch execution, and cache/error
tracking for provider-backed input tensors used during export.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.specs import SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig
from ..providers import gee_utils as _gee_utils
from ..tools.normalization import normalize_input_chw
from ..providers.prefetch_plan import (
    build_gee_prefetch_plan,
    select_prefetched_channels,
)
from .runner import run_with_retry


class PrefetchManager:
    """Manages GEE input prefetching, caching, and error tracking.

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
        provider: Optional[Any],
        models: List[str],
        resolved_sensor: Dict[str, Optional[SensorSpec]],
        model_type: Dict[str, str],
        config: ExportConfig,
        fetch_fn: Optional[Callable[..., np.ndarray]] = None,
        inspect_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.provider = provider
        self.models = models
        self.resolved_sensor = resolved_sensor
        self.model_type = model_type
        self.config = config
        self.fetch_fn = fetch_fn or _gee_utils.fetch_gee_patch_raw
        self.inspect_fn = inspect_fn or _gee_utils.inspect_input_raw

        # Caches populated by fetch_chunk / restored from checkpoint
        self.cache: Dict[Tuple[int, str], np.ndarray] = {}
        self.errors: Dict[Tuple[int, str], str] = {}
        self.input_reports: Dict[Tuple[int, str], Dict[str, Any]] = {}

        # Populated by plan()
        self.sensor_by_key: Dict[str, SensorSpec] = {}
        self.fetch_sensor_by_key: Dict[str, SensorSpec] = {}
        self.sensor_to_fetch: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
        self.sensor_models: Dict[str, List[str]] = {}
        self.fetch_members: Dict[str, List[str]] = {}

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    # ── plan ───────────────────────────────────────────────────────

    def plan(self, resolve_bands_fn: Optional[Callable[..., Any]] = None) -> None:
        """Build the prefetch plan (sensor dedup + band unions)."""
        (
            self.sensor_by_key,
            self.fetch_sensor_by_key,
            self.sensor_to_fetch,
            self.sensor_models,
            self.fetch_members,
        ) = build_gee_prefetch_plan(
            models=self.models,
            resolved_sensor=self.resolved_sensor,
            model_type=self.model_type,
            resolve_bands_fn=resolve_bands_fn,
        )

    # ── fetch ──────────────────────────────────────────────────────

    def build_tasks(
        self, idxs: List[int], spatials: List[SpatialSpec]
    ) -> List[Tuple[int, str, SensorSpec]]:
        """Return list of ``(spatial_idx, fetch_key, fetch_sensor)`` needing fetch."""
        tasks: List[Tuple[int, str, SensorSpec]] = []
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
        idxs: List[int],
        spatials: List[SpatialSpec],
        temporal: Optional[TemporalSpec],
        *,
        progress: Any = None,
    ) -> None:
        """Prefetch GEE inputs for *idxs* into ``self.cache``."""
        if not self.enabled or self.provider is None:
            return

        tasks = self.build_tasks(idxs, spatials)
        if not tasks:
            return

        cfg = self.config
        provider = self.provider

        def _fetch_one(
            i: int, skey: str, sspec: SensorSpec
        ) -> Tuple[int, str, np.ndarray]:
            x = run_with_retry(
                lambda: self.fetch_fn(
                    provider, spatial=spatials[i], temporal=temporal, sensor=sspec
                ),
                retries=cfg.max_retries,
                backoff_s=cfg.retry_backoff_s,
            )
            return i, skey, x

        mw = max(1, cfg.num_workers)
        with ThreadPoolExecutor(max_workers=mw) as ex:
            fut_map = {
                ex.submit(_fetch_one, i, sk, ss): (i, sk) for (i, sk, ss) in tasks
            }
            for fut in as_completed(fut_map):
                i, skey = fut_map[fut]
                try:
                    i, skey, x = fut.result()
                except Exception as e:
                    if not cfg.continue_on_error:
                        raise
                    err_s = repr(e)
                    for member_skey in self.fetch_members.get(skey, []):
                        self.errors[(i, member_skey)] = err_s
                    if progress is not None:
                        progress.update(1)
                    continue

                for member_skey in self.fetch_members.get(skey, []):
                    member_idx = self.sensor_to_fetch[member_skey][1]
                    x_member = normalize_input_chw(
                        select_prefetched_channels(x, member_idx),
                        expected_channels=len(member_idx),
                        name=f"gee_input_{member_skey}",
                    )
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
                            continue
                        self.input_reports[(i, member_skey)] = rep
                    self.cache[(i, member_skey)] = x_member

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
            raise RuntimeError(
                f"Prefetch failed for index={idx}, sensor={sensor_key}: {err}"
            )
        raise RuntimeError(
            f"Missing prefetched input for index={idx}, sensor={sensor_key}"
        )

    def get_or_fetch(
        self,
        idx: int,
        skey: str,
        sspec: SensorSpec,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
    ) -> np.ndarray:
        """Return cached input, or fetch on demand if not cached."""
        hit = self.cache.get((idx, skey))
        if hit is not None:
            return hit
        err = self.errors.get((idx, skey))
        if err:
            raise RuntimeError(
                f"Prefetch previously failed for index={idx}, sensor={skey}: {err}"
            )
        if self.provider is None:
            raise RuntimeError(
                f"Missing provider for input fetch: index={idx}, sensor={skey}"
            )
        cfg = self.config
        x = run_with_retry(
            lambda: self.fetch_fn(
                self.provider, spatial=spatial, temporal=temporal, sensor=sspec
            ),
            retries=cfg.max_retries,
            backoff_s=cfg.retry_backoff_s,
        )
        rep = self.inspect_fn(x, sensor=sspec, name=f"gee_input_{skey}")
        if cfg.fail_on_bad_input and (not bool(rep.get("ok", True))):
            issues = (rep.get("report", {}) or {}).get("issues", [])
            raise RuntimeError(
                f"Input inspection failed for index={idx}, sensor={skey}: {issues}"
            )
        self.cache[(idx, skey)] = x
        self.input_reports[(idx, skey)] = rep
        return x

    def has_error(self, idx: int, sensor_key: str) -> Optional[str]:
        return self.errors.get((idx, sensor_key))

    def clear_chunk(self) -> None:
        """Clear caches between chunks to bound memory."""
        self.cache.clear()
        self.errors.clear()
        self.input_reports.clear()

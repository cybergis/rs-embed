from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from ..core.specs import SensorSpec
from ..tools.serialization import sensor_cache_key as _sensor_cache_key

_LEGACY_RESOLVE_BANDS_WARNED = False

def sensor_fetch_group_key(sensor: SensorSpec) -> tuple[str, int, int, float, str]:
    """Fetch identity excluding bands; used to build reusable band supersets."""
    cloudy = -1 if getattr(sensor, "cloudy_pct", None) is None else int(sensor.cloudy_pct)
    return (
        str(sensor.collection),
        int(sensor.scale_m),
        cloudy,
        float(sensor.fill_value),
        str(sensor.composite),
    )

def select_prefetched_channels(x_chw: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    if len(idx) == x_chw.shape[0] and all(i == j for j, i in enumerate(idx)):
        return x_chw
    return x_chw[list(idx), :, :]

def build_prefetch_plan(
    *,
    models: list[str],
    resolved_sensor: dict[str, SensorSpec | None],
    model_type: dict[str, str],
    resolve_bands_fn: Callable[..., tuple[str, ...]] | None = None,
) -> tuple[
    dict[str, SensorSpec],  # sensor_by_key
    dict[str, SensorSpec],  # fetch_sensor_by_key
    dict[str, tuple[str, tuple[int, ...]]],  # sensor_key -> (fetch_key, channel_idx)
    dict[str, list[str]],  # sensor_models
    dict[str, list[str]],  # fetch_members
]:
    sensor_by_key: dict[str, SensorSpec] = {}
    sensor_models: dict[str, list[str]] = {}
    for m in models:
        sspec = resolved_sensor.get(m)
        if sspec is None or "precomputed" in (model_type.get(m) or ""):
            continue
        skey = _sensor_cache_key(sspec)
        sensor_by_key.setdefault(skey, sspec)
        sensor_models.setdefault(skey, []).append(m)

    groups: dict[
        tuple[str, int, int, float, str], list[tuple[str, SensorSpec, tuple[str, ...]]]
    ] = {}
    for skey, sspec in sensor_by_key.items():
        gkey = sensor_fetch_group_key(sspec)
        if resolve_bands_fn is None:
            rbands = tuple(str(b) for b in sspec.bands)
        else:
            # Prefer keyword-style call to match ProviderBase.normalize_bands(*, collection, bands).
            # Fall back to positional call for backward-compatible test stubs/lambdas.
            try:
                rbands = resolve_bands_fn(
                    collection=str(sspec.collection),
                    bands=tuple(sspec.bands),
                )
            except TypeError:
                global _LEGACY_RESOLVE_BANDS_WARNED
                if not _LEGACY_RESOLVE_BANDS_WARNED:
                    warnings.warn(
                        "Legacy compatibility path used for `resolve_bands_fn`: "
                        "called with positional args `(collection, bands)`. "
                        "Please update to keyword-style signature "
                        "`resolve_bands_fn(*, collection, bands)`.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                    _LEGACY_RESOLVE_BANDS_WARNED = True
                rbands = resolve_bands_fn(str(sspec.collection), tuple(sspec.bands))
        groups.setdefault(gkey, []).append((skey, sspec, rbands))

    fetch_sensor_by_key: dict[str, SensorSpec] = {}
    sensor_to_fetch: dict[str, tuple[str, tuple[int, ...]]] = {}
    fetch_members: dict[str, list[str]] = {}

    for members in groups.values():
        union_bands: list[str] = []
        seen: set[str] = set()
        for _, _, rbands in members:
            for b in rbands:
                if b not in seen:
                    seen.add(b)
                    union_bands.append(b)
        if not union_bands:
            continue

        base = members[0][1]
        fetch_sensor = SensorSpec(
            collection=str(base.collection),
            bands=tuple(union_bands),
            scale_m=int(base.scale_m),
            cloudy_pct=(
                base.cloudy_pct
                if getattr(base, "cloudy_pct", None) is None
                else int(base.cloudy_pct)
            ),
            fill_value=float(base.fill_value),
            composite=str(base.composite),
            check_input=bool(getattr(base, "check_input", False)),
            check_raise=bool(getattr(base, "check_raise", True)),
            check_save_dir=getattr(base, "check_save_dir", None),
        )
        fetch_key = _sensor_cache_key(fetch_sensor)
        fetch_sensor_by_key[fetch_key] = fetch_sensor
        fetch_members.setdefault(fetch_key, [])

        band_pos = {b: i for i, b in enumerate(fetch_sensor.bands)}
        for member_key, _member_sensor, member_bands in members:
            idx = tuple(band_pos[b] for b in member_bands)
            sensor_to_fetch[member_key] = (fetch_key, idx)
            if member_key not in fetch_members[fetch_key]:
                fetch_members[fetch_key].append(member_key)

    return (
        sensor_by_key,
        fetch_sensor_by_key,
        sensor_to_fetch,
        sensor_models,
        fetch_members,
    )

# Backwards-compatible alias kept for existing imports/tests.
build_gee_prefetch_plan = build_prefetch_plan

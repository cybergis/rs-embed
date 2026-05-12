"""Utilities for inspecting raw patches downloaded from provider backends."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .core.errors import ModelError, ProviderError
from .core.specs import SensorSpec, SpatialSpec, TemporalSpec
from .providers.fetch import fetch_sensor_patch_chw as _fetch_sensor_patch_chw
from .providers.resolution import create_provider_for_backend
from .tools.inspection import checks_save_dir, inspect_chw, save_quicklook_rgb


def inspect_provider_patch(
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: tuple[float, float] | None = None,
    return_array: bool = False,
) -> dict[str, Any]:
    """Download a patch from a provider and return an input inspection report.

    Does **not** run any embedding model. Useful for verifying that a spatial
    location and sensor configuration produce valid imagery before committing
    to a full export.

    Parameters
    ----------
    spatial : SpatialSpec
        Spatial location to inspect.
    temporal : TemporalSpec or None
        Optional temporal filter.
    sensor : SensorSpec
        Sensor/collection configuration used for the download.
    backend : str
        Provider backend name (default ``"gee"``).
    name : str
        Label used in the report and quicklook filename (default
        ``"gee_patch"``).
    value_range : tuple[float, float] or None
        Optional ``(min, max)`` range for value-range checks in the report.
    return_array : bool
        If ``True``, attach the raw ``np.ndarray`` as ``array_chw`` in the
        returned dict (not JSON-serializable).

    Returns
    -------
    dict[str, Any]
        JSON-serializable inspection report with keys ``ok``, ``report``,
        ``sensor``, ``temporal``, ``backend``, and ``artifacts``.
        When ``return_array=True``, also includes ``array_chw``.

    Raises
    ------
    ProviderError
        If the backend name is empty or the provider fails to initialize.
    """

    backend_name = str(backend).strip().lower()
    if not backend_name:
        raise ProviderError("backend must be a non-empty provider name.")
    try:
        provider = create_provider_for_backend(backend_name, allow_auto=False)
    except ModelError as exc:
        raise ProviderError(str(exc)) from exc
    x_chw = _fetch_sensor_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
    )

    report = inspect_chw(
        x_chw,
        name=name,
        expected_channels=len(sensor.bands),
        value_range=value_range,
        fill_value=sensor.fill_value,
    )

    # Save quicklook if requested (best-effort)
    artifacts: dict[str, Any] = {}
    save_dir = checks_save_dir(sensor)
    if save_dir and x_chw.ndim == 3 and x_chw.shape[0] >= 3:
        try:
            import datetime as _dt
            import os

            ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(save_dir, f"{name}_{ts}.png")
            save_quicklook_rgb(x_chw, path=path, bands=(0, 1, 2))
            artifacts["quicklook_rgb"] = path
        except Exception as e:
            artifacts["quicklook_rgb_error"] = repr(e)

    out: dict[str, Any] = {
        "ok": bool(report.get("ok", False)),
        "report": report,
        "sensor": asdict(sensor),
        "temporal": asdict(temporal) if temporal is not None else None,
        "backend": backend_name,
        "artifacts": artifacts or None,
    }
    if return_array:
        out["array_chw"] = x_chw
    return out


def inspect_gee_patch(
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: tuple[float, float] | None = None,
    return_array: bool = False,
) -> dict[str, Any]:
    """Backwards-compatible wrapper around inspect_provider_patch."""
    return inspect_provider_patch(
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        backend=backend,
        name=name,
        value_range=value_range,
        return_array=return_array,
    )

from __future__ import annotations

"""Utilities for inspecting raw patches downloaded from provider backends."""

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple


from .core.errors import ProviderError
from .tools.inspection import inspect_chw, checks_save_dir, save_quicklook_rgb
from .core.specs import SensorSpec, SpatialSpec, TemporalSpec
from .providers.gee_utils import fetch_provider_patch_raw
from .providers import get_provider


def inspect_provider_patch(
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: Optional[Tuple[float, float]] = None,
    return_array: bool = False,
) -> Dict[str, Any]:
    """Download a patch from a provider and return an input inspection report.

    This does **not** run any embedding model.

    Returns
    -------
    dict
        A JSON-serializable report. If `return_array=True`, the report also
        includes a non-serializable `array_chw` entry with the numpy array.
    """

    backend_name = str(backend).strip().lower()
    if not backend_name:
        raise ProviderError("backend must be a non-empty provider name.")
    kwargs = {"auto_auth": True} if backend_name == "gee" else {}
    provider = get_provider(backend_name, **kwargs)
    provider.ensure_ready()
    x_chw = fetch_provider_patch_raw(
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
    artifacts: Dict[str, Any] = {}
    save_dir = checks_save_dir(sensor)
    if save_dir and x_chw.ndim == 3 and x_chw.shape[0] >= 3:
        try:
            import os
            import datetime as _dt

            ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(save_dir, f"{name}_{ts}.png")
            save_quicklook_rgb(x_chw, path=path, bands=(0, 1, 2))
            artifacts["quicklook_rgb"] = path
        except Exception as e:
            artifacts["quicklook_rgb_error"] = repr(e)

    out: Dict[str, Any] = {
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
    temporal: Optional[TemporalSpec] = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: Optional[Tuple[float, float]] = None,
    return_array: bool = False,
) -> Dict[str, Any]:
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

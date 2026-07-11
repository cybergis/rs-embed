from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from ..core.errors import ModelError
from ..core.specs import SensorSpec, TemporalSpec

# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------


def temporal_to_range(
    temporal: TemporalSpec | None,
    default: tuple[str, str] = ("2022-06-01", "2022-09-01"),
) -> TemporalSpec:
    """
    Normalize TemporalSpec to a start/end range.
    - None -> default range (with a UserWarning: the window is otherwise invisible)
    - year -> [year-01-01, (year+1)-01-01)  # end-exclusive
    - range -> unchanged
    """
    if temporal is None:
        import warnings

        warnings.warn(
            f"temporal=None: using the package default window "
            f"{default[0]}..{default[1]}. Pass temporal=TemporalSpec.range(...)/"
            "year(...) for reproducible, self-documented results.",
            UserWarning,
            stacklevel=2,
        )
        return TemporalSpec.range(default[0], default[1])
    temporal.validate()
    if temporal.mode == "range":
        return temporal
    if temporal.mode == "year":
        y = int(temporal.year)
        return TemporalSpec.range(f"{y}-01-01", f"{y + 1}-01-01")
    raise ModelError(f"Unknown TemporalSpec mode: {temporal.mode}")


def temporal_to_dict(temporal: TemporalSpec | None) -> dict[str, Any]:
    """
    Convert TemporalSpec into a serializable dictionary.
    """
    if temporal is None:
        return {"mode": None}
    temporal.validate()
    if temporal.mode == "range":
        return {"mode": "range", "start": temporal.start, "end": temporal.end}
    if temporal.mode == "year":
        return {
            "mode": "year",
            "year": temporal.year,
            "start": f"{temporal.year}-01-01",
            "end": f"{int(temporal.year) + 1}-01-01",
        }
    return {"mode": temporal.mode}


def temporal_midpoint_str(temporal: TemporalSpec | None) -> str | None:
    """
    Return an ISO date string representing the midpoint of the temporal window.
    For yearly mode, returns the mid-year date.
    """
    if temporal is None:
        return None
    temporal = temporal_to_range(temporal)
    if temporal.mode == "range" and temporal.start and temporal.end:
        start_dt = datetime.fromisoformat(temporal.start)
        end_dt = datetime.fromisoformat(temporal.end)
        mid_dt = start_dt + (end_dt - start_dt) / 2
        return mid_dt.date().isoformat()
    if temporal.mode == "year" and temporal.year is not None:
        return f"{int(temporal.year)}-07-01"
    return None


# ---------------------------------------------------------------------------
# Meta builder
# ---------------------------------------------------------------------------

META_REQUIRED_KEYS: tuple[str, ...] = (
    "model",
    "type",
    "backend",
    "source",
    "sensor",
    "temporal",
    "image_size",
)


def _sensor_to_dict(
    sensor: SensorSpec | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if sensor is None:
        return None
    if is_dataclass(sensor):
        return asdict(sensor)  # type: ignore[arg-type]
    if isinstance(sensor, dict):
        return sensor
    try:
        return asdict(sensor)  # type: ignore[arg-type]
    except Exception as exc:
        raise ModelError(f"Unsupported sensor meta type: {type(sensor)}") from exc


def build_meta(
    *,
    model: str,
    kind: str,
    backend: str,
    source: str | None,
    sensor: SensorSpec | dict[str, Any] | None,
    temporal: TemporalSpec | None,
    image_size: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Construct a consistent meta dictionary across embedders.
    Standard keys:
      model, type, backend, source, sensor, temporal, image_size

    Convention: *temporal* must be the **resolved** spec the data was actually
    fetched/served with (the ``temporal_to_range`` result, tessera's served
    year, ...), not the raw request — ``{"mode": None}`` in an embedding's meta
    would leave the real window unrecorded and the result irreproducible.
    """
    t_dict = temporal_to_dict(temporal)

    meta: dict[str, Any] = {
        "model": model,
        "type": kind,
        "backend": backend,
        "source": source,
        "sensor": _sensor_to_dict(sensor),
        "temporal": t_dict,
        "image_size": int(image_size) if image_size is not None else None,
    }

    for key in META_REQUIRED_KEYS:
        meta.setdefault(key, None)

    if extra:
        meta.update(extra)

    return meta


def base_meta(
    *,
    model_name,
    hf_id,
    backend,
    image_size,
    sensor,
    temporal=None,
    source=None,
    embed_type="on_the_fly",
    extra=None,
):
    """``build_meta`` wrapper for on-the-fly embedders that record an ``hf_id``."""
    m = build_meta(
        model=model_name,
        kind=embed_type,
        backend=backend,
        source=source or getattr(sensor, "collection", None),
        sensor=sensor,
        temporal=temporal,
        image_size=image_size,
    )
    m["hf_id"] = hf_id
    if extra:
        m.update(extra)
    return m

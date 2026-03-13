from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Optional

from ..core.errors import ModelError
from ..core.registry import get_embedder_cls
from ..core.specs import SensorSpec


def _probe_model_desc(model_id: str) -> dict:
    cls = get_embedder_cls(model_id)
    try:
        desc = cls().describe() or {}
    except Exception:
        desc = {}
    return desc if isinstance(desc, dict) else {}


def _normalize_modality_name(modality: Optional[str]) -> Optional[str]:
    if modality is None:
        return None
    key = str(modality).strip().lower().replace("-", "_")
    aliases = {
        "sentinel1": "s1",
        "sentinel_1": "s1",
        "sentinel2": "s2",
        "sentinel_2": "s2",
        "s2l1c": "s2_l1c",
        "s2l2a": "s2_l2a",
    }
    return aliases.get(key, key)


def _mk_sensor(
    *,
    collection: str,
    bands: Iterable[str],
    defaults: dict,
    modality: Optional[str] = None,
) -> SensorSpec:
    return SensorSpec(
        collection=str(collection),
        bands=tuple(str(b) for b in bands),
        scale_m=int(defaults.get("scale_m", 10)),
        cloudy_pct=int(defaults.get("cloudy_pct", 30)),
        composite=str(defaults.get("composite", "median")),
        fill_value=float(defaults.get("fill_value", 0.0)),
        modality=_normalize_modality_name(modality),
        orbit=defaults.get("orbit"),
        use_float_linear=bool(defaults.get("use_float_linear", True)),
    )


def modality_profiles_for_model(model_id: str) -> Dict[str, SensorSpec]:
    desc = _probe_model_desc(model_id)

    typ = str(desc.get("type", "")).lower()
    if "precomputed" in typ:
        return {}

    inputs = desc.get("inputs")
    defaults = desc.get("defaults", {}) or {}
    profiles: Dict[str, SensorSpec] = {}

    explicit = desc.get("modalities")
    if isinstance(explicit, dict):
        for raw_name, entry in explicit.items():
            if not isinstance(entry, dict):
                continue
            if "collection" not in entry or "bands" not in entry:
                continue
            name = _normalize_modality_name(str(raw_name))
            if not name:
                continue
            profile_defaults = {**defaults, **(entry.get("defaults") or {})}
            profiles[name] = _mk_sensor(
                collection=str(entry["collection"]),
                bands=entry["bands"],
                defaults=profile_defaults,
                modality=name,
            )

    if profiles:
        return profiles

    if isinstance(inputs, dict) and "collection" in inputs and "bands" in inputs:
        return {}
    if isinstance(inputs, dict) and "s2_sr" in inputs:
        s2 = inputs["s2_sr"]
        if isinstance(s2, dict) and "collection" in s2 and "bands" in s2:
            profiles["s2"] = _mk_sensor(
                collection=str(s2["collection"]),
                bands=s2["bands"],
                defaults=defaults,
                modality="s2",
            )
    if isinstance(inputs, dict) and "s1" in inputs:
        s1 = inputs["s1"]
        if isinstance(s1, dict) and "bands" in s1:
            collection = str(
                s1.get(
                    "collection",
                    "COPERNICUS/S1_GRD_FLOAT" if defaults.get("use_float_linear", True) else "COPERNICUS/S1_GRD",
                )
            )
            profiles["s1"] = _mk_sensor(
                collection=collection,
                bands=s1["bands"],
                defaults=defaults,
                modality="s1",
            )
    return profiles


def supports_modality_for_model(model_id: str, modality: str) -> bool:
    modality_n = _normalize_modality_name(modality)
    if modality_n is None:
        return False
    profiles = modality_profiles_for_model(model_id)
    if modality_n in profiles:
        return True
    desc = _probe_model_desc(model_id)
    default_modality = _normalize_modality_name(
        (desc.get("defaults") or {}).get("modality")
    )
    return modality_n == default_modality


def default_sensor_for_model(
    model_id: str, modality: Optional[str] = None
) -> Optional[SensorSpec]:
    desc = _probe_model_desc(model_id)

    typ = str(desc.get("type", "")).lower()
    if "precomputed" in typ:
        return None

    defaults = desc.get("defaults", {}) or {}
    profiles = modality_profiles_for_model(model_id)
    requested_modality = _normalize_modality_name(modality)
    default_modality = _normalize_modality_name(defaults.get("modality"))

    if requested_modality is not None:
        if requested_modality in profiles:
            return profiles[requested_modality]
        if default_modality == requested_modality:
            base = default_sensor_for_model(model_id, modality=None)
            if base is None:
                return None
            return replace(base, modality=requested_modality)
        return None

    if default_modality is not None and default_modality in profiles:
        return profiles[default_modality]

    inputs = desc.get("inputs")

    if isinstance(inputs, dict) and "collection" in inputs and "bands" in inputs:
        return _mk_sensor(
            collection=str(inputs["collection"]),
            bands=inputs["bands"],
            defaults=defaults,
        )
    if isinstance(inputs, dict) and "s2_sr" in inputs:
        s2 = inputs["s2_sr"]
        if isinstance(s2, dict) and "collection" in s2 and "bands" in s2:
            return _mk_sensor(
                collection=str(s2["collection"]),
                bands=s2["bands"],
                defaults=defaults,
                modality="s2",
            )
    if isinstance(inputs, dict) and "provider_default" in inputs:
        provider_default = inputs["provider_default"]
        if (
            isinstance(provider_default, dict)
            and "collection" in provider_default
            and "bands" in provider_default
        ):
            return _mk_sensor(
                collection=str(provider_default["collection"]),
                bands=provider_default["bands"],
                defaults=defaults,
            )
    if "input_bands" in desc:
        return _mk_sensor(
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=desc["input_bands"],
            defaults=defaults,
        )

    return None


def resolve_sensor_for_model(
    model_id: str,
    *,
    sensor: Optional[SensorSpec],
    modality: Optional[str] = None,
    default_when_missing: bool = False,
) -> Optional[SensorSpec]:
    sensor_modality = _normalize_modality_name(getattr(sensor, "modality", None))
    requested_modality = _normalize_modality_name(modality) or sensor_modality

    if (
        requested_modality is not None
        and sensor_modality is not None
        and requested_modality != sensor_modality
    ):
        raise ModelError(
            f"Conflicting modality values: sensor.modality={sensor_modality!r}, modality={requested_modality!r}."
        )

    if requested_modality is not None and not supports_modality_for_model(
        model_id, requested_modality
    ):
        raise ModelError(
            f"Model '{model_id}' does not expose modality={requested_modality!r}."
        )

    if sensor is not None:
        if requested_modality is None or sensor.modality == requested_modality:
            return sensor
        return replace(sensor, modality=requested_modality)

    if requested_modality is not None:
        resolved = default_sensor_for_model(model_id, modality=requested_modality)
        if resolved is None:
            raise ModelError(
                f"Model '{model_id}' has no default sensor for modality={requested_modality!r}."
            )
        return resolved

    if default_when_missing:
        return default_sensor_for_model(model_id)

    return None

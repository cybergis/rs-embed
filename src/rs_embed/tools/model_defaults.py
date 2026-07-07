from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from ..core.errors import ModelError
from ..core.registry import get_embedder_cls
from ..core.specs import FetchSpec, SensorSpec


def _probe_model_desc(model_id: str) -> dict:
    cls = get_embedder_cls(model_id)
    try:
        desc = cls().describe() or {}
    except Exception as _e:
        desc = {}
    return desc if isinstance(desc, dict) else {}


def _normalize_modality_name(modality: str | None) -> str | None:
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
    modality: str | None = None,
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
        s1_require_iw=bool(defaults.get("s1_require_iw", True)),
        s1_relax_iw_on_empty=bool(defaults.get("s1_relax_iw_on_empty", True)),
    )


def apply_fetch_to_sensor(sensor: SensorSpec, fetch: FetchSpec | None) -> SensorSpec:
    """Apply a :class:`FetchSpec` override to a :class:`SensorSpec`.

    Only fields explicitly set on *fetch* (non-``None``) are applied; all
    other sensor fields are preserved unchanged.

    Parameters
    ----------
    sensor : SensorSpec
        Base sensor configuration to update.
    fetch : FetchSpec or None
        Lightweight fetch-policy override. Returns *sensor* unchanged when
        ``None``.

    Returns
    -------
    SensorSpec
        Updated sensor with fetch-policy fields applied.
    """
    if fetch is None:
        return sensor

    updates: dict[str, object] = {}
    if fetch.scale_m is not None:
        updates["scale_m"] = int(fetch.scale_m)
    if fetch.cloudy_pct is not None:
        updates["cloudy_pct"] = int(fetch.cloudy_pct)
    if fetch.fill_value is not None:
        updates["fill_value"] = float(fetch.fill_value)
    if fetch.composite is not None:
        updates["composite"] = str(fetch.composite)

    if not updates:
        return sensor
    return replace(sensor, **updates)


def _fetch_override_sensor_for_model(model_id: str) -> SensorSpec | None:
    desc = _probe_model_desc(model_id)
    source = str(desc.get("source", "")).strip()
    defaults = desc.get("defaults", {}) or {}

    if source == "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL":
        return SensorSpec(
            collection=source,
            bands=tuple(),
            scale_m=int(defaults.get("scale_m", 10)),
            cloudy_pct=int(defaults.get("cloudy_pct", 100)),
            composite=str(defaults.get("composite", "mosaic")),
            fill_value=float(defaults.get("fill_value", -9999.0)),
        )
    return None


def modality_profiles_for_model(model_id: str) -> dict[str, SensorSpec]:
    """Return the named sensor profiles for each modality exposed by a model.

    Parameters
    ----------
    model_id : str
        Canonical model identifier.

    Returns
    -------
    dict[str, SensorSpec]
        Mapping from normalized modality name (e.g. ``"s2"``, ``"s1"``) to
        the corresponding default :class:`SensorSpec`. Returns an empty dict
        for precomputed models or models with no explicit modality profiles.
    """
    desc = _probe_model_desc(model_id)

    typ = str(desc.get("type", "")).lower()
    if "precomputed" in typ:
        return {}

    inputs = desc.get("inputs")
    defaults = desc.get("defaults", {}) or {}
    profiles: dict[str, SensorSpec] = {}

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
                    "COPERNICUS/S1_GRD_FLOAT"
                    if defaults.get("use_float_linear", True)
                    else "COPERNICUS/S1_GRD",
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
    """Return whether a model supports a given modality name.

    Parameters
    ----------
    model_id : str
        Canonical model identifier.
    modality : str
        Modality name to check (e.g. ``"s1"``, ``"s2"``).

    Returns
    -------
    bool
        ``True`` if the model exposes the modality as a named profile or as
        its default modality.
    """
    modality_n = _normalize_modality_name(modality)
    if modality_n is None:
        return False
    profiles = modality_profiles_for_model(model_id)
    if modality_n in profiles:
        return True
    desc = _probe_model_desc(model_id)
    default_modality = _normalize_modality_name((desc.get("defaults") or {}).get("modality"))
    return modality_n == default_modality


def default_sensor_for_model(model_id: str, modality: str | None = None) -> SensorSpec | None:
    """Return the default :class:`SensorSpec` for a model and optional modality.

    Parameters
    ----------
    model_id : str
        Canonical model identifier.
    modality : str or None
        Optional modality name. When provided, returns the sensor profile for
        that specific modality rather than the overall default.

    Returns
    -------
    SensorSpec or None
        Default sensor configuration, or ``None`` for precomputed models or
        models that do not declare input collection/band metadata.
    """
    desc = _probe_model_desc(model_id)
    cls = get_embedder_cls(model_id)

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

    if requested_modality is None:
        try:
            emb = cls()
            default_sensor = getattr(emb, "_default_sensor", None)
            if callable(default_sensor):
                sensor = default_sensor()
                if isinstance(sensor, SensorSpec):
                    return sensor
        except Exception as _e:
            pass

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
    sensor: SensorSpec | None,
    fetch: FetchSpec | None = None,
    modality: str | None = None,
    default_when_missing: bool = False,
) -> SensorSpec | None:
    """Resolve the effective :class:`SensorSpec` for a model call.

    Combines explicit ``sensor`` / ``fetch`` overrides with per-model
    defaults and modality profiles, applying validation along the way.

    Parameters
    ----------
    model_id : str
        Canonical model identifier.
    sensor : SensorSpec or None
        Explicit sensor override. Mutually exclusive with *fetch*.
    fetch : FetchSpec or None
        Lightweight fetch-policy override applied on top of the model
        default. Mutually exclusive with *sensor*.
    modality : str or None
        Optional modality selector (e.g. ``"s1"``, ``"s2"``).
    default_when_missing : bool
        When ``True`` and no explicit sensor is provided, fall back to the
        model's default sensor instead of returning ``None``.

    Returns
    -------
    SensorSpec or None
        Resolved sensor configuration, or ``None`` when no sensor is
        applicable (e.g. precomputed models without a provider backend).

    Raises
    ------
    ModelError
        If *sensor* and *fetch* are both provided, or if the requested
        modality is unsupported or ambiguous.
    """
    if sensor is not None and fetch is not None:
        raise ModelError("Use either sensor=... or fetch=..., not both.")

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
        raise ModelError(f"Model '{model_id}' does not expose modality={requested_modality!r}.")

    if sensor is not None:
        if requested_modality is None or sensor.modality == requested_modality:
            return sensor
        return replace(sensor, modality=requested_modality)

    if fetch is not None:
        resolved = _fetch_override_sensor_for_model(model_id)
        if resolved is None:
            resolved = default_sensor_for_model(model_id, modality=requested_modality)
        if resolved is None:
            raise ModelError(f"Model '{model_id}' does not support fetch=... overrides.")
        return apply_fetch_to_sensor(resolved, fetch)

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

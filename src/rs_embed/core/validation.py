from __future__ import annotations

from .errors import ModelError
from .specs import OutputSpec, SpatialSpec, TemporalSpec
from ..providers import has_provider

def validate_specs(
    *, spatial: SpatialSpec, temporal: TemporalSpec | None, output: OutputSpec
) -> None:
    """Validate spatial/temporal/output specs before inference.

    Parameters
    ----------
    spatial : SpatialSpec
        Spatial request definition.
    temporal : TemporalSpec or None
        Optional temporal constraint.
    output : OutputSpec
        Output shape/pooling policy.

    Raises
    ------
    ModelError
        If any spec is malformed or contains unsupported values at the
        interface level (for example invalid types or unsupported output
        configuration).
    SpecError
        If the underlying spatial or temporal spec validation fails.
    """

    if not hasattr(spatial, "validate"):
        raise ModelError(f"Invalid spatial spec type: {type(spatial)}")
    spatial.validate()  # type: ignore[call-arg]

    if temporal is not None:
        temporal.validate()

    if output.mode not in ("grid", "pooled"):
        raise ModelError(f"Unknown output mode: {output.mode}")
    if output.scale_m <= 0:
        raise ModelError("output.scale_m must be positive.")
    if output.mode == "pooled" and output.pooling not in ("mean", "max"):
        raise ModelError(f"Unknown pooling method: {output.pooling}")
    if getattr(output, "grid_orientation", "north_up") not in ("north_up", "native"):
        raise ModelError(
            f"Unknown grid orientation policy: {getattr(output, 'grid_orientation', None)}"
        )

def validate_spatial_list(
    *,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None,
    output: OutputSpec,
) -> None:
    """Validate a non-empty list of spatial specs against shared settings."""
    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty list[SpatialSpec].")
    for spatial in spatials:
        validate_specs(spatial=spatial, temporal=temporal, output=output)

def assert_supported(
    embedder, *, backend: str, output: OutputSpec, temporal: TemporalSpec | None
) -> None:
    """Check whether an embedder supports the requested execution settings.

    Parameters
    ----------
    embedder : Any
        Embedder instance exposing ``describe()`` metadata.
    backend : str
        Requested backend (for example ``"auto"`` or ``"gee"``).
    output : OutputSpec
        Requested output mode/pooling configuration.
    temporal : TemporalSpec or None
        Optional temporal request to validate against model constraints.

    Raises
    ------
    ModelError
        If the embedder metadata cannot be read or does not support the
        requested backend/output/temporal configuration.
    """

    try:
        desc = embedder.describe() or {}
    except Exception as e:
        name = getattr(embedder, "model_name", type(embedder).__name__)
        raise ModelError(
            f"Model '{name}' describe() failed during capability validation: {e}"
        ) from e

    backends = desc.get("backend")
    if isinstance(backends, list):
        allowed = [str(b).lower() for b in backends]
        auto_provider_compatible = backend == "auto" and ("provider" in allowed or "gee" in allowed)
        provider_compatible = has_provider(backend) and ("provider" in allowed or "gee" in allowed)
        if backend not in allowed and not provider_compatible and not auto_provider_compatible:
            raise ModelError(
                f"Model '{embedder.model_name}' does not support backend='{backend}'. Supported: {backends}"
            )

    outputs = desc.get("output")
    if isinstance(outputs, list) and output.mode not in outputs:
        raise ModelError(
            f"Model '{embedder.model_name}' does not support output.mode='{output.mode}'. Supported: {outputs}"
        )

    temporal_hint = desc.get("temporal")
    if isinstance(temporal_hint, dict) and "mode" in temporal_hint:
        mode_hint = str(temporal_hint["mode"])
        if (
            "year" in mode_hint
            and temporal is not None
            and getattr(temporal, "mode", None) != "year"
        ):
            raise ModelError(
                f"Model '{embedder.model_name}' expects TemporalSpec.mode='year' (or None)."
            )

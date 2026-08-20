"""Match user-provided imagery declarations against model input sensors.

The bring-your-own-data entrypoints let callers compute embeddings from
arrays they already have instead of provider-fetched imagery. The caller
declares what the array is (:class:`~rs_embed.core.types.UserData`:
collection + band names, raw provider units); this module decides whether
that declaration satisfies a model's resolved :class:`SensorSpec` and, when
it does, which user channels to feed the model in which order.

Policy: the model's required bands must be a subset of the declared bands
(superset data is sliced and reordered automatically); a collection mismatch
or a missing band refuses the request with a :class:`ModelError` naming what
is missing. Band vocabulary is shared with provider fetches via
:func:`~rs_embed.providers.gee_utils.resolve_band_aliases`.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..core.errors import ModelError
from ..core.specs import SensorSpec
from ..core.types import UserData
from ..providers.gee_utils import resolve_band_aliases

# Short user-facing aliases for provider collection ids. Full ids always pass
# through unchanged, so this stays a convenience layer, not a registry.
_COLLECTION_ALIASES: dict[str, str] = {
    "s2": "COPERNICUS/S2_SR_HARMONIZED",
    "s2_sr": "COPERNICUS/S2_SR_HARMONIZED",
    "s2_l2a": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel_2": "COPERNICUS/S2_SR_HARMONIZED",
    "s1": "COPERNICUS/S1_GRD",
    "s1_grd": "COPERNICUS/S1_GRD",
    "sentinel1": "COPERNICUS/S1_GRD",
    "sentinel_1": "COPERNICUS/S1_GRD",
}

# Collections whose raw units are surface-reflectance DN in 0..10000; used
# only for the best-effort "looks already normalized" warning below.
_DN_0_10000_COLLECTION_MARKERS: tuple[str, ...] = ("COPERNICUS/S2",)


def normalize_collection_id(collection: str) -> str:
    """Resolve a user-facing collection alias to a full collection id."""
    raw = str(collection or "").strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    return _COLLECTION_ALIASES.get(key, raw)


def canonical_band_names(collection_id: str, bands: tuple[str, ...]) -> tuple[str, ...]:
    """Alias-resolve band names and fold case for comparison."""
    resolved = resolve_band_aliases(collection_id, tuple(str(b) for b in bands))
    return tuple(b.upper() for b in resolved)


def match_user_data_to_sensor(
    data: UserData,
    sensor: SensorSpec,
    *,
    model_name: str,
) -> tuple[int, ...]:
    """Match a user-data declaration against a model's input sensor.

    Parameters
    ----------
    data : UserData
        User declaration (collection, bands, array). Must already be
        validated via :meth:`UserData.validate`.
    sensor : SensorSpec
        The model's resolved input sensor to satisfy.
    model_name : str
        Model name used in refusal messages.

    Returns
    -------
    tuple[int, ...]
        Channel indices into the user array's band axis, in the model's band
        order, suitable for
        :func:`~rs_embed.providers.prefetch_plan.select_prefetched_channels`.

    Raises
    ------
    ModelError
        If the declared collection does not match the model's, the
        declaration repeats a band name, or a required band is missing.
    """
    user_collection = normalize_collection_id(data.collection)
    model_collection = normalize_collection_id(sensor.collection)
    if user_collection.upper() != model_collection.upper():
        raise ModelError(
            f"Model '{model_name}' expects imagery from collection "
            f"'{sensor.collection}', but the provided data is declared as "
            f"'{data.collection}'. Raw units differ across collections, so "
            "this data cannot serve the model."
        )

    user_bands = canonical_band_names(user_collection, tuple(data.bands))
    if len(set(user_bands)) != len(user_bands):
        dupes = sorted({b for b in user_bands if user_bands.count(b) > 1})
        raise ModelError(
            f"UserData.bands declares duplicate band name(s) {dupes}; "
            "channel selection would be ambiguous."
        )
    model_bands = canonical_band_names(model_collection, tuple(sensor.bands))

    missing = [b for b in model_bands if b not in user_bands]
    if missing:
        raise ModelError(
            f"Model '{model_name}' needs bands {list(model_bands)} from "
            f"'{sensor.collection}', but the provided data lacks {missing} "
            f"(declared bands: {list(user_bands)})."
        )
    return tuple(user_bands.index(b) for b in model_bands)


def warn_on_suspicious_value_range(data: UserData) -> None:
    """Warn when values look already normalized for a raw-DN collection.

    The user-data contract expects raw provider units; reflectance already
    scaled to ``0..1`` fed into a DN-normalizing embedder produces silently
    wrong embeddings, which this best-effort check surfaces early.
    """
    collection_id = normalize_collection_id(data.collection).upper()
    if not any(marker in collection_id for marker in _DN_0_10000_COLLECTION_MARKERS):
        return
    arr = np.asarray(data.data)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    max_value = float(finite.max())
    if 0.0 < max_value <= 1.5:
        warnings.warn(
            f"UserData declared as '{data.collection}' has max value "
            f"{max_value:.3g}; raw surface-reflectance DN (0..10000) is "
            "expected. If your data is scaled reflectance, multiply by 10000 "
            "before embedding.",
            UserWarning,
            stacklevel=3,
        )

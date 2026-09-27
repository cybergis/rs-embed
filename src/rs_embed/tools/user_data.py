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

Georeferenced declarations (``crs`` + ``transform``) are also put on the
package's common grid here (:func:`align_to_common_grid`), so a user raster
reaches the model on the same EPSG:3857 lattice a provider fetch would have
produced.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..core.errors import ModelError, SpecError
from ..core.specs import SensorSpec, SpatialSpec
from ..core.types import UserData
from ..providers.gee_utils import resolve_band_aliases
from .projection import (
    common_grid,
    crs_label,
    raster_bounds_4326,
    resample_to_common_grid,
    warn_if_utm_boundary,
)

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

# Canonical full band order per collection, used when a declaration omits
# ``bands``. Only collections with one unambiguous canonical order belong
# here; a declaration whose channel order differs must name its bands.
_DEFAULT_BANDS_BY_COLLECTION: dict[str, tuple[str, ...]] = {
    "COPERNICUS/S2_SR_HARMONIZED": (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    ),
}


def normalize_collection_id(collection: str) -> str:
    """Resolve a user-facing collection alias to a full collection id."""
    raw = str(collection or "").strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    return _COLLECTION_ALIASES.get(key, raw)


def canonical_band_names(collection_id: str, bands: tuple[str, ...]) -> tuple[str, ...]:
    """Alias-resolve band names and fold case for comparison."""
    resolved = resolve_band_aliases(collection_id, tuple(str(b) for b in bands))
    return tuple(b.upper() for b in resolved)


def resolve_declared_bands(data: UserData) -> tuple[str, ...]:
    """Return the band names a declaration covers, defaulting when omitted.

    A ``bands=None`` declaration means "the collection's canonical full band
    order"; that default only exists for collections listed in
    ``_DEFAULT_BANDS_BY_COLLECTION`` and only when the array's channel count
    matches exactly — anything else must name its bands, because guessing
    band identity from channel count is precisely the silent-wrongness this
    layer exists to refuse.

    Raises
    ------
    SpecError
        If ``bands`` is omitted and the collection has no canonical order,
        or the channel count does not match that order.
    """
    if data.bands is not None:
        return tuple(data.bands)
    collection_id = normalize_collection_id(data.collection)
    default = _DEFAULT_BANDS_BY_COLLECTION.get(collection_id.upper())
    if default is None:
        raise SpecError(
            f"UserData.bands was omitted, but collection '{data.collection}' "
            "has no canonical band order to default to; declare one band "
            "name per channel."
        )
    channels = int(np.asarray(data.data).shape[-3])
    if channels != len(default):
        raise SpecError(
            f"UserData.bands was omitted; the canonical order for "
            f"'{collection_id}' has {len(default)} bands {list(default)}, but "
            f"the array has {channels} channels. Declare bands explicitly."
        )
    return default


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
    SpecError
        If ``bands`` was omitted and no canonical default applies (see
        :func:`resolve_declared_bands`).
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

    user_bands = canonical_band_names(user_collection, resolve_declared_bands(data))
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


def resolve_user_data_spatial(data: UserData) -> SpatialSpec | None:
    """Where the declaration is: ``spatial`` if given, else the raster footprint.

    A georeferenced declaration (``crs`` + ``transform``) carries its own
    location, so ``spatial`` need not be repeated; an explicit ``spatial``
    still wins when both are present.
    """
    if data.spatial is not None or data.crs is None:
        return data.spatial
    return raster_bounds_4326(data.crs, data.transform, np.asarray(data.data).shape[-2:])


def align_to_common_grid(
    array: np.ndarray,
    data: UserData,
    *,
    scale_m: float,
    fill_value: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample a georeferenced user array onto the common grid at *scale_m*.

    Provider fetches return pixels on the common EPSG:3857 lattice at the
    model's scale; a raster declared with ``crs``/``transform`` is put on that
    same grid (nearest neighbour, *fill_value* outside the raster) so the model
    sees what a fetch would have produced. Declarations without georeferencing
    pass through untouched. Returns the array and provenance for
    ``meta['user_input']['projection']``.
    """
    if data.crs is None:
        return array, {"crs": None, "aligned": False}
    src_label = crs_label(data.crs)
    footprint = raster_bounds_4326(data.crs, data.transform, np.asarray(array).shape[-2:])
    warn_if_utm_boundary(crss=[src_label], footprint=footprint, context=f"user data in {src_label}")
    grid = common_grid(footprint, scale_m=scale_m)
    out = resample_to_common_grid(
        array,
        src_crs=data.crs,
        src_transform=data.transform,
        grid=grid,
        fill_value=fill_value,
    )
    return out, {
        "crs": src_label,
        "input_transform": data.transform,
        "input_hw": tuple(int(v) for v in np.asarray(array).shape[-2:]),
        "aligned": True,
        "resampling": "nearest",
        **grid.meta(),
    }

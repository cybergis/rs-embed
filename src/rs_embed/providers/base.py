from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.errors import ProviderError
from ..core.specs import SensorSpec, SpatialSpec, TemporalSpec


class ProviderBase:
    """Base interface for provider-specific data access implementations."""

    name: str = "base"

    def ensure_ready(self) -> None:
        """Initialize provider dependencies or authenticate remote services.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete provider subclasses.
        """

        raise NotImplementedError

    def get_region(self, spatial: SpatialSpec) -> Any:
        """Convert a spatial spec into a provider-native region object.

        Parameters
        ----------
        spatial : SpatialSpec
            Spatial request definition.

        Returns
        -------
        Any
            Provider-native geometry/region representation.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete provider subclasses.
        """

        raise NotImplementedError

    def build_image(
        self,
        *,
        sensor: SensorSpec,
        temporal: TemporalSpec | None,
        region: Any | None = None,
    ) -> Any:
        """Build a provider-native image object for a sensor/time request.

        Parameters
        ----------
        sensor : SensorSpec
            Sensor/collection and band configuration.
        temporal : TemporalSpec or None
            Optional temporal filter.
        region : Any or None
            Optional provider-native region constraint.

        Returns
        -------
        Any
            Provider-native image/collection object.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete provider subclasses.
        """

        raise NotImplementedError

    def fetch_array_chw(
        self,
        *,
        image: Any,
        bands: tuple[str, ...],
        region: Any,
        scale_m: int,
        fill_value: float,
        collection: str | None = None,
    ) -> np.ndarray:
        """Fetch raster pixels as a ``[C,H,W]`` float array.

        Parameters
        ----------
        image : Any
            Provider-native image object.
        bands : tuple[str, ...]
            Band names and order to fetch.
        region : Any
            Provider-native region object.
        scale_m : int
            Pixel scale in meters.
        fill_value : float
            Fill value for missing pixels.
        collection : str or None
            Optional collection name for provider-specific behavior.

        Returns
        -------
        np.ndarray
            Array with shape ``[C,H,W]``.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete provider subclasses.
        """

        raise NotImplementedError

    def normalize_bands(
        self,
        *,
        collection: str,
        bands: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize band aliases to provider-preferred names.

        Parameters
        ----------
        collection : str
            Collection identifier.
        bands : tuple[str, ...]
            User requested band names.

        Returns
        -------
        tuple[str, ...]
            Normalized band names in the same order.
        """
        return tuple(str(b) for b in bands)

    def fetch_multiframe_collection_raw_tchw(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec,
        collection: str,
        bands: Sequence[str],
        n_frames: int = 8,
        scale_m: int = 10,
        cloudy_pct: int | None = 30,
        composite: str = "median",
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Fetch multi-frame collection patch as ``[T,C,H,W]``.

        Parameters
        ----------
        spatial : SpatialSpec
            Spatial request definition.
        temporal : TemporalSpec
            Temporal range/year request.
        collection : str
            Collection identifier.
        bands : Sequence[str]
            Band names and order to fetch.
        n_frames : int
            Number of temporal frames to sample.
        scale_m : int
            Pixel scale in meters.
        cloudy_pct : int or None
            Optional cloud threshold.
        composite : str
            Temporal compositing strategy.
        fill_value : float
            Fill value for missing pixels.

        Returns
        -------
        np.ndarray
            Array with shape ``[T,C,H,W]``.

        Raises
        ------
        ProviderError
            If this provider does not support multi-frame collection fetch.
        """
        raise ProviderError(
            f"Provider '{self.name}' does not implement multi-frame collection fetch support."
        )

    def fetch_collection_patch_all_bands_chw(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        collection: str,
        scale_m: int = 10,
        fill_value: float = 0.0,
        composite: str = "median",
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Fetch a collection patch containing all available bands.

        Parameters
        ----------
        spatial : SpatialSpec
            Spatial request definition.
        temporal : TemporalSpec or None
            Optional temporal filter.
        collection : str
            Collection identifier.
        scale_m : int
            Pixel scale in meters.
        fill_value : float
            Fill value for missing pixels.
        composite : str
            Temporal compositing strategy.

        Returns
        -------
        tuple[np.ndarray, tuple[str, ...]]
            ``(array_chw, band_names)`` tuple.

        Raises
        ------
        ProviderError
            If the provider does not implement all-band collection patch fetch support.
        """
        raise ProviderError(
            f"Provider '{self.name}' does not implement all-band collection patch fetch support."
        )

    def fetch_sensor_patch_chw(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec,
        to_float_image: bool = False,
    ) -> np.ndarray:
        """Fetch a sensor patch and validate shape/channel count.

        Parameters
        ----------
        spatial : SpatialSpec
            Spatial request definition.
        temporal : TemporalSpec or None
            Optional temporal filter.
        sensor : SensorSpec
            Sensor/collection and expected band configuration.
        to_float_image : bool
            If ``True``, request float image conversion when available.

        Returns
        -------
        np.ndarray
            Sanitized float32 array with shape ``[C,H,W]``.

        Raises
        ------
        ProviderError
            If this provider does not implement sensor patch fetch support.
        """
        raise ProviderError(
            f"Provider '{self.name}' does not implement sensor patch fetch support."
        )

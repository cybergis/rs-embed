from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..core.errors import ProviderError
from ..core.specs import SensorSpec, SpatialSpec, TemporalSpec


class ProviderBase:
    name: str = "base"

    def ensure_ready(self) -> None:
        raise NotImplementedError

    def get_region(self, spatial: SpatialSpec) -> Any:
        raise NotImplementedError

    def build_image(
        self,
        *,
        sensor: SensorSpec,
        temporal: Optional[TemporalSpec],
        region: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError

    def fetch_array_chw(
        self,
        *,
        image: Any,
        bands: Tuple[str, ...],
        region: Any,
        scale_m: int,
        fill_value: float,
        collection: Optional[str] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def normalize_bands(
        self,
        *,
        collection: str,
        bands: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        """Best-effort band normalization for provider-specific aliases."""
        return tuple(str(b) for b in bands)

    def fetch_s1_vvvh_raw_chw(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec,
        scale_m: int = 10,
        orbit: Optional[str] = None,
        use_float_linear: bool = True,
        composite: str = "median",
        fill_value: float = 0.0,
    ) -> np.ndarray:
        raise ProviderError(
            f"Provider '{self.name}' does not implement Sentinel-1 VV/VH fetch support."
        )

    def fetch_multiframe_collection_raw_tchw(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec,
        collection: str,
        bands: Sequence[str],
        n_frames: int = 8,
        scale_m: int = 10,
        cloudy_pct: Optional[int] = 30,
        composite: str = "median",
        fill_value: float = 0.0,
    ) -> np.ndarray:
        raise ProviderError(
            f"Provider '{self.name}' does not implement multi-frame collection fetch support."
        )

    def fetch_collection_patch_all_bands_chw(
        self,
        *,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        collection: str,
        scale_m: int = 10,
        fill_value: float = 0.0,
        composite: str = "median",
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        raise ProviderError(
            f"Provider '{self.name}' does not implement all-band collection patch fetch support."
        )

    def fetch_sensor_patch_chw(
        self,
        *,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        sensor: SensorSpec,
        to_float_image: bool = False,
    ) -> np.ndarray:
        from ..internal.api.api_helpers import fetch_provider_patch_raw

        x = fetch_provider_patch_raw(
            self,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            to_float_image=bool(to_float_image),
        )
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 3:
            raise ProviderError(f"Expected CHW array from provider fetch, got shape={getattr(arr, 'shape', None)}")
        if int(arr.shape[0]) != len(sensor.bands):
            raise ProviderError(
                f"Provider fetch channel mismatch: got C={int(arr.shape[0])}, "
                f"expected C={len(sensor.bands)} for collection={sensor.collection}"
            )
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pyproj import Transformer

from ..core.errors import ProviderError, SpecError
from ..core.specs import BBox, PointBuffer, SensorSpec, SpatialSpec, TemporalSpec
from ..tools.temporal import split_date_range as _split_date_range_core
from .base import ProviderBase

_ALIAS_S2 = {
    "BLUE": "B2",
    "GREEN": "B3",
    "RED": "B4",
    # NIR
    "NIR": "B8",
    "NIR_BROAD": "B8",
    "NIR_WIDE": "B8",
    # Narrow NIR (S2 band B8A)
    "NIR_NARROW": "B8A",
    "NIRN": "B8A",
    "NIRNARROW": "B8A",
    "NIR_N": "B8A",
    # Red edge (optional but common)
    "RE1": "B5",
    "RED_EDGE_1": "B5",
    "RE2": "B6",
    "RED_EDGE_2": "B6",
    "RE3": "B7",
    "RED_EDGE_3": "B7",
    "RE4": "B8A",
    "RED_EDGE_4": "B8A",
    # SWIR
    "SWIR1": "B11",
    "SWIR_1": "B11",
    "SWIR2": "B12",
    "SWIR_2": "B12",
}
_ALIAS_LS89_SR = {
    "BLUE": "SR_B2",
    "GREEN": "SR_B3",
    "RED": "SR_B4",
    "NIR": "SR_B5",
    "SWIR1": "SR_B6",
    "SWIR2": "SR_B7",
}

_ALIAS_LS457_SR = {
    "BLUE": "SR_B1",
    "GREEN": "SR_B2",
    "RED": "SR_B3",
    "NIR": "SR_B4",
    "SWIR1": "SR_B5",
    "SWIR2": "SR_B7",
}

_NO_IMAGES_FOUND_MSG = "No images found for the selected region/time window."

def _resolve_band_aliases(collection: str, bands: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve semantic band aliases to real band names based on collection id."""
    if not bands:
        return bands

    c = (collection or "").upper()
    # Sentinel-2 (SR/TOA/HARMONIZED etc.)
    if "COPERNICUS/S2" in c:
        amap = _ALIAS_S2
    # Landsat Collection 2 L2 SR (typical ids)
    elif "LANDSAT/LC08/C02/T1_L2" in c or "LANDSAT/LC09/C02/T1_L2" in c:
        amap = _ALIAS_LS89_SR
    elif (
        "LANDSAT/LE07/C02/T1_L2" in c
        or "LANDSAT/LT05/C02/T1_L2" in c
        or "LANDSAT/LT04/C02/T1_L2" in c
    ):
        amap = _ALIAS_LS457_SR
    else:
        # Unknown collection: do not map
        amap = {}

    out = []
    for b in bands:
        key = (b or "").upper()
        out.append(amap.get(key, b))
    return tuple(out)

def _split_date_range(start: str, end: str, n_parts: int) -> tuple[tuple[str, str], ...]:
    try:
        return _split_date_range_core(start, end, n_parts)
    except SpecError as e:
        raise ProviderError(str(e)) from e

def _no_images_found_message(*, collection: str | None = None) -> str:
    if collection:
        return f"{_NO_IMAGES_FOUND_MSG} collection={collection!r}"
    return _NO_IMAGES_FOUND_MSG

def _collection_size_or_none(col: Any) -> int | None:
    try:
        return int(col.size().getInfo())
    except Exception as _e:
        return None

def _raise_if_empty_collection(col: Any, *, collection: str | None = None) -> None:
    n = _collection_size_or_none(col)
    if n == 0:
        raise ProviderError(_no_images_found_message(collection=collection))

def _sample_image_bands_raw_chw(
    img: Any,
    *,
    region: Any,
    bands: Sequence[str],
    scale_m: int,
    fill_value: float,
) -> np.ndarray:
    img = img.select(list(bands)).reproject(crs="EPSG:3857", scale=int(scale_m))
    rect = img.sampleRectangle(region=region, defaultValue=float(fill_value)).getInfo()
    props = rect.get("properties", {}) if isinstance(rect, dict) else {}
    if not props:
        raise ProviderError(_NO_IMAGES_FOUND_MSG)
    arrs = [np.array(props.get(str(b), []), dtype=np.float32) for b in bands]
    try:
        raw = np.stack(arrs, axis=0).astype(np.float32)
    except Exception as e:
        raise ProviderError("Failed to sample rectangle from GEE image.") from e
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(raw, 0.0, 10000.0).astype(np.float32)

class GEEProvider(ProviderBase):
    name = "gee"

    def __init__(self, auto_auth: bool = True):
        self.auto_auth = auto_auth

    def ensure_ready(self) -> None:
        try:
            import ee

            ee.Initialize()
        except Exception as _e:
            if not self.auto_auth:
                raise ProviderError(
                    "Earth Engine not initialized. Run `earthengine authenticate` and try again."
                )
            try:
                import geemap

                geemap.ee_initialize()
            except Exception as e:
                raise ProviderError(f"Failed to initialize GEE: {e!r}")

    def _to_ee_region_3857(self, spatial: SpatialSpec):
        import ee

        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        if isinstance(spatial, PointBuffer):
            spatial.validate()
            x, y = to_3857.transform(spatial.lon, spatial.lat)
            half = spatial.buffer_m
            minx, miny, maxx, maxy = x - half, y - half, x + half, y + half
            return ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:3857", geodesic=False)

        if isinstance(spatial, BBox):
            spatial.validate()
            minx, miny = to_3857.transform(spatial.minlon, spatial.minlat)
            maxx, maxy = to_3857.transform(spatial.maxlon, spatial.maxlat)
            return ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:3857", geodesic=False)

        raise ProviderError(f"Unsupported spatial type: {type(spatial)}")

    def get_region_3857(self, spatial: SpatialSpec):
        self.ensure_ready()
        return self._to_ee_region_3857(spatial)

    def get_region(self, spatial: SpatialSpec):
        return self.get_region_3857(spatial)

    def build_image(
        self,
        *,
        sensor: SensorSpec,
        temporal: TemporalSpec | None,
        region: Any | None = None,
    ) -> Any:
        """Build an ee.Image from SensorSpec and TemporalSpec."""
        import ee

        temporal_range: tuple[str, str] | None = None
        if temporal is not None:
            temporal.validate()
            if temporal.mode == "range":
                temporal_range = (temporal.start, temporal.end)
            elif temporal.mode == "year":
                y = int(temporal.year)
                temporal_range = (f"{y}-01-01", f"{y + 1}-01-01")
            else:
                raise ProviderError(f"Unknown TemporalSpec mode: {temporal.mode}")

        try:
            ic = ee.ImageCollection(sensor.collection)
            if region is not None:
                ic = ic.filterBounds(region)
            if temporal_range is not None:
                ic = ic.filterDate(temporal_range[0], temporal_range[1])
            if sensor.cloudy_pct is not None:
                try:
                    ic = ic.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", int(sensor.cloudy_pct)))
                except Exception as _e:
                    pass
            _raise_if_empty_collection(ic, collection=str(sensor.collection))

            if sensor.composite == "median":
                img = ic.median()
            elif sensor.composite == "mosaic":
                img = ic.mosaic()
            else:
                img = ic.median()
        except ProviderError:
            raise
        except Exception as _e:
            img = ee.Image(sensor.collection)

        return img

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
        """Download a rectangular patch as CHW array.

        - Resolves band aliases like BLUE/GREEN/RED -> B2/B3/B4 (S2) etc.
        - Forces deterministic pixel grid by reprojecting to EPSG:3857 at `scale_m`
          before sampleRectangle (prevents accidental (C,1,1)).
        """
        import ee

        # 1) Resolve aliases using collection hint when provided
        resolved = self.normalize_bands(
            collection=(collection or ""),
            bands=bands,
        )

        # 2) Select resolved bands (this will error at compute-time if typo)
        img = image.select(list(resolved))

        # 3) Force pixel grid at desired scale
        proj = ee.Projection("EPSG:3857").atScale(int(scale_m))
        img = img.reproject(proj).clip(region)

        # 4) Sample and build CHW
        rect = img.sampleRectangle(region=region, defaultValue=fill_value).getInfo()
        props = rect.get("properties", {})
        if not props:
            raise ProviderError(_no_images_found_message(collection=collection))

        arrs = []
        missing = []
        for b in resolved:
            if b not in props:
                missing.append(b)
            else:
                arrs.append(np.array(props[b], dtype=np.float32))

        if missing:
            avail = sorted(list(props.keys()))
            raise ProviderError(
                f"Band(s) {missing} not in sampled properties. "
                f"Requested={resolved}. Available bands={avail}"
            )

        return np.stack(arrs, axis=0)

    def normalize_bands(
        self,
        *,
        collection: str,
        bands: tuple[str, ...],
    ) -> tuple[str, ...]:
        return _resolve_band_aliases(collection, tuple(str(b) for b in bands))

    def fetch_s1_vvvh_raw_chw(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec,
        scale_m: int = 10,
        orbit: str | None = None,
        use_float_linear: bool = True,
        composite: str = "median",
        fill_value: float = 0.0,
    ) -> np.ndarray:
        import ee

        temporal.validate()

        region = self.get_region(spatial)
        collection_id = "COPERNICUS/S1_GRD_FLOAT" if bool(use_float_linear) else "COPERNICUS/S1_GRD"
        col = (
            ee.ImageCollection(collection_id)
            .filterDate(temporal.start, temporal.end)
            .filterBounds(region)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        if orbit:
            col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit))
        _raise_if_empty_collection(col, collection=collection_id)

        comp = str(composite).lower().strip()
        if comp == "median":
            img = col.median()
        elif comp == "mosaic":
            img = col.mosaic()
        else:
            raise ProviderError(f"Unknown composite='{composite}'. Use 'median' or 'mosaic'.")

        img = img.select(["VV", "VH"]).reproject(crs="EPSG:3857", scale=int(scale_m))
        rect = img.sampleRectangle(region=region, defaultValue=float(fill_value)).getInfo()
        props = rect.get("properties", {}) if isinstance(rect, dict) else {}
        if not props:
            raise ProviderError(_no_images_found_message(collection=collection_id))
        vv = np.array(props.get("VV", []), dtype=np.float32)
        vh = np.array(props.get("VH", []), dtype=np.float32)
        try:
            arr = np.stack([vv, vh], axis=0).astype(np.float32)
        except Exception as e:
            raise ProviderError("Failed to sample S1 VV/VH rectangle from GEE image.") from e
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if arr.ndim != 3 or int(arr.shape[0]) != 2:
            raise ProviderError(
                f"Expected S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}"
            )
        return arr

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
        import ee

        temporal.validate()

        region = self.get_region(spatial)
        resolved_bands = self.normalize_bands(
            collection=str(collection),
            bands=tuple(str(b) for b in bands),
        )
        col_all = (
            ee.ImageCollection(str(collection))
            .filterDate(temporal.start, temporal.end)
            .filterBounds(region)
        )
        if cloudy_pct is not None:
            try:
                col_all = col_all.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", int(cloudy_pct)))
            except Exception as _e:
                pass
        _raise_if_empty_collection(col_all, collection=str(collection))

        comp = str(composite).lower().strip()
        if comp not in {"median", "mosaic"}:
            raise ProviderError(f"Unknown composite='{composite}'. Use 'median' or 'mosaic'.")

        def _reduce(c: Any) -> Any:
            return c.median() if comp == "median" else c.mosaic()

        fallback_frame = None
        try:
            if int(col_all.size().getInfo()) > 0:
                fallback_frame = _sample_image_bands_raw_chw(
                    _reduce(col_all),
                    region=region,
                    bands=resolved_bands,
                    scale_m=scale_m,
                    fill_value=fill_value,
                )
        except Exception as _e:
            fallback_frame = None

        frames = []
        bins = _split_date_range(temporal.start, temporal.end, max(1, int(n_frames)))
        for start_i, end_i in bins:
            col_i = col_all.filterDate(start_i, end_i)
            try:
                has_data = int(col_i.size().getInfo()) > 0
            except Exception as _e:
                has_data = False

            if has_data:
                frames.append(
                    _sample_image_bands_raw_chw(
                        _reduce(col_i),
                        region=region,
                        bands=resolved_bands,
                        scale_m=scale_m,
                        fill_value=fill_value,
                    )
                )
            elif fallback_frame is not None:
                frames.append(fallback_frame.copy())

        if not frames:
            if fallback_frame is not None:
                frames = [fallback_frame.copy() for _ in range(max(1, int(n_frames)))]
            else:
                raise ProviderError(_no_images_found_message(collection=str(collection)))

        t = max(1, int(n_frames))
        if len(frames) < t:
            frames.extend([frames[-1].copy() for _ in range(t - len(frames))])
        elif len(frames) > t:
            frames = frames[:t]

        arr = np.stack(frames, axis=0).astype(np.float32)
        if arr.ndim != 4:
            raise ProviderError(f"Expected TCHW array, got shape={getattr(arr, 'shape', None)}")
        if int(arr.shape[1]) != len(tuple(resolved_bands)):
            raise ProviderError(
                f"Time series channel mismatch: got C={int(arr.shape[1])}, expected C={len(tuple(resolved_bands))}"
            )
        return arr

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
        import ee

        region = self.get_region(spatial)

        temporal_range: tuple[str, str] | None = None
        if temporal is not None:
            temporal.validate()
            if temporal.mode == "range":
                temporal_range = (temporal.start, temporal.end)
            elif temporal.mode == "year":
                y = int(temporal.year)
                temporal_range = (f"{y}-01-01", f"{y + 1}-01-01")
            else:
                raise ProviderError(f"Unknown TemporalSpec mode: {temporal.mode}")

        col = ee.ImageCollection(str(collection))
        col = col.filterBounds(region)
        if temporal_range is not None:
            col = col.filterDate(temporal_range[0], temporal_range[1])
        _raise_if_empty_collection(col, collection=str(collection))

        comp = str(composite).lower().strip()
        if comp == "median":
            img = col.median()
        elif comp == "mosaic":
            img = col.mosaic()
        else:
            raise ProviderError(f"Unknown composite='{composite}'. Use 'median' or 'mosaic'.")

        img = img.reproject(crs="EPSG:3857", scale=int(scale_m)).clip(region)
        band_names_raw = img.bandNames().getInfo()
        band_names = tuple(str(b) for b in (band_names_raw or []))
        if not band_names:
            raise ProviderError(f"No bands found for collection={collection!r}.")

        rect = img.sampleRectangle(region=region, defaultValue=float(fill_value)).getInfo()
        props = rect.get("properties", {}) if isinstance(rect, dict) else {}
        if not props:
            raise ProviderError(_no_images_found_message(collection=str(collection)))
        arrs = [np.array(props.get(b, []), dtype=np.float32) for b in band_names]
        try:
            arr = np.stack(arrs, axis=0).astype(np.float32)
        except Exception as e:
            raise ProviderError("Failed to sample rectangle for all collection bands.") from e
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr, band_names

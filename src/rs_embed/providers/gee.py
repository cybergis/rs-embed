from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from pyproj import Transformer

from ..core.errors import ModelError, ProviderError
from ..core.specs import BBox, PointBuffer, SensorSpec, SpatialSpec, TemporalSpec
from .base import ProviderBase
from .gee_utils import (
    _bbox_recursive_fallback,
    _build_s1_dualpol_collection,
    _collection_size_or_none,
    _fetch_with_bbox_fallback,
    _format_s1_empty_collection_message,
    _gee_error_message,
    _gee_init_kwargs,
    _no_images_found_message,
    _order_collection_for_mosaic,
    _raise_if_empty_collection,
    _resolve_band_aliases,
    _sample_image_bands_raw_chw,
    _split_date_range,
    _stitch_bbox_split_arrays,
    fetch_provider_patch_raw,
)


class GEEProvider(ProviderBase):
    name = "gee"

    def __init__(self, auto_auth: bool = True, project: str | None = None):
        self.auto_auth = auto_auth
        self.project = (
            project or os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )

    def ensure_ready(self) -> None:
        try:
            import ee

            ee.Initialize(**_gee_init_kwargs(self.project))
            return
        except Exception as e:
            if not self.auto_auth:
                raise ProviderError(_gee_error_message(e)) from None
            try:
                import geemap

                geemap.ee_initialize(**_gee_init_kwargs(self.project))
            except Exception as geemap_exc:
                raise ProviderError(_gee_error_message(geemap_exc)) from geemap_exc

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
                img = _order_collection_for_mosaic(ic).mosaic()
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

    def fetch_sensor_patch_chw(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec,
        to_float_image: bool = False,
    ) -> np.ndarray:
        return fetch_provider_patch_raw(
            self,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            to_float_image=bool(to_float_image),
        )

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
        require_iw: bool = True,
        relax_iw_on_empty: bool = True,
    ) -> np.ndarray:
        arr, _ = self.fetch_s1_vvvh_raw_chw_with_meta(
            spatial=spatial,
            temporal=temporal,
            scale_m=scale_m,
            orbit=orbit,
            use_float_linear=use_float_linear,
            composite=composite,
            fill_value=fill_value,
            require_iw=require_iw,
            relax_iw_on_empty=relax_iw_on_empty,
        )
        return arr

    def fetch_s1_vvvh_raw_chw_with_meta(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec,
        scale_m: int = 10,
        orbit: str | None = None,
        use_float_linear: bool = True,
        composite: str = "median",
        fill_value: float = 0.0,
        require_iw: bool = True,
        relax_iw_on_empty: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        def _fetch(sp: SpatialSpec) -> tuple[np.ndarray, dict[str, Any]]:
            return self._fetch_s1_impl(
                spatial=sp,
                temporal=temporal,
                scale_m=scale_m,
                orbit=orbit,
                use_float_linear=use_float_linear,
                composite=composite,
                fill_value=fill_value,
                require_iw=require_iw,
                relax_iw_on_empty=relax_iw_on_empty,
            )

        def _stitch(
            res_a: Any, res_b: Any, bbox: Any, axis: str, scale_m: int, fill_value: float
        ) -> tuple[np.ndarray, dict[str, Any]]:
            arr_a, meta = res_a
            arr_b, _ = res_b
            return _stitch_bbox_split_arrays(
                arr_a=arr_a,
                arr_b=arr_b,
                parent_spatial=bbox,
                axis=axis,
                scale_m=scale_m,
                fill_value=fill_value,
            ), meta

        return _bbox_recursive_fallback(
            spatial=spatial,
            scale_m=scale_m,
            fill_value=fill_value,
            fetch_fn=_fetch,
            stitch_fn=_stitch,
        )

    def _fetch_s1_impl(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec,
        scale_m: int = 10,
        orbit: str | None = None,
        use_float_linear: bool = True,
        composite: str = "median",
        fill_value: float = 0.0,
        require_iw: bool = True,
        relax_iw_on_empty: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import ee

        temporal.validate()

        region = self.get_region(spatial)
        collection_id = "COPERNICUS/S1_GRD_FLOAT" if bool(use_float_linear) else "COPERNICUS/S1_GRD"
        base = (
            ee.ImageCollection(collection_id)
            .filterDate(temporal.start, temporal.end)
            .filterBounds(region)
        )
        col = _build_s1_dualpol_collection(base, require_iw=bool(require_iw))
        iw_relaxed = False
        iw_applied = bool(require_iw)
        n = _collection_size_or_none(col)
        fallback_n: int | None = None
        if n == 0 and bool(require_iw) and bool(relax_iw_on_empty):
            relaxed_col = _build_s1_dualpol_collection(base, require_iw=False)
            fallback_n = _collection_size_or_none(relaxed_col)
            if fallback_n is not None and fallback_n > 0:
                col = relaxed_col
                n = fallback_n
                iw_relaxed = True
                iw_applied = False
        if n == 0:
            counts = {
                "base": _collection_size_or_none(base),
                "iw": _collection_size_or_none(base.filter(ee.Filter.eq("instrumentMode", "IW"))),
                "vv": _collection_size_or_none(
                    base.filter(ee.Filter.eq("instrumentMode", "IW")).filter(
                        ee.Filter.listContains("transmitterReceiverPolarisation", "VV")
                    )
                ),
                "vh": _collection_size_or_none(
                    base.filter(ee.Filter.eq("instrumentMode", "IW"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                ),
            }
            if fallback_n is not None:
                counts["vh_no_iw"] = fallback_n
            raise ProviderError(
                _format_s1_empty_collection_message(
                    collection_id=collection_id,
                    temporal=temporal,
                    counts=counts,
                    require_iw=bool(require_iw),
                    relax_iw_on_empty=bool(relax_iw_on_empty),
                )
            )

        comp = str(composite).lower().strip()
        if comp == "median":
            img = col.median()
        elif comp == "mosaic":
            img = _order_collection_for_mosaic(col).mosaic()
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
        meta = {
            "s1_iw_requested": bool(require_iw),
            "s1_iw_applied": bool(iw_applied),
            "s1_iw_relaxed_on_empty": bool(iw_relaxed),
            "s1_relax_iw_on_empty": bool(relax_iw_on_empty),
            "s1_orbit_requested": orbit,
            "s1_collection_used": collection_id,
        }
        return arr, meta

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
        def _do(sp: SpatialSpec) -> np.ndarray:
            return self._fetch_multiframe_impl(
                spatial=sp,
                temporal=temporal,
                collection=collection,
                bands=bands,
                n_frames=n_frames,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
                fill_value=fill_value,
            )

        return _fetch_with_bbox_fallback(
            spatial=spatial,
            scale_m=int(scale_m),
            fill_value=float(fill_value),
            fetch_fn=_do,
        )

    def _fetch_multiframe_impl(
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
            return c.median() if comp == "median" else _order_collection_for_mosaic(c).mosaic()

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
        def _fetch(sp: SpatialSpec) -> tuple[np.ndarray, tuple[str, ...]]:
            return self._fetch_all_bands_impl(
                spatial=sp,
                temporal=temporal,
                collection=collection,
                scale_m=scale_m,
                fill_value=fill_value,
                composite=composite,
            )

        def _stitch(
            res_a: Any, res_b: Any, bbox: Any, axis: str, scale_m: int, fill_value: float
        ) -> tuple[np.ndarray, tuple[str, ...]]:
            arr_a, names_a = res_a
            arr_b, names_b = res_b
            if tuple(names_a) != tuple(names_b):
                raise ModelError("Band names mismatch while stitching all-band bbox tiles.")
            return _stitch_bbox_split_arrays(
                arr_a=arr_a,
                arr_b=arr_b,
                parent_spatial=bbox,
                axis=axis,
                scale_m=scale_m,
                fill_value=fill_value,
            ), tuple(names_a)

        return _bbox_recursive_fallback(
            spatial=spatial,
            scale_m=scale_m,
            fill_value=fill_value,
            fetch_fn=_fetch,
            stitch_fn=_stitch,
        )

    def _fetch_all_bands_impl(
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
            img = _order_collection_for_mosaic(col).mosaic()
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

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import (
    BBox,
    OutputSpec,
    PointBuffer,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from ._vendor.copernicus_embed import CopernicusEmbedGeoTiff
from .base import EmbedderBase
from .meta_utils import build_meta

SUPPORTED_YEARS = {2021}
_COPERNICUS_PROJECTION_WARNED = False


def _buffer_m_to_deg(lat: float, buffer_m: float) -> tuple[float, float]:
    """
    Approximate meters to degrees at given latitude.
    Good enough for precomputed tile selection / dataset slicing (v0.1).
    """
    # ~ meters per degree latitude
    m_per_deg_lat = 111_320.0
    dlat = buffer_m / m_per_deg_lat

    # longitude shrinks with cos(lat)
    import math

    cos_lat = max(1e-6, math.cos(math.radians(lat)))
    m_per_deg_lon = m_per_deg_lat * cos_lat
    dlon = buffer_m / m_per_deg_lon
    return dlon, dlat


def _spatial_to_bbox_4326(spatial: SpatialSpec) -> BBox:
    if isinstance(spatial, BBox):
        spatial.validate()
        return spatial
    if isinstance(spatial, PointBuffer):
        spatial.validate()
        dlon, dlat = _buffer_m_to_deg(spatial.lat, spatial.buffer_m)
        return BBox(
            minlon=spatial.lon - dlon,
            minlat=spatial.lat - dlat,
            maxlon=spatial.lon + dlon,
            maxlat=spatial.lat + dlat,
            crs="EPSG:4326",
        )
    raise ModelError(f"Unsupported SpatialSpec type: {type(spatial)}")


def _pool_chw(chw: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return chw.mean(axis=(1, 2)).astype(np.float32)
    if pooling == "max":
        return chw.max(axis=(1, 2)).astype(np.float32)
    raise ModelError(f"Unknown pooling='{pooling}' (expected 'mean' or 'max').")


def _projection_note() -> str:
    return (
        "Copernicus returns embeddings on its fixed product grid in EPSG:4326 "
        "(0.25 deg pixels), which differs from the common provider-backed default "
        "grid in EPSG:3857. Input spatial specs still use EPSG:4326."
    )


def _warn_projection_once() -> None:
    global _COPERNICUS_PROJECTION_WARNED
    if _COPERNICUS_PROJECTION_WARNED:
        return
    warnings.warn(
        _projection_note(),
        category=UserWarning,
        stacklevel=2,
    )
    _COPERNICUS_PROJECTION_WARNED = True


@register("copernicus")
class CopernicusEmbedder(EmbedderBase):
    """
    Precomputed embeddings via a vendored Copernicus GeoTIFF reader.

    Output:
      - OutputSpec.pooled(): (D,)
      - OutputSpec.grid():   xarray.DataArray (d,y,x) from CHW
    """

    DEFAULT_BATCH_WORKERS = 4

    def describe(self) -> dict[str, Any]:
        return {
            "type": "precomputed",
            "backend": ["auto"],
            "inputs": {"spatial": "BBox or PointBuffer (EPSG:4326)"},
            "temporal": {
                "mode": "ignored",
                "supported_years": sorted(SUPPORTED_YEARS),
            },
            "output": ["pooled", "grid"],
            "defaults": {
                "data_dir_env": "RS_EMBED_COP_DIR",
                "data_dir_default": "data/copernicus_embed",
                "download": True,
            },
            "notes": [
                "Uses a vendored GeoTIFF reader with bbox slicing ds[minlon:maxlon, minlat:maxlat].",
                "ROIs smaller than a single Copernicus pixel raise an error.",
                "Embeddings stay on the fixed product EPSG:4326 grid rather than the common provider-backed EPSG:3857 grid.",
            ],
        }

    def __init__(self) -> None:
        super().__init__()
        self._ds_cache: dict[str, Any] = {}

    def _get_dataset(self, *, data_dir: str, download: bool):
        # Cache per data_dir to avoid reopening the large GeoTIFF repeatedly.
        key = f"{data_dir}|download={int(bool(download))}"
        if key not in self._ds_cache:
            os.makedirs(data_dir, exist_ok=True)
            self._ds_cache[key] = CopernicusEmbedGeoTiff(paths=data_dir, download=download)
        return self._ds_cache[key]

    @staticmethod
    def _resolve_batch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_COPERNICUS_BATCH_WORKERS",
                str(CopernicusEmbedder.DEFAULT_BATCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: np.ndarray | None = None,
    ) -> Embedding:
        if temporal is None:
            raise ModelError("copernicus_embed requires TemporalSpec.year(YYYY).")
        temporal.validate()
        if temporal.mode == "range":
            year = int(str(temporal.start)[:4])
            warnings.warn(
                f"copernicus_embed only supports annual embeddings; "
                f"TemporalSpec.range will use start year {year} for lookup.",
                UserWarning,
                stacklevel=2,
            )
            temporal = TemporalSpec.year(year)
        if temporal.year not in SUPPORTED_YEARS:
            raise ModelError(
                f"copernicus_embed only provides embeddings for year(s) {sorted(SUPPORTED_YEARS)}; "
                f"got {temporal.year}. "
                f"Tip: use TemporalSpec.year(2021)."
            )

        backend_n = str(backend).strip().lower()
        if backend_n == "local":
            backend_n = "auto"
        if backend_n != "auto":
            raise ModelError("copernicus_embed is precomputed; use backend='auto'.")

        bbox = _spatial_to_bbox_4326(spatial)

        # data_dir: env var override OR (optional) sensor.collection override
        data_dir = os.environ.get("RS_EMBED_COP_DIR", "data/copernicus_embed")
        if sensor and isinstance(sensor.collection, str):
            # convention: collection="dir:/path/to/cop"
            if sensor.collection.startswith("dir:"):
                data_dir = sensor.collection.replace("dir:", "", 1).strip()

        download = True  # v0.1 default

        ds = self._get_dataset(data_dir=data_dir, download=download)
        minlon, minlat, maxlon, maxlat = (
            bbox.minlon,
            bbox.minlat,
            bbox.maxlon,
            bbox.maxlat,
        )

        # Vendored GeoTIFF bbox slicing
        sample = ds[minlon:maxlon, minlat:maxlat]
        img = sample["image"]
        if hasattr(img, "detach"):
            chw = img.detach().cpu().numpy().astype(np.float32)
        else:
            chw = np.asarray(img, dtype=np.float32)
        _warn_projection_once()

        meta = build_meta(
            model=self.model_name,
            kind="precomputed",
            backend="vendored_geotiff",
            source="hf://torchgeo/copernicus_embed/embed_map_310k.tif",
            sensor=None,
            temporal=None,
            image_size=None,
            extra={
                "data_dir": data_dir,
                "download": download,
                "bbox_4326": (minlon, minlat, maxlon, maxlat),
                "chw_shape": tuple(chw.shape),
                "dataset_path": getattr(ds, "path", None),
                "input_crs": "EPSG:4326",
                "output_crs": "EPSG:4326",
                "projection_mode": "product_native_fixed",
                "projection_note": _projection_note(),
                "product_resolution_deg": (0.25, 0.25),
            },
        )

        if output.mode == "pooled":
            vec = _pool_chw(chw, output.pooling)
            meta["pooling"] = f"{output.pooling}_hw"
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            try:
                import xarray as xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

            da = xr.DataArray(
                chw,
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(chw.shape[0]),
                    "y": np.arange(chw.shape[1]),
                    "x": np.arange(chw.shape[2]),
                },
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        raise ModelError(f"Unknown output mode: {output.mode}")

    def get_embeddings_batch(
        self,
        *,
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []

        n = len(spatials)
        out: list[Embedding | None] = [None] * n

        def _one(i: int, sp: SpatialSpec) -> tuple[int, Embedding]:
            emb = self.get_embedding(
                spatial=sp,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
            )
            return i, emb

        mw = self._resolve_batch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                _, emb = _one(i, sp)
                out[i] = emb
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, emb = fut.result()
                    out[i] = emb

        if any(e is None for e in out):
            raise ModelError("copernicus_embed batch failed to produce all outputs.")
        return [e for e in out if e is not None]

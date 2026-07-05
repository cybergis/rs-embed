from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import BBox, OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..providers.fetch import (
    fetch_collection_patch_all_bands_chw as _fetch_collection_patch_all_bands_chw,
)
from ..providers.resolution import (
    is_provider_backend,
)
from ..tools.tiling import _tile_subspatial, _tile_yx_starts
from .base import EmbedderBase
from .meta import build_meta

_GSE_DEFAULT_MAX_PIXELS = 512 * 512


def _gse_pixel_threshold() -> int:
    raw_value = os.environ.get("RS_EMBED_GSE_MAX_PIXELS", str(_GSE_DEFAULT_MAX_PIXELS))
    try:
        threshold = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ModelError("Invalid RS_EMBED_GSE_MAX_PIXELS value: must be an integer.") from exc
    if threshold < 1:
        raise ModelError(f"RS_EMBED_GSE_MAX_PIXELS must be >= 1, got {threshold}.")
    return threshold


def _estimate_pixel_dims(spatial: SpatialSpec, scale_m: int) -> tuple[int, int]:
    if not isinstance(spatial, BBox):
        return (1, 1)
    lat_c = (spatial.minlat + spatial.maxlat) / 2.0
    w_px = max(
        1,
        int(
            abs(spatial.maxlon - spatial.minlon)
            * math.cos(math.radians(lat_c))
            * 111320.0
            / scale_m
        ),
    )
    h_px = max(1, int(abs(spatial.maxlat - spatial.minlat) * 111320.0 / scale_m))
    return h_px, w_px


@register("gse")
class GSEAnnualEmbedder(EmbedderBase):
    """
    Precomputed embeddings on Provider:
      ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    """

    DEFAULT_BATCH_WORKERS = 4
    _allow_auto_backend = True
    _is_precomputed = True

    def describe(self) -> dict[str, Any]:
        return {
            "type": "precomputed",
            "backend": ["provider"],
            "temporal": {"mode": "year"},
            "output": ["grid", "pooled"],
            "source": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            "defaults": {
                "scale_m": 10,
                "fill_value": -9999.0,
                "composite": "mosaic",
            },
            "notes": "Uses sampleRectangle in EPSG:3857; returns [C,H,W] or pooled [C].",
        }

    @staticmethod
    def _resolve_batch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_GSE_BATCH_WORKERS",
                str(GSEAnnualEmbedder.DEFAULT_BATCH_WORKERS),
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
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("gse_annual expects a provider backend (or 'auto').")
        if temporal is None:
            raise ModelError("gse_annual requires TemporalSpec.year(year=...).")
        temporal.validate()
        if temporal.mode == "range":
            year = int(str(temporal.start)[:4])
            warnings.warn(
                f"gse_annual only supports annual embeddings; "
                f"TemporalSpec.range will use start year {year} for lookup.",
                UserWarning,
                stacklevel=2,
            )
            temporal = TemporalSpec.year(year)

        scale_m = int(getattr(sensor, "scale_m", 10) if sensor is not None else 10)

        provider = self._get_provider(backend)
        h_px, w_px = _estimate_pixel_dims(spatial, scale_m)
        if h_px * w_px > _gse_pixel_threshold():
            emb_chw, band_names = self._fetch_tiled(
                provider,
                spatial=spatial,
                temporal=temporal,
                scale_m=scale_m,
                h_px=h_px,
                w_px=w_px,
            )
        else:
            emb_chw, band_names = _fetch_collection_patch_all_bands_chw(
                provider,
                spatial=spatial,
                temporal=temporal,
                collection="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
                scale_m=scale_m,
                fill_value=-9999.0,
                composite="mosaic",
            )
        emb_chw = np.asarray(emb_chw, dtype=np.float32)
        emb_chw[emb_chw == -9999] = np.nan
        nodata_fraction = float(np.isnan(emb_chw[0]).mean()) if emb_chw.size else 0.0

        meta = build_meta(
            model=self.model_name,
            kind="precomputed",
            backend=str(backend).lower(),
            source="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            sensor=None,
            temporal=temporal,
            image_size=None,
            extra={
                "year": temporal.year,
                "scale_m": scale_m,
                "bands": band_names,
                "nodata_fraction": nodata_fraction,
            },
        )

        if output.mode == "pooled":
            # NaN-aware pooling: a single nodata pixel (tile edge, coastline)
            # must not poison the whole vector. All-nodata ROIs still error.
            if nodata_fraction >= 1.0:
                raise ModelError(
                    "gse_annual: the requested ROI contains no valid embedding "
                    "pixels (all nodata)."
                )
            if output.pooling == "mean":
                vec = np.nanmean(emb_chw, axis=(-2, -1)).astype(np.float32)
            elif output.pooling == "max":
                vec = np.nanmax(emb_chw, axis=(-2, -1)).astype(np.float32)
            else:
                raise ModelError(f"Unknown pooling='{output.pooling}' (expected 'mean' or 'max').")
            return Embedding(data=vec, meta={**meta, "pooling": output.pooling})

        # grid: return xarray with dims (band,y,x)
        da = xr.DataArray(
            emb_chw,
            dims=("d", "y", "x"),
            coords={"d": list(band_names)},
            name="embedding",
            attrs=meta,
        )
        return Embedding(data=da, meta=meta)

    def _fetch_tiled(
        self,
        provider: Any,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        scale_m: int,
        h_px: int,
        w_px: int,
    ) -> tuple[np.ndarray, Any]:
        tile_size = int(math.isqrt(_gse_pixel_threshold()))
        ys, xs = _tile_yx_starts(h=h_px, w=w_px, tile_size=tile_size, stride=tile_size)
        band_names: Any = None
        rows: list[np.ndarray] = []
        for y0 in ys:
            row: list[np.ndarray] = []
            for x0 in xs:
                y1 = min(h_px, y0 + tile_size)
                x1 = min(w_px, x0 + tile_size)
                sub = _tile_subspatial(
                    spatial, full_h=h_px, full_w=w_px, y0=y0, y1=y1, x0=x0, x1=x1
                )
                tile_chw, bn = _fetch_collection_patch_all_bands_chw(
                    provider,
                    spatial=sub,
                    temporal=temporal,
                    collection="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
                    scale_m=scale_m,
                    fill_value=-9999.0,
                    composite="mosaic",
                )
                if band_names is None:
                    band_names = bn
                row.append(np.asarray(tile_chw, dtype=np.float32))
            rows.append(np.concatenate(row, axis=-1))
        return np.concatenate(rows, axis=-2), band_names

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
        if not is_provider_backend(backend, allow_auto=True):
            raise ModelError("gse_annual expects a provider backend (or 'auto').")

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
            raise ModelError("gse_annual batch failed to produce all outputs.")
        return [e for e in out if e is not None]

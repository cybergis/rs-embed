from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers import ProviderBase
from ..ops.pooling import pool_chw_to_vec
from .base import EmbedderBase
from .meta_utils import build_meta, temporal_midpoint_str
from .runtime_utils import (
    fetch_collection_patch_all_bands_chw as _fetch_collection_patch_all_bands_chw,
    get_cached_provider,
    is_provider_backend,
)


@register("gse")
class GSEAnnualEmbedder(EmbedderBase):
    """
    Precomputed embeddings on Provider:
      ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    """

    DEFAULT_BATCH_WORKERS = 4

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "precomputed",
            "backend": ["provider"],
            "temporal": {"mode": "year"},
            "output": ["grid", "pooled"],
            "source": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            "notes": "Uses sampleRectangle in EPSG:3857; returns [C,H,W] or pooled [C].",
        }

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderBase] = {}

    def _get_provider(self, backend: str) -> ProviderBase:
        return get_cached_provider(
            self._providers,
            backend=backend,
            allow_auto=False,
        )

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
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str,
        device: str = "auto",
    ) -> Embedding:
        if not is_provider_backend(backend, allow_auto=False):
            raise ModelError("gse_annual only supports a provider backend in v0.1.")
        if temporal is None:
            raise ModelError("gse_annual requires TemporalSpec.year(year=...).")
        temporal.validate()
        if temporal.mode != "year":
            raise ModelError("gse_annual only supports TemporalSpec.year in v0.1.")

        provider = self._get_provider(backend)
        emb_chw, band_names = _fetch_collection_patch_all_bands_chw(
            provider,
            spatial=spatial,
            temporal=temporal,
            collection="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            scale_m=int(output.scale_m),
            fill_value=-9999.0,
            composite="mosaic",
        )
        emb_chw = np.asarray(emb_chw, dtype=np.float32)
        emb_chw[emb_chw == -9999] = np.nan

        meta = build_meta(
            model=self.model_name,
            kind="precomputed",
            backend=str(backend).lower(),
            source="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            sensor=None,
            temporal=temporal,
            image_size=None,
            input_time=temporal_midpoint_str(temporal),
            extra={
                "year": temporal.year,
                "scale_m": output.scale_m,
                "bands": band_names,
            },
        )

        if output.mode == "pooled":
            vec = pool_chw_to_vec(emb_chw, method=output.pooling)
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

    def get_embeddings_batch(
        self,
        *,
        spatials: list[SpatialSpec],
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []
        if not is_provider_backend(backend, allow_auto=False):
            raise ModelError("gse_annual only supports a provider backend in v0.1.")

        n = len(spatials)
        out: List[Optional[Embedding]] = [None] * n

        def _one(i: int, sp: SpatialSpec) -> Tuple[int, Embedding]:
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

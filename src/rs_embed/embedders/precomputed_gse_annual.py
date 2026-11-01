from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import EmbedderCapabilities
from ..providers.fetch import (
    fetch_collection_patch_all_bands_chw as _fetch_collection_patch_all_bands_chw,
)
from ..providers.resolution import (
    is_provider_backend,
)
from .base import EmbedderBase
from .meta import build_meta

_LEGACY_MAX_PIXELS_ENV = "RS_EMBED_GSE_MAX_PIXELS"


def _warn_if_legacy_max_pixels_env() -> None:
    if os.environ.get(_LEGACY_MAX_PIXELS_ENV) is None:
        return
    warnings.warn(
        f"{_LEGACY_MAX_PIXELS_ENV} is no longer used and will be removed: requests above "
        "GEE's request-size limit are split by the provider along pixel boundaries, "
        "so the result matches a single request "
        "(https://github.com/cybergis/rs-embed/issues/134).",
        FutureWarning,
        stacklevel=3,
    )


@register("gse")
class GSEAnnualEmbedder(EmbedderBase):
    """
    Precomputed embeddings on Provider:
      ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    """

    DEFAULT_BATCH_WORKERS = 4
    _allow_auto_backend = True
    _is_precomputed = True
    # GSE returns the precomputed grid as-is (the provider splits requests
    # above GEE's pixel limit); API-side input_prep is ignored (a non-resize
    # request warns).
    _manages_own_input_prep = True

    # Explicit pipeline-routing capabilities; the contract test asserts these
    # match the actual method signatures (tests/test_capabilities_contract.py).
    capabilities = EmbedderCapabilities(
        batch_fetch_metas=True,
        model_config_batch=True,
        model_config_batch_inputs=True,
    )

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
            "notes": "Downloads the EPSG:3857 common grid with computePixels; returns [C,H,W] or pooled [C].",
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

        _warn_if_legacy_max_pixels_env()
        provider = self._get_provider(backend)
        # Requests above GEE's request-size limit are split and
        # re-joined inside the provider, on the common grid's pixel lattice.
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
                    "gse_annual: the requested ROI contains no valid embedding pixels (all nodata)."
                )
            if output.pooling == "mean":
                vec = np.nanmean(emb_chw, axis=(-2, -1)).astype(np.float32)
            elif output.pooling == "max":
                vec = np.nanmax(emb_chw, axis=(-2, -1)).astype(np.float32)
            else:
                raise ModelError(f"Unknown pooling='{output.pooling}' (expected 'mean' or 'max').")
            return Embedding(data=vec, meta={**meta, "pooling": output.pooling})

        if output.mode == "grid":
            # grid: return xarray with dims (band,y,x)
            da = xr.DataArray(
                emb_chw,
                dims=("d", "y", "x"),
                coords={"d": list(band_names)},
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
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if model_config is not None:
            # Same contract as the base batch path: an unsupported
            # model_config raises ModelError instead of TypeError/silence.
            self._require_model_config_support(model_config)
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

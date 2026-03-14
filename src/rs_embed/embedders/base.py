from __future__ import annotations
from typing import Any

import numpy as np

from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..core.embedding import Embedding
from ..providers.base import ProviderBase

class EmbedderBase:
    """Base interface for all embedder implementations.

    Subclasses implement model-specific inference while keeping a common call
    signature used by higher-level API and pipeline code.
    """

    model_name: str = "base"
    _allow_auto_backend: bool = True

    def __init__(self) -> None:
        self._providers: dict[str, ProviderBase] = {}

    def _get_provider(self, backend: str) -> ProviderBase:
        from .runtime_utils import get_cached_provider

        return get_cached_provider(
            self._providers, backend=backend, allow_auto=self._allow_auto_backend
        )

    def describe(self) -> dict[str, Any]:
        """Return model capabilities and input/output requirements.

        Returns
        -------
        dict[str, Any]
            Metadata describing supported backends, outputs, and temporal hints.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete embedder subclasses.
        """
        raise NotImplementedError

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
        """Compute a single embedding.

        Parameters
        ----------
        spatial : SpatialSpec
            Spatial request definition.
        temporal : TemporalSpec or None
            Optional temporal filter.
        sensor : SensorSpec or None
            Optional sensor override for provider-backed models.
        output : OutputSpec
            Requested output layout.
        backend : str
            Backend/provider selector.
        device : str
            Target inference device.
        input_chw : np.ndarray or None
            Optional prefetched CHW input to bypass provider fetch.

        Returns
        -------
        Embedding
            Embedding payload and metadata.

        Raises
        ------
        NotImplementedError
            Must be implemented by concrete embedder subclasses.
        """

        raise NotImplementedError

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
        """Default batch implementation: loop over spatials.

        Embedders that can do true batching (e.g. torch models) should override.

        Parameters
        ----------
        spatials : list[SpatialSpec]
            Spatial requests to process.
        temporal : TemporalSpec or None
            Optional temporal filter applied to all inputs.
        sensor : SensorSpec or None
            Optional sensor override.
        output : OutputSpec
            Requested output layout.
        backend : str
            Backend/provider selector.
        device : str
            Target inference device.

        Returns
        -------
        list[Embedding]
            Embeddings in the same order as ``spatials``.
        """
        return [
            self.get_embedding(
                spatial=s,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
            )
            for s in spatials
        ]

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        """Batch inference with prefetched CHW inputs.

        Default implementation keeps existing behavior by looping through
        get_embedding(..., input_chw=...).
        Embedders that can do true batched model forward should override.

        Parameters
        ----------
        spatials : list[SpatialSpec]
            Spatial requests aligned with ``input_chws``.
        input_chws : list[np.ndarray]
            Prefetched CHW arrays, one per spatial.
        temporal : TemporalSpec or None
            Optional temporal filter.
        sensor : SensorSpec or None
            Optional sensor override.
        output : OutputSpec
            Requested output layout.
        backend : str
            Backend/provider selector.
        device : str
            Target inference device.

        Returns
        -------
        list[Embedding]
            Embeddings in the same order as inputs.

        Raises
        ------
        ValueError
            If ``spatials`` and ``input_chws`` lengths differ.
        """
        if len(spatials) != len(input_chws):
            raise ValueError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        return [
            self.get_embedding(
                spatial=s,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
                input_chw=x,
            )
            for s, x in zip(spatials, input_chws)
        ]

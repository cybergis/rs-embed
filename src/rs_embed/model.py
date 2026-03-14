"""Public class-based API for rs-embed embeddings.

``Model`` mirrors the TorchGeo ``RasterDataset`` pattern: heavy setup
once in ``__init__``, then lightweight calls to ``get_embedding`` /
``get_embeddings_batch`` that take only the varying inputs.
"""

from __future__ import annotations

from typing import Any

from .core.embedding import Embedding
from .core.errors import ModelError
from .core.specs import (
    InputPrepSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from .embedders.catalog import MODEL_ALIASES, MODEL_SPECS
from .tools.normalization import (
    _resolve_embedding_api_backend,
    normalize_backend_name,
    normalize_device_name,
    normalize_model_name,
)
from .tools.model_defaults import (
    default_sensor_for_model,
    resolve_sensor_for_model,
)
from .tools.runtime import (
    _EmbeddingRequestContext,
    get_embedder_bundle_cached,
    run_embedding_request,
    sensor_key,
)
from .tools.tiling import (
    _resolve_input_prep_spec,
)
from .core.validation import assert_supported, validate_specs

class Model:
    """A ready-to-use embedding model.

    Parameters
    ----------
    name : str
        Model identifier (e.g. ``"satclip-vit16-l40"``).
    backend : str
        Provider/inference backend (``"auto"`` / ``"gee"`` / …).
    device : str
        Target device (``"auto"`` / ``"cpu"`` / ``"cuda"`` / …).
    sensor : SensorSpec or None
        Sensor spec override.
    modality : str or None
        Optional modality selector for models that expose multiple input
        branches.
    output : OutputSpec
        Embedding output spec (default: pooled).
    input_prep : str or InputPrepSpec or None
        Input preparation mode.
    """

    def __init__(
        self,
        name: str,
        *,
        backend: str = "auto",
        device: str = "auto",
        sensor: SensorSpec | None = None,
        modality: str | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        input_prep: InputPrepSpec | str | None = "resize",
    ) -> None:
        self._model_n = normalize_model_name(name)
        self._backend_n = _resolve_embedding_api_backend(
            self._model_n, normalize_backend_name(backend)
        )
        self._device = normalize_device_name(device)
        self._output = output
        self._input_prep = input_prep
        self._input_prep_resolved = _resolve_input_prep_spec(input_prep)

        self._sensor = resolve_sensor_for_model(
            self._model_n,
            sensor=sensor,
            modality=modality,
            default_when_missing=(self._input_prep_resolved.mode == "tile"),
        )
        if self._input_prep_resolved.mode == "tile" and self._sensor is None:
            self._sensor = default_sensor_for_model(self._model_n)

        sensor_k = sensor_key(self._sensor)
        self._embedder, self._lock = get_embedder_bundle_cached(
            self._model_n, self._backend_n, self._device, sensor_k
        )
        assert_supported(
            self._embedder,
            backend=self._backend_n,
            output=self._output,
            temporal=None,
        )
        self._ctx = _EmbeddingRequestContext(
            model_n=self._model_n,
            backend_n=self._backend_n,
            device=self._device,
            sensor_eff=self._sensor,
            input_prep=self._input_prep,
            input_prep_resolved=self._input_prep_resolved,
            embedder=self._embedder,
            lock=self._lock,
        )

    # ── embedding methods ──────────────────────────────────────────

    def get_embedding(
        self,
        spatial: SpatialSpec,
        *,
        temporal: TemporalSpec | None = None,
    ) -> Embedding:
        """Compute one embedding with this model instance.

        Parameters
        ----------
        spatial : SpatialSpec
            Spatial location/extent to embed.
        temporal : TemporalSpec or None
            Optional temporal filter.

        Returns
        -------
        Embedding
            Normalized embedding output matching this model's ``output`` spec.
        """
        validate_specs(spatial=spatial, temporal=temporal, output=self._output)
        return run_embedding_request(
            spatials=[spatial],
            temporal=temporal,
            sensor=self._sensor,
            output=self._output,
            ctx=self._ctx,
        )[0]

    def get_embeddings_batch(
        self,
        spatials: list[SpatialSpec],
        *,
        temporal: TemporalSpec | None = None,
    ) -> list[Embedding]:
        """Compute embeddings for multiple spatial locations.

        Parameters
        ----------
        spatials : list[SpatialSpec]
            Non-empty list of spatial requests.
        temporal : TemporalSpec or None
            Optional temporal filter applied to all requests.

        Returns
        -------
        list[Embedding]
            Embeddings in the same order as ``spatials``.

        Raises
        ------
        ModelError
            If ``spatials`` is empty or not a list.
        """
        if not isinstance(spatials, list) or len(spatials) == 0:
            raise ModelError("spatials must be a non-empty list[SpatialSpec].")
        for sp in spatials:
            validate_specs(spatial=sp, temporal=temporal, output=self._output)
        return run_embedding_request(
            spatials=spatials,
            temporal=temporal,
            sensor=self._sensor,
            output=self._output,
            ctx=self._ctx,
        )

    def describe(self) -> dict[str, Any]:
        """Return model capabilities from the underlying embedder.

        Returns
        -------
        dict[str, Any]
            Capability metadata. Returns ``{}`` if unavailable.
        """
        try:
            desc = self._embedder.describe()
            return desc if isinstance(desc, dict) else {}
        except Exception as _e:
            return {}

    # ── static catalog ─────────────────────────────────────────────

    @staticmethod
    def list_models(*, include_aliases: bool = False) -> list[str]:
        """Return the stable model catalog.

        Parameters
        ----------
        include_aliases : bool
            If ``True``, include alias names in addition to canonical ids.

        Returns
        -------
        list[str]
            Sorted model names available in the catalog.
        """
        model_ids = set(MODEL_SPECS.keys())
        if include_aliases:
            model_ids.update(MODEL_ALIASES.keys())
        return sorted(model_ids)

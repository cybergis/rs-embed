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
    FetchSpec,
    InputPrepSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from .core.validation import assert_supported, validate_specs
from .embedders.catalog import MODEL_ALIASES, MODEL_SPECS
from .tools.model_defaults import resolve_sensor_for_model
from .tools.normalization import normalize_model_name
from .tools.runtime import (
    _prepare_embedding_request_context,
    require_model_config_support,
    run_embedding_request,
)


class Model:
    """A ready-to-use embedding model.

    Model-specific settings (e.g. variant selection) are passed as keyword
    arguments directly.  For example, ``Model("dofa", variant="large")``
    selects the large DOFA variant.  Call :func:`rs_embed.describe_model` to
    see which keyword arguments a model accepts.

    Parameters
    ----------
    name : str
        Model identifier (e.g. ``"dofa"``, ``"prithvi"``).
    backend : str
        Provider/inference backend (``"auto"`` / ``"gee"`` / …).
    device : str
        Target device (``"auto"`` / ``"cpu"`` / ``"cuda"`` / …).
    sensor : SensorSpec or None
        Sensor spec override.
    fetch : FetchSpec or None
        Lightweight fetch-policy override applied to the model default sensor.
        Cannot be combined with ``sensor``.
    modality : str or None
        Optional modality selector for models that expose multiple input
        branches.
    output : OutputSpec
        Embedding output spec (default: pooled).
    input_prep : str or InputPrepSpec or None
        Input preparation mode. ``None`` (the default) uses the package default,
        which is ``"tile"``: large on-the-fly inputs are tiled and stitched so
        native resolution is preserved. Pass ``"resize"`` to downsample the whole
        input to the model's image size instead, or ``"auto"`` to tile only when
        beneficial.
    **model_kwargs
        Model-specific settings (e.g. ``variant="large"``).  The accepted
        keys depend on the model; see :func:`rs_embed.describe_model`.

    Examples
    --------
    >>> model = Model("dofa", variant="large")
    >>> emb = model.get_embedding(spatial)
    """

    def __init__(
        self,
        name: str,
        *,
        backend: str = "auto",
        device: str = "auto",
        sensor: SensorSpec | None = None,
        fetch: FetchSpec | None = None,
        modality: str | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        input_prep: InputPrepSpec | str | None = None,
        **model_kwargs: Any,
    ) -> None:
        model_config = model_kwargs or None
        self._output = output

        # Same resolution sequence as api.get_embedding: resolve explicit
        # sensor/fetch/modality overrides first, then let the shared request
        # context apply normalization, model-aware input_prep, the tile-mode
        # sensor default, embedder caching, and backend/output support checks.
        sensor_eff = resolve_sensor_for_model(
            normalize_model_name(name),
            sensor=sensor,
            fetch=fetch,
            modality=modality,
            default_when_missing=False,
        )
        self._ctx = _prepare_embedding_request_context(
            model=name,
            temporal=None,
            sensor=sensor_eff,
            model_config=model_config,
            output=output,
            backend=backend,
            device=device,
            input_prep=input_prep,
        )
        # Fail fast at construction when model kwargs are unsupported, instead
        # of on the first embedding call.
        require_model_config_support(
            embedder=self._ctx.embedder,
            model_config=model_config,
            method_name="get_embedding",
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
        assert_supported(
            self._ctx.embedder,
            backend=self._ctx.backend_n,
            output=self._output,
            temporal=temporal,
        )
        return run_embedding_request(
            spatials=[spatial],
            temporal=temporal,
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
        assert_supported(
            self._ctx.embedder,
            backend=self._ctx.backend_n,
            output=self._output,
            temporal=temporal,
        )
        return run_embedding_request(
            spatials=spatials,
            temporal=temporal,
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
            desc = self._ctx.embedder.describe()
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

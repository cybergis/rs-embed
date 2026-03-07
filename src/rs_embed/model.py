"""Public class-based API for rs-embed embeddings.

``Model`` mirrors the TorchGeo ``RasterDataset`` pattern: heavy setup
once in ``__init__``, then lightweight calls to ``get_embedding`` /
``get_embeddings_batch`` that take only the varying inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
from .internal.api.api_helpers import (
    normalize_backend_name,
    normalize_device_name,
    normalize_model_name,
)
from .internal.api.model_defaults_helpers import default_sensor_for_model
from .internal.api.output_helpers import normalize_embedding_output
from .internal.api.runtime_helpers import (
    get_embedder_bundle_cached,
    run_with_retry,
    sensor_key,
)
from .internal.api.tiling_helpers import (
    _call_embedder_get_embedding_with_input_prep,
    _resolve_input_prep_spec,
)
from .internal.api.validation_helpers import assert_supported, validate_specs


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
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        input_prep: Optional[InputPrepSpec | str] = "resize",
    ) -> None:
        # Import here to share the backend resolution logic with api.py
        from .api import _resolve_embedding_api_backend

        self._model_n = normalize_model_name(name)
        self._backend_n = _resolve_embedding_api_backend(
            self._model_n, normalize_backend_name(backend)
        )
        self._device = normalize_device_name(device)
        self._output = output
        self._input_prep = input_prep
        self._input_prep_resolved = _resolve_input_prep_spec(input_prep)

        self._sensor = sensor
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

    # ── embedding methods ──────────────────────────────────────────

    def get_embedding(
        self,
        spatial: SpatialSpec,
        *,
        temporal: Optional[TemporalSpec] = None,
    ) -> Embedding:
        """Compute a single embedding."""
        validate_specs(spatial=spatial, temporal=temporal, output=self._output)
        return self._run([spatial], temporal=temporal, sensor=self._sensor)[0]

    def get_embeddings_batch(
        self,
        spatials: List[SpatialSpec],
        *,
        temporal: Optional[TemporalSpec] = None,
    ) -> List[Embedding]:
        """Compute embeddings for multiple spatial locations."""
        if not isinstance(spatials, list) or len(spatials) == 0:
            raise ModelError("spatials must be a non-empty List[SpatialSpec].")
        for sp in spatials:
            validate_specs(spatial=sp, temporal=temporal, output=self._output)
        return self._run(spatials, temporal=temporal, sensor=self._sensor)

    def describe(self) -> Dict[str, Any]:
        """Return the model's self-description dict."""
        try:
            desc = self._embedder.describe()
            return desc if isinstance(desc, dict) else {}
        except Exception:
            return {}

    # ── static catalog ─────────────────────────────────────────────

    @staticmethod
    def list_models(*, include_aliases: bool = False) -> List[str]:
        """Return the stable model catalog."""
        model_ids = set(MODEL_SPECS.keys())
        if include_aliases:
            model_ids.update(MODEL_ALIASES.keys())
        return sorted(model_ids)

    # ── internal execution ─────────────────────────────────────────

    def _run(
        self,
        spatials: List[SpatialSpec],
        *,
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
    ) -> List[Embedding]:
        """Execute embeddings, using API-side input fetch when appropriate."""
        prefetched = self._maybe_fetch_inputs(spatials, temporal)
        if prefetched is not None:
            out: List[Embedding] = []
            for spatial, raw in zip(spatials, prefetched):
                with self._lock:
                    emb = _call_embedder_get_embedding_with_input_prep(
                        embedder=self._embedder,
                        spatial=spatial,
                        temporal=temporal,
                        sensor=self._sensor,
                        output=self._output,
                        backend=self._backend_n,
                        device=self._device,
                        input_chw=raw,
                        input_prep=self._input_prep,
                    )
                out.append(emb)
            return out

        if len(spatials) == 1:
            with self._lock:
                emb = self._embedder.get_embedding(
                    spatial=spatials[0],
                    temporal=temporal,
                    sensor=sensor,
                    output=self._output,
                    backend=self._backend_n,
                    device=self._device,
                )
            return [normalize_embedding_output(emb=emb, output=self._output)]

        with self._lock:
            embs = self._embedder.get_embeddings_batch(
                spatials=spatials,
                temporal=temporal,
                sensor=sensor,
                output=self._output,
                backend=self._backend_n,
                device=self._device,
            )
        return [normalize_embedding_output(emb=e, output=self._output) for e in embs]

    def _maybe_fetch_inputs(
        self,
        spatials: List[SpatialSpec],
        temporal: Optional[TemporalSpec],
    ) -> Optional[List["np.ndarray"]]:
        """Fetch API-side inputs when input_prep requires it."""
        import numpy as np

        from .api import _provider_factory_for_backend
        from .internal.api.runtime_helpers import embedder_accepts_input_chw

        use_api_side = (self._input_prep is not None) and (
            self._input_prep_resolved.mode in {"tile", "auto"}
        )
        if not use_api_side:
            return None

        factory = _provider_factory_for_backend(self._backend_n)
        if factory is None:
            if self._input_prep_resolved.mode == "tile":
                raise ModelError(
                    "input_prep.mode='tile' requires a provider backend (e.g. gee)."
                )
            return None
        if self._sensor is None:
            if self._input_prep_resolved.mode == "tile":
                raise ModelError(
                    "input_prep.mode='tile' requires a sensor spec."
                )
            return None
        if not embedder_accepts_input_chw(type(self._embedder)):
            if self._input_prep_resolved.mode == "tile":
                raise ModelError(
                    f"Model {self._model_n} does not accept input_chw; cannot use input_prep.mode='tile'."
                )
            return None

        from .internal.api.api_helpers import fetch_gee_patch_raw

        provider = factory()
        ensure_ready = getattr(provider, "ensure_ready", None)
        if callable(ensure_ready):
            run_with_retry(lambda: ensure_ready(), retries=0, backoff_s=0.0)
        return [
            fetch_gee_patch_raw(
                provider, spatial=sp, temporal=temporal, sensor=self._sensor
            )
            for sp in spatials
        ]

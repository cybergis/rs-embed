"""rs-embed — location embeddings from remote-sensing imagery.

This top-level package re-exports the public API surface so that users can
write ``from rs_embed import Model, BBox`` without reaching into subpackages.

See :mod:`rs_embed.api` for the function-based API and :class:`rs_embed.Model`
for the class-based (stateful) interface.
"""

from ._version import __version__
from .api import (
    describe_model,
    export_batch,
    get_embedding,
    get_embeddings_batch,
    inspect_gee_patch,
    inspect_provider_patch,
    list_models,
    reset_runtime,
)
from .core._warnings import disable_pretty_warnings, enable_pretty_warnings
from .core.specs import (
    BBox,
    FetchSpec,
    InputPrepSpec,
    OutputSpec,
    PointBuffer,
    SensorSpec,
    TemporalSpec,
)
from .core.types import (
    ExportConfig,
    ExportLayout,
    ExportModelRequest,
    ExportTarget,
    ModelConfig,
)
from .load import ExportResult, ModelResult, load_export
from .model import Model
from .pipelines.exporter import BatchExporter

# Render rs-embed's own warnings as a structured, colourised block instead of
# Python's terse default. Opt out with ``RS_EMBED_PLAIN_WARNINGS=1``.
enable_pretty_warnings()

__all__ = [
    # Specs
    "BBox",
    "PointBuffer",
    "TemporalSpec",
    "SensorSpec",
    "FetchSpec",
    "OutputSpec",
    "InputPrepSpec",
    # Types
    "ExportConfig",
    "ExportLayout",
    "ExportModelRequest",
    "ExportTarget",
    "ModelConfig",
    # Embedding API (class-based)
    "Model",
    "BatchExporter",
    # Embedding API (function-based, backward compat)
    "get_embedding",
    "get_embeddings_batch",
    "list_models",
    "describe_model",
    "reset_runtime",
    # Export API
    "export_batch",
    # Load API
    "load_export",
    "ExportResult",
    "ModelResult",
    # Inspection
    "inspect_provider_patch",
    # Backward-compatible alias for inspect_provider_patch
    "inspect_gee_patch",
    # Warning display controls
    "enable_pretty_warnings",
    "disable_pretty_warnings",
    "__version__",
]

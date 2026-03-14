# src/rs_embed/core/registry.py
from __future__ import annotations

import importlib
from typing import Any

from rs_embed.embedders.catalog import MODEL_SPECS, canonical_model_id

from .errors import ModelError

_REGISTRY: dict[str, type[Any]] = {}
_REGISTRY_IMPORT_ERRORS: dict[str, BaseException] = {}

def register(name: str):
    """Create a decorator that registers an embedder class under ``name``.

    Parameters
    ----------
    name : str
        Model identifier or alias to register.

    Returns
    -------
    Callable[[type[Any]], type[Any]]
        Decorator that stores the class in the runtime model registry.
    """

    def deco(cls: type[Any]):
        model_id = canonical_model_id(name)
        _REGISTRY[model_id] = cls
        cls.model_name = model_id
        return cls

    return deco

def _try_lazy_load_model(name: str) -> None:
    """Load only the module that owns `name`, then backfill registration if needed."""
    model_id = canonical_model_id(name)
    if model_id in _REGISTRY:
        return
    spec = MODEL_SPECS.get(model_id)
    if spec is None:
        return
    module_name, class_name = spec
    fqmn = f"rs_embed.embedders.{module_name}"
    try:
        mod = importlib.import_module(fqmn)
    except Exception as e:
        _REGISTRY_IMPORT_ERRORS[model_id] = e
        return

    try:
        cls = getattr(mod, class_name)
    except Exception as e:
        _REGISTRY_IMPORT_ERRORS[model_id] = e
        return

    # If decorators did not run in this process state (e.g. registry was cleared),
    # repopulate from the imported module class symbol.
    _REGISTRY[model_id] = cls
    cls.model_name = model_id
    _REGISTRY_IMPORT_ERRORS.pop(model_id, None)

def get_embedder_cls(name: str) -> type[Any]:
    """Resolve and return the embedder class for ``name``.

    Parameters
    ----------
    name : str
        Model identifier or alias.

    Returns
    -------
    type[Any]
        Embedder class registered for the canonical model id.

    Raises
    ------
    ModelError
        If the model is unknown or its module/class cannot be imported.
    """

    k = canonical_model_id(name)
    if k not in _REGISTRY:
        _try_lazy_load_model(k)
    if k not in _REGISTRY:
        msg = (
            f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}. "
            f"If this list is empty, ensure requested embedder module is importable "
            f"(e.g. optional deps like torch/ee are installed)."
        )
        if k in _REGISTRY_IMPORT_ERRORS:
            err = _REGISTRY_IMPORT_ERRORS[k]
            msg += f" Import error for '{k}': {type(err).__name__}: {err}"
        elif _REGISTRY_IMPORT_ERRORS:
            parts = [
                f"{mid}: {type(e).__name__}: {e}" for mid, e in _REGISTRY_IMPORT_ERRORS.items()
            ]
            msg += f" Embedder import errors: {'; '.join(parts)}"
        raise ModelError(msg)
    return _REGISTRY[k]

def list_models():
    """Return sorted list of currently-loaded model IDs from the runtime registry.

    Note: only models that have been lazy-imported so far will appear here.
    Use rs_embed.api.list_models() for a stable catalog-backed list.

    Returns
    -------
    list[str]
        Canonical model ids currently present in the in-memory registry.
    """
    return sorted(_REGISTRY.keys())

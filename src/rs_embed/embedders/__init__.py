"""
Model Implementations (Plugins).

This module contains the specific inference logic for each supported AI model.

To add a new model:

1. Create a new file (e.g., ``my_model.py``).
2. Inherit from ``rs_embed.embedders.base.EmbedderBase``.
3. Implement ``get_embedding()``.

Rules
-----
- Models should be self-contained.
- Only inference logic belongs here — no training code.
- Lazy-loading: embedder classes are imported on first access via
  ``__getattr__`` so unused models incur no startup cost.
"""

from __future__ import annotations

import importlib
from typing import Any

from .catalog import CLASS_TO_MODULE, MODEL_SPECS

__all__ = ["MODEL_SPECS"]


def __getattr__(name: str) -> Any:
    """Lazily expose embedder classes without importing all submodules."""
    module_name = CLASS_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"{__name__}.{module_name}")
    try:
        return getattr(mod, name)
    except AttributeError as e:
        raise AttributeError(f"module {mod.__name__!r} has no attribute {name!r}") from e

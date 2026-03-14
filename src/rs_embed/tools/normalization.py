from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import ModelError
from ..core.registry import get_embedder_cls
from ..embedders.catalog import canonical_model_id
from ..providers import has_provider


def normalize_model_name(model: str) -> str:
    return canonical_model_id(model)

def normalize_backend_name(backend: str) -> str:
    return str(backend).strip().lower()

def normalize_device_name(device: str | None) -> str:
    if device is None:
        return "auto"
    dev = str(device).strip().lower()
    return dev or "auto"

def normalize_input_chw(
    x_chw: np.ndarray,
    *,
    expected_channels: int | None = None,
    name: str = "input_chw",
) -> np.ndarray:
    x = np.asarray(x_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ModelError(f"{name} must be CHW with ndim=3, got shape={getattr(x, 'shape', None)}")
    if expected_channels is not None and int(x.shape[0]) != int(expected_channels):
        raise ModelError(
            f"{name} channel mismatch: got C={int(x.shape[0])}, expected C={int(expected_channels)}"
        )
    return x

def _probe_model_describe(model_n: str) -> dict[str, Any]:
    """Best-effort model describe() probe used for API-level routing decisions."""
    try:
        cls = get_embedder_cls(model_n)
        emb = cls()
        desc = emb.describe() or {}
        return desc if isinstance(desc, dict) else {}
    except Exception as _e:
        return {}

def _default_provider_backend_for_api() -> str:
    from ..embedders.runtime_utils import default_provider_backend_name

    return default_provider_backend_name() or "gee"

def _resolve_embedding_api_backend(model_n: str, backend_n: str) -> str:
    """Normalize backend semantics for precomputed models."""
    desc = _probe_model_describe(model_n)
    if str(desc.get("type", "")).strip().lower() != "precomputed":
        return backend_n

    backends = desc.get("backend")
    if not isinstance(backends, list):
        return backend_n
    allowed = [str(b).strip().lower() for b in backends if str(b).strip()]
    if not allowed:
        return backend_n

    provider_allowed = ("provider" in allowed) or ("gee" in allowed)
    if backend_n in allowed:
        if backend_n == "auto" and provider_allowed:
            return _default_provider_backend_for_api()
        return backend_n
    if backend_n == "local" and "auto" in allowed and not provider_allowed:
        return "auto"
    if has_provider(backend_n) and provider_allowed:
        return backend_n

    if backend_n in {"gee", "auto"}:
        if "auto" in allowed:
            return "auto"
        if "local" in allowed:
            return "local"
        if provider_allowed:
            return _default_provider_backend_for_api()

    return backend_n

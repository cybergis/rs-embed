"""
Data Source Adapters.

This module handles fetching raw imagery from external services
(Google Earth Engine, Sentinel Hub, local files, etc.).

To add a new data source:

1. Inherit from ``rs_embed.providers.base.ProviderBase``.
2. Implement ``fetch_array_chw()`` to return a standardised NumPy array
   shaped ``(C, H, W)``.
3. Handle authentication and networking internally.

Built-in providers are lazy-loaded so optional dependencies stay optional.
"""

from __future__ import annotations

import importlib

from .base import ProviderBase

_PROVIDER_REGISTRY: dict[str, type[ProviderBase]] = {}
_BUILTINS_LOADED = False

def _register_builtin_providers() -> None:
    """Lazy-load built-in providers so optional deps stay optional."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    try:
        mod = importlib.import_module(f"{__name__}.gee")
        cls = mod.GEEProvider
        _PROVIDER_REGISTRY.setdefault("gee", cls)
    except Exception as _e:
        # Keep registry usable even when optional backend deps are unavailable.
        pass

def register_provider(name: str, provider_cls: type[ProviderBase]) -> None:
    _register_builtin_providers()
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Provider name must be a non-empty string.")
    _PROVIDER_REGISTRY[key] = provider_cls

def get_provider(name: str, **kwargs) -> ProviderBase:
    _register_builtin_providers()
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Provider name must be a non-empty string.")
    provider_cls = _PROVIDER_REGISTRY.get(key)
    if provider_cls is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unknown provider '{name}'. Available providers: {available}")
    return provider_cls(**kwargs)

def list_providers() -> tuple[str, ...]:
    _register_builtin_providers()
    return tuple(sorted(_PROVIDER_REGISTRY.keys()))

def has_provider(name: str) -> bool:
    _register_builtin_providers()
    key = str(name).strip().lower()
    if not key:
        return False
    return key in _PROVIDER_REGISTRY

def __getattr__(name: str):
    if name == "GEEProvider":
        _register_builtin_providers()
        cls = _PROVIDER_REGISTRY.get("gee")
        if cls is None:
            raise AttributeError("GEEProvider is unavailable in this environment.")
        return cls
    raise AttributeError(name)

__all__ = [
    "ProviderBase",
    "GEEProvider",
    "get_provider",
    "register_provider",
    "list_providers",
    "has_provider",
]

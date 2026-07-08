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
# Import failures of optional built-in providers, keyed by provider name, so
# an unavailable provider can report WHY it is missing instead of silently
# looking unregistered (mirrors core.registry._REGISTRY_IMPORT_ERRORS).
_PROVIDER_IMPORT_ERRORS: dict[str, Exception] = {}
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
        _PROVIDER_IMPORT_ERRORS.pop("gee", None)
    except Exception as e:
        # Keep registry usable even when optional backend deps are unavailable,
        # but record the cause for get_provider's error message.
        _PROVIDER_IMPORT_ERRORS["gee"] = e


def register_provider(name: str, provider_cls: type[ProviderBase]) -> None:
    _register_builtin_providers()
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Provider name must be a non-empty string.")
    _PROVIDER_REGISTRY[key] = provider_cls
    _PROVIDER_IMPORT_ERRORS.pop(key, None)


def get_provider(name: str, **kwargs) -> ProviderBase:
    _register_builtin_providers()
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Provider name must be a non-empty string.")
    provider_cls = _PROVIDER_REGISTRY.get(key)
    if provider_cls is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        msg = f"Unknown provider '{name}'. Available providers: {available}."
        if key in _PROVIDER_IMPORT_ERRORS:
            err = _PROVIDER_IMPORT_ERRORS[key]
            msg += f" Import error for '{key}': {type(err).__name__}: {err}"
        elif _PROVIDER_IMPORT_ERRORS:
            parts = [
                f"{pid}: {type(e).__name__}: {e}" for pid, e in _PROVIDER_IMPORT_ERRORS.items()
            ]
            msg += f" Provider import errors: {'; '.join(parts)}"
        raise ValueError(msg)
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

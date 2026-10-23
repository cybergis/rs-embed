from __future__ import annotations

import os
from numbers import Integral, Real
from typing import Any

from ..core.errors import ModelError


def model_config_value(
    model_config: dict[str, Any] | None,
    key: str,
) -> Any | None:
    if model_config is None:
        return None
    if isinstance(model_config, dict):
        return model_config.get(key)
    return getattr(model_config, key, None)


def coerce_config_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    raise ModelError(f"model_config['{key}'] must be a boolean-like value, got {value!r}.")


def _exact_int(value: Any) -> int | None:
    """*value* as an int, or ``None`` for a bool, a fractional number or a non-number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return int(value) if float(value).is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def resolve_config_positive_int(
    model_config: dict[str, Any] | None,
    key: str,
    *,
    env_var: str,
    default: int,
    model_name: str,
    max_value: int | None = None,
) -> int:
    """``model_config[key]``, else ``$env_var``, else *default*, as a positive int.

    A bool or a fractional number raises instead of being truncated.
    """
    value = model_config_value(model_config, key)
    source = "model_config"
    if value is None:
        value = os.environ.get(env_var, "").strip() or None
        source = env_var
    if value is None:
        return int(default)
    n = _exact_int(value)
    if n is None:
        raise ModelError(f"{model_name} {key} from {source} must be an integer, got {value!r}.")
    if n < 1 or (max_value is not None and n > max_value):
        bound = "positive" if max_value is None else f"1–{max_value}"
        raise ModelError(f"{model_name} {key} must be {bound}, got {n}.")
    return n

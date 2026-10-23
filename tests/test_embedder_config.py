"""Tests for the shared model_config helpers (embedders/config.py)."""

from __future__ import annotations

import numpy as np
import pytest

from rs_embed.core.errors import ModelError
from rs_embed.embedders.config import resolve_config_positive_int

_ENV = "RS_EMBED_TEST_KNOB"


def _resolve(model_config=None, *, max_value=None):
    return resolve_config_positive_int(
        model_config,
        "knob",
        env_var=_ENV,
        default=3,
        model_name="Test",
        max_value=max_value,
    )


def test_model_config_then_env_then_default(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert _resolve() == 3
    monkeypatch.setenv(_ENV, " 5 ")
    assert _resolve() == 5
    assert _resolve({"knob": 7}) == 7
    monkeypatch.setenv(_ENV, "")
    assert _resolve() == 3


@pytest.mark.parametrize("value", [4, "4", 4.0, np.int64(4), np.float32(4.0)])
def test_accepts_exact_integers(value):
    assert _resolve({"knob": value}) == 4


@pytest.mark.parametrize("value", [2.9, np.float32(3.7), True, "2.0", "four", [2]])
def test_rejects_non_integers_instead_of_truncating(value):
    with pytest.raises(ModelError, match="Test knob from model_config must be an integer"):
        _resolve({"knob": value})


def test_env_source_is_named(monkeypatch):
    monkeypatch.setenv(_ENV, "abc")
    with pytest.raises(ModelError, match=f"from {_ENV} must be an integer"):
        _resolve()


def test_bounds():
    with pytest.raises(ModelError, match="Test knob must be positive, got 0"):
        _resolve({"knob": 0})
    with pytest.raises(ModelError, match="Test knob must be 1–8, got 9"):
        _resolve({"knob": 9}, max_value=8)
    assert _resolve({"knob": 8}, max_value=8) == 8

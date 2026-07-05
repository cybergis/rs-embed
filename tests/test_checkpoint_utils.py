"""Tests for rs_embed.tools.checkpoint_utils helpers."""

import numpy as np

from rs_embed.tools.checkpoint_utils import drop_model_arrays


def _sanitize(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def test_drop_model_arrays_removes_exact_and_ragged_keys():
    arrays = {
        "embeddings__satmae": np.zeros(2),
        "embedding__satmae__00003": np.zeros(2),
        "inputs_bchw__satmae": np.zeros(2),
        "input_chw__satmae__00003": np.zeros(2),
        "other": np.zeros(2),
    }
    drop_model_arrays(arrays, "satmae", sanitize_key=_sanitize)
    assert set(arrays) == {"other"}


def test_drop_model_arrays_does_not_drop_prefix_sibling_model():
    """satmae must not delete satmaepp's arrays (prefix collision)."""
    arrays = {
        "embeddings__satmae": np.zeros(2),
        "embeddings__satmaepp": np.ones(2),
        "inputs_bchw__satmaepp": np.ones(2),
        "embedding__satmaepp__00001": np.ones(2),
    }
    drop_model_arrays(arrays, "satmae", sanitize_key=_sanitize)
    assert set(arrays) == {
        "embeddings__satmaepp",
        "inputs_bchw__satmaepp",
        "embedding__satmaepp__00001",
    }

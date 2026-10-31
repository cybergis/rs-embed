from pathlib import Path

import numpy as np
import xarray as xr

from rs_embed.core.embedding import Embedding
from rs_embed.embedders.meta import META_REQUIRED_KEYS, build_meta


def test_embedding_numpy():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    meta = {"model": "test"}
    emb = Embedding(data=data, meta=meta)
    assert emb.data is data
    assert emb.meta["model"] == "test"


def test_embedding_xarray():
    da = xr.DataArray(np.zeros((3, 4, 4)), dims=["C", "H", "W"])
    emb = Embedding(data=da, meta={"mode": "grid"})
    assert emb.data.dims == ("C", "H", "W")
    assert emb.data.shape == (3, 4, 4)


def test_build_meta_contains_stable_contract_keys():
    meta = build_meta(
        model="demo",
        kind="on_the_fly",
        backend="auto",
        source="demo_source",
        sensor=None,
        temporal=None,
        image_size=224,
    )

    for key in META_REQUIRED_KEYS:
        assert key in meta


def test_builtin_embedders_use_common_meta_helpers():
    root = Path(__file__).resolve().parents[1] / "src" / "rs_embed" / "embedders"
    ignored = {
        "__init__.py",
        "_vit_mae_utils.py",
        "base.py",
        "catalog.py",
        "config_utils.py",
        "meta_utils.py",
        "runtime_utils.py",
        # Helper module, not an embedder: its builders take a base_meta the
        # calling embedder made with build_meta().
        "shared.py",
    }
    missing: list[str] = []
    for path in sorted(root.glob("*.py")):
        if path.name in ignored:
            continue
        text = path.read_text(encoding="utf-8")
        if "return Embedding(" not in text:
            continue
        if "build_meta(" not in text and "base_meta(" not in text:
            missing.append(path.name)

    assert missing == []

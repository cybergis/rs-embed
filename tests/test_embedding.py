import numpy as np
import xarray as xr

from rs_embed.core.embedding import Embedding


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

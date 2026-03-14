from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

@dataclass
class Embedding:
    """Container for embedding values and associated metadata.

    Attributes
    ----------
    data : np.ndarray or xr.DataArray
        Embedding payload, typically pooled vectors or grid embeddings.
    meta : dict[str, Any]
        Metadata describing provenance, orientation, model details, and
        optional diagnostics.
    """

    data: np.ndarray | xr.DataArray
    meta: dict[str, Any]

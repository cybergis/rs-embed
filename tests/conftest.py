import sys
from pathlib import Path

import numpy as np

# Ensure `src/` is importable when running tests without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rs_embed.core.embedding import Embedding  # noqa: E402
from rs_embed.embedders.base import EmbedderBase  # noqa: E402

# ── shared mock embedders ────────────────────────────────────────────
# Used by test_api.py, test_model.py, and others.


class MockEmbedder(EmbedderBase):
    """Deterministic embedder with no I/O."""

    def describe(self):
        return {"type": "mock", "dim": 8}

    def get_embedding(
        self,
        *,
        spatial,
        temporal,
        sensor,
        output,
        backend,
        device="auto",
        input_chw=None,
    ):
        vec = np.arange(8, dtype=np.float32)
        return Embedding(
            data=vec,
            meta={
                "model": self.model_name,
                "output": output.mode,
                "sensor": sensor,
            },
        )


class MockPrecomputedLocalEmbedder(EmbedderBase):
    """Precomputed embedder using local/auto backends."""

    def describe(self):
        return {
            "type": "precomputed",
            "backend": ["local", "auto"],
            "output": ["pooled"],
            "source": "mock.fixed.source",
        }

    def get_embedding(
        self,
        *,
        spatial,
        temporal,
        sensor,
        output,
        backend,
        device="auto",
        input_chw=None,
    ):
        return Embedding(
            data=np.arange(4, dtype=np.float32),
            meta={
                "model": self.model_name,
                "backend_used": backend,
                "source": "mock.fixed.source",
            },
        )


class MockMultimodalEmbedder(EmbedderBase):
    """Embedder exposing s2/s1 modality branches."""

    def describe(self):
        return {
            "type": "on_the_fly",
            "backend": ["provider"],
            "output": ["pooled"],
            "modalities": {
                "s2": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": ["B4", "B3", "B2"],
                },
                "s1": {
                    "collection": "COPERNICUS/S1_GRD_FLOAT",
                    "bands": ["VV", "VH"],
                    "defaults": {"use_float_linear": True},
                },
            },
            "defaults": {"modality": "s2", "scale_m": 10},
        }

    def get_embedding(
        self,
        *,
        spatial,
        temporal,
        sensor,
        output,
        backend,
        device="auto",
        input_chw=None,
    ):
        vec = np.arange(4, dtype=np.float32)
        return Embedding(
            data=vec,
            meta={
                "model": self.model_name,
                "sensor": sensor,
                "backend_used": backend,
            },
        )

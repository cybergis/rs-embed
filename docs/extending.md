
## Overview

To add a new model, you typically do five things:

1. **Create an embedder class** in `src/rs_embed/embedders/`
2. Decorate it with **`@register("your_model_name")`**
3. Add it to `src/rs_embed/embedders/catalog.py` (`MODEL_SPECS`)
4. Implement: `describe()`, `get_embedding(...)`
5. (Optional, recommended) Override:
    - `get_embeddings_batch(...)` for true batched inference (no prefetched inputs)
    - `get_embeddings_batch_from_inputs(...)` for true batched inference with prefetched `input_chw`

---

## The Registry

Models are discovered through the registry in `rs_embed.core.registry`:

- `@register("name")` registers an embedder class.
- `get_embedder_cls("name")` resolves the class.
- `list_models()` lists models that have already been loaded in the current process.
- `rs_embed.list_models()` returns the stable public model catalog from `MODEL_SPECS`.

Model loading is **lazy**:

- `get_embedder_cls("name")` looks up `name` in `MODEL_SPECS`.
- Then it imports the mapped module and reads the mapped class.
- The class is inserted into the runtime registry.

!!! tip
    Put your embedder in `rs_embed/embedders/` and add it to `src/rs_embed/embedders/catalog.py`.
    If it's not in `MODEL_SPECS`, string-based lookup (`get_embedding("...")`) will not find it.

---

## Embedder Contract

All models implement `EmbedderBase`:

```python
from rs_embed.embedders.base import EmbedderBase

class EmbedderBase:
    def describe(self) -> dict: ...
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
    ): ...
    def get_embeddings_batch(...): ...
    def get_embeddings_batch_from_inputs(...): ...
```

### `describe()`

`describe()` should return a JSON-serializable dictionary describing capabilities and requirements. A typical structure is:

```python
{
  "type": "on_the_fly" | "precomputed",
  "backend": ["provider" | "gee" | "auto" | "tensor", ...],
  "inputs": {
    "sensor_required": true/false,
    "default_sensor": {...} | null,
    "notes": "..."
  },
  "temporal": {"mode": "year" | "range"} | null,
  "output": ["pooled", "grid"]
}
```

!!! note
    `describe()` should be **fast** and should not trigger heavy downloads or model loading.
    In current rs-embed, `describe()["backend"]`, `describe()["output"]`, and `describe()["temporal"]`
    may be used for runtime validation and capability checks.

---

## Template: Minimal Model (Hello World)

This is the smallest possible embedder you can add. It returns a deterministic random vector.

Create `src/rs_embed/embedders/toy_model.py`:

```python
from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any, Dict, Optional
import numpy as np

from rs_embed.core.registry import register
from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from rs_embed.embedders.base import EmbedderBase


@register("toy_model_v1")
class ToyModelV1(EmbedderBase):
    def describe(self) -> Dict[str, Any]:
        return {
            "type": "precomputed",
            "backend": ["auto"],  # use "provider"/"gee" for on-the-fly fetchers
            "inputs": {"sensor_required": False, "default_sensor": None},
            "output": ["pooled"],
        }

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str = "auto",
        device: str = "auto",
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        # Use a stable hash so results are reproducible across processes.
        seed_bytes = hashlib.blake2s(
            f"{spatial!r}|{temporal!r}|{self.model_name}".encode("utf-8"),
            digest_size=4,
        ).digest()
        seed = int.from_bytes(seed_bytes, "little")
        rng = np.random.default_rng(seed)

        if output.mode != "pooled":
            raise ValueError("toy_model_v1 only supports pooled output")

        vec = rng.standard_normal(512).astype("float32")
        meta = {
            "model": self.model_name,
            "backend": backend,
            "device": device,
            "spatial": asdict(spatial),
            "temporal": asdict(temporal) if temporal else None,
        }
        return Embedding(data=vec, meta=meta)
```

Then register it in `src/rs_embed/embedders/catalog.py`:

```python
MODEL_SPECS["toy_model_v1"] = ("toy_model", "ToyModelV1")
```

---

## On-the-fly Models (GEE patch → model → embedding)

Most vision models in rs-embed work like this:

1. Use a provider (e.g., Earth Engine) to fetch an **input patch** (CHW numpy).
2. Preprocess it (normalize/resample).
3. Run inference.
4. Return `Embedding(data=..., meta=...)`.

### Recommended pattern

- Use `SensorSpec` to define:
  - collection
  - bands
  - scale_m
  - composite strategy
- Use the provider to fetch inputs.

You can follow existing implementations in:
- `rs_embed/embedders/onthefly_*.py`

!!! tip
    Keep network IO (fetching patches) separate from model inference whenever possible.
    This makes batching and caching much easier.

---

## Supporting `export_batch` Input Reuse (`input_chw`)

`export_batch` can prefetch the input patch once and reuse it for both:

- saving `inputs`
- computing `embeddings`

To benefit from this optimization, your `get_embedding` should follow this rule:

> If `input_chw` is provided, **do not fetch inputs again**. Use `input_chw` as the model input.

Example snippet:

```python
if input_chw is None:
    # fetch from backend/provider
    input_chw = provider.fetch_array_chw(...)
# now preprocess + infer using input_chw
```

!!! important
    This is the key to avoiding “download twice” when `save_inputs=True` and `save_embeddings=True`.

---

## True Batch Inference (`get_embeddings_batch` / `get_embeddings_batch_from_inputs`)

`EmbedderBase.get_embeddings_batch` defaults to a Python loop calling `get_embedding`.  
`EmbedderBase.get_embeddings_batch_from_inputs` defaults to a Python loop calling
`get_embedding(..., input_chw=...)`.

If your model supports vectorized/batched inference (common for torch models), override one or both:

```python
def get_embeddings_batch(
    self,
    *,
    spatials,
    temporal=None,
    sensor=None,
    output=OutputSpec.pooled(),
    backend="gee",
    device="auto",
):
    # 1) fetch/preprocess inputs for all spatials
    # 2) stack into a batch tensor
    # 3) run a single forward pass
    # 4) split outputs back into Embedding objects
```

If your model supports `input_chw` reuse (recommended for on-the-fly models), also consider:

```python
def get_embeddings_batch_from_inputs(
    self,
    *,
    spatials,
    input_chws,
    temporal=None,
    sensor=None,
    output=OutputSpec.pooled(),
    backend="auto",
    device="auto",
):
    # 1) preprocess/stack prefetched CHW inputs
    # 2) run a single batched forward pass
    # 3) split outputs back into Embedding objects
```

`export_batch(...)` prefers `get_embeddings_batch_from_inputs(...)` when it has prefetched
provider inputs available, so overriding this method usually gives the biggest speedup for
on-the-fly models.

Best practice:
- Batch **inference** (GPU-friendly).
- Parallelize **IO** (provider fetch) with threads if needed.
- Keep memory stable by using chunking (see `export_batch(chunk_size=...)`).

---

## Handling `OutputSpec` (pooled vs grid)

`OutputSpec` controls output shape:

- `OutputSpec.pooled()` → `(D,)`
- `OutputSpec.grid(...)` → `(D, H, W)`

If your model does not support a mode, raise a clear error:

```python
if output.mode == "grid" and not supported:
    raise ValueError("model_x does not support grid output")
```

---

## Optional Dependencies (Packaging)

Many embedders rely on optional packages (e.g., `torch`, `ee`).  
Follow this pattern:

- Import heavy dependencies **inside** methods or within a `try/except` at module import.
- If the dependency is missing, raise a **helpful** error (`ModelError`) explaining what to install.

Example:

```python
from rs_embed.core.errors import ModelError

try:
    import torch
except Exception as e:
    torch = None
    _torch_err = e

def _require_torch():
    if torch is None:
        raise ModelError('Torch is required. Install with: pip install "rs-embed[torch]"')
```

---

## Testing Your New Model

### 1) Registry test (fast)

Add a test ensuring registration works:

```python
from rs_embed.core.registry import get_embedder_cls

def test_toy_model_registered():
    cls = get_embedder_cls("toy_model_v1")
    assert cls is not None
```

### 2) API-level test (recommended)

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

def test_toy_model_get_embedding():
    emb = get_embedding(
        "toy_model_v1",
        spatial=PointBuffer(0, 0, 1000),
        temporal=TemporalSpec.year(2022),
        output=OutputSpec.pooled(),
        backend="auto",
    )
    assert emb.data.shape == (512,)
```

### 3) Export integration test (optional)

If your model supports input reuse and batch export, add a small `export_batch` test using `monkeypatch` to avoid real network calls.  
See existing patterns in:
- `tests/test_export_batch.py`
- `tests/test_gee_provider.py`

Run tests:

```bash
pytest -q
```

---

## Documenting the Model

Update docs in one of these places:

- `docs/models.md` (add model name and usage)


---

## Checklist

Before opening a PR / shipping the model:

- [ ] `@register("...")` added and entry added in `src/rs_embed/embedders/catalog.py`
- [ ] `describe()` is fast and accurate
- [ ] `get_embedding()` supports `input_chw` reuse (if on-the-fly)
- [ ] override `get_embeddings_batch_from_inputs()` if your model can batch prefetched inputs
- [ ] clear errors for missing optional dependencies
- [ ] unit tests added (`pytest -q` passes)
- [ ] minimal usage example in docs or notebook

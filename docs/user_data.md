# API: User-Provided Data

This page covers the bring-your-own-data API: computing embeddings from imagery you already have, instead of provider-fetched imagery.

Related pages: [API: Embedding](api_embedding.md), [API: Specs and Data Structures](api_specs.md).

---

## Concept

The flow has two steps:

1. **Register** each piece of imagery as a `UserData` — the pixels plus everything that describes them: which collection they came from, one band name per channel, and where/when they were acquired.
2. **Embed** by naming a model: `get_embedding_from_data("galileo", data)`. Nothing else is needed — the declaration already carries the full context.

Every on-the-fly model declares an input sensor (a collection plus an ordered band list). Your declaration is matched against it:

- **Superset data is accepted**: if your declaration covers all bands the model needs, the needed channels are sliced out and reordered automatically. One 12-band Sentinel-2 L2A cube can serve models that need 3, 6, or 10 of those bands.
- **Insufficient data is refused**: a collection mismatch (e.g. S2 data offered to a MODIS model) or a missing band raises `ModelError` naming exactly what is missing. Precomputed models (`tessera`, `gse`, `copernicus`) are always refused — they have no imagery input.

Values must be **raw provider units** for the declared collection (e.g. Sentinel-2 L2A surface-reflectance DN in `0..10000`), exactly what a provider fetch would return. Per-model normalization stays inside each embedder, so you never need to know a model's normalization. Data that looks already normalized (max ≤ 1.5 on an S2 declaration) triggers a warning.

---

## UserData

```python
from rs_embed import UserData

UserData(
    data: np.ndarray,                    # [C,H,W] or [T,C,H,W], raw provider values
    collection: str,                     # e.g. "COPERNICUS/S2_SR_HARMONIZED" or alias "s2"
    spatial: SpatialSpec | None = None,  # where the imagery is (PointBuffer / BBox)
    bands: tuple[str, ...] | None = None,  # one band name per channel; None = canonical order
    temporal: TemporalSpec | None = None,  # when the imagery was acquired
    scale_m: int | None = None,          # optional nominal pixel size, provenance only
)
```

- **`spatial` is optional but supply it whenever you have it** — models whose forward pass conditions on geometry (lat/lon or GSD encodings: `clay`, `prithvi`) refuse declarations without it, because coordinates are never fabricated. All other models accept ungeoreferenced data; they just lose location provenance in the metadata. `list_models_for_data` on a spatial-less declaration reports which models refuse for this reason.
- **`temporal` travels with the data** — it is the acquisition time of *this* imagery, so it lives here rather than on the API call. Models that condition on time read it; omitting it falls back to the package default window.
- **`bands` may be omitted only for the canonical case**: an S2 L2A declaration with exactly 12 channels defaults to the canonical order `B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12`. Any other channel count, order, or collection must name its bands — band identity is never guessed from channel count.

Collection aliases: `"s2"` / `"sentinel-2"` / `"s2-l2a"` → `COPERNICUS/S2_SR_HARMONIZED`; `"s1"` / `"sentinel-1"` → `COPERNICUS/S1_GRD`. Full collection ids pass through unchanged. Band aliases resolve the same way provider fetches resolve them (`"RED"` → `"B4"`, `"NIR_NARROW"` → `"B8A"`, …).

Multi-frame `[T,C,H,W]` arrays are only meaningful for time-series models (galileo, prithvi, olmoearth, anysat, agrifm); single-frame models reject them.

---

## Functions

### get_embedding_from_data

```python
import numpy as np
from rs_embed import UserData, get_embedding_from_data
from rs_embed.core.specs import PointBuffer, TemporalSpec

data = UserData(
    data=cube,                                     # [12, H, W] raw S2 L2A DN
    collection="s2",
    spatial=PointBuffer(lon=-88.2, lat=40.1, buffer_m=640),
    temporal=TemporalSpec.year(2022),
)
emb = get_embedding_from_data("galileo", data)
```

Returns one `Embedding`; `meta["user_input"]` records the declaration and the channel selection actually fed to the model (`declared_bands`, `bands_used`, `channel_indices`).

### get_embeddings_batch_from_data

```python
embs = get_embeddings_batch_from_data("galileo", datas)   # datas: list[UserData]
```

Each item is matched independently and carries its own `spatial` / `temporal`, so one batch can mix locations, dates, and even band orders. Items sharing a temporal are dispatched together (models with true batching benefit); results always come back in input order.

### list_models_for_data

```python
from rs_embed import list_models_for_data

report = list_models_for_data(data)
[r["model"] for r in report if r["compatible"]]
```

Runs the same matching against every catalog model without loading weights. Each entry has `model`, `compatible`, `bands_used`, and `reason` (why the model is incompatible).

---

## Refusal semantics

| Situation | Result |
|---|---|
| Declaration covers all model bands | Accepted; channels sliced/reordered |
| Missing band(s) | `ModelError` listing the missing band names |
| Collection mismatch | `ModelError` (raw units differ across collections) |
| Precomputed model | `ModelError` (no imagery input) |
| Channel count ≠ declared bands | `SpecError` from `UserData.validate()` |
| `bands=None` outside the canonical case | `SpecError` (declare bands explicitly) |
| Missing `spatial` on a georef-conditioned model (clay, prithvi) | `ModelError` (coordinates are never fabricated) |
| S2 values look normalized (max ≤ 1.5) | `UserWarning`, request still runs |

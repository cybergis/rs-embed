# API: User-Provided Data

This page covers the bring-your-own-data API: computing embeddings from imagery you already have, instead of provider-fetched imagery.

Related pages: [API: Embedding](api_embedding.md), [API: Specs and Data Structures](api_specs.md).

---

## Concept

Every on-the-fly model declares an input sensor (a collection plus an ordered band list). The bring-your-own-data path asks you to declare what your array is — the collection it came from and one band name per channel — and matches that declaration against the model's sensor:

- **Superset data is accepted**: if your declaration covers all bands the model needs, the needed channels are sliced out and reordered automatically. One 12-band Sentinel-2 L2A cube can serve models that need 3, 6, or 10 of those bands.
- **Insufficient data is refused**: a collection mismatch (e.g. S2 data offered to a MODIS model) or a missing band raises `ModelError` naming exactly what is missing. Precomputed models (`tessera`, `gse`, `copernicus`) are always refused — they have no imagery input.

Values must be **raw provider units** for the declared collection (e.g. Sentinel-2 L2A surface-reflectance DN in `0..10000`), exactly what a provider fetch would return. Per-model normalization stays inside each embedder, so you never need to know a model's normalization. Data that looks already normalized (max ≤ 1.5 on an S2 declaration) triggers a warning.

`spatial` is still required: several models condition on geometry (lat/lon, GSD), and embedding metadata records provenance. Pass the location your imagery covers.

---

## UserData

```python
from rs_embed import UserData

UserData(
    data: np.ndarray,            # [C,H,W] or [T,C,H,W], raw provider values
    collection: str,             # e.g. "COPERNICUS/S2_SR_HARMONIZED" or alias "s2"
    bands: tuple[str, ...],      # one band name per channel, e.g. ("B2", "B3", ...)
    scale_m: int | None = None,  # optional nominal pixel size, recorded as provenance
)
```

Collection aliases: `"s2"` / `"sentinel-2"` / `"s2-l2a"` → `COPERNICUS/S2_SR_HARMONIZED`; `"s1"` / `"sentinel-1"` → `COPERNICUS/S1_GRD`. Full collection ids pass through unchanged. Band aliases resolve the same way provider fetches resolve them (`"RED"` → `"B4"`, `"NIR_NARROW"` → `"B8A"`, …).

Multi-frame `[T,C,H,W]` arrays are only meaningful for time-series models (galileo, prithvi, olmoearth, anysat, agrifm); single-frame models reject them.

---

## Functions

### get_embedding_from_data

```python
from rs_embed import UserData, get_embedding_from_data
from rs_embed.core.specs import PointBuffer, TemporalSpec

emb = get_embedding_from_data(
    "galileo",
    data=UserData(data=cube, collection="s2", bands=bands),
    spatial=PointBuffer(lon=-88.2, lat=40.1, buffer_m=640),
    temporal=TemporalSpec.year(2022),
)
```

Returns one `Embedding`; `meta["user_input"]` records the declaration and the channel selection actually fed to the model (`bands_used`, `channel_indices`).

### get_embeddings_batch_from_data

Batch counterpart: `datas: list[UserData]` aligned with `spatials: list[SpatialSpec]`. Each item is matched independently, so items may declare different band orders.

### list_models_for_data

```python
from rs_embed import list_models_for_data

report = list_models_for_data(my_declaration)
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
| S2 values look normalized (max ≤ 1.5) | `UserWarning`, request still runs |

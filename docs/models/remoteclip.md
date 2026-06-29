# RemoteCLIP (`remoteclip`)

## Quick Facts

| Field                | Value                                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Model ID             | `remoteclip`                                                                                                     |
| Aliases              | `remoteclip_s2rgb`                                                                                               |
| Family / Backbone    | RemoteCLIP (CLIP-style ViT via `rshf.remoteclip.RemoteCLIP`)                                                     |
| Adapter type         | `on-the-fly`                                                                                                     |
| Training alignment   | High — uses open_clip's official CLIP preprocess (Resize 224 bicubic + CenterCrop + CLIP normalize), matching how RemoteCLIP was trained |

!!! success "RemoteCLIP In 30 Seconds"
    RemoteCLIP is a CLIP-style vision-language ViT continually fine-tuned on remote-sensing image-text pairs, so its embeddings live in a *shared* image/text space that supports caption-based retrieval — in `rs-embed` you are getting the visual side of that shared space from a 3-band RGB Sentinel-2 input.

    In `rs-embed`, its most important characteristics are:

    - RGB-only (`B4,B3,B2`) with a fixed `224×224` preprocessing path: see [Input Contract](#input-contract)
    - checkpoint override goes through `sensor.collection="hf:<repo>"` rather than an environment variable: see [Environment Variables / Tuning Knobs](#environment-variables-tuning-knobs)
    - `pooled` is the **projected CLIP image embedding** (`encode_image`, e.g. 512-d for the default ViT-B/32), identical across single / batch / tiled calls — not a token mean: see [Output Semantics](#output-semantics)

---

## Input Contract

| Field                 | Value                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------- |
| Backend               | provider (`auto` recommended)                                                          |
| `TemporalSpec`        | **required** `TemporalSpec.range(start, end)` — treated as filter-and-composite window |
| Default collection    | `COPERNICUS/S2_SR_HARMONIZED`                                                          |
| Default bands (order) | `B4, B3, B2`                                                                           |
| Default fetch         | `scale_m=10`, `cloudy_pct=30`, `composite="median"`                                    |
| `input_chw`           | `CHW`, `C=3` in `(B4,B3,B2)` order                                                     |
| Side inputs           | none                                                                                   |

!!! note "Checkpoint override via `sensor.collection`"
    Use `sensor.collection="hf:<repo_or_path>"` (e.g. `hf:MVRL/remote-clip-vit-base-patch32`) to swap in a different RemoteCLIP checkpoint — the `hf:` prefix is how this adapter distinguishes checkpoint overrides from regular provider collections.

---

## Preprocessing Pipeline

!!! note "`tile` is the default `input_prep`"
    RemoteCLIP follows the package-wide default: `input_prep=None` resolves to `"tile"` (large inputs are tiled + stitched to preserve native resolution). Note that RemoteCLIP `grid` output is an image-level CLIP ViT patch-token grid rather than a seamless dense geospatial field, so tiled grids can show stitching seams; pass `input_prep="resize"` if you prefer a single resized forward pass.

```mermaid
flowchart LR
    INPUT["S2 RGB"] --> PREP["Normalize → uint8\n→ CLIP preprocess (Resize 224 + CenterCrop + normalize)"]
    PREP --> FWD["CLIP ViT forward"]
    FWD --> POOL["pooled: encode_image (projected CLIP embedding)"]
    FWD --> GRID["grid: patch-token (D,H,W) from forward_intermediates"]
```

!!! note "Current adapter image size"
    The image size is fixed at `224` in this adapter path.

!!! note "Preprocessing path"
    `rshf.remoteclip.RemoteCLIP` exposes no `model.transform`, so the adapter always
    uses the open_clip-equivalent CLIP preprocess (it never silently diverges from a
    wrapper transform).

---

## Architecture Concept

```mermaid
flowchart LR
    subgraph Input
        RGB["S2 RGB\n(B4,B3,B2)"]
    end
    subgraph "CLIP ViT"
        RGB --> PRE["Preprocess\n(CLIP: Resize 224\n+ CenterCrop + normalize)"]
        PRE --> FWD["CLIP ViT\nforward"]
    end
    subgraph "Shared Image ↔ Text Space"
        FWD --> EMB["Embeddings support\ncaption-based\nsimilarity & retrieval"]
        EMB --> POOL["pooled:\nencode_image\n(projected embedding)"]
        EMB --> GRID["grid:\npatch-token (D,H,W)"]
    end
```

---

## Environment Variables / Tuning Knobs

| Env var                                                  | Default            | Effect                                                   |
| -------------------------------------------------------- | ------------------ | -------------------------------------------------------- |
| `RS_EMBED_REMOTECLIP_FETCH_WORKERS`                      | `8`                | Provider prefetch worker count for batch APIs            |
| `RS_EMBED_REMOTECLIP_BATCH_SIZE`                         | CPU:`8`, CUDA:`64` | Inference batch size for batch APIs                      |
| `HUGGINGFACE_HUB_CACHE` / `HF_HOME` / `HUGGINGFACE_HOME` | unset              | Controls HF cache path used for model snapshot downloads |

!!! info "Checkpoint override"
    Set `sensor.collection="hf:<repo_or_local_path>"` (not env-based in this adapter).

---

## Output Semantics

**`pooled`**: returns the projected CLIP image embedding from `encode_image` (e.g. 512-d for the default ViT-B/32 checkpoint) — the canonical RemoteCLIP visual representation that lives in the shared image/text space, suitable for similarity search and retrieval. This is computed identically across the single (`get_embedding`), batch, and tiled paths, so the output dimensionality and values do not depend on ROI size or which API you call.

**`grid`**: exposes the ViT patch-token layout, extracted from open_clip's `forward_intermediates` as the deepest dense feature map (`[D, Ht, Wt]`, e.g. `768×7×7` for ViT-B/32 @ 224; `tokens_kind="tokens_intermediates"`). The single and batch paths produce identical grids. Default/auto input preparation resolves to resize, and metadata records `input_prep.model_policy="resize_default_for_image_level_vit_patch_grid"`, `grid_semantics="vit_patch_tokens"`, and `grid_tile_recommended=false`.

---

## Examples

### Minimal example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="auto",
)
```

### Custom checkpoint via `sensor.collection="hf:..."`

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec, SensorSpec

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    sensor=SensorSpec(
        collection="hf:MVRL/remote-clip-vit-base-patch32",
        bands=("B4", "B3", "B2"),
        scale_m=10,
        cloudy_pct=30,
        composite="median",
    ),
    output=OutputSpec.grid(),
    backend="auto",
)
```

---

## Paper & Links

- **Publication**: [TGRS 2024](https://arxiv.org/abs/2306.11029)
- **Code**: [ChenDelong1999/RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)

---

## Reference

- Provider-only — `backend="tensor"` is not supported.
- The `rshf` wrapper exposes no `model.transform`, so the adapter always applies the open_clip-equivalent CLIP preprocess — there is no second, divergent preprocessing path.
- `pooled` uses `encode_image` (projected CLIP embedding) on every path (single / batch / tiled); it is not a token-mean and its dimensionality does not change with ROI size.
- `grid` tokens come from open_clip's `forward_intermediates` patch grid; if a wrapper exposes no dense features the adapter falls back to a vision-transformer forward hook (batch-first / sequence-first safe).
- `input_prep` defaults to `tile` (the package-wide default); tiled RemoteCLIP patch-token grids can show stitching seams, so pass `input_prep="resize"` for a single resized forward pass when that matters.

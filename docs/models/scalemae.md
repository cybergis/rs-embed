# ScaleMAE RGB (`scalemae`)

> Sentinel-2 RGB on-the-fly adapter for ScaleMAE (`rshf.scalemae.ScaleMAE`), with explicit scale conditioning via `sensor.scale_m -> effective input_res_m`.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `scalemae` |
| Aliases | `scalemae_rgb` |
| Family / Backbone | ScaleMAE via `rshf.scalemae.ScaleMAE` |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | S2 RGB (`B4,B3,B2`) + `input_res_m` |
| Default resolution | 10m default provider fetch / source scale (`sensor.scale_m`) |
| Temporal mode | range window in practice (normalized via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | **required effective scale** (`input_res_m` derived from `sensor.scale_m` after preprocess) |
| Training alignment (adapter path) | High when `sensor.scale_m` matches the source patch resolution semantics |

---

## When To Use This Model

ScaleMAE is the right choice for RGB experiments where spatial scale conditioning matters, for comparisons against SatMAE or RemoteCLIP with an explicitly scale-aware backbone, and for robustness studies across resolution changes where `scale_m` is part of the logged setup.

It becomes harder to interpret when `sensor.scale_m` is missing or semantically wrong for the input patch. You should also avoid assuming that `grid` is always available, because some wrapper outputs are pooled vectors only, and cross-run comparisons should record the output type through fields such as `tokens_kind`.

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

The current adapter path is provider-only, so use `backend="gee"` or another provider-compatible backend. `TemporalSpec` is normalized through the shared helper, and `TemporalSpec.range(...)` remains the clearest option for reproducible runs.

### Sensor / channels + scale

Default `SensorSpec` if omitted:

The default sensor is `COPERNICUS/S2_SR_HARMONIZED` with bands `("B4", "B3", "B2")`, `scale_m=10`, `cloudy_pct=30`, and `composite="median"`.

The declarative `input_spec.normalization` for this adapter is `s2_sr_raw`, which here means the provider contract remains raw Sentinel-2 SR and the adapter applies its own RGB conversion plus ImageNet eval preprocessing afterward.

`input_chw` contract:

`input_chw` must be `CHW` with 3 channels in `(B4,B3,B2)` order, and the adapter expects raw Sentinel-2 SR values in `0..10000`.

Scale requirement:

The adapter derives `input_res_m` from `sensor.scale_m` and the actual preprocessed tensor geometry (`Resize(short side) + CenterCrop`). If `sensor.scale_m` does not match the source patch resolution semantics, the resulting embeddings are not meaningfully comparable.

---

## Preprocessing Pipeline (Current rs-embed Path)

<pre class="pipeline-flow"><code><span class="pipeline-root">INPUT</span>  provider fetch / input_chw
  <span class="pipeline-arrow">-&gt;</span> S2 RGB patch
     <span class="pipeline-detail">input_chw path: raw SR -&gt; [0,1] -&gt; uint8</span>
  <span class="pipeline-arrow">-&gt;</span> ImageNet eval preprocess
     <span class="pipeline-detail">Resize(short side) -&gt; CenterCrop -&gt; ToTensor -&gt; Normalize(ImageNet)</span>
  <span class="pipeline-arrow">-&gt;</span> derive effective input_res tensor from source `sensor.scale_m`
  <span class="pipeline-arrow">-&gt;</span> ScaleMAE forward
     <span class="pipeline-branch">path:</span> official `forward_features(...)`
     <span class="pipeline-detail">adapter unwraps common rshf wrappers (for example nested `.model`) and passes patch_size + input_res compatibly</span>
  <span class="pipeline-arrow">-&gt;</span> normalize output format
     <span class="pipeline-branch">tokens:</span> [N,D]
     <span class="pipeline-branch">pooled:</span> [D]
     <span class="pipeline-branch">feature map:</span> [C,H,W] -&gt; tokens
  <span class="pipeline-arrow">-&gt;</span> output projection
     <span class="pipeline-branch">pooled:</span> vector
     <span class="pipeline-branch">grid:</span>   token grid</code></pre>

Important:

`grid` output requires a token sequence after adapter normalization. If the model or wrapper returns pooled vectors only, `OutputSpec.grid()` raises a clear error instead of silently fabricating a grid.

---

## Environment Variables / Tuning Knobs

| Env var                           | Default                      | Effect                                          |
| --------------------------------- | ---------------------------- | ----------------------------------------------- |
| `RS_EMBED_SCALEMAE_ID`            | `MVRL/scalemae-vitlarge-800` | HF model ID for `ScaleMAE.from_pretrained(...)` |
| `RS_EMBED_SCALEMAE_IMG`           | `224`                        | Resize / preprocess image size                  |
| `RS_EMBED_SCALEMAE_FETCH_WORKERS` | `8`                          | Provider prefetch workers for batch APIs        |
| `RS_EMBED_SCALEMAE_BATCH_SIZE`    | CPU:`8`, CUDA:`32`           | Inference batch size for batch APIs             |

Non-env but critical:

Even though it is not an environment variable, `sensor.scale_m` is a critical runtime setting because the adapter uses it to derive the effective `input_res_m` passed to ScaleMAE.

---

## Output Semantics

### `OutputSpec.pooled()`

If the adapter receives a token sequence `[N,D]`, `OutputSpec.pooled()` pools patch tokens with `mean` or `max`. If the model already returns a pooled vector `[D]`, that vector is returned directly and metadata marks the path as `model_pooled`. Metadata also records fields such as `tokens_kind`, `used_patch_size`, and `used_scale_m`.

### `OutputSpec.grid()`

`OutputSpec.grid()` requires a token sequence after adapter normalization and returns a patch-token grid as `xarray.DataArray` with shape `(D,H,W)`. As with the other ViT-like pages, this is model token layout rather than georeferenced raster pixels.

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "scalemae",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example tuning (env + scale semantics)

```python
# Example (shell):
# export RS_EMBED_SCALEMAE_ID=MVRL/scalemae-vitlarge-800
# export RS_EMBED_SCALEMAE_IMG=224
#
# In code, keep sensor.scale_m correct (the adapter derives effective input_res_m from it).
```

---

## Common Failure Modes / Debugging

- backend mismatch (`scalemae` is provider-only)
- wrong `input_chw` shape / band order (`CHW`, 3 channels, `(B4,B3,B2)`)
- missing `rshf.scalemae.ScaleMAE`
- wrapper mismatch in older/newer `rshf` versions, especially wrappers that hide the real ScaleMAE backbone and do not expose `forward_features()` even through common nested attributes such as `.model`
- `grid` requested when model output is pooled vector only
- incorrect `sensor.scale_m` causing silent comparison drift
- mismatch between expected source resolution and the post-preprocess effective `input_res_m`

Recommended first checks:

Start by inspecting metadata such as `tokens_kind`, `used_patch_size`, `input_res_m`, `resize_short_side`, and `used_scale_m`. Then verify `sensor.scale_m` and `RS_EMBED_SCALEMAE_IMG`, and use `OutputSpec.pooled()` first if you are isolating a grid-layout issue.

---

## Reproducibility Notes

For reproducibility, keep `RS_EMBED_SCALEMAE_ID`, `RS_EMBED_SCALEMAE_IMG`, and especially `sensor.scale_m` fixed, along with the temporal window, compositing settings, output mode, pooling choice, and `rshf` version.

---

## Source of Truth (Code Pointers)

The main implementation points are `src/rs_embed/embedders/catalog.py`, `src/rs_embed/embedders/onthefly_scalemae.py`, and the shared helpers in `src/rs_embed/embedders/_vit_mae_utils.py`.

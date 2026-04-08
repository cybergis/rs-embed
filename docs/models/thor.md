# THOR (`thor`)

> Vendored THOR adapter for Sentinel-2 SR 10-band inputs, with THOR-specific normalization and flexible group-grid aggregation (`mean` / `sum` / `concat`).

## Quick Facts

| Field                             | Value                                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------- |
| Model ID                          | `thor`                                                                                            |
| Aliases                           | `thor_1_0_base`                                                                                   |
| Family / Backbone                 | Fully vendored THOR runtime (`thor_v1_tiny` / `thor_v1_small` / `thor_v1_base` / `thor_v1_large`) |
| Adapter type                      | `on-the-fly`                                                                                      |
| Typical backend                   | provider backend (`gee`)                                                                          |
| Primary input                     | S2 SR 10-band `CHW`                                                                               |
| Default resolution                | 10m default provider fetch (`sensor.scale_m`)                                                     |
| Temporal mode                     | `range` in practice (composite window)                                                            |
| Output modes                      | `pooled`, `grid`                                                                                  |
| Model config keys                 | `variant` (default: `base`; choices: `tiny`, `small`, `base`, `large`)                            |
| Extra side inputs                 | none required in current adapter                                                                  |
| Training alignment (adapter path) | High when `thor_stats` normalization and default S2 SR setup are preserved                        |

---

## When To Use This Model

THOR is a strong Sentinel-2 SR baseline when you want pretrained THOR weights, both pooled and token-grid outputs, or experiments where group-wise token aggregation through `group_merge` is part of the analysis.

As with the other model pages, the runtime knobs here affect representation semantics. Moving away from `thor_stats` normalization, changing `patch_size` or `image_size` without logging `ground_cover_m`, or assuming `grid` is always available can all make comparisons harder to interpret.

One THOR-specific characteristic is that `RS_EMBED_THOR_PATCH_SIZE` directly changes how finely the vendored THOR backbone tokenizes each Sentinel-2 group. Smaller patch sizes keep the same geographic crop but produce denser token layouts and higher compute cost; larger patch sizes coarsen the token grid and reduce compute. In the current S2 10m/20m adapter path, `RS_EMBED_THOR_PATCH_SIZE=8` is the default because it stays divisible with `RS_EMBED_THOR_IMG=288` while giving a denser grid than the previous `16`-pixel default.

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

The current adapter path is provider-only, so use `backend="gee"` or another provider-compatible backend. `TemporalSpec` is normalized to a range via the shared helper, with `TemporalSpec.range(...)` recommended for reproducibility. The temporal window controls compositing rather than locking the request to a single source scene.

### Sensor / channels

Default `SensorSpec` if omitted:

The default sensor is `COPERNICUS/S2_SR_HARMONIZED` with adapter band order `B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`, `scale_m=10`, `cloudy_pct=30`, `composite="median"`, and `fill_value=0.0`.

`input_chw` contract:

`input_chw` must be `CHW` with `C=10`, and raw Sentinel-2 SR values are expected in `0..10000`. Before normalization, the adapter clips NaN and Inf values and clamps the range to `0..10000`.

---

## Preprocessing Pipeline (Current rs-embed Path)

<pre class="pipeline-flow"><code><span class="pipeline-root">INPUT</span>  provider fetch / input_chw
  <span class="pipeline-arrow">-&gt;</span> S2 SR 10-band composite patch
  <span class="pipeline-arrow">-&gt;</span> optional raw-value inspection
     <span class="pipeline-detail">expected_channels=10, value range 0..10000</span>
  <span class="pipeline-arrow">-&gt;</span> normalize with RS_EMBED_THOR_NORMALIZE
     <span class="pipeline-branch">thor_stats:</span> /10000 -&gt; THOR z-score stats
     <span class="pipeline-branch">unit_scale:</span> /10000 -&gt; clip [0,1]
     <span class="pipeline-branch">none / raw:</span> keep clipped raw values
  <span class="pipeline-arrow">-&gt;</span> resize to RS_EMBED_THOR_IMG=288
  <span class="pipeline-arrow">-&gt;</span> build/load THOR backbone
     <span class="pipeline-detail">ground_cover_m = sensor.scale_m * image_size</span>
     <span class="pipeline-detail">patch_size passed through THOR build params</span>
  <span class="pipeline-arrow">-&gt;</span> forward model to token sequence [N,D]
  <span class="pipeline-arrow">-&gt;</span> grid construction
     <span class="pipeline-branch">preferred:</span> THOR group-aware token layout (group_merge)
     <span class="pipeline-branch">fallback:</span>  generic square patch-token reshape
     <span class="pipeline-branch">failure:</span>   grid unavailable</code></pre>

---

## Environment Variables / Tuning Knobs

| Env var                       | Default        | Effect                                               |
| ----------------------------- | -------------- | ---------------------------------------------------- |
| `RS_EMBED_THOR_MODEL_KEY`     | `thor_v1_base` | THOR backbone key for the vendored runtime           |
| `RS_EMBED_THOR_CKPT`          | unset          | Local checkpoint path override                       |
| `RS_EMBED_THOR_PRETRAINED`    | `1`            | Use pretrained weights (HF default path)             |
| `RS_EMBED_THOR_IMG`           | `288`          | Resize target image size                             |
| `RS_EMBED_THOR_NORMALIZE`     | `thor_stats`   | `thor_stats`, `unit_scale`, or `none`                |
| `RS_EMBED_THOR_GROUP_MERGE`   | `mean`         | THOR group-grid aggregation: `mean`, `sum`, `concat` |
| `RS_EMBED_THOR_PATCH_SIZE`    | `8`            | THOR flexi patch size parameter                      |
| `RS_EMBED_THOR_FETCH_WORKERS` | `8`            | Provider prefetch workers for batch APIs             |

Notes:

`RS_EMBED_THOR_PATCH_SIZE` and `RS_EMBED_THOR_IMG` jointly affect token layout and `ground_cover_m`. Changing `group_merge` also changes grid channel semantics and dimensionality, especially when `concat` is used.

For the current Sentinel-2 SR 10-band path, a practical rule is to keep `RS_EMBED_THOR_IMG` divisible by `2 * RS_EMBED_THOR_PATCH_SIZE`. That keeps the default 10m and 20m THOR groups aligned to valid patch grids. With the default `RS_EMBED_THOR_IMG=288`, common valid choices include `4`, `6`, `8`, `12`, `16`, and `18`.

If you want to preserve the same geographic footprint (`ground_cover_m=2880` at the default 10m fetch), keep `RS_EMBED_THOR_IMG=288` and only change `RS_EMBED_THOR_PATCH_SIZE`. If you choose a patch size that no longer divides cleanly, change `RS_EMBED_THOR_IMG` together with it.

![patch size](assets/thor.png)

## Model-specific Settings

| Key       | Type     | Default | Choices                          | Notes                                                                                                         |
| --------- | -------- | ------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `variant` | `string` | `base`  | `tiny`, `small`, `base`, `large` | Backbone size selector. For export jobs, pass it through `ExportModelRequest.configure("thor", variant=...)`. |

Example:

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "thor",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
    variant="large",
)
```

---

## Output Semantics

### `OutputSpec.pooled()`

`OutputSpec.pooled()` pools the token sequence through `_pool_thor_tokens(...)`. When the expected THOR patch-token count is available, the adapter uses it to avoid pooling non-patch tokens incorrectly. Metadata records the pooling mode and `cls_removed`.

### `OutputSpec.grid()`

`OutputSpec.grid()` first tries to build a THOR group-aware grid with `grid_kind="thor_group_grid"` from the channel groups. If that fails, it falls back to a generic ViT-style patch-token reshape with `grid_kind="patch_tokens"`. Some token layouts still cannot be reshaped into a grid, in which case the adapter raises a clear error and suggests using pooled output.

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "thor",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example THOR tuning (env-controlled)

```python
# Example (shell):
# export RS_EMBED_THOR_NORMALIZE=thor_stats
# export RS_EMBED_THOR_GROUP_MERGE=mean
# export RS_EMBED_THOR_IMG=288
# export RS_EMBED_THOR_PATCH_SIZE=8
```

### Example: keep the same coverage, use a denser token grid

```bash
export RS_EMBED_THOR_IMG=288
export RS_EMBED_THOR_PATCH_SIZE=4
```

This keeps the same `ground_cover_m` as the default path, but increases the token density substantially. Use this when spatial detail matters more than runtime or memory.

### Example: keep the same coverage, use a coarser token grid

```bash
export RS_EMBED_THOR_IMG=288
export RS_EMBED_THOR_PATCH_SIZE=12
```

This preserves the same geographic crop while reducing token count compared with `patch_size=8`.

### Example: choose a patch size that requires changing image size too

```bash
export RS_EMBED_THOR_IMG=280
export RS_EMBED_THOR_PATCH_SIZE=10
```

With `patch_size=10`, the default `288` image size does not divide cleanly for THOR's 10m/20m S2 groups, so `RS_EMBED_THOR_IMG` needs to move to a compatible value such as `280` or `300`.

### Example with variant selection

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "thor",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.grid(pooling="mean"),
    backend="gee",
    variant="small",
)
```

Use `variant` only for backbone size selection.
Other THOR runtime knobs such as image size, normalization, patch size, and checkpoint override
still use the existing environment-variable path.

---

## Common Failure Modes / Debugging

- broken runtime deps (`torch`, `timm`, `einops`)
- wrong `input_chw` shape (`C` must be `10`)
- invalid `RS_EMBED_THOR_GROUP_MERGE` (must be `mean` / `sum` / `concat`)
- grid unavailable for chosen config (token layout not square and group parsing failed)
- normalization mismatch causing unstable comparison across runs

Recommended first checks:

Start by verifying metadata such as `model_key`, `normalization`, `group_merge`, `patch_size`, and `ground_cover_m`. `OutputSpec.pooled()` is the faster way to isolate grid-layout issues, and the default `thor_stats` plus `group_merge=mean` combination is the best baseline before benchmarking alternatives.

---

## Reproducibility Notes

For reproducibility, keep `RS_EMBED_THOR_MODEL_KEY`, `RS_EMBED_THOR_PRETRAINED`, and any local checkpoint path fixed, together with normalization mode, image size, patch size, `group_merge`, temporal window, and provider compositing settings.

---

## Source of Truth (Code Pointers)

The main implementation files are `src/rs_embed/embedders/catalog.py`, `src/rs_embed/embedders/onthefly_thor.py`, and the shared token/grid helpers in `src/rs_embed/embedders/_vit_mae_utils.py`.

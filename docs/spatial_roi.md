# Spatial ROI Handling: From Region to Embedding

You pass `rs-embed` a geographic **region** and get back an embedding **of that
region**. But most on-the-fly model encodes a **fixed-size square** input (e.g.
256×256 tokens). This page explains the contract that bridges the two — how an
arbitrary ROI (a thin rectangle, a near-square box, or a large non-square area)
becomes the square the encoder wants, and how the output is mapped back to your
ROI so the embedding still describes the region you asked for.

This is the spatial counterpart to [Temporal Sampling](temporal_sampling.md): the
same way a `TemporalSpec.range` window is turned into the frames each model wants,
a spatial ROI is turned into the square frame each encoder wants — and then mapped
back.

!!! abstract "The one idea"
    A rectangular ROI is **enlarged to a square of real imagery** (not stretched),
    encoded, and the output is **cropped back to the ROI**. So a 1.8:1 field is
    never smeared into striped embeddings, and the returned vector/grid describes
    your region, not the enlarged square.

---

## The contract, in three stages

```mermaid
flowchart LR
    ROI["Your ROI\n(BBox / PointBuffer)"] --> SQ["1. Square the request\nenlarge rectangle → centered\nsquare of real imagery\n(record roi_window)"]
    SQ --> ENC["2. Encode the square\nresize to model size,\nor tile at native res"]
    ENC --> CROP["3. Crop output back\ngrid → crop to roi_window\npooled → pool ROI tokens only"]
    CROP --> OUT["Embedding of your ROI"]
```

**1. Square the request — fetch real pixels, don't stretch.**
If the ROI is a rectangle, `rs_embed.tools.spatial.square_spatial` enlarges it to a
**centered square in EPSG:3857**, so the model sees the *real* surrounding imagery
instead of a stretched rectangle. It returns the ROI's normalized position inside
that square (`roi_window = (y0, y1, x0, x1)`). A `PointBuffer` or an already-square
`BBox` is left unchanged (`roi_window` = the full frame).

**2. Encode the square.**
The undistorted square is run through the encoder — resized to the model's fixed
input size on the resize path, or split into native-resolution tiles on the tile
path (`input_prep="tile"`, the package default for most models).

**3. Crop the output back to the ROI.**
`grid` output is cropped to `roi_window` (`crop_grid_for_output`). `roi_window` is relative to the fetched square, so a model that pads, crops or center-crops that array on the way in first maps the window into its own input's frame (`model_roi_window`, through `meta["shape_prep"]["input_window"]`), and the crop also drops whole cells of such padding. `pooled` output pools the ROI's tokens only when the ROI is a sub-window, and otherwise keeps the model's own pooled vector (`olmoearth` and `thor` pool only the input's valid region even then); see [Pooled output for an ROI sub-window](#pooled-output-for-an-roi-sub-window). The stitched grid from the tile path is cropped the same way, and tiled `pooled` embeddings are weighted by each tile's ROI-overlap area. The result describes **your** ROI, not the enlarged square.

### Pooled output for an ROI sub-window

An ROI-pooled vector is pooled by the model's own rule, so it is comparable with the same model's pooled vector for a whole input: it has the same shape and lies in the same space (`shape.pool_for_output` checks the shape).

- **Models that pool tokens** (`satmae`, `satmaepp`, `scalemae`, `prithvi`, `terramind`, `galileo`, `agrifm`, `fomo`, `satmaepp_s2`, `anysat` with the default `pooled_source="patch"`, and `wildsat` with token pooling) reduce the ROI's tokens with `pooling` (`mean` or `max`), as they reduce all tokens for a whole input. Models with several tokens per cell (spectral groups, modalities, frames) reduce those with the same operation.
- **Models whose pooled vector is a CLS token or a fixed mean** (`clay`, `dofa`, `terrafm`, `remoteclip`, `wildsat`, `anysat` with `pooled_source="tile"`, `satvision_toa`) mean-pool the ROI's tokens and pass the mean through the same final norm, head or projection: `fc_norm` for `dofa`, `ln_post` and `proj` for `remoteclip`, torchvision's `heads` and the image head for `wildsat`. `pooling="max"` does not change these vectors, for a whole input or an ROI, and their tiles from `input_prep="tile"` merge as a mean too; `meta["requested_pooling"]` records the request. For a CLS model the ROI vector is a token mean in the CLS token's space, not a CLS token: it is the closest comparable vector without running the model on the ROI alone.

An ROI-pooled vector is labelled `meta["pooling"] = "roi_grid_<mean|max>"` (`prithvi` multi-frame: `roi_grid_temporal_<mean|max>`) with `meta["roi_pooled"] = True` and the pooled cells' extent in `meta["roi_grid_hw"]`. An almost-square ROI whose margin is under one token on every side covers every token, so it keeps the model's own pooled vector. When a model has no token grid for the input (a wrapper that only returns pooled vectors, a backbone without patch tokens, or `fomo` tokens that do not match its patch grid), the pooled vector covers the whole fetched square instead: a `UserWarning` says so and `meta["roi_pooled"]` is `False`.

---

## By ROI shape

| Your ROI | What happens | Output |
| -------- | ------------ | ------ |
| **Small rectangle** (e.g. 1.8:1) | Enlarge to the bounding square, encode, crop the grid back to the rectangle | Rectangular token grid / ROI-only pooled vector |
| **Near-square** (≈ model input) | No enlargement; straight resize + encode | Full square grid |
| **Large non-square** | Tiled at native resolution; stitched grid cropped back, pooled weighted by ROI overlap | ROI-shaped stitched field / ROI-weighted vector |
| **Point** (`PointBuffer`) | Already square; encode directly | Full square grid |

---

## Scope: which models

This contract applies to **all on-the-fly models** — anything that runs a
fixed-input-size encoder (`prithvi`, `olmoearth`, `galileo`, `anysat`, `agrifm`,
`dofa`, `fomo`, `terramind`, `terrafm`, `thor`, `satmae`, `satmaepp`, `scalemae`,
`remoteclip`, `wildsat`, `satvision`).

**Precomputed products (`tessera`, `gse`, `copernicus`) are exempt.** They sample
your ROI directly from a precomputed global grid — there is no fixed model input to
square against, so no enlarge/crop step is needed (`gse` does its own large-request
tiling; see its page).

!!! note "Token granularity limits the crop"
    Cropping back is only as fine as the model's token grid. A coarse grid (e.g.
    `satvision`'s 4×4 SwinV2-Giant output) cannot isolate a sub-2:1 ROI — that is a
    token-granularity limit, not a bug. Finer-grid models crop cleanly.

---

## Fallback: when the request can't be fetch-squared

`square_spatial` cannot always enlarge — the square might run past valid lat/lon at
the poles or antimeridian, or you may pass a non-square array directly via
`input_chw` (bypassing the fetch). In those cases the array itself is squared by
`rs_embed.tools.shape.prepare_square`:

| Case | What happens | `meta["shape_prep"]["applied"]` |
| ---- | ------------ | ------------------------------- |
| Already square | Resize to `image_size` | `none` |
| Any rectangle | Make square (pad **or** crop), then resize; the input's position in the square is recorded (`input_window`) so ROI windows map through it and the output is cropped back | `pad_to_square` / `crop_to_square` |

`pad` (default) reflect-pads the short side (keeps the whole ROI, adds synthetic
border); `crop` center-crops the long side (no synthetic data, discards margins).
There is no silent stretch fallback: extremely rectangular inputs (`aspect ≥ 2.0`)
still pad, with a `UserWarning` that most of the model input is synthetic border —
prefer a square fetch of real imagery (automatic on provider-backed paths) or
`input_prep="tile"` for such ROIs.

### The `shape_adjust` knob

For most models this fallback is internal and fixed to `pad`. One model exposes the
choice:

| Model | How to set | Default |
| ----- | ---------- | ------- |
| [`olmoearth`](models/olmoearth.md) | `shape_adjust="pad"`/`"crop"` (keyword / `model_config`) | `pad` |

[`thor`](models/thor.md) also squares non-square inputs, but through its own bounded
`native_snap` path (`RS_EMBED_THOR_SHAPE_ADJUST`, default `crop`) — separate from
this fallback; see the THOR page.

---

## What gets recorded

| Field | Meaning |
| ----- | ------- |
| `meta["input_prep"]["roi_cropped"]` | `True` when the output was cropped back to a sub-ROI |
| `meta["shape_prep"]["applied"]` | `none` / `pad_to_square` / `crop_to_square` (fallback path); `pad_bottom_right` for `prithvi` with `RS_EMBED_PRITHVI_PREP=pad`, which pads only the bottom and right edges up to a patch multiple and records just `orig_hw`, `square_hw` and the windows |
| `meta["shape_prep"]["orig_hw"]` / `["square_hw"]` / `["target_hw"]` | ROI shape, squared shape, final model size |
| `meta["shape_prep"]["aspect"]` | `long_side / short_side` (≥ 2.0 triggers the heavy-padding warning) |
| `meta["shape_prep"]["input_window"]` | Where the input lies in the squared array, as exact fractions `(y0, y1, x0, x1)`; past `[0, 1]` on a cropped axis. Recorded by models that square their input (`clay` and `dofa` under `meta["preprocess"]["resize_meta"]["shape_prep"]`; `remoteclip` only with its built-in preprocessing) |
| `meta["shape_prep"]["roi_window"]` | `input_window` clipped to the square |

`roi_cropped=True` means you got the ROI, not the square.

---

## Practical advice

The contract removes aspect distortion but **cannot invent spatial detail** the ROI
never had. For tiny ROIs the real fix is to request a **larger, roughly square**
BBox (e.g. ~2.56 km at 10 m → native 256×256) so no upsampling is needed at all.
Keeping `aspect < 2.0` also keeps the pad fallback (direct `input_chw` rectangles)
mostly real imagery instead of synthetic border.

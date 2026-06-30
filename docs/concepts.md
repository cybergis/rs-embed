# Core Concepts

This page explains the semantic meaning of the main rs-embed abstractions.
It is a supplement to [Quickstart](quickstart.md) and [API: Specs and Data Structures](api_specs.md), not a separate getting-started path.

---

## The Core Abstraction

`rs-embed` exposes a unified interface:

`(model, spatial, temporal, sensor, output) -> Embedding`

In practice, most users work with `spatial` for where, `temporal` for when, `output` for what shape they want back, and `backend` for the data-access route. In normal public usage, `backend="auto"` is the default, and `gee` is the most common explicit provider override.

This page explains what those words mean in practice.

---

## Spatial Specs: What Area You Want

Two common ways to define a region are `PointBuffer(lon, lat, buffer_m)` for a square ROI centered at one point and `BBox(minlon, minlat, maxlon, maxlat)` for explicit latitude and longitude bounds.

The public API currently accepts only `crs="EPSG:4326"` for `BBox` and `PointBuffer`.

For exact constructors and validation rules, see [API: Specs and Data Structures](api_specs.md).

---

## Temporal Specs: Window, Not Necessarily a Single Scene

This is usually the most important semantic distinction to get right.

### `TemporalSpec.year(...)`

Used mainly for annual precomputed products (for example `gse`).

### `TemporalSpec.range(start, end)`

For most on-the-fly GEE-backed models, this means:

1. Filter imagery within the half-open window `[start, end)`
2. Composite the images (default `median`, optional `mosaic`)
3. Feed the composite patch into the model

It usually does **not** mean "pick a single image acquired exactly on this date."

!!! tip
    If you want a near single-day query, use a one-day window such as
    `TemporalSpec.range("2022-06-01", "2022-06-02")`.

See detailed model-specific temporal behavior in [Supported Models](models.md).

---

## Output Specs: `pooled` vs `grid`

### `OutputSpec.pooled()`

Returns one vector `(D,)` for the whole ROI.

Use `pooled` for classification, retrieval, clustering, and most cross-model benchmarking work.

Conceptually, this is one ROI in and one embedding vector out. It is also the easiest format to compare across different model families. The main advantage is semantic simplicity and small output size, not a guarantee of much lower inference cost.

### `OutputSpec.grid()`

Returns a spatial feature grid `(D, H, W)`.

Use `grid` for visualization, patch-wise analysis, and spatial structure inspection.

!!! note
    For ViT-like models, `grid` is often a token/patch grid, not guaranteed georeferenced raster pixels.

Conceptually, this is one ROI in and one spatial embedding field out. It is useful when spatial layout matters more than a single pooled descriptor. For many ViT-like models, `grid` is not much more expensive at forward time than `pooled`; the extra cost mostly comes from reconstruction, larger outputs, and downstream processing.

---

## Backends and Providers

Think of backend as the input retrieval and runtime path. `backend="auto"` is the recommended default and lets rs-embed choose the model-compatible path. `backend="gee"` explicitly fetches imagery from Google Earth Engine and is common for on-the-fly models.

You usually do not need to customize providers directly unless you are debugging inputs or extending the library.

---

## `sensor`: Only Needed for Some Paths

For on-the-fly models, `SensorSpec(...)` describes the collection, bands, sampling scale, cloud filtering, and compositing mode.

For most precomputed models, `sensor` is often `None` or ignored.

---

## Input Prep (`resize` / `tile` / `auto`)

`input_prep` is an API-level policy for large on-the-fly inputs. `"tile"` is the default, preserving native resolution by running API-side tiled inference and stitching the result; `"resize"` is the faster opt-in that downsamples the whole ROI to a single input; and `"auto"` is a conservative automatic choice that mainly matters for some `grid` outputs.

Pass `input_prep="resize"` when a downsampled whole-ROI input is good enough and you want to skip the cost of tiling.

Some image-level ViT adapters expose `grid` as patch-token layout rather than a seamless dense geospatial field. For `scalemae`, `satmae`, and `satmaepp`, `input_prep=None` or `"auto"` with `OutputSpec.grid()` resolves to `"tile"` (the package-wide default) and emits a warning about stitching seams; explicit `"tile"` warns the same way. Explicit `"resize"` is the seamless (downsampled) opt-in and does not warn.

This is a runtime policy choice, not a model identity choice.
Use it when the same model needs different large-ROI handling in different workflows.

---

## Precomputed vs On-the-fly Models

### Precomputed

Precomputed models read embeddings from existing embedding products. They are usually faster and simpler to run, but their temporal coverage, resolution, and sometimes projection are fixed by the product. Typical examples are `tessera`, `gse`, and `copernicus`.

For example, `tessera` and `copernicus` currently keep product-native grid semantics instead of following the common provider-backed EPSG:3857 sampling default used by many on-the-fly paths. That distinction matters for `grid` outputs and should be recorded when comparing models.

### On-the-fly

On-the-fly models fetch an imagery patch, preprocess it, and then run inference. They are more flexible, but they come with heavier runtime dependencies and require more care around bands, temporal windows, and normalization. Typical examples are `remoteclip`, `prithvi`, `anysat`, and `terramind`.

---

## Where To Go Next

For runnable examples, go to [Quickstart](quickstart.md). For task recipes, use [Workflows](workflows.md). For model-specific assumptions, use [Models](models.md). For exact type definitions, use [API: Specs and Data Structures](api_specs.md).

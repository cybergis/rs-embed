# Changelog

All notable changes to `rs-embed` will be documented in this file.

The project keeps this changelog as the canonical release record. GitHub Releases should summarize the same versioned changes rather than introducing a second source of truth.

The format is based on Keep a Changelog, and the project follows Semantic Versioning with extra care around model and embedding semantics.

## [Unreleased]

### Changed

- **`satmaepp_s2_10b` is no longer a standalone model; the Sentinel-2 10-band path is now the `"s2_10b"` modality of `satmaepp`.** SatMAE++'s two sensor configurations live under one model name, selected with `modality=` (the codebase's mechanism for same-model/different-sensor, like `terrafm`'s s2/s1): `get_embedding("satmaepp")` is the default fMoW-RGB 3-band path and `get_embedding("satmaepp", modality="s2_10b")` is the grouped-channel 10-band path; for exports use `export_batch(..., per_model_modalities={"satmaepp": "s2_10b"})`. The `satmaepp_s2_10b`, `satmaepp_s2`, and `satmaepp_sentinel10` model names (and their `variant`/env knobs) are removed — **migrate to `modality="s2_10b"`.** Embeddings and metadata for the 10-band path are unchanged (now labelled under `model="satmaepp"`); the model catalog drops from 20 to 19 entries.

### Fixed

- **`export_batch` emitted spurious `All-NaN slice` / `Mean of empty slice` RuntimeWarnings for multi-temporal models with an empty leading temporal bin.** The prefetch input inspection (`inspect_fetch_result`) inspected frame 0 of a multi-frame `[T,C,H,W]` stack unconditionally; when that bin had no usable imagery (a NaN-sentinel frame that the embedder drops downstream), the per-band nan-reductions warned over an all-NaN frame. It now inspects the first frame carrying finite data, falling back to frame 0 only when every frame is empty (still flagged `ok=False`), and reports the chosen frame via a new `inspected_frame` field. Only the prefetch/export path runs this inspection, so `get_embedding`/`get_embeddings_batch` were never affected.

## [0.2.0] — 2026-06-29

This release adds the OlmoEarth model family, makes the temporal models (`prithvi`, `galileo`, `olmoearth`) sample window-adaptively by default, and fixes several `export_batch` correctness issues where a point's embedding differed between the single and batch/tiled paths. Some defaults that change embedding behavior are noted below; pin explicit options where strict reproducibility across versions is required.

### Added

- **OlmoEarth v1/v1.1 embedder (`olmoearth`).** Adds the [OlmoEarth](https://huggingface.co/collections/allenai/olmoearth) foundation model family (Allen AI), all 7 variants (`nano`/`tiny`/`base`/`large` v1, `nano_v1_1`/`tiny_v1_1`/`base_v1_1` v1.1; dims 128/192/768/1024), encoding 12 Sentinel-2 L2A bands with the FlexiViT encoder. Both `pooled` and `grid` outputs are supported; requires `pip install rs-embed[olmoearth]`.
- **Multi-frame temporal mode for Prithvi (`temporal_mode="multi"`).** A new `temporal_mode` key (default `"auto"`) feeds Prithvi-EO 2.0 a real multi-timestep HLS series instead of always collapsing the window into a single median composite; the frame count is derived from the window (`T = clamp(window_days // 28, 1, max_frames)`, default `max_frames=4`) and output shapes are unchanged. Configurable via `model_config` or `RS_EMBED_PRITHVI_*` env vars.
- **`gse` now automatically tiles large spatial requests.** When a `BBox`'s estimated pixel footprint exceeds the GEE `sampleRectangle` limit, the region is split into a sub-BBox grid, each tile fetched and concatenated — removing the previous hard cap on request size while matching a single large fetch.
- **GEE fetch statistics reporting in `export_batch`.** With `show_progress=True`, a thread-safe `[gee_fetch]` summary line (planned / completed / failed / cache-hit counts) is printed to stderr after each prefetch chunk, giving visibility into GEE quota use and cache reuse.
- **Orientation regression test for `_fetch_all_bands_impl`** that locks in GEE's north-up row pass-through so any accidental flip is caught immediately.

### Changed

- **`olmoearth` `temporal_mode` now defaults to `"auto"` (was `"single"`).** `auto` picks `single` for ≤~1-month windows and `multi` otherwise, so multi-month/-year ranges sample temporally by default. **Cost note:** any range >~1 month now fetches up to 12 composites (≈12× single mode) — pass `temporal_mode="single"` to force the cheaper path.
- **`galileo` now samples a window-adaptive frame count instead of a fixed 8, with a `temporal_mode` knob (default `"auto"`).** Frames track Galileo's ~monthly cadence (≤12); windows beyond 12 months are equal-divided into 12 frames covering the whole range and flagged out-of-distribution. Sub-month windows are now cheaper (1 frame instead of 8); `temporal_mode`/`n_frames` are settable via `model_config` or `RS_EMBED_GALILEO_*`.
- **Image-level ViT patch-token grid models tile by default with a seam warning.** For `scalemae`, `satmae`, and `satmaepp`, `input_prep=None`/`"auto"` resolves to `"tile"` (the package-wide default, same as `remoteclip`), so these models tile like every other model. On `grid` output the default/auto path and explicit `"tile"` emit a `UserWarning` that the tiled patch-token mosaic can show stitching seams — inherent to mosaicking independent token grids, not a per-model bug — and point to `input_prep="resize"` for a seamless (downsampled) grid; explicit `"resize"` stays silent. `get_embedding` and `export_batch` resolve this identically per model. Metadata records `input_prep.model_policy="tile_default_for_image_level_vit_patch_grid"` (or `"explicit_tile_for_image_level_vit_patch_grid"`) and `resolved_mode="tile"`.
- **Normalization responsibility moved entirely to embedders.** `NormalizationSpec` was removed from `ModelInputSpec`/`rs_embed.core.specs` and `apply_normalization()` from `providers.fetch`; all fetch helpers now return raw provider values (S2 DN, S1 linear, etc.) and each embedder normalizes inside `get_embedding()`, removing a misleading never-applied contract and a normalize→denormalize round-trip.
- **`ProviderBase.fetch_array_chw` contract tightened to north-up.** Implementations must return north-up CHW float32 (or apply `_flip_sample_tile_y` internally); `GEEProvider.fetch_array_chw` now does this itself rather than relying on the caller.
- **Internal refactor: provider/embedder layer separation.** The `embedders/runtime_utils.py` grab-bag was removed and split by responsibility into `providers/resolution.py`, `providers/fetch.py`, `tools/runtime.py`, and `tools/normalization.py`, fixing an inverted `tools/`→`embedders/` dependency.
- **Internal refactor: embedder utilities made self-contained.** `embedders/_vit_mae_utils.py`, `image_utils.py`, and `token_utils.py` were removed and their helpers inlined into each embedder; `meta_utils.py`→`meta.py` and `config_utils.py`→`config.py` were renamed (no functional change).

### Fixed

- **Tiled grid embeddings showed a flat "dead band" along a short ROI edge** (most visible on `prithvi` over wide/flat ROIs). The stitcher cropped against the unpadded extent and computed boundary patches over zero-padding; fixes crop against the padded extent, pad edge tiles with edge replication, and let resize-capable models skip padding entirely (short axis fed at native size and resized).
- **`export_batch` resolved `input_prep` globally instead of per-model, so a point's embedding could differ from `get_embedding`.** The export pipeline now uses `resolve_model_aware_input_prep` so the same point yields identical embeddings via `get_embedding` and `export_batch` (verified across all models × input_prep × output modes).
- **`remoteclip` produced black-image (garbage) embeddings on every `export_batch` path** due to a double `/10000` normalization in the `input_chw` branches; raw values are now passed through so export embeddings match `get_embedding`, and the provider `value_range` was corrected to `(0, 10000)`.
- **`satvision` export batch re-normalized already-unit-scaled inputs under `norm="raw"`.** `get_embeddings_batch_from_inputs` now honors the `already_unit_scaled` provenance (via a threaded `fetch_metas` argument) and forces `unit` per item, matching the single path; other models are unaffected.
- **Multi-temporal models no longer silently hide how many distinct frames a window actually had.** All five temporal adapters (`olmoearth`, `galileo`, `prithvi`, `anysat`, `agrifm`) now record frame-diversity metadata (`n_distinct_frames`/`n_backfilled_frames`, or `n_bins`/`dropped_bins` for `olmoearth`) and emit a `UserWarning` when real frames are sparser than requested — across the single, tiled, batch, and `export_batch` paths.
- **`prithvi` no longer silently degrades to a single composite on the `tile`/`auto`/`export_batch` paths.** It now overrides `fetch_input` to prefetch the same window-adaptive `[T,6,H,W]` raw-DN series as the direct path (single-frame windows defer to the generic composite fetch), so `num_frames` is identical across all APIs.
- **`olmoearth` `temporal_mode="multi"` silently dropped everything beyond the first ~12 months.** Windows over ~1 year are now equal-divided into 12 frames covering the whole period, surfaced via `temporal_sampling="equal_divided"` and a `UserWarning`; ≤12-month windows keep the in-distribution 30-day cadence.
- **`remoteclip` `pooled` output was inconsistent between single and batch/tiled paths** (768-d raw token mean vs 512-d projected CLIP). The single path now also uses `encode_image`, so `pooled` is the canonical 512-d CLIP embedding everywhere, and `grid` extraction was unified across paths.
- **Combined `export_batch` manifest lost all but the first model entry.** `combined_write_checkpoint` returned a copy instead of the caller's mutated dict, dropping every model after the first; it now merges the writer's path keys back into the original manifest. Per-item exports were unaffected.
- **`tessera` no longer re-fetches every tile twice per point.** `_mosaic_and_crop_strict_roi` invoked its `tiles_rows_factory` twice (re-downloading on a cold cache); the factory is now materialized once and shared, roughly halving first-chunk I/O at the cost of holding the point's tile blocks in memory simultaneously.
- **CLI `ModuleNotFoundError` on import.** `rs_embed.cli` imported from the non-existent `rs_embed.export`/`rs_embed.inspect`; imports now point to `rs_embed.api` and the `export-npz` call site matches `export_batch`'s current signature.
- **GEE tile orientation made explicit and self-contained.** `GEEProvider.fetch_array_chw` now applies `_flip_sample_tile_y` internally for a documented north-up contract, instead of leaving the flip to the caller (which double-flipped any subclass returning north-up data).
- **`export_batch` no longer raises `ModelError` when a `TemporalSpec.range` is passed with precomputed models (`gse`, `copernicus`).** Both now extract the start year and warn (`copernicus` still raises for unsupported years); the `assert_supported` check was also tightened to an exact `mode_hint == "year"` match, fixing a latent bug that wrongly matched `tessera`'s `"year_or_range"`.
- **`gse` batch inference now runs concurrently on CPU in `export_batch`** instead of only when a GPU was detected, benefiting precomputed models that do their own remote IO.
- **`get_embedding`/`get_embeddings_batch` no longer raise on `input_prep='tile'` with `model='gse'`, and `export_batch` no longer ignores `input_prep` for GSE.** All now emit a `UserWarning` clarifying that GSE manages its own tiling.
- **`input_prep='tile'` no longer disables `get_embeddings_batch` for precomputed models in `export_batch`.** The `allow_nonresize` guard now applies only to prefetched-input batch APIs; no-input batch APIs are unaffected.

## [0.1.3] — 2026-04-12

### Added

- `wildsat` now supports `variant` selection via `model_config={"variant": "..."}` (or the `variant=` keyword). Available variants are `vitb16` (default), `resnet50`, and `swint`, each backed by its own ImageNet-pretrained checkpoint that auto-downloads from Google Drive. The previous `vitl16` arch has been removed as no upstream checkpoint exists. Users who need other pre-training initializations (CLIP, Prithvi, SatCLIP, etc.) can still point `RS_EMBED_WILDSAT_CKPT` at a local checkpoint. The `RS_EMBED_WILDSAT_ARCH` environment variable is still respected as a fallback but is now overridden by `variant` when both are set.

### Changed

- `GEEProvider` initialization now prefers an explicit Google Cloud project when one is provided via `project=...`, `EE_PROJECT`, or `GOOGLE_CLOUD_PROJECT`, but no longer hard-requires rs-embed callers to pass one explicitly. When no project is supplied, rs-embed now lets `ee.Initialize()` and `geemap.ee_initialize()` resolve Earth Engine's configured default project first, while still surfacing a clear error message when authentication is missing or no usable Cloud/quota project is configured.
- `thor` now defaults `RS_EMBED_THOR_PATCH_SIZE` to `8` instead of `16`, increasing the default token-grid density while keeping `RS_EMBED_THOR_IMG=288`. THOR also now defaults to a bounded `native_snap` preprocessing policy for ordinary resize-mode inputs: near-square inputs can keep a snapped native side when they stay within configured side/token limits, while tiled inputs still force fixed per-tile resize so stitched grids remain geometrically stable. The THOR model page now documents the patch-size/image-size coupling, native-snap limits, and concrete environment-variable examples for common tuning patterns.

### Fixed

- `anysat` and `satmaepp_s2_10b` now validate the `variant` keyword in a single place instead of accepting a wider alias set in `_normalize_*_variant` and then raising a second `ModelError` deeper in the runtime resolver. Previously, passing `variant="tiny"` / `"small"` to `anysat` or `variant="base"` to `satmaepp_s2_10b` would first be silently normalized and then rejected with a confusingly-located "currently exposes only variant=..." error. The normalize helpers now only accept the variants that actually map to a wired checkpoint (`anysat` → `base`, `satmaepp_s2_10b` → `large`), raise an immediate and descriptive `ModelError` for anything else, and the duplicate runtime guards have been removed. The `satmaepp_s2_10b` env-var path (`RS_EMBED_SATMAEPP_S2_MODEL_FN`) now also raises a clear error for unknown `model_fn` values instead of silently producing a `variant=None` runtime config. The `describe()` output for both adapters already advertises `choices: ["base"]` / `choices: ["large"]`, so the schema side was already correct; this fix just makes the validation code match it.
- `BBox.validate()` now enforces geographic bounds: longitudes must be in `[-180, 180]` and latitudes in `[-90, 90]`. Out-of-range coordinates previously passed validation and caused confusing downstream errors from the GEE provider.
- `describe_model()` now returns a cached copy of the embedder's `describe()` output instead of instantiating a new embedder class on every call. The cache is keyed by canonical model name and is cleared by `reset_runtime()`. The returned dict is always a shallow copy so callers cannot mutate the cached entry.
- `fetch_api_side_inputs()` now wraps per-spatial fetch errors in a `ModelError` that includes the spatial index and the original exception, making it easier to pinpoint which location caused a failure when running `get_embeddings_batch()` with `input_prep="tile"` or `"auto"`.
- `run_embedding_request()` now uses `strict=True` in the `zip` of spatials and prefetched inputs. A length mismatch between the two lists now raises immediately instead of silently truncating the result.
- Loading checkpoint arrays during combined-export resume now emits a `warnings.warn` instead of silently swallowing the exception. Users will see a clear message indicating that array loading failed and that all inputs will be re-fetched.
- `_write_per_item_chunk` no longer accesses the private `_shutdown` attribute of `ThreadPoolExecutor` to guard against double-shutdown. The outer `finally` block now relies on the documented idempotency of `ThreadPoolExecutor.shutdown()` instead, removing a fragile dependency on CPython internals that could break on future Python versions.
- `sensor_key()` no longer applies `int()` truncation to `scale_m` and `cloudy_pct` when building the embedder instance cache key. Previously, float values such as `10.1` and `10.9` were both mapped to `10`, allowing two sensors with different resolutions to share a cached embedder instance incorrectly. The raw field values are now used directly.
- `_run_per_item` now closes all progress bars (main and per-model) inside the `finally` block of the chunk-pipeline loop. Previously the cleanup ran after the `try/finally`, so an unhandled exception (e.g. `continue_on_error=False`) would leave progress bars open and leak display resources in notebook environments.

## [0.1.2] - 2026-04-03

This release rolls up upstream-alignment work and correctness fixes that may change default embedding behavior for some model adapters compared with `0.1.1`. Users who need strict reproducibility across versions should review the model-specific changes below and pin explicit options where needed.

### Changed

- Standardised NumPy docstrings across all public functions and classes in `export.py`, `inspect.py`, `writers.py`, and the `tools/` and `providers/` layers. No behaviour changes.

### Added

- Versioned documentation with a version selector powered by `mike` and MkDocs Material. Each release tag deploys a pinned version; pushes to `main` update a `dev` alias. The `mike`, `mkdocs-material`, and `pymdown-extensions` packages are now included in the `[dev]` optional group.

- `load_export(path)` reader API that loads any export produced by `export_batch(...)` — both combined (single file) and per-item (directory) layouts — and returns a structured `ExportResult`. Failed points are NaN-filled rather than dropped, partial model runs are surfaced via `status="partial"`, and `ExportResult.embedding(model)` provides a typed shortcut to the embedding array.

### Changed

- `anysat` now defaults `grid` output to native `dense` features while keeping pooled output on patch-grid pooling by default, with new AnySat-specific switches for `grid_feature_mode` (`dense`/`patch`) and `pooled_source` (`patch`/`tile`).
- `galileo` now aligns more closely with the upstream NASA Harvest runtime: `grid` output prefers Galileo's own patch-level token averaging path, automatic NDVI derivation has been removed, and the default normalization mode is now `none` with an `official_stats` option for upstream pretraining statistics.
- `satvision_toa` now uses the vendored official SatVision runtime as its only model path and narrows provider-side preprocessing to the default MODIS proxy route (`MOD09GA` reflectance + `MOD21A1D` thermal proxy). Custom collections are no longer treated as implicit GEE fallbacks; callers should pass calibrated `input_chw` directly for non-default inputs.

### Fixed

- `device="auto"` now correctly selects MPS on Apple Silicon instead of silently falling back to CPU. Follows the PyTorch-recommended priority (`cuda > mps > cpu`), giving an approximately 4x speedup on Apple M-series hardware for all API calls that use the default device.
- `galileo` month overrides now use the official zero-based month indexing expected by Galileo embeddings, fixing the previous one-month offset in `RS_EMBED_GALILEO_MONTH`.
- `satvision_toa` `grid` output now consistently extracts spatial features from the official-style SwinV2 path instead of misinterpreting pooled vectors as token grids, and the tightened fetch path records explicit proxy provenance in metadata.
- `scalemae` now follows the official feature-extraction path more closely: the adapter unwraps common `rshf` wrappers to call the nested ScaleMAE backbone's `forward_features(...)` instead of falling back to wrapper `forward()`, uses ImageNet eval preprocessing (`Resize(short side) + CenterCrop + Normalize`) with effective post-preprocess `input_res_m`, and fixes the declarative input metadata to reflect raw Sentinel-2 SR inputs plus adapter-managed preprocessing.
- `satmaepp` and `satmaepp_s2_10b` now align more closely with the official SatMAE++ preprocessing paths. The RGB adapter now defaults to `rgb` channel order for the published fMoW-RGB checkpoint and no longer pre-resizes provider/input overrides before the official eval transform, while the Sentinel-2 10-band adapter no longer sanitizes inputs with adapter-side `clip` / `nan_to_num` before the source-style `SentinelNormalize -> ToTensor -> Resize(short side) -> CenterCrop` pipeline.

## [0.1.1] - 2026-04-01

### Added

- Automated pull request changelog enforcement with a `skip-changelog` escape hatch for docs, tests, CI, and other internal-only changes.
- Tag-driven GitHub Release publishing that uses the matching `CHANGELOG.md` section as the release notes.
- Trusted Publishing release automation for PyPI and TestPyPI, including a manual TestPyPI dry run and install smoke test.

### Changed

- The contribution and release workflow now treats `CHANGELOG.md` as the canonical source for user-visible release notes.
- The tag-triggered release flow now validates `src/rs_embed/_version.py`, publishes to PyPI, and only then creates the GitHub Release.
- The tag-triggered release flow now validates the matching `CHANGELOG.md` entry before publishing to PyPI, so a missing release-notes section fails early instead of after package upload.
- The base package installation now includes the Copernicus GeoTIFF runtime (`tifffile` and `imagecodecs`) instead of requiring a separate extra.
- The public docs now default to `pip install rs-embed` for published releases while keeping editable installs documented for repository development.

### Deprecated

### Removed

### Fixed

- The TestPyPI smoke test now verifies package importability and the `rs-embed` CLI entry point, not just installability and version metadata.

## [0.1.0] - 2026-03-31

### Added

- Initial public alpha release of `rs-embed`.
- Unified ROI to embedding API centered on `get_embedding(...)`, `get_embeddings_batch(...)`, `export_batch(...)`, and `inspect_provider_patch(...)`.
- Support for precomputed embedding products including `tessera`, `gse`, and `copernicus`.
- Support for on-the-fly model adapters including `satmae`, `satmaepp`, `satmaepp_s2_10b`, `prithvi`, `scalemae`, `remoteclip`, `dofa`, `satvision`, `anysat`, `galileo`, `wildsat`, `fomo`, `terramind`, `terrafm`, `thor`, and `agrifm`.
- Documentation site covering quickstart, model behavior, API contracts, and extension guidance.

[Unreleased]: https://github.com/cybergis/rs-embed/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/cybergis/rs-embed/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.3
[0.1.2]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.2
[0.1.1]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.1
[0.1.0]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.0

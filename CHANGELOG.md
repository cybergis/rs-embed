# Changelog

All notable changes to `rs-embed` will be documented in this file.

The project keeps this changelog as the canonical release record. GitHub Releases should summarize the same versioned changes rather than introducing a second source of truth.

The format is based on Keep a Changelog, and the project follows Semantic Versioning with extra care around model and embedding semantics.

## [Unreleased]

### Added

- **GEE fetch statistics reporting in `export_batch`.** When `show_progress=True`, a `[gee_fetch]` summary line is now printed to stderr after each prefetch chunk completes, reporting total planned fetches, completed, failed, cache hits, and the most recently processed point/sensor. This gives users visibility into GEE quota consumption, cache reuse, and whether runtime is dominated by fetching vs. inference. No output is emitted when `show_progress=False` or when no GEE provider is involved (e.g. precomputed models). The underlying `FetchStats` class in `tools/progress.py` is thread-safe and accumulates counts cumulatively across chunks.

### Fixed

- **CLI `ModuleNotFoundError` on import.** `rs_embed.cli` was importing from `rs_embed.export` and `rs_embed.inspect`, two modules that do not exist in the current package layout. The imports now point directly to `rs_embed.api` (`export_batch`, `inspect_gee_patch`). The `export-npz` subcommand call site has been updated to match `export_batch`'s current signature: a single spatial argument is wrapped in `spatials=[...]`, the output path becomes `ExportTarget.combined(args.out)`, and the flat boolean flags (`save_inputs`, `save_manifest`, etc.) are grouped into an `ExportConfig` object. The stub injection in `tests/test_cli_parsers.py` that was masking the broken imports has been removed and the integration test updated to patch `cli.export_batch` instead of `cli.export_npz`.

- **GEE tile orientation made explicit and self-contained.** `GEEProvider.fetch_array_chw` now applies `_flip_sample_tile_y` internally before returning, giving the method a documented north-up contract. Previously the flip was the caller's responsibility inside `_fetch_provider_array_chw_with_bbox_fallback`, which created a leaky abstraction: any `ProviderBase` subclass overriding `fetch_array_chw` had to know to return south-up raw data or risk being flipped twice. The caller no longer applies a second flip. The `_sample_image_bands_raw_chw` docstring now documents that GEE's `reproject(crs=..., scale=...)` **without** `.clip()` naturally returns north-up rows, and warns explicitly that adding `.clip()` would change the row ordering to south-up and break the multiframe fetch path. `_flip_sample_tile_y` docstring updated to describe exactly which call pattern requires it and which does not.

### Changed

- **`ProviderBase.fetch_array_chw` contract tightened to north-up.** The method is now required to return north-up CHW float32. Custom provider implementations must either return north-up data directly or apply `_flip_sample_tile_y` internally. Test fake providers in `test_api_helpers.py` have been updated accordingly (north-up row generation, renamed test functions).

### Added

- **Orientation regression test for `_fetch_all_bands_impl`.** `test_gee_provider.py` now includes `test_fetch_all_bands_impl_passes_through_gee_row_order`, which injects a mock GEE `sampleRectangle` response with a known row order and asserts the function output matches exactly (no flip applied). This locks in the pass-through behaviour so that any accidental flip addition is caught immediately, and documents that the north-up guarantee for `fetch_collection_patch_all_bands_chw` comes from GEE's empirical behaviour for the `reproject(crs=..., scale=...) + clip` call pattern rather than from Python-level normalization.

### Changed

- **Normalization responsibility moved entirely to embedders.** `NormalizationSpec` has been removed from `ModelInputSpec` and from `rs_embed.core.specs`. The `apply_normalization()` helper in `rs_embed.providers.fetch` has been removed. `fetch_input()` and all fetch helpers (`fetch_collection_patch_chw`, `fetch_s2_rgb_chw`, `fetch_s2_multiframe_raw_tchw`, `fetch_s1_vvvh_raw_chw`) now consistently return raw provider values (S2 DN in [0, 10000], S1 linear float, etc.); each embedder applies its own normalization inside `get_embedding()`. This eliminates a misleading contract where `ModelInputSpec.normalization` was declared but never automatically applied, and removes a normalize→denormalize round-trip that existed in some batch paths (remoteclip, wildsat).

- **Internal refactor: provider and embedder layer separation.** The `embedders/runtime_utils.py` grab-bag module has been removed and its contents redistributed by responsibility:
  - Provider selection and lifecycle management → `providers/resolution.py` (`default_provider_backend_name`, `resolve_provider_backend_name`, `is_provider_backend`, `get_cached_provider`, `create_provider_for_backend`)
  - Provider fetch helpers and satellite normalization → `providers/fetch.py` (`fetch_sensor_patch_chw`, `fetch_collection_patch_chw`, `fetch_s2_rgb_chw`, `fetch_s1_vvvh_raw_chw`, `fetch_s2_multiframe_raw_tchw`, `normalize_s1_vvvh_chw`, `apply_normalization`, etc.)
  - Device resolution and embedder instance loading → `tools/runtime.py` (`resolve_device_auto_torch`, `load_cached_with_device`)
  - Input coercion → `tools/normalization.py` (`coerce_input_to_tchw`, `coerce_single_input_chw`)
  - This fixes an inverted dependency where `tools/` was importing from `embedders/`.
- **Internal refactor: embedder utility modules.** `embedders/_vit_mae_utils.py`, `embedders/image_utils.py`, and `embedders/token_utils.py` have been removed. Image preprocessing (`ensure_torch`, `resize_rgb_u8`, `fetch_s2_rgb_u8_from_provider`, CLIP norm, etc.) and ViT token operations (`pool_from_tokens`, `tokens_to_grid_dhw`) are now defined locally in each embedder that needs them. This makes each embedder file self-contained and eliminates hidden shared state between model implementations.
- **Internal rename:** `embedders/meta_utils.py` → `embedders/meta.py`, `embedders/config_utils.py` → `embedders/config.py`. No functional change; the `_utils` suffix was removed to match the naming convention of other modules in the package.

### Added

- `gse` now automatically tiles large spatial requests. If the estimated pixel footprint of a `BBox` at the requested `scale_m` exceeds the GEE `sampleRectangle` limit (default 512×512, overridable via `RS_EMBED_GSE_MAX_PIXELS`), the region is split into a sub-BBox grid, each tile is fetched from `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, and the embedding arrays are concatenated before pooling or grid output is applied. This removes the previous hard cap on request size and makes the result equivalent to a single large fetch.

### Fixed

- `export_batch` no longer raises `ModelError` when a `TemporalSpec.range` is passed alongside precomputed models (`gse`, `copernicus`) that only index by year. Both embedders now extract the start year from the range, emit a `UserWarning`, and continue. `copernicus` still raises if the extracted year is not in `SUPPORTED_YEARS`. The early-exit validation in `assert_supported` was also tightened from `"year" in mode_hint` to an exact `mode_hint == "year"` check, which fixes a latent bug where `tessera`'s `"year_or_range"` hint was incorrectly matched and would have blocked range inputs before the embedder's own graceful handling could run.
- `gse` batch inference now runs concurrently on CPU in `export_batch`. Previously, chunk-level batching was only enabled when a GPU was detected; precomputed models that do their own remote IO benefit from batching regardless of device.
- `get_embedding` and `get_embeddings_batch` no longer raise `ModelError` when `input_prep='tile'` is passed with `model='gse'`. `export_batch` no longer silently ignores `input_prep` for GSE. Both APIs now emit a `UserWarning` when a non-default `input_prep` is passed alongside `model='gse'`, clarifying that GSE manages its own tiling.
- `input_prep='tile'` no longer disables `get_embeddings_batch` for precomputed models in `export_batch`. The `allow_nonresize` guard in `_evaluate_batch_capability` now applies only to prefetched-input batch APIs; no-input batch APIs (which never receive an `input_chw` to tile) are unaffected.

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

[Unreleased]: https://github.com/cybergis/rs-embed/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.2
[0.1.1]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.1
[0.1.0]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.0

"""
Stateless Utilities and Helper Functions.

A collection of pure functions for generic tasks.  Code here must be
stateless (inputs -> outputs) and reusable across different pipelines.

Categories
----------
- **Tiling** (``tiling``) — geometric math for cutting images into patches.
- **Projection** (``projection``) — the common EPSG:3857 grid, resampling onto it, UTM-zone checks.
- **Inspection** (``inspection``) — histogram calculation and data-quality checks.
- **Serialization** (``serialization``) — JSON / NPZ saving and hashing.
- **Temporal** (``temporal``) — date parsing and time-range helpers.
- **Normalization** (``normalization``) — value scaling and string normalisation.
- **Export Requests** (``export_requests``) — export API argument normalization and request assembly.
- **Checkpoints** (``checkpoint_utils``) — resume-friendly progress tracking.
- **Manifests** (``manifest``) — metadata bookkeeping for export runs.
- **Output** (``output``) — path and directory helpers for saved artefacts.
- **Progress** (``progress``) — progress-bar wrappers.
- **Runtime** (``runtime``) — environment detection (device, memory, threads).
- **Model Defaults** (``model_defaults``) — canonical per-model fallback configs.
"""

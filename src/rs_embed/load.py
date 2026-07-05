"""Utilities for loading exports produced by :func:`rs_embed.export_batch`.

Two export layouts are supported:

* **combined** — a single ``.npz``/``.nc`` file (plus a ``.json`` manifest)
  containing all spatial points and models in one array file.
* **per_item** — a directory of ``p<index>.npz``/``.nc`` and ``.json`` files,
  one pair per spatial point.

The public entry point is :func:`load_export`, which auto-detects the layout
from the path argument and returns an :class:`ExportResult`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .tools.serialization import sanitize_key

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

_EMBEDDING_KEY_SINGLE = "embedding__{model}"
_EMBEDDING_KEY_BATCH = "embeddings__{model}"
_INPUT_KEY_SINGLE = "input_chw__{model}"
_INPUT_KEY_BATCH = "inputs_bchw__{model}"


@dataclass
class ModelResult:
    """Loaded embeddings and metadata for one model across all spatial points.

    Attributes
    ----------
    name : str
        Canonical model identifier (e.g. ``"remoteclip"``).
    status : str
        Aggregate status: ``"ok"``, ``"partial"``, or ``"failed"``.
    embeddings : np.ndarray or None
        Float32 array of shape ``(n_items, dim)`` for pooled output, or
        ``(n_items, C, H, W)`` for grid output.  ``None`` when the model
        failed for every point.  Individual failed points within a
        partially-succeeded run are filled with ``NaN``.
    inputs : np.ndarray or None
        Raw input patches if they were saved during export
        (``ExportConfig(save_inputs=True)``).  Shape
        ``(n_items, C, H, W)``.  ``None`` when inputs were not saved.
    meta : list[dict]
        Per-point embedding metadata dicts (one entry per spatial point).
        Empty dicts are used for failed points.
    error : str or None
        Error string if *every* point for this model failed, else ``None``.
    """

    name: str
    status: str
    embeddings: np.ndarray | None
    inputs: np.ndarray | None
    meta: list[dict[str, Any]]
    error: str | None = None


@dataclass
class ExportResult:
    """Loaded export from :func:`rs_embed.export_batch`.

    Attributes
    ----------
    layout : str
        ``"combined"`` or ``"per_item"``.
    spatials : list[dict]
        Spatial specs (one per point), as JSON-serializable dicts matching
        the original :class:`~rs_embed.BBox` / :class:`~rs_embed.PointBuffer`
        fields.
    temporal : dict or None
        Temporal spec used during export, or ``None`` if not specified.
    n_items : int
        Number of spatial points.
    status : str
        Overall export status: ``"ok"``, ``"partial"``, or ``"failed"``.
    models : dict[str, ModelResult]
        Loaded results keyed by model name.
    manifest : dict
        Raw manifest dict for advanced inspection.
    """

    layout: str
    spatials: list[dict[str, Any]]
    temporal: dict[str, Any] | None
    n_items: int
    status: str
    models: dict[str, ModelResult]
    manifest: dict[str, Any]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def embedding(self, model: str) -> np.ndarray:
        """Return the embedding array for *model*.

        Parameters
        ----------
        model : str
            Model name as it appears in :attr:`models`.

        Returns
        -------
        np.ndarray
            Shape ``(n_items, dim)`` for pooled, ``(n_items, C, H, W)``
            for grid.

        Raises
        ------
        KeyError
            If *model* was not part of this export.
        ValueError
            If *model* failed for every point (``status == "failed"``).
        """
        if model not in self.models:
            available = sorted(self.models)
            raise KeyError(f"Model {model!r} not found in export. Available models: {available}")
        result = self.models[model]
        if result.embeddings is None:
            raise ValueError(
                f"Model {model!r} has no embeddings (status={result.status!r}). "
                f"Error: {result.error}"
            )
        return result.embeddings

    @property
    def ok_models(self) -> list[str]:
        """Model names whose status is ``"ok"``."""
        return [name for name, r in self.models.items() if r.status == "ok"]

    @property
    def failed_models(self) -> list[str]:
        """Model names whose status is ``"failed"``."""
        return [name for name, r in self.models.items() if r.status == "failed"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_export(path: str | os.PathLike[str]) -> ExportResult:
    """Load an export produced by :func:`rs_embed.export_batch`.

    Auto-detects the export layout:

    * Pass a **file path** (``.npz``, ``.nc``, or ``.json``) to load a
      **combined** export (all spatial points in one file).
    * Pass a **directory path** to load a **per-item** export (one file
      pair per spatial point).

    Parameters
    ----------
    path : str or path-like
        Path to the export file or directory.

    Returns
    -------
    ExportResult
        Loaded embeddings, inputs, metadata, and spatial information.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the path exists but cannot be interpreted as an rs-embed export.

    Examples
    --------
    Load a combined export:

    >>> result = load_export("embeddings.npz")
    >>> result.ok_models
    ['prithvi', 'remoteclip']
    >>> emb = result.embedding("prithvi")   # shape (n_items, dim)

    Load a per-item export directory:

    >>> result = load_export("embeddings/")
    >>> result.n_items
    50
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Export path does not exist: {path!r}")
    if os.path.isdir(path):
        return _load_per_item(path)
    return _load_combined(path)


# ---------------------------------------------------------------------------
# Combined layout
# ---------------------------------------------------------------------------


def _resolve_combined_paths(path: str) -> tuple[str, str, str]:
    """Return ``(arrays_path, json_path, fmt)`` for a combined export path.

    Accepts a ``.npz``, ``.nc``, or ``.json`` path and finds the paired files.
    """
    base, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".json":
        # Find the paired array file
        for candidate_ext, fmt in ((".npz", "npz"), (".nc", "netcdf")):
            arrays_path = base + candidate_ext
            if os.path.exists(arrays_path):
                return arrays_path, path, fmt
        raise FileNotFoundError(
            f"Found manifest {path!r} but no paired .npz or .nc file at {base!r}."
        )

    if ext == ".npz":
        fmt = "npz"
    elif ext in (".nc", ".netcdf"):
        fmt = "netcdf"
    else:
        raise ValueError(
            f"Unrecognised file extension {ext!r}. "
            "Expected .npz, .nc, or .json for a combined export."
        )

    json_path = base + ".json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Array file {path!r} found but no paired manifest at {json_path!r}."
        )
    return path, json_path, fmt


def _load_arrays(arrays_path: str, fmt: str) -> dict[str, np.ndarray]:
    """Load all arrays from an npz or netcdf file."""
    if fmt == "npz":
        with np.load(arrays_path, allow_pickle=False) as payload:
            return {str(k): np.asarray(payload[k]) for k in payload.files}
    if fmt == "netcdf":
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "xarray is required to load NetCDF exports. Install with: pip install xarray"
            ) from exc
        ds = xr.open_dataset(arrays_path)
        try:
            return {str(k): np.asarray(ds[k].values) for k in ds.data_vars}
        finally:
            ds.close()
    raise ValueError(f"Unknown format {fmt!r}.")  # pragma: no cover


def _load_json(json_path: str) -> dict[str, Any]:
    """Load and validate a JSON manifest."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to load manifest {json_path!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Manifest {json_path!r} is not a JSON object.")
    return data


def _gather_ragged_rows(
    ref: dict[str, Any],
    arrays: dict[str, np.ndarray],
    n_items: int,
) -> np.ndarray | None:
    """Assemble a ragged combined reference (``npz_keys``/``indices``) into a
    dense ``(n_items, ...)`` array with NaN rows at missing points.

    The exporter writes this format on partial success (per-point keys like
    ``embedding__<model>__00012``). When ``indices`` is absent, the point
    index is parsed from the key's trailing ``__<index>`` suffix. Genuinely
    unstackable per-point shapes return ``None`` — there is no dense
    representation to promise.
    """
    keys = ref.get("npz_keys")
    if not (isinstance(keys, list) and keys):
        return None
    indices = ref.get("indices")
    per_point: list[np.ndarray | None] = [None] * n_items
    for j, k in enumerate(keys):
        if not (isinstance(k, str) and k in arrays):
            continue
        if isinstance(indices, list) and j < len(indices):
            idx = int(indices[j])
        else:
            tail = k.rsplit("__", 1)[-1]
            idx = int(tail) if tail.isdigit() else j
        if 0 <= idx < n_items:
            per_point[idx] = np.asarray(arrays[k], dtype=np.float32)
    try:
        return _stack_per_item_embeddings(per_point)
    except ValueError:
        return None


def _combined_ref_array(
    ref: Any,
    arrays: dict[str, np.ndarray],
    *,
    default_key: str,
    n_items: int,
) -> np.ndarray | None:
    """Resolve a combined manifest array reference: dense key or ragged keys."""
    ref = ref if isinstance(ref, dict) else {}
    key = ref.get("npz_key") or default_key
    if key in arrays:
        return np.asarray(arrays[key], dtype=np.float32)
    return _gather_ragged_rows(ref, arrays, n_items)


def _build_combined_model_result(
    entry: dict[str, Any],
    arrays: dict[str, np.ndarray],
    n_items: int,
) -> ModelResult:
    """Build a ModelResult from one entry in a combined manifest's models list."""
    name = str(entry.get("model", ""))
    status = str(entry.get("status", "failed"))
    error = entry.get("error") or None

    key = sanitize_key(name)

    embeddings = _combined_ref_array(
        entry.get("embeddings"),
        arrays,
        default_key=_EMBEDDING_KEY_BATCH.format(model=key),
        n_items=n_items,
    )
    inputs = _combined_ref_array(
        entry.get("inputs"),
        arrays,
        default_key=_INPUT_KEY_BATCH.format(model=key),
        n_items=n_items,
    )

    # Per-point metadata list
    raw_metas = entry.get("metas") or []
    meta: list[dict[str, Any]] = []
    for i in range(n_items):
        m = raw_metas[i] if i < len(raw_metas) else {}
        meta.append(m if isinstance(m, dict) else {})

    return ModelResult(
        name=name,
        status=status,
        embeddings=embeddings,
        inputs=inputs,
        meta=meta,
        error=str(error) if error is not None else None,
    )


def _load_combined(path: str) -> ExportResult:
    arrays_path, json_path, fmt = _resolve_combined_paths(path)
    arrays = _load_arrays(arrays_path, fmt)
    manifest = _load_json(json_path)

    n_items = int(manifest.get("n_items", 0))
    spatials: list[dict[str, Any]] = list(manifest.get("spatials") or [])
    temporal = manifest.get("temporal") or None
    status = str(manifest.get("status", "ok"))

    model_entries: list[dict[str, Any]] = list(manifest.get("models") or [])
    models: dict[str, ModelResult] = {}
    for entry in model_entries:
        result = _build_combined_model_result(entry, arrays, n_items)
        models[result.name] = result

    return ExportResult(
        layout="combined",
        spatials=spatials,
        temporal=temporal,
        n_items=n_items,
        status=status,
        models=models,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Per-item layout
# ---------------------------------------------------------------------------


def _find_per_item_files(directory: str) -> list[tuple[str, str, str, str]]:
    """Return sorted ``(arrays_path, json_path, fmt, base)`` tuples for a per-item dir.

    Any ``<base>.(npz|nc)`` with a paired ``<base>.json`` manifest is a point
    file — custom ``target.names`` load the same as the default ``p<index>``
    names. The manifest's ``point_index`` decides row placement.
    """
    entries: list[tuple[str, str, str, str]] = []
    for fname in sorted(os.listdir(directory)):
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in (".npz", ".nc", ".netcdf"):
            continue
        json_path = os.path.join(directory, base + ".json")
        if not os.path.exists(json_path):
            continue
        fmt = "npz" if ext == ".npz" else "netcdf"
        entries.append((os.path.join(directory, fname), json_path, fmt, base))

    if not entries:
        raise ValueError(
            f"No per-item export files found in {directory!r}. "
            "Expected '<name>.npz' or '<name>.nc' files with paired "
            "'.json' manifests."
        )
    return entries


def _stack_per_item_embeddings(
    per_point: list[np.ndarray | None],
) -> np.ndarray | None:
    """Stack per-point embedding arrays into ``(n_items, ...)``.

    Failed points (``None``) are filled with NaN using the shape inferred
    from the first successful array.
    """
    sample = next((a for a in per_point if a is not None), None)
    if sample is None:
        return None
    fill = np.full(sample.shape, fill_value=np.nan, dtype=np.float32)
    rows = [a.astype(np.float32) if a is not None else fill for a in per_point]
    return np.stack(rows, axis=0)


def _per_item_point_index(manifest: dict[str, Any], base: str) -> int | None:
    """Row index for one per-item file, or ``None`` when the pair is not a
    point file.

    Every point manifest the exporter writes (build/resume/failure paths)
    carries ``point_index``; the default ``p<digits>`` filename convention is
    kept for legacy exports. Anything else — e.g. a combined ``run.npz`` +
    ``run.json`` dropped into the same directory — is not a point file and
    must not be ingested as one.
    """
    idx = manifest.get("point_index")
    if isinstance(idx, int) and idx >= 0:
        return idx
    if base.startswith("p") and base[1:].isdigit():
        return int(base[1:])
    return None


def _load_per_item(directory: str) -> ExportResult:
    files = _find_per_item_files(directory)

    # Row i of the result must correspond to the caller's spatials[i]: place
    # each file at its recorded point_index (failed points write no file under
    # continue_on_error, so dense renumbering would silently misalign rows).
    # Manifests load first so non-point pairs are skipped before their arrays
    # are read, and a duplicated index fails instead of overwriting a row.
    loaded: list[tuple[int, dict[str, np.ndarray], dict[str, Any]]] = []
    claimed_by: dict[int, str] = {}
    for arrays_path, json_path, fmt, base in files:
        manifest = _load_json(json_path)
        idx = _per_item_point_index(manifest, base)
        if idx is None:
            continue
        if idx in claimed_by:
            raise ValueError(
                f"Per-item files {claimed_by[idx]!r} and {os.path.basename(arrays_path)!r} "
                f"both claim point_index={idx}; refusing to overwrite a row. "
                "Remove the stale file or separate the exports."
            )
        claimed_by[idx] = os.path.basename(arrays_path)
        loaded.append((idx, _load_arrays(arrays_path, fmt), manifest))
    if not loaded:
        raise ValueError(
            f"No per-item point files found in {directory!r}. Point files carry "
            "a 'point_index' in their .json manifest (or use the default "
            "'p<index>' naming)."
        )
    loaded.sort(key=lambda t: t[0])
    n_items = max(idx for idx, _, _ in loaded) + 1

    spatials: list[dict[str, Any]] = [{} for _ in range(n_items)]
    temporal: dict[str, Any] | None = None
    overall_status: str | None = None

    # Collect per-point data keyed by model name
    per_model_embeddings: dict[str, list[np.ndarray | None]] = {}
    per_model_inputs: dict[str, list[np.ndarray | None]] = {}
    per_model_meta: dict[str, list[dict[str, Any]]] = {}
    per_model_status: dict[str, list[str]] = {}
    per_model_error: dict[str, str | None] = {}

    manifest_first: dict[str, Any] = {}

    for pos, (i, arrays, manifest) in enumerate(loaded):
        if pos == 0:
            manifest_first = manifest
            temporal = manifest.get("temporal") or None

        spatial = manifest.get("spatial")
        spatials[i] = spatial if isinstance(spatial, dict) else {}

        for entry in manifest.get("models") or []:
            name = str(entry.get("model", ""))
            if not name:
                continue

            key = sanitize_key(name)
            entry_status = str(entry.get("status", "failed"))

            # Embeddings
            emb_meta = entry.get("embedding") or {}
            emb_key = (
                emb_meta.get("npz_key") if isinstance(emb_meta, dict) else None
            ) or _EMBEDDING_KEY_SINGLE.format(model=key)
            emb_arr = arrays.get(emb_key)

            # Inputs
            inp_meta = entry.get("input") or {}
            inp_key = (
                inp_meta.get("npz_key") if isinstance(inp_meta, dict) else None
            ) or _INPUT_KEY_SINGLE.format(model=key)
            inp_arr = arrays.get(inp_key)

            per_model_embeddings.setdefault(name, [None] * n_items)
            per_model_inputs.setdefault(name, [None] * n_items)
            per_model_meta.setdefault(name, [{} for _ in range(n_items)])
            per_model_status.setdefault(name, [])
            per_model_error.setdefault(name, None)

            per_model_embeddings[name][i] = (
                np.asarray(emb_arr, dtype=np.float32) if emb_arr is not None else None
            )
            per_model_inputs[name][i] = (
                np.asarray(inp_arr, dtype=np.float32) if inp_arr is not None else None
            )
            raw_meta = entry.get("meta")
            per_model_meta[name][i] = raw_meta if isinstance(raw_meta, dict) else {}
            per_model_status[name].append(entry_status)

            if entry_status == "failed" and per_model_error[name] is None:
                per_model_error[name] = entry.get("error") or None

    # Build per-model results
    models: dict[str, ModelResult] = {}
    for name in per_model_embeddings:
        emb_arrays = per_model_embeddings[name]
        inp_arrays = per_model_inputs[name]
        statuses = per_model_status[name]

        embeddings = _stack_per_item_embeddings(emb_arrays)
        inputs = _stack_per_item_embeddings(inp_arrays)

        n_ok = sum(1 for s in statuses if s == "ok")
        n_failed = sum(1 for s in statuses if s == "failed")
        if n_failed == 0:
            agg_status = "ok"
        elif n_ok == 0:
            agg_status = "failed"
        else:
            agg_status = "partial"

        models[name] = ModelResult(
            name=name,
            status=agg_status,
            embeddings=embeddings if agg_status != "failed" else None,
            inputs=inputs if any(a is not None for a in inp_arrays) else None,
            meta=per_model_meta[name],
            error=str(per_model_error[name]) if agg_status == "failed" else None,
        )

    # Aggregate overall status
    all_statuses = [r.status for r in models.values()]
    n_ok_models = sum(1 for s in all_statuses if s == "ok")
    n_failed_models = sum(1 for s in all_statuses if s == "failed")
    if n_failed_models == 0:
        overall_status = "ok" if all_statuses else "ok"
    elif n_ok_models == 0:
        overall_status = "failed"
    else:
        overall_status = "partial"

    return ExportResult(
        layout="per_item",
        spatials=spatials,
        temporal=temporal,
        n_items=n_items,
        status=overall_status,
        models=models,
        manifest=manifest_first,
    )

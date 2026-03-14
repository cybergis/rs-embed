"""High-level export entrypoints.

`export_npz` is a convenience wrapper around `rs_embed.api.export_batch`.
"""

from __future__ import annotations

import os
from typing import Any

from .core.specs import InputPrepSpec, OutputSpec, SensorSpec, SpatialSpec, TemporalSpec


def export_npz(
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    models: list[str],
    out_path: str,
    backend: str = "auto",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: SensorSpec | None = None,
    per_model_sensors: dict[str, SensorSpec] | None = None,
    save_inputs: bool = True,
    save_embeddings: bool = True,
    save_manifest: bool = True,
    fail_on_bad_input: bool = False,
    infer_batch_size: int | None = None,
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    input_prep: InputPrepSpec | str | None = "resize",
) -> dict[str, Any]:
    """Export inputs + embeddings for one spatial query to a single `.npz`."""
    from .api import export_batch as _api_export_batch

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not out_path.endswith(".npz"):
        out_path = out_path + ".npz"

    return _api_export_batch(
        spatials=[spatial],
        temporal=temporal,
        models=models,
        out_path=out_path,
        backend=backend,
        device=device,
        output=output,
        sensor=sensor,
        per_model_sensors=per_model_sensors,
        format="npz",
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        save_manifest=save_manifest,
        fail_on_bad_input=fail_on_bad_input,
        infer_batch_size=infer_batch_size,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        input_prep=input_prep,
    )

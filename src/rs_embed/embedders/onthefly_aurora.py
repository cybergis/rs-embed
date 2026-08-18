from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import (
    BBox,
    ModelInputSpec,
    OutputSpec,
    PointBuffer,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from ..core.types import EmbedderCapabilities, FetchResult
from ..providers import ProviderBase
from ..providers.fetch import (
    fetch_latlon_grid_bins_tchw as _fetch_latlon_grid_bins_tchw,
)
from ..tools.runtime import load_cached_with_device as _load_cached_with_device
from ..tools.shape import (
    crop_grid_and_pool,
    crop_grid_to_roi,
    geo_roi_from_meta,
    roi_fetch_meta,
)
from .base import EmbedderBase
from .config import model_config_value
from .meta import build_meta, temporal_to_range
from .shared import grid_to_dataarray, verify_loaded_params

# ---------------------------------------------------------------------------
# Constants — Aurora 0.25° input contract
# ---------------------------------------------------------------------------

# GEE surface-level source: the four ERA5 single-level variables Aurora needs,
# in Aurora's canonical surf_vars order (2t, 10u, 10v, msl).
_GEE_COLLECTION = "ECMWF/ERA5_HOURLY"
_SURF_BANDS_GEE: tuple[str, ...] = (
    "temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "mean_sea_level_pressure",
)
_SURF_VARS: tuple[str, ...] = ("2t", "10u", "10v", "msl")

# Pressure-level variables (Aurora names → ARCO-ERA5 names) and the 13
# pressure levels Aurora was pretrained on (hPa). Fixed, not configurable:
# they change the fetched channel stack, and fetch-affecting model_config is
# not forwarded to prefetch paths (see docs/models/aurora.md).
_ATMOS_VARS: tuple[str, ...] = ("t", "u", "v", "q", "z")
_ATMOS_VARS_ARCO: dict[str, str] = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}
_ATMOS_LEVELS: tuple[int, ...] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
_ARCO_ZARR_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Static variables come from the official checkpoint repo, not from GEE.
_STATIC_VARS: tuple[str, ...] = ("lsm", "slt", "z")
_HF_REPO = "microsoft/aurora"
_STATIC_PICKLE = "aurora-0.25-static.pickle"

# ERA5 0.25° grid: GEE nominal scale, global raster shape, Aurora patch size.
_SCALE_M = 27830
_GRID_DEG = 0.25
_GLOBAL_HW = (721, 1440)
_PATCH_SIZE = 4
_WINDOW_PX = 32  # fixed lattice window side (8° × 8°); multiple of _PATCH_SIZE

# Aurora consumes two states 6 h apart: (t-6h, t).
_HISTORY_STEPS = 2
_STEP_HOURS = 6

_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "pretrained": {"ckpt": "aurora-0.25-pretrained.ckpt", "dim": 512, "small": False},
    "small": {"ckpt": "aurora-0.25-small-pretrained.ckpt", "dim": 256, "small": True},
}
_DEFAULT_VARIANT = "pretrained"

# Channel layout of the combined fetch array (TCHW, T=2): surf | atmos | static.
_N_SURF = len(_SURF_VARS)
_N_ATMOS = len(_ATMOS_VARS) * len(_ATMOS_LEVELS)
_N_STATIC = len(_STATIC_VARS)
_N_CHANNELS = _N_SURF + _N_ATMOS + _N_STATIC


# ---------------------------------------------------------------------------
# Temporal resolution — snapshot pair, not a window composite
# ---------------------------------------------------------------------------


def resolve_time_pair(temporal: TemporalSpec | None) -> tuple[datetime, datetime]:
    """Resolve a temporal spec to Aurora's (t-6h, t) snapshot pair.

    ``t`` is the last 00/06/12/18 UTC boundary strictly before the resolved
    range end (Aurora embeds an instantaneous atmospheric state, not a window
    composite).
    """
    t = temporal_to_range(temporal)
    end = datetime.fromisoformat(str(t.end))
    boundary = end.replace(minute=0, second=0, microsecond=0)
    boundary -= timedelta(hours=boundary.hour % _STEP_HOURS)
    if boundary >= end:
        boundary -= timedelta(hours=_STEP_HOURS)
    return boundary - timedelta(hours=_STEP_HOURS), boundary


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _hour_bins(times: tuple[datetime, ...]) -> list[tuple[str, str]]:
    """One [t, t+1h) bin per snapshot — selects exactly one hourly ERA5 asset."""
    return [(_iso(t), _iso(t + timedelta(hours=1))) for t in times]


# ---------------------------------------------------------------------------
# Lattice window — geometry on the global 0.25° grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LatticeWindow:
    """A window of grid *points* on the global 0.25° lattice, north-up.

    ``lat_max``/``lon_min`` locate the NW corner point; rows step ``-grid_deg``
    in latitude and columns ``+grid_deg`` in longitude (``lon`` in [0, 360)).
    """

    lat_max: float
    lon_min: float
    n: int = _WINDOW_PX
    grid_deg: float = _GRID_DEG

    @property
    def lat_min(self) -> float:
        return self.lat_max - (self.n - 1) * self.grid_deg

    @property
    def lon_max(self) -> float:
        return self.lon_min + (self.n - 1) * self.grid_deg


def _spatial_bounds(spatial: SpatialSpec) -> tuple[float, float, float, float]:
    """ROI bounds as (minlon, minlat, maxlon, maxlat) in EPSG:4326 degrees."""
    if isinstance(spatial, PointBuffer):
        spatial.validate()
        dlat = float(spatial.buffer_m) / 111_320.0
        coslat = max(math.cos(math.radians(float(spatial.lat))), 1e-2)
        dlon = float(spatial.buffer_m) / (111_320.0 * coslat)
        return (
            float(spatial.lon) - dlon,
            float(spatial.lat) - dlat,
            float(spatial.lon) + dlon,
            float(spatial.lat) + dlat,
        )
    if isinstance(spatial, BBox):
        spatial.validate()
        return (
            float(spatial.minlon),
            float(spatial.minlat),
            float(spatial.maxlon),
            float(spatial.maxlat),
        )
    raise ModelError(
        f"Aurora supports PointBuffer/BBox spatial specs, got {type(spatial).__name__}."
    )


def _lattice_window(spatial: SpatialSpec) -> _LatticeWindow:
    """The ``_WINDOW_PX``-sized lattice window centered on the ROI.

    The window is snapped to the global 0.25° lattice and shifted (not
    shrunk) when centering would run past a pole or the lon-0 seam, so the
    ROI stays inside and Aurora's Metadata constraints (lat in [-90, 90]
    strictly decreasing, lon in [0, 360) strictly increasing) always hold.
    """
    minlon, minlat, maxlon, maxlat = _spatial_bounds(spatial)
    lat_c = 0.5 * (minlat + maxlat)
    lon_c = (0.5 * (minlon + maxlon)) % 360.0

    n, g = _WINDOW_PX, _GRID_DEG
    half = n // 2
    lat_max = round(lat_c / g) * g + half * g
    lat_max = min(lat_max, 90.0)
    lat_max = max(lat_max, -90.0 + (n - 1) * g)
    # Snap before wrapping so centers just west of 360° keep an unwrapped
    # window ([352, 359.75] rather than a clamp to the 0° edge).
    lon_min = round(lon_c / g) * g - half * g
    lon_min = min(max(lon_min, 0.0), 360.0 - n * g)
    return _LatticeWindow(lat_max=round(lat_max, 6), lon_min=round(lon_min, 6), n=n, grid_deg=g)


def _window_roi(spatial: SpatialSpec, win: _LatticeWindow) -> tuple[float, float, float, float]:
    """The ROI's normalized (y0, y1, x0, x1) window inside the lattice window.

    Same convention as ``tools.spatial.square_spatial``: fractions in [0, 1]
    of the window's edge-to-edge extent, row 0 = north. Degenerate/oversized
    ROIs clamp to the window (a full window means "nothing to crop").
    """
    minlon, minlat, maxlon, maxlat = _spatial_bounds(spatial)
    extent = win.n * win.grid_deg
    top = win.lat_max + win.grid_deg / 2.0
    left = win.lon_min - win.grid_deg / 2.0
    x0 = (minlon % 360.0 - left) / extent
    x1 = (maxlon % 360.0 - left) / extent
    if x1 < x0:  # ROI wrapped the lon seam relative to the window: keep order
        x0, x1 = x1, x0
    y0 = (top - maxlat) / extent
    y1 = (top - minlat) / extent
    clamp = [min(max(v, 0.0), 1.0) for v in (y0, y1, x0, x1)]
    return tuple(round(v, 6) for v in clamp)  # type: ignore[return-value]


def _window_from_global(arr: np.ndarray, win: _LatticeWindow) -> np.ndarray:
    """Slice a lattice window out of a global (…, 721, 1440) 0.25° field."""
    if arr.shape[-2:] != _GLOBAL_HW:
        raise ModelError(
            f"Expected a global 0.25° field with shape (..., {_GLOBAL_HW[0]}, "
            f"{_GLOBAL_HW[1]}), got {arr.shape}."
        )
    r0 = int(round((90.0 - win.lat_max) / win.grid_deg))
    c0 = int(round(win.lon_min / win.grid_deg))
    return np.ascontiguousarray(arr[..., r0 : r0 + win.n, c0 : c0 + win.n])


# ---------------------------------------------------------------------------
# ARCO-ERA5 pressure levels + HF static fields
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _arco_dataset() -> Any:
    try:
        import xarray as xr
    except ModuleNotFoundError as e:  # pragma: no cover - xarray is a base dep
        raise ModelError("Aurora needs xarray to read ARCO-ERA5.") from e
    try:
        return xr.open_zarr(_ARCO_ZARR_PATH, chunks=None, storage_options={"token": "anon"})
    except Exception as e:
        raise ModelError(
            f"Failed to open the ARCO-ERA5 zarr store at {_ARCO_ZARR_PATH!r}. "
            "Anonymous GCS access needs the 'aurora' extra (gcsfs, zarr): "
            f"pip install 'rs-embed[aurora]'. Original error: {type(e).__name__}: {e}"
        ) from e


@lru_cache(maxsize=_HISTORY_STEPS * 2)
def _arco_global_slab(time_iso: str) -> np.ndarray:
    """Global pressure-level state at one timestamp: (5, 13, 721, 1440) float32.

    ARCO-ERA5 is chunked per timestep with whole-globe slabs, so any window
    read downloads the full field anyway (~270 MB per timestamp). Caching the
    global slab makes every additional point at the same timestamp a local
    array slice — the intended pattern for batch runs.
    """
    ds = _arco_dataset()
    t64 = np.datetime64(time_iso)
    times = ds["time"].values
    if t64 < times[0] or t64 > times[-1]:
        raise ModelError(
            f"Timestamp {time_iso} is outside ARCO-ERA5 coverage "
            f"({np.datetime_as_string(times[0], unit='h')} .. "
            f"{np.datetime_as_string(times[-1], unit='h')}; the mirror lags "
            "ERA5 by a few months). Choose a temporal range ending earlier."
        )
    try:
        sel = ds.sel(time=t64, level=list(_ATMOS_LEVELS))
    except KeyError as e:
        raise ModelError(
            f"ARCO-ERA5 selection failed for time={time_iso}, levels={_ATMOS_LEVELS}: {e}"
        ) from e
    lat_desc = bool(ds["latitude"].values[0] > ds["latitude"].values[-1])
    fields: list[np.ndarray] = []
    for k in _ATMOS_VARS:
        v = np.asarray(sel[_ATMOS_VARS_ARCO[k]].values, dtype=np.float32)
        if v.shape != (len(_ATMOS_LEVELS), *_GLOBAL_HW):
            raise ModelError(
                f"ARCO-ERA5 variable {_ATMOS_VARS_ARCO[k]!r} has shape {v.shape}, "
                f"expected {(len(_ATMOS_LEVELS), *_GLOBAL_HW)}."
            )
        fields.append(v if lat_desc else v[:, ::-1, :])
    return np.stack(fields, axis=0)


@lru_cache(maxsize=1)
def _static_fields() -> dict[str, np.ndarray]:
    """Aurora's official static fields (lsm, slt, z), each (721, 1440) float32."""
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as e:  # pragma: no cover - hub is a base dep
        raise ModelError("Aurora needs huggingface-hub to download static fields.") from e
    try:
        path = hf_hub_download(repo_id=_HF_REPO, filename=_STATIC_PICKLE)
    except Exception as e:
        raise ModelError(
            f"Failed to download {_STATIC_PICKLE!r} from {_HF_REPO!r}: {type(e).__name__}: {e}"
        ) from e
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out: dict[str, np.ndarray] = {}
    for k in _STATIC_VARS:
        if k not in raw:
            raise ModelError(f"Static pickle {_STATIC_PICKLE!r} is missing variable {k!r}.")
        v = raw[k]
        v = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
        v = np.asarray(v, dtype=np.float32)
        v = v.reshape(v.shape[-2], v.shape[-1])
        if v.shape != _GLOBAL_HW:
            raise ModelError(f"Static variable {k!r} has shape {v.shape}, expected {_GLOBAL_HW}.")
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Encoder-only model loading + token capture
# ---------------------------------------------------------------------------


class _EncoderTapDone(Exception):
    """Control-flow sentinel: stops ``Aurora.forward`` right after the encoder."""


@lru_cache(maxsize=2)
def _load_aurora_encoder_cached(*, variant: str, dev: str) -> tuple[Any, dict[str, Any]]:
    spec = _VARIANT_SPECS[variant]
    try:
        import torch
        from aurora import AuroraPretrained, AuroraSmallPretrained
    except ModuleNotFoundError as e:
        raise ModelError(
            "Aurora requires the optional dependency 'microsoft-aurora'. "
            "Install with: pip install 'rs-embed[aurora]'."
        ) from e
    cls = AuroraSmallPretrained if spec["small"] else AuroraPretrained
    model = cls()
    try:
        model.load_checkpoint()
    except Exception as e:
        raise ModelError(
            f"Failed to download/load Aurora checkpoint {spec['ckpt']!r} from "
            f"{_HF_REPO!r}: {type(e).__name__}: {e}"
        ) from e
    # Encoder-only: swap the Swin3D backbone and decoder for stubs so ~99% of
    # the loaded parameters can be released. They are never called — the
    # encoder tap aborts the forward pass first.
    model.backbone = torch.nn.Identity()
    model.decoder = torch.nn.Identity()
    model.eval()
    try:
        model.to(dev)
    except Exception as e:
        raise ModelError(
            f"Failed to move Aurora encoder to device {dev!r}: {type(e).__name__}: {e}"
        ) from e
    stats = verify_loaded_params(model.encoder, model_name="aurora")
    meta = {
        "hf_id": f"{_HF_REPO}/{spec['ckpt']}",
        "variant": variant,
        "dim": int(spec["dim"]),
        "encoder_only": True,
        "latent_levels": int(model.encoder.latent_levels),
        "patch_size": int(model.patch_size),
        **{f"encoder_{k}": v for k, v in stats.items()},
    }
    return model, meta


def _encoder_tokens(model: Any, batch: Any) -> np.ndarray:
    """Run the official ``Aurora.forward`` and capture the encoder output.

    Running the real forward (instead of re-implementing its preamble) keeps
    the official preprocessing — dtype cast, ``batch.normalise`` with the
    published statistics, patch-size crop, device move, static broadcast,
    positive-variable clamps, lead-time resolution — with zero duplication;
    the tap aborts before the stubbed backbone. Returns tokens ``(L, D)``.
    """
    import torch

    grabbed: list[Any] = []

    def _tap(_module: Any, _inputs: Any, out: Any) -> None:
        grabbed.append(out)
        raise _EncoderTapDone

    handle = model.encoder.register_forward_hook(_tap)
    try:
        with torch.inference_mode():
            try:
                model.forward(batch)
            except _EncoderTapDone:
                pass
    finally:
        handle.remove()
    if not grabbed:
        raise ModelError(
            "Aurora.forward finished without the encoder tap firing; cannot extract encoder tokens."
        )
    x = grabbed[0]
    if not torch.is_tensor(x) or x.ndim != 3:
        raise ModelError(f"Aurora encoder returned {type(x).__name__}, expected (B, L, D) tokens.")
    return x.detach().to(device="cpu", dtype=torch.float32).numpy()[0]


def _tokens_to_level_grid(
    tokens_ld: np.ndarray, *, latent_levels: int, hw: tuple[int, int]
) -> np.ndarray:
    """Latent tokens (L, D) → level-averaged column-feature grid (D, h, w)."""
    h, w = hw
    n_tokens, d = tokens_ld.shape
    if n_tokens != latent_levels * h * w:
        raise ModelError(
            f"Aurora encoder returned {n_tokens} tokens, expected "
            f"latent_levels*h*w = {latent_levels}*{h}*{w}."
        )
    grid_hwd = tokens_ld.reshape(latent_levels, h, w, d).mean(axis=0)
    return np.ascontiguousarray(np.transpose(grid_hwd, (2, 0, 1))).astype(np.float32)


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def _assemble_batch(
    arr_tchw: np.ndarray, *, win: _LatticeWindow, times: tuple[datetime, datetime]
) -> Any:
    """Split the combined (2, 72, n, n) array into an ``aurora.Batch``."""
    try:
        import torch
        from aurora import Batch, Metadata
    except ModuleNotFoundError as e:
        raise ModelError(
            "Aurora requires the optional dependency 'microsoft-aurora'. "
            "Install with: pip install 'rs-embed[aurora]'."
        ) from e

    n_levels = len(_ATMOS_LEVELS)
    surf = arr_tchw[:, :_N_SURF]
    atmos = arr_tchw[:, _N_SURF : _N_SURF + _N_ATMOS].reshape(
        arr_tchw.shape[0], len(_ATMOS_VARS), n_levels, win.n, win.n
    )
    static = arr_tchw[0, _N_SURF + _N_ATMOS :]
    lat = torch.from_numpy((win.lat_max - np.arange(win.n) * win.grid_deg).astype(np.float32))
    lon = torch.from_numpy((win.lon_min + np.arange(win.n) * win.grid_deg).astype(np.float32))
    try:
        return Batch(
            surf_vars={
                k: torch.from_numpy(np.ascontiguousarray(surf[None, :, i]))
                for i, k in enumerate(_SURF_VARS)
            },
            static_vars={
                k: torch.from_numpy(np.ascontiguousarray(static[i]))
                for i, k in enumerate(_STATIC_VARS)
            },
            atmos_vars={
                k: torch.from_numpy(np.ascontiguousarray(atmos[None, :, i]))
                for i, k in enumerate(_ATMOS_VARS)
            },
            metadata=Metadata(
                lat=lat, lon=lon, time=(times[1],), atmos_levels=tuple(_ATMOS_LEVELS)
            ),
        )
    except ValueError as e:
        raise ModelError(f"Failed to assemble the Aurora Batch: {e}") from e


def _check_finite(arr_tchw: np.ndarray, times: tuple[datetime, datetime]) -> None:
    if np.isfinite(arr_tchw).all():
        return
    empty = [_iso(t) for i, t in enumerate(times) if bool(np.isnan(arr_tchw[i, :_N_SURF]).all())]
    if empty:
        raise ModelError(
            f"Aurora input has empty time frames for {empty} — the hourly ERA5 "
            "asset is missing (ERA5 publishes with ~5 days delay). Choose a "
            "temporal range ending earlier."
        )
    raise ModelError("Aurora input contains non-finite values; cannot run the encoder.")


def _resolve_variant(model_config: dict[str, Any] | None) -> str:
    raw = model_config_value(model_config, "variant")
    if raw is None:
        return _DEFAULT_VARIANT
    v = str(raw).strip().lower()
    if v not in _VARIANT_SPECS:
        raise ModelError(f"Unknown Aurora variant {raw!r}; choices: {sorted(_VARIANT_SPECS)}.")
    return v


@register("aurora")
class AuroraEmbedder(EmbedderBase):
    """Microsoft Aurora 0.25° atmospheric foundation model, encoder-only.

    Unlike the imagery models in this package, Aurora embeds ERA5-style
    weather states: given a point/ROI and a temporal range, the adapter takes
    a fixed 32×32 window on the global 0.25° lat/lon lattice around the ROI
    at the last 6 h boundary before the range end (plus the state 6 h
    earlier), assembles the official ``aurora.Batch``, runs the encoder only,
    and pools the latent column tokens.

    Inputs (three sources, one Batch — see docs/models/aurora.md):
      - surf_vars  (2t, 10u, 10v, msl): GEE ``ECMWF/ERA5_HOURLY``
      - atmos_vars (t, u, v, q, z @ 13 hPa levels): ARCO-ERA5 public zarr
      - static_vars (lsm, slt, z): HF ``microsoft/aurora`` static pickle

    Outputs:
      - pooled: mean/max over the level-averaged column-token grid,
        ROI-cropped when the ROI is a window subset
      - grid  : ``[D, 8, 8]`` column-token map (window/patch_size)

    Variants (via ``model_config={"variant": ...}``): ``pretrained``
    (default, D=512) and ``small`` (D=256).
    """

    input_spec = ModelInputSpec(
        collection=_GEE_COLLECTION,
        bands=_SURF_BANDS_GEE,
        scale_m=_SCALE_M,
        temporal_mode="multi",
        n_frames=_HISTORY_STEPS,
    )

    # Explicit pipeline-routing capabilities; the contract test asserts these
    # match the actual method signatures (tests/test_capabilities_contract.py).
    capabilities = EmbedderCapabilities(
        input_chw=True,
        fetch_meta=True,
        batch_fetch_metas=True,
        model_config_single=True,
        model_config_batch=True,
        model_config_batch_inputs=True,
    )

    DEFAULT_VARIANT = _DEFAULT_VARIANT
    WINDOW_PX = _WINDOW_PX

    def describe(self) -> dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["provider", "tensor"],
            "inputs": {
                "provider_default": {
                    "collection": _GEE_COLLECTION,
                    "bands": list(_SURF_BANDS_GEE),
                },
                "atmos": {
                    "source": _ARCO_ZARR_PATH,
                    "variables": list(_ATMOS_VARS),
                    "levels_hpa": list(_ATMOS_LEVELS),
                },
                "static": {
                    "source": f"hf://{_HF_REPO}/{_STATIC_PICKLE}",
                    "variables": list(_STATIC_VARS),
                },
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "variant": _DEFAULT_VARIANT,
                "window_px": _WINDOW_PX,
                "scale_m": _SCALE_M,
                "atmos_levels": list(_ATMOS_LEVELS),
            },
            "model_config": {
                "variant": {
                    "type": "string",
                    "default": _DEFAULT_VARIANT,
                    "choices": sorted(_VARIANT_SPECS),
                    "description": (
                        "Checkpoint size: 'pretrained' (aurora-0.25-pretrained, D=512) "
                        "or 'small' (aurora-0.25-small-pretrained, D=256)."
                    ),
                },
            },
            "notes": [
                "Aurora embeds ERA5 weather states, not satellite imagery.",
                "Encoder-only: the Swin3D backbone and decoder are replaced by "
                "stubs after checkpoint load; tokens are per-patch (1°×1°) "
                "atmospheric column features with no cross-patch attention.",
                "Temporal semantics: snapshot at the last 6 h UTC boundary "
                "before the range end, plus the state 6 h earlier (T=2).",
                "The 32-cell window and the 13 pressure levels are fixed: both "
                "change the fetched input, and fetch-affecting model_config is "
                "not forwarded to prefetch paths.",
                "First fetch per timestamp downloads a global ARCO-ERA5 "
                "pressure-level slab (~270 MB) and caches it in-process; "
                "further points at the same timestamp are local slices.",
                "Requires the 'aurora' extra (microsoft-aurora, gcsfs, zarr).",
                "All variables enter the model in native physical units; "
                "normalization happens inside Aurora with its official "
                "statistics.",
            ],
        }

    def _default_sensor(self) -> SensorSpec:
        assert self.input_spec is not None
        return self.input_spec.to_sensor_spec()

    def fetch_input(
        self,
        provider: ProviderBase,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec,
        square_input: bool = True,
    ) -> FetchResult | None:
        """Fetch the combined Aurora input stack as (2, 72, 32, 32) float32.

        Channel layout per frame: 4 surface vars (GEE) | 5 atmos vars × 13
        levels (ARCO-ERA5) | 3 static vars (HF pickle, repeated per frame).
        ``square_input`` is ignored — the lattice window replaces
        fetch-squaring for this model.
        """
        times = resolve_time_pair(temporal)
        win = _lattice_window(spatial)
        ss = sensor if sensor is not None else self._default_sensor()
        if len(ss.bands) != _N_SURF:
            raise ModelError(
                f"Aurora needs exactly {_N_SURF} surface bands in order "
                f"{_SURF_VARS}, got {len(ss.bands)}: {ss.bands}."
            )
        surf_tchw, bin_meta = _fetch_latlon_grid_bins_tchw(
            provider,
            collection=ss.collection,
            bands=ss.bands,
            bins=_hour_bins(times),
            lat_max=win.lat_max,
            lon_min=win.lon_min,
            n_lat=win.n,
            n_lon=win.n,
            grid_deg=win.grid_deg,
        )
        if bin_meta.get("n_empty"):
            missing = [f["start"] for f in bin_meta.get("frames", []) if f.get("empty")]
            raise ModelError(
                f"No hourly {ss.collection} asset for {missing} (ERA5 publishes "
                "with ~5 days delay). Choose a temporal range ending earlier."
            )
        atmos = np.stack([_window_from_global(_arco_global_slab(_iso(t)), win) for t in times])
        static = np.stack([_window_from_global(_static_fields()[k], win) for k in _STATIC_VARS])
        data = np.concatenate(
            [
                surf_tchw,
                atmos.reshape(len(times), _N_ATMOS, win.n, win.n),
                np.repeat(static[None], len(times), axis=0),
            ],
            axis=1,
        ).astype(np.float32)
        meta = {
            "time_pair": [_iso(t) for t in times],
            "window": {
                "lat_max": win.lat_max,
                "lon_min": win.lon_min,
                "n": win.n,
                "grid_deg": win.grid_deg,
            },
            **(roi_fetch_meta(_window_roi(spatial, win)) or {}),
        }
        return FetchResult(data=data, meta=meta)

    @staticmethod
    def _build_embedding(
        grid_dhw: np.ndarray,
        *,
        geo_roi: tuple[float, float, float, float],
        output: OutputSpec,
        meta: dict[str, Any],
    ) -> Embedding:
        if output.mode == "pooled":
            if output.pooling not in ("mean", "max"):
                raise ModelError(f"Unknown pooling={output.pooling!r} (expected 'mean' or 'max').")
            _, vec = crop_grid_and_pool(grid_dhw, geo_roi, pooling=output.pooling)
            if vec is None:
                reduce = grid_dhw.max if output.pooling == "max" else grid_dhw.mean
                vec = np.asarray(reduce(axis=(1, 2)), dtype=np.float32)
                pooling_name = f"token_{output.pooling}"
            else:
                pooling_name = f"roi_grid_{output.pooling}"
            ometa = {**meta, "pooling": pooling_name, "pooled_shape": tuple(vec.shape)}
            return Embedding(data=vec.astype(np.float32), meta=ometa)
        if output.mode == "grid":
            g = crop_grid_to_roi(grid_dhw, geo_roi)
            gmeta = {
                **meta,
                "grid_type": "aurora_column_tokens",
                "grid_hw": (int(g.shape[1]), int(g.shape[2])),
                "grid_shape": tuple(g.shape),
            }
            return Embedding(data=grid_to_dataarray(g.astype(np.float32), meta=gmeta), meta=gmeta)
        raise ModelError(f"Unknown output mode: {output.mode}")

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        input_chw: np.ndarray | None = None,
        fetch_meta: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> Embedding:
        variant = _resolve_variant(model_config)
        t = temporal_to_range(temporal)
        times = resolve_time_pair(temporal)
        win = _lattice_window(spatial)
        ss = sensor if sensor is not None else self._default_sensor()
        backend_l = str(backend).lower()

        if input_chw is not None:
            arr = np.asarray(input_chw, dtype=np.float32)
            expected = (_HISTORY_STEPS, _N_CHANNELS, win.n, win.n)
            if arr.shape != expected:
                raise ModelError(
                    f"Aurora input_chw must be the combined TCHW stack {expected} "
                    f"(surf | atmos×levels | static), got {getattr(arr, 'shape', None)}."
                )
            geo_roi = geo_roi_from_meta(fetch_meta)
            fetch_note: dict[str, Any] = {"input_override": True}
        else:
            if backend_l == "tensor":
                raise ModelError("backend='tensor' requires input_chw for Aurora.")
            provider = self._get_provider(backend)
            fr = self.fetch_input(provider, spatial=spatial, temporal=temporal, sensor=ss)
            assert fr is not None
            arr = np.asarray(fr.data, dtype=np.float32)
            geo_roi = geo_roi_from_meta(fr.meta)
            fetch_note = {k: v for k, v in fr.meta.items() if k != "roi_window_geo"}
        _check_finite(arr, times)

        batch = _assemble_batch(arr, win=win, times=times)
        (model, mmeta), dev = _load_cached_with_device(
            _load_aurora_encoder_cached, device=device, variant=variant
        )
        tokens = _encoder_tokens(model, batch)
        grid = _tokens_to_level_grid(
            tokens,
            latent_levels=int(mmeta["latent_levels"]),
            hw=(win.n // _PATCH_SIZE, win.n // _PATCH_SIZE),
        )

        meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend=backend_l,
            source=ss.collection,
            sensor=ss,
            temporal=t,
            image_size=win.n,
            extra={
                "output_mode": output.mode,
                "device": dev,
                "time_pair": [_iso(x) for x in times],
                "window": {
                    "lat_max": win.lat_max,
                    "lon_min": win.lon_min,
                    "n": win.n,
                    "grid_deg": win.grid_deg,
                },
                "atmos_levels": list(_ATMOS_LEVELS),
                "atmos_source": _ARCO_ZARR_PATH,
                "static_source": f"hf://{_HF_REPO}/{_STATIC_PICKLE}",
                **fetch_note,
                **mmeta,
            },
        )
        return self._build_embedding(grid, geo_roi=geo_roi, output=output, meta=meta)

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
        fetch_metas: list[dict[str, Any] | None] | None = None,
    ) -> list[Embedding]:
        """Batch inference over prefetched combined input stacks.

        A true stacked forward is not possible for Aurora: every item has its
        own lattice window, and ``aurora.Batch`` carries a single lat/lon
        grid for the whole batch. Instead the encoder is loaded once up
        front and every item runs through the exact single-item path (same
        Batch assembly, same ``_build_embedding``).
        """
        if len(spatials) != len(input_chws):
            raise ValueError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        if fetch_metas is not None and len(fetch_metas) != len(spatials):
            raise ValueError(
                f"spatials/fetch_metas length mismatch: {len(spatials)} != {len(fetch_metas)}"
            )
        variant = _resolve_variant(model_config)
        _load_cached_with_device(_load_aurora_encoder_cached, device=device, variant=variant)
        return [
            self.get_embedding(
                spatial=s,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
                input_chw=x,
                fetch_meta=(fetch_metas[k] if fetch_metas is not None else None),
                model_config=model_config,
            )
            for k, (s, x) in enumerate(zip(spatials, input_chws, strict=True))
        ]

from __future__ import annotations

import glob
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.errors import ModelError

HF_REPO_ID = "torchgeo/copernicus_embed"
HF_FILENAME = "embed_map_310k.tif"
HF_REVISION = "435b4a7bdce6f6fdbf4272f9d6e54f2604f35fdb"
HF_MIN_BYTES = 700 * 1024 * 1024
_COPERNICUS_MEMMAP_FALLBACK_WARNED = False


@dataclass(frozen=True)
class GeoTiffMeta:
    path: str
    bands: int
    height: int
    width: int
    left: float
    bottom: float
    right: float
    top: float
    xres: float
    yres: float
    axis_order: str


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _find_embed_tif(path: str) -> str | None:
    p = _normalize_path(path)
    if os.path.isfile(p):
        return p if p.lower().endswith((".tif", ".tiff")) else None
    if not os.path.isdir(p):
        return None

    matches = sorted(glob.glob(os.path.join(p, "embed_map_*.tif")))
    if matches:
        return matches[0]

    fallback = os.path.join(p, HF_FILENAME)
    if os.path.exists(fallback):
        return fallback
    return None


def _validate_large_file(path: str, *, min_bytes: int = HF_MIN_BYTES) -> str:
    if not os.path.exists(path):
        raise ModelError(f"Copernicus embed GeoTIFF not found: {path}")
    size = os.path.getsize(path)
    if size < int(min_bytes):
        raise ModelError(
            f"Found '{path}' but it's only {size} bytes, which is too small to be the "
            "real Copernicus embed GeoTIFF. It may be an incomplete pointer/download."
        )
    return path


def _download_embed_tif(data_dir: str) -> str:
    from ..shared import hf_hub_download_cache_first

    root = _normalize_path(data_dir)
    os.makedirs(root, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    kwargs = dict(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        revision=HF_REVISION,
        repo_type="dataset",
        token=token,
    )
    try:
        path = hf_hub_download_cache_first(local_dir=root, **kwargs)
    except TypeError:
        path = hf_hub_download_cache_first(cache_dir=root, **kwargs)
    except Exception as e:
        raise ModelError(
            f"Failed to download Copernicus embed GeoTIFF from Hugging Face dataset '{HF_REPO_ID}'."
        ) from e

    return _validate_large_file(path)


def resolve_embed_tif_path(*, data_dir: str, auto_download: bool) -> str:
    found = _find_embed_tif(data_dir)
    if found is not None:
        return _validate_large_file(found)
    if not auto_download:
        raise ModelError(
            f"Copernicus embed GeoTIFF not found under '{_normalize_path(data_dir)}'. "
            f"Expected a file like '{HF_FILENAME}'."
        )
    return _download_embed_tif(data_dir)


def _require_tifffile():
    try:
        import tifffile
    except Exception as e:
        raise ModelError(
            "CopernicusEmbed requires tifffile for GeoTIFF access. "
            "Reinstall rs-embed or run: pip install tifffile"
        ) from e
    return tifffile


def _raise_for_missing_imagecodecs(exc: Exception) -> None:
    msg = str(exc)
    if "imagecodecs" not in msg:
        return
    raise ModelError(
        "Copernicus embed GeoTIFF decoding requires 'imagecodecs'. "
        "Reinstall rs-embed or run: pip install imagecodecs"
    ) from exc


def _infer_axis_order(shape: tuple[int, ...]) -> str:
    if len(shape) != 3:
        raise ModelError(f"Expected a 3D GeoTIFF array, got shape={shape}.")
    spatial_pairs = {(721, 1440), (1440, 721)}
    if tuple(shape[1:]) in spatial_pairs:
        return "chw"
    if tuple(shape[:2]) in spatial_pairs:
        return "hwc"
    raise ModelError(f"Unsupported Copernicus embed GeoTIFF shape={shape}.")


def _get_tiff_tag(tags: Any, name: str, code: int):
    tag = None
    if hasattr(tags, "get"):
        tag = tags.get(name)
        if tag is None:
            tag = tags.get(code)
    if tag is None:
        try:
            tag = tags[name]
        except Exception:
            try:
                tag = tags[code]
            except Exception:
                tag = None
    return tag


def _fallback_global_georef(
    shape: tuple[int, ...], axis_order: str
) -> tuple[float, float, float, float]:
    if axis_order == "chw":
        _, height, width = shape
    else:
        height, width, _ = shape
    if height == 721 and width == 1440:
        return -180.0, 90.125, 0.25, 0.25
    raise ModelError("Copernicus embed GeoTIFF is missing GeoTIFF georeferencing tags.")


def load_geotiff_meta(path: str) -> GeoTiffMeta:
    tifffile = _require_tifffile()
    path = _validate_large_file(path, min_bytes=1)

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(v) for v in series.shape)
        axis_order = _infer_axis_order(shape)
        page = tif.pages[0]
        tags = page.tags

        scale_tag = _get_tiff_tag(tags, "ModelPixelScaleTag", 33550)
        tie_tag = _get_tiff_tag(tags, "ModelTiepointTag", 33922)

        if scale_tag is None or tie_tag is None:
            left, top, xres, yres = _fallback_global_georef(shape, axis_order)
        else:
            scale = tuple(float(v) for v in scale_tag.value)
            tie = tuple(float(v) for v in tie_tag.value)
            if len(scale) < 2 or len(tie) < 6:
                raise ModelError("Invalid GeoTIFF georeferencing tags in Copernicus embed file.")
            xres = abs(scale[0])
            yres = abs(scale[1])
            left = tie[3] - tie[0] * xres
            top = tie[4] + tie[1] * yres

    if axis_order == "chw":
        bands, height, width = shape
    else:
        height, width, bands = shape

    right = left + width * xres
    bottom = top - height * yres
    return GeoTiffMeta(
        path=path,
        bands=int(bands),
        height=int(height),
        width=int(width),
        left=float(left),
        bottom=float(bottom),
        right=float(right),
        top=float(top),
        xres=float(xres),
        yres=float(yres),
        axis_order=axis_order,
    )


def _bbox_to_window(
    *,
    meta: GeoTiffMeta,
    minlon: float,
    minlat: float,
    maxlon: float,
    maxlat: float,
) -> tuple[int, int, int, int]:
    req_width = float(maxlon) - float(minlon)
    req_height = float(maxlat) - float(minlat)
    if req_width < meta.xres or req_height < meta.yres:
        raise ModelError(
            "Requested Copernicus bbox is smaller than one dataset pixel "
            f"({meta.xres} deg x {meta.yres} deg). "
            f"got=({req_width} deg x {req_height} deg)"
        )

    left = max(float(minlon), meta.left)
    right = min(float(maxlon), meta.right)
    bottom = max(float(minlat), meta.bottom)
    top = min(float(maxlat), meta.top)

    if right <= left or top <= bottom:
        raise ModelError(
            "Requested Copernicus bbox does not overlap dataset coverage. "
            f"coverage=({meta.left}, {meta.bottom}, {meta.right}, {meta.top})"
        )

    col0 = int(math.floor((left - meta.left) / meta.xres))
    col1 = int(math.ceil((right - meta.left) / meta.xres))
    row0 = int(math.floor((meta.top - top) / meta.yres))
    row1 = int(math.ceil((meta.top - bottom) / meta.yres))

    col0 = max(0, min(meta.width - 1, col0))
    row0 = max(0, min(meta.height - 1, row0))
    col1 = min(meta.width, col1)
    row1 = min(meta.height, row1)
    if col1 <= col0 or row1 <= row0:
        raise ModelError(
            "Requested Copernicus bbox does not resolve to any pixels after windowing. "
            f"window=({row0}, {row1}, {col0}, {col1})"
        )
    return row0, row1, col0, col1


class CopernicusEmbedGeoTiff:
    """Minimal local reader for the Copernicus embedding GeoTIFF."""

    def __init__(self, *, paths: str = "data", download: bool = False) -> None:
        self.root = _normalize_path(paths)
        self.path = resolve_embed_tif_path(data_dir=self.root, auto_download=download)
        self._meta: GeoTiffMeta | None = None
        self._array: np.ndarray | None = None

    @property
    def meta(self) -> GeoTiffMeta:
        if self._meta is None:
            self._meta = load_geotiff_meta(self.path)
        return self._meta

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            tifffile = _require_tifffile()
            try:
                self._array = tifffile.memmap(self.path)
            except ValueError as e:
                _raise_for_missing_imagecodecs(e)
                if "memory-mappable" not in str(e):
                    raise
                global _COPERNICUS_MEMMAP_FALLBACK_WARNED
                if not _COPERNICUS_MEMMAP_FALLBACK_WARNED:
                    warnings.warn(
                        "Copernicus GeoTIFF is not directly memory-mappable; "
                        "falling back to tifffile.asarray(out='memmap'). "
                        "This may be slower on first access.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                    _COPERNICUS_MEMMAP_FALLBACK_WARNED = True
                try:
                    with tifffile.TiffFile(self.path) as tif:
                        self._array = tif.asarray(series=0, out="memmap")
                except ValueError as e2:
                    _raise_for_missing_imagecodecs(e2)
                    raise
        return self._array

    def _slice_chw(self, row0: int, row1: int, col0: int, col1: int) -> np.ndarray:
        arr = self.array
        if self.meta.axis_order == "chw":
            out = arr[:, row0:row1, col0:col1]
        else:
            out = np.moveaxis(arr[row0:row1, col0:col1, :], -1, 0)
        return np.asarray(out, dtype=np.float32)

    def __getitem__(self, key: tuple[slice, slice]) -> dict[str, np.ndarray]:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("CopernicusEmbedGeoTiff expects ds[minlon:maxlon, minlat:maxlat].")

        lon_slice, lat_slice = key
        if not isinstance(lon_slice, slice) or not isinstance(lat_slice, slice):
            raise TypeError("CopernicusEmbedGeoTiff expects slice objects for lon/lat.")
        if lon_slice.start is None or lon_slice.stop is None:
            raise TypeError("Longitude slice must define start and stop.")
        if lat_slice.start is None or lat_slice.stop is None:
            raise TypeError("Latitude slice must define start and stop.")

        row0, row1, col0, col1 = _bbox_to_window(
            meta=self.meta,
            minlon=float(lon_slice.start),
            minlat=float(lat_slice.start),
            maxlon=float(lon_slice.stop),
            maxlat=float(lat_slice.stop),
        )
        return {
            "image": self._slice_chw(row0, row1, col0, col1),
            "window": np.array([row0, row1, col0, col1], dtype=np.int64),
        }

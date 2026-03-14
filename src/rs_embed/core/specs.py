from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal

from .errors import SpecError

@dataclass(frozen=True)
class BBox:
    """Axis-aligned geographic bounding box in EPSG:4326.

    Attributes
    ----------
    minlon : float
        Western longitude bound.
    minlat : float
        Southern latitude bound.
    maxlon : float
        Eastern longitude bound.
    maxlat : float
        Northern latitude bound.
    crs : str
        Coordinate reference system. Must currently be ``"EPSG:4326"``.
    """

    minlon: float
    minlat: float
    maxlon: float
    maxlat: float
    crs: str = "EPSG:4326"

    def validate(self) -> None:
        """Validate bbox bounds and CRS.

        Raises
        ------
        SpecError
            If CRS is unsupported or bounds are not strictly increasing.
        """
        if self.crs != "EPSG:4326":
            raise SpecError("BBox currently must be EPSG:4326 (v0.1).")
        if not (self.minlon < self.maxlon and self.minlat < self.maxlat):
            raise SpecError("Invalid bbox bounds.")

@dataclass(frozen=True)
class PointBuffer:
    """Point-centered spatial request with radius in meters.

    Attributes
    ----------
    lon : float
        Center longitude.
    lat : float
        Center latitude.
    buffer_m : float
        Buffer radius in meters.
    crs : str
        Coordinate reference system. Must currently be ``"EPSG:4326"``.
    """

    lon: float
    lat: float
    buffer_m: float
    crs: str = "EPSG:4326"

    def validate(self) -> None:
        """Validate point-buffer parameters.

        Raises
        ------
        SpecError
            If CRS is unsupported or ``buffer_m`` is not positive.
        """
        if self.crs != "EPSG:4326":
            raise SpecError("PointBuffer currently must be EPSG:4326 (v0.1).")
        if self.buffer_m <= 0:
            raise SpecError("buffer_m must be positive.")

SpatialSpec = BBox | PointBuffer

@dataclass(frozen=True)
class TemporalSpec:
    """Temporal request for annual or date-range model inputs.

    Attributes
    ----------
    mode : {"year", "range"}
        Interpretation mode for the remaining fields.
    year : int or None
        Year value used when ``mode="year"``.
    start : str or None
        ISO date string (``YYYY-MM-DD``) for range start.
    end : str or None
        ISO date string (``YYYY-MM-DD``) for range end.
    """

    mode: Literal["year", "range"]
    year: int | None = None
    start: str | None = None
    end: str | None = None

    @staticmethod
    def year(y: int) -> "TemporalSpec":
        """Build a year-mode temporal spec.

        Parameters
        ----------
        y : int
            Calendar year.

        Returns
        -------
        TemporalSpec
            Temporal spec with ``mode="year"``.
        """
        return TemporalSpec(mode="year", year=y)

    @staticmethod
    def range(start: str, end: str) -> "TemporalSpec":
        """Build a range-mode temporal spec.

        Parameters
        ----------
        start : str
            Start date (``YYYY-MM-DD``).
        end : str
            End date (``YYYY-MM-DD``).

        Returns
        -------
        TemporalSpec
            Temporal spec with ``mode="range"``.
        """
        return TemporalSpec(mode="range", start=start, end=end)

    def validate(self) -> None:
        """Validate temporal fields according to ``mode``.

        Raises
        ------
        SpecError
            If required fields are missing or date/year values are invalid.
        """
        if self.mode == "year":
            if self.year is None:
                raise SpecError("TemporalSpec.year requires year.")
            try:
                y = int(self.year)
            except Exception as e:
                raise SpecError("TemporalSpec.year requires an integer year.") from e
            if y < 1 or y > 9999:
                raise SpecError("TemporalSpec.year must be in [1, 9999].")
        elif self.mode == "range":
            if not self.start or not self.end:
                raise SpecError("TemporalSpec.range requires start and end.")
            try:
                start_d = date.fromisoformat(str(self.start))
                end_d = date.fromisoformat(str(self.end))
            except Exception as e:
                raise SpecError("TemporalSpec.range expects ISO dates 'YYYY-MM-DD'.") from e
            if start_d >= end_d:
                raise SpecError("TemporalSpec.range requires start < end.")
        else:
            raise SpecError(f"Unknown TemporalSpec mode: {self.mode}")

@dataclass(frozen=True)
class SensorSpec:
    """Sensor/source definition for on-the-fly provider fetching.

    Attributes
    ----------
    collection : str
        Provider collection identifier.
    bands : tuple[str, ...]
        Band names and order expected by the model.
    scale_m : int
        Pixel scale in meters.
    cloudy_pct : int
        Cloud threshold used by cloud-aware providers.
    fill_value : float
        Fill value used for missing data.
    composite : {"median", "mosaic"}
        Compositing strategy for multi-image windows.
    modality : str or None
        Optional model-facing modality selector (for example ``"s1"`` or
        ``"s2"``) when a model exposes multiple input branches.
    orbit : str or None
        Optional orbit/pass filter for sensors that support it.
    use_float_linear : bool
        Whether the provider path should use linear-scale floating-point
        values when a sensor family offers both linear and dB products.
    check_input : bool
        If ``True``, run input diagnostics.
    check_raise : bool
        If ``True``, raise on failed diagnostics.
    check_save_dir : str or None
        Optional directory to persist diagnostics artifacts.
    """

    collection: str
    bands: tuple[str, ...]
    scale_m: int = 10
    cloudy_pct: int = 30
    fill_value: float = 0.0
    composite: Literal["median", "mosaic"] = "median"
    modality: str | None = None
    orbit: str | None = None
    use_float_linear: bool = True

    # Optional: on-the-fly input inspection for GEE downloads.
    # If enabled, embedders can attach a compact stats report into Embedding.meta
    # (and optionally raise if issues are detected).
    check_input: bool = False
    check_raise: bool = True
    check_save_dir: str | None = None

@dataclass(frozen=True)
class OutputSpec:
    """Embedding output layout and post-processing policy.

    Attributes
    ----------
    mode : {"grid", "pooled"}
        Output representation mode.
    scale_m : int
        Output resolution in meters.
    pooling : {"mean", "max"}
        Pooling reducer for pooled output mode.
    grid_orientation : {"north_up", "native"}
        Orientation policy for grid outputs.
    """

    mode: Literal["grid", "pooled"]
    scale_m: int = 10
    pooling: Literal["mean", "max"] = "mean"
    # Grid orientation policy:
    # - north_up: normalize y-axis to north->south when metadata permits.
    # - native: keep model/provider native orientation.
    grid_orientation: Literal["north_up", "native"] = "north_up"

    @staticmethod
    def grid(
        scale_m: int = 10,
        *,
        grid_orientation: Literal["north_up", "native"] = "north_up",
    ) -> "OutputSpec":
        """Build a grid-output specification.

        Parameters
        ----------
        scale_m : int
            Output pixel scale in meters.
        grid_orientation : {"north_up", "native"}
            Orientation policy for returned grid embeddings.

        Returns
        -------
        OutputSpec
            Output specification with ``mode="grid"``.
        """
        return OutputSpec(mode="grid", scale_m=scale_m, grid_orientation=grid_orientation)

    @staticmethod
    def pooled(pooling: Literal["mean", "max"] = "mean") -> "OutputSpec":
        """Build a pooled-output specification.

        Parameters
        ----------
        pooling : {"mean", "max"}
            Pooling reducer.

        Returns
        -------
        OutputSpec
            Output specification with ``mode="pooled"``.
        """
        return OutputSpec(mode="pooled", scale_m=10, pooling=pooling, grid_orientation="north_up")

@dataclass(frozen=True)
class InputPrepSpec:
    """Policy controlling API-side preprocessing for large on-the-fly inputs.

    Attributes
    ----------
    mode : {"auto", "resize", "tile"}
        Preprocessing strategy.
    tile_size : int or None
        Tile edge length for tile-based modes.
    tile_stride : int or None
        Tile stride for tile-based modes.
    max_tiles : int
        Maximum number of tiles to process.
    pad_edges : bool
        If ``True``, pad boundary tiles to preserve shape.
    """

    mode: Literal["auto", "resize", "tile"] = "resize"
    tile_size: int | None = None
    tile_stride: int | None = None
    max_tiles: int = 9
    pad_edges: bool = True

    @staticmethod
    def auto(
        *,
        tile_size: int | None = None,
        tile_stride: int | None = None,
        max_tiles: int = 9,
        pad_edges: bool = True,
    ) -> "InputPrepSpec":
        """Build an adaptive preprocessing policy.

        Parameters
        ----------
        tile_size : int or None
            Optional tile edge length.
        tile_stride : int or None
            Optional tile stride.
        max_tiles : int
            Maximum number of tiles to process.
        pad_edges : bool
            If ``True``, pad edge tiles.

        Returns
        -------
        InputPrepSpec
            Spec with ``mode="auto"``.
        """
        return InputPrepSpec(
            mode="auto",
            tile_size=tile_size,
            tile_stride=tile_stride,
            max_tiles=max_tiles,
            pad_edges=pad_edges,
        )

    @staticmethod
    def resize() -> "InputPrepSpec":
        """Build a resize-only preprocessing policy.

        Returns
        -------
        InputPrepSpec
            Spec with ``mode="resize"``.
        """
        return InputPrepSpec(mode="resize")

    @staticmethod
    def tile(
        *,
        tile_size: int | None = None,
        tile_stride: int | None = None,
        max_tiles: int = 9,
        pad_edges: bool = True,
    ) -> "InputPrepSpec":
        """Build a tile-based preprocessing policy.

        Parameters
        ----------
        tile_size : int or None
            Optional tile edge length.
        tile_stride : int or None
            Optional tile stride.
        max_tiles : int
            Maximum number of tiles to process.
        pad_edges : bool
            If ``True``, pad edge tiles.

        Returns
        -------
        InputPrepSpec
            Spec with ``mode="tile"``.
        """
        return InputPrepSpec(
            mode="tile",
            tile_size=tile_size,
            tile_stride=tile_stride,
            max_tiles=max_tiles,
            pad_edges=pad_edges,
        )

import pytest

from rs_embed.core.errors import SpecError
from rs_embed.core.specs import BBox, InputPrepSpec, OutputSpec, PointBuffer, SensorSpec, TemporalSpec


# ══════════════════════════════════════════════════════════════════════
# BBox
# ══════════════════════════════════════════════════════════════════════

def test_bbox_validate_ok():
    bbox = BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0)
    bbox.validate()


def test_bbox_validate_invalid_bounds():
    bbox = BBox(minlon=1.0, minlat=0.0, maxlon=0.0, maxlat=1.0)
    with pytest.raises(SpecError, match="Invalid bbox bounds"):
        bbox.validate()


def test_bbox_validate_equal_lon():
    bbox = BBox(minlon=1.0, minlat=0.0, maxlon=1.0, maxlat=1.0)
    with pytest.raises(SpecError, match="Invalid bbox bounds"):
        bbox.validate()


def test_bbox_validate_equal_lat():
    bbox = BBox(minlon=0.0, minlat=1.0, maxlon=1.0, maxlat=1.0)
    with pytest.raises(SpecError, match="Invalid bbox bounds"):
        bbox.validate()


def test_bbox_validate_non_4326_crs():
    bbox = BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0, crs="EPSG:3857")
    with pytest.raises(SpecError, match="EPSG:4326"):
        bbox.validate()


@pytest.mark.parametrize("obj, attr", [
    (BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0), "minlon"),
    (PointBuffer(lon=1.0, lat=2.0, buffer_m=100.0), "lon"),
    (TemporalSpec.year(2024), "year"),
    (OutputSpec.pooled(), "mode"),
    (SensorSpec(collection="C", bands=("B1",)), "collection"),
    (InputPrepSpec.resize(), "mode"),
])
def test_specs_are_frozen(obj, attr):
    with pytest.raises(AttributeError):
        setattr(obj, attr, "x")


# ══════════════════════════════════════════════════════════════════════
# PointBuffer
# ══════════════════════════════════════════════════════════════════════

def test_pointbuffer_validate_ok():
    pb = PointBuffer(lon=1.0, lat=2.0, buffer_m=100.0)
    pb.validate()


def test_pointbuffer_validate_zero_buffer():
    pb = PointBuffer(lon=1.0, lat=2.0, buffer_m=0.0)
    with pytest.raises(SpecError, match="buffer_m must be positive"):
        pb.validate()


def test_pointbuffer_validate_negative_buffer():
    pb = PointBuffer(lon=1.0, lat=2.0, buffer_m=-50.0)
    with pytest.raises(SpecError, match="buffer_m must be positive"):
        pb.validate()


def test_pointbuffer_validate_non_4326_crs():
    pb = PointBuffer(lon=1.0, lat=2.0, buffer_m=100.0, crs="EPSG:3857")
    with pytest.raises(SpecError, match="EPSG:4326"):
        pb.validate()




# ══════════════════════════════════════════════════════════════════════
# TemporalSpec
# ══════════════════════════════════════════════════════════════════════

def test_temporal_spec_year():
    ts = TemporalSpec.year(2024)
    ts.validate()
    assert ts.mode == "year"
    assert ts.year == 2024


def test_temporal_spec_range():
    ts = TemporalSpec.range("2022-01-01", "2022-06-01")
    ts.validate()
    assert ts.mode == "range"
    assert ts.start == "2022-01-01"
    assert ts.end == "2022-06-01"


def test_temporal_spec_year_missing():
    ts = TemporalSpec(mode="year", year=None)
    with pytest.raises(SpecError, match="requires year"):
        ts.validate()


def test_temporal_spec_range_missing_start():
    ts = TemporalSpec(mode="range", start=None, end="2022-06-01")
    with pytest.raises(SpecError, match="requires start and end"):
        ts.validate()


def test_temporal_spec_range_missing_end():
    ts = TemporalSpec(mode="range", start="2022-01-01", end=None)
    with pytest.raises(SpecError, match="requires start and end"):
        ts.validate()


def test_temporal_spec_invalid_mode():
    ts = TemporalSpec(mode="oops")
    with pytest.raises(SpecError, match="Unknown TemporalSpec mode"):
        ts.validate()


def test_temporal_spec_range_bad_date_format():
    ts = TemporalSpec.range("2022/01/01", "2022-06-01")
    with pytest.raises(SpecError, match="ISO dates"):
        ts.validate()


def test_temporal_spec_range_start_must_be_before_end():
    ts = TemporalSpec.range("2022-06-01", "2022-06-01")
    with pytest.raises(SpecError, match="start < end"):
        ts.validate()


def test_temporal_spec_year_out_of_range():
    ts = TemporalSpec.year(0)
    with pytest.raises(SpecError, match="\\[1, 9999\\]"):
        ts.validate()




# ══════════════════════════════════════════════════════════════════════
# OutputSpec
# ══════════════════════════════════════════════════════════════════════

def test_output_spec_grid():
    grid = OutputSpec.grid(scale_m=20)
    assert grid.mode == "grid"
    assert grid.scale_m == 20


def test_output_spec_pooled_mean():
    pooled = OutputSpec.pooled()
    assert pooled.mode == "pooled"
    assert pooled.pooling == "mean"


def test_output_spec_pooled_max():
    pooled = OutputSpec.pooled(pooling="max")
    assert pooled.pooling == "max"


def test_output_spec_grid_default_scale():
    grid = OutputSpec.grid()
    assert grid.scale_m == 10




# ══════════════════════════════════════════════════════════════════════
# SensorSpec
# ══════════════════════════════════════════════════════════════════════

def test_sensor_spec_defaults():
    s = SensorSpec(collection="COPERNICUS/S2_SR_HARMONIZED", bands=("B4", "B3", "B2"))
    assert s.scale_m == 10
    assert s.cloudy_pct == 30
    assert s.fill_value == 0.0
    assert s.composite == "median"
    assert s.check_input is False
    assert s.check_raise is True
    assert s.check_save_dir is None


def test_sensor_spec_custom():
    s = SensorSpec(
        collection="C",
        bands=("B1",),
        scale_m=20,
        cloudy_pct=10,
        fill_value=-9999.0,
        composite="mosaic",
        check_input=True,
        check_raise=False,
        check_save_dir="/tmp/out",
    )
    assert s.scale_m == 20
    assert s.fill_value == -9999.0
    assert s.check_input is True
    assert s.check_raise is False
    assert s.check_save_dir == "/tmp/out"


# ══════════════════════════════════════════════════════════════════════
# InputPrepSpec
# ══════════════════════════════════════════════════════════════════════

def test_input_prep_spec_resize_defaults():
    s = InputPrepSpec.resize()
    assert s.mode == "resize"
    assert s.tile_size is None
    assert s.tile_stride is None
    assert s.max_tiles == 9
    assert s.pad_edges is True


def test_input_prep_spec_tile_defaults():
    s = InputPrepSpec.tile(tile_size=224)
    assert s.mode == "tile"
    assert s.tile_size == 224
    assert s.tile_stride is None
    assert s.max_tiles == 9
    assert s.pad_edges is True


def test_input_prep_spec_tile_custom():
    s = InputPrepSpec.tile(tile_size=128, tile_stride=64, max_tiles=4, pad_edges=False)
    assert s.tile_size == 128
    assert s.tile_stride == 64
    assert s.max_tiles == 4
    assert s.pad_edges is False


def test_input_prep_spec_auto_defaults():
    s = InputPrepSpec.auto()
    assert s.mode == "auto"
    assert s.tile_size is None
    assert s.tile_stride is None
    assert s.max_tiles == 9
    assert s.pad_edges is True


def test_input_prep_spec_auto_custom():
    s = InputPrepSpec.auto(tile_size=256, max_tiles=16, pad_edges=False)
    assert s.tile_size == 256
    assert s.max_tiles == 16
    assert s.pad_edges is False

from rs_embed.core.specs import TemporalSpec
from rs_embed.tools.temporal import split_temporal_range, temporal_frame_midpoints
from rs_embed.embedders.meta_utils import temporal_to_dict, temporal_to_range


def test_temporal_to_range_year_is_end_exclusive():
    t = temporal_to_range(TemporalSpec.year(2022))
    assert t.mode == "range"
    assert t.start == "2022-01-01"
    assert t.end == "2023-01-01"


def test_temporal_to_dict_year_matches_end_exclusive_convention():
    d = temporal_to_dict(TemporalSpec.year(2022))
    assert d["mode"] == "year"
    assert d["start"] == "2022-01-01"
    assert d["end"] == "2023-01-01"


def test_split_temporal_range_keeps_requested_frame_count():
    bins = split_temporal_range(TemporalSpec.range("2020-06-01", "2020-06-09"), 4)
    assert len(bins) == 4
    assert bins == (
        ("2020-06-01", "2020-06-03"),
        ("2020-06-03", "2020-06-05"),
        ("2020-06-05", "2020-06-07"),
        ("2020-06-07", "2020-06-09"),
    )


def test_temporal_frame_midpoints_follow_split_bins():
    mids = temporal_frame_midpoints(TemporalSpec.range("2020-06-01", "2020-06-09"), 4)
    assert mids == ("2020-06-02", "2020-06-04", "2020-06-06", "2020-06-08")

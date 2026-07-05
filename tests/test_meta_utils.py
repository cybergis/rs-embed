from rs_embed.core.specs import TemporalSpec
from rs_embed.embedders.meta import temporal_to_dict, temporal_to_range
from rs_embed.tools.temporal import split_temporal_range, temporal_frame_midpoints


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


def test_temporal_to_range_none_warns_about_default_window():
    """The silent hardcoded 2022 summer window must be visible.

    Regression: temporal=None quietly embedded 2022-06..09 imagery while the
    meta recorded {"mode": None} - the actual window was unrecorded and the
    result irreproducible.
    """
    import warnings as _w

    import pytest

    from rs_embed.core.specs import TemporalSpec
    from rs_embed.embedders.meta import temporal_to_range

    with pytest.warns(UserWarning, match="2022-06-01..2022-09-01"):
        t = temporal_to_range(None)
    assert (t.start, t.end) == ("2022-06-01", "2022-09-01")

    # Explicit temporal stays silent.
    with _w.catch_warnings():
        _w.simplefilter("error")
        t2 = temporal_to_range(TemporalSpec.year(2021))
    assert (t2.start, t2.end) == ("2021-01-01", "2022-01-01")

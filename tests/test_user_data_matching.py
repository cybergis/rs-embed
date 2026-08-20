"""Tests for the user-data declaration + sensor matching layer.

Pure unit tests over ``core.types.UserData`` and ``tools.user_data``; no
provider access, no model weights.
"""

import numpy as np
import pytest

from rs_embed.core.errors import ModelError, SpecError
from rs_embed.core.specs import SensorSpec
from rs_embed.core.types import UserData
from rs_embed.providers.prefetch_plan import select_prefetched_channels
from rs_embed.tools.user_data import (
    canonical_band_names,
    match_user_data_to_sensor,
    normalize_collection_id,
    warn_on_suspicious_value_range,
)

S2 = "COPERNICUS/S2_SR_HARMONIZED"


def _user_data(bands, *, collection=S2, shape_hw=(4, 4), tchw=False, fill=None):
    c = len(bands)
    if fill is None:
        # channel i is constant i, so tests can assert selection order by value
        base = np.stack([np.full(shape_hw, i, dtype=np.float32) for i in range(c)])
    else:
        base = np.full((c, *shape_hw), fill, dtype=np.float32)
    data = np.repeat(base[None, ...], 2, axis=0) if tchw else base
    return UserData(data=data, collection=collection, bands=tuple(bands))


def _sensor(bands, *, collection=S2):
    return SensorSpec(collection=collection, bands=tuple(bands))


# ── UserData.validate ──────────────────────────────────────────────


def test_validate_accepts_chw_and_tchw():
    _user_data(["B2", "B3"]).validate()
    _user_data(["B2", "B3"], tchw=True).validate()


def test_validate_rejects_channel_band_mismatch():
    bad = UserData(data=np.zeros((3, 4, 4), dtype=np.float32), collection=S2, bands=("B2", "B3"))
    with pytest.raises(SpecError, match="3 channels"):
        bad.validate()


def test_validate_rejects_bad_ndim_and_empty_fields():
    with pytest.raises(SpecError, match=r"\[C,H,W\]"):
        UserData(data=np.zeros((4, 4)), collection=S2, bands=("B2",)).validate()
    with pytest.raises(SpecError, match="collection"):
        UserData(data=np.zeros((1, 4, 4)), collection="  ", bands=("B2",)).validate()
    with pytest.raises(SpecError, match="bands"):
        UserData(data=np.zeros((1, 4, 4)), collection=S2, bands=()).validate()
    with pytest.raises(SpecError, match="scale_m"):
        UserData(data=np.zeros((1, 4, 4)), collection=S2, bands=("B2",), scale_m=0).validate()


# ── collection + band normalization ───────────────────────────────


def test_collection_aliases_resolve_and_full_ids_pass_through():
    assert normalize_collection_id("s2") == S2
    assert normalize_collection_id("Sentinel-2") == S2
    assert normalize_collection_id("s2-l2a") == S2
    assert normalize_collection_id("s1") == "COPERNICUS/S1_GRD"
    assert normalize_collection_id(S2) == S2
    assert normalize_collection_id("SOME/OTHER/COLLECTION") == "SOME/OTHER/COLLECTION"


def test_canonical_band_names_resolve_aliases_and_case():
    assert canonical_band_names(S2, ("RED", "green", "b2")) == ("B4", "B3", "B2")
    assert canonical_band_names(S2, ("SWIR_1", "NIR_NARROW")) == ("B11", "B8A")


# ── matching ───────────────────────────────────────────────────────


def test_superset_data_is_sliced_into_model_band_order():
    data = _user_data(["B1", "B2", "B3", "B4", "B8"])
    idx = match_user_data_to_sensor(data, _sensor(["B4", "B3", "B2"]), model_name="m")
    assert idx == (3, 2, 1)
    sliced = select_prefetched_channels(data.data, idx)
    assert sliced.shape == (3, 4, 4)
    assert [float(sliced[i, 0, 0]) for i in range(3)] == [3.0, 2.0, 1.0]


def test_exact_match_is_identity():
    data = _user_data(["B2", "B3", "B4"])
    idx = match_user_data_to_sensor(data, _sensor(["B2", "B3", "B4"]), model_name="m")
    assert idx == (0, 1, 2)


def test_tchw_slicing_selects_channel_axis():
    data = _user_data(["B1", "B2", "B3", "B4"], tchw=True)
    idx = match_user_data_to_sensor(data, _sensor(["B4", "B2"]), model_name="m")
    sliced = select_prefetched_channels(data.data, idx)
    assert sliced.shape == (2, 2, 4, 4)
    assert [float(sliced[0, i, 0, 0]) for i in range(2)] == [3.0, 1.0]


def test_alias_bands_match_model_alias_bands():
    # Prithvi-style: model declares HLS-style names (NIR_NARROW -> B8A), user
    # declares S2 names; both sides canonicalize to the same vocabulary.
    data = _user_data(["B2", "B3", "B4", "B8A", "B11", "B12"])
    sensor = _sensor(["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"])
    idx = match_user_data_to_sensor(data, sensor, model_name="m")
    assert idx == (0, 1, 2, 3, 4, 5)


def test_alias_bands_refuse_when_canonical_band_missing():
    # NIR_NARROW canonicalizes to B8A; broad-NIR B8 does not satisfy it.
    data = _user_data(["B2", "B3", "B4", "B8", "B11", "B12"])
    sensor = _sensor(["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"])
    with pytest.raises(ModelError, match=r"lacks \['B8A'\]"):
        match_user_data_to_sensor(data, sensor, model_name="m")


def test_missing_band_refuses_with_names():
    data = _user_data(["B2", "B3", "B4"])
    with pytest.raises(ModelError, match=r"lacks \['B8'\]"):
        match_user_data_to_sensor(data, _sensor(["B2", "B3", "B4", "B8"]), model_name="m")


def test_collection_mismatch_refuses():
    data = _user_data(["VV", "VH"], collection="s1")
    with pytest.raises(ModelError, match="collection"):
        match_user_data_to_sensor(data, _sensor(["B2", "B3"]), model_name="m")


def test_duplicate_user_bands_refuse():
    data = UserData(
        data=np.zeros((3, 4, 4), dtype=np.float32),
        collection=S2,
        bands=("B2", "RED", "B4"),  # RED aliases to B4 -> duplicate
    )
    with pytest.raises(ModelError, match="duplicate"):
        match_user_data_to_sensor(data, _sensor(["B2", "B4"]), model_name="m")


# ── value-range warning ────────────────────────────────────────────


def test_normalized_looking_s2_values_warn():
    data = _user_data(["B2", "B3"], fill=0.3)
    with pytest.warns(UserWarning, match="0..10000"):
        warn_on_suspicious_value_range(data)


def test_raw_dn_s2_values_do_not_warn():
    data = _user_data(["B2", "B3"], fill=4321.0)
    with warnings_disabled_check():
        warn_on_suspicious_value_range(data)


def test_non_s2_collections_never_warn():
    data = _user_data(["VV", "VH"], collection="s1", fill=0.02)
    with warnings_disabled_check():
        warn_on_suspicious_value_range(data)


class warnings_disabled_check:
    """Context asserting no UserWarning was emitted inside the block."""

    def __enter__(self):
        import warnings as _warnings

        self._catcher = _warnings.catch_warnings(record=True)
        self._records = self._catcher.__enter__()
        _warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc, tb):
        self._catcher.__exit__(exc_type, exc, tb)
        assert not [w for w in self._records if issubclass(w.category, UserWarning)]
        return False

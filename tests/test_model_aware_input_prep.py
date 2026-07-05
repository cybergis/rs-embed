import numpy as np
import pytest

import rs_embed.api as api
from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import InputPrepSpec, OutputSpec, PointBuffer, TemporalSpec

_SPATIAL = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
_TEMPORAL = TemporalSpec.range("2022-06-01", "2022-09-01")


class _FakeImageLevelVitGridEmbedder:
    def describe(self):
        return {"type": "mock", "output": ["pooled", "grid"]}

    def get_embedding(
        self,
        *,
        spatial,
        temporal,
        sensor,
        output,
        backend,
        device="auto",
        input_chw=None,
    ):
        if output.mode == "grid":
            return Embedding(data=np.zeros((2, 2, 2), dtype=np.float32), meta={})
        return Embedding(data=np.zeros((2,), dtype=np.float32), meta={})


def _with_registered_fake_model(model_id):
    previous = registry._REGISTRY.get(model_id)
    registry.register(model_id)(_FakeImageLevelVitGridEmbedder)
    api.reset_runtime()
    return previous


def _restore_registered_model(model_id, previous):
    if previous is None:
        registry._REGISTRY.pop(model_id, None)
    else:
        registry._REGISTRY[model_id] = previous
    api.reset_runtime()


@pytest.mark.parametrize(
    "model_id",
    ["satmae", "satmaepp", "scalemae"],
)
@pytest.mark.parametrize(
    ("input_prep", "requested_mode"),
    [(None, "default"), (InputPrepSpec.auto(), "auto")],
)
def test_image_level_vit_grid_default_input_prep_resolves_to_tile_with_metadata(
    model_id,
    input_prep,
    requested_mode,
):
    previous = _with_registered_fake_model(model_id)
    try:
        with pytest.warns(UserWarning, match="can show seams"):
            emb = api.get_embedding(
                model_id,
                spatial=_SPATIAL,
                temporal=_TEMPORAL,
                output=OutputSpec.grid(),
                backend="local",
                input_prep=input_prep,
            )
    finally:
        _restore_registered_model(model_id, previous)

    prep = emb.meta["input_prep"]
    assert prep["requested_mode"] == requested_mode
    assert prep["resolved_mode"] == "tile"
    assert prep["model_policy"] == "tile_default_for_image_level_vit_patch_grid"
    assert prep["resolved_by_model_policy"] is True
    assert prep["tiled_grid_seam_risk"] == "high"
    assert prep["tiled_grid_recommended"] is False
    assert emb.meta["grid_semantics"] == "vit_patch_tokens"
    assert emb.meta["grid_tile_recommended"] is False
    assert emb.meta["preferred_output"] == "pooled"


@pytest.mark.parametrize(
    "model_id",
    ["satmae", "satmaepp", "scalemae"],
)
def test_image_level_vit_grid_explicit_tile_is_allowed_and_warns_about_seams(model_id):
    previous = _with_registered_fake_model(model_id)
    try:
        with pytest.warns(UserWarning, match="can show seams"):
            emb = api.get_embedding(
                model_id,
                spatial=_SPATIAL,
                temporal=_TEMPORAL,
                output=OutputSpec.grid(),
                backend="local",
                input_prep=InputPrepSpec.tile(tile_size=224),
            )
    finally:
        _restore_registered_model(model_id, previous)

    prep = emb.meta["input_prep"]
    assert prep["requested_mode"] == "tile"
    assert prep["resolved_mode"] == "tile"
    assert prep["model_policy"] == "explicit_tile_for_image_level_vit_patch_grid"
    assert prep["resolved_by_model_policy"] is False
    assert prep["tiled_grid_seam_risk"] == "high"
    assert prep["tiled_grid_recommended"] is False
    assert emb.meta["grid_semantics"] == "vit_patch_tokens"
    assert emb.meta["grid_tile_recommended"] is False


def test_input_prep_spec_bare_construction_defaults_to_tile():
    """InputPrepSpec(tile_size=512) must mean tiling with that size.

    Regression: the dataclass default was mode='resize' while the unset
    (None) default resolved to tile - two different defaults for the same
    concept, so this natural construction silently resized and ignored
    tile_size.
    """
    from rs_embed.tools.tiling import _resolve_input_prep_spec

    spec = InputPrepSpec(tile_size=512)
    resolved = _resolve_input_prep_spec(spec)
    assert resolved.mode == "tile"
    assert resolved.tile_size == 512
    # And the two defaults agree.
    assert InputPrepSpec().mode == _resolve_input_prep_spec(None).mode == "tile"


def test_vit_grid_model_policy_preserves_user_auto_fields():
    """The satmae/scalemae tile policy must override only the mode.

    Regression: the policy replaced the user's auto spec wholesale with
    InputPrepSpec.tile() defaults, dropping tile_size/max_tiles/....
    """
    from rs_embed.tools.runtime import resolve_model_aware_input_prep

    user = InputPrepSpec.auto(tile_size=512, max_tiles=16, tile_snap_frac=0.0)
    effective, resolved, requested_mode, policy = resolve_model_aware_input_prep(
        model_n="satmae",
        input_prep=user,
        output=OutputSpec.pooled(),
    )
    assert policy == "tile_default_for_image_level_vit_patch_grid"
    assert requested_mode == "auto"
    assert resolved.mode == "tile"
    assert resolved.tile_size == 512
    assert resolved.max_tiles == 16
    assert resolved.tile_snap_frac == 0.0

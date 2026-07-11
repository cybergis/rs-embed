"""Contract tests for explicit embedder capability declarations.

Routing decisions (prefetched inputs, fetch metadata, model_config forwarding)
are driven by each embedder class's ``capabilities`` declaration; signature
introspection remains only as a fallback for third-party subclasses.  These
tests keep the declarations honest:

1. every in-tree (catalog-registered) embedder must declare ``capabilities``;
2. every declared field must match what the method signatures actually accept,
   so a signature change without a matching declaration update fails loudly
   instead of silently rerouting requests.
"""

from __future__ import annotations

import inspect

import pytest

from rs_embed.core.registry import get_embedder_cls
from rs_embed.core.types import (
    CAPABILITY_FIELD_BY_METHOD_PARAM,
    EmbedderCapabilities,
    declared_capability,
)
from rs_embed.embedders.catalog import MODEL_SPECS


def _signature_accepts(cls: type, method_name: str, param_name: str) -> bool:
    """Reference introspection: mirrors the runtime fallback logic."""
    fn = getattr(cls, method_name, None)
    if fn is None:
        return False
    sig = inspect.signature(fn)
    if param_name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


@pytest.fixture(scope="module", params=sorted(MODEL_SPECS))
def embedder_cls(request):
    return get_embedder_cls(request.param)


def test_all_registered_embedders_declare_capabilities(embedder_cls):
    caps = embedder_cls.__dict__.get("capabilities")
    assert isinstance(caps, EmbedderCapabilities), (
        f"{embedder_cls.__name__} must declare its own `capabilities = "
        "EmbedderCapabilities(...)` class attribute (inheriting the base "
        "default is not enough — routing would fall back to signature "
        "introspection)."
    )


def test_declared_capabilities_match_signatures(embedder_cls):
    mismatches = []
    for (method_name, param_name), field in sorted(CAPABILITY_FIELD_BY_METHOD_PARAM.items()):
        declared = declared_capability(embedder_cls, method_name, param_name)
        actual = _signature_accepts(embedder_cls, method_name, param_name)
        if declared is not None and declared != actual:
            mismatches.append(
                f"{field}: declared={declared} but {method_name}({param_name}=...) accepts={actual}"
            )
    assert not mismatches, (
        f"{embedder_cls.__name__} capability declaration drifted from its "
        "method signatures:\n  " + "\n  ".join(mismatches)
    )


def test_satmaepp_s2_delegate_declares_matching_capabilities():
    """The unregistered S2 delegate is also an EmbedderBase entry point."""
    from rs_embed.embedders.onthefly_satmaepp_s2 import SatMAEPPSentinel10Embedder

    caps = SatMAEPPSentinel10Embedder.__dict__.get("capabilities")
    assert isinstance(caps, EmbedderCapabilities)
    for (method_name, param_name), _field in CAPABILITY_FIELD_BY_METHOD_PARAM.items():
        declared = declared_capability(SatMAEPPSentinel10Embedder, method_name, param_name)
        assert declared == _signature_accepts(
            SatMAEPPSentinel10Embedder, method_name, param_name
        ), (method_name, param_name)


# ── model-policy flags (replaced hardcoded model-name lists) ────────


def test_image_level_vit_grid_flag_matches_expected_models():
    """Behavior parity with the removed _IMAGE_LEVEL_VIT_GRID_MODELS set."""
    flagged = {
        model_id
        for model_id in MODEL_SPECS
        if getattr(get_embedder_cls(model_id), "_image_level_vit_patch_grid", False)
    }
    assert flagged == {"satmae", "satmaepp", "scalemae"}


def test_manages_own_input_prep_flag_matches_expected_models():
    """Behavior parity with the removed api-level 'gse' hardcode."""
    flagged = {
        model_id
        for model_id in MODEL_SPECS
        if getattr(get_embedder_cls(model_id), "_manages_own_input_prep", False)
    }
    assert flagged == {"gse"}


def test_tiled_dispatch_hook_only_overridden_by_thor():
    """Behavior parity with the removed tiling-level 'thor' hardcode."""
    from rs_embed.embedders.base import EmbedderBase

    overriding = {
        model_id
        for model_id in MODEL_SPECS
        if get_embedder_cls(model_id).tiled_dispatch_model_config
        is not EmbedderBase.tiled_dispatch_model_config
    }
    assert overriding == {"thor"}

    thor_cls = get_embedder_cls("thor")
    cfg = thor_cls().tiled_dispatch_model_config({"variant": "base"}, tile_size=288)
    assert cfg == {
        "variant": "base",
        "_input_prep_mode": "tile",
        "_input_prep_tile_size": 288,
    }


def test_declaration_overrides_signature_for_routing():
    """Routing must trust the declaration, not the signature, when declared."""
    from rs_embed.embedders.base import EmbedderBase
    from rs_embed.tools import runtime as rt

    class _DeclaredNoKwargs(EmbedderBase):
        model_name = "declared_no_kwargs"
        capabilities = EmbedderCapabilities()  # everything False

        def get_embedding(self, **kwargs):  # signature says "accepts anything"
            raise NotImplementedError

    class _UndeclaredKwargs(EmbedderBase):
        model_name = "undeclared_kwargs"

        def get_embedding(self, **kwargs):
            raise NotImplementedError

    rt._embedder_method_accepts_parameter.cache_clear()
    # Declared False wins over the **kwargs signature.
    assert rt.embedder_accepts_input_chw(_DeclaredNoKwargs) is False
    # Undeclared classes keep the introspection fallback.
    assert rt.embedder_accepts_input_chw(_UndeclaredKwargs) is True
    rt._embedder_method_accepts_parameter.cache_clear()

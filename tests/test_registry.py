import sys
import types

import pytest

from rs_embed.core import registry
from rs_embed.embedders import catalog

# ── fixture to isolate registry between tests ──────────────────────


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after every test in this module."""
    registry._REGISTRY.clear()
    if hasattr(registry, "_REGISTRY_IMPORT_ERRORS"):
        registry._REGISTRY_IMPORT_ERRORS.clear()
    yield
    registry._REGISTRY.clear()
    if hasattr(registry, "_REGISTRY_IMPORT_ERRORS"):
        registry._REGISTRY_IMPORT_ERRORS.clear()


def _install_fake_lazy_model(
    monkeypatch,
    *,
    model_id: str,
    module_name: str,
    class_name: str,
    alias: str | None = None,
):
    monkeypatch.setitem(catalog.MODEL_SPECS, model_id, (module_name, class_name))
    if alias is not None:
        monkeypatch.setitem(catalog.MODEL_ALIASES, alias, model_id)

    fqmn = f"rs_embed.embedders.{module_name}"
    mod = types.ModuleType(fqmn)
    cls = type(class_name, (), {})
    setattr(mod, class_name, cls)
    monkeypatch.setitem(sys.modules, fqmn, mod)
    return fqmn, cls


# ══════════════════════════════════════════════════════════════════════
# register + get_embedder_cls
# ══════════════════════════════════════════════════════════════════════


def test_register_and_get_embedder_cls():
    @registry.register("TestModel")
    class DummyEmbedder:
        pass

    assert DummyEmbedder.model_name == "testmodel"
    assert registry.get_embedder_cls("testmodel") is DummyEmbedder
    assert registry.get_embedder_cls("TESTMODEL") is DummyEmbedder
    assert "testmodel" in registry.list_models()


def test_get_embedder_cls_missing():
    from rs_embed.core.errors import ModelError

    with pytest.raises(ModelError, match="Unknown model"):
        registry.get_embedder_cls("missing-model")


# ── case insensitivity ─────────────────────────────────────────────


def test_register_case_insensitive():
    @registry.register("MiXeD_CaSe")
    class M:
        pass

    assert registry.get_embedder_cls("mixed_case") is M
    assert registry.get_embedder_cls("MIXED_CASE") is M


# ── multiple registrations ─────────────────────────────────────────


def test_register_multiple():
    @registry.register("alpha")
    class A:
        pass

    @registry.register("beta")
    class B:
        pass

    assert registry.get_embedder_cls("alpha") is A
    assert registry.get_embedder_cls("beta") is B
    assert registry.list_models() == ["alpha", "beta"]


# ── overwrite same name ────────────────────────────────────────────


def test_register_overwrite():
    @registry.register("dup")
    class First:
        pass

    @registry.register("dup")
    class Second:
        pass

    assert registry.get_embedder_cls("dup") is Second


# ── empty registry ─────────────────────────────────────────────────


def test_list_models_empty():
    assert registry.list_models() == []


def test_get_embedder_cls_empty_shows_available():
    from rs_embed.core.errors import ModelError

    with pytest.raises(ModelError, match="Available: \\[\\]"):
        registry.get_embedder_cls("anything")


def test_get_embedder_cls_includes_last_import_error(monkeypatch):
    from rs_embed.core.errors import ModelError

    registry._REGISTRY.clear()
    registry._REGISTRY_IMPORT_ERRORS["anything"] = RuntimeError("boom")

    with pytest.raises(ModelError) as ei:
        registry.get_embedder_cls("anything")
    msg = str(ei.value)
    assert "Import error for 'anything'" in msg
    assert "RuntimeError: boom" in msg


def test_get_embedder_cls_lazy_imports_builtin_without_bulk_package_import(monkeypatch):
    fqmn, cls_expected = _install_fake_lazy_model(
        monkeypatch,
        model_id="fake_lazy",
        module_name="onthefly_test_lazy",
        class_name="FakeLazyEmbedder",
    )
    calls = []
    orig_import_module = registry.importlib.import_module

    def _spy(name, *args, **kwargs):
        calls.append(name)
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(registry.importlib, "import_module", _spy)

    cls = registry.get_embedder_cls("fake_lazy")
    assert cls is cls_expected
    assert "fake_lazy" in registry.list_models()
    assert "rs_embed.embedders" not in calls
    assert fqmn in calls


def test_get_embedder_cls_cleans_failed_lazy_import_modules(monkeypatch):
    from rs_embed.core.errors import ModelError

    model_id = "fake_lazy_fail"
    module_name = "onthefly_test_lazy_fail"
    class_name = "FakeLazyFailEmbedder"
    monkeypatch.setitem(catalog.MODEL_SPECS, model_id, (module_name, class_name))
    fqmn = f"rs_embed.embedders.{module_name}"
    vendor_name = "rs_embed.embedders._vendor.fake_test_lazy_fail"
    orig_import_module = registry.importlib.import_module

    def _boom(name, *args, **kwargs):
        if name == fqmn:
            sys.modules[fqmn] = types.ModuleType(fqmn)
            sys.modules[vendor_name] = types.ModuleType(vendor_name)
            raise RuntimeError("boom")
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(registry.importlib, "import_module", _boom)

    with pytest.raises(ModelError, match="RuntimeError: boom"):
        registry.get_embedder_cls(model_id)

    assert fqmn not in sys.modules
    assert vendor_name not in sys.modules


def test_get_embedder_cls_accepts_legacy_alias(monkeypatch):
    _, cls_expected = _install_fake_lazy_model(
        monkeypatch,
        model_id="fake_alias_target",
        module_name="onthefly_test_lazy_alias",
        class_name="FakeAliasEmbedder",
        alias="fake_alias_old",
    )

    cls_new = registry.get_embedder_cls("fake_alias_target")
    cls_old = registry.get_embedder_cls("fake_alias_old")
    assert cls_new is cls_expected
    assert cls_old is cls_new


def test_get_embedder_cls_accepts_satmaepp_aliases():
    cls_new = registry.get_embedder_cls("satmaepp")
    cls_old = registry.get_embedder_cls("satmaepp_rgb")
    cls_pp = registry.get_embedder_cls("satmae++")
    assert cls_old is cls_new
    assert cls_pp is cls_new


def test_get_embedder_cls_can_reregister_when_registry_was_cleared(monkeypatch):
    _, cls_expected = _install_fake_lazy_model(
        monkeypatch,
        model_id="fake_reregister",
        module_name="onthefly_test_lazy_reregister",
        class_name="FakeReregisterEmbedder",
        alias="fake_reregister_old",
    )

    cls1 = registry.get_embedder_cls("fake_reregister")
    registry._REGISTRY.clear()
    cls2 = registry.get_embedder_cls("fake_reregister_old")
    assert cls1 is cls_expected
    assert cls2 is cls1
    assert registry.get_embedder_cls("fake_reregister") is cls1

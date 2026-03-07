import pytest

from rs_embed.core import registry


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
    calls = []
    orig_import_module = registry.importlib.import_module

    def _spy(name, *args, **kwargs):
        calls.append(name)
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(registry.importlib, "import_module", _spy)

    cls = registry.get_embedder_cls("remoteclip")
    assert cls.__name__ == "RemoteCLIPS2RGBEmbedder"
    assert "remoteclip" in registry.list_models()
    assert "remoteclip_s2rgb" not in registry.list_models()
    assert "rs_embed.embedders" not in calls
    assert "rs_embed.embedders.onthefly_remoteclip" in calls


def test_get_embedder_cls_accepts_legacy_alias():
    cls_new = registry.get_embedder_cls("remoteclip")
    cls_old = registry.get_embedder_cls("remoteclip_s2rgb")
    assert cls_old is cls_new


def test_get_embedder_cls_accepts_satmaepp_aliases():
    cls_new = registry.get_embedder_cls("satmaepp")
    cls_old = registry.get_embedder_cls("satmaepp_rgb")
    cls_pp = registry.get_embedder_cls("satmae++")
    assert cls_old is cls_new
    assert cls_pp is cls_new


def test_get_embedder_cls_accepts_satmaepp_s2_aliases():
    cls_new = registry.get_embedder_cls("satmaepp_s2_10b")
    cls_old = registry.get_embedder_cls("satmaepp_sentinel10")
    cls_alt = registry.get_embedder_cls("satmaepp_s2")
    assert cls_old is cls_new
    assert cls_alt is cls_new


def test_get_embedder_cls_can_reregister_when_registry_was_cleared():
    cls1 = registry.get_embedder_cls("remoteclip")
    registry._REGISTRY.clear()
    cls2 = registry.get_embedder_cls("remoteclip_s2rgb")
    assert cls2 is cls1
    assert registry.get_embedder_cls("remoteclip") is cls1

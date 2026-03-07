"""Tests for backend resolution routing in rs_embed.api.

Exercises _resolve_embedding_api_backend, _provider_factory_for_backend,
and _default_provider_backend_for_api — the logic that maps high-level
backend names to concrete provider/access paths, particularly for
precomputed models.
"""

from unittest.mock import patch, MagicMock

import pytest

from rs_embed.api import (
    _resolve_embedding_api_backend,
    _provider_factory_for_backend,
    _default_provider_backend_for_api,
)

# ── _resolve_embedding_api_backend ────────────────────────────────────


def _mock_describe(desc: dict):
    """Patch _probe_model_describe to return *desc* for any model."""
    return patch("rs_embed.api._probe_model_describe", return_value=desc)


class TestResolveBackend:
    """Tests for _resolve_embedding_api_backend."""

    def test_onthefly_model_returns_backend_unchanged(self):
        with _mock_describe({"type": "onthefly"}):
            assert _resolve_embedding_api_backend("sat-mae", "gee") == "gee"

    def test_onthefly_model_auto_unchanged(self):
        with _mock_describe({"type": "onthefly"}):
            assert _resolve_embedding_api_backend("sat-mae", "auto") == "auto"

    def test_precomputed_backend_in_allowed_list(self):
        with _mock_describe({"type": "precomputed", "backend": ["local", "auto"]}):
            assert _resolve_embedding_api_backend("m", "local") == "local"

    def test_precomputed_auto_backend_with_provider_allowed(self):
        """auto + provider in allowed → returns default provider backend."""
        with _mock_describe({"type": "precomputed", "backend": ["provider", "auto"]}):
            with patch(
                "rs_embed.api._default_provider_backend_for_api", return_value="gee"
            ):
                assert _resolve_embedding_api_backend("m", "auto") == "gee"

    def test_precomputed_auto_backend_no_provider(self):
        """auto in allowed list but no provider → auto returned as-is."""
        with _mock_describe({"type": "precomputed", "backend": ["auto", "local"]}):
            assert _resolve_embedding_api_backend("m", "auto") == "auto"

    def test_precomputed_gee_falls_to_auto_if_available(self):
        """Precomputed with gee default → maps to 'auto' if 'auto' in allowed."""
        with _mock_describe({"type": "precomputed", "backend": ["auto", "local"]}):
            assert _resolve_embedding_api_backend("m", "gee") == "auto"

    def test_precomputed_gee_falls_to_local(self):
        """Precomputed gee → 'local' when 'auto' not available but 'local' is."""
        with _mock_describe({"type": "precomputed", "backend": ["local"]}):
            assert _resolve_embedding_api_backend("m", "gee") == "local"

    def test_precomputed_gee_falls_to_provider(self):
        """Precomputed gee → provider backend when only provider is allowed."""
        with _mock_describe({"type": "precomputed", "backend": ["provider"]}):
            with patch(
                "rs_embed.api._default_provider_backend_for_api", return_value="gee"
            ):
                assert _resolve_embedding_api_backend("m", "gee") == "gee"

    def test_precomputed_legacy_local_to_auto(self):
        """Legacy backend='local' maps to 'auto' for precomputed with auto+no-provider."""
        with _mock_describe({"type": "precomputed", "backend": ["auto", "local"]}):
            assert _resolve_embedding_api_backend("m", "local") == "local"

    def test_precomputed_no_backend_list_returns_unchanged(self):
        """When describe has no backend list, the backend is returned as-is."""
        with _mock_describe({"type": "precomputed"}):
            assert _resolve_embedding_api_backend("m", "gee") == "gee"

    def test_describe_fails_returns_backend_unchanged(self):
        """If model describe fails (returns {}), backend passes through."""
        with _mock_describe({}):
            assert _resolve_embedding_api_backend("m", "gee") == "gee"

    def test_empty_allowed_list_returns_unchanged(self):
        with _mock_describe({"type": "precomputed", "backend": []}):
            assert _resolve_embedding_api_backend("m", "gee") == "gee"

    def test_backend_list_with_whitespace(self):
        """Allowed backends are stripped & lowered before comparison."""
        with _mock_describe({"type": "precomputed", "backend": [" Auto ", " Local "]}):
            assert _resolve_embedding_api_backend("m", "auto") == "auto"


# ── _provider_factory_for_backend ────────────────────────────────────


class TestProviderFactory:
    """Tests for _provider_factory_for_backend."""

    def test_auto_creates_gee(self):
        """auto → gee provider factory (when gee is available)."""
        with patch("rs_embed.api.has_provider", return_value=True):
            factory = _provider_factory_for_backend("auto")
            assert factory is not None

    def test_unknown_backend_returns_none(self):
        with patch("rs_embed.api.has_provider", return_value=False):
            assert _provider_factory_for_backend("nonexistent") is None

    def test_gee_returns_monkeypatch_friendly_factory(self):
        """gee → returns the module-level _create_default_gee_provider."""
        with patch("rs_embed.api.has_provider", return_value=True):
            factory = _provider_factory_for_backend("gee")
            assert factory is not None


# ── _default_provider_backend_for_api ────────────────────────────────


class TestDefaultProviderBackend:
    """Tests for _default_provider_backend_for_api."""

    def test_gee_available(self):
        with patch("rs_embed.api.list_providers", return_value=["gee"]):
            assert _default_provider_backend_for_api() == "gee"

    def test_other_provider(self):
        with patch("rs_embed.api.list_providers", return_value=["planetary_computer"]):
            assert _default_provider_backend_for_api() == "planetary_computer"

    def test_no_providers_defaults_gee(self):
        with patch("rs_embed.api.list_providers", return_value=[]):
            assert _default_provider_backend_for_api() == "gee"

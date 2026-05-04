"""Oracle Code Assist provider wiring."""

from __future__ import annotations

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    _OCA_MODEL_API_MODE_OVERRIDES,
    _PROVIDER_MODELS,
    normalize_provider,
    oca_model_api_mode,
    provider_model_ids,
)


@pytest.fixture(autouse=True)
def _clear_oca_env(monkeypatch):
    _OCA_MODEL_API_MODE_OVERRIDES.clear()
    for key in (
        "OCA_ACCESS_TOKEN",
        "OCA_API_KEY",
        "OCI_CODE_ASSIST_TOKEN",
        "OCA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_oca_provider_registry():
    pconfig = PROVIDER_REGISTRY["oca"]
    assert pconfig.name == "Oracle Code Assist"
    assert pconfig.auth_type == "api_key"
    assert pconfig.inference_base_url == (
        "https://code-internal.aiservice.us-chicago-1.oci.oraclecloud.com/20250206/app/litellm"
    )
    assert pconfig.api_key_env_vars == ("OCA_ACCESS_TOKEN", "OCA_API_KEY", "OCI_CODE_ASSIST_TOKEN")
    assert pconfig.base_url_env_var == "OCA_BASE_URL"


@pytest.mark.parametrize("alias", ["oca", "oracle", "oracle-code-assist", "oci-code-assist", "code-assist"])
def test_oca_aliases(alias, monkeypatch):
    monkeypatch.setenv("OCA_ACCESS_TOKEN", "oca-token")
    assert resolve_provider(alias) == "oca"
    assert normalize_provider(alias) == "oca"


def test_oca_credentials_from_token_env(monkeypatch):
    monkeypatch.setenv("OCA_API_KEY", "token-from-env")
    creds = resolve_api_key_provider_credentials("oca")
    assert creds["api_key"] == "token-from-env"
    assert creds["base_url"] == PROVIDER_REGISTRY["oca"].inference_base_url


def test_oca_base_url_override(monkeypatch):
    monkeypatch.setenv("OCA_ACCESS_TOKEN", "oca-token")
    monkeypatch.setenv("OCA_BASE_URL", "https://oca.example.com/litellm")
    creds = resolve_api_key_provider_credentials("oca")
    assert creds["base_url"] == "https://oca.example.com/litellm"


def test_oca_model_catalog_static_fallback():
    assert "oca" in _PROVIDER_MODELS
    assert "oca/gpt-oss-120b" in provider_model_ids("oca")
    assert "oca/gpt-4.1" in provider_model_ids("oca")
    assert "oca/gpt-5.5" in provider_model_ids("oca")


def test_oca_model_catalog_prefers_models_endpoint(monkeypatch):
    import httpx

    monkeypatch.setenv("OCA_ACCESS_TOKEN", "token-from-env")
    monkeypatch.setenv("OCA_BASE_URL", "https://oca.example.com/litellm")
    calls = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"id": "oca/llama4", "object": "model"},
                    {"id": "oca/gpt-4.1", "object": "model"},
                ]
            }

    def _get(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr(httpx, "get", _get)

    assert provider_model_ids("oca") == ["oca/llama4", "oca/gpt-4.1"]
    assert [call[0] for call in calls] == [
        "https://oca.example.com/litellm/v1/models",
        "https://oca.example.com/litellm/v1/model/info",
    ]
    assert calls[0][1]["headers"]["Authorization"] == "Bearer token-from-env"


def test_oca_model_catalog_falls_back_to_model_info(monkeypatch):
    import httpx

    monkeypatch.setenv("OCA_ACCESS_TOKEN", "token-from-env")
    monkeypatch.setenv("OCA_BASE_URL", "https://oca.example.com/litellm")
    calls = []

    class _FailingResponse:
        def raise_for_status(self):
            raise RuntimeError("models endpoint unavailable")

    class _InfoResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "litellm_params": {"max_tokens": 131072, "model": "oca/gpt-oss-120b"},
                        "model_info": {
                            "context_window": 131072,
                            "is_reasoning_model": True,
                            "reasoning_effort_options": ["low", "medium", "high"],
                            "supported_api_list": ["CHAT_COMPLETIONS"],
                        },
                        "model_name": "OpenAI GPT OSS 120b hosted by Oracle Code Assist",
                    },
                    {
                        "litellm_params": {"max_tokens": 524288, "model": "oca/llama4"},
                        "model_info": {
                            "context_window": 524288,
                            "supported_api_list": ["CHAT_COMPLETIONS"],
                        },
                        "model_name": "Llama4 hosted by Oracle Code Assist",
                    },
                    {
                        "litellm_params": {"max_tokens": 1050000, "model": "oca/gpt-5.5"},
                        "model_info": {
                            "context_window": 1050000,
                            "reasoning_effort_options": ["none", "low", "medium", "high", "xhigh"],
                            "supported_api_list": ["RESPONSES", "CHAT_COMPLETIONS"],
                        },
                        "model_name": "OpenAI GPT 5.5",
                    },
                ]
            }

    def _get(url, **kwargs):
        calls.append(url)
        if url.endswith("/v1/models"):
            return _FailingResponse()
        return _InfoResponse()

    monkeypatch.setattr(httpx, "get", _get)

    assert provider_model_ids("oca") == ["oca/gpt-oss-120b", "oca/llama4", "oca/gpt-5.5"]
    assert calls == [
        "https://oca.example.com/litellm/v1/models",
        "https://oca.example.com/litellm/v1/model/info",
    ]


def test_oca_model_api_mode_known_responses_capable_models():
    assert oca_model_api_mode("oca/gpt-5.3-codex") == "codex_responses"
    assert oca_model_api_mode("oca/gpt-5.5-pro") == "codex_responses"
    assert oca_model_api_mode("gpt-5.3-codex") == "codex_responses"
    assert oca_model_api_mode("gpt-5.5-pro") == "codex_responses"
    assert oca_model_api_mode("oca/gpt-5.5") == "codex_responses"
    assert oca_model_api_mode("gpt-5.5") == "codex_responses"
    assert oca_model_api_mode("oca/gpt-oss-120b") == "chat_completions"


def test_oca_model_api_mode_uses_live_supported_api_list():
    from hermes_cli.models import _extract_oca_model_ids

    _extract_oca_model_ids({
        "data": [
            {
                "litellm_params": {"model": "oca/custom-responses-model"},
                "model_info": {"supported_api_list": ["RESPONSES", "CHAT_COMPLETIONS"]},
            },
            {
                "litellm_params": {"model": "oca/custom-chat-model"},
                "model_info": {"supported_api_list": ["CHAT_COMPLETIONS"]},
            },
        ]
    })

    assert oca_model_api_mode("oca/custom-responses-model") == "codex_responses"
    assert oca_model_api_mode("custom-responses-model") == "codex_responses"
    assert oca_model_api_mode("oca/custom-chat-model") == "chat_completions"


def test_oca_model_normalization_preserves_provider_prefix():
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider("oca/gpt-5.5", "oca") == "oca/gpt-5.5"


def test_oca_canonical_provider_entry():
    slugs = [entry.slug for entry in CANONICAL_PROVIDERS]
    assert "oca" in slugs


def test_oca_overlay():
    from hermes_cli.providers import HERMES_OVERLAYS

    overlay = HERMES_OVERLAYS["oca"]
    assert overlay.transport == "openai_chat"
    assert overlay.extra_env_vars == ("OCA_ACCESS_TOKEN", "OCA_API_KEY", "OCI_CODE_ASSIST_TOKEN")
    assert overlay.base_url_env_var == "OCA_BASE_URL"

"""Oracle Code Assist provider wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    _OCA_MODEL_API_MODE_OVERRIDES,
    _OCA_MODEL_REASONING_OVERRIDES,
    _PROVIDER_MODELS,
    normalize_provider,
    oca_model_api_mode,
    oca_model_is_reasoning_model,
    provider_model_ids,
)


@pytest.fixture(autouse=True)
def _clear_oca_env(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _OCA_MODEL_API_MODE_OVERRIDES.clear()
    _OCA_MODEL_REASONING_OVERRIDES.clear()
    for key in (
        "OCA_API_KEY",
        "OCA_BASE_URL",
        "HERMES_OCA_MODEL_LIST_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_oca_provider_registry():
    pconfig = PROVIDER_REGISTRY["oca"]
    assert pconfig.name == "Oracle Code Assist"
    assert pconfig.auth_type == "api_key"
    assert pconfig.inference_base_url == (
        "https://code-internal.aiservice.us-chicago-1.oci.oraclecloud.com/20250206/app/litellm"
    )
    assert pconfig.api_key_env_vars == ("OCA_API_KEY",)
    assert pconfig.base_url_env_var == "OCA_BASE_URL"


@pytest.mark.parametrize("alias", ["oca", "oracle", "oracle-code-assist", "oci-code-assist", "code-assist"])
def test_oca_aliases(alias, monkeypatch):
    monkeypatch.setenv("OCA_API_KEY", "oca-token")
    assert resolve_provider(alias) == "oca"
    assert normalize_provider(alias) == "oca"


def test_oca_credentials_from_token_env(monkeypatch):
    monkeypatch.setenv("OCA_API_KEY", "token-from-env")
    creds = resolve_api_key_provider_credentials("oca")
    assert creds["api_key"] == "token-from-env"
    assert creds["base_url"] == PROVIDER_REGISTRY["oca"].inference_base_url


def test_oca_base_url_override(monkeypatch):
    monkeypatch.setenv("OCA_API_KEY", "oca-token")
    monkeypatch.setenv("OCA_BASE_URL", "https://oca.example.com/litellm")
    creds = resolve_api_key_provider_credentials("oca")
    assert creds["base_url"] == "https://oca.example.com/litellm"


def test_oca_model_catalog_uses_credential_pool(monkeypatch, tmp_path):
    import httpx

    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.auth import write_credential_pool

    write_credential_pool(
        "oca",
        [
            {
                "id": "oca001",
                "label": "test-user@example.com",
                "auth_type": "oauth",
                "priority": 0,
                "source": "manual:oca_pkce",
                "access_token": "pool-token",
                "refresh_token": "refresh-token",
                "base_url": "https://oca.example.com/litellm",
            }
        ],
    )
    calls = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"litellm_params": {"model": "oca/gpt-5.5"}}]}

    def _get(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr(httpx, "get", _get)

    assert provider_model_ids("oca") == ["oca/gpt-5.5"]
    assert calls[0][0] == "https://oca.example.com/litellm/v1/model/info"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer pool-token"


def test_oca_model_catalog_static_fallback():
    assert "oca" in _PROVIDER_MODELS
    assert "oca/gpt-oss-120b" in provider_model_ids("oca")
    assert "oca/gpt-4.1" in provider_model_ids("oca")
    assert "oca/gpt-5.5" in provider_model_ids("oca")


def test_oca_model_catalog_reads_model_info_endpoint(monkeypatch):
    import httpx

    monkeypatch.setenv("OCA_API_KEY", "token-from-env")
    monkeypatch.setenv("OCA_BASE_URL", "https://oca.example.com/litellm")
    calls = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "litellm_params": {"model": "oca/llama4"},
                        "model_info": {"supported_api_list": ["CHAT_COMPLETIONS"]},
                    },
                    {
                        "litellm_params": {"model": "oca/gpt-4.1"},
                        "model_info": {"supported_api_list": ["RESPONSES", "CHAT_COMPLETIONS"]},
                    },
                ]
            }

    def _get(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr(httpx, "get", _get)

    assert provider_model_ids("oca") == ["oca/llama4", "oca/gpt-4.1"]
    assert [call[0] for call in calls] == ["https://oca.example.com/litellm/v1/model/info"]
    assert calls[0][1]["headers"]["Authorization"] == "Bearer token-from-env"
    assert oca_model_api_mode("oca/gpt-4.1") == "codex_responses"


def test_oca_model_catalog_uses_model_info_shape(monkeypatch):
    import httpx

    monkeypatch.setenv("OCA_API_KEY", "token-from-env")
    monkeypatch.setenv("OCA_BASE_URL", "https://oca.example.com/litellm")
    calls = []

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
        return _InfoResponse()

    monkeypatch.setattr(httpx, "get", _get)

    assert provider_model_ids("oca") == ["oca/gpt-oss-120b", "oca/llama4", "oca/gpt-5.5"]
    assert calls == ["https://oca.example.com/litellm/v1/model/info"]


def test_oca_model_catalog_falls_back_to_static_on_network_error(monkeypatch):
    import httpx

    monkeypatch.setenv("OCA_API_KEY", "token-from-env")
    monkeypatch.setenv("OCA_BASE_URL", "https://oca.example.com/litellm")

    def _get(url, **kwargs):
        raise httpx.TimeoutException("vpn path timed out")

    monkeypatch.setattr(httpx, "get", _get)

    assert provider_model_ids("oca") == list(_PROVIDER_MODELS["oca"])


def test_oca_model_catalog_honors_timeout_env_and_refreshes_each_call(monkeypatch):
    import httpx

    monkeypatch.setenv("OCA_API_KEY", "token-from-env")
    monkeypatch.setenv("OCA_BASE_URL", "https://oca.example.com/litellm")
    monkeypatch.setenv("HERMES_OCA_MODEL_LIST_TIMEOUT", "0.25")
    calls = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"litellm_params": {"model": "oca/gpt-5.5"}}]}

    def _get(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr(httpx, "get", _get)

    assert provider_model_ids("oca") == ["oca/gpt-5.5"]
    assert provider_model_ids("oca") == ["oca/gpt-5.5"]
    assert len(calls) == 2
    assert calls[0][1]["timeout"] == 0.25
    assert calls[1][1]["timeout"] == 0.25


def test_oca_model_picker_uses_live_catalog(monkeypatch):
    from hermes_cli.model_switch import list_authenticated_providers

    monkeypatch.setenv("OCA_API_KEY", "token-from-env")

    with patch("agent.models_dev.fetch_models_dev", return_value={}), patch(
        "hermes_cli.models._fetch_oca_models",
        return_value=["oca/live-model-a", "oca/live-model-b"],
    ) as fetch_oca:
        providers = list_authenticated_providers(current_provider="oca", max_models=50)

    oca_provider = next((p for p in providers if p["slug"] == "oca"), None)
    assert oca_provider is not None
    assert oca_provider["models"] == ["oca/live-model-a", "oca/live-model-b"]
    assert oca_provider["total_models"] == 2
    fetch_oca.assert_called_once_with(force_refresh=True)


def test_oca_model_api_mode_defaults_to_chat_without_live_capability(monkeypatch):
    monkeypatch.setattr("hermes_cli.models._fetch_oca_models", lambda: [])

    assert oca_model_api_mode("oca/gpt-5.5") == "chat_completions"
    assert oca_model_api_mode("gpt-5.5") == "chat_completions"
    assert oca_model_api_mode("oca/gpt-oss-120b") == "chat_completions"


def test_oca_model_api_mode_refreshes_live_capability_cache(monkeypatch):
    from hermes_cli.models import _extract_oca_model_ids

    def _fetch(*args, **kwargs):
        return _extract_oca_model_ids({
            "data": [
                {
                    "litellm_params": {"model": "oca/gpt-5.5"},
                    "model_info": {"supported_api_list": ["RESPONSES", "CHAT_COMPLETIONS"]},
                },
                {
                    "litellm_params": {"model": "oca/gpt-oss-120b"},
                    "model_info": {"supported_api_list": ["CHAT_COMPLETIONS"]},
                },
            ]
        })

    monkeypatch.setattr("hermes_cli.models._fetch_oca_models", _fetch)

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


def test_oca_model_reasoning_flag_uses_live_model_info():
    from hermes_cli.models import _extract_oca_model_ids

    _extract_oca_model_ids({
        "data": [
            {
                "litellm_params": {"model": "oca/gpt-4.1"},
                "model_info": {
                    "is_reasoning_model": False,
                    "supported_api_list": ["RESPONSES", "CHAT_COMPLETIONS"],
                },
            },
            {
                "litellm_params": {"model": "oca/gpt-5.5"},
                "model_info": {
                    "is_reasoning_model": True,
                    "supported_api_list": ["RESPONSES", "CHAT_COMPLETIONS"],
                },
            },
        ]
    })

    assert oca_model_is_reasoning_model("oca/gpt-4.1") is False
    assert oca_model_is_reasoning_model("gpt-4.1") is False
    assert oca_model_is_reasoning_model("oca/gpt-5.5") is True


def test_oca_model_normalization_preserves_provider_prefix():
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider("oca/gpt-5.5", "oca") == "oca/gpt-5.5"
    assert normalize_model_for_provider("gpt-5.5", "oca") == "oca/gpt-5.5"
    assert normalize_model_for_provider("openai/gpt-5.5", "oca") == "oca/gpt-5.5"


def test_oca_canonical_provider_entry():
    entry = next((entry for entry in CANONICAL_PROVIDERS if entry.slug == "oca"), None)
    assert entry is not None
    assert entry.label == "Oracle Code Assist"
    assert entry.tui_desc == "Oracle Code Assist (OCA; Oracle SSO, requires OCA access)"


def test_oca_overlay():
    from hermes_cli.providers import HERMES_OVERLAYS

    overlay = HERMES_OVERLAYS["oca"]
    assert overlay.transport == "openai_chat"
    assert overlay.extra_env_vars == ("OCA_API_KEY",)
    assert overlay.base_url_env_var == "OCA_BASE_URL"


def test_oca_model_flow_persists_provider_and_api_mode(monkeypatch, tmp_path):
    import yaml

    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    (home / ".env").write_text("", encoding="utf-8")
    (home / "config.yaml").write_text("model: old-model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.auth import write_credential_pool
    from hermes_cli.config import load_config
    from hermes_cli.main import _model_flow_oca
    from hermes_cli.models import _extract_oca_model_ids

    write_credential_pool(
        "oca",
        [
            {
                "id": "oca001",
                "label": "test-user@example.com",
                "auth_type": "oauth",
                "priority": 0,
                "source": "manual:oca_pkce",
                "access_token": "pool-token",
                "refresh_token": "refresh-token",
                "base_url": "https://oca.example.com/litellm",
            }
        ],
    )
    _extract_oca_model_ids({
        "data": [
            {
                "litellm_params": {"model": "oca/gpt-5.5"},
                "model_info": {"supported_api_list": ["RESPONSES", "CHAT_COMPLETIONS"]},
            }
        ]
    })

    with patch("builtins.input", return_value="1"), patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["oca/gpt-5.5", "oca/gpt-oss-120b"],
    ), patch(
        "hermes_cli.auth._prompt_model_selection",
        return_value="oca/gpt-5.5",
    ), patch(
        "hermes_cli.auth_commands.auth_add_command",
    ) as auth_add:
        _model_flow_oca(load_config(), "old-model")

    auth_add.assert_not_called()
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
    model = config.get("model")
    assert isinstance(model, dict)
    assert model["provider"] == "oca"
    assert model["default"] == "oca/gpt-5.5"
    assert model["base_url"] == "https://oca.example.com/litellm"
    assert model["api_mode"] == "codex_responses"

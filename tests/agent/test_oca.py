"""Tests for Oracle Code Assist helpers."""

from __future__ import annotations

import base64
import json
import time

import pytest

import agent.oca as oca
from agent.oca import create_oca_headers, get_oca_config


@pytest.fixture(autouse=True)
def _clear_oca_env(monkeypatch):
    for key in (
        "OCA_API_KEY",
        "OCA_BASE_URL",
        "OCA_IDCS_URL",
        "OCA_IDCS_CLIENT_ID",
        "OCA_IDCS_SCOPES",
        "OCA_REDIRECT_HOST",
        "OCA_REDIRECT_PATH",
        "OCA_REDIRECT_PORTS",
        "OCA_REDIRECT_URI",
    ):
        monkeypatch.delenv(key, raising=False)


def _jwt_with_exp(exp: int) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def test_create_oca_headers():
    headers = create_oca_headers("access-token", "task-123")
    assert headers["Authorization"] == "Bearer access-token"
    assert headers["client"] == "Hermes"
    assert headers["opc-request-id"]
    assert len(headers["opc-request-id"]) == 32


def test_oca_config_env_overrides(monkeypatch):
    monkeypatch.setenv("OCA_BASE_URL", "https://oca.example.com/litellm")
    monkeypatch.setenv("OCA_IDCS_URL", "https://idcs.example.com")
    monkeypatch.setenv("OCA_IDCS_CLIENT_ID", "client-id")
    monkeypatch.setenv("OCA_IDCS_SCOPES", "openid offline_access profile")
    monkeypatch.setenv("OCA_REDIRECT_HOST", "localhost")
    monkeypatch.setenv("OCA_REDIRECT_PATH", "/custom/oca")
    monkeypatch.setenv("OCA_REDIRECT_PORTS", "48805,48806")
    cfg = get_oca_config()
    assert cfg["base_url"] == "https://oca.example.com/litellm"
    assert cfg["idcs_url"] == "https://idcs.example.com"
    assert cfg["client_id"] == "client-id"
    assert cfg["scopes"] == "openid offline_access profile"
    assert cfg["redirect_host"] == "localhost"
    assert cfg["redirect_path"] == "/custom/oca"
    assert cfg["ports"] == [48805, 48806]


def test_oca_config_defaults_match_cline_loopback_callback():
    cfg = get_oca_config()
    assert cfg["redirect_host"] == "127.0.0.1"
    assert cfg["redirect_path"] == "/auth/oca"
    assert cfg["ports"] == list(range(48801, 48812))


def test_oca_pool_refresh_persists_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    for key in ("OCA_API_KEY",):
        monkeypatch.delenv(key, raising=False)

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    expired = _jwt_with_exp(int(time.time()) - 60)
    refreshed = _jwt_with_exp(int(time.time()) + 3600)
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(json.dumps({
        "version": 1,
        "credential_pool": {
            "oca": [
                {
                    "id": "oca-1",
                    "label": "OCA SSO",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "manual:oca_pkce",
                    "access_token": expired,
                    "refresh_token": "refresh-token",
                    "client_id": "client-id",
                    "idcs_url": "https://idcs.example.com",
                    "base_url": "https://oca.example.com/litellm",
                }
            ]
        },
    }))

    from agent.credential_pool import load_pool

    def _refresh(refresh_token, *, client_id="", idcs_url=""):
        assert refresh_token == "refresh-token"
        assert client_id == "client-id"
        assert idcs_url == "https://idcs.example.com"
        return {
            "access_token": refreshed,
            "refresh_token": "next-refresh-token",
            "expires_at": "2099-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(oca, "refresh_oca_access_token", _refresh)

    entry = load_pool("oca").select()

    assert entry is not None
    assert entry.access_token == refreshed
    assert entry.refresh_token == "next-refresh-token"
    saved = json.loads(auth_path.read_text())
    saved_entry = saved["credential_pool"]["oca"][0]
    assert saved_entry["access_token"] == refreshed
    assert saved_entry["refresh_token"] == "next-refresh-token"

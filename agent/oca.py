"""Oracle Code Assist (OCA) auth and request helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import string
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlencode, urlparse

import httpx


DEFAULT_IDCS_CLIENT_ID = "a8331954c0cf48ba99b5dd223a14c6ea"
DEFAULT_IDCS_URL = "https://idcs-9dc693e80d9b469480d7afe00e743931.identity.oraclecloud.com"
DEFAULT_IDCS_SCOPES = "openid offline_access"
DEFAULT_OCA_BASE_URL = "https://code-internal.aiservice.us-chicago-1.oci.oraclecloud.com/20250206/app/litellm"
DEFAULT_OCA_PORTS = tuple(range(48801, 48812))
DEFAULT_OCA_REDIRECT_HOST = "127.0.0.1"
DEFAULT_OCA_REDIRECT_PATH = "/auth/oca"

_TOKEN_ENV_VARS = ("OCA_API_KEY",)


def _jwt_expiry_seconds(token: str) -> int:
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(part.encode("ascii")))
        exp = payload.get("exp")
        return int(exp) if exp else 0
    except Exception:
        return 0


def _expires_at_from_token(token: str) -> str:
    exp = _jwt_expiry_seconds(token)
    if not exp:
        return ""
    return datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()


def _is_token_expiring(token: str, skew_seconds: int = 300) -> bool:
    exp = _jwt_expiry_seconds(token)
    if not exp:
        return False
    return time.time() > exp - skew_seconds


def is_oca_token_expiring(token: str, skew_seconds: int = 300) -> bool:
    return _is_token_expiring(token, skew_seconds)


def _token_subject(token: str) -> str:
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(part.encode("ascii")))
        sub = payload.get("sub") or payload.get("email") or payload.get("preferred_username")
        return str(sub or "").strip()
    except Exception:
        return ""


def _pkce_verifier(length: int = 96) -> str:
    chars = string.ascii_letters + string.digits + "-._~"
    return "".join(secrets.choice(chars) for _ in range(length))


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _random_state(length: int = 32) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def get_oca_config() -> Dict[str, Any]:
    """Load OCA config from environment overrides and built-in defaults."""
    raw_ports = os.getenv("OCA_REDIRECT_PORTS", "").strip() or DEFAULT_OCA_PORTS
    if isinstance(raw_ports, str):
        raw_ports = [part.strip() for part in raw_ports.split(",") if part.strip()]
    if not isinstance(raw_ports, (list, tuple)):
        raw_ports = DEFAULT_OCA_PORTS
    ports = []
    for port in raw_ports:
        try:
            p = int(port)
        except (TypeError, ValueError):
            continue
        if 1 <= p <= 65535:
            ports.append(p)

    return {
        "client_id": os.getenv("OCA_IDCS_CLIENT_ID", "").strip() or DEFAULT_IDCS_CLIENT_ID,
        "idcs_url": (os.getenv("OCA_IDCS_URL", "").strip() or DEFAULT_IDCS_URL).rstrip("/"),
        "scopes": os.getenv("OCA_IDCS_SCOPES", "").strip() or DEFAULT_IDCS_SCOPES,
        "ports": ports or list(DEFAULT_OCA_PORTS),
        "redirect_host": (
            os.getenv("OCA_REDIRECT_HOST", "").strip()
            or DEFAULT_OCA_REDIRECT_HOST
        ),
        "redirect_uri": (
            os.getenv("OCA_REDIRECT_URI", "").strip()
            or ""
        ),
        "redirect_path": (
            os.getenv("OCA_REDIRECT_PATH", "").strip()
            or DEFAULT_OCA_REDIRECT_PATH
        ),
        "base_url": (
            os.getenv("OCA_BASE_URL", "").strip()
            or os.getenv("OCI_CODE_ASSIST_BASE_URL", "").strip()
            or DEFAULT_OCA_BASE_URL
        ).rstrip("/"),
    }


def _make_callback_handler(expected_path: str) -> tuple[type[BaseHTTPRequestHandler], Dict[str, Any]]:
    result: Dict[str, Any] = {"code": None, "state": None, "error": None, "error_description": None}

    class _OcaCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return
            params = parse_qs(parsed.query)
            result["code"] = params.get("code", [None])[0]
            result["state"] = params.get("state", [None])[0]
            result["error"] = params.get("error", [None])[0]
            result["error_description"] = params.get("error_description", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if result["error"]:
                body = "<html><body><h1>Oracle Code Assist login failed.</h1>You can close this tab.</body></html>"
            elif result["code"]:
                body = "<html><body><h1>Oracle Code Assist login received.</h1>You can close this tab.</body></html>"
            else:
                body = "<html><body><h1>Oracle Code Assist callback is running.</h1>No authorization code was received yet.</body></html>"
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _OcaCallbackHandler, result


def _bind_callback_server(ports: list[int], expected_path: str = DEFAULT_OCA_REDIRECT_PATH) -> tuple[HTTPServer, int, Dict[str, Any]]:
    handler_cls, result = _make_callback_handler(expected_path)

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    last_error: OSError | None = None
    for port in ports:
        try:
            return _ReuseHTTPServer(("127.0.0.1", port), handler_cls), port, result
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"Could not bind OCA callback server on ports {ports}: {last_error}")


def _discover_token_endpoint(idcs_url: str, timeout: float = 15.0) -> str:
    url = f"{idcs_url.rstrip('/')}/.well-known/openid-configuration"
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    endpoint = str(resp.json().get("token_endpoint") or "").strip()
    if not endpoint:
        raise RuntimeError("OCA IDCS discovery document did not include token_endpoint.")
    return endpoint


def _exchange_token(endpoint: str, payload: Dict[str, str], timeout: float = 20.0) -> Dict[str, Any]:
    resp = httpx.post(
        endpoint,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=payload,
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OCA token request failed ({resp.status_code}): {resp.text.strip()}")
    data = resp.json()
    if not isinstance(data, dict) or not str(data.get("access_token") or "").strip():
        raise RuntimeError("OCA token response did not include access_token.")
    return data


def refresh_oca_access_token(refresh_token: str, *, client_id: str = "", idcs_url: str = "") -> Dict[str, Any]:
    cfg = get_oca_config()
    client_id = client_id or cfg["client_id"]
    idcs_url = (idcs_url or cfg["idcs_url"]).rstrip("/")
    endpoint = _discover_token_endpoint(idcs_url)
    token_payload = _exchange_token(endpoint, {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    })
    token_payload["expires_at"] = _expires_at_from_token(token_payload.get("access_token", ""))
    return token_payload


def run_oca_oauth_login_pure(*, open_browser: bool = True, timeout_seconds: float = 180.0) -> Dict[str, Any]:
    cfg = get_oca_config()
    redirect_uri = str(cfg.get("redirect_uri") or "").strip()
    if redirect_uri:
        parsed_redirect = urlparse(redirect_uri)
        if not parsed_redirect.scheme or not parsed_redirect.hostname:
            raise RuntimeError(f"Invalid OCA redirect_uri configured: {redirect_uri}")
        redirect_path = parsed_redirect.path or "/"
        redirect_ports = [parsed_redirect.port] if parsed_redirect.port else list(cfg["ports"])
        server, port, callback = _bind_callback_server(redirect_ports, redirect_path)
    else:
        redirect_path = str(cfg.get("redirect_path") or DEFAULT_OCA_REDIRECT_PATH)
        if not redirect_path.startswith("/"):
            redirect_path = f"/{redirect_path}"
        server, port, callback = _bind_callback_server(list(cfg["ports"]), redirect_path)
        redirect_uri = f"http://{cfg['redirect_host']}:{port}{redirect_path}"
    verifier = _pkce_verifier()
    state = _random_state()
    nonce = _random_state()
    auth_url = (
        f"{cfg['idcs_url']}/oauth2/v1/authorize?"
        + urlencode({
            "client_id": cfg["client_id"],
            "response_type": "code",
            "scope": cfg["scopes"],
            "code_challenge": _pkce_challenge(verifier),
            "code_challenge_method": "S256",
            "redirect_uri": redirect_uri,
            "state": state,
            "nonce": nonce,
        })
    )

    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        print("Starting Oracle Code Assist SSO login...")
        print(f"Open this URL if your browser does not open automatically:\n{auth_url}\n")
        if open_browser:
            webbrowser.open(auth_url)
        deadline = time.time() + max(5.0, timeout_seconds)
        while time.time() < deadline:
            if callback["code"] or callback["error"]:
                break
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    if callback["error"]:
        detail = callback.get("error_description") or callback["error"]
        raise RuntimeError(f"OCA authorization failed: {detail}")
    if callback["state"] != state:
        raise RuntimeError("OCA authorization failed: state mismatch.")
    code = str(callback.get("code") or "").strip()
    if not code:
        raise RuntimeError("OCA authorization timed out waiting for the local callback.")

    endpoint = _discover_token_endpoint(cfg["idcs_url"])
    token_payload = _exchange_token(endpoint, {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": cfg["client_id"],
        "code_verifier": verifier,
    })
    id_token = str(token_payload.get("id_token") or "")
    if id_token:
        try:
            part = id_token.split(".")[1]
            part += "=" * (-len(part) % 4)
            claims = json.loads(base64.urlsafe_b64decode(part.encode("ascii")))
            if claims.get("nonce") != nonce:
                raise RuntimeError("OCA ID token nonce verification failed.")
        except RuntimeError:
            raise
        except Exception:
            pass
    access_token = str(token_payload["access_token"])
    return {
        "access_token": access_token,
        "refresh_token": token_payload.get("refresh_token"),
        "expires_at": _expires_at_from_token(access_token),
        "token_type": token_payload.get("token_type", "Bearer"),
        "scope": token_payload.get("scope") or cfg["scopes"],
        "client_id": cfg["client_id"],
        "idcs_url": cfg["idcs_url"],
        "base_url": cfg["base_url"],
        "subject": _token_subject(id_token or access_token),
    }


def create_oca_headers(access_token: str, task_id: str = "") -> Dict[str, str]:
    request_seed = f"{access_token}:{task_id or uuid.uuid4().hex}:{int(time.time())}:{uuid.uuid4().hex}"
    opc_request_id = hashlib.sha256(request_seed.encode("utf-8")).hexdigest()[:32]
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "client": "hermes-agent",
        "client-version": "hermes-agent",
        "client-ide": "Hermes CLI",
        "client-ide-version": "hermes-agent",
        "opc-request-id": opc_request_id,
    }


def resolve_oca_runtime_credentials(*, explicit_api_key: str = "", explicit_base_url: str = "") -> Dict[str, Any]:
    cfg = get_oca_config()
    token = (explicit_api_key or "").strip()
    source = "explicit"
    if not token:
        for key in _TOKEN_ENV_VARS:
            token = os.getenv(key, "").strip()
            if token:
                source = key
                break
    if token:
        return {
            "provider": "oca",
            "api_key": token,
            "base_url": (explicit_base_url or cfg["base_url"]).rstrip("/"),
            "source": source,
            "api_mode": "chat_completions",
        }

    return {
        "provider": "oca",
        "api_key": "",
        "base_url": (explicit_base_url or cfg["base_url"]).rstrip("/"),
        "source": "default",
        "api_mode": "chat_completions",
    }


def maybe_refresh_oca_pool_entry(entry: Any) -> Any:
    token = str(getattr(entry, "access_token", "") or "").strip()
    refresh_token = str(getattr(entry, "refresh_token", "") or "").strip()
    if token and not _is_token_expiring(token):
        return entry
    if not refresh_token:
        return entry
    refreshed = refresh_oca_access_token(
        refresh_token,
        client_id=str(getattr(entry, "client_id", "") or ""),
        idcs_url=str(getattr(entry, "idcs_url", "") or ""),
    )
    entry.access_token = str(refreshed.get("access_token") or token)
    entry.refresh_token = str(refreshed.get("refresh_token") or refresh_token)
    if refreshed.get("expires_at"):
        entry.expires_at = refreshed["expires_at"]
    return entry

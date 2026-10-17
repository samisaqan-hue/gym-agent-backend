import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from test_auth_routes import (  # reuse the auth test setup
    API_VERSION, EMAIL, PASSWORD, SESSION_JSON, USER_ID, fake_supabase, network_down, supabase_error,
)

OLD_REFRESH = "refresh-token-456"
NEW_SESSION = {**SESSION_JSON, "access_token": "new-access-token-789",
               "refresh_token": "new-refresh-token-abc", "expires_in": 3600}

LOG_IN_AGAIN = "Please log in again."
EXPIRED = "Your session has expired. Please log in again."
TOO_MANY = "Too many requests. Please wait a bit before trying again."
UNAVAILABLE = "The account service is unavailable right now. Please try again in a moment."


@pytest.fixture
def client(fake_supabase):
    return TestClient(main.app)


@pytest.fixture
def made_clients(monkeypatch):
    """Records every per-request client the app creates."""
    made = []
    real_factory = main._new_auth_client

    def factory():
        new_client = real_factory()
        made.append(new_client)
        return new_client

    monkeypatch.setattr(main, "_new_auth_client", factory)
    return made


def ok(body):
    return httpx.Response(200, json=body, headers=API_VERSION)


def assert_shared_client_is_clean():
    shared = main.supabase.auth
    assert shared.get_session() is None
    assert shared._refresh_token_timer is None


# ── POST /login ────────────────────────────────────────────────

def test_login_keeps_old_fields_and_adds_refresh_token(client, fake_supabase):
    fake_supabase["handler"] = ok(SESSION_JSON)

    response = client.post("/login", json={"email": EMAIL, "password": PASSWORD})

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "access-token-123"
    assert body["user_id"] == USER_ID
    assert body["refresh_token"] == "refresh-token-456"
    assert body["expires_in"] == 3600
    assert set(body) == {"access_token", "user_id", "refresh_token", "expires_in"}


def test_login_leaves_no_session_or_refresh_timer_anywhere(client, fake_supabase, made_clients):
    fake_supabase["handler"] = ok(SESSION_JSON)

    client.post("/login", json={"email": EMAIL, "password": PASSWORD})

    assert_shared_client_is_clean()
    (per_request,) = made_clients
    assert per_request._auto_refresh_token is False
    assert per_request._refresh_token_timer is None


def test_login_unexpected_error_from_the_per_request_client_is_503(client, monkeypatch):
    class Exploding:
        def sign_in_with_password(self, credentials):
            raise RuntimeError("something odd")

    monkeypatch.setattr(main, "_new_auth_client", Exploding)

    response = client.post("/login", json={"email": EMAIL, "password": PASSWORD})

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}


# ── POST /refresh ──────────────────────────────────────────────

def test_refresh_returns_new_tokens(client, fake_supabase):
    fake_supabase["handler"] = ok(NEW_SESSION)

    response = client.post("/refresh", json={"refresh_token": OLD_REFRESH})

    assert response.status_code == 200
    assert response.json() == {
        "access_token": "new-access-token-789",
        "refresh_token": "new-refresh-token-abc",
        "expires_in": 3600,
    }
    sent = fake_supabase["requests"][0]
    assert sent.url.path == "/auth/v1/token"
    assert sent.url.params["grant_type"] == "refresh_token"
    assert json.loads(sent.content) == {"refresh_token": OLD_REFRESH}


def test_refresh_needs_no_authorization_header(client, fake_supabase):
    fake_supabase["handler"] = ok(NEW_SESSION)

    response = client.post("/refresh", json={"refresh_token": OLD_REFRESH}, headers={})

    assert response.status_code == 200


def test_refresh_leaves_no_session_on_the_shared_client(client, fake_supabase, made_clients):
    fake_supabase["handler"] = ok(NEW_SESSION)

    client.post("/refresh", json={"refresh_token": OLD_REFRESH})
    client.post("/refresh", json={"refresh_token": "another-refresh-token"})

    assert_shared_client_is_clean()
    first, second = made_clients
    assert first is not second
    for per_request in made_clients:
        assert per_request is not main.supabase.auth
        assert per_request._auto_refresh_token is False and per_request._persist_session is False
        assert per_request._refresh_token_timer is None
    # It is only dropped, never signed out (that would revoke the new session).
    assert all(r.url.path != "/auth/v1/logout" for r in fake_supabase["requests"])


@pytest.mark.parametrize("handler", [
    supabase_error(400, "refresh_token_not_found", "Invalid Refresh Token: Refresh Token Not Found"),
    supabase_error(400, "refresh_token_already_used", "Invalid Refresh Token: Already Used"),
    supabase_error(403, "session_expired", "Session Expired"),
    supabase_error(403, "session_not_found", "Session from session_id claim in JWT does not exist"),
    supabase_error(400, "validation_failed", "Invalid Refresh Token"),
], ids=["invalid", "already-used", "expired", "revoked", "malformed"])
def test_bad_refresh_token_is_401(client, fake_supabase, handler):
    fake_supabase["handler"] = handler

    response = client.post("/refresh", json={"refresh_token": "bad-token"})

    assert response.status_code == 401
    assert response.json() == {"detail": EXPIRED}


def test_reusing_a_refresh_token_is_401(client, fake_supabase):
    fake_supabase["handler"] = ok(NEW_SESSION)
    assert client.post("/refresh", json={"refresh_token": OLD_REFRESH}).status_code == 200

    # Supabase rotates refresh tokens: the old one is now used up.
    fake_supabase["handler"] = supabase_error(400, "refresh_token_already_used", "Invalid Refresh Token: Already Used")
    response = client.post("/refresh", json={"refresh_token": OLD_REFRESH})

    assert response.status_code == 401
    assert response.json() == {"detail": EXPIRED}


@pytest.mark.parametrize("body", [
    {}, {"refresh_token": ""}, {"refresh_token": "   "}, {"refresh_token": None}, {"refresh_token": 123},
])
def test_missing_refresh_token_is_400(client, fake_supabase, body):
    response = client.post("/refresh", json=body)

    assert response.status_code == 400
    assert response.json() == {"detail": LOG_IN_AGAIN}
    assert fake_supabase["requests"] == []


def test_refresh_with_no_body_is_400(client, fake_supabase):
    response = client.post("/refresh")
    assert response.status_code == 400
    assert response.json() == {"detail": LOG_IN_AGAIN}


def test_refresh_rate_limited_is_429(client, fake_supabase):
    fake_supabase["handler"] = supabase_error(429, "over_request_rate_limit", "Request rate limit reached")

    response = client.post("/refresh", json={"refresh_token": OLD_REFRESH})

    assert response.status_code == 429
    assert response.json() == {"detail": TOO_MANY}


@pytest.mark.parametrize("handler", [
    httpx.Response(503, text="Service Unavailable"),
    httpx.Response(502, text="Bad Gateway"),
    httpx.Response(500, json={"code": "unexpected_failure", "msg": "Database error"}, headers=API_VERSION),
    httpx.Response(200, text="<html>not json</html>"),
    ok({"user": SESSION_JSON["user"]}),  # a reply without a session
    network_down,
], ids=["503", "502", "500", "not-json", "no-session", "network-down"])
def test_refresh_outage_is_503_not_401(client, fake_supabase, handler):
    fake_supabase["handler"] = handler

    response = client.post("/refresh", json={"refresh_token": OLD_REFRESH})

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}
    assert_shared_client_is_clean()


def test_refresh_unexpected_exception_is_503(client, monkeypatch):
    class Exploding:
        def refresh_session(self, refresh_token):
            raise RuntimeError("something odd")

    monkeypatch.setattr(main, "_new_auth_client", Exploding)

    response = client.post("/refresh", json={"refresh_token": OLD_REFRESH})

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}


# ── Nothing sensitive in responses or logs ─────────────────────

SECRETS = (OLD_REFRESH, "new-access-token-789", "new-refresh-token-abc", "access-token-123",
           EMAIL, "Invalid Refresh Token", "Already Used", "Session Expired",
           "Service Unavailable", "connection refused", "supabase.co")


@pytest.mark.parametrize("handler", [
    ok(NEW_SESSION),
    supabase_error(400, "refresh_token_already_used", "Invalid Refresh Token: Already Used"),
    supabase_error(403, "session_expired", "Session Expired"),
    supabase_error(429, "over_request_rate_limit", "Request rate limit reached"),
    httpx.Response(503, text="Service Unavailable"),
    network_down,
], ids=["success", "reused", "expired", "rate-limited", "outage", "network-down"])
def test_no_tokens_in_logs_or_error_responses(client, fake_supabase, caplog, handler):
    fake_supabase["handler"] = handler

    with caplog.at_level(logging.DEBUG, logger="main"):
        response = client.post("/refresh", json={"refresh_token": OLD_REFRESH})

    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == "main")
    for secret in SECRETS:
        assert secret not in logs
    if response.status_code != 200:
        assert isinstance(response.json()["detail"], str)
        for secret in SECRETS:
            assert secret not in response.text


def test_login_logs_no_tokens(client, fake_supabase, caplog):
    fake_supabase["handler"] = ok(SESSION_JSON)

    with caplog.at_level(logging.DEBUG, logger="main"):
        client.post("/login", json={"email": EMAIL, "password": PASSWORD})

    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == "main")
    for secret in ("access-token-123", "refresh-token-456", EMAIL, PASSWORD):
        assert secret not in logs

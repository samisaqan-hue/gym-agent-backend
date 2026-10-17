import json
import logging
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from supabase_auth import SyncGoTrueClient

import main
from test_auth_routes import (  # reuse the auth test setup
    API_VERSION, EMAIL, SESSION_JSON, USER_JSON, fake_supabase, network_down, supabase_error,
)

CODE = "482913"
NEW_PASSWORD = "brand-new-Passw0rd!"

REQUESTED = {"message": "If an account exists for that email, we sent a code."}
DONE = {"message": "Password updated. Please log in."}
INVALID_EMAIL = "Please enter a valid email address."
MISSING = "Please enter your email, the code and a new password."
INVALID_CODE = "That code is invalid or has expired."
WEAK = "That password is too weak. Please choose a stronger one."
TOO_MANY = "Too many requests. Please wait a bit before trying again."
UNAVAILABLE = "The account service is unavailable right now. Please try again in a moment."

# Like a real recovery session: valid for about an hour from now.
RECOVERY_SESSION = {**SESSION_JSON, "access_token": "recovery-access-token", "refresh_token": "recovery-refresh",
                    "expires_at": int(time.time()) + 3600}


@pytest.fixture
def client(fake_supabase):
    return TestClient(main.app)


@pytest.fixture
def fresh_clients(monkeypatch):
    """Answers the per-request client's calls by path. Each test sets the
    responses for /verify, PUT /user and /logout."""
    state = {"responses": {}, "requests": [], "clients": []}

    def transport(request):
        state["requests"].append(request)
        key = (request.method, request.url.path.removeprefix("/auth/v1/"))
        handler = state["responses"].get(key)
        if handler is None:
            raise AssertionError(f"unexpected call {key}")
        return handler(request) if callable(handler) else handler

    def factory():
        new_client = SyncGoTrueClient(
            url="https://example.supabase.co/auth/v1",
            headers={"apikey": "anon", "Authorization": "Bearer anon"},
            auto_refresh_token=False,
            persist_session=False,
            http_client=httpx.Client(transport=httpx.MockTransport(transport)),
        )
        state["clients"].append(new_client)
        return new_client

    monkeypatch.setattr(main, "_new_auth_client", factory)
    return state


def ok(body=None, status=200):
    return httpx.Response(status, json=body if body is not None else {}, headers=API_VERSION)


def happy_path(fresh_clients):
    fresh_clients["responses"] = {
        ("POST", "verify"): ok(RECOVERY_SESSION),
        ("PUT", "user"): ok(USER_JSON),
        ("POST", "logout"): httpx.Response(204),
    }


def confirm_body(**overrides):
    return {"email": EMAIL, "code": CODE, "new_password": NEW_PASSWORD, **overrides}


# ── POST /password-reset/request ───────────────────────────────

def test_request_sends_recovery_email_and_returns_generic_200(client, fake_supabase):
    fake_supabase["handler"] = ok()

    response = client.post("/password-reset/request", json={"email": EMAIL})

    assert response.status_code == 200
    assert response.json() == REQUESTED
    sent = fake_supabase["requests"][0]
    assert sent.url.path == "/auth/v1/recover"
    assert json.loads(sent.content)["email"] == EMAIL


def test_request_gives_the_same_response_for_existing_and_unknown_emails(client, fake_supabase):
    # Supabase itself answers 200 for both; even if it rejected an unknown
    # email with a 4xx, the response must not change.
    fake_supabase["handler"] = ok()
    existing = client.post("/password-reset/request", json={"email": EMAIL})
    fake_supabase["handler"] = ok()
    unknown = client.post("/password-reset/request", json={"email": "nobody@example.com"})
    fake_supabase["handler"] = supabase_error(400, "user_not_found", "User not found")
    rejected = client.post("/password-reset/request", json={"email": "nobody@example.com"})

    assert existing.status_code == unknown.status_code == rejected.status_code == 200
    assert existing.content == unknown.content == rejected.content


@pytest.mark.parametrize("body", [
    {}, {"email": ""}, {"email": "   "}, {"email": None}, {"email": "not-an-email"},
    {"email": "a@b"}, {"email": 12345}, {"email": ["x@y.co"]},
])
def test_request_rejects_empty_or_malformed_email(client, fake_supabase, body):
    response = client.post("/password-reset/request", json=body)

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_EMAIL}
    assert fake_supabase["requests"] == []


def test_request_with_no_body_is_400(client, fake_supabase):
    response = client.post("/password-reset/request")
    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_EMAIL}


def test_request_email_rejected_by_supabase_is_400(client, fake_supabase):
    fake_supabase["handler"] = supabase_error(400, "email_address_invalid", 'Email address "x" is invalid')

    response = client.post("/password-reset/request", json={"email": EMAIL})

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_EMAIL}


@pytest.mark.parametrize("handler", [
    supabase_error(429, "over_email_send_rate_limit", "For security purposes, you can only request this after 45 seconds."),
    supabase_error(429, "over_request_rate_limit", "Request rate limit reached"),
])
def test_request_rate_limited_is_429(client, fake_supabase, handler):
    fake_supabase["handler"] = handler

    response = client.post("/password-reset/request", json={"email": EMAIL})

    assert response.status_code == 429
    assert response.json() == {"detail": TOO_MANY}


@pytest.mark.parametrize("handler", [
    httpx.Response(503, text="Service Unavailable"),
    httpx.Response(500, json={"code": "unexpected_failure", "msg": "Error sending recovery email"}, headers=API_VERSION),
    network_down,
])
def test_request_outage_is_503(client, fake_supabase, handler):
    fake_supabase["handler"] = handler

    response = client.post("/password-reset/request", json={"email": EMAIL})

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}


# ── POST /password-reset/confirm ───────────────────────────────

def test_confirm_success_updates_password_without_logging_in(client, fake_supabase, fresh_clients):
    happy_path(fresh_clients)

    response = client.post("/password-reset/confirm", json=confirm_body())

    assert response.status_code == 200
    assert response.json() == DONE
    assert "token" not in response.text

    calls = [(r.method, r.url.path) for r in fresh_clients["requests"]]
    assert calls == [("POST", "/auth/v1/verify"), ("PUT", "/auth/v1/user"), ("POST", "/auth/v1/logout")]
    verify = json.loads(fresh_clients["requests"][0].content)
    assert (verify["email"], verify["token"], verify["type"]) == (EMAIL, CODE, "recovery")
    update = fresh_clients["requests"][1]
    assert json.loads(update.content) == {"password": NEW_PASSWORD}
    assert update.headers["authorization"] == "Bearer recovery-access-token"
    logout = fresh_clients["requests"][2]
    assert logout.url.params["scope"] == "local"


def test_confirm_never_touches_the_shared_client(client, fake_supabase, fresh_clients):
    happy_path(fresh_clients)

    client.post("/password-reset/confirm", json=confirm_body())

    assert fake_supabase["requests"] == []
    assert main.supabase.auth.get_session() is None
    # The per-request client doesn't keep the session either.
    assert fresh_clients["clients"][0].get_session() is None


def test_each_confirm_uses_its_own_client(client, fake_supabase, fresh_clients):
    happy_path(fresh_clients)

    client.post("/password-reset/confirm", json=confirm_body())
    client.post("/password-reset/confirm", json=confirm_body())

    first, second = fresh_clients["clients"]
    assert first is not second
    assert first._auto_refresh_token is False and first._persist_session is False


def test_real_client_factory_is_isolated_and_does_not_refresh():
    new_client = main._new_auth_client()
    assert new_client is not main.supabase.auth
    assert new_client._auto_refresh_token is False
    assert new_client._persist_session is False
    assert new_client.get_session() is None


@pytest.mark.parametrize("handler", [
    supabase_error(403, "otp_expired", "Token has expired or is invalid"),
    supabase_error(400, "validation_failed", "Verify requires a token"),
    supabase_error(404, "user_not_found", "User not found"),
])
def test_wrong_or_expired_code_is_400(client, fake_supabase, fresh_clients, handler):
    fresh_clients["responses"] = {("POST", "verify"): handler}

    response = client.post("/password-reset/confirm", json=confirm_body(code="000000"))

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_CODE}
    # No password update was attempted.
    assert [r.url.path for r in fresh_clients["requests"]] == ["/auth/v1/verify"]


def test_already_used_code_is_400(client, fake_supabase, fresh_clients):
    happy_path(fresh_clients)
    assert client.post("/password-reset/confirm", json=confirm_body()).status_code == 200

    fresh_clients["responses"][("POST", "verify")] = supabase_error(403, "otp_expired", "Token has expired or is invalid")
    response = client.post("/password-reset/confirm", json=confirm_body())

    assert response.status_code == 400
    assert response.json() == {"detail": INVALID_CODE}


def test_weak_password_is_400_and_session_is_revoked(client, fake_supabase, fresh_clients):
    happy_path(fresh_clients)
    fresh_clients["responses"][("PUT", "user")] = supabase_error(
        422, "weak_password", "Password should be at least 6 characters.", weak_password={"reasons": ["length"]})

    response = client.post("/password-reset/confirm", json=confirm_body(new_password="abc"))

    assert response.status_code == 400
    assert response.json() == {"detail": WEAK}
    assert fresh_clients["requests"][-1].url.path == "/auth/v1/logout"
    assert fresh_clients["clients"][0].get_session() is None


def test_same_password_is_400(client, fake_supabase, fresh_clients):
    happy_path(fresh_clients)
    fresh_clients["responses"][("PUT", "user")] = supabase_error(
        422, "same_password", "New password should be different from the old password.")

    response = client.post("/password-reset/confirm", json=confirm_body())

    assert response.status_code == 400
    assert response.json() == {"detail": "Your new password must be different from your current one."}


@pytest.mark.parametrize("body", [
    {},
    {"email": EMAIL, "code": CODE},
    {"email": EMAIL, "new_password": NEW_PASSWORD},
    {"code": CODE, "new_password": NEW_PASSWORD},
    {"email": "", "code": CODE, "new_password": NEW_PASSWORD},
    {"email": EMAIL, "code": "  ", "new_password": NEW_PASSWORD},
    {"email": EMAIL, "code": CODE, "new_password": ""},
    {"email": EMAIL, "code": 482913, "new_password": NEW_PASSWORD},
])
def test_confirm_missing_fields_is_400(client, fake_supabase, fresh_clients, body):
    response = client.post("/password-reset/confirm", json=body)

    assert response.status_code == 400
    assert response.json() == {"detail": MISSING}
    assert fresh_clients["requests"] == []


def test_confirm_rate_limited_is_429(client, fake_supabase, fresh_clients):
    fresh_clients["responses"] = {("POST", "verify"): supabase_error(429, "over_request_rate_limit", "Request rate limit reached")}

    response = client.post("/password-reset/confirm", json=confirm_body())

    assert response.status_code == 429
    assert response.json() == {"detail": TOO_MANY}


@pytest.mark.parametrize("step, handler", [
    ("verify", httpx.Response(503, text="Service Unavailable")),
    ("verify", network_down),
    ("verify", ok(USER_JSON)),  # a reply with no session
    ("update", httpx.Response(502, text="Bad Gateway")),
    ("update", network_down),
])
def test_confirm_outage_is_503(client, fake_supabase, fresh_clients, step, handler):
    happy_path(fresh_clients)
    key = ("POST", "verify") if step == "verify" else ("PUT", "user")
    fresh_clients["responses"][key] = handler

    response = client.post("/password-reset/confirm", json=confirm_body())

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}
    assert all(c.get_session() is None for c in fresh_clients["clients"])


def test_confirm_succeeds_even_if_revoking_the_session_fails(client, fake_supabase, fresh_clients):
    happy_path(fresh_clients)
    fresh_clients["responses"][("POST", "logout")] = network_down

    response = client.post("/password-reset/confirm", json=confirm_body())

    assert response.status_code == 200
    assert response.json() == DONE


# ── Nothing sensitive in responses or logs ─────────────────────

SECRETS = (EMAIL, CODE, NEW_PASSWORD, "recovery-access-token", "recovery-refresh",
           "Token has expired", "at least 6", "For security purposes", "Service Unavailable",
           "connection refused", "supabase.co")


@pytest.mark.parametrize("path, body, setup", [
    ("/password-reset/request", {"email": EMAIL}, lambda s, f: s.update(handler=ok())),
    ("/password-reset/request", {"email": EMAIL}, lambda s, f: s.update(handler=supabase_error(
        429, "over_email_send_rate_limit", "For security purposes, you can only request this after 45 seconds."))),
    ("/password-reset/request", {"email": EMAIL}, lambda s, f: s.update(handler=network_down)),
    ("/password-reset/confirm", confirm_body(), lambda s, f: happy_path(f)),
    ("/password-reset/confirm", confirm_body(), lambda s, f: f["responses"].update({
        ("POST", "verify"): supabase_error(403, "otp_expired", "Token has expired or is invalid")})),
    ("/password-reset/confirm", confirm_body(), lambda s, f: (happy_path(f), f["responses"].update({
        ("PUT", "user"): supabase_error(422, "weak_password", "Password should be at least 6 characters.",
                                        weak_password={"reasons": ["length"]})}))),
    ("/password-reset/confirm", confirm_body(), lambda s, f: f["responses"].update({("POST", "verify"): network_down})),
])
def test_no_secrets_in_responses_or_logs(client, fake_supabase, fresh_clients, caplog, path, body, setup):
    setup(fake_supabase, fresh_clients)

    with caplog.at_level(logging.DEBUG, logger="main"):
        response = client.post(path, json=body)

    assert isinstance(response.json().get("detail", response.json().get("message")), str)
    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == "main")
    for secret in SECRETS:
        assert secret not in response.text
        assert secret not in logs


def test_other_routes_keep_fastapis_default_422(client):
    main.app.dependency_overrides[main.get_current_user] = lambda: "3d12cc3a-f847-4c1e-b395-c11915bc71c2"
    try:
        response = client.patch("/workouts/1", json={"weight": "heavy", "reps": 5})
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)

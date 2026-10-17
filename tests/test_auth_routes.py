import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from supabase_auth import SyncGoTrueClient

import main

USER_ID = "3d12cc3a-f847-4c1e-b395-c11915bc71c2"
EMAIL = "lifter@example.com"
PASSWORD = "correct-horse-battery"
API_VERSION = {"X-Supabase-Api-Version": "2024-01-01"}

SIGNUP_BODY = {
    "email": EMAIL, "password": PASSWORD,
    "height": "180cm", "bodyweight": "80kg", "training_start_date": "2024-01-01",
}
LOGIN_BODY = {"email": EMAIL, "password": PASSWORD}

EMAIL_TAKEN = "An account with this email already exists. Try logging in instead."
WRONG_LOGIN = "Incorrect email or password."
UNAVAILABLE = "The account service is unavailable right now. Please try again in a moment."

USER_JSON = {
    "id": USER_ID, "aud": "authenticated", "role": "authenticated", "email": EMAIL,
    "app_metadata": {"provider": "email"}, "user_metadata": {},
    "created_at": "2026-10-17T00:00:00Z",
}
SESSION_JSON = {
    "access_token": "access-token-123", "refresh_token": "refresh-token-456",
    "token_type": "bearer", "expires_in": 3600, "expires_at": 1792000000,
    "user": USER_JSON,
}


def supabase_error(status, code, msg, **extra):
    """A response shaped like Supabase Auth's own error responses."""
    return httpx.Response(status, json={"code": code, "msg": msg, **extra}, headers=API_VERSION)


def network_down(request):
    raise httpx.ConnectError("connection refused")


@pytest.fixture
def fake_supabase(monkeypatch):
    """The real supabase_auth client, with its HTTP calls answered here, so the
    library itself decides which exception each response turns into."""
    state = {"handler": None, "requests": []}

    def transport(request):
        state["requests"].append(request)
        handler = state["handler"]
        return handler(request) if callable(handler) else handler

    client = SyncGoTrueClient(
        url="https://example.supabase.co/auth/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    monkeypatch.setattr(main, "supabase", SimpleNamespace(auth=client))
    return state


class FakeProfileStore:
    """Stands in for SessionLocal and records the profiles that get committed."""

    def __init__(self):
        self.saved = []

    def __call__(self):
        store = self

        class Session:
            def __init__(self):
                self.pending = []

            def add(self, row):
                self.pending.append(row)

            def commit(self):
                store.saved.extend(self.pending)

            def close(self):
                pass

        return Session()


@pytest.fixture
def db(monkeypatch):
    store = FakeProfileStore()
    monkeypatch.setattr(main, "SessionLocal", store)
    return store


@pytest.fixture
def client(fake_supabase, db):
    return TestClient(main.app)


def profile_count(store):
    return len(store.saved)


# ── POST /signup ───────────────────────────────────────────────

def test_signup_success_response_is_unchanged(client, fake_supabase, db):
    fake_supabase["handler"] = httpx.Response(200, json=SESSION_JSON, headers=API_VERSION)

    response = client.post("/signup", json=SIGNUP_BODY)

    assert response.status_code == 200, response.json()
    assert response.json() == {"message": "Signup successful", "user_id": USER_ID}
    assert profile_count(db) == 1
    profile = db.saved[0]
    assert (profile.id, profile.height, profile.bodyweight) == (USER_ID, "180cm", "80kg")
    assert fake_supabase["requests"][0].url.path == "/auth/v1/signup"


@pytest.mark.parametrize("code", ["user_already_exists", "email_exists"])
def test_signup_duplicate_email_is_409_and_creates_no_profile(client, fake_supabase, db, code):
    fake_supabase["handler"] = supabase_error(422, code, "User already registered")

    response = client.post("/signup", json=SIGNUP_BODY)

    assert response.status_code == 409
    assert response.json() == {"detail": EMAIL_TAKEN}
    assert profile_count(db) == 0


@pytest.mark.parametrize("supabase_response, message", [
    (supabase_error(422, "weak_password", "Password should be at least 6 characters.",
                    weak_password={"reasons": ["length"]}),
     "That password is too weak. Please choose a stronger one."),
    (supabase_error(400, "email_address_invalid", 'Email address "x" is invalid'),
     "Please enter a valid email address."),
    (supabase_error(400, "validation_failed", "Unable to validate email address: invalid format"),
     "Please enter a valid email address."),
    (supabase_error(422, "signup_disabled", "Signups not allowed for this instance"),
     "Sign-up failed. Please check your details and try again."),
])
def test_other_signup_errors_are_400_with_a_readable_message(client, fake_supabase, db, supabase_response, message):
    fake_supabase["handler"] = supabase_response

    response = client.post("/signup", json=SIGNUP_BODY)

    assert response.status_code == 400
    assert response.json() == {"detail": message}
    assert profile_count(db) == 0


def test_signup_with_empty_password_is_400(client, fake_supabase, db):
    response = client.post("/signup", json={**SIGNUP_BODY, "password": ""})

    assert response.status_code == 400
    assert response.json() == {"detail": "Please enter an email and password."}
    assert fake_supabase["requests"] == []  # the library rejects it before sending
    assert profile_count(db) == 0


@pytest.mark.parametrize("handler", [
    httpx.Response(503, text="Service Unavailable"),
    httpx.Response(500, json={"code": "unexpected_failure", "msg": "Database error saving new user"}, headers=API_VERSION),
    supabase_error(429, "over_request_rate_limit", "Request rate limit reached"),
    network_down,
])
def test_signup_outage_is_503(client, fake_supabase, db, handler):
    fake_supabase["handler"] = handler

    response = client.post("/signup", json=SIGNUP_BODY)

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}
    assert profile_count(db) == 0


def test_signup_profile_failure_keeps_its_existing_behaviour(fake_supabase, monkeypatch):
    fake_supabase["handler"] = httpx.Response(200, json=SESSION_JSON, headers=API_VERSION)

    class BrokenSession:
        def add(self, row): pass
        def commit(self): raise RuntimeError("database is down")
        def close(self): pass

    monkeypatch.setattr(main, "SessionLocal", BrokenSession)

    response = TestClient(main.app).post("/signup", json=SIGNUP_BODY)

    assert response.status_code == 400
    assert response.json() == {"detail": "Profile creation failed: database is down"}


# ── POST /login ────────────────────────────────────────────────

def test_login_success_response_is_unchanged(client, fake_supabase):
    fake_supabase["handler"] = httpx.Response(200, json=SESSION_JSON, headers=API_VERSION)

    response = client.post("/login", json=LOGIN_BODY)

    assert response.status_code == 200
    body = response.json()
    # The original fields, exactly as before (old app versions rely on them).
    assert body["access_token"] == "access-token-123"
    assert body["user_id"] == USER_ID
    # Plus the refresh-token fields, with the values Supabase returned.
    assert body["refresh_token"] == SESSION_JSON["refresh_token"] == "refresh-token-456"
    assert body["expires_in"] == SESSION_JSON["expires_in"] == 3600
    # Nothing else.
    assert set(body) == {"access_token", "user_id", "refresh_token", "expires_in"}


def test_login_wrong_password_is_401(client, fake_supabase):
    fake_supabase["handler"] = supabase_error(400, "invalid_credentials", "Invalid login credentials")

    response = client.post("/login", json={"email": EMAIL, "password": "wrong"})

    assert response.status_code == 401
    assert response.json() == {"detail": WRONG_LOGIN}


def test_unknown_email_gets_exactly_the_same_response_as_wrong_password(client, fake_supabase):
    fake_supabase["handler"] = supabase_error(400, "invalid_credentials", "Invalid login credentials")
    wrong_password = client.post("/login", json={"email": EMAIL, "password": "wrong"})
    unknown_email = client.post("/login", json={"email": "nobody@example.com", "password": "anything"})

    assert unknown_email.status_code == wrong_password.status_code == 401
    assert unknown_email.content == wrong_password.content
    assert unknown_email.headers.get("www-authenticate") == wrong_password.headers.get("www-authenticate")


def test_login_with_empty_password_is_the_same_401(client, fake_supabase):
    response = client.post("/login", json={"email": EMAIL, "password": ""})

    assert response.status_code == 401
    assert response.json() == {"detail": WRONG_LOGIN}
    assert fake_supabase["requests"] == []


@pytest.mark.parametrize("handler", [
    httpx.Response(503, text="Service Unavailable"),
    httpx.Response(502, text="Bad Gateway"),
    httpx.Response(500, json={"code": "unexpected_failure", "msg": "boom"}, headers=API_VERSION),
    supabase_error(429, "over_request_rate_limit", "Request rate limit reached"),
    httpx.Response(200, text="<html>not json</html>"),
    network_down,
])
def test_login_outage_or_unexpected_error_is_503(client, fake_supabase, handler):
    fake_supabase["handler"] = handler

    response = client.post("/login", json=LOGIN_BODY)

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}


def test_login_unexpected_exception_is_503(client, monkeypatch):
    def explode(credentials):
        raise RuntimeError("something odd")

    monkeypatch.setattr(main, "supabase", SimpleNamespace(auth=SimpleNamespace(sign_in_with_password=explode)))

    response = client.post("/login", json=LOGIN_BODY)

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}


# ── Nothing sensitive in responses or logs ─────────────────────

@pytest.mark.parametrize("path, body, handler", [
    ("/signup", SIGNUP_BODY, supabase_error(422, "user_already_exists", "User already registered")),
    ("/signup", SIGNUP_BODY, supabase_error(422, "weak_password", "Password should be at least 6 characters.",
                                            weak_password={"reasons": ["length"]})),
    ("/signup", SIGNUP_BODY, httpx.Response(503, text="Service Unavailable")),
    ("/login", LOGIN_BODY, supabase_error(400, "invalid_credentials", "Invalid login credentials")),
    ("/login", LOGIN_BODY, network_down),
])
def test_no_password_email_or_supabase_text_in_response_or_logs(client, fake_supabase, caplog, path, body, handler):
    fake_supabase["handler"] = handler

    with caplog.at_level(logging.DEBUG, logger="main"):
        response = client.post(path, json=body)

    detail = response.json()["detail"]
    assert isinstance(detail, str)
    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == "main")
    for leaked in (PASSWORD, EMAIL, "already registered", "at least 6", "Invalid login",
                   "Service Unavailable", "connection refused", "supabase.co"):
        assert leaked not in detail
        assert leaked not in logs


# ── Signup: placeholder user for an existing email ─────────────

IDENTITY_JSON = {
    "id": USER_ID, "identity_id": "9b1c2d3e-0000-4000-8000-000000000001", "user_id": USER_ID,
    "identity_data": {"sub": USER_ID, "email": EMAIL}, "provider": "email",
    "created_at": "2026-10-17T00:00:00Z",
}
# What Supabase Auth sends back instead of an error for an email that already
# has an account: a user (with a made-up id) whose identities list is empty,
# and no session.
PLACEHOLDER_USER_JSON = {
    "id": "0f5d9a2e-1111-4222-8333-444455556666", "aud": "authenticated", "role": "",
    "email": EMAIL, "phone": "", "app_metadata": {"provider": "email", "providers": ["email"]},
    "user_metadata": {}, "identities": [], "created_at": "2026-10-17T00:00:00Z",
    "updated_at": "2026-10-17T00:00:00Z", "is_anonymous": False,
}
NEW_USER_JSON = {**USER_JSON, "identities": [IDENTITY_JSON]}


def test_signup_placeholder_user_is_409_and_creates_no_profile(client, fake_supabase, db):
    fake_supabase["handler"] = httpx.Response(200, json=PLACEHOLDER_USER_JSON, headers=API_VERSION)

    response = client.post("/signup", json=SIGNUP_BODY)

    assert response.status_code == 409
    assert response.json() == {"detail": EMAIL_TAKEN}
    assert profile_count(db) == 0


def test_placeholder_user_gets_the_same_response_as_the_duplicate_email_error(client, fake_supabase, db):
    fake_supabase["handler"] = httpx.Response(200, json=PLACEHOLDER_USER_JSON, headers=API_VERSION)
    placeholder = client.post("/signup", json=SIGNUP_BODY)

    fake_supabase["handler"] = supabase_error(422, "user_already_exists", "User already registered")
    raised = client.post("/signup", json=SIGNUP_BODY)

    assert placeholder.status_code == raised.status_code == 409
    assert placeholder.content == raised.content
    assert profile_count(db) == 0


@pytest.mark.parametrize("supabase_body", [
    # Email confirmation off: a session and the new user come back.
    {**SESSION_JSON, "user": NEW_USER_JSON},
    # Email confirmation on: just the new user, with its identity, no session.
    NEW_USER_JSON,
], ids=["with-session", "user-only"])
def test_new_signup_with_an_identity_still_succeeds(client, fake_supabase, db, supabase_body):
    fake_supabase["handler"] = httpx.Response(200, json=supabase_body, headers=API_VERSION)

    response = client.post("/signup", json=SIGNUP_BODY)

    assert response.status_code == 200
    assert response.json() == {"message": "Signup successful", "user_id": USER_ID}
    assert profile_count(db) == 1
    assert db.saved[0].id == USER_ID


def test_placeholder_user_logs_nothing_sensitive(client, fake_supabase, caplog):
    fake_supabase["handler"] = httpx.Response(200, json=PLACEHOLDER_USER_JSON, headers=API_VERSION)

    with caplog.at_level(logging.DEBUG, logger="main"):
        client.post("/signup", json=SIGNUP_BODY)

    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == "main")
    assert "placeholder user" in logs
    for leaked in (PASSWORD, EMAIL, PLACEHOLDER_USER_JSON["id"]):
        assert leaked not in logs

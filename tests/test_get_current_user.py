from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from supabase_auth.errors import AuthApiError, AuthRetryableError

import main


class FakeAuth:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.tokens = []

    def get_user(self, token):
        self.tokens.append(token)
        if self.error is not None:
            raise self.error
        return self.result


def use_fake_supabase(monkeypatch, result=None, error=None):
    fake_auth = FakeAuth(result=result, error=error)
    monkeypatch.setattr(main, "supabase", SimpleNamespace(auth=fake_auth))
    return fake_auth


def call(token="some-token"):
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    return main.get_current_user(credentials)


def assert_invalid_token(exc_info):
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or expired token"
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


def test_valid_token_returns_user_id(monkeypatch):
    user_response = SimpleNamespace(user=SimpleNamespace(id="user-123"))
    fake_auth = use_fake_supabase(monkeypatch, result=user_response)

    assert call("good-token") == "user-123"
    assert fake_auth.tokens == ["good-token"]


def test_expired_token_gives_401(monkeypatch):
    error = AuthApiError("invalid JWT: token is expired", 403, "bad_jwt")
    use_fake_supabase(monkeypatch, error=error)

    with pytest.raises(HTTPException) as exc_info:
        call()
    assert_invalid_token(exc_info)


def test_invalid_token_gives_401(monkeypatch):
    error = AuthApiError("invalid JWT: unable to parse or verify signature", 401, "bad_jwt")
    use_fake_supabase(monkeypatch, error=error)

    with pytest.raises(HTTPException) as exc_info:
        call("not-a-real-token")
    assert_invalid_token(exc_info)


@pytest.mark.parametrize("user_response", [None, SimpleNamespace(user=None)])
def test_no_user_gives_401(monkeypatch, user_response):
    use_fake_supabase(monkeypatch, result=user_response)

    with pytest.raises(HTTPException) as exc_info:
        call()
    assert_invalid_token(exc_info)


def test_network_failure_is_not_turned_into_401(monkeypatch):
    use_fake_supabase(monkeypatch, error=httpx.ConnectError("connection refused"))

    with pytest.raises(httpx.ConnectError):
        call()


def test_supabase_outage_is_not_turned_into_401(monkeypatch):
    use_fake_supabase(monkeypatch, error=AuthRetryableError("Service Unavailable", 503))

    with pytest.raises(AuthRetryableError):
        call()


def test_supabase_server_error_is_not_turned_into_401(monkeypatch):
    use_fake_supabase(monkeypatch, error=AuthApiError("Internal Server Error", 500, None))

    with pytest.raises(AuthApiError):
        call()


def test_protected_route_returns_401_with_header(monkeypatch):
    error = AuthApiError("invalid JWT: token is expired", 403, "bad_jwt")
    use_fake_supabase(monkeypatch, error=error)

    client = TestClient(main.app)
    response = client.get("/workouts", headers={"Authorization": "Bearer expired-token"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired token"}
    assert response.headers["www-authenticate"] == "Bearer"

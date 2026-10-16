from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from supabase import AuthApiError

import main
from test_critique_limit import (  # reuse the POST /critique test setup
    USER_A, USER_B, central, clock, current_user, db, gemini, rows_for,
)

AVAILABLE = {"available": True, "retry_after_seconds": None}


@pytest.fixture
def client(db, clock, gemini, current_user):
    return TestClient(main.app)


def log_critique(client, clock, local_time):
    clock.set(local_time)
    assert client.post("/critique").status_code == 200


def status(client, clock, local_time):
    clock.set(local_time)
    response = client.get("/critique/status")
    assert response.status_code == 200
    return response.json()


def test_no_critique_today_is_available(client, clock, gemini, db):
    assert status(client, clock, central(2026, 10, 15, 9, 0)) == AVAILABLE
    assert gemini.calls == []
    assert rows_for(db, USER_A) == 0


def test_critique_logged_today_is_unavailable_until_midnight(client, clock, gemini, db):
    log_critique(client, clock, central(2026, 10, 15, 9, 0))

    assert status(client, clock, central(2026, 10, 15, 22, 0)) == {
        "available": False,
        "retry_after_seconds": 2 * 60 * 60,  # 10 PM -> midnight Central
    }
    # Checking status never calls Gemini or writes a row.
    assert gemini.calls == [USER_A]
    assert rows_for(db, USER_A) == 1


def test_critique_logged_yesterday_central_is_available(client, clock):
    # 11 PM Central on Oct 14 is already Oct 15 in UTC, but it was yesterday in Central.
    log_critique(client, clock, central(2026, 10, 14, 23, 0))

    assert status(client, clock, central(2026, 10, 15, 0, 30)) == AVAILABLE


def test_midnight_boundary_matches_post_critique(client, clock):
    log_critique(client, clock, central(2026, 10, 15, 23, 55))

    clock.set(central(2026, 10, 15, 23, 59, 30))
    blocked = client.post("/critique")
    assert status(client, clock, central(2026, 10, 15, 23, 59, 30)) == {
        "available": False,
        "retry_after_seconds": int(blocked.headers["Retry-After"]),
    }
    assert blocked.headers["Retry-After"] == "30"

    assert status(client, clock, central(2026, 10, 16, 0, 5)) == AVAILABLE
    clock.set(central(2026, 10, 16, 0, 5))
    assert client.post("/critique").status_code == 200


def test_day_follows_central_time_not_utc(client, clock):
    # 8 PM Central is already the next day in UTC, but it's the same Central day.
    log_critique(client, clock, central(2026, 10, 15, 8, 0))

    assert status(client, clock, central(2026, 10, 15, 20, 0))["available"] is False


def test_retry_after_across_daylight_saving_change(client, clock):
    log_critique(client, clock, central(2026, 11, 1, 0, 10))

    assert status(client, clock, central(2026, 11, 1, 0, 30))["retry_after_seconds"] == int(24.5 * 60 * 60)


def test_another_users_critique_does_not_count(client, clock, current_user):
    current_user["id"] = USER_B
    log_critique(client, clock, central(2026, 10, 15, 9, 0))

    current_user["id"] = USER_A
    assert status(client, clock, central(2026, 10, 15, 10, 0)) == AVAILABLE

    current_user["id"] = USER_B
    assert status(client, clock, central(2026, 10, 15, 10, 0))["available"] is False


def test_no_token_returns_401(db, clock):
    response = TestClient(main.app).get("/critique/status")
    assert response.status_code == 401


def test_expired_token_returns_401(db, clock, monkeypatch):
    class FakeAuth:
        def get_user(self, token):
            raise AuthApiError("invalid JWT: token is expired", 403, "bad_jwt")

    monkeypatch.setattr(main, "supabase", SimpleNamespace(auth=FakeAuth()))

    response = TestClient(main.app).get("/critique/status", headers={"Authorization": "Bearer expired"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired token"}

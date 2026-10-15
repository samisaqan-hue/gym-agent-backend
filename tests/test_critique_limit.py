import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
from database import CritiqueLog

CENTRAL = ZoneInfo("America/Chicago")
USER_A = str(uuid.uuid4())
USER_B = str(uuid.uuid4())
LIMIT_MESSAGE = (
    "You've reached your daily critique limit. Your next critique is available "
    "tomorrow at 12:00 AM Central Time."
)

def central(*args):
    return datetime(*args, tzinfo=CENTRAL)


class Clock:
    """Stands in for main.datetime so tests control what "now" is."""

    def __init__(self):
        self.now_utc = None

    def set(self, local_time):
        self.now_utc = local_time.astimezone(timezone.utc)

    def datetime_class(self):
        clock = self

        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.now_utc.astimezone(tz) if tz else clock.now_utc.replace(tzinfo=None)

        return FrozenDatetime


class FakeGemini:
    def __init__(self):
        self.calls = []
        self.error = None

    def __call__(self, user_id, height, bodyweight, months_training):
        self.calls.append(user_id)
        if self.error is not None:
            raise self.error
        return f"critique for {user_id}"


@pytest.fixture
def db(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # critique_log's foreign key points at Supabase's auth schema; SQLite needs
    # a schema with that name to exist before it can create the table.
    @event.listens_for(engine, "connect")
    def attach_auth_schema(dbapi_connection, _):
        dbapi_connection.execute("ATTACH DATABASE ':memory:' AS auth")

    CritiqueLog.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(main, "SessionLocal", Session)
    yield Session
    engine.dispose()


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    clock.set(central(2026, 10, 15, 9, 0))
    monkeypatch.setattr(main, "datetime", clock.datetime_class())
    return clock


@pytest.fixture
def gemini(monkeypatch):
    fake = FakeGemini()
    monkeypatch.setattr(main, "get_critique", fake)
    monkeypatch.setattr(
        main,
        "get_profile",
        lambda user_id: {"height": "180cm", "bodyweight": "80kg", "training_start_date": "2024-01-01"},
    )
    monkeypatch.setattr(main, "calculate_months_training", lambda start: 12)
    return fake


@pytest.fixture
def current_user():
    holder = {"id": USER_A}
    main.app.dependency_overrides[main.get_current_user] = lambda: holder["id"]
    yield holder
    main.app.dependency_overrides.clear()


@pytest.fixture
def client(db, clock, gemini, current_user):
    return TestClient(main.app, raise_server_exceptions=False)


def rows_for(Session, user_id):
    session = Session()
    try:
        return (
            session.query(func.count(CritiqueLog.id))
            .filter(CritiqueLog.user_id == uuid.UUID(user_id))
            .scalar()
        )
    finally:
        session.close()


def test_first_request_of_the_day_succeeds_and_records_a_row(client, db, gemini):
    response = client.post("/critique")

    assert response.status_code == 200
    assert response.json() == {"critique": f"critique for {USER_A}", "remaining": 0}
    assert gemini.calls == [USER_A]
    assert rows_for(db, USER_A) == 1


def test_second_request_same_day_gets_429_without_calling_gemini(client, db, clock, gemini):
    clock.set(central(2026, 10, 15, 9, 0))
    assert client.post("/critique").status_code == 200

    clock.set(central(2026, 10, 15, 22, 0))
    response = client.post("/critique")

    assert response.status_code == 429
    assert response.json() == {"detail": LIMIT_MESSAGE}
    assert response.headers["Retry-After"] == str(2 * 60 * 60)  # 10 PM -> midnight
    assert gemini.calls == [USER_A]
    assert rows_for(db, USER_A) == 1


def test_day_follows_central_time_not_utc(client, clock, gemini):
    # 8 PM Central on Oct 15 is already Oct 16 in UTC, but it is the same
    # Central calendar day as the morning critique, so it is still blocked.
    clock.set(central(2026, 10, 15, 8, 0))
    assert client.post("/critique").status_code == 200

    clock.set(central(2026, 10, 15, 20, 0))
    assert client.post("/critique").status_code == 429
    assert len(gemini.calls) == 1


def test_request_just_after_midnight_central_works_again(client, db, clock, gemini):
    clock.set(central(2026, 10, 15, 23, 55))
    assert client.post("/critique").status_code == 200

    clock.set(central(2026, 10, 15, 23, 59, 30))
    blocked = client.post("/critique")
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "30"

    clock.set(central(2026, 10, 16, 0, 5))
    response = client.post("/critique")

    assert response.status_code == 200
    assert response.json()["remaining"] == 0
    assert len(gemini.calls) == 2
    assert rows_for(db, USER_A) == 2


def test_retry_after_counts_real_seconds_across_daylight_saving_change(client, clock):
    # Clocks fall back at 2 AM on Nov 1, 2026, so that day is 25 hours long.
    clock.set(central(2026, 11, 1, 0, 10))
    assert client.post("/critique").status_code == 200

    clock.set(central(2026, 11, 1, 0, 30))
    response = client.post("/critique")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == str(int(24.5 * 60 * 60))


def test_failed_gemini_call_does_not_record_a_row(client, db, gemini):
    gemini.error = RuntimeError("Gemini is down")

    response = client.post("/critique")

    assert response.status_code == 500
    assert rows_for(db, USER_A) == 0

    # The failed attempt didn't use up the day's critique.
    gemini.error = None
    assert client.post("/critique").status_code == 200
    assert rows_for(db, USER_A) == 1


def test_one_users_limit_does_not_affect_another_user(client, db, gemini, current_user):
    current_user["id"] = USER_A
    assert client.post("/critique").status_code == 200
    assert client.post("/critique").status_code == 429

    current_user["id"] = USER_B
    response = client.post("/critique")

    assert response.status_code == 200
    assert response.json()["critique"] == f"critique for {USER_B}"
    assert gemini.calls == [USER_A, USER_B]
    assert rows_for(db, USER_A) == 1
    assert rows_for(db, USER_B) == 1

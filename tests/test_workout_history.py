import uuid
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
from database import Workout

USER_A = str(uuid.uuid4())
USER_B = str(uuid.uuid4())


@pytest.fixture
def db(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Workout.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(main, "SessionLocal", Session)
    yield Session
    engine.dispose()


@pytest.fixture
def current_user():
    holder = {"id": USER_A}
    main.app.dependency_overrides[main.get_current_user] = lambda: holder["id"]
    yield holder
    main.app.dependency_overrides.clear()


@pytest.fixture
def client(db, current_user):
    return TestClient(main.app)


def add_set(Session, user_id, day, exercise, set_number, weight, reps):
    session = Session()
    row = Workout(user_id=uuid.UUID(user_id), date=day, exercise=exercise,
                  set_number=set_number, weight=weight, reps=reps)
    session.add(row)
    session.commit()
    workout_id = row.id
    session.close()
    return workout_id


def load(Session, workout_id):
    session = Session()
    row = session.get(Workout, workout_id)
    session.close()
    return row


# ── GET /workouts ──────────────────────────────────────────────

def test_get_returns_ids_and_only_the_current_users_sets(client, db):
    a2 = add_set(db, USER_A, date(2026, 10, 16), "bench", 1, 210, 3)
    add_set(db, USER_B, date(2026, 10, 15), "squat", 1, 300, 5)
    a1 = add_set(db, USER_A, date(2026, 10, 14), "bench", 1, 200, 5)

    response = client.get("/workouts")

    assert response.status_code == 200
    assert response.json() == {"workouts": [
        {"id": a1, "date": "2026-10-14", "exercise": "bench", "set_number": 1, "weight": 200.0, "reps": 5},
        {"id": a2, "date": "2026-10-16", "exercise": "bench", "set_number": 1, "weight": 210.0, "reps": 3},
    ]}


def test_get_with_no_sets_returns_empty_list(client):
    assert client.get("/workouts").json() == {"workouts": []}


def test_prompt_workout_data_still_has_no_id(monkeypatch):
    import pipeline

    row = SimpleNamespace(id=7, date=date(2026, 10, 16), exercise="bench", set_number=1, weight=200, reps=5)

    class FakeQuery:
        def filter(self, *args): return self
        def order_by(self, *args): return self
        def all(self): return [row]

    class FakeSession:
        def query(self, model): return FakeQuery()
        def close(self): pass

    monkeypatch.setattr(pipeline, "SessionLocal", FakeSession)

    assert pipeline.get_all_workouts(USER_A) == [
        {"date": date(2026, 10, 16), "exercise": "bench", "set_number": 1, "weight": 200, "reps": 5}
    ]


# ── PATCH /workouts/{id} ───────────────────────────────────────

def test_patch_updates_weight_and_reps(client, db):
    workout_id = add_set(db, USER_A, date(2026, 10, 16), "bench", 2, 2000, 5)

    response = client.patch(f"/workouts/{workout_id}", json={"weight": 200, "reps": 6})

    assert response.status_code == 200
    assert response.json() == {
        "id": workout_id, "date": "2026-10-16", "exercise": "bench",
        "set_number": 2, "weight": 200.0, "reps": 6,
    }
    row = load(db, workout_id)
    assert (row.weight, row.reps) == (200.0, 6)


def test_patch_accepts_decimal_weight_and_zero_weight(client, db):
    workout_id = add_set(db, USER_A, date(2026, 10, 16), "curl", 1, 30, 10)

    assert client.patch(f"/workouts/{workout_id}", json={"weight": 32.5, "reps": 10}).json()["weight"] == 32.5
    assert client.patch(f"/workouts/{workout_id}", json={"weight": 0, "reps": 10}).json()["weight"] == 0.0


@pytest.mark.parametrize("field, value", [
    ("date", "2026-01-01"),
    ("exercise", "squat"),
    ("set_number", 9),
    ("user_id", USER_B),
    ("id", 999),
])
def test_patch_rejects_changing_other_fields(client, db, field, value):
    workout_id = add_set(db, USER_A, date(2026, 10, 16), "bench", 1, 200, 5)

    response = client.patch(f"/workouts/{workout_id}", json={"weight": 210, "reps": 4, field: value})

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", field]
    assert error["msg"] == "Extra inputs are not permitted"
    row = load(db, workout_id)
    assert (row.date, row.exercise, row.set_number, row.weight, row.reps) == (date(2026, 10, 16), "bench", 1, 200, 5)


def test_patch_on_another_users_set_returns_404_and_changes_nothing(client, db):
    other_id = add_set(db, USER_B, date(2026, 10, 16), "squat", 1, 300, 5)

    response = client.patch(f"/workouts/{other_id}", json={"weight": 1, "reps": 1})

    assert response.status_code == 404
    assert response.json() == {"detail": "Workout set not found"}
    row = load(db, other_id)
    assert (row.weight, row.reps) == (300, 5)


def test_patch_on_missing_id_returns_the_same_404(client):
    response = client.patch("/workouts/123456", json={"weight": 100, "reps": 5})

    assert response.status_code == 404
    assert response.json() == {"detail": "Workout set not found"}


@pytest.mark.parametrize("body, field, message", [
    ({"weight": -5, "reps": 5}, "weight", "Input should be greater than or equal to 0"),
    ({"weight": "heavy", "reps": 5}, "weight", "Input should be a valid number"),
    ({"weight": "100", "reps": 5}, "weight", "Input should be a valid number"),
    ({"weight": None, "reps": 5}, "weight", "Input should be a valid number"),
    ({"weight": True, "reps": 5}, "weight", "Input should be a valid number"),
    ({"reps": 5}, "weight", "Field required"),
    ({"weight": 100, "reps": 0}, "reps", "Input should be greater than or equal to 1"),
    ({"weight": 100, "reps": -3}, "reps", "Input should be greater than or equal to 1"),
    ({"weight": 100, "reps": 5.5}, "reps", "Input should be a valid integer"),
    ({"weight": 100, "reps": "5"}, "reps", "Input should be a valid integer"),
    ({"weight": 100, "reps": True}, "reps", "Input should be a valid integer"),
    ({"weight": 100}, "reps", "Field required"),
])
def test_patch_rejects_invalid_weight_or_reps(client, db, body, field, message):
    workout_id = add_set(db, USER_A, date(2026, 10, 16), "bench", 1, 200, 5)

    response = client.patch(f"/workouts/{workout_id}", json=body)

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", field]
    assert error["msg"] == message
    row = load(db, workout_id)
    assert (row.weight, row.reps) == (200, 5)


def test_patch_rejects_non_finite_weight(db, current_user):
    # A hand-crafted body with Infinity is rejected and nothing is saved.
    # FastAPI can't write its 422 body here (the error echoes the value, and
    # Infinity isn't valid JSON), so the client sees a 500. Apps can't send
    # this: JSON.stringify turns Infinity into null, which gets a normal 422.
    client = TestClient(main.app, raise_server_exceptions=False)
    workout_id = add_set(db, USER_A, date(2026, 10, 16), "bench", 1, 200, 5)

    response = client.patch(
        f"/workouts/{workout_id}",
        content='{"weight": Infinity, "reps": 5}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code != 200
    assert load(db, workout_id).weight == 200


# ── Authentication ─────────────────────────────────────────────

@pytest.mark.parametrize("method, path, body", [
    ("get", "/workouts", None),
    ("patch", "/workouts/1", {"weight": 100, "reps": 5}),
])
def test_no_token_returns_401(db, method, path, body):
    client = TestClient(main.app)
    response = client.request(method.upper(), path, json=body)
    assert response.status_code == 401


def test_expired_token_returns_401_on_patch(db, monkeypatch):
    from supabase import AuthApiError

    class FakeAuth:
        def get_user(self, token):
            raise AuthApiError("invalid JWT: token is expired", 403, "bad_jwt")

    monkeypatch.setattr(main, "supabase", SimpleNamespace(auth=FakeAuth()))
    workout_id = add_set(db, USER_A, date(2026, 10, 16), "bench", 1, 200, 5)

    response = TestClient(main.app).patch(
        f"/workouts/{workout_id}",
        json={"weight": 1, "reps": 1},
        headers={"Authorization": "Bearer expired"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired token"}
    assert load(db, workout_id).weight == 200

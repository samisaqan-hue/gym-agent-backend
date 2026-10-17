import uuid
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SASession, sessionmaker

import main
import pipeline
from database import Workout
from test_workout_history import USER_A, USER_B, add_set, current_user, db, load  # reuse the workout test setup

DAY = date(2026, 10, 16)
OTHER_DAY = date(2026, 10, 14)
NOT_FOUND = {"detail": "Workout set not found"}


class PostgresLikeSession(SASession):
    """POST /workouts saves the date and user id as the strings it receives.
    Postgres converts them; SQLite doesn't. This test session converts them
    the same way so POST can be tested against the in-memory database."""

    def add(self, instance, *args, **kwargs):
        if isinstance(instance, Workout):
            if isinstance(instance.date, str):
                instance.date = date.fromisoformat(instance.date)
            if isinstance(instance.user_id, str):
                instance.user_id = uuid.UUID(instance.user_id)
        return super().add(instance, *args, **kwargs)


@pytest.fixture
def client(db, current_user, monkeypatch):
    monkeypatch.setattr(main, "SessionLocal", sessionmaker(bind=db.kw["bind"], class_=PostgresLikeSession))
    return TestClient(main.app)


def sets_of(client, exercise=None, day=None):
    rows = client.get("/workouts").json()["workouts"]
    return [
        (r["id"], r["set_number"], r["weight"])
        for r in rows
        if (exercise is None or r["exercise"] == exercise) and (day is None or r["date"] == day.isoformat())
    ]


def add_sets(db, user, day, exercise, weights):
    return [add_set(db, user, day, exercise, n, w, 5) for n, w in enumerate(weights, start=1)]


# ── Deleting ───────────────────────────────────────────────────

def test_delete_removes_the_set_and_keeps_the_others(client, db):
    bench = add_sets(db, USER_A, DAY, "bench", [200, 205])
    squat = add_set(db, USER_A, DAY, "squat", 1, 300, 5)

    response = client.delete(f"/workouts/{bench[1]}")

    assert response.status_code == 200
    assert response.json() == {"message": "Workout set deleted", "id": bench[1], "renumbered": []}
    remaining_ids = [r[0] for r in sets_of(client)]
    assert bench[1] not in remaining_ids
    assert set(remaining_ids) == {bench[0], squat}
    assert load(db, bench[1]) is None


@pytest.mark.parametrize("delete_index, expected_renumbered", [
    (0, [(0, 1), (1, 2), (2, 3), (3, 4)]),  # first: every later set moves up
    (2, [(2, 3), (3, 4)]),                  # middle: only the sets after it move
    (4, []),                                # last: nothing moves
], ids=["first", "middle", "last"])
def test_renumbering_after_delete(client, db, delete_index, expected_renumbered):
    ids = add_sets(db, USER_A, DAY, "bench", [200, 205, 210, 215, 220])
    deleted = ids[delete_index]
    remaining = [i for i in ids if i != deleted]

    response = client.delete(f"/workouts/{deleted}")

    # expected_renumbered holds (index into remaining, new number).
    assert response.json()["renumbered"] == [
        {"id": remaining[index], "set_number": number} for index, number in expected_renumbered
    ]
    assert [(r[0], r[1]) for r in sets_of(client, "bench", DAY)] == [
        (set_id, n) for n, set_id in enumerate(remaining, start=1)
    ]


def test_renumbering_keeps_order_and_breaks_ties_by_id(client, db):
    # Messy existing numbers: a gap and a duplicate.
    a = add_set(db, USER_A, DAY, "bench", 1, 200, 5)
    b = add_set(db, USER_A, DAY, "bench", 3, 205, 5)
    c = add_set(db, USER_A, DAY, "bench", 3, 210, 5)
    d = add_set(db, USER_A, DAY, "bench", 7, 215, 5)

    response = client.delete(f"/workouts/{a}")

    assert response.json()["renumbered"] == [
        {"id": b, "set_number": 1}, {"id": c, "set_number": 2}, {"id": d, "set_number": 3},
    ]


def test_delete_never_touches_other_exercises_days_or_users(client, db):
    bench_today = add_sets(db, USER_A, DAY, "bench", [200, 205, 210])
    squat_today = add_sets(db, USER_A, DAY, "squat", [300, 305, 310])
    bench_other_day = add_sets(db, USER_A, OTHER_DAY, "bench", [190, 195, 200])
    other_user = add_sets(db, USER_B, DAY, "bench", [100, 105, 110])

    client.delete(f"/workouts/{bench_today[0]}")

    for ids in (squat_today, bench_other_day, other_user):
        assert [load(db, i).set_number for i in ids] == [1, 2, 3]
    assert [load(db, i).set_number for i in bench_today[1:]] == [1, 2]


def test_deleting_a_missing_id_is_404(client, db):
    response = client.delete("/workouts/999999")
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_deleting_another_users_set_is_404_and_changes_nothing(client, db):
    theirs = add_sets(db, USER_B, DAY, "bench", [100, 105, 110])

    response = client.delete(f"/workouts/{theirs[0]}")

    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert [(load(db, i).set_number, load(db, i).weight) for i in theirs] == [(1, 100), (2, 105), (3, 110)]


def test_missing_and_other_users_ids_get_identical_responses(client, db):
    theirs = add_set(db, USER_B, DAY, "bench", 1, 100, 5)
    other = client.delete(f"/workouts/{theirs}")
    missing = client.delete("/workouts/999999")
    assert other.status_code == missing.status_code == 404
    assert other.content == missing.content


def test_second_delete_of_the_same_id_is_404(client, db):
    ids = add_sets(db, USER_A, DAY, "bench", [200, 205])

    assert client.delete(f"/workouts/{ids[0]}").status_code == 200
    second = client.delete(f"/workouts/{ids[0]}")

    assert second.status_code == 404
    assert second.json() == NOT_FOUND
    assert load(db, ids[1]).set_number == 1


def test_no_token_is_401(db):
    workout_id = add_set(db, USER_A, DAY, "bench", 1, 200, 5)

    response = TestClient(main.app).delete(f"/workouts/{workout_id}")

    assert response.status_code == 401
    assert load(db, workout_id) is not None


def test_invalid_token_is_401(db, monkeypatch):
    from types import SimpleNamespace
    from supabase import AuthApiError

    class FakeAuth:
        def get_user(self, token):
            raise AuthApiError("invalid JWT", 403, "bad_jwt")

    monkeypatch.setattr(main, "supabase", SimpleNamespace(auth=FakeAuth()))
    workout_id = add_set(db, USER_A, DAY, "bench", 1, 200, 5)

    response = TestClient(main.app).delete(f"/workouts/{workout_id}", headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired token"}
    assert load(db, workout_id) is not None


def test_failure_after_delete_and_renumber_rolls_back_both(client, db, monkeypatch):
    ids = add_sets(db, USER_A, DAY, "bench", [200, 205, 210])

    class FailingCommitSession(PostgresLikeSession):
        def commit(self):
            # The DELETE and the UPDATEs have already run inside the
            # transaction; fail before it is committed.
            self.flush()
            raise RuntimeError("database went away")

    monkeypatch.setattr(main, "SessionLocal", sessionmaker(bind=db.kw["bind"], class_=FailingCommitSession))

    response = client.delete(f"/workouts/{ids[0]}")

    assert response.status_code == 500
    assert response.json() == {"detail": "Could not delete the workout set. Please try again."}
    assert "database went away" not in response.text
    assert [(load(db, i).id, load(db, i).set_number) for i in ids] == [(ids[0], 1), (ids[1], 2), (ids[2], 3)]


# ── POST /workouts and the next set number ─────────────────────

def test_post_returns_the_new_rows_id(client, db):
    response = client.post("/workouts", json={
        "date": DAY.isoformat(), "exercise": "bench", "set_number": 1, "weight": 200, "reps": 5,
    })

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["id"], int)
    assert load(db, body["id"]).weight == 200
    assert set(body) == {"id", "user_id", "date", "exercise", "set_number", "weight", "reps"}


def test_next_set_after_a_delete_gets_the_next_number(client, db):
    # Log three sets the way the app does, delete the middle one, then log
    # another: the client sends (sets it now has for that exercise and day) + 1.
    logged = []
    for n, weight in enumerate([200, 205, 210], start=1):
        logged.append(client.post("/workouts", json={
            "date": DAY.isoformat(), "exercise": "bench", "set_number": n, "weight": weight, "reps": 5,
        }).json()["id"])

    deleted = client.delete(f"/workouts/{logged[1]}").json()
    assert deleted["renumbered"] == [{"id": logged[2], "set_number": 2}]

    next_number = len(sets_of(client, "bench", DAY)) + 1
    assert next_number == 3
    new_id = client.post("/workouts", json={
        "date": DAY.isoformat(), "exercise": "bench", "set_number": next_number, "weight": 215, "reps": 5,
    }).json()["id"]

    assert [(r[0], r[1]) for r in sets_of(client, "bench", DAY)] == [(logged[0], 1), (logged[2], 2), (new_id, 3)]


# ── Trends and critique after deletes ──────────────────────────

def history(Session, user):
    session = Session()
    try:
        rows = (session.query(Workout).filter(Workout.user_id == uuid.UUID(user))
                .order_by(Workout.date, Workout.id).all())
        return [{"date": r.date, "exercise": r.exercise, "set_number": r.set_number,
                 "weight": r.weight, "reps": r.reps} for r in rows]
    finally:
        session.close()


@pytest.fixture
def pipeline_on_test_db(db, monkeypatch):
    """Point the trend code at the same test database."""
    monkeypatch.setattr(pipeline, "get_all_workouts", lambda user: history(db, user))
    monkeypatch.setattr(pipeline, "get_unique_exercises",
                        lambda user: sorted({h["exercise"] for h in history(db, user)}))
    monkeypatch.setattr(pipeline, "filter_by_exercise",
                        lambda user, exercise: [h for h in history(db, user) if h["exercise"] == exercise])


def seed_five_bench_sessions(db):
    ids = {}
    for i, day in enumerate([date(2026, 10, d) for d in (1, 4, 7, 10, 13)]):
        ids[day] = add_sets(db, USER_A, day, "bench", [150 + 10 * i, 140 + 10 * i])
    return ids


def test_trends_after_deleting_one_set_from_a_multi_set_session(client, db, pipeline_on_test_db):
    ids = seed_five_bench_sessions(db)
    last_day = max(ids)
    client.delete(f"/workouts/{ids[last_day][0]}")  # the best set of the latest session

    trends = pipeline.get_all_exercise_trends(USER_A)
    latest = pipeline.get_latest_workout(history(db, USER_A), today=last_day)

    assert trends["bench"]["sessions_logged"] == 5
    assert trends["bench"]["current_1rm"] == pytest.approx(180 * (1 + 5 / 30))
    assert latest["sets"] == [{"exercise": "bench", "set_number": 1, "weight": 180, "reps": 5}]
    assert "LATEST WORKOUT" in pipeline.build_prompt(USER_A, "180cm", "80kg", 12)


def test_trends_after_deleting_the_only_set_of_a_day(client, db, pipeline_on_test_db):
    ids = seed_five_bench_sessions(db)
    last_day = max(ids)
    for workout_id in ids[last_day]:
        client.delete(f"/workouts/{workout_id}")

    trends = pipeline.get_all_exercise_trends(USER_A)
    latest = pipeline.get_latest_workout(history(db, USER_A), today=last_day)

    assert trends["bench"]["sessions_logged"] == 4
    assert latest["date"] == "2026-10-10"
    assert latest["is_today"] is False
    pipeline.build_prompt(USER_A, "180cm", "80kg", 12)


def test_trends_and_prompt_for_a_user_with_no_workouts_left(client, db, pipeline_on_test_db):
    only = add_set(db, USER_A, DAY, "bench", 1, 200, 5)
    client.delete(f"/workouts/{only}")

    assert pipeline.get_all_exercise_trends(USER_A) == {}
    assert pipeline.get_latest_workout(history(db, USER_A)) is None
    prompt = pipeline.build_prompt(USER_A, "180cm", "80kg", 12)
    assert "If no workouts have been logged, say so" in prompt


def test_critique_for_a_user_with_no_workouts_left_still_works(client, db, pipeline_on_test_db, monkeypatch):
    only = add_set(db, USER_A, DAY, "bench", 1, 200, 5)
    client.delete(f"/workouts/{only}")

    prompts = []
    monkeypatch.setattr(main, "check_daily_critique_limit", lambda user: (0, None))
    monkeypatch.setattr(main, "get_profile",
                        lambda user: {"height": "180cm", "bodyweight": "80kg", "training_start_date": "2024-01-01"})
    monkeypatch.setattr(main, "get_critique",
                        lambda **kw: prompts.append(pipeline.build_prompt(**kw)) or "No workouts logged yet.")

    class NoopSession:
        def add(self, row): pass
        def commit(self): pass
        def close(self): pass

    monkeypatch.setattr(main, "SessionLocal", NoopSession)

    response = client.post("/critique")

    assert response.status_code == 200
    assert response.json()["critique"] == "No workouts logged yet."
    # Gemini was asked with an empty history and told what to do about it.
    assert "LATEST WORKOUT" in prompts[0]
    assert "If no workouts have been logged, say so" in prompts[0]

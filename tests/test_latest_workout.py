from datetime import date

import pytest

import pipeline
from pipeline import get_latest_workout

TODAY = date(2026, 10, 16)


def workout_set(day, exercise, set_number, weight, reps):
    return {"date": day, "exercise": exercise, "set_number": set_number, "weight": weight, "reps": reps}


def by_exercise(latest):
    return {e["exercise"]: e for e in latest["exercises"]}


def test_latest_workout_is_every_set_on_the_most_recent_date():
    history = [
        workout_set(date(2026, 10, 10), "squat", 1, 300, 5),
        workout_set(date(2026, 10, 10), "bench", 1, 200, 5),
        workout_set(date(2026, 10, 16), "bench", 2, 205, 4),
        workout_set(date(2026, 10, 16), "row", 1, 150, 8),
        workout_set(date(2026, 10, 16), "bench", 1, 210, 3),
        workout_set(date(2026, 10, 13), "deadlift", 1, 400, 3),
    ]

    latest = get_latest_workout(history, today=TODAY)

    assert latest["date"] == "2026-10-16"
    assert latest["sets"] == [
        {"exercise": "bench", "set_number": 1, "weight": 210, "reps": 3},
        {"exercise": "bench", "set_number": 2, "weight": 205, "reps": 4},
        {"exercise": "row", "set_number": 1, "weight": 150, "reps": 8},
    ]
    assert [e["exercise"] for e in latest["exercises"]] == ["bench", "row"]


def test_no_workouts_gives_none():
    assert get_latest_workout([], today=TODAY) is None


@pytest.mark.parametrize("previous, latest_weight", [(200, 180), (20, 18)])
def test_percent_change_vs_previous_session_heavy_and_light(previous, latest_weight):
    history = [
        workout_set(date(2026, 10, 9), "lift", 1, previous, 5),
        workout_set(date(2026, 10, 16), "lift", 1, latest_weight, 5),
    ]

    lift = by_exercise(get_latest_workout(history, today=TODAY))["lift"]

    assert lift["first_time_logged"] is False
    assert lift["previous_session_date"] == "2026-10-09"
    assert lift["percent_change_vs_previous"] == -10.0


def test_comparison_uses_best_set_of_each_session_and_only_the_previous_session():
    history = [
        workout_set(date(2026, 10, 1), "bench", 1, 100, 5),   # older session, ignored
        workout_set(date(2026, 10, 9), "bench", 1, 190, 5),
        workout_set(date(2026, 10, 9), "bench", 2, 200, 5),   # best set on Oct 9
        workout_set(date(2026, 10, 16), "bench", 1, 210, 5),  # best set on Oct 16
        workout_set(date(2026, 10, 16), "bench", 2, 180, 5),
    ]

    bench = by_exercise(get_latest_workout(history, today=TODAY))["bench"]

    assert bench["previous_session_date"] == "2026-10-09"
    assert bench["percent_change_vs_previous"] == 5.0


def test_exercise_with_no_earlier_session_is_first_time_logged():
    history = [
        workout_set(date(2026, 10, 9), "bench", 1, 200, 5),
        workout_set(date(2026, 10, 16), "bench", 1, 200, 5),
        workout_set(date(2026, 10, 16), "overhead press", 1, 95, 5),
    ]

    exercises = by_exercise(get_latest_workout(history, today=TODAY))

    press = exercises["overhead press"]
    assert press["first_time_logged"] is True
    assert "percent_change_vs_previous" not in press
    assert "previous_session_date" not in press
    assert exercises["bench"]["percent_change_vs_previous"] == 0.0


def test_exercises_are_matched_by_exact_name():
    history = [
        workout_set(date(2026, 10, 9), "bench press", 1, 200, 5),
        workout_set(date(2026, 10, 16), "Barbell Bench Press", 1, 200, 5),
    ]

    latest = get_latest_workout(history, today=TODAY)

    assert by_exercise(latest)["Barbell Bench Press"]["first_time_logged"] is True


def test_workout_today():
    latest = get_latest_workout([workout_set(TODAY, "bench", 1, 200, 5)], today=TODAY)
    assert latest["is_today"] is True
    assert latest["days_ago"] == 0


def test_workout_not_today():
    latest = get_latest_workout([workout_set(date(2026, 10, 12), "bench", 1, 200, 5)], today=TODAY)
    assert latest["date"] == "2026-10-12"
    assert latest["is_today"] is False
    assert latest["days_ago"] == 4


def test_string_dates_are_accepted():
    latest = get_latest_workout([workout_set("2026-10-16", "bench", 1, 200, 5)], today=TODAY)
    assert latest["date"] == "2026-10-16"
    assert latest["is_today"] is True


HEADINGS = (
    "1. Workout Summary\n"
    "2. Strong Points\n"
    "3. Weak Points / Plateaus\n"
    "4. Muscles Not Being Trained\n"
    "5. Next Workout Goals\n"
)


def build_prompt_with(monkeypatch, history):
    monkeypatch.setattr(pipeline, "get_all_workouts", lambda user_id: history)
    monkeypatch.setattr(pipeline, "get_all_exercise_trends", lambda user_id: {})
    return pipeline.build_prompt("user", "180cm", "80kg", 12)


def test_prompt_includes_latest_workout_block_and_section_1_rules(monkeypatch):
    history = [
        workout_set(date(2026, 10, 9), "bench", 1, 200, 5),
        workout_set(date(2026, 10, 16), "bench", 1, 180, 5),
    ]

    prompt = build_prompt_with(monkeypatch, history)

    assert "LATEST WORKOUT:" in prompt
    assert '"percent_change_vs_previous": -10.0' in prompt
    assert "SECTION 1 (Workout Summary) covers ONLY the latest workout" in prompt
    assert "critique of their latest workout" in prompt
    assert prompt.endswith("Respond in exactly this structure, using these section headers:\n\n" + HEADINGS)


def test_prompt_builds_with_no_workouts(monkeypatch):
    prompt = build_prompt_with(monkeypatch, [])

    assert "LATEST WORKOUT:" in prompt
    assert prompt.endswith(HEADINGS)

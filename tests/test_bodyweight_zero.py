from datetime import date

import pytest

import pipeline
from pipeline import _percent_change, classify_trend, get_latest_workout

TODAY = date(2026, 10, 16)


def summaries(one_rms):
    return [
        {"Date": date(2026, 9, 1 + 3 * i), "estimated_1rm": v, "actual_weight": v, "actual_reps": 8}
        for i, v in enumerate(one_rms)
    ]


def workout_set(day, exercise, set_number, weight, reps):
    return {"date": day, "exercise": exercise, "set_number": set_number, "weight": weight, "reps": reps}


@pytest.mark.parametrize("new", [0, 10, -5])
def test_percent_change_from_zero_is_none(new):
    assert _percent_change(0, new) is None


def test_percent_change_from_negative_is_none():
    assert _percent_change(-10, 5) is None


def test_percent_change_from_positive_is_unchanged():
    assert _percent_change(200, 180) == -10.0
    assert _percent_change(200, 0) == -100.0


@pytest.mark.parametrize("count", [4, 5])
def test_all_zero_sessions_are_not_enough_data(count):
    assert classify_trend(summaries([0] * count)) == {
        "trend": "not enough data",
        "sessions_logged": count,
    }


@pytest.mark.parametrize("one_rms", [
    [0, 100, 100, 100, 100],      # starting session at 0
    [100, 100, 0, 0, 100],        # earlier window averages 0
    [100, 100, 100, 0, 0],        # the two sessions before the latest average 0... and the last
    [100, 0, 0, 100, 100],
    [0, 0, 0, 0, 100],
])
def test_mixed_zero_sessions_do_not_raise(one_rms):
    result = classify_trend(summaries(one_rms))
    assert isinstance(result["trend"], str)


def test_zero_denominator_gives_not_enough_data_without_percentages():
    result = classify_trend(summaries([0, 100, 100, 100, 100]))
    assert result == {"trend": "not enough data", "sessions_logged": 5}


def test_positive_history_dropping_to_zero_still_classifies():
    # Every denominator is positive, so the normal rules apply.
    result = classify_trend(summaries([200, 200, 200, 200, 0]))
    assert result["trend"] == "sharp recent drop (unconfirmed)"
    assert result["drop_percent"] == 100.0


def test_latest_workout_with_previous_zero_gives_null_change():
    history = [
        workout_set(date(2026, 10, 9), "pull-up", 1, 0, 8),
        workout_set(date(2026, 10, 16), "pull-up", 1, 0, 10),
        workout_set(date(2026, 10, 9), "bench", 1, 200, 5),
        workout_set(date(2026, 10, 16), "bench", 1, 180, 5),
    ]

    exercises = {e["exercise"]: e for e in get_latest_workout(history, today=TODAY)["exercises"]}

    pull_up = exercises["pull-up"]
    assert pull_up["first_time_logged"] is False
    assert pull_up["previous_session_date"] == "2026-10-09"
    assert pull_up["percent_change_vs_previous"] is None
    assert exercises["bench"]["percent_change_vs_previous"] == -10.0


def test_latest_workout_from_zero_to_added_weight_gives_null_change():
    history = [
        workout_set(date(2026, 10, 9), "pull-up", 1, 0, 8),
        workout_set(date(2026, 10, 16), "pull-up", 1, 25, 5),
    ]

    pull_up = get_latest_workout(history, today=TODAY)["exercises"][0]
    assert pull_up["percent_change_vs_previous"] is None
    assert pull_up["first_time_logged"] is False


def test_full_prompt_builds_with_bodyweight_exercise_at_zero(monkeypatch):
    days = [date(2026, 9, 1 + 3 * i) for i in range(6)]
    history = []
    for i, day in enumerate(days):
        history.append(workout_set(day, "pull-up", 1, 0, 8 + i % 2))
        history.append(workout_set(day, "bench", 1, 200 + 5 * i, 5))
    # One mixed exercise: started at bodyweight, later added weight.
    history.append(workout_set(days[0], "dip", 1, 0, 10))
    history.append(workout_set(days[1], "dip", 1, 0, 12))
    history.append(workout_set(days[2], "dip", 1, 10, 8))
    history.append(workout_set(days[3], "dip", 1, 15, 8))
    history.append(workout_set(days[4], "dip", 1, 0, 15))
    history.sort(key=lambda s: s["date"])

    monkeypatch.setattr(pipeline, "get_all_workouts", lambda user_id: history)
    monkeypatch.setattr(pipeline, "get_unique_exercises",
                        lambda user_id: sorted({s["exercise"] for s in history}))
    monkeypatch.setattr(pipeline, "filter_by_exercise",
                        lambda user_id, exercise: [s for s in history if s["exercise"] == exercise])

    prompt = pipeline.build_prompt("user", "180cm", "80kg", 12)

    assert '"percent_change_vs_previous": null' in prompt
    assert "no percentage" in prompt
    assert prompt.endswith(
        "1. Workout Summary\n2. Strong Points\n3. Weak Points / Plateaus\n"
        "4. Muscles Not Being Trained\n5. Next Workout Goals\n"
    )

    trends = pipeline.get_all_exercise_trends("user")
    assert trends["pull-up"] == {"trend": "not enough data", "sessions_logged": 6}
    assert trends["bench"]["trend"] == "improving"

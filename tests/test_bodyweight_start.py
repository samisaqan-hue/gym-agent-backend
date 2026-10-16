from datetime import date, timedelta

import pytest

import pipeline
from pipeline import MIN_SESSIONS, classify_trend, get_latest_workout

TODAY = date(2026, 10, 16)


def summaries(one_rms, reps=5):
    return [
        {"Date": date(2026, 9, 1) + timedelta(days=3 * i), "estimated_1rm": v,
         "actual_weight": v, "actual_reps": 0 if v == 0 else reps}
        for i, v in enumerate(one_rms)
    ]


WEIGHTED = {
    "improving": [150, 160, 170, 180, 190],
    "flat": [200, 200, 200, 155, 200],
    "declining": [200, 195, 190, 185],
    "plateau": [200, 200, 200, 200],
    "sharp recent drop (unconfirmed)": [200, 205, 195, 200, 155],
}


@pytest.mark.parametrize("label, weighted", WEIGHTED.items())
@pytest.mark.parametrize("leading_zeros", [1, 3, 5])
def test_leading_bodyweight_sessions_classify_like_the_weighted_history(label, weighted, leading_zeros):
    expected = classify_trend(summaries(weighted))
    assert expected["trend"] == label

    result = classify_trend(summaries([0] * leading_zeros + weighted))

    # Same label, same percentages, same extra fields, same session count.
    assert result == expected


@pytest.mark.parametrize("weighted_count", range(1, MIN_SESSIONS))
def test_too_few_weighted_sessions_after_zeros_is_not_enough_data(weighted_count):
    weighted = [100 + 5 * i for i in range(weighted_count)]
    assert classify_trend(summaries([0] * 5 + weighted)) == {
        "trend": "not enough data",
        "sessions_logged": weighted_count,
    }


@pytest.mark.parametrize("count", [1, 3, 4, 5, 8])
def test_all_bodyweight_history_is_unchanged(count):
    assert classify_trend(summaries([0] * count)) == {
        "trend": "not enough data",
        "sessions_logged": count,
    }


@pytest.mark.parametrize("one_rms, expected", [
    # Zero in the middle: kept, classified exactly as before this change.
    ([10, 15, 0, 20, 25], {"trend": "improving", "sessions_logged": 5,
                           "starting_1rm": 10, "current_1rm": 25, "percent_change": 150.0}),
    # Zero at the end: a real drop back to bodyweight.
    ([10, 15, 20, 25, 0], {"trend": "sharp recent drop (unconfirmed)", "percent_change": -100.0,
                           "drop_percent": 100.0, "sessions_logged": 5}),
    # Zero in the window the trend divides by: still "not enough data", as before.
    ([100, 100, 0, 0, 100], {"trend": "not enough data", "sessions_logged": 5}),
])
def test_zero_in_the_middle_or_at_the_end_is_unchanged(one_rms, expected):
    result = classify_trend(summaries(one_rms))
    for key, value in expected.items():
        assert result[key] == value


def test_only_leading_zeros_are_skipped():
    # The leading zeros go; the later zero stays and is part of the trend.
    assert classify_trend(summaries([0, 0, 10, 15, 0, 20, 25])) == classify_trend(summaries([10, 15, 0, 20, 25]))


def workout_set(day, exercise, set_number, weight, reps):
    return {"date": day, "exercise": exercise, "set_number": set_number, "weight": weight, "reps": reps}


def use_fake_workouts(monkeypatch, weights_by_exercise):
    history = [
        workout_set(date(2026, 9, 1) + timedelta(days=3 * i), exercise, 1, w, 5)
        for exercise, weights in weights_by_exercise.items()
        for i, w in enumerate(weights)
    ]
    monkeypatch.setattr(pipeline, "get_unique_exercises", lambda user_id: list(weights_by_exercise))
    monkeypatch.setattr(pipeline, "filter_by_exercise",
                        lambda user_id, exercise: [s for s in history if s["exercise"] == exercise])
    return history


def test_other_lifts_check_uses_the_same_starting_point(monkeypatch):
    use_fake_workouts(monkeypatch, {
        "bench": [200, 200, 200, 200, 170],
        # With its leading zero this looks like a 33% drop (150 -> 100), but
        # from its first weighted session there are too few sessions to say.
        "dip": [0, 300, 100],
    })

    trends = pipeline.get_all_exercise_trends("user")

    assert trends["bench"]["trend"] == "sharp recent drop (unconfirmed)"
    assert trends["bench"]["other_lifts_also_dropped"] is False


def test_other_lifts_check_still_sees_a_weighted_drop_after_zeros(monkeypatch):
    use_fake_workouts(monkeypatch, {
        "bench": [200, 200, 200, 200, 170],
        "dip": [0, 0, 50, 50, 50, 50, 40],
    })

    trends = pipeline.get_all_exercise_trends("user")

    assert trends["dip"]["trend"] == "sharp recent drop (unconfirmed)"
    assert trends["dip"]["sessions_logged"] == 5
    assert trends["bench"]["other_lifts_also_dropped"] is True
    assert trends["dip"]["other_lifts_also_dropped"] is True


def test_latest_workout_bodyweight_then_weighted_has_no_percent(monkeypatch):
    history = [
        workout_set(date(2026, 10, 9), "pull-up", 1, 0, 8),
        workout_set(date(2026, 10, 16), "pull-up", 1, 10, 6),
    ]

    pull_up = get_latest_workout(history, today=TODAY)["exercises"][0]

    assert pull_up == {
        "exercise": "pull-up",
        "best_estimated_1rm": 12.0,
        "first_time_logged": False,
        "previous_session_date": "2026-10-09",
        "previous_best_estimated_1rm": 0.0,
        "percent_change_vs_previous": None,
        "latest_session_bodyweight": False,
    }

    monkeypatch.setattr(pipeline, "get_all_workouts", lambda user_id: history)
    monkeypatch.setattr(pipeline, "get_all_exercise_trends", lambda user_id: {})
    prompt = pipeline.build_prompt("user", "180cm", "80kg", 12)
    assert '"percent_change_vs_previous": null' in prompt
    assert prompt.endswith(
        "1. Workout Summary\n2. Strong Points\n3. Weak Points / Plateaus\n"
        "4. Muscles Not Being Trained\n5. Next Workout Goals\n"
    )


SECTION_1_NULL_RULE = """- If "percent_change_vs_previous" is null for an exercise, the previous session was 
  logged at weight 0, so there is no percentage comparison. Do not calculate or invent 
  any comparison or percentage yourself.
  - If "latest_session_bodyweight" is true, say there is no percentage comparison 
    available for it (it is probably a bodyweight exercise logged at weight 0).
  - If "latest_session_bodyweight" is false, say the user added weight this session 
    after a previous session at bodyweight, and that no percentage comparison is 
    available. Do not call it a bodyweight exercise.
"""

OLD_BODYWEIGHT_WORDING = (
    "say there is no percentage comparison \n"
    "    available for it (it is probably a bodyweight exercise logged at weight 0)."
)


def prompt_for(monkeypatch, history):
    monkeypatch.setattr(pipeline, "get_all_workouts", lambda user_id: history)
    monkeypatch.setattr(pipeline, "get_all_exercise_trends", lambda user_id: {})
    return pipeline.build_prompt("user", "180cm", "80kg", 12)


def test_prompt_for_weight_added_after_bodyweight(monkeypatch):
    history = [
        workout_set(date(2026, 10, 9), "pull-up", 1, 0, 8),
        workout_set(date(2026, 10, 16), "pull-up", 1, 10, 6),
    ]

    pull_up = get_latest_workout(history)["exercises"][0]
    assert pull_up["percent_change_vs_previous"] is None
    assert pull_up["latest_session_bodyweight"] is False
    assert pull_up["first_time_logged"] is False

    prompt = prompt_for(monkeypatch, history)

    assert '"percent_change_vs_previous": null' in prompt
    assert '"latest_session_bodyweight": false' in prompt
    assert SECTION_1_NULL_RULE in prompt
    assert prompt.endswith(
        "1. Workout Summary\n2. Strong Points\n3. Weak Points / Plateaus\n"
        "4. Muscles Not Being Trained\n5. Next Workout Goals\n"
    )


def test_prompt_for_always_bodyweight_lift_keeps_the_old_wording(monkeypatch):
    history = [
        workout_set(date(2026, 10, 9), "pull-up", 1, 0, 8),
        workout_set(date(2026, 10, 16), "pull-up", 1, 0, 10),
    ]

    pull_up = get_latest_workout(history)["exercises"][0]
    assert pull_up["percent_change_vs_previous"] is None
    assert pull_up["latest_session_bodyweight"] is True

    prompt = prompt_for(monkeypatch, history)

    assert '"latest_session_bodyweight": true' in prompt
    assert OLD_BODYWEIGHT_WORDING in prompt


def test_lifts_with_a_percentage_get_no_new_field():
    history = [
        workout_set(date(2026, 10, 9), "bench", 1, 200, 5),
        workout_set(date(2026, 10, 16), "bench", 1, 180, 5),
        workout_set(date(2026, 10, 16), "dip", 1, 0, 10),
    ]

    exercises = {e["exercise"]: e for e in get_latest_workout(history, today=TODAY)["exercises"]}

    assert "latest_session_bodyweight" not in exercises["bench"]
    assert "latest_session_bodyweight" not in exercises["dip"]  # first time logged

import json
from datetime import date, timedelta

import pytest

import pipeline
from pipeline import classify_trend

UNCONFIRMED = "sharp recent drop (unconfirmed)"


def make_summaries(one_rms, reps=5, start=date(2026, 9, 1), gap_days=3, last_gap_days=None, last_reps=None):
    """Session summaries shaped like build_session_summaries' output."""
    summaries = []
    day = start
    for i, one_rm in enumerate(one_rms):
        is_last = i == len(one_rms) - 1
        if i > 0:
            day += timedelta(days=last_gap_days if is_last and last_gap_days is not None else gap_days)
        summaries.append({
            "Date": day,
            "estimated_1rm": one_rm,
            "actual_weight": one_rm,
            "actual_reps": last_reps if is_last and last_reps is not None else reps,
        })
    return summaries


def trend_of(one_rms):
    return classify_trend(make_summaries(one_rms))["trend"]


def test_sudden_drop_after_steady_sessions_is_unconfirmed_not_plateau():
    result = classify_trend(make_summaries([200, 205, 195, 200, 155]))

    assert result["trend"] == UNCONFIRMED
    assert result["percent_change"] == -22.5
    assert result["baseline_1rm"] == 197.5
    assert result["latest_1rm"] == 155
    assert result["drop_percent"] == 21.5


def test_drop_after_earlier_gains_is_not_improving():
    assert trend_of([150, 170, 190, 200, 150]) != "improving"


def test_consecutive_drops_are_declining():
    assert trend_of([200, 195, 190, 185]) == "declining"


@pytest.mark.parametrize("one_rms", [
    [200, 200, 200, 155, 155],
    [200, 200, 200, 200, 155, 155],
])
def test_two_low_sessions_in_a_row_are_declining(one_rms):
    assert trend_of(one_rms) == "declining"


@pytest.mark.parametrize("fourth", [185, 189.9, 150])
def test_three_drops_in_a_row_are_declining(fourth):
    result = classify_trend(make_summaries([200, 195, 190, fourth]))
    assert result["trend"] == "declining"
    assert result["declining_streak"] == 4


def test_two_drops_in_a_row_are_not_declining():
    assert trend_of([200, 200, 195, 190]) != "declining"


def test_old_losing_streak_that_recovered_is_not_declining():
    assert trend_of([200, 190, 180, 170, 175, 185, 195, 205]) != "declining"


def test_losing_streak_must_end_at_latest_session():
    # Three drops, then one session up: the streak is broken.
    assert trend_of([200, 195, 190, 185, 186]) != "declining"


def test_flat_sessions_are_plateau():
    assert trend_of([200, 200, 200, 200]) == "plateau"


def test_rising_sessions_are_improving():
    assert trend_of([150, 160, 170, 180, 190]) == "improving"


@pytest.mark.parametrize("one_rms", [[], [200], [200, 190], [200, 150, 100]])
def test_fewer_than_four_sessions_is_not_enough_data(one_rms):
    assert classify_trend(make_summaries(one_rms)) == {
        "trend": "not enough data",
        "sessions_logged": len(one_rms),
    }


def test_single_drop_then_bounce_back_is_neither_declining_nor_unconfirmed():
    trend = trend_of([200, 200, 200, 155, 200])
    assert trend not in ("declining", UNCONFIRMED)


def test_unconfirmed_drop_then_second_low_session_becomes_declining():
    assert trend_of([200, 200, 200, 200, 155]) == UNCONFIRMED
    assert trend_of([200, 200, 200, 200, 155, 155]) == "declining"


def test_rules_use_percentages_so_light_and_heavy_lifts_match():
    heavy = [200, 205, 195, 200, 155]
    light = [v / 10 for v in heavy]  # same shape, a tenth of the weight
    assert trend_of(heavy) == trend_of(light) == UNCONFIRMED


def test_same_percentage_drop_gives_same_result_for_heavy_and_light_lift():
    heavy = classify_trend(make_summaries([300, 300, 300, 300, 240]))
    light = classify_trend(make_summaries([30, 30, 30, 30, 24]))

    assert heavy["trend"] == light["trend"] == UNCONFIRMED
    for key in ("drop_percent", "percent_change", "likely_typo"):
        assert heavy[key] == light[key]
    assert heavy["drop_percent"] == 20.0


def test_small_dip_is_not_a_sharp_drop():
    # 8% below the two sessions before it: under SHARP_DROP_PCT.
    assert trend_of([200, 200, 200, 200, 184]) == "plateau"


def test_extra_facts_when_nothing_unusual():
    result = classify_trend(make_summaries([200, 200, 200, 200, 170], reps=5, gap_days=3))

    assert result["trend"] == UNCONFIRMED
    assert result["drop_percent"] == 15.0
    assert result["days_since_previous_session"] == 3
    assert result["long_gap"] is False
    assert result["latest_reps"] == 5
    assert result["typical_reps"] == 5
    assert result["high_rep_set"] is False
    assert result["likely_typo"] is False


def test_long_gap_flag():
    result = classify_trend(make_summaries([200, 200, 200, 200, 170], last_gap_days=14))
    assert result["days_since_previous_session"] == 14
    assert result["long_gap"] is True

    result = classify_trend(make_summaries([200, 200, 200, 200, 170], last_gap_days=13))
    assert result["long_gap"] is False


def test_high_rep_set_flag():
    result = classify_trend(make_summaries([200, 200, 200, 200, 170], reps=5, last_reps=11))
    assert result["latest_reps"] == 11
    assert result["typical_reps"] == 5
    assert result["high_rep_set"] is True

    result = classify_trend(make_summaries([200, 200, 200, 200, 170], last_reps=10))
    assert result["high_rep_set"] is False


def test_likely_typo_flag():
    assert classify_trend(make_summaries([200, 200, 200, 200, 160]))["likely_typo"] is True   # 20% drop
    assert classify_trend(make_summaries([200, 200, 200, 200, 162]))["likely_typo"] is False  # 19%


def test_existing_keys_are_kept():
    for one_rms in ([200, 195, 190, 185], [200, 200, 200, 200], [150, 160, 170, 180], [200, 200, 200, 200, 155]):
        result = classify_trend(make_summaries(one_rms))
        for key in ("trend", "sessions_logged", "starting_1rm", "current_1rm", "percent_change"):
            assert key in result


def use_fake_workouts(monkeypatch, sessions_by_exercise):
    monkeypatch.setattr(pipeline, "get_unique_exercises", lambda user_id: list(sessions_by_exercise))

    def fake_filter(user_id, exercise):
        return [
            {"date": s["Date"], "exercise": exercise, "set_number": 1,
             "weight": s["estimated_1rm"] / (1 + s["actual_reps"] / 30), "reps": s["actual_reps"]}
            for s in sessions_by_exercise[exercise]
        ]

    monkeypatch.setattr(pipeline, "filter_by_exercise", fake_filter)


def test_other_lifts_also_dropped_when_another_lift_drops_the_same_week(monkeypatch):
    use_fake_workouts(monkeypatch, {
        "bench": make_summaries([200, 200, 200, 200, 170]),
        "squat": make_summaries([300, 300, 300, 300, 250]),
        "row": make_summaries([100, 100, 100, 100, 100]),
    })

    trends = pipeline.get_all_exercise_trends("user")

    assert trends["bench"]["trend"] == UNCONFIRMED
    assert trends["bench"]["other_lifts_also_dropped"] is True
    assert trends["squat"]["other_lifts_also_dropped"] is True
    assert "other_lifts_also_dropped" not in trends["row"]


def test_other_lifts_also_dropped_is_false_when_only_one_lift_dropped(monkeypatch):
    use_fake_workouts(monkeypatch, {
        "bench": make_summaries([200, 200, 200, 200, 170]),
        "row": make_summaries([100, 100, 100, 100, 100]),
    })

    trends = pipeline.get_all_exercise_trends("user")

    assert trends["bench"]["other_lifts_also_dropped"] is False


def test_other_lifts_drop_long_before_does_not_count(monkeypatch):
    use_fake_workouts(monkeypatch, {
        "bench": make_summaries([200, 200, 200, 200, 170], start=date(2026, 9, 1)),
        "squat": make_summaries([300, 300, 300, 300, 250], start=date(2026, 6, 1)),
    })

    trends = pipeline.get_all_exercise_trends("user")

    assert trends["bench"]["other_lifts_also_dropped"] is False


def test_prompt_includes_new_label_guidance_and_trends(monkeypatch):
    use_fake_workouts(monkeypatch, {"bench": make_summaries([200, 205, 195, 200, 155])})
    monkeypatch.setattr(pipeline, "get_all_workouts", lambda user_id: [])

    prompt = pipeline.build_prompt("user", "180cm", "80kg", 12)

    assert '"trend": "sharp recent drop (unconfirmed)"' in prompt
    assert '"drop_percent": 21.5' in prompt
    assert "cannot know the cause" in prompt
    assert "see a professional" in prompt
    assert 'Only use the word "plateau"' in prompt
    for header in ("1. Workout Summary", "2. Strong Points", "3. Weak Points / Plateaus",
                   "4. Muscles Not Being Trained", "5. Next Workout Goals"):
        assert header in prompt
    assert "edit" not in prompt.lower()

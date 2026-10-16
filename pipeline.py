import json
import os
import statistics
from datetime import date, datetime
from zoneinfo import ZoneInfo
from google import genai
from database import SessionLocal, Workout, Profile

# ─────────────────────────────────────────────
# TREND SETTINGS (all drops and gains are percentages, not raw weight)
# ─────────────────────────────────────────────

MIN_SESSIONS = 4                # fewer sessions than this -> "not enough data"
DECLINING_STREAK_DROPS = 3      # this many drops in a row, ending at the latest session -> "declining"
DECLINE_WINDOW_DROP_PCT = 5     # both of the last two sessions this far below the two before them -> "declining"
SHARP_DROP_PCT = 10             # latest session this far below the two before it -> "sharp recent drop (unconfirmed)"
LIKELY_TYPO_DROP_PCT = 20       # a sharp drop at least this big may be a logging mistake
LONG_GAP_DAYS = 14              # this many days since the previous session counts as a long break
HIGH_REP_THRESHOLD = 10         # more reps than this makes the estimated 1RM less reliable
PLATEAU_WINDOW_DROP_PCT = 5     # plateau: latest two-session average no more than this % below the prior two...
IMPROVING_WINDOW_GAIN_PCT = 2   # ...and less than this % above them; this % or more -> "improving"
OTHER_LIFTS_WINDOW_DAYS = 7     # other lifts' sharp drops within this many days count as the same period

# "Today" for the latest-workout summary. Matches CRITIQUE_TIMEZONE in main.py,
# so a workout logged in the evening in Central Time still counts as today.
LATEST_WORKOUT_TIMEZONE = "America/Chicago"

# ─────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ─────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────

def filter_by_exercise(user_id, exercise):
    session = SessionLocal()
    results = (
        session.query(Workout)
        .filter(Workout.user_id == user_id)
        .filter(Workout.exercise == exercise)
        .order_by(Workout.date)
        .all()
    )
    session.close()

    return [
        {
            "date": row.date,
            "exercise": row.exercise,
            "set_number": row.set_number,
            "weight": row.weight,
            "reps": row.reps
        }
        for row in results
    ]


def group_by_date(filtered_entries):
    grouped_dates = {}
    for entry in filtered_entries:
        grouped_dates.setdefault(entry["date"], []).append(entry)
    return grouped_dates


def summarize_session(session_sets):
    best_1rm = 0
    best_weight = 0
    best_reps = 0
    session_date = session_sets[0]["date"]

    for s in session_sets:
        weight = s["weight"]
        reps = s["reps"]
        estimated_1rm = weight * (1 + reps / 30)

        if estimated_1rm > best_1rm:
            best_1rm = estimated_1rm
            best_weight = weight
            best_reps = reps

    return {
        "Date": session_date,
        "estimated_1rm": best_1rm,
        "actual_weight": best_weight,
        "actual_reps": best_reps
    }


def build_session_summaries(grouped_dates):
    summaries = []
    for date_key, sets in grouped_dates.items():
        summaries.append(summarize_session(sets))
    return summaries


def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    return value


def _percent_change(old, new):
    """Percentage change from old to new. Negative means a drop."""
    return ((new - old) / old) * 100


def latest_change_percent(sorted_summaries):
    """Percentage change from the average of the two sessions before the latest
    one to the latest one. None if there are fewer than 3 sessions."""
    if len(sorted_summaries) < 3:
        return None
    baseline = (sorted_summaries[-3]["estimated_1rm"] + sorted_summaries[-2]["estimated_1rm"]) / 2
    return _percent_change(baseline, sorted_summaries[-1]["estimated_1rm"])


def classify_trend(sorted_summaries, months_training=None):
    num_sessions = len(sorted_summaries)

    if num_sessions < MIN_SESSIONS:
        return {
            "trend": "not enough data",
            "sessions_logged": num_sessions
        }

    one_rms = [s["estimated_1rm"] for s in sorted_summaries]

    # Count drops in a row that end at the latest session. An older losing
    # streak that has since recovered doesn't count.
    recent_drops = 0
    for i in range(num_sessions - 1, 0, -1):
        if one_rms[i] < one_rms[i - 1]:
            recent_drops += 1
        else:
            break

    starting_1rm = one_rms[0]
    current_1rm = one_rms[-1]
    percent_change = round(_percent_change(starting_1rm, current_1rm), 1)

    base_result = {
        "sessions_logged": num_sessions,
        "starting_1rm": starting_1rm,
        "current_1rm": current_1rm,
        "percent_change": percent_change
    }

    # Both of the last two sessions clearly below the two before them. One bad
    # session on its own never counts.
    prior_two_avg = (one_rms[-4] + one_rms[-3]) / 2
    last_two_both_low = all(
        _percent_change(prior_two_avg, value) <= -DECLINE_WINDOW_DROP_PCT
        for value in one_rms[-2:]
    )

    if recent_drops >= DECLINING_STREAK_DROPS or last_two_both_low:
        return {
            "trend": "declining",
            # Number of sessions in the current downward run (drops + 1).
            "declining_streak": recent_drops + 1,
            **base_result
        }

    # A single low latest session: flag it, but don't call it a decline yet.
    latest_change = latest_change_percent(sorted_summaries)
    if latest_change <= -SHARP_DROP_PCT:
        latest = sorted_summaries[-1]
        previous = sorted_summaries[-2]
        baseline_1rm = (one_rms[-3] + one_rms[-2]) / 2
        drop_percent = round(-latest_change, 1)  # reported as a positive number
        days_since_previous = (_as_date(latest["Date"]) - _as_date(previous["Date"])).days
        latest_reps = latest["actual_reps"]
        return {
            "trend": "sharp recent drop (unconfirmed)",
            **base_result,
            "drop_percent": drop_percent,
            "baseline_1rm": round(baseline_1rm, 1),
            "latest_1rm": round(one_rms[-1], 1),
            "days_since_previous_session": days_since_previous,
            "long_gap": days_since_previous >= LONG_GAP_DAYS,
            "latest_reps": latest_reps,
            "typical_reps": statistics.median(s["actual_reps"] for s in sorted_summaries[:-1]),
            "high_rep_set": latest_reps > HIGH_REP_THRESHOLD,
            "likely_typo": drop_percent >= LIKELY_TYPO_DROP_PCT
        }

    # Judge the most recent window only, not whether the lifter ever improved.
    recent_two_avg = (one_rms[-2] + one_rms[-1]) / 2
    window_change = _percent_change(prior_two_avg, recent_two_avg)

    if window_change >= IMPROVING_WINDOW_GAIN_PCT:
        trend_label = "improving"
    elif window_change >= -PLATEAU_WINDOW_DROP_PCT:
        trend_label = "plateau"
    else:
        trend_label = "flat"

    return {
        "trend": trend_label,
        **base_result
    }


def get_unique_exercises(user_id):
    session = SessionLocal()
    results = (
        session.query(Workout.exercise)
        .filter(Workout.user_id == user_id)
        .distinct()
        .all()
    )
    session.close()
    return [row.exercise for row in results]


def get_all_exercise_trends(user_id):
    exercise_trends = {}
    sharp_drop_dates = {}
    exercise_names = get_unique_exercises(user_id)
    for exercise in exercise_names:
        filtered = filter_by_exercise(user_id, exercise)
        grouped = group_by_date(filtered)
        summaries = build_session_summaries(grouped)
        exercise_trends[exercise] = classify_trend(summaries)

        latest_change = latest_change_percent(summaries)
        if latest_change is not None and latest_change <= -SHARP_DROP_PCT:
            sharp_drop_dates[exercise] = _as_date(summaries[-1]["Date"])

    # If several lifts dropped sharply around the same time, the cause may be
    # general (sleep, illness, stress) rather than specific to one lift.
    for exercise, trend in exercise_trends.items():
        if trend["trend"] != "sharp recent drop (unconfirmed)":
            continue
        own_date = sharp_drop_dates[exercise]
        trend["other_lifts_also_dropped"] = any(
            abs((other_date - own_date).days) <= OTHER_LIFTS_WINDOW_DAYS
            for other, other_date in sharp_drop_dates.items()
            if other != exercise
        )
    return exercise_trends


def get_all_workouts(user_id):
    session = SessionLocal()
    results = (
        session.query(Workout)
        .filter(Workout.user_id == user_id)
        .order_by(Workout.date)
        .all()
    )
    session.close()

    return [
        {
            "date": row.date,
            "exercise": row.exercise,
            "set_number": row.set_number,
            "weight": row.weight,
            "reps": row.reps
        }
        for row in results
    ]


def get_profile(user_id):
    session = SessionLocal()
    result = session.query(Profile).filter(Profile.id == user_id).first()
    session.close()

    return {
        "height": result.height,
        "bodyweight": result.bodyweight,
        "training_start_date": result.training_start_date
    }


def calculate_months_training(training_start_date):
    if isinstance(training_start_date, str):
        start_date = datetime.strptime(training_start_date, "%Y-%m-%d").date()
    else:
        start_date = training_start_date

    today = date.today()
    months = (today.year - start_date.year) * 12 + (today.month - start_date.month)
    return max(months, 0)


def get_latest_workout(workout_data, today=None):
    """The user's most recent workout: every set logged on the latest date,
    plus, for each exercise in it, the percentage change in best estimated
    1RM versus that exercise's previous session. None if nothing is logged."""
    if not workout_data:
        return None

    if today is None:
        today = datetime.now(ZoneInfo(LATEST_WORKOUT_TIMEZONE)).date()

    latest_date = max(_as_date(entry["date"]) for entry in workout_data)
    latest_sets = [entry for entry in workout_data if _as_date(entry["date"]) == latest_date]

    exercise_order = []
    for entry in latest_sets:
        if entry["exercise"] not in exercise_order:
            exercise_order.append(entry["exercise"])
    latest_sets.sort(key=lambda entry: (exercise_order.index(entry["exercise"]), entry["set_number"]))

    exercises = []
    for exercise in exercise_order:
        # Exercises are matched by exact name.
        entries = [entry for entry in workout_data if entry["exercise"] == exercise]
        sessions = build_session_summaries(group_by_date(entries))
        sessions.sort(key=lambda summary: _as_date(summary["Date"]))
        latest = sessions[-1]
        earlier = [summary for summary in sessions if _as_date(summary["Date"]) < latest_date]

        comparison = {
            "exercise": exercise,
            "best_estimated_1rm": round(latest["estimated_1rm"], 1),
        }
        if earlier:
            previous = earlier[-1]
            comparison.update({
                "first_time_logged": False,
                "previous_session_date": _as_date(previous["Date"]).isoformat(),
                "previous_best_estimated_1rm": round(previous["estimated_1rm"], 1),
                "percent_change_vs_previous": round(
                    _percent_change(previous["estimated_1rm"], latest["estimated_1rm"]), 1
                ),
            })
        else:
            comparison["first_time_logged"] = True
        exercises.append(comparison)

    return {
        "date": latest_date.isoformat(),
        "is_today": latest_date == today,
        "days_ago": (today - latest_date).days,
        "sets": [
            {
                "exercise": entry["exercise"],
                "set_number": entry["set_number"],
                "weight": entry["weight"],
                "reps": entry["reps"],
            }
            for entry in latest_sets
        ],
        "exercises": exercises,
    }


# ─────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────

def build_prompt(user_id, height, bodyweight, months_training):
    workout_data = get_all_workouts(user_id)
    all_trends = get_all_exercise_trends(user_id)

    latest_workout = get_latest_workout(workout_data)

    workout_json = json.dumps(workout_data, indent=2, default=str)
    trends_json = json.dumps(all_trends, indent=2)
    latest_workout_json = json.dumps(latest_workout, indent=2)

    prompt = f"""You are an expert AI personal trainer. After each workout, you give the user an 
honest, statistics-driven critique of their latest workout, in the context of their 
full workout history.

USER CONTEXT:
- Height: {height}
- Bodyweight: {bodyweight}
- Experience: {months_training} months of training

WORKOUT DATA:
You will receive the user's full workout history as JSON, structured like this:

{workout_json}

Each workout entry contains one exercise, the date, and one set's weight/reps.

LATEST WORKOUT:
This is the user's most recent workout (calculated in code, not by you): its date, 
whether that date is today ("is_today") and how many days ago it was ("days_ago"), 
every set logged that day, and for each exercise a comparison with that exercise's 
previous session. "percent_change_vs_previous" is the percentage change in best 
estimated 1RM versus the previous session; "first_time_logged" is true when there 
is no earlier session to compare with. If it is null, no workouts have been logged.

{latest_workout_json}

COMPUTED TRENDS:
For each exercise, you will also be given pre-computed trend data (calculated in code, 
not by you) in this format:

{trends_json}

Treat this computed trend data as ground truth for whether something is a plateau, 
improvement, or decline — do not independently judge trend direction from the raw 
numbers alone. Use the raw set/rep/weight data to explain and add color to why 
the trend looks the way it does, not to override the computed trend label.

The possible trend labels are: "improving", "plateau", "flat", "declining", 
"sharp recent drop (unconfirmed)" and "not enough data".

"sharp recent drop (unconfirmed)" means only the latest session came in well below 
the sessions before it. It is NOT yet a decline. For an exercise with this label:
- State the drop neutrally using the exact "drop_percent" value. Do not calculate 
  any numbers yourself.
- Say the app cannot know the cause. Do not diagnose, assume an injury, or claim 
  any cause.
- Mention only the possible reasons supported by its fields, plus the always-valid 
  possibility that it was simply an easy or off day:
  - "likely_typo" is true: it may be a logging mistake; suggest double-checking the 
    weight and reps that were entered.
  - "long_gap" is true: there was a break of "days_since_previous_session" days since 
    the previous session.
  - "high_rep_set" is true: the latest set had "latest_reps" reps, and estimates 
    from high-rep sets are less reliable.
  - "other_lifts_also_dropped" is true: other lifts dropped around the same time, so 
    the cause may be general, such as sleep, stress or illness.
  Do not mention a reason whose field is false or missing.
- Say it will be flagged as a decline if the next sessions stay low.
- Include this safety line: if they feel pain or something feels wrong, they should 
  not push through it and should see a professional.

YOUR TASK:
Using the user's context, workout history, and computed trends, analyze:
1. The latest workout only (see SECTION 1 below)
2. Which exercises/muscle groups are trending well
3. Which exercises/muscle groups are plateauing or declining
4. Muscle groups that appear to be undertrained or skipped entirely
5. Specific next-workout goals: for each major exercise, state an exact weight and 
   rep target for their next session, calibrated to their experience level and 
   current trend

RULES:
- Be direct and honest. Do not use encouragement filler like "great job" or "keep 
  it up." Base every statement strictly on the data provided.
- If an exercise's trend is "not enough data", you must NOT mention any percentage 
  change, direction (up/down), or improvement/decline for that exercise anywhere in 
  your response — not even as a tentative observation. Simply state there is not 
  enough data yet, and nothing more about its trend. The only exception is Section 1: 
  there you may state its exact "percent_change_vs_previous" as a comparison with the 
  previous session only, while saying there is not yet enough data to call a trend.
- When referencing a percentage change for any exercise, use the exact "percent_change" 
  value provided in the computed trends data (or, in Section 1, the exact 
  "percent_change_vs_previous" value from the latest workout) — do not calculate or 
  estimate your own percentage.
- Only use the word "plateau" (or "plateauing") for an exercise whose computed trend 
  is exactly "plateau".
- Do not invent data that wasn't provided.

SECTION 1 (Workout Summary) covers ONLY the latest workout, not the full history:
- State the workout's date. If "is_today" is false, say plainly that this workout 
  was not today and was "days_ago" days ago.
- List which exercises were done and the sets (weight x reps) for each.
- For each exercise, say how it compared with its previous session using the exact 
  "percent_change_vs_previous" value and the "previous_session_date". If 
  "first_time_logged" is true, say it is the first time this exercise was logged, 
  so there is nothing to compare with yet.
- If no workouts have been logged, say so and nothing else in this section.
Sections 2-5 use the full workout history and the computed trends.

OUTPUT FORMAT:
Respond in exactly this structure, using these section headers:

1. Workout Summary
2. Strong Points
3. Weak Points / Plateaus
4. Muscles Not Being Trained
5. Next Workout Goals
"""

    return prompt


# ─────────────────────────────────────────────
# SEND TO GEMINI
# ─────────────────────────────────────────────

def get_critique(user_id, height, bodyweight, months_training):
    final_prompt = build_prompt(user_id, height, bodyweight, months_training)

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=final_prompt
    )

    return response.text
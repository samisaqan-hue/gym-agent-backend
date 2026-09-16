import json
import os
from datetime import date, datetime
from google import genai
from database import SessionLocal, Workout, Profile

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


def classify_trend(sorted_summaries, months_training=None):
    num_sessions = len(sorted_summaries)

    if num_sessions < 4:
        return {
            "trend": "not enough data",
            "sessions_logged": num_sessions
        }

    declining_streak = 1
    max_declining_streak = 1
    for i in range(1, num_sessions):
        current_1rm = sorted_summaries[i]["estimated_1rm"]
        previous_1rm = sorted_summaries[i - 1]["estimated_1rm"]
        if current_1rm < previous_1rm:
            declining_streak += 1
            max_declining_streak = max(max_declining_streak, declining_streak)
        else:
            declining_streak = 1

    starting_1rm = sorted_summaries[0]["estimated_1rm"]
    current_1rm = sorted_summaries[-1]["estimated_1rm"]
    percent_change = round(((current_1rm - starting_1rm) / starting_1rm) * 100, 1)

    if max_declining_streak >= 3:
        return {
            "trend": "declining",
            "sessions_logged": num_sessions,
            "declining_streak": max_declining_streak,
            "starting_1rm": starting_1rm,
            "current_1rm": current_1rm,
            "percent_change": percent_change
        }

    ever_improved = False
    plateau_streak = 0
    max_plateau_streak = 0

    for i in range(4, num_sessions + 1):
        recent_two = sorted_summaries[i - 2:i]
        prior_two = sorted_summaries[i - 4:i - 2]

        recent_avg = sum(s["estimated_1rm"] for s in recent_two) / 2
        prior_avg = sum(s["estimated_1rm"] for s in prior_two) / 2

        window_change = ((recent_avg - prior_avg) / prior_avg) * 100

        if window_change >= 2:
            ever_improved = True
            plateau_streak = 0
        else:
            plateau_streak += 1
            max_plateau_streak = max(max_plateau_streak, plateau_streak)

    if not ever_improved and max_plateau_streak >= 1 and num_sessions >= 4:
        return {
            "trend": "plateau",
            "sessions_logged": num_sessions,
            "starting_1rm": starting_1rm,
            "current_1rm": current_1rm,
            "percent_change": percent_change
        }

    if ever_improved:
        trend_label = "improving"
    else:
        trend_label = "flat"

    return {
        "trend": trend_label,
        "sessions_logged": num_sessions,
        "starting_1rm": starting_1rm,
        "current_1rm": current_1rm,
        "percent_change": percent_change
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
    exercise_names = get_unique_exercises(user_id)
    for exercise in exercise_names:
        filtered = filter_by_exercise(user_id, exercise)
        grouped = group_by_date(filtered)
        summaries = build_session_summaries(grouped)
        exercise_trends[exercise] = classify_trend(summaries)
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


# ─────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────

def build_prompt(user_id, height, bodyweight, months_training):
    workout_data = get_all_workouts(user_id)
    all_trends = get_all_exercise_trends(user_id)

    workout_json = json.dumps(workout_data, indent=2, default=str)
    trends_json = json.dumps(all_trends, indent=2)

    prompt = f"""You are an expert AI personal trainer. You analyze a user's workout history and give 
them an honest, statistics-driven progress critique after each workout.

USER CONTEXT:
- Height: {height}
- Bodyweight: {bodyweight}
- Experience: {months_training} months of training

WORKOUT DATA:
You will receive the user's full workout history as JSON, structured like this:

{workout_json}

Each workout entry contains one exercise, the date, and one set's weight/reps.

COMPUTED TRENDS:
For each exercise, you will also be given pre-computed trend data (calculated in code, 
not by you) in this format:

{trends_json}

Treat this computed trend data as ground truth for whether something is a plateau, 
improvement, or decline — do not independently judge trend direction from the raw 
numbers alone. Use the raw set/rep/weight data to explain and add color to why 
the trend looks the way it does, not to override the computed trend label.

YOUR TASK:
Using the user's context, workout history, and computed trends, analyze:
1. Overall progress since their last workout and over their full recorded history
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
  enough data yet, and nothing more about its trend.
- When referencing a percentage change for any exercise, use the exact "percent_change" 
  value provided in the computed trends data — do not calculate or estimate your own 
  percentage.
- Do not invent data that wasn't provided.

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
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from supabase import create_client, AuthApiError
from pipeline import get_critique, get_profile, calculate_months_training
from database import Workout, SessionLocal, Profile, CritiqueLog
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import func
import math
import os
import uuid

DAILY_CRITIQUE_LIMIT = 1
CRITIQUE_TIMEZONE = "America/Chicago"

load_dotenv()

app = FastAPI()

# ─────────────────────────────────────────────
# SUPABASE AUTH CLIENT (uses the anon key, not the secret key)
# ─────────────────────────────────────────────

SUPABASE_URL = "https://kkgtytvbdvmtqbmuirxe.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────────
# AUTHENTICATION DEPENDENCY
# ─────────────────────────────────────────────

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    invalid_token = HTTPException(
        status_code=401,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        user_response = supabase.auth.get_user(token)
    except AuthApiError as e:
        # Supabase rejected the token itself (expired, bad signature, user gone).
        # Anything else (outage, 5xx, network failure) is left alone so it
        # surfaces as a server error instead of signing the user out.
        if e.status in (401, 403):
            raise invalid_token
        raise

    if user_response is None or user_response.user is None:
        raise invalid_token

    return user_response.user.id


# ─────────────────────────────────────────────
# REQUEST SHAPES
# ─────────────────────────────────────────────

class WorkoutRequest(BaseModel):
    date: str
    exercise: str
    set_number: int
    weight: float
    reps: int


class WorkoutUpdateRequest(BaseModel):
    # Only weight and reps can be corrected. Any other field (date, exercise,
    # set_number, ...) is rejected rather than silently ignored.
    model_config = ConfigDict(extra="forbid")

    weight: float = Field(ge=0, allow_inf_nan=False, strict=True)
    reps: int = Field(ge=1, strict=True)


class SignupRequest(BaseModel):
    email: str
    password: str
    height: str
    bodyweight: str
    training_start_date: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ProfileUpdateRequest(BaseModel):
    height: str
    bodyweight: str
    training_start_date: str


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "Gym agent backend is running"}


def check_daily_critique_limit(user_uuid):
    """How many critiques the user has had today, and, if they've reached the
    limit, the seconds until it resets (None while they're under it).

    The limit resets at midnight in CRITIQUE_TIMEZONE (a calendar day, not a
    rolling 24 hours). created_at is stored in UTC, so the day's start is
    converted to UTC before comparing. Read-only: nothing is written."""
    tz = ZoneInfo(CRITIQUE_TIMEZONE)
    now_utc = datetime.now(timezone.utc)
    today_local = now_utc.astimezone(tz).date()
    start_of_today_utc = datetime.combine(today_local, time.min, tzinfo=tz).astimezone(timezone.utc)
    start_of_tomorrow_utc = datetime.combine(today_local + timedelta(days=1), time.min, tzinfo=tz).astimezone(timezone.utc)

    session = SessionLocal()
    try:
        used_today = (
            session.query(func.count(CritiqueLog.id))
            .filter(CritiqueLog.user_id == user_uuid, CritiqueLog.created_at >= start_of_today_utc)
            .scalar()
        )
    finally:
        session.close()

    retry_after = None
    if used_today >= DAILY_CRITIQUE_LIMIT:
        retry_after = max(1, math.ceil((start_of_tomorrow_utc - now_utc).total_seconds()))
    return used_today, retry_after


@app.get("/critique/status")
def critique_status(current_user_id: str = Depends(get_current_user)):
    # Lets the app show or hide the critique button without generating one.
    _, retry_after = check_daily_critique_limit(uuid.UUID(current_user_id))
    return {"available": retry_after is None, "retry_after_seconds": retry_after}


@app.post("/critique")
def critique_endpoint(current_user_id: str = Depends(get_current_user)):
    user_uuid = uuid.UUID(current_user_id)
    used_today, retry_after = check_daily_critique_limit(user_uuid)

    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="You've reached your daily critique limit. Your next critique is available tomorrow at 12:00 AM Central Time.",
            headers={"Retry-After": str(retry_after)},
        )

    profile = get_profile(current_user_id)
    months_training = calculate_months_training(profile["training_start_date"])

    critique_text = get_critique(
        user_id=current_user_id,
        height=profile["height"],
        bodyweight=profile["bodyweight"],
        months_training=months_training
    )

    # Only record the critique once Gemini has succeeded, so a failed attempt
    # doesn't use up the user's daily critique.
    session = SessionLocal()
    try:
        session.add(CritiqueLog(user_id=user_uuid, created_at=datetime.now(timezone.utc)))
        session.commit()
    finally:
        session.close()

    return {"critique": critique_text, "remaining": max(0, DAILY_CRITIQUE_LIMIT - used_today - 1)}


@app.post("/workouts")
def log_set(set_data: WorkoutRequest, current_user_id: str = Depends(get_current_user)):
    try:
        new_row = Workout(
            user_id=current_user_id,
            date=set_data.date,
            exercise=set_data.exercise,
            set_number=set_data.set_number,
            weight=set_data.weight,
            reps=set_data.reps
        )
        session = SessionLocal()
        session.add(new_row)
        session.commit()
        session.refresh(new_row)
        session.close()

        return {
            "id": new_row.id,
            "user_id": new_row.user_id,
            "date": new_row.date,
            "exercise": new_row.exercise,
            "set_number": new_row.set_number,
            "weight": new_row.weight,
            "reps": new_row.reps
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not save workout: {str(e)}")


def workout_to_dict(row):
    return {
        "id": row.id,
        "date": row.date,
        "exercise": row.exercise,
        "set_number": row.set_number,
        "weight": row.weight,
        "reps": row.reps
    }


@app.get("/workouts")
def get_workouts_endpoint(current_user_id: str = Depends(get_current_user)):
    # Own query rather than pipeline.get_all_workouts, so adding "id" here
    # doesn't change the data sent to Gemini.
    session = SessionLocal()
    try:
        rows = (
            session.query(Workout)
            .filter(Workout.user_id == uuid.UUID(current_user_id))
            .order_by(Workout.date, Workout.id)
            .all()
        )
    finally:
        session.close()
    return {"workouts": [workout_to_dict(row) for row in rows]}


@app.patch("/workouts/{workout_id}")
def update_workout(
    workout_id: int,
    changes: WorkoutUpdateRequest,
    current_user_id: str = Depends(get_current_user),
):
    session = SessionLocal()
    try:
        # Missing and someone else's set get the same 404, so ids can't be probed.
        row = (
            session.query(Workout)
            .filter(Workout.id == workout_id, Workout.user_id == uuid.UUID(current_user_id))
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Workout set not found")

        row.weight = changes.weight
        row.reps = changes.reps
        session.commit()
        session.refresh(row)
        return workout_to_dict(row)
    finally:
        session.close()


@app.post("/signup")
def signup(request: SignupRequest):
    try:
        auth_response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password
        })
        new_user_id = auth_response.user.id
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signup failed: {str(e)}")

    try:
        new_profile = Profile(
            id=new_user_id,
            height=request.height,
            bodyweight=request.bodyweight,
            training_start_date=request.training_start_date
        )

        session = SessionLocal()
        session.add(new_profile)
        session.commit()
        session.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Profile creation failed: {str(e)}")

    return {"message": "Signup successful", "user_id": new_user_id}


@app.post("/login")
def login(request: LoginRequest):
    auth_response = supabase.auth.sign_in_with_password({
        "email": request.email,
        "password": request.password
    })

    return {
        "access_token": auth_response.session.access_token,
        "user_id": auth_response.user.id
    }


@app.patch("/profile")
def update_profile(request: ProfileUpdateRequest, current_user_id: str = Depends(get_current_user)):
    try:
        session = SessionLocal()
        existing_profile = session.query(Profile).filter(Profile.id == current_user_id).first()

        existing_profile.height = request.height
        existing_profile.bodyweight = request.bodyweight
        existing_profile.training_start_date = request.training_start_date

        session.commit()
        session.refresh(existing_profile)
        session.close()

        return {
            "message": "Profile updated",
            "height": existing_profile.height,
            "bodyweight": existing_profile.bodyweight,
            "training_start_date": existing_profile.training_start_date
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not update profile: {str(e)}")

@app.get("/profile")
def get_profile_endpoint(current_user_id: str = Depends(get_current_user)):
    try:
        profile = get_profile(current_user_id)
        return profile
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch profile: {str(e)}")
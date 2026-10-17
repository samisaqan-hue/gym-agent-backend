from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from dotenv import load_dotenv
from supabase import (
    create_client,
    AuthApiError,
    AuthInvalidCredentialsError,
    AuthWeakPasswordError,
)
from supabase_auth import SyncGoTrueClient
from pipeline import get_critique, get_profile, calculate_months_training
from database import Workout, SessionLocal, Profile, CritiqueLog
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import func
import logging
import math
import os
import re
import uuid

logger = logging.getLogger(__name__)

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


# Fields are optional here so a missing one gets our own 400 with a plain
# message (checked in the routes) instead of FastAPI's 422 list.
class PasswordResetRequest(BaseModel):
    email: Optional[str] = None


class PasswordResetConfirm(BaseModel):
    email: Optional[str] = None
    code: Optional[str] = None
    new_password: Optional[str] = None


class RefreshRequest(BaseModel):
    refresh_token: Optional[str] = None


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


@app.delete("/workouts/{workout_id}")
def delete_workout(workout_id: int, current_user_id: str = Depends(get_current_user)):
    user_uuid = uuid.UUID(current_user_id)
    session = SessionLocal()
    try:
        # Missing and someone else's set get the same 404, so ids can't be probed.
        row = (
            session.query(Workout)
            .filter(Workout.id == workout_id, Workout.user_id == user_uuid)
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Workout set not found")

        day, exercise = row.date, row.exercise
        session.delete(row)
        session.flush()

        # Close the gap: this user's remaining sets of the same exercise on the
        # same day become 1..n, in their current order. Same transaction as
        # the delete, so either both happen or neither does.
        remaining = (
            session.query(Workout)
            .filter(Workout.user_id == user_uuid, Workout.date == day, Workout.exercise == exercise)
            .order_by(Workout.set_number, Workout.id)
            .all()
        )
        renumbered = []
        for new_number, other in enumerate(remaining, start=1):
            if other.set_number != new_number:
                other.set_number = new_number
                renumbered.append({"id": other.id, "set_number": new_number})

        session.commit()
        return {"message": "Workout set deleted", "id": workout_id, "renumbered": renumbered}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.warning("Workout delete failed: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="Could not delete the workout set. Please try again.")
    finally:
        session.close()


# Supabase error codes for an email that already has an account.
EMAIL_TAKEN_CODES = {"user_already_exists", "email_exists"}
EMAIL_INVALID_CODES = {"email_address_invalid", "validation_failed"}
EMAIL_TAKEN_DETAIL = "An account with this email already exists. Try logging in instead."

AUTH_UNAVAILABLE = "The account service is unavailable right now. Please try again in a moment."


def _is_user_error(e):
    # A 4xx answer from Supabase about the request itself, or a check the client
    # library does before sending (e.g. a missing email or password).
    if isinstance(e, AuthInvalidCredentialsError):
        return True
    return isinstance(e, AuthApiError) and 400 <= e.status < 500 and e.status != 429


def _log_auth_failure(action, e):
    # Only the error class and Supabase's error code: never the message, the
    # email, the password or a token.
    logger.warning("%s failed: %s (code=%s)", action, type(e).__name__, getattr(e, "code", None))


@app.post("/signup")
def signup(request: SignupRequest):
    try:
        auth_response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password
        })
    except Exception as e:
        _log_auth_failure("Signup", e)
        code = getattr(e, "code", None)
        if code in EMAIL_TAKEN_CODES:
            raise HTTPException(status_code=409, detail=EMAIL_TAKEN_DETAIL)
        if isinstance(e, AuthWeakPasswordError) or code == "weak_password":
            raise HTTPException(status_code=400, detail="That password is too weak. Please choose a stronger one.")
        if isinstance(e, AuthInvalidCredentialsError):
            raise HTTPException(status_code=400, detail="Please enter an email and password.")
        if code in EMAIL_INVALID_CODES:
            raise HTTPException(status_code=400, detail="Please enter a valid email address.")
        if _is_user_error(e):
            raise HTTPException(status_code=400, detail="Sign-up failed. Please check your details and try again.")
        raise HTTPException(status_code=503, detail=AUTH_UNAVAILABLE)

    if auth_response is None or auth_response.user is None:
        # No account was created, so don't create a profile for one.
        raise HTTPException(status_code=400, detail="Sign-up failed. Please check your details and try again.")

    # For an email that already has an account, Supabase can answer with a
    # placeholder user instead of an error (so the response doesn't reveal
    # which emails exist). That user has an empty identities list; a real new
    # user always has one identity. No account was created, so no profile.
    if auth_response.user.identities is not None and len(auth_response.user.identities) == 0:
        logger.warning("Signup failed: existing email (placeholder user returned)")
        raise HTTPException(status_code=409, detail=EMAIL_TAKEN_DETAIL)
    new_user_id = auth_response.user.id

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
    try:
        # Per-request client: signing in on the shared client would keep this
        # user's session there and start a background refresh that rotates the
        # refresh token the app now holds, making it look reused.
        auth_response = _new_auth_client().sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
    except Exception as e:
        _log_auth_failure("Login", e)
        # Wrong password and unknown email get the same answer, so this can't
        # be used to find out which emails have accounts.
        if _is_user_error(e):
            raise HTTPException(status_code=401, detail="Incorrect email or password.")
        raise HTTPException(status_code=503, detail=AUTH_UNAVAILABLE)

    if auth_response is None or auth_response.session is None or auth_response.user is None:
        logger.warning("Login failed: Supabase returned no session")
        raise HTTPException(status_code=503, detail=AUTH_UNAVAILABLE)

    return {
        "access_token": auth_response.session.access_token,
        "user_id": auth_response.user.id,
        "refresh_token": auth_response.session.refresh_token,
        "expires_in": auth_response.session.expires_in,
    }


# ─────────────────────────────────────────────
# PASSWORD RESET (6-digit code emailed by Supabase)
# ─────────────────────────────────────────────

RESET_REQUESTED = "If an account exists for that email, we sent a code."
RESET_DONE = "Password updated. Please log in."
INVALID_EMAIL = "Please enter a valid email address."
MISSING_RESET_FIELDS = "Please enter your email, the code and a new password."
INVALID_CODE = "That code is invalid or has expired."
WEAK_PASSWORD = "That password is too weak. Please choose a stronger one."
SAME_PASSWORD = "Your new password must be different from your current one."
TOO_MANY_REQUESTS = "Too many requests. Please wait a bit before trying again."
RATE_LIMIT_CODES = {"over_email_send_rate_limit", "over_request_rate_limit"}
PASSWORD_RESET_PATHS = {"/password-reset/request", "/password-reset/confirm"}

# Deliberately loose: Supabase does the real check. This only catches empty
# or obviously malformed input before asking Supabase to send an email.
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Password-reset errors are always a plain string. Other routes keep
    # FastAPI's default 422 response.
    if request.url.path == "/password-reset/request":
        return JSONResponse(status_code=400, content={"detail": INVALID_EMAIL})
    if request.url.path == "/password-reset/confirm":
        return JSONResponse(status_code=400, content={"detail": MISSING_RESET_FIELDS})
    if request.url.path == "/refresh":
        return JSONResponse(status_code=400, content={"detail": LOG_IN_AGAIN})
    return await request_validation_exception_handler(request, exc)


def _is_rate_limited(e):
    return getattr(e, "code", None) in RATE_LIMIT_CODES or getattr(e, "status", None) == 429


def _new_auth_client():
    """A throwaway Supabase Auth client for one request. Signing in, verifying
    a recovery code or refreshing a session saves that user's session on
    whichever client does it, so this must never be the shared `supabase`
    client. No storage, no background token refresh. It reuses the shared
    client's HTTP connection pool (which holds no user state) instead of
    opening a new one each time."""
    return SyncGoTrueClient(
        url=f"{SUPABASE_URL}/auth/v1",
        headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
        auto_refresh_token=False,
        persist_session=False,
        http_client=supabase.auth._http_client,
    )


@app.post("/password-reset/request")
def request_password_reset(body: PasswordResetRequest):
    email = (body.email or "").strip()
    if not EMAIL_PATTERN.match(email):
        raise HTTPException(status_code=400, detail=INVALID_EMAIL)

    try:
        # Sends the recovery email; nothing is stored on the client.
        supabase.auth.reset_password_for_email(email)
    except Exception as e:
        _log_auth_failure("Password reset request", e)
        if _is_rate_limited(e):
            raise HTTPException(status_code=429, detail=TOO_MANY_REQUESTS)
        if getattr(e, "code", None) in EMAIL_INVALID_CODES:
            raise HTTPException(status_code=400, detail=INVALID_EMAIL)
        if not _is_user_error(e):
            raise HTTPException(status_code=503, detail=AUTH_UNAVAILABLE)
        # Any other rejection gets the same answer as success, so this
        # endpoint never reveals whether an email has an account.

    return {"message": RESET_REQUESTED}


@app.post("/password-reset/confirm")
def confirm_password_reset(body: PasswordResetConfirm):
    email = (body.email or "").strip()
    code = (body.code or "").strip()
    new_password = body.new_password or ""
    if not email or not code or not new_password:
        raise HTTPException(status_code=400, detail=MISSING_RESET_FIELDS)

    client = _new_auth_client()
    try:
        try:
            verified = client.verify_otp({"email": email, "token": code, "type": "recovery"})
        except Exception as e:
            _log_auth_failure("Password reset code check", e)
            if _is_rate_limited(e):
                raise HTTPException(status_code=429, detail=TOO_MANY_REQUESTS)
            if _is_user_error(e):
                raise HTTPException(status_code=400, detail=INVALID_CODE)
            raise HTTPException(status_code=503, detail=AUTH_UNAVAILABLE)

        if verified is None or verified.session is None:
            logger.warning("Password reset code check failed: Supabase returned no session")
            raise HTTPException(status_code=503, detail=AUTH_UNAVAILABLE)

        try:
            client.update_user({"password": new_password})
        except Exception as e:
            _log_auth_failure("Password update", e)
            code_name = getattr(e, "code", None)
            if isinstance(e, AuthWeakPasswordError) or code_name == "weak_password":
                raise HTTPException(status_code=400, detail=WEAK_PASSWORD)
            if code_name == "same_password":
                raise HTTPException(status_code=400, detail=SAME_PASSWORD)
            if _is_rate_limited(e):
                raise HTTPException(status_code=429, detail=TOO_MANY_REQUESTS)
            raise HTTPException(status_code=503, detail=AUTH_UNAVAILABLE)
    finally:
        # Don't leave the user logged in: revoke the recovery session and drop
        # it from this throwaway client.
        try:
            if client.get_session() is not None:
                client.sign_out({"scope": "local"})
        except Exception as e:
            _log_auth_failure("Password reset sign-out", e)

    return {"message": RESET_DONE}


# ─────────────────────────────────────────────
# SESSION REFRESH
# ─────────────────────────────────────────────

LOG_IN_AGAIN = "Please log in again."
SESSION_EXPIRED = "Your session has expired. Please log in again."


@app.post("/refresh")
def refresh_session(body: RefreshRequest):
    # No Authorization header: the access token has usually expired by now.
    refresh_token = (body.refresh_token or "").strip()
    if not refresh_token:
        raise HTTPException(status_code=400, detail=LOG_IN_AGAIN)

    try:
        # Per-request client, never the shared one. It is simply dropped
        # afterwards: signing it out would revoke the session just issued.
        refreshed = _new_auth_client().refresh_session(refresh_token)
    except Exception as e:
        _log_auth_failure("Session refresh", e)
        if _is_rate_limited(e):
            raise HTTPException(status_code=429, detail=TOO_MANY_REQUESTS)
        if _is_user_error(e):
            # Invalid, expired, revoked or already-used refresh token.
            raise HTTPException(status_code=401, detail=SESSION_EXPIRED)
        raise HTTPException(status_code=503, detail=AUTH_UNAVAILABLE)

    if refreshed is None or refreshed.session is None:
        logger.warning("Session refresh failed: Supabase returned no session")
        raise HTTPException(status_code=503, detail=AUTH_UNAVAILABLE)

    return {
        "access_token": refreshed.session.access_token,
        "refresh_token": refreshed.session.refresh_token,
        "expires_in": refreshed.session.expires_in,
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
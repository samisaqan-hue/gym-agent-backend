from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
from pipeline import get_critique, get_profile, calculate_months_training, get_all_workouts
from database import Workout, SessionLocal, Profile
import os

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
    user_response = supabase.auth.get_user(token)
    return user_response.user.id


# ─────────────────────────────────────────────
# REQUEST SHAPES
# ─────────────────────────────────────────────

class WorkoutRequest(BaseModel):
    user_id: str
    date: str
    exercise: str
    set_number: int
    weight: float
    reps: int


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


@app.post("/critique")
def critique_endpoint(current_user_id: str = Depends(get_current_user)):
    profile = get_profile(current_user_id)
    months_training = calculate_months_training(profile["training_start_date"])

    critique_text = get_critique(
        user_id=current_user_id,
        height=profile["height"],
        bodyweight=profile["bodyweight"],
        months_training=months_training
    )
    return {"critique": critique_text}


@app.post("/workouts")
def log_set(set_data: WorkoutRequest):
    try:
        new_row = Workout(
            user_id=set_data.user_id,
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


@app.get("/workouts")
def get_workouts_endpoint(current_user_id: str = Depends(get_current_user)):
    workouts = get_all_workouts(current_user_id)
    return {"workouts": workouts}


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
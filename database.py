from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

db_password = os.getenv("DATABASE_PASSWORD")
DATABASE_URL = f"postgresql://postgres:{db_password}@db.kkgtytvbdvmtqbmuirxe.supabase.co:5432/postgres"

engine = create_engine(DATABASE_URL)

# A "session" is what you'll actually use to run queries and insert data.
SessionLocal = sessionmaker(bind=engine)


# ─────────────────────────────────────────────
# TABLE MODELS
# ─────────────────────────────────────────────

Base = declarative_base()

class Workout(Base):
    __tablename__ = "workouts"

    id = Column(Integer, primary_key=True)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    date = Column(Date, nullable=False)
    exercise = Column(String, nullable=False)
    set_number = Column(Integer, nullable=False)
    weight = Column(Float, nullable=False)
    reps = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(UUID(as_uuid=True), primary_key=True)
    height = Column(String, nullable=False)
    bodyweight = Column(String, nullable=False)
    training_start_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
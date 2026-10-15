from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, ForeignKey, Table, func
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from dotenv import load_dotenv
from datetime import datetime, timezone
import os

load_dotenv()

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

db_password = os.getenv("DATABASE_PASSWORD")
DATABASE_URL = f"postgresql://postgres.kkgtytvbdvmtqbmuirxe:{db_password}@aws-0-us-west-2.pooler.supabase.com:5432/postgres"

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


# Supabase's own auth.users table. Supabase manages it; it is declared here only
# so SQLAlchemy can resolve the critique_log.user_id foreign key.
auth_users = Table(
    "users",
    Base.metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    schema="auth",
)


class CritiqueLog(Base):
    # One row per successful critique, used to enforce the daily limit.
    __tablename__ = "critique_log"

    id = Column(Integer, primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id"), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

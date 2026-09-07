import requests

BASE_URL = "http://127.0.0.1:8000"
REAL_USER_ID = "3d12cc3a-f847-4c1e-b395-c11915bc71c2"  # your real authenticated user_id

sets_to_log = [
    # Deadlift — only 2 sessions, deliberately under the 4-session minimum,
    # to test the "not enough data" rule
    {"date": "2026-09-05", "exercise": "deadlift", "set_number": 1, "weight": 275, "reps": 8},
    {"date": "2026-09-05", "exercise": "deadlift", "set_number": 2, "weight": 275, "reps": 7},
    {"date": "2026-09-05", "exercise": "deadlift", "set_number": 3, "weight": 275, "reps": 6},

    {"date": "2026-09-10", "exercise": "deadlift", "set_number": 1, "weight": 280, "reps": 8},
    {"date": "2026-09-10", "exercise": "deadlift", "set_number": 2, "weight": 280, "reps": 7},
    {"date": "2026-09-10", "exercise": "deadlift", "set_number": 3, "weight": 280, "reps": 6},
]

if __name__ == "__main__":
    for s in sets_to_log:
        payload = {
            "user_id": REAL_USER_ID,
            "date": s["date"],
            "exercise": s["exercise"],
            "set_number": s["set_number"],
            "weight": s["weight"],
            "reps": s["reps"]
        }
        response = requests.post(f"{BASE_URL}/workouts", json=payload)
        print(response.status_code, response.json())
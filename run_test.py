from pipeline import get_critique

if __name__ == "__main__":
    user_id = "a1b2c3d4-5678-90ab-cdef-1234567890ab"  # your fake test user's UUID

    critique = get_critique(
        user_id=user_id,
        height="5'10\"",
        bodyweight="180 lbs",
        months_training=8
    )

    print(critique)

import os

# main.py builds the Supabase and Gemini clients at import time, so they need
# some key to exist. Tests replace the Supabase client with a fake; these
# dummy values are never sent anywhere.
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

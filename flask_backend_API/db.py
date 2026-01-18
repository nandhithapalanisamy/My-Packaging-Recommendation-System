from sqlalchemy import create_engine
import os

DATABASE_URL = os.environ.get("DATABASE_URL")

try:
    if DATABASE_URL is None:
        raise Exception("DATABASE_URL environment variable not set")

    engine = create_engine(DATABASE_URL)
    print("Database connected successfully")
except Exception as e:
    print("Database connection failed:", e)

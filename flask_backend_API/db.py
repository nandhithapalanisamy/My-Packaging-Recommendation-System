from sqlalchemy import create_engine

try:
    engine = create_engine(
    "postgresql+psycopg2://postgres:PostgreSQL#11@localhost:5432/infosys_database"
)
    print("Database connected successfully")
except Exception as e:
    print("Database connection failed:", e)
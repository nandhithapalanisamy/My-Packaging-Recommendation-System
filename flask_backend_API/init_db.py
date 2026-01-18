import pandas as pd
from sqlalchemy import create_engine
import os

# -------------------------------
# Base directory (important)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# -------------------------------
# CSV path (your data_set.csv)
# -------------------------------
CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "Dataset_Preparation",
    "data_set.csv"
)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

# -------------------------------
#  Database connection
# -------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

print(" Connected to Render PostgreSQL")

# -------------------------------
# 4. Load CSV
# -------------------------------
df = pd.read_csv(CSV_PATH)

print(f" Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")

# -------------------------------
# 5. Create table: target_material_data
# -------------------------------
df.to_sql(
    name="target_material_data",
    con=engine,
    if_exists="replace",   # drops old table if exists
    index=False
)

print(" target_material_data table created and populated successfully")

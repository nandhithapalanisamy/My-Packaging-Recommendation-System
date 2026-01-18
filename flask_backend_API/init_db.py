import os
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    print("DATABASE_URL not set, skipping DB init")
    exit(0)

engine = create_engine(DATABASE_URL)

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Dataset_Preparation",
    "data_set.csv"
)

def init_db():
    with engine.connect() as conn:
        # check if table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'target_material_data'
            );
        """)).scalar()

        if result:
            print("Table already exists. Skipping initialization.")
            return

    print("Creating table and inserting CSV data...")
    df = pd.read_csv(CSV_PATH)

    df.to_sql(
        "target_material_data",
        engine,
        if_exists="replace",
        index=False
    )

    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()

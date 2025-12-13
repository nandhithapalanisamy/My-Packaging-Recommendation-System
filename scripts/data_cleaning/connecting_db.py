import psycopg2
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from fill_missing_value import missing_value  
from fill_missing_value import normalize_numeric 
from fill_missing_value import encode_categorical
from fill_missing_value import feature_engineering  

# connecting to postgresql database
connection = psycopg2.connect(
    host="localhost",
    database="infosys_database",
    user="postgres",
    password="Pass_word",
    port="5432"
)

cursor = connection.cursor()
cursor.execute("SELECT * FROM material_data;")
rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]  
print(columns)

print("Connected to DB")

# Convert DB rows to DataFrame with correct column names
df = pd.DataFrame(rows, columns=columns)
print(df.head())
# numeric column 
numeric_cols = df.select_dtypes(include=np.number).columns

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
df_clean = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]


# Calling missing value processing
df_updated = missing_value(df_clean)
print("Missing values filled.") 


# calling normalization function
df_updated= normalize_numeric(df_updated)
print("Numeric values normalized.")
# calling encoding function

df_updated = encode_categorical(df_updated)
print("Categorical values encoded.")

# calling feature engineering function
df_updated = feature_engineering(df_updated)
print("Feature engineering completed.")

#  statistical analysis
print(df_updated.describe())
print(df_updated.isnull().sum())
print(df_updated.info())

df_updated.columns = df_updated.columns.str.replace("-", "_")
df_updated.columns = df_updated.columns.str.replace(" ", "_")

engine = create_engine(
    "postgresql+psycopg2://postgres:PostgreSQL#11@localhost:5432/infosys_database"
)

# Transfer DataFrame to PostgreSQL
df_updated.to_sql(
    name="target_material_data",   
    con=engine,
    if_exists="replace",   
    index=False
)

print("DataFrame successfully inserted into PostgreSQL!")
connection.commit()
cursor.close()
connection.close()
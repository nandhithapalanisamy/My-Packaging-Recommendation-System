import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read CSV
df = pd.read_csv("C:/Users/nandh/Downloads/before_cleaning_material_data.csv")

# Correct structural errors first (strip spaces, lowercase, replace spaces, remove special chars)
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("[^a-z0-9_]", "", regex=True)
)

# Remove rows with missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Ensure productCategory and productName are strings
df = df[df["material_type"].apply(lambda x: isinstance(x, str))]
df = df[df["strength"].apply(lambda x: isinstance(x, str))]
df = df[df["compostable"].apply(lambda x: isinstance(x, str))]
df = df[df["water_resistance"].apply(lambda x: isinstance(x, str))]
# Columns that should be numeric
numeric_cols = ["id", "weight_capacity", "biodegradability_score", "co2_emission"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with invalid numeric data
df.dropna(subset=[c for c in numeric_cols if c in df.columns], inplace=True)

# Outlier detection and removal for all numeric columns
for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

# Convert all text to lowercase
df["material_type"] = df["material_type"].str.lower()
df["strength"] = df["strength"].str.lower()
df["compostable"] = df["compostable"].str.lower()
df["water_resistance"] = df["water_resistance"].str.lower()

df[numeric_cols] = df[numeric_cols].round(2)
# Reset product ID
df["id"] = range(1, len(df) + 1)

# Save cleaned data
df.to_csv("C:/Users/nandh/Downloads/cleaned_material_data.csv", index=False)
print("DATA CLEANED")
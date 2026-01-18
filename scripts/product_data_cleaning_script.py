import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read CSV
df = pd.read_csv(r"C:\InfosysInternshipRepos\Packaging-Recommendation-System\data_source\product_data_before_cleaning.csv")

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
df = df[df["productcategory"].apply(lambda x: isinstance(x, str))]
df = df[df["productname"].apply(lambda x: isinstance(x, str))]

# Columns that should be numeric
numeric_cols = ["productid", "productweight", "width", "height"]
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
df["productcategory"] = df["productcategory"].str.lower()
df["productname"] = df["productname"].str.lower()

df[numeric_cols] = df[numeric_cols].round(2)
# Reset product ID
df["productid"] = range(1, len(df) + 1)

# Save cleaned data
df.to_csv(r"C:\InfosysInternshipRepos\Packaging-Recommendation-System\data_source\cleaned_product_data.csv", index=False)
print("DATA CLEANED")


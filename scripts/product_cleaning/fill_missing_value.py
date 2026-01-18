import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
# Read CSV
df = pd.read_csv(r"C:\InfosysInternshipRepos\Packaging-Recommendation-System\data_source\cleaned_product_data.csv")

null_summary = df.isnull().sum()
total_null = null_summary.sum()

if total_null == 0:  
    print("No null values found in the DataFrame.")
else:   
    # Predict missing productweight
    w_X_train = df[df['productweight'].notnull()][['width', 'height']]
    w_y_train = df[df['productweight'].notnull()]['productweight']

    w_X_missing = df[df['productweight'].isnull()][['width', 'height']]

    weight_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", LinearRegression())
    ])

    weight_pipeline.fit(w_X_train, w_y_train)

    predicted_weights = weight_pipeline.predict(w_X_missing)

    df.loc[df['productweight'].isnull(), 'productweight'] = predicted_weights

    # Predict missing height

    h_X_train = df[df['height'].notnull()][['width', 'productweight']]
    h_y_train = df[df['height'].notnull()]['height']

    h_X_missing = df[df['height'].isnull()][['width', 'productweight']]

    height_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", LinearRegression())
    ])

    height_pipeline.fit(h_X_train, h_y_train)

    predicted_heights = height_pipeline.predict(h_X_missing)

    df.loc[df['height'].isnull(), 'height'] = predicted_heights

    # Predict missing width

    wd_X_train = df[df['width'].notnull()][['height', 'productweight']]
    wd_y_train = df[df['width'].notnull()]['width']

    wd_X_missing = df[df['width'].isnull()][['height', 'productweight']]

    width_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", LinearRegression())
    ])

    width_pipeline.fit(wd_X_train, wd_y_train)

    predicted_widths = width_pipeline.predict(wd_X_missing)

    df.loc[df['width'].isnull(), 'width'] = predicted_widths
    print("Missing values filled using prediction models.")

# Database connection setup
engine = create_engine(
    "postgresql+psycopg2://postgres:PostgreSQL#11@localhost:5432/infosys_database")

# Transfer DataFrame to PostgreSQL
df.to_sql(
    name="target_product_data",   
    con=engine,
    if_exists="replace",   
    index=False
)

print("DataFrame successfully inserted into PostgreSQL!")


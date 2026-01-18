import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Missing value function
def missing_value(df):
    numeric_columns = ["id", "weight_capacity", "biodegradability_score", "co2_emission", "recyclability"]
    categorical_columns = ["material_type", "strength", "compostable", "water_resistance"]

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    df_filled = preprocessor.fit_transform(df)
    df_filled = pd.DataFrame(df_filled, columns=numeric_columns + categorical_columns)
    return df_filled

# normaling the numeric values using minmaxscaler
def normalize_numeric(df):
    numeric_columns = ["id", "weight_capacity", "biodegradability_score", "co2_emission", "recyclability"]
    scaler = MinMaxScaler()

    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Round to 2 decimal place
    df[numeric_columns] = df[numeric_columns].round(2)
    # changing the id
    df["id"] = range(1, len(df) + 1)
    return df

# encoding the categorical values
def encode_categorical(df):
    # Standardize text to lowercase and strip whitespace
    df["strength"] = df["strength"].str.lower().str.strip()
    df["compostable"] = df["compostable"].astype(str).str.lower().str.strip()
    df["water_resistance"] = df["water_resistance"].astype(str).str.lower().str.strip()
    df["material_type"] = df["material_type"].str.lower().str.strip()

    # Strength mapping (ordered & meaningful)
    strength_map = {
        "low": 0,
        "low-medium": 1,
        "medium-low": 1,
        "medium": 2,
        "medium-high": 3,
        "high": 4
    }
    df["strength"] = df["strength"].map(strength_map)
    # Water resistance mapping
    water_map = {
        "low": 0,
        "medium": 1,
        "high": 2
    }
    df["water_resistance"] = df["water_resistance"].map(water_map)

    # Compostable mapping
    df["compostable"] = df["compostable"].map({"yes": 1, "no": 0})


    #  One-hot encoding for material_type
    df = pd.get_dummies(df, columns=["material_type"], prefix="material_type")

    # for material_type dummies convert to int
    dummy_cols = [col for col in df.columns if col.startswith("material_type_")]
    df[dummy_cols] = df[dummy_cols].astype(int)

    return df

#feature engineering function
def feature_engineering(df):
    #c02 impact index
    # Lower CO2 emission + higher biodegradability = better
    df["co2_impact_index"] = (1 - df["co2_emission"]) * 0.6 + df["biodegradability_score"] * 0.4
    df["co2_impact_index"] = df["co2_impact_index"].round(3)

    #cost efficient index
    # Higher recyclability + lower weight = cheaper & efficient
    df["cost_efficiency_index"] = df["recyclability"] * 0.7 + (1 - df["weight_capacity"]) * 0.3
    df["cost_efficiency_index"] = df["cost_efficiency_index"].round(3)

    #material sustainability score
    #strength + water resistance + compostability + cost efficiency
    df["material_suitability_score"] = (
        df["strength"] * 0.3 +
        df["water_resistance"] * 0.2 +
        df["compostable"] * 0.1 +
        df["cost_efficiency_index"] * 0.4
    )
    df["material_suitability_score"] = df["material_suitability_score"].round(3)
    return df


    
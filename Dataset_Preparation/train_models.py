import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import joblib


# Enable MLflow autolog
mlflow.sklearn.autolog()

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(
    "C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\data_set.csv"
)

features = [
    'weight_capacity', 'biodegradability_score', 'co2_emission',
    'recyclability', 'strength', 'compostable', 'water_resistance'
]

X = df[features]
y_cost = df['cost_efficiency_index']
y_co2 = df['co2_impact_index']

# ----------------------------
# TRAIN-TEST SPLIT
# ----------------------------
X_train, X_test, y_cost_train, y_cost_test, y_co2_train, y_co2_test = train_test_split(
    X, y_cost, y_co2, test_size=0.2, random_state=42
)

# ----------------------------
# SAVE TRAIN & TEST DATASETS
# ----------------------------
train_df = X_train.copy()
train_df["cost_efficiency_index"] = y_cost_train
train_df["co2_impact_index"] = y_co2_train
train_df.to_csv("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\data_outputs\\train_data.csv", index=False)

test_df = X_test.copy()
test_df["cost_efficiency_index"] = y_cost_test
test_df["co2_impact_index"] = y_co2_test
test_df.to_csv("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\data_outputs\\test_data.csv", index=False)

# ----------------------------
# 5. LINEAR REGRESSION MODELS (BASELINE)
# ----------------------------
with mlflow.start_run(run_name="Linear_Cost_Prediction"):
    cost_lr = Pipeline([("model", LinearRegression())])
    cost_lr.fit(X_train, y_cost_train)

    mlflow.sklearn.log_model(cost_lr, "linear_cost_model")

    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/linear_cost_model",
        "Linear_Cost_Prediction"
    )

    joblib.dump(cost_lr, "C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\saved_models\\linear_cost_model.pkl")

with mlflow.start_run(run_name="Linear_CO2_Prediction"):
    co2_lr = Pipeline([("model", LinearRegression())])
    co2_lr.fit(X_train, y_co2_train)

    mlflow.sklearn.log_model(co2_lr, "linear_co2_model")

    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/linear_co2_model",
        "Linear_CO2_Prediction"
    )

    joblib.dump(co2_lr, "C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\saved_models\\linear_co2_model.pkl")

# ----------------------------
# RANDOM FOREST (COST)
# ----------------------------
with mlflow.start_run(run_name="RandomForest_Cost_Model"):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_cost_train)

    rf_pred = rf_model.predict(X_test)

    mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(y_cost_test, rf_pred)))
    mlflow.log_metric("MAE", mean_absolute_error(y_cost_test, rf_pred))
    mlflow.log_metric("R2", r2_score(y_cost_test, rf_pred))

    mlflow.sklearn.log_model(rf_model, "rf_cost_model")

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/rf_cost_model"
    mlflow.register_model(model_uri, "RF_Cost_Prediction")

    joblib.dump(rf_model, "C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\saved_models\\random_forest_cost.pkl")

# ----------------------------
# XGBOOST (CO2)
# ----------------------------
with mlflow.start_run(run_name="XGBoost_CO2_Model"):
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train, y_co2_train)

    xgb_pred = xgb_model.predict(X_test)

    mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(y_co2_test, xgb_pred)))
    mlflow.log_metric("MAE", mean_absolute_error(y_co2_test, xgb_pred))
    mlflow.log_metric("R2", r2_score(y_co2_test, xgb_pred))

    mlflow.sklearn.log_model(xgb_model, "xgb_co2_model")

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/xgb_co2_model"
    mlflow.register_model(model_uri, "XGB_CO2_Prediction")

    joblib.dump(xgb_model, "C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\saved_models\\xgboost_co2.pkl")

# ----------------------------
# RANKING FILE (BASED ON PREDICTIONS)
# ----------------------------
ranking_df = X_test.copy()
ranking_df["Predicted_Cost"] = rf_pred
ranking_df["Predicted_CO2"] = xgb_pred

ranking_df["Final_Score"] = (
    ranking_df["Predicted_Cost"].rank(ascending=False) +
    ranking_df["Predicted_CO2"].rank(ascending=True)
)

ranking_df = ranking_df.sort_values("Final_Score")
ranking_df.to_csv("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\data_outputs\\ranking.csv", index=False)

print("All outputs generated successfully!")

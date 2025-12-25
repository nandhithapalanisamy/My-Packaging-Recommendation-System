import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# module 3
# Load your dataset
df = pd.read_csv("C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\data_set.csv")

#Select ML features for prediction
features = ['weight_capacity', 'biodegradability_score', 'co2_emission', 'recyclability', 
            'strength', 'compostable', 'water_resistance']
X = df[features]

#  Generate target values
y_cost = df['cost_efficiency_index']
y_co2 = df['co2_impact_index']

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_cost_train, y_cost_test, y_co2_train, y_co2_test = train_test_split(
    X, y_cost, y_co2, test_size=0.2, random_state=42
)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)


#Cost prediction pipeline
cost_pipeline = Pipeline([
    ('model', LinearRegression())
])

#Train cost prediction model
cost_pipeline.fit(X_train, y_cost_train)

#Make predictions
cost_predictions = cost_pipeline.predict(X_test)
print("Cost Efficiency Predictions:", cost_predictions)

#CO2 impact prediction pipeline
co2_pipeline = Pipeline([
    ('model', LinearRegression())
])

#Train CO2 impact prediction model
co2_pipeline.fit(X_train, y_co2_train)

#Make predictions
co2_predictions = co2_pipeline.predict(X_test)
print("CO2 Impact Predictions:", co2_predictions)

# module 4

#Train Random Forest for Cost Prediction
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_cost_train)

rf_cost_pred = rf_model.predict(X_test)
print("Random Forest Cost Efficiency Predictions:", rf_cost_pred)

#Train XGBoost for CO₂ Prediction
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_co2_train)

xgb_co2_pred = xgb_model.predict(X_test)
print("XGBoost CO2 Impact Predictions:", xgb_co2_pred)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print()
    return rmse, mae, r2

rf_metrics = evaluate_model(y_cost_test, rf_cost_pred, "Random Forest (Cost)")
xgb_metrics = evaluate_model(y_co2_test, xgb_co2_pred, "XGBoost (CO2)")

# Material Ranking System
# Assume you have a list of material identifiers in X_test.index or a column in df_test
# Higher cost is worse, higher CO2 is worse. Lower scores = better rank.

ranking_df = pd.DataFrame({
    "Material": X_test.index,  # replace with actual material names if available
    "Predicted_Cost": rf_cost_pred,
    "Predicted_CO2": xgb_co2_pred
})
#Min-Max normalization done manually
# Normalize both predictions to 0-1 range for combined ranking
ranking_df["Cost_Score"] = (ranking_df["Predicted_Cost"] - ranking_df["Predicted_Cost"].min()) / \
                           (ranking_df["Predicted_Cost"].max() - ranking_df["Predicted_Cost"].min())

ranking_df["CO2_Score"] = (ranking_df["Predicted_CO2"] - ranking_df["Predicted_CO2"].min()) / \
                          (ranking_df["Predicted_CO2"].max() - ranking_df["Predicted_CO2"].min())

# Simple combined score: equal weight
# Combine cost and CO2 scores
# Equal weight is given to cost and CO2 for balanced sustainability ranking
ranking_df["Combined_Score"] = ranking_df["Cost_Score"] + ranking_df["CO2_Score"]

# Rank materials: lower combined score = better
ranking_df["Rank"] = ranking_df["Combined_Score"].rank(method="min")
ranking_df = ranking_df.sort_values("Rank")

print("--- Material Ranking Based on ML Predictions ---")
print(ranking_df[["Material", "Predicted_Cost", "Predicted_CO2", "Combined_Score", "Rank"]])

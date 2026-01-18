
import joblib
from sqlalchemy import text
from db import engine
from feature_engineering import generate_features
from flask import Flask, request, jsonify, render_template
import os
import pandas as pd

# -------------------------------
# Decode One-Hot Encoded Material
# -------------------------------
def decode_material_type(row):
    for key, value in row.items():
        if key.startswith("material_type_") and value == 1:
            return key.replace("material_type_", "").replace("_", " ").title()
    return "Unknown Material"
# -------------------------------
# Constants for real-world units
# -------------------------------
COST_SCALE_INR = 100          # model → ₹
CO2_SCALE_PERCENT = 100      # model → %

# -------------------------------
# Load ML Models
# -------------------------------
cost_model = joblib.load(
    "C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\saved_models\\random_forest_cost.pkl"
)
co2_model = joblib.load(
    "C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\saved_models\\xgboost_co2.pkl"
)


def recommend_material_data(weight, fragility):

    weight = float(weight)
    fragility = int(fragility)  # 1–5

    query = text("SELECT * FROM target_material_data")
    with engine.connect() as conn:
        materials = [dict(row._mapping) for row in conn.execute(query)]

    
    r = []

    for m in materials:

        # -------------------------------
        # THEORY → FEATURE ADJUSTMENT
        # -------------------------------

        material_capacity = float(m.get("weight_capacity", 0))
        material_strength = float(m.get("strength", 0))

        # Weight vs Capacity logic
        #If product is heavier than material capacity, cost & CO₂ increase.
        # If product weight is high relative to capacity → penalty
        if material_capacity > 0:
            capacity_ratio = weight / material_capacity
        else:
            capacity_ratio = 2.0  # heavy penalty if no capacity

        capacity_penalty = max(0, capacity_ratio - 1)

        #  Fragility vs Strength logic
        #Fragile products need stronger materials.
        # Higher fragility demands higher strength
        strength_requirement = fragility / 5  # normalize
        strength_penalty = max(0, strength_requirement - (material_strength / 5))

        #  Inject penalties into features
        m["effective_weight"] = weight * (1 + capacity_penalty)
        m["effective_fragility"] = fragility * (1 + strength_penalty)

        # -------------------------------
        # Feature Engineering
        # -------------------------------
        features = generate_features(m)

        # -------------------------------
        # Model Predictions
        # -------------------------------
        cost_raw = float(cost_model.predict([features])[0])
        co2_raw = float(co2_model.predict([features])[0])

        # Penalize bad material matches
        adjusted_cost = cost_raw * (1 + capacity_penalty + strength_penalty)
        adjusted_co2 = co2_raw * (1 + capacity_penalty)

        # Scale to real-world values
        cost_inr = adjusted_cost * COST_SCALE_INR
        co2_percent = adjusted_co2 * CO2_SCALE_PERCENT
        # Clamp CO₂ impact to 0–100%
        co2_percent = max(0, min(co2_percent, 100))
        # Final ranking score (lower is better)
        score = (0.6 * adjusted_co2) + (0.4 * adjusted_cost)
        # -------------------------------
        #collecting data for the dashboard
        # -------------------------------
        r.append({
    "material": decode_material_type(m),
    "predicted_cost_raw": round(cost_raw * COST_SCALE_INR, 2),    # before penalty
    "predicted_co2_raw": round(co2_raw * CO2_SCALE_PERCENT, 2),   # before penalty
    "predicted_cost": round(cost_inr, 2),                          # after penalty
    "co2_impact": round(co2_percent, 2),                           # after penalty
    "score": round(score, 4)
    })
        return sorted(r, key=lambda x: x["score"])[:10]  # Top 10 recommendations
# Imports
import joblib
from sqlalchemy import text
from db import engine
from feature_engineering import generate_features
from flask import Flask, request, jsonify, render_template
import os

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
# Flask App Setup
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
TEMPLATE_DIR = os.path.join(FRONTEND_DIR, "templates")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Load ML Models
# -------------------------------
cost_model = joblib.load(
    "C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\saved_models\\random_forest_cost.pkl"
)
co2_model = joblib.load(
    "C:\\InfosysInternshipRepos\\Packaging-Recommendation-System\\Dataset_Preparation\\saved_models\\xgboost_co2.pkl"
)

# -------------------------------
# Recommendation API
# -------------------------------
@app.route("/recommend", methods=["POST"])
def recommend_material():
    data = request.json
    weight = float(data.get("weight", 1.0))
    fragility = int(data.get("fragility", 1))  # 1–5

    query = text("SELECT * FROM target_material_data")
    with engine.connect() as conn:
        materials = [dict(row._mapping) for row in conn.execute(query)]

    recommendations = []

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

        recommendations.append({
            "material": decode_material_type(m),
            "predicted_cost": round(cost_inr, 2),
            "co2_impact": round(co2_percent, 2),
            "score": round(score, 4)
        })

    recommendations.sort(key=lambda x: x["score"])
    return jsonify(recommendations[:10])

# -------------------------------
# Environmental Score API
# -------------------------------
@app.route("/environment-score", methods=["POST"])
def environment_score():
    data = request.get_json(force=True)

    # -------------------------------
    # User Inputs
    # -------------------------------
    weight = float(data.get("weight", 1.0))       # kg
    fragility = int(data.get("fragility", 1))     # 1–5

    # -------------------------------
    # Fetch Materials from DB
    # -------------------------------
    query = text("SELECT * FROM target_material_data")
    with engine.connect() as conn:
        materials = [dict(row._mapping) for row in conn.execute(query)]

    if not materials:
        return jsonify({"error": "No materials found"}), 400

    scored_materials = []

    for m in materials:

        # -------------------------------
        # Material Properties (DB)
        # -------------------------------
        material_capacity = float(m.get("weight_capacity", 0))
        material_strength = float(m.get("strength", 0))

        base_biodegradability = float(m.get("biodegradability_score", 0)) * 100
        base_recyclability = float(m.get("recyclability", 0)) * 100

        # -------------------------------
        # Product vs Material Penalties
        # -------------------------------
        capacity_ratio = weight / material_capacity if material_capacity > 0 else 2.0
        capacity_penalty = max(0, capacity_ratio - 1)

        strength_requirement = fragility / 5
        strength_penalty = max(
            0, strength_requirement - (material_strength / 5)
        )

        # -------------------------------
        # Dynamic Eco Adjustments (FIX)
        # -------------------------------
        eco_penalty = (0.6 * capacity_penalty) + (0.4 * strength_penalty)

        biodegradability = base_biodegradability * (1 - eco_penalty)
        recyclability = base_recyclability * (1 - (strength_penalty * 1.2))

        biodegradability = max(0, min(biodegradability, 100))
        recyclability = max(0, min(recyclability, 100))

        # -------------------------------
        # CO₂ Impact Index
        # -------------------------------
        effective_weight = weight * (1 + capacity_penalty)
        effective_fragility = fragility * (1 + strength_penalty)

        co2_percent = min(
            (0.6 * effective_weight / 5 + 0.4 * effective_fragility / 5) * 100,
            100
        )

        eco_benefit = 0.5 * biodegradability + 0.5 * recyclability

        co2_impact_index = co2_percent * (100 - eco_benefit) / 100
        co2_impact_index = max(0, min(co2_impact_index, 100))

        # -------------------------------
        # Material Suitability Score
        # -------------------------------
        material_suitability = 100 - co2_impact_index

        scored_materials.append({
            "biodegradability_percent": round(biodegradability, 2),
            "recyclability_percent": round(recyclability, 2),
            "co2_impact_index_percent": round(co2_impact_index, 2),
            "material_suitability_score_percent": round(material_suitability, 2)
        })

    # -------------------------------
    # Best Material Selection
    # -------------------------------
    scored_materials.sort(
        key=lambda x: x["material_suitability_score_percent"],
        reverse=True
    )

    return jsonify(scored_materials[0])

 
# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
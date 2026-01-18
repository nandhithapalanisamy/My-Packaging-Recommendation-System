# Imports
import joblib
from sqlalchemy import text
from db import engine
from feature_engineering import generate_features

from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from init_db import init_db

init_db()  # runs once on startup

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
model_dir = os.path.join(BASE_DIR, "..", "Dataset_Preparation", "saved_models")

cost_model_path = os.path.join(model_dir, "random_forest_cost.pkl")
co2_model_path = os.path.join(model_dir, "xgboost_co2.pkl")



LAST_USER_INPUT = { "weight": None, "fragility": None }

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Load ML Models
# -------------------------------
cost_model = joblib.load(cost_model_path)
co2_model = joblib.load(co2_model_path)

# -------------------------------
# Recommendation API
# -------------------------------
@app.route("/recommend", methods=["POST"])
def recommend_material():
    data = request.json
    weight = float(data.get("weight", 1.0))
    fragility = int(data.get("fragility", 1))  # 1–5
    LAST_USER_INPUT["weight"] = weight 
    LAST_USER_INPUT["fragility"] = fragility
    query = text("SELECT * FROM target_material_data")
    with engine.connect() as conn:
        materials = [dict(row._mapping) for row in conn.execute(query)]

    recommendations = []
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
 

        recommendations.append({
            "material": decode_material_type(m),
            "predicted_cost": round(cost_inr, 2),
            "co2_impact": round(co2_percent, 2),
            "score": round(score, 4)
        })

    recommendations.sort(key=lambda x: x["score"])
    return jsonify(recommendations[:10])
#-------------------------------
# Export Recommendations to Excel
#-------------------------------
@app.route("/api/export/recommendations/excel", methods=["POST"])
def export_recommendations_excel():
    import pandas as pd
    from flask import request, send_file, jsonify
    import io

    data = request.get_json(force=True)
    recommendations = data.get("recommendations", [])

    if not recommendations:
        return jsonify({"error": "No recommendation data received"}), 400

    df = pd.DataFrame(recommendations)

    df.rename(columns={
        "material": "Material Name",
        "predicted_cost": "Predicted Cost (INR)",
        "co2_impact": "CO2 Impact (%)",
        "score": "AI Score"
    }, inplace=True)

    output = io.BytesIO()
    # Correct usage: no writer.save()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Recommendations')

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="AI_Recommended_Materials.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -------------------------------
# Material Prediction Function
# -------------------------------
def recommend_material_data(weight, fragility, baseline_only=None):
    # --------------------------------
    # Fetch materials
    # --------------------------------
    if baseline_only:
        materials = [baseline_only]
    else:
        query = text("SELECT * FROM target_material_data")
        with engine.connect() as conn:
            materials = [dict(row._mapping) for row in conn.execute(query)]

    recommendations = []

    for m in materials:

        # --------------------------------
        # SAFE MATERIAL VALUES
        # --------------------------------
        material_capacity = float(m.get("weight_capacity") or 0.0)
        material_strength = float(m.get("strength") or 0.0)

        # --------------------------------
        # WEIGHT vs CAPACITY
        # --------------------------------
        if material_capacity > 0:
            capacity_ratio = weight / material_capacity
        else:
            capacity_ratio = 2.0

        capacity_penalty = max(0, capacity_ratio - 1)

        # --------------------------------
        # FRAGILITY vs STRENGTH
        # --------------------------------
        strength_requirement = fragility / 5
        strength_penalty = max(
            0, strength_requirement - (material_strength / 5)
        )

        # --------------------------------
        # FEATURE ENGINEERING
        # --------------------------------
        m["effective_weight"] = weight * (1 + capacity_penalty)
        m["effective_fragility"] = fragility * (1 + strength_penalty)

        features = generate_features(m)

        feature_columns = [
            "weight_capacity",
            "biodegradability_score",
            "co2_emission",
            "recyclability",
            "strength",
            "compostable",
            "water_resistance",
        ]

        features_df = pd.DataFrame([features], columns=feature_columns)


        # --------------------------------
        # MODEL PREDICTION
        # --------------------------------
        cost_raw = float(cost_model.predict(features_df)[0])
        co2_raw = float(co2_model.predict(features_df)[0])

        # --------------------------------
        # APPLY PENALTIES
        # --------------------------------
        adjusted_cost = cost_raw * (1 + capacity_penalty + strength_penalty)
        adjusted_co2 = co2_raw * (1 + capacity_penalty)

        # --------------------------------
        # SCALE
        # --------------------------------
        cost_inr = adjusted_cost * COST_SCALE_INR
        co2_percent = min(max(adjusted_co2 * CO2_SCALE_PERCENT, 0), 100)

        score = (0.6 * adjusted_co2) + (0.4 * adjusted_cost)

        recommendations.append({
            "material": decode_material_type(m),
            "predicted_cost": round(cost_inr, 2),
            "predicted_co2": round(co2_percent, 2),
            "score": round(score, 4)
        })

    recommendations.sort(key=lambda x: x["score"])

    # Baseline → return single prediction
    if baseline_only:
        return recommendations[0]

    return recommendations[:10]


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
# Additional BI Dashboard Routes
# -------------------------------

# Dashboard page
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")  # Use separate HTML file


#function for calculating the reductions
def calculate_reduction(baseline_value, target_value):
    if baseline_value == 0:
        return 0.0
    return round(((baseline_value - target_value) / baseline_value) * 100, 2)



# -------------------------------
# Material Comparison Endpoint
# Baseline Comparison API
# -------------------------------
@app.route("/api/material-comparison", methods=["POST"])
def material_comparison():
    data = request.get_json(force=True)

    weight = float(data.get("weight") or 1.0)
    fragility = float(data.get("fragility") or 1.0)
    baseline_name = data.get("selected_material")

    if not baseline_name:
        return jsonify({"error": "Baseline material not provided"}), 400

    # --------------------------------
    # FIND BASELINE FROM FULL DB
    # --------------------------------
    query = text("SELECT * FROM target_material_data")
    with engine.connect() as conn:
        materials = [dict(row._mapping) for row in conn.execute(query)]

    baseline_row = next(
        (m for m in materials
         if decode_material_type(m).lower() ==
         baseline_name.lower()),
        None
    )

    if not baseline_row:
        return jsonify({"error": "Baseline material not found"}), 400

    # --------------------------------
    # BASELINE PREDICTION
    # --------------------------------
    baseline = recommend_material_data(
        weight, fragility, baseline_only=baseline_row
    )

    baseline_cost = baseline["predicted_cost"]
    baseline_co2 = baseline["predicted_co2"]

    # --------------------------------
    # AI RECOMMENDATIONS
    # --------------------------------
    recommendations = recommend_material_data(weight, fragility)

    def reduction(base, value):
        return round(max(0, ((base - value) / base) * 100), 2) if base > 0 else 0

    table = []
    total_cost_red = 0
    total_co2_red = 0

    for m in recommendations:
        cost_red = reduction(baseline_cost, m["predicted_cost"])
        co2_red = reduction(baseline_co2, m["predicted_co2"])

        total_cost_red += cost_red
        total_co2_red += co2_red

        table.append({
            "material": m["material"],
            "predicted_cost": m["predicted_cost"],
            "predicted_co2": m["predicted_co2"],
            "cost_reduction_percent": cost_red,
            "co2_reduction_percent": co2_red
        })

    count = len(recommendations)

    return jsonify({
        "average_cost_reduction": round(total_cost_red / count, 2),
        "average_co2_reduction": round(total_co2_red / count, 2),
        "materials": table
    })

@app.route("/api/export/material-comparison/excel", methods=["POST"])
def export_material_comparison_excel():
    import pandas as pd
    from flask import request, send_file, jsonify
    import io

    data = request.get_json(force=True)
    comparison = data.get("comparison", [])

    if not comparison:
        return jsonify({"error": "No material comparison data received"}), 400

    df = pd.DataFrame(comparison)

    df.rename(columns={
        "material": "Material Name",
        "actual_cost": "Actual Cost (INR)",
        "predicted_cost": "Predicted Cost (INR)",
        "actual_co2": "Actual CO2 (%)",
        "predicted_co2": "Predicted CO2 (%)",
        "cost_saving": "Cost Saving (INR)",
        "co2_reduction": "CO2 Reduction (%)"
    }, inplace=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Material Comparison")

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="Material_Comparison_Report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.route("/api/get-top-materials", methods=["POST"])
def get_top_materials():
    data = request.get_json()

    weight = float(data.get("weight"))
    fragility = int(data.get("fragility"))

    # Call your existing function
    top_materials = recommend_material_data(weight, fragility)

    return jsonify(top_materials)
def calculate_material_performance(cost, co2):
    # Lower cost & CO2 → higher performance
    cost_score = 1 / (cost + 1)
    co2_score = 1 / (co2 + 1)
    return ((cost_score + co2_score) / 2) * 100

def calculate_material_performance(material_row):
    """
    Returns a single performance score (0–100)
    based on ACTUAL cost (₹) and CO2 (%)
    """

    cost_inr = float(material_row.get("cost_efficiency_index", 0)) * COST_SCALE_INR
    co2_percent = float(material_row.get("co2_impact_index", 0)) * CO2_SCALE_PERCENT

    # Guard
    if cost_inr <= 0 and co2_percent <= 0:
        return 0.0

    # Reference values (standards)
    MAX_COST = 10000    # ₹ — standard upper bound
    MAX_CO2 = 100       # %

    cost_score = max(0, 1 - (cost_inr / MAX_COST))
    co2_score = max(0, 1 - (co2_percent / MAX_CO2))

    performance = ((cost_score + co2_score) / 2) * 100

    return round(performance, 2)

@app.route("/api/material-performance", methods=["POST"])
def material_performance():

    selected_material = request.json.get("material")

    if not selected_material:
        return jsonify({"error": "Material not provided"}), 400

    query = text("SELECT * FROM target_material_data")

    with engine.connect() as conn:
        materials = [dict(row._mapping) for row in conn.execute(query)]

    material_row = next(
        (m for m in materials if decode_material_type(m) == selected_material),
        None
    )

    if not material_row:
        return jsonify({"error": "Material not found"}), 404

    performance = calculate_material_performance(material_row)

    return jsonify({
        "material": selected_material,
        "performance": performance
    })




@app.route("/api/material-performance-trend", methods=["GET"])
def material_performance_trend():
    """
    Returns:
    - Actual Cost in ₹
    - Actual CO2 in %
    - Performance % based on real cost & CO2
    """

    query = text("SELECT * FROM target_material_data")

    with engine.connect() as conn:
        materials = [dict(row._mapping) for row in conn.execute(query)]

    if not materials:
        return jsonify({"data": []})

    # Convert model outputs to real-world values
    cost_values = [
        float(m.get("cost_efficiency_index", 0)) * COST_SCALE_INR
        for m in materials
    ]
    co2_values = [
        float(m.get("co2_impact_index", 0)) * CO2_SCALE_PERCENT
        for m in materials
    ]

    max_cost = max(cost_values)
    max_co2 = max(co2_values)

    result_list = []

    for m in materials:
        cost_inr = float(m.get("cost_efficiency_index", 0)) * COST_SCALE_INR
        co2_percent = float(m.get("co2_impact_index", 0)) * CO2_SCALE_PERCENT

        # Performance calculation (lower cost & CO2 → higher score)
        cost_score = 0 if max_cost == 0 else (1 - (cost_inr / max_cost))
        co2_score = 0 if max_co2 == 0 else (1 - (co2_percent / max_co2))

        performance_percent = ((cost_score + co2_score) / 2) * 100

        material_name = decode_material_type(m)

        result_list.append({
            "material": material_name,
            "co2_percent": round(co2_percent, 2),          # ACTUAL %
            "cost_percent": round(cost_inr, 2),            # ACTUAL ₹
            "performance_percent": round(performance_percent, 2)
        })

    # Best material first
    result_list.sort(key=lambda x: x["performance_percent"], reverse=True)

    return jsonify({"data": result_list[:5]})

@app.route("/api/export/material-performance/excel", methods=["POST"])
def export_material_performance_excel():
    import pandas as pd
    from flask import request, send_file, jsonify
    import io

    data = request.get_json(force=True)
    performance_data = data.get("performance", [])

    if not performance_data:
        return jsonify({"error": "No material performance data received"}), 400

    df = pd.DataFrame(performance_data)

    df.rename(columns={
        "material": "Material Name",
        "performance": "Overall Performance (%)",
        "cost_efficiency": "Cost Efficiency (%)",
        "co2_efficiency": "CO2 Efficiency (%)"
    }, inplace=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Performance Metrics")

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="Material_Performance_Report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



@app.route("/api/sustainability-report", methods=["GET"])
def sustainability_report():
    query = text("SELECT * FROM target_material_data")

    with engine.connect() as conn:
        rows = [dict(row._mapping) for row in conn.execute(query)]

    report = []

    for row in rows:
        # 🔹 extract material name from one-hot columns
        material_name = "Unknown Material"
        for key, value in row.items():
            if key.startswith("material_type_") and value == 1:
                material_name = key.replace("material_type_", "").replace("_", " ").title()
                break

        # 🔹 compute sustainability score (0–100)
        sustainability_score = (
            row["biodegradability_score"] * 25 +
            row["recyclability"] * 20 +
            (1 - row["co2_impact_index"]) * 25 +
            row["cost_efficiency_index"] * 20 +
            row["material_suitability_score"] * 10
        )

        report.append({
            "material": material_name,
            "sustainability_score": round(sustainability_score, 2),
            "co2_impact": round(row["co2_impact_index"] * 100, 2),
            "cost_efficiency": round(row["cost_efficiency_index"] * 100, 2)
        })

    #  sort & pick top 10
    top_10 = sorted(
        report,
        key=lambda x: x["sustainability_score"],
        reverse=True
    )[:10]

    return jsonify({"data": top_10})
@app.route("/api/export/sustainability-report/excel", methods=["POST"])
def export_sustainability_report_excel():
    import pandas as pd
    from flask import request, send_file
    import io

    data = request.get_json(force=True)
    report_data = data.get("report", [])

    if not report_data:
        return jsonify({"error": "No report data received"}), 400

    df = pd.DataFrame(report_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sustainability Report")

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="Sustainability_Report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
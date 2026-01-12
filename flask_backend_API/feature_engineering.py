def generate_features(material):
    return [
        material["weight_capacity"],
        material["biodegradability_score"],
        material["co2_emission"],
        material["recyclability"],
        material["strength"],
        material["compostable"],
        material["water_resistance"]
    ]
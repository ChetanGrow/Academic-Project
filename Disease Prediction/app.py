from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load model and feature names
try:
    model, feature_names = joblib.load('disease_prediction_model.joblib')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Scaling and normal ranges
scaling_ranges = {
    "RBC": (3.5, 6.0),
    "HGB": (10.0, 18.0),
    "HCT": (30.0, 55.0),
    "MCV": (70.0, 110.0),
    "MCHC": (28.0, 38.0),
    "RDW": (10.0, 20.0),
    "PLT": (100, 600),
    "WBC": (3.0, 15.0)
}

normal_ranges = {
    "RBC": (4.5, 5.5),
    "HGB": (12.0, 16.0),
    "HCT": (36.0, 48.0),
    "MCV": (80.0, 100.0),
    "MCHC": (32.0, 36.0),
    "RDW": (11.5, 14.5),
    "PLT": (150, 450),
    "WBC": (4.0, 11.0)
}

# Create abnormal feature columns
def create_abnormal_features(df, normal_ranges):
    for col, (low, high) in normal_ranges.items():
        if col in df.columns:
            df[f"{col}_abnormal"] = ((df[col] < low) | (df[col] > high)).astype(int)
    return df

# Normalize input values using scaling ranges
def normalize_input(input_dict, scaling_ranges):
    normalized = {}
    for param, value in input_dict.items():
        if param in scaling_ranges:
            min_val, max_val = scaling_ranges[param]
            norm = (value - min_val) / (max_val - min_val)
            normalized[param] = max(0, min(1, norm))
        else:
            normalized[param] = value
    return normalized

# Preprocess input to handle unit mismatches
def preprocess_input(raw_params):
    processed_params = raw_params.copy()
    for param in ["RBC", "PLT", "WBC"]:
        if param in processed_params and processed_params[param] > scaling_ranges[param][1] * 10:
            processed_params[param] /= 10
    return processed_params

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "parameters" not in data:
        return jsonify({"error": "Missing 'parameters' in input JSON"}), 400

    try:
        raw_params = data["parameters"]

        # Preprocess input to handle unit mismatches
        processed_params = preprocess_input(raw_params)

        # Create DataFrame from processed parameters
        df = pd.DataFrame([processed_params]).astype(float)

        # Add abnormal features
        df = create_abnormal_features(df, normal_ranges)

        # Normalize the data
        normalized_params = normalize_input(df.iloc[0].to_dict(), scaling_ranges)
        df = pd.DataFrame([normalized_params])

        # Ensure all expected model features are present
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0

        # Reorder columns to match model
        df = df[feature_names]

        # Make prediction
        prediction = model.predict(df)[0]
        # Get prediction probabilities
        probs = model.predict_proba(df)[0]
        class_labels = model.classes_
        prob_dict = {label: float(prob) for label, prob in zip(class_labels, probs)}
        
        return jsonify({"prediction": "Possibally " + prediction, "probabilities": prob_dict})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
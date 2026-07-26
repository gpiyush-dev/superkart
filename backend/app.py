"""
Flask backend API for the SuperKart Sales Forecasting model.

Endpoints
---------
GET  /health   -> simple health check
POST /predict  -> accepts a single record (dict) or a list of records,
                   returns predicted Product_Store_Sales_Total value(s)
"""
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = os.path.join(os.path.dirname(__file__), "superkart_best_model.pkl")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

EXPECTED_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    if payload is None:
        return jsonify({"error": "No JSON payload received"}), 400

    # Accept either a single record (dict) or a list of records
    records = payload if isinstance(payload, list) else [payload]

    try:
        input_df = pd.DataFrame(records)
        missing = set(EXPECTED_COLUMNS) - set(input_df.columns)
        if missing:
            return jsonify({"error": f"Missing required fields: {sorted(missing)}"}), 400

        input_df = input_df[EXPECTED_COLUMNS]
        predictions = model.predict(input_df)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    result = [{"Predicted_Product_Store_Sales_Total": float(p)} for p in predictions]
    return jsonify(result if isinstance(payload, list) else result[0]), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

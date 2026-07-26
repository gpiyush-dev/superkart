"""
Streamlit frontend for the SuperKart Sales Forecasting model.
Talks to the Flask backend's /predict endpoint for single and batch inference.
"""
import os
import requests
import pandas as pd
import streamlit as st

# The backend URL - when running via docker-compose / a forwarded port,
# set this via an environment variable, e.g. BACKEND_URL=http://backend:8000
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart — Sales Revenue Forecast")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Inference"])

# ---------------- Single prediction ----------------
with tab1:
    st.subheader("Enter product / store details")

    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05)
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=140.0)
        product_id_char = st.selectbox("Product Id Prefix", ["FD", "DR", "NC"])
    with col2:
        store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1",
                                                   "Supermarket Type2", "Food Mart"])
        store_age_years = st.number_input("Store Age (Years)", min_value=0, value=15)
        product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age_years,
            "Product_Type_Category": product_type_category,
        }
        try:
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=15)
            response.raise_for_status()
            prediction = response.json()["Predicted_Product_Store_Sales_Total"]
            st.success(f"Predicted Sales Total: {prediction:,.2f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ---------------- Batch inference ----------------
with tab2:
    st.subheader("Upload a CSV for batch inference")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_input = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", batch_input.head())

        if st.button("Run Batch Prediction"):
            records = batch_input.to_dict(orient="records")
            try:
                response = requests.post(f"{BACKEND_URL}/predict", json=records, timeout=60)
                response.raise_for_status()
                preds = [r["Predicted_Product_Store_Sales_Total"] for r in response.json()]
                batch_output = batch_input.copy()
                batch_output["Predicted_Product_Store_Sales_Total"] = preds
                st.success("Batch prediction complete.")
                st.dataframe(batch_output)
                st.download_button(
                    "Download Predictions as CSV",
                    batch_output.to_csv(index=False).encode("utf-8"),
                    "superkart_batch_predictions.csv",
                    "text/csv",
                )
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")

import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit App Title
st.title("💳 Customer Segmentation App")
st.markdown("Enter transaction details to predict the customer segment.")

# Input Fields
transaction_frequency = st.number_input("Transaction Frequency", min_value=1, step=1)
avg_transaction_value = st.number_input("Average Transaction Value", min_value=1.0, step=0.1)

# Predict Button
if st.button("Predict Customer Segment"):
    # Prepare input data
    input_data = np.array([[transaction_frequency, avg_transaction_value]])
    input_data_scaled = scaler.transform(input_data)

    # Predict cluster
    cluster = kmeans.predict(input_data_scaled)[0]

    # Display result
    st.success(f"🛒 The customer belongs to **Segment {cluster}**")

import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("loan_default_model.pkl")

st.title("Loan Default Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=70, value=30)
income = st.number_input("Annual Income", min_value=120000, max_value=600000, value=120000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
loan_amount = st.number_input("Loan Amount", min_value=5000, max_value=400000, value=5000)
interest_rate = st.number_input("Interest Rate", min_value=2.5, max_value=15.0, value=5.5)
loan_term = st.selectbox("Loan Term (years)", [15, 20, 30])
income_to_loan_ratio = income / loan_amount
monthly_income  = income / 12
monthly_payment = loan_amount * (interest_rate / 100) / (1 - (1 + interest_rate / 100)**(-loan_term))

# Prediction button
if st.button("Predict Default Risk"):
    features = np.array([age, income, credit_score, loan_amount, interest_rate, loan_term, income_to_loan_ratio, monthly_income, monthly_payment]).reshape(1, -1)
    prediction = model.predict(features)[0]
    result = "Default" if prediction == 1 else "No Default"
    st.write(f"Prediction: {result}")

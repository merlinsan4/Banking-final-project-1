import streamlit as st
import joblib
import numpy as np
from scipy.sparse import csr_matrix

# Load the trained model and data
knn = joblib.load("knn_model.pkl")
interaction_matrix = joblib.load("interaction_matrix.pkl")
product_mapping = joblib.load("product_mapping.pkl")

# Function to recommend products
def recommend_products(customer_id):
    if customer_id not in interaction_matrix.index:
        return ["Customer not found"]
    
    customer_index = interaction_matrix.index.get_loc(customer_id)
    customer_vector = csr_matrix(interaction_matrix.iloc[customer_index]).toarray()
    distances, indices = knn.kneighbors(customer_vector)
    
    recommended_products = set()
    for idx in indices[0]:
        if idx != customer_index:
            similar_products = interaction_matrix.iloc[idx]
            recommended_products.update(similar_products[similar_products > 0].index)
    
    return [product_mapping.get(p, "Unknown Product") for p in recommended_products][:5]

# Streamlit UI
st.title("Banking Product Recommendation System")
customer_id = st.text_input("Enter Customer ID:")

if st.button("Get Recommendations"):
    if customer_id:
        recommendations = recommend_products(customer_id)
        st.write("Recommended Products:")
        for prod in recommendations:
            st.write(f"- {prod}")

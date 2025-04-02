import streamlit as st
import numpy as np
import pandas as pd
# Uncomment the next two lines if loading the model from a pickle file:
import pickle
best_model = pickle.load(open('wheat_model_proto.pkl', 'rb'))

# If the model is already defined in this context:
# from your_model_module import best_model

st.title("Price Prediction App")

st.write("Enter the following details to predict prices:")

# Input widgets
month = st.number_input("Month", min_value=1, max_value=12, value=1)
year = st.number_input("Year", min_value=2000, max_value=2100, value=2020)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
district_name_enc = st.number_input("District Encoding", value=0)
commodity_name_enc = st.number_input("Commodity Encoding", value=0)
state_name_enc = st.number_input("State Encoding", value=0)

if st.button("Predict"):
    # Create input as a 2D array (model expects multiple samples)
    input_data = np.array([[month, year, day, district_name_enc, commodity_name_enc, state_name_enc]])
    
    # Make prediction using the trained model
    predictions = best_model.predict(input_data)
    
    # Extract predicted values for each target
    pred_modal, pred_min, pred_max = predictions[0]
    
    st.subheader("Predicted Prices")
    st.write(f"**Average Modal Price:** {pred_modal:.2f}")
    st.write(f"**Average Min Price:** {pred_min:.2f}")
    st.write(f"**Average Max Price:** {pred_max:.2f}")

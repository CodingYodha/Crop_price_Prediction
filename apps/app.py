import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# -------------------------------
# Custom transformer for preprocessing
# -------------------------------
class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, freq_mapping_district=None, freq_mapping_commodity=None, freq_mapping_state=None):
        self.freq_mapping_district = freq_mapping_district
        self.freq_mapping_commodity = freq_mapping_commodity
        self.freq_mapping_state = freq_mapping_state

    def fit(self, X, y=None):
        # Expecting X with columns: 'date', 'district_name', 'commodity_name', 'state_name'
        if self.freq_mapping_district is None:
            self.freq_mapping_district = X['district_name'].value_counts().to_dict()
        if self.freq_mapping_commodity is None:
            self.freq_mapping_commodity = X['commodity_name'].value_counts().to_dict()
        if self.freq_mapping_state is None:
            self.freq_mapping_state = X['state_name'].value_counts().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        # Convert the date column and extract day, month, year
        X['date'] = pd.to_datetime(X['date'])
        X['day'] = X['date'].dt.day
        X['month'] = X['date'].dt.month
        X['year'] = X['date'].dt.year
        # Frequency encoding for the categorical columns
        X['district_name_enc'] = X['district_name'].map(self.freq_mapping_district)
        X['commodity_name_enc'] = X['commodity_name'].map(self.freq_mapping_commodity)
        X['state_name_enc'] = X['state_name'].map(self.freq_mapping_state)
        # Drop the original categorical columns and date
        X.drop(['district_name', 'commodity_name', 'state_name', 'date'], axis=1, inplace=True)
        # Return columns in the order expected by the model
        return X[['month', 'year', 'day', 'district_name_enc', 'commodity_name_enc', 'state_name_enc']]

# -------------------------------
# Load training data and the trained model
# -------------------------------
# Load your training data (ensure the CSV has columns: date, district_name, commodity_name, state_name, etc.)
training_data = pd.read_csv('edited_training_data.csv')

# Load the trained model from pickle (update filename if needed)
with open('wheat_model_proto.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Fit the preprocessor on the training data
preprocessor = DataPreprocessor()
preprocessor.fit(training_data)

# Build the pipeline using the fitted preprocessor and loaded model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("Crop Price Prediction App")
st.write("Enter the details below to predict prices:")

# User inputs: raw values (not encoded)
input_date = st.date_input("Select Date", date.today())
district_name = st.text_input("District Name", value="Enter district name")
commodity_name = st.text_input("Commodity Name", value="Enter commodity name")
state_name = st.text_input("State Name", value="Enter state name")

if st.button("Predict"):
    # Create a DataFrame with a single row of raw input
    input_df = pd.DataFrame({
        'date': [input_date],
        'district_name': [district_name],
        'commodity_name': [commodity_name],
        'state_name': [state_name]
    })
    
    # Run the pipeline to process inputs and generate prediction
    prediction = pipeline.predict(input_df)
    # prediction is an array with three outputs: [avg_modal_price, avg_min_price, avg_max_price]
    pred_modal, pred_min, pred_max = prediction[0]
    
    st.subheader("Predicted Prices")
    st.write(f"**Average Modal Price:** {pred_modal:.2f}")
    st.write(f"**Average Min Price:** {pred_min:.2f}")
    st.write(f"**Average Max Price:** {pred_max:.2f}")

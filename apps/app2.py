import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
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
        # Drop original categorical columns and date
        X.drop(['district_name', 'commodity_name', 'state_name', 'date'], axis=1, inplace=True)
        # Return columns in the order expected by the model
        return X[['month', 'year', 'day', 'district_name_enc', 'commodity_name_enc', 'state_name_enc']]

# -------------------------------
# Load training data and the trained model
# -------------------------------
# Load your training data. This CSV should have columns:
# 'date', 'district_name', 'commodity_name', 'state_name', etc.
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

st.write("### Single Date Prediction")
st.write("Enter details to predict prices for a specific date:")

# Inputs for single date prediction
input_date = st.date_input("Select Date", date.today())
district_name = st.text_input("District Name", value="Enter district name")
commodity_name = st.text_input("Commodity Name", value="Enter commodity name")
state_name = st.text_input("State Name", value="Enter state name")

if st.button("Predict for Selected Date"):
    input_df = pd.DataFrame({
        'date': [input_date],
        'district_name': [district_name],
        'commodity_name': [commodity_name],
        'state_name': [state_name]
    })
    prediction = pipeline.predict(input_df)
    pred_modal, pred_min, pred_max = prediction[0]
    
    st.subheader("Predicted Prices")
    st.write(f"**Average Modal Price:** {pred_modal:.2f}")
    st.write(f"**Average Min Price:** {pred_min:.2f}")
    st.write(f"**Average Max Price:** {pred_max:.2f}")

st.markdown("---")
st.write("### Forecast Price Trend")
st.write("Select a forecast horizon to see how prices might change in the coming days:")

# Option for forecast horizon
horizon_option = st.selectbox("Forecast Horizon", ["Next Week", "Next Month", "Next Year"])

# Determine forecast days based on selection
if horizon_option == "Next Week":
    num_days = 7
elif horizon_option == "Next Month":
    num_days = 30
elif horizon_option == "Next Year":
    num_days = 365
else:
    num_days = 7

if st.button("Generate Forecast Plot"):
    # Generate a date range starting tomorrow
    start_date = date.today() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # For forecasting, we use the same district, commodity, and state inputs
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'district_name': [district_name]*num_days,
        'commodity_name': [commodity_name]*num_days,
        'state_name': [state_name]*num_days
    })
    
    # Predict for the future dates
    forecast_predictions = pipeline.predict(forecast_df)
    
    # Create a DataFrame with predictions and corresponding dates
    forecast_results = pd.DataFrame(forecast_predictions, columns=['avg_modal_price', 'avg_min_price', 'avg_max_price'])
    forecast_results['date'] = future_dates
    
    # Plot the forecasted prices
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(forecast_results['date'], forecast_results['avg_modal_price'], label="Avg Modal Price", marker='o')
    ax.plot(forecast_results['date'], forecast_results['avg_min_price'], label="Avg Min Price", marker='o')
    ax.plot(forecast_results['date'], forecast_results['avg_max_price'], label="Avg Max Price", marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"Forecasted Prices for {horizon_option}")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.write("Note: For more robust time-series forecasting, consider training dedicated forecasting models (e.g., Prophet, ARIMA).")

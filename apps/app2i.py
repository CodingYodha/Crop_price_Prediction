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
        # Fit only if mappings are not provided (e.g., during initial training)
        # For prediction, we assume mappings are loaded or passed in
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

        # --- Frequency encoding for the categorical columns ---
        # Use .get() with a default value (e.g., 0 or 1) for unseen categories during prediction
        default_freq = 1 # Assign a low frequency to unseen categories

        # Check if mappings exist before applying .map()
        if self.freq_mapping_district:
             X['district_name_enc'] = X['district_name'].map(lambda name: self.freq_mapping_district.get(name, default_freq))
        else:
             st.error("District frequency mapping is missing in the preprocessor.")
             X['district_name_enc'] = 0 # Or handle error appropriately

        if self.freq_mapping_commodity:
             X['commodity_name_enc'] = X['commodity_name'].map(lambda name: self.freq_mapping_commodity.get(name, default_freq))
        else:
             st.error("Commodity frequency mapping is missing in the preprocessor.")
             X['commodity_name_enc'] = 0 # Or handle error appropriately

        if self.freq_mapping_state:
             X['state_name_enc'] = X['state_name'].map(lambda name: self.freq_mapping_state.get(name, default_freq))
        else:
             st.error("State frequency mapping is missing in the preprocessor.")
             X['state_name_enc'] = 0 # Or handle error appropriately
        # -----------------------------------------------------

        # Drop original categorical columns and date
        X.drop(['district_name', 'commodity_name', 'state_name', 'date'], axis=1, inplace=True)

        # Return columns in the order expected by the model
        # Ensure this order matches the training phase exactly
        expected_cols = ['month', 'year', 'day', 'district_name_enc', 'commodity_name_enc', 'state_name_enc']
        # Add missing columns if any (though ideally handled by .get in map)
        for col in expected_cols:
             if col not in X.columns:
                  X[col] = 0 # Add missing column with default value

        return X[expected_cols]

# -------------------------------
# Load training data (for fitting preprocessor) and the trained model
# -------------------------------
try:
    # Load your training data. This is primarily needed to FIT the preprocessor
    # to get the frequency mappings if they aren't saved separately.
    training_data = pd.read_csv('edited_training_data.csv') # Ensure this file exists and has the required columns

    # Fit the preprocessor on the training data to generate mappings
    preprocessor = DataPreprocessor()
    preprocessor.fit(training_data[['date', 'district_name', 'commodity_name', 'state_name']])

    # --- Ideally, save and load the FITTED preprocessor ---
    # Example:
    # with open('preprocessor.pkl', 'wb') as f:
    #     pickle.dump(preprocessor, f)
    # with open('preprocessor.pkl', 'rb') as f:
    #     preprocessor = pickle.load(f)
    # This avoids needing training_data.csv during deployment.
    # For this example, we fit it here using the loaded training_data.

    # Load the trained prediction model from pickle
    with open('wheat_model_proto.pkl', 'rb') as file:
        best_model = pickle.load(file)

    # Build the pipeline using the fitted preprocessor and loaded model
    # Note: The preprocessor is already fitted here.
    pipeline = Pipeline([
        ('preprocessor', preprocessor), # Use the already fitted preprocessor instance
        ('model', best_model)
    ])
    MODEL_LOADED = True
except FileNotFoundError as e:
    st.error(f"Error loading file: {e}. Make sure 'edited_training_data.csv' and 'wheat_model_proto.pkl' are in the correct directory.")
    MODEL_LOADED = False
except Exception as e:
    st.error(f"An error occurred during model or data loading: {e}")
    MODEL_LOADED = False

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🌾 Crop Price Prediction App (Maharashtra Focus)")

if MODEL_LOADED: # Only show UI if model and preprocessor loaded successfully

    # --- Define Maharashtra Districts ---
    # You can expand this list based on your actual dataset coverage
    maharashtra_districts = [
        "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara",
        "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli",
        "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban",
        "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar",
        "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
        "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"
        # Add more districts if they appear in your training data for Maharashtra
    ]
    # Sort alphabetically for user convenience
    maharashtra_districts.sort()
    # ------------------------------------

    st.write("### Single Date Prediction")
    st.write("Enter details to predict prices for a specific date:")

    # Inputs for single date prediction
    input_date = st.date_input("Select Date", value=date.today()) # Use value= for default

    # --- Use Selectbox for District ---
    district_name = st.selectbox("Select District (Maharashtra)", options=maharashtra_districts)
    # ----------------------------------

    commodity_name = st.text_input("Commodity Name", value="Wheat") # Example default
    # For state, since we focus on MH districts, we can fix it or keep it flexible
    # Option 1: Fix state to Maharashtra
    state_name = "Maharashtra"
    st.text_input("State Name", value=state_name, disabled=True) # Show fixed state
    # Option 2: Allow other states (keep original text input)
    # state_name = st.text_input("State Name", value="Maharashtra")

    if st.button("Predict for Selected Date"):
        if not district_name or not commodity_name or not state_name:
             st.warning("Please ensure all fields (District, Commodity, State) are filled.")
        else:
            try:
                input_df = pd.DataFrame({
                    'date': [input_date],
                    'district_name': [district_name],
                    'commodity_name': [commodity_name],
                    'state_name': [state_name]
                })

                # Use the pipeline to preprocess and predict
                prediction = pipeline.predict(input_df)
                pred_modal, pred_min, pred_max = prediction[0] # Assumes model outputs 3 values

                st.subheader("Predicted Prices")
                st.metric(label="Predicted Avg Modal Price", value=f"{pred_modal:.2f}")
                st.metric(label="Predicted Avg Min Price", value=f"{pred_min:.2f}")
                st.metric(label="Predicted Avg Max Price", value=f"{pred_max:.2f}")

            except KeyError as e:
                 st.error(f"Prediction Error: Input data might be missing expected columns or have unexpected values. Missing key: {e}")
            except Exception as e:
                 st.error(f"An error occurred during prediction: {e}")


    st.markdown("---")
    st.write("### Forecast Price Trend")
    st.write("Select a forecast horizon (uses the same District/Commodity/State from above):")

    # Option for forecast horizon
    horizon_option = st.selectbox("Forecast Horizon", ["Next Week (7 days)", "Next Month (30 days)", "Next 3 Months (90 days)"])

    # Determine forecast days based on selection
    if "Next Week" in horizon_option:
        num_days = 7
    elif "Next Month" in horizon_option:
        num_days = 30
    elif "Next 3 Months" in horizon_option:
        num_days = 90
    else:
        num_days = 7 # Default

    if st.button("Generate Forecast Plot"):
        if not district_name or not commodity_name or not state_name:
             st.warning("Please ensure District, Commodity, and State are selected/entered above before forecasting.")
        else:
            try:
                # Generate a date range starting tomorrow
                start_date = date.today() + timedelta(days=1)
                # Ensure future_dates are Timestamp objects for consistency if needed
                future_dates = pd.date_range(start=start_date, periods=num_days, freq='D')

                # For forecasting, use the same district, commodity, and state inputs from above
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'district_name': [district_name] * num_days,
                    'commodity_name': [commodity_name] * num_days,
                    'state_name': [state_name] * num_days
                })

                # Predict for the future dates using the pipeline
                forecast_predictions = pipeline.predict(forecast_df)

                # Create a DataFrame with predictions and corresponding dates
                forecast_results = pd.DataFrame(forecast_predictions, columns=['avg_modal_price', 'avg_min_price', 'avg_max_price'])
                forecast_results['date'] = future_dates # Assign the generated dates

                # Plot the forecasted prices using Matplotlib
                fig, ax = plt.subplots(figsize=(12, 6)) # Wider plot
                ax.plot(forecast_results['date'], forecast_results['avg_modal_price'], label="Avg Modal Price", marker='.', linestyle='-')
                ax.plot(forecast_results['date'], forecast_results['avg_min_price'], label="Avg Min Price", marker='.', linestyle='-')
                ax.plot(forecast_results['date'], forecast_results['avg_max_price'], label="Avg Max Price", marker='.', linestyle='-')
                ax.set_xlabel("Date")
                ax.set_ylabel("Predicted Price")
                ax.set_title(f"Forecasted Price Trend for {commodity_name} in {district_name}\n({horizon_option})")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6) # Add grid
                plt.xticks(rotation=45, ha='right') # Improve date readability
                plt.tight_layout() # Adjust layout
                st.pyplot(fig)

                st.info("Disclaimer: This plot shows predictions from the regression model applied to future dates. It reflects trends based on historical patterns learned by the model but is NOT a dedicated time-series forecast (like ARIMA or Prophet) which explicitly models temporal dependencies and seasonality.")

            except KeyError as e:
                 st.error(f"Forecasting Error: Input data might be missing expected columns or have unexpected values. Missing key: {e}")
            except Exception as e:
                 st.error(f"An error occurred during forecasting: {e}")

else:
    st.warning("Model or supporting files could not be loaded. Please check the file paths and try again.")
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler # Needed for scaler type hint potentially
from tensorflow.keras.models import load_model # To load saved model
import joblib # To load saved scaler
import plotly.graph_objects as go
import os

# --- Configuration ---
# Paths where you SAVED the models, scalers, and sequences (adjust if needed)
MODEL_DIR = "E:\elevatetrsest\crop price predictor\Crop_price_Prediction\saved_lstm_models"
TARGET_COLUMNS = ['avg_min_price', 'avg_max_price', 'avg_modal_price']
SEQUENCE_LENGTH = 60 # MUST match the sequence length used during training
PREDICTION_START_DATE_STR = "2024-01-01" # Approx end of training data + 1 day

# --- Resource Loading (Cached) ---
# Use @st.cache_resource for things that shouldn't be copied/hashed like models/scalers
@st.cache_resource
def load_keras_model(target):
    """Loads the saved Keras LSTM model for a specific target."""
    model_path = os.path.join(MODEL_DIR, f"lstm_model_{target}.h5")
    try:
        model = load_model(model_path)
        print(f"Loaded model for {target}")
        return model
    except Exception as e:
        st.error(f"Error loading model for {target} from {model_path}: {e}")
        return None

@st.cache_resource
def load_joblib_scaler(target):
    """Loads the saved scikit-learn scaler for a specific target."""
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{target}.joblib")
    try:
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler for {target}")
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler for {target} from {scaler_path}: {e}")
        return None

# Use @st.cache_data for data that can be hashed (like numpy arrays)
@st.cache_data
def load_last_sequence(target):
    """Loads the last sequence of scaled training data needed to start prediction."""
    sequence_path = os.path.join(MODEL_DIR, f"last_sequence_{target}.npy")
    try:
        last_sequence = np.load(sequence_path)
        print(f"Loaded last sequence for {target}")
        # Ensure it has the correct shape (SEQUENCE_LENGTH, 1)
        if last_sequence.shape == (SEQUENCE_LENGTH,):
            last_sequence = last_sequence.reshape(-1, 1)
        elif last_sequence.shape == (SEQUENCE_LENGTH, 1):
            pass # Shape is already correct
        else:
             raise ValueError(f"Loaded sequence for {target} has unexpected shape: {last_sequence.shape}")
        return last_sequence
    except Exception as e:
        st.error(f"Error loading sequence for {target} from {sequence_path}: {e}")
        return None

# --- Prediction Function ---
def predict_future_prices(models, scalers, last_sequences, n_days_ahead, sequence_length):
    """Predicts prices autoregressively for n days ahead for all targets."""
    if not all(models.values()) or not all(scalers.values()) or not all(last_sequences.values()):
        st.error("One or more models, scalers, or sequences failed to load. Cannot predict.")
        return None, None

    # Store predictions sequence for each target
    predictions_scaled = {target: [] for target in TARGET_COLUMNS}
    # Keep track of the evolving input sequence for each target
    current_sequences = {target: seq.copy() for target, seq in last_sequences.items()} # Use copies

    with st.spinner(f"Generating predictions for {n_days_ahead} days..."):
        for i in range(n_days_ahead):
            for target in TARGET_COLUMNS:
                model = models[target]
                current_seq = current_sequences[target]

                # Reshape input for LSTM: (1, sequence_length, 1 feature)
                input_seq = current_seq.reshape((1, sequence_length, 1))

                # Predict the next step (scaled)
                next_pred_scaled = model.predict(input_seq, verbose=0)[0, 0] # Get scalar value

                # Store scaled prediction
                predictions_scaled[target].append(next_pred_scaled)

                # Update the current sequence for the next prediction (autoregressive step)
                # Append new prediction and remove oldest value
                current_sequences[target] = np.vstack((current_seq[1:], [[next_pred_scaled]]))

    # Inverse transform the predictions
    predictions_inv = {target: [] for target in TARGET_COLUMNS}
    try:
        for target in TARGET_COLUMNS:
            scaler = scalers[target]
            # Reshape predictions for scaler [(pred1,), (pred2,), ...]
            preds_reshaped = np.array(predictions_scaled[target]).reshape(-1, 1)
            predictions_inv[target] = scaler.inverse_transform(preds_reshaped).flatten()
    except Exception as e:
        st.error(f"Error during inverse transformation: {e}")
        return None, predictions_scaled # Return scaled if inverse fails

    return predictions_inv, predictions_scaled

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🌾 Nashik Wheat Price Predictor (LSTM)")
st.markdown("Predicts future average minimum, maximum, and modal prices based on historical data (2002-2023).")
st.markdown(f"*Note: Predictions start from approx. **{PREDICTION_START_DATE_STR}** and use an autoregressive method, which may accumulate errors over longer horizons.*")

# --- Load Resources ---
models = {target: load_keras_model(target) for target in TARGET_COLUMNS}
scalers = {target: load_joblib_scaler(target) for target in TARGET_COLUMNS}
last_sequences = {target: load_last_sequence(target) for target in TARGET_COLUMNS}

# Check if all resources loaded
resources_loaded = all(models.values()) and all(scalers.values()) and all(last_sequences.values())

if not resources_loaded:
    st.error("Failed to load necessary model files. Please ensure models, scalers, and sequences are saved correctly in the 'saved_lstm_models' directory.")
else:
    st.sidebar.header("Prediction Input")
    days_ahead = st.sidebar.number_input(
        "Number of days to predict ahead:",
        min_value=1,
        max_value=90, # Limit prediction horizon
        value=7,
        step=1,
        help=f"How many days after {PREDICTION_START_DATE_STR} to predict?"
    )

    predict_button = st.sidebar.button("Predict Prices", type="primary")

    st.divider()

    if predict_button:
        # Perform prediction
        predictions_inv, predictions_scaled = predict_future_prices(
            models, scalers, last_sequences, days_ahead, SEQUENCE_LENGTH
        )

        if predictions_inv:
            st.header(f"Predicted Prices for {days_ahead} Days Ahead")

            # Calculate prediction dates
            try:
                start_date = datetime.datetime.strptime(PREDICTION_START_DATE_STR, "%Y-%m-%d")
                prediction_dates = [start_date + datetime.timedelta(days=i) for i in range(days_ahead)]
                prediction_dates_str = [d.strftime("%Y-%m-%d") for d in prediction_dates]
            except ValueError:
                st.warning("Could not parse PREDICTION_START_DATE_STR. Using relative days.")
                prediction_dates_str = [f"Day {i+1}" for i in range(days_ahead)]


            # Display final predicted values (for the last day)
            st.subheader(f"Prediction for Day {days_ahead} ({prediction_dates_str[-1]}):")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Price", f"₹{predictions_inv['avg_min_price'][-1]:.2f}")
            with col2:
                st.metric("Modal Price", f"₹{predictions_inv['avg_modal_price'][-1]:.2f}")
            with col3:
                st.metric("Max Price", f"₹{predictions_inv['avg_max_price'][-1]:.2f}")

            st.divider()

            # --- Plot Predictions ---
            st.subheader("Prediction Trend")
            fig = go.Figure()

            colors = {'avg_min_price': 'orange', 'avg_modal_price': 'red', 'avg_max_price': 'green'}
            names = {'avg_min_price': 'Min Price', 'avg_modal_price': 'Modal Price', 'avg_max_price': 'Max Price'}

            for target in TARGET_COLUMNS:
                fig.add_trace(go.Scatter(
                    x=prediction_dates_str,
                    y=predictions_inv[target],
                    mode='lines+markers',
                    name=f'Predicted {names[target]}',
                    line=dict(color=colors[target])
                ))

            fig.update_layout(
                title=f'Predicted Wheat Prices for Next {days_ahead} Days',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Optional: Display Prediction Table ---
            with st.expander("View Prediction Data Table"):
                df_preds = pd.DataFrame({
                    'Date': prediction_dates_str,
                    'Predicted Min Price': predictions_inv['avg_min_price'],
                    'Predicted Modal Price': predictions_inv['avg_modal_price'],
                    'Predicted Max Price': predictions_inv['avg_max_price']
                })
                st.dataframe(df_preds.style.format({
                    'Predicted Min Price': '{:.2f}',
                    'Predicted Modal Price': '{:.2f}',
                    'Predicted Max Price': '{:.2f}'
                }))

        else:
             st.warning("Prediction could not be generated.")
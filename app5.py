import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import datetime
import os # Import os to check file existence

# --- Configuration ---

# Dictionary to manage settings for each commodity
COMMODITY_CONFIG = {
    "Wheat": {
        "path": "wheat_nashik_price_data.csv",
        "states": ["Maharashtra"],
        "districts": ["Nashik"]
    },
    "Cotton": {
        "path": "cotton_data (2003-23).csv",
        "states": ["Maharashtra"],
        "districts": ["Yavatmal"] # As per requirement
    }
}

# Define common columns (assuming they are consistent across files)
# IMPORTANT: Verify these column names exist in BOTH CSV files
TARGET_COLUMNS = ['avg_min_price', 'avg_max_price', 'avg_modal_price']
DATE_COLUMN = 'date' # Ensure this is the correct date column name in both files
STATE_COLUMN = 'state_name' # Ensure this column exists
DISTRICT_COLUMN = 'district_name' # Ensure this column exists
COMMODITY_COLUMN = 'commodity_name' # Ensure this column exists

MIN_DATA_POINTS = 30 # Minimum data points required to train a model

# --- Data Loading and Caching ---
# Cache data loading based on the file path
@st.cache_data
def load_data(path):
    """Loads and preprocesses the data from a given path."""
    if not os.path.exists(path):
        st.error(f"Error: Data file not found at {path}")
        return None
    try:
        df = pd.read_csv(path)

        # --- Basic Preprocessing ---
        # Convert date column
        if DATE_COLUMN not in df.columns:
            st.error(f"Error: Date column '{DATE_COLUMN}' not found in {path}.")
            return None
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
        df.dropna(subset=[DATE_COLUMN], inplace=True) # Drop rows where date conversion failed

        # Ensure price columns are numeric, coerce errors to NaN
        missing_targets = []
        for col in TARGET_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                missing_targets.append(col)
        if missing_targets:
            st.error(f"Error: Target columns {missing_targets} not found in {path}.")
            return None

        # Check for essential filter columns
        for col in [STATE_COLUMN, DISTRICT_COLUMN, COMMODITY_COLUMN]:
             if col not in df.columns:
                 st.warning(f"Warning: Column '{col}' not found in {path}. Filtering might be affected.")
                 # Optionally create a dummy column if needed for filtering consistency
                 # df[col] = "Unknown" # Example if needed


        # Drop rows with missing target values
        df.dropna(subset=TARGET_COLUMNS, inplace=True)

        # Sort by date (important for time series)
        df.sort_values(DATE_COLUMN, inplace=True)

        # Optional: Impute missing prices if needed (example using ffill)
        # df[TARGET_COLUMNS] = df.groupby([STATE_COLUMN, DISTRICT_COLUMN, COMMODITY_COLUMN])[TARGET_COLUMNS].ffill()
        # df.dropna(subset=TARGET_COLUMNS, inplace=True) # Drop again if ffill didn't cover the start

        # Standardize commodity name based on file (optional but good practice)
        # This assumes the file name corresponds to the commodity if the column is messy
        # filename_commodity = os.path.splitext(os.path.basename(path))[0].split('_')[0].capitalize() # Simple guess
        # if COMMODITY_COLUMN in df.columns:
        #     # You might want to clean/standardize df[COMMODITY_COLUMN] here
        #     pass
        # else:
        #     df[COMMODITY_COLUMN] = filename_commodity # Add if missing

        return df

    except pd.errors.EmptyDataError:
        st.error(f"Error: The file at {path} is empty.")
        return None
    except Exception as e:
        st.error(f"Error loading or preprocessing data from {path}: {e}")
        return None

# --- Modeling Function (Unchanged) ---
def train_and_forecast(data, target_column, forecast_periods):
    """Trains a Prophet model and returns the forecast starting from today."""
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = data[[DATE_COLUMN, target_column]].rename(columns={DATE_COLUMN: 'ds', target_column: 'y'})

    # Check for sufficient data points for training
    if len(prophet_df) < MIN_DATA_POINTS:
        st.warning(f"Not enough historical data points ({len(prophet_df)}) for '{target_column}' in the selected group to train. Need at least {MIN_DATA_POINTS}. Skipping forecast.")
        return None, None # Return None for model and forecast

    try:
        # Instantiate and fit Prophet model on historical data
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False, # Adjust based on expected patterns
            daily_seasonality=False   # Adjust based on expected patterns
        )
        model.fit(prophet_df) # Fit the model using historical data

        # --- Create future dataframe STARTING FROM TODAY ---
        current_date = pd.Timestamp.now().normalize() # Get today's date (e.g., 2025-04-03 00:00:00)
        # Create a sequence of dates starting from today for the forecast period
        future_dates = pd.date_range(start=current_date, periods=forecast_periods, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        # ----------------------------------------------------

        # Generate forecast using the dates starting from today
        forecast = model.predict(future_df)
        return model, forecast # Return model (optional) and forecast DataFrame
    except Exception as e:
        st.error(f"Error during forecasting for {target_column}: {e}")
        return None, None

# --- Plotting Function for a Single Target (Unchanged) ---
def plot_single_forecast(historical_data, forecast_data, target_column, title):
    """Creates a Plotly figure for one target's historical data and forecast."""
    fig = go.Figure()
    target_label = target_column.replace("avg_", "").replace("_price", "") # Clean label for display

    # Add historical data trace
    hist_data_col = historical_data[[DATE_COLUMN, target_column]].dropna()
    fig.add_trace(go.Scatter(
        x=hist_data_col[DATE_COLUMN],
        y=hist_data_col[target_column],
        mode='lines',
        name=f'Historical {target_label}',
        line=dict(color='blue')
    ))

    # Add forecast trace
    # Note: forecast_data['ds'] now starts from today
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        mode='lines',
        name=f'Forecast {target_label}',
        line=dict(color='red', dash='dash')
    ))

    # Add uncertainty interval
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_upper'],
        mode='lines', name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_lower'],
        mode='lines', name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(255, 0, 0, 0.2)', # Light red fill for uncertainty
        fill='tonexty',
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=f'Price ({target_label})',
        hovermode="x unified"
    )
    return fig

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🌾 Crop Price Time Series Forecasting (Future Dates)")
st.write(f"Forecasting from current date: {pd.Timestamp.now().normalize().strftime('%Y-%m-%d')}")

# --- User Selections in Sidebar ---
st.sidebar.header("Select Criteria")

# 1. Select Commodity FIRST
selected_commodity = st.sidebar.selectbox(
    "Select Commodity",
    options=list(COMMODITY_CONFIG.keys())
)

# Initialize selection variables
selected_state = None
selected_district = None
df_loaded = None # Variable to hold the loaded dataframe for the chosen commodity

if selected_commodity:
    config = COMMODITY_CONFIG[selected_commodity]
    data_path = config["path"]

    # Attempt to load data for the selected commodity
    df_loaded = load_data(data_path)

    if df_loaded is not None:
        try:
            # 2. Select State (Restricted based on Commodity Config)
            allowed_states = config["states"]
            if not allowed_states:
                 st.sidebar.warning(f"No states configured for {selected_commodity}.")
            elif len(allowed_states) == 1:
                 st.sidebar.text_input("State", value=allowed_states[0], disabled=True)
                 selected_state = allowed_states[0]
                 # Verify state exists in data (optional but good)
                 if STATE_COLUMN in df_loaded.columns and selected_state not in df_loaded[STATE_COLUMN].unique():
                     st.sidebar.warning(f"Configured state '{selected_state}' not found in the actual data of {data_path}.")
                     selected_state = None # Invalidate if not found
            else: # Allow selection if multiple states are configured
                 # Filter available states based on loaded data for robustness
                 available_states_in_data = []
                 if STATE_COLUMN in df_loaded.columns:
                     available_states_in_data = sorted([s for s in df_loaded[STATE_COLUMN].unique() if s in allowed_states])
                 if not available_states_in_data:
                     st.sidebar.warning(f"No data found for configured states {allowed_states} in {data_path}.")
                 else:
                    selected_state = st.sidebar.selectbox("Select State", available_states_in_data)


            # 3. Select District (Restricted based on Commodity Config and Selected State)
            if selected_state:
                allowed_districts = config["districts"]
                if not allowed_districts:
                     st.sidebar.warning(f"No districts configured for {selected_commodity} in {selected_state}.")
                elif len(allowed_districts) == 1:
                     st.sidebar.text_input("District", value=allowed_districts[0], disabled=True)
                     selected_district = allowed_districts[0]
                     # Verify district exists in data for the state (optional but good)
                     if DISTRICT_COLUMN in df_loaded.columns:
                         mask = (df_loaded[STATE_COLUMN] == selected_state)
                         if selected_district not in df_loaded[mask][DISTRICT_COLUMN].unique():
                             st.sidebar.warning(f"Configured district '{selected_district}' not found for state '{selected_state}' in the actual data of {data_path}.")
                             selected_district = None # Invalidate if not found
                else: # Allow selection if multiple districts are configured
                    # Filter available districts based on loaded data and selected state
                    available_districts_in_data = []
                    if DISTRICT_COLUMN in df_loaded.columns and STATE_COLUMN in df_loaded.columns:
                        mask = (df_loaded[STATE_COLUMN] == selected_state)
                        available_districts_in_data = sorted([
                            d for d in df_loaded[mask][DISTRICT_COLUMN].unique() if d in allowed_districts
                        ])
                    if not available_districts_in_data:
                         st.sidebar.warning(f"No data found for configured districts {allowed_districts} in state {selected_state} within {data_path}.")
                    else:
                         selected_district = st.sidebar.selectbox("Select District", available_districts_in_data)

            else: # If state selection failed or wasn't possible
                selected_district = None

        except Exception as e:
            st.sidebar.error(f"Error setting up selection widgets: {e}")
            selected_state = selected_district = None

    else:
        # Data loading failed for the selected commodity
        st.sidebar.error(f"Could not load data for {selected_commodity}. Check file path and format.")
        # Reset selections
        selected_state = selected_district = None


# 4. Forecast Period
forecast_days = st.sidebar.number_input(
    "Select Forecast Period (Days)",
    min_value=7,
    max_value=365,
    value=30
)

# 5. Run Button
run_forecast = st.sidebar.button("Run Forecast")


# --- Filtering and Forecasting ---
if run_forecast:
    # Check if all necessary selections are made and data is loaded
    if selected_commodity and selected_state and selected_district and df_loaded is not None:
        st.header(f"Forecast for {selected_commodity} in {selected_district}, {selected_state}")

        # Filter the already loaded dataframe (df_loaded)
        # Commodity check might be redundant if files are commodity-specific, but good practice
        filter_mask = (
            (df_loaded[STATE_COLUMN] == selected_state) &
            (df_loaded[DISTRICT_COLUMN] == selected_district)
        )
        # Add commodity filter only if the column exists and contains relevant data
        if COMMODITY_COLUMN in df_loaded.columns:
             # This assumes the commodity name in the file matches the selection key
             # You might need cleaning/mapping if they differ (e.g., 'Cotton(Unginned)' vs 'Cotton')
             filter_mask = filter_mask & (df_loaded[COMMODITY_COLUMN].str.contains(selected_commodity, case=False, na=False))


        filtered_df = df_loaded[filter_mask].copy() # Use copy to avoid SettingWithCopyWarning

        # Ensure data is sorted by date AFTER filtering
        filtered_df.sort_values(by=DATE_COLUMN, inplace=True)

        if filtered_df.empty:
            st.warning(f"No historical data found for {selected_commodity} in {selected_district}, {selected_state} after filtering.")
            st.info(f"Please check the contents of '{COMMODITY_CONFIG[selected_commodity]['path']}' for matching entries.")
        else:
            last_hist_date = filtered_df[DATE_COLUMN].max().strftime('%Y-%m-%d')
            st.info(f"Found {len(filtered_df)} historical data points for the selection (latest: {last_hist_date}). Training on this data.")

            all_forecasts = {} # Store forecasts if needed later

            with st.spinner(f"Training models and generating forecasts for {selected_commodity}..."):
                # Loop through each target price type
                for target in TARGET_COLUMNS:
                    st.subheader(f"Forecast: {target.replace('avg_', '').replace('_price', '').capitalize()} Price")

                    # Train model and get forecast starting from TODAY
                    model, forecast = train_and_forecast(filtered_df, target, forecast_days)

                    if forecast is not None:
                        all_forecasts[target] = forecast # Store the forecast

                        # Plot historical data and the specific forecast
                        plot_title = f'{selected_commodity} - {target.replace("avg_", "").replace("_price", "").capitalize()} Price: Historical & Forecast ({forecast_days} days)'
                        fig = plot_single_forecast(filtered_df, forecast, target, plot_title)
                        st.plotly_chart(fig, use_container_width=True)

                        # Display forecast data table for this target (optional)
                        with st.expander(f"View Forecast Data Table for {target}"):
                            f_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                            f_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                            f_display['Date'] = f_display['Date'].dt.strftime('%Y-%m-%d') # Format date
                            st.dataframe(f_display.set_index('Date').style.format("{:.2f}"))
                    else:
                        # Warning message already displayed in train_and_forecast if skipped
                        pass # Do nothing further for this target if forecast failed

    elif run_forecast: # Button clicked but selections are incomplete or data failed loading
        if df_loaded is None and selected_commodity:
             st.warning(f"Could not proceed: Data loading failed for {selected_commodity}. Please check the file and console/terminal for errors.")
        elif not selected_commodity:
            st.warning("Please select a Commodity.")
        elif not selected_state:
             st.warning("Please select a State (or check data/configuration if it's missing).")
        elif not selected_district:
             st.warning("Please select a District (or check data/configuration if it's missing).")
        else:
             st.warning("Please ensure Commodity, State, and District are selected and data is available.")


# Display message if no data was loaded initially for the first selected commodity
# (This check is outside the 'run_forecast' block)
# if selected_commodity and df_loaded is None and not run_forecast:
#    st.warning(f"Data for {selected_commodity} could not be loaded. Please ensure the file '{COMMODITY_CONFIG[selected_commodity]['path']}' exists and is correctly formatted.")
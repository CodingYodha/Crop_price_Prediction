import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import datetime

# --- Configuration ---
DATA_PATH = "price_wheat_daily.csv"  # Use the correct path to your CSV
TARGET_COLUMNS = ['avg_min_price', 'avg_max_price', 'avg_modal_price']
DATE_COLUMN = 'date'
MIN_DATA_POINTS = 30 # Minimum data points required to train a model

# --- Data Loading and Caching ---
@st.cache_data # Cache data loading for performance
def load_data(path):
    """Loads and preprocesses the data."""
    try:
        df = pd.read_csv(path)
        # Basic Preprocessing
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
        df.dropna(subset=[DATE_COLUMN], inplace=True) # Drop rows where date conversion failed

        # Ensure price columns are numeric, coerce errors to NaN
        for col in TARGET_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Optional: Impute missing prices if needed (e.g., ffill)
        # df.sort_values(DATE_COLUMN, inplace=True)
        # df[TARGET_COLUMNS] = df.groupby(['state_name', 'district_name', 'commodity_name'])[TARGET_COLUMNS].ffill()

        df.dropna(subset=TARGET_COLUMNS, inplace=True) # Drop rows with missing target values after potential imputation
        df.sort_values(DATE_COLUMN, inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Modeling Function ---
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

# --- Plotting Function for a Single Target ---
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


# Load data
df_full = load_data(DATA_PATH)

if df_full is not None:
    # --- User Selections ---
    st.sidebar.header("Select Criteria")
    try:
        states = sorted(df_full['state_name'].unique())
        selected_state = st.sidebar.selectbox("Select State", states)

        districts = sorted(df_full[df_full['state_name'] == selected_state]['district_name'].unique())
        if not districts:
            st.sidebar.warning(f"No districts found for state '{selected_state}'.")
            selected_district = None
        else:
            selected_district = st.sidebar.selectbox("Select District", districts)

        if selected_district:
            commodities = sorted(df_full[(df_full['state_name'] == selected_state) &
                                         (df_full['district_name'] == selected_district)]['commodity_name'].unique())
            if not commodities:
                 st.sidebar.warning(f"No commodities found for '{selected_district}' in '{selected_state}'.")
                 selected_commodity = None
            else:
                selected_commodity = st.sidebar.selectbox("Select Commodity", commodities)
        else:
            selected_commodity = None

        forecast_days = st.sidebar.number_input("Select Forecast Period (Days)", min_value=7, max_value=365, value=30) # Default to 30 days

        run_forecast = st.sidebar.button("Run Forecast")

    except Exception as e:
        st.sidebar.error(f"Error setting up selection widgets: {e}")
        selected_state = selected_district = selected_commodity = None
        run_forecast = False

    # --- Filtering and Forecasting ---
    if run_forecast and selected_state and selected_district and selected_commodity:
        st.header(f"Forecast for {selected_commodity} in {selected_district}, {selected_state}")

        # Filter data based on selections
        filtered_df = df_full[
            (df_full['state_name'] == selected_state) &
            (df_full['district_name'] == selected_district) &
            (df_full['commodity_name'] == selected_commodity)
        ].copy() # Use copy to avoid SettingWithCopyWarning

        # Ensure data is sorted by date (important for plotting historical correctly)
        filtered_df.sort_values(by=DATE_COLUMN, inplace=True)

        if filtered_df.empty:
            st.warning("No historical data found for the selected combination.")
        else:
            last_hist_date = filtered_df[DATE_COLUMN].max().strftime('%Y-%m-%d')
            st.info(f"Found {len(filtered_df)} historical data points (latest: {last_hist_date}). Training on this data.")

            all_forecasts = {} # Store forecasts if needed later

            with st.spinner("Training models and generating forecasts..."):
                # Loop through each target price type
                for target in TARGET_COLUMNS:
                    st.subheader(f"Forecast: {target.replace('avg_', '').replace('_price', '').capitalize()} Price")

                    # Train model and get forecast starting from TODAY
                    model, forecast = train_and_forecast(filtered_df, target, forecast_days)

                    if forecast is not None:
                        all_forecasts[target] = forecast # Store the forecast

                        # Plot historical data and the specific forecast
                        plot_title = f'{target.replace("avg_", "").replace("_price", "").capitalize()} Price: Historical & Forecast ({forecast_days} days)'
                        fig = plot_single_forecast(filtered_df, forecast, target, plot_title)
                        st.plotly_chart(fig, use_container_width=True)

                        # Display forecast data table for this target (optional)
                        with st.expander(f"View Forecast Data Table for {target}"):
                            f_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                            f_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                            f_display['Date'] = f_display['Date'].dt.strftime('%Y-%m-%d') # Format date
                            st.dataframe(f_display.set_index('Date').style.format("{:.2f}"))
                    else:
                        # Message already displayed in train_and_forecast if skipped
                        pass


    elif run_forecast:
        st.warning("Please make sure State, District, and Commodity are selected.")

else:
    st.error("Failed to load data. Cannot start the application.")
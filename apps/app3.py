import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from prophet.plot import plot_plotly # Optional for Prophet's default plots

# --- Configuration ---
DATA_PATH = "price_wheat_daily.csv" # Use the correct path to your CSV
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
        # Consider basic imputation for missing prices if needed (e.g., ffill)
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
    """Trains a Prophet model and returns the forecast."""
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = data[[DATE_COLUMN, target_column]].rename(columns={DATE_COLUMN: 'ds', target_column: 'y'})

    # Check for sufficient data points
    if len(prophet_df) < MIN_DATA_POINTS:
        st.warning(f"Not enough historical data points ({len(prophet_df)}) for '{target_column}' in the selected group. Need at least {MIN_DATA_POINTS}. Skipping forecast.")
        return None, None # Return None for model and forecast

    try:
        # Instantiate and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False, # Adjust based on expected patterns (daily data might not have weekly patterns)
            daily_seasonality=False   # Adjust based on expected patterns
        )
        # Suppress Prophet's informational messages during fitting
        # See: https://github.com/facebook/prophet/issues/223
        # This approach might change based on Prophet versions.
        # Alternative: redirect stdout temporarily if needed.
        # from contextlib import redirect_stdout
        # import io
        # f = io.StringIO()
        # with redirect_stdout(f):
        #     model.fit(prophet_df)
        #fit_output = f.getvalue() # If you need to capture the output

        model.fit(prophet_df) # Fit the model

        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_periods)

        # Generate forecast
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        st.error(f"Error during forecasting for {target_column}: {e}")
        return None, None

# --- Plotting Function ---
def plot_forecast(historical_data, forecasts, target_columns, title):
    """Creates a Plotly figure for historical data and forecasts."""
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Colors for min, max, modal

    for i, target_col in enumerate(target_columns):
        if target_col in forecasts and forecasts[target_col] is not None:
            forecast_df = forecasts[target_col]
            hist_data_col = historical_data[[DATE_COLUMN, target_col]].dropna() # Ensure no NaNs in plotting data

            # Add historical data trace
            fig.add_trace(go.Scatter(
                x=hist_data_col[DATE_COLUMN],
                y=hist_data_col[target_col],
                mode='lines',
                name=f'Historical {target_col.replace("avg_", "").replace("_price", "")}',
                line=dict(color=colors[i % len(colors)])
            ))

            # Add forecast trace
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                mode='lines',
                name=f'Forecast {target_col.replace("avg_", "").replace("_price", "")}',
                line=dict(color=colors[i % len(colors)], dash='dash')
            ))

            # Add uncertainty interval (optional but recommended)
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_upper'],
                mode='lines', name=f'Upper Bound {target_col.replace("avg_", "").replace("_price", "")}',
                line=dict(width=0),
                showlegend=False # Don't show legend for bounds
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_lower'],
                mode='lines', name=f'Lower Bound {target_col.replace("avg_", "").replace("_price", "")}',
                line=dict(width=0),
                fillcolor=f'rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.2)', # Use color with opacity
                fill='tonexty', # Fill area between lower and upper bound traces
                showlegend=False
            ))
        else:
             st.info(f"Forecast for {target_col} was skipped due to insufficient data or errors.")


    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode="x unified" # Improved hover experience
    )
    return fig

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🌾 Crop Price Time Series Forecasting")

# Load data
df_full = load_data(DATA_PATH)

if df_full is not None:
    # --- User Selections ---
    st.sidebar.header("Select Criteria")
    
    # Get unique sorted lists for selections
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

        forecast_days = st.sidebar.number_input("Select Forecast Period (Days)", min_value=7, max_value=365, value=90)

        run_forecast = st.sidebar.button("Run Forecast")

    except Exception as e:
        st.sidebar.error(f"Error setting up selection widgets: {e}")
        selected_state = selected_district = selected_commodity = None
        run_forecast = False

    # --- Filtering and Forecasting ---
    if run_forecast and selected_state and selected_district and selected_commodity:
        st.subheader(f"Forecast for {selected_commodity} in {selected_district}, {selected_state}")

        # Filter data based on selections
        filtered_df = df_full[
            (df_full['state_name'] == selected_state) &
            (df_full['district_name'] == selected_district) &
            (df_full['commodity_name'] == selected_commodity)
        ].copy() # Use copy to avoid SettingWithCopyWarning

        if filtered_df.empty:
            st.warning("No historical data found for the selected combination.")
        else:
            st.write(f"Found {len(filtered_df)} historical data points.")
            
            forecasts = {}
            models = {} # Optional: store models if needed later

            with st.spinner("Training models and generating forecasts..."):
                for target in TARGET_COLUMNS:
                    model, forecast = train_and_forecast(filtered_df, target, forecast_days)
                    models[target] = model # Store model (optional)
                    forecasts[target] = forecast # Store forecast results


            # --- Display Results ---
            plot_title = f'{selected_commodity} Price Forecast - {selected_district}, {selected_state}'
            fig = plot_forecast(filtered_df, forecasts, TARGET_COLUMNS, plot_title)
            st.plotly_chart(fig, use_container_width=True)

            # Optionally display Prophet's default plots
            # st.subheader("Prophet Model Components (Example: Modal Price)")
            # if models.get('avg_modal_price') and forecasts.get('avg_modal_price') is not None:
            #     try:
            #         fig_prophet = plot_plotly(models['avg_modal_price'], forecasts['avg_modal_price'])
            #         st.plotly_chart(fig_prophet, use_container_width=True)
            #     except Exception as e:
            #         st.warning(f"Could not generate Prophet components plot: {e}")

            # Display forecast data table (optional)
            st.subheader("Forecast Data (Last 10 points)")
            display_forecasts = []
            for target, forecast_df in forecasts.items():
                if forecast_df is not None:
                    # Select relevant columns and rename for clarity
                    f_display = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).copy()
                    f_display.columns = ['Date', f'Forecast_{target}', f'Lower_{target}', f'Upper_{target}']
                    f_display.set_index('Date', inplace=True)
                    display_forecasts.append(f_display)
            
            if display_forecasts:
                # Concatenate the forecast dataframes side-by-side
                all_forecast_data = pd.concat(display_forecasts, axis=1)
                st.dataframe(all_forecast_data.style.format("{:.2f}")) # Format numbers

    elif run_forecast:
        st.warning("Please make sure State, District, and Commodity are selected.")

else:
    st.error("Failed to load data. Cannot start the application.")
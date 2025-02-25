import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")

# App title and description
st.title("Stock Price Prediction Dashboard")
st.markdown("""
This application uses machine learning models to predict stock prices.
Select a stock symbol, date range, and prediction parameters to get started.
""")

# Sidebar for inputs
st.sidebar.header("Parameters")

# Stock symbol input
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")

# Date range selection
today = datetime.today()
start_date = st.sidebar.date_input("Start Date", today - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", today-timedelta(days=1))

# Prediction parameters
prediction_days = st.sidebar.slider("Prediction Window (Days)", 7, 90, 30)
feature_days = st.sidebar.slider("Days to use for features", 7, 100, 60)

# Model selection
models_to_use = st.sidebar.multiselect(
    "Select Models for Prediction",
    ["Linear Regression", "Random Forest", "LSTM Neural Network"],
    ["Linear Regression", "LSTM Neural Network"]
)

# Function to load stock data
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        print(data)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the stock symbol.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to create features
def create_features(data, feature_days):
    df = data.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Return'].rolling(window=20).std()

    # Create lagged features
    for i in range(1, feature_days + 1):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
        if i % 5 == 0:  # Create some lagged returns to avoid too many features
            df[f'Return_lag_{i}'] = df['Return'].shift(i)

    df.dropna(inplace=True)
    return df

# Function to prepare data for ML models
def prepare_ml_data(df, target_days):
    # Define X and y
    X = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Return'], axis=1)

    # Create target variable (future price)
    y = df['Close'].shift(-target_days)

    # Remove NaN values
    X = X[:-target_days]
    y = y.dropna()

    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, X.columns

# Function to prepare data for LSTM
def prepare_lstm_data(data, lookback):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_data, y_data = [], []

    for i in range(lookback, len(scaled_data) - 1):
        x_data.append(scaled_data[i-lookback:i, 0])
        y_data.append(scaled_data[i, 0])

    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

    # Split data
    train_size = int(len(x_data) * 0.8)
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]

    return x_train, x_test, y_train, y_test, scaler

# Function to create LSTM model
def create_lstm_model(lookback):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train models
def train_models(data, feature_days, prediction_days, models_to_use):
    results = {}

    # Create features
    df_features = create_features(data, feature_days)

    # Prepare data for ML models
    X_train, X_test, y_train, y_test, feature_names = prepare_ml_data(df_features, prediction_days)

    if "Linear Regression" in models_to_use:
        with st.spinner("Training Linear Regression Model..."):
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_pred = lr_model.predict(X_test)

            results["Linear Regression"] = {
                "model": lr_model,
                "predictions": lr_pred,
                "actual": y_test,
                "mse": mean_squared_error(y_test, lr_pred),
                "mae": mean_absolute_error(y_test, lr_pred),
                "r2": r2_score(y_test, lr_pred),
                "feature_importance": pd.Series(lr_model.coef_, index=feature_names).abs().sort_values(ascending=False)
            }

    if "Random Forest" in models_to_use:
        with st.spinner("Training Random Forest Model..."):
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)

            results["Random Forest"] = {
                "model": rf_model,
                "predictions": rf_pred,
                "actual": y_test,
                "mse": mean_squared_error(y_test, rf_pred),
                "mae": mean_absolute_error(y_test, rf_pred),
                "r2": r2_score(y_test, rf_pred),
                "feature_importance": pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
            }

    if "LSTM Neural Network" in models_to_use:
        with st.spinner("Training LSTM Neural Network..."):
            lookback = 60  # Number of days to look back for LSTM
            x_train, x_test, y_train, y_test, scaler = prepare_lstm_data(data, lookback)

            lstm_model = create_lstm_model(lookback)
            lstm_model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)

            # Make predictions
            lstm_pred = lstm_model.predict(x_test)

            # Inverse transform the predictions
            lstm_pred_actual = scaler.inverse_transform(lstm_pred)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            results["LSTM Neural Network"] = {
                "model": lstm_model,
                "predictions": lstm_pred_actual.flatten(),
                "actual": y_test_actual.flatten(),
                "mse": mean_squared_error(y_test_actual, lstm_pred_actual),
                "mae": mean_absolute_error(y_test_actual, lstm_pred_actual),
                "r2": r2_score(y_test_actual, lstm_pred_actual),
                "scaler": scaler,
                "lookback": lookback
            }

    return results, df_features

# Function to make future predictions
def predict_future(models_results, data, feature_days, prediction_days):
    future_predictions = {}

    # Get the latest data for features
    latest_data = data.copy().iloc[-feature_days-1:]
    latest_df = create_features(latest_data, feature_days).iloc[-1:]

    # Extract features for traditional ML models
    if latest_df.empty:
        st.warning("Not enough data to create features for prediction")
        return {}

    features = latest_df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Return'], axis=1)

    # Make predictions with each model
    for model_name, results in models_results.items():
        if model_name in ["Linear Regression", "Random Forest"]:
            model = results["model"]
            future_predictions[model_name] = model.predict(features)[0]
        elif model_name == "LSTM Neural Network":
            model = results["model"]
            scaler = results["scaler"]
            lookback = results["lookback"]

            # Prepare data for LSTM prediction
            scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))
            x_input = scaled_data[-lookback:].reshape(1, lookback, 1)

            # Predict
            lstm_pred = model.predict(x_input)
            future_price = scaler.inverse_transform(lstm_pred)[0][0]
            future_predictions[model_name] = future_price

    return future_predictions

# Main app logic
try:
    # Load data
    data = load_data(stock_symbol, start_date, end_date)

    if data is not None:
        # Display stock info
        st.subheader("Stock Information")
        col1, col2, col3 = st.columns(3)

        # Get latest price and calculate some stats
        latest_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        percent_change = (price_change / data['Close'].iloc[-2]) * 100

        col1.metric("Latest Price", f"${latest_price:.2f}", f"{price_change:.2f} ({percent_change:.2f}%)")
        col2.metric("Average (30d)", f"${data['Close'].iloc[-30:].mean():.2f}")
        col3.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")

        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(data)

        # Plot historical data
        st.subheader("Historical Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        fig.update_layout(
            title=f"{stock_symbol} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Train models
        with st.spinner("Training models..."):
            models_results, featured_data = train_models(data, feature_days, prediction_days, models_to_use)

        # Display model results
        st.subheader("Model Performance")

        # Create a table to compare models
        if models_results:
            performance_df = pd.DataFrame({
                model_name: {
                    "MSE": results["mse"],
                    "MAE": results["mae"],
                    "R²": results["r2"]
                }
                for model_name, results in models_results.items()
            }).T

            st.dataframe(performance_df)

            # Show predictions vs actual
            st.subheader("Prediction Results")
            fig = go.Figure()

            # Add actual prices
            first_model = next(iter(models_results.values()))
            actual_dates = data.index[-len(first_model["actual"]):]

            fig.add_trace(go.Scatter(
                x=actual_dates,
                y=first_model["actual"],
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            ))

            # Add predictions for each model
            colors = ['blue', 'green', 'red', 'purple']
            for i, (model_name, results) in enumerate(models_results.items()):
                fig.add_trace(go.Scatter(
                    x=actual_dates,
                    y=results["predictions"],
                    mode='lines',
                    name=f'{model_name} Predictions',
                    line=dict(color=colors[i % len(colors)])
                ))

            fig.update_layout(
                title="Model Predictions vs Actual Prices",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show feature importance for applicable models
            if "Linear Regression" in models_results or "Random Forest" in models_results:
                st.subheader("Feature Importance")

                cols = st.columns(len([m for m in models_results if m in ["Linear Regression", "Random Forest"]]))

                col_idx = 0
                for model_name, results in models_results.items():
                    if model_name in ["Linear Regression", "Random Forest"]:
                        with cols[col_idx]:
                            st.write(f"**{model_name} Feature Importance**")

                            # Display top 10 features
                            fi = results["feature_importance"].head(10)
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                y=fi.index,
                                x=fi.values,
                                orientation='h'
                            ))
                            fig.update_layout(
                                height=400,
                                xaxis_title="Importance",
                                margin=dict(l=10, r=10, t=10, b=10)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            col_idx += 1

            # Future predictions
            st.subheader(f"Price Prediction for {prediction_days} Days Ahead")

            future_preds = predict_future(models_results, data, feature_days, prediction_days)

            if future_preds:
                # Format predictions
                current_price = data['Close'].iloc[-1]

                model_cols = st.columns(len(future_preds))
                for i, (model_name, prediction) in enumerate(future_preds.items()):
                    change = prediction - current_price
                    pct_change = (change / current_price) * 100

                    with model_cols[i]:
                        st.metric(
                            f"{model_name} Prediction",
                            f"${prediction:.2f}",
                            f"{change:.2f} ({pct_change:.2f}%)"
                        )
            else:
                st.warning("Could not generate future predictions with the current data")

        else:
            st.warning("No models were trained. Please select at least one model.")

    # Add additional info
    st.subheader("About This Dashboard")
    st.markdown("""
    This dashboard uses machine learning models to predict stock prices:

    1. **Linear Regression**: A simple model that establishes linear relationships between features and target.
    2. **Random Forest**: An ensemble method that builds multiple decision trees and merges their predictions.
    3. **LSTM Neural Network**: A recurrent neural network architecture that can capture temporal dependencies in time series data.

    **Disclaimer**: Stock price predictions are inherently uncertain. This tool is for educational purposes only and should not be used for financial decisions.
    """)

except Exception as e:
    st.error(f"An error occurred: {e}")
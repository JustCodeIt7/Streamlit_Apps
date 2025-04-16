import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb  # Added XGBoost import

# Page layout
st.set_page_config(layout="wide", page_title="Stock Market Prediction")

# Create a title for the dashboard
st.title("Stock Market Prediction App")


# Functions for data retrieval
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    return stock.info


def get_historical_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist


# Feature engineering functions
def create_features(data):
    data = data.copy()
    # Add technical indicators
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()
    # Price momentum
    data["Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Return"].rolling(window=20).std()
    # Date features
    data["day"] = data.index.dayofweek
    data["month"] = data.index.month
    data["year"] = data.index.year

    # Additional features for advanced models
    # Relative Strength Index (RSI)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    return data.dropna()


# Machine learning functions
def prepare_ml_data(data, target_col="Close", forecast_days=5):
    # Create target variable (future price)
    data["Target"] = data[target_col].shift(-forecast_days)
    data = data.dropna()

    # Features and target
    feature_cols = [
        "MA5",
        "MA20",
        "MA50",
        "Return",
        "Volatility",
        "day",
        "month",
        "year",
        "RSI",
        "MACD",
        "MACD_Signal",
    ]
    X = data[feature_cols]
    y = data["Target"]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def train_model(X_train, y_train, model_type):
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":  # Added XGBoost model
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:  # SVR
        model = SVR(kernel="rbf")

    model.fit(X_train, y_train)
    return model


def predict_future(model, last_data, scaler, feature_cols, days=30):
    future_dates = pd.date_range(
        start=last_data.index[-1] + timedelta(days=1), periods=days
    )
    future_dates = pd.DatetimeIndex(
        [date for date in future_dates if date.weekday() < 5]
    )  # Remove weekends

    predictions = []
    last_features = last_data[feature_cols].iloc[-1:].values
    current_prediction = last_data["Close"].iloc[-1]

    for i in range(len(future_dates)):
        # Update features based on previous predictions
        input_features = last_features.copy()
        # Update date features
        input_features[0, feature_cols.index("day")] = future_dates[i].dayofweek
        input_features[0, feature_cols.index("month")] = future_dates[i].month
        input_features[0, feature_cols.index("year")] = future_dates[i].year

        # Make prediction
        scaled_input = scaler.transform(input_features)
        prediction = model.predict(scaled_input)[0]
        predictions.append(prediction)

        # Update for next prediction
        if i < len(future_dates) - 1:
            # Update MAs with new prediction
            input_features[0, feature_cols.index("MA5")] = (
                input_features[0, feature_cols.index("MA5")] * 4 + prediction
            ) / 5
            input_features[0, feature_cols.index("MA20")] = (
                input_features[0, feature_cols.index("MA20")] * 19 + prediction
            ) / 20
            input_features[0, feature_cols.index("MA50")] = (
                input_features[0, feature_cols.index("MA50")] * 49 + prediction
            ) / 50

            # Update return and volatility
            new_return = (prediction - current_prediction) / current_prediction
            input_features[0, feature_cols.index("Return")] = new_return

            # Simple estimates for RSI and MACD updates (simplified approximation)
            # These are approximations as properly updating these would require more historical data
            if "RSI" in feature_cols:
                if prediction > current_prediction:
                    input_features[0, feature_cols.index("RSI")] = min(
                        70, input_features[0, feature_cols.index("RSI")] + 2
                    )
                else:
                    input_features[0, feature_cols.index("RSI")] = max(
                        30, input_features[0, feature_cols.index("RSI")] - 2
                    )

            if "MACD" in feature_cols and "MACD_Signal" in feature_cols:
                # Simple update based on trend
                macd_idx = feature_cols.index("MACD")
                macd_signal_idx = feature_cols.index("MACD_Signal")
                if prediction > current_prediction:
                    input_features[0, macd_idx] = input_features[0, macd_idx] * 1.01
                    input_features[0, macd_signal_idx] = (
                        input_features[0, macd_signal_idx] * 1.005
                    )
                else:
                    input_features[0, macd_idx] = input_features[0, macd_idx] * 0.99
                    input_features[0, macd_signal_idx] = (
                        input_features[0, macd_signal_idx] * 0.995
                    )

            # Save current values for next iteration
            last_features = input_features.copy()
            current_prediction = prediction

    return pd.DataFrame(
        {"Predicted_Close": predictions}, index=future_dates[: len(predictions)]
    )


# Additional function for model comparison
def compare_models(hist_data, feature_cols, forecast_days=5):
    results = {}
    models = ["Linear Regression", "Random Forest", "SVR", "XGBoost"]

    for model_type in models:
        feature_data = create_features(hist_data)
        X_train, X_test, y_train, y_test, scaler, _ = prepare_ml_data(
            feature_data, forecast_days=forecast_days
        )

        model = train_model(X_train, y_train, model_type)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[model_type] = {"MAE": mae, "RMSE": rmse, "MSE": mse, "R2": r2}

    return pd.DataFrame(results).T


# Sidebar controls
st.sidebar.header("Stock Analysis")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
period = st.sidebar.selectbox(
    "Historical Data Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3
)

# ML controls
st.sidebar.header("ML Prediction Settings")
model_type = st.sidebar.selectbox(
    "Select Model",
    options=["Linear Regression", "Random Forest", "SVR", "XGBoost"],
    index=3,  # Added XGBoost
)
forecast_days = st.sidebar.slider("Training Forecast Horizon (Days)", 1, 30, 5)
future_days = st.sidebar.slider("Future Prediction Days", 5, 60, 30)

# Option for model comparison
compare_models_option = st.sidebar.checkbox("Compare All Models", value=False)

if ticker:
    try:
        # Get data
        with st.spinner("Loading stock data..."):
            stock_info = get_stock_info(ticker)
            hist_data = get_historical_data(ticker, period=period)

        # Display stock info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${stock_info.get('currentPrice', 0):.2f}")
        with col2:
            st.metric("Market Cap", f"${stock_info.get('marketCap', 0):,.0f}")
        with col3:
            st.metric("Sector", stock_info.get("sector", "N/A"))

        # Historical chart
        st.subheader("Historical Price Chart")
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=hist_data.index,
                    open=hist_data["Open"],
                    high=hist_data["High"],
                    low=hist_data["Low"],
                    close=hist_data["Close"],
                    name="Price",
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

        # ML Section
        st.header("Machine Learning Price Prediction")

        # Model comparison section
        if compare_models_option:
            with st.spinner("Comparing all models... This may take a minute."):
                comparison_df = compare_models(hist_data, forecast_days=forecast_days)

                st.subheader("Model Performance Comparison")
                st.dataframe(
                    comparison_df.style.format(
                        {
                            "MAE": "${:.2f}",
                            "RMSE": "${:.2f}",
                            "MSE": "${:.2f}",
                            "R2": "{:.4f}",
                        }
                    )
                )

                # Bar chart comparison
                metrics = ["RMSE", "MAE", "R2"]
                for metric in metrics:
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=comparison_df.index,
                                y=comparison_df[metric],
                                text=comparison_df[metric].round(4),
                                textposition="auto",
                            )
                        ]
                    )
                    fig.update_layout(title=f"Model Comparison - {metric}")
                    st.plotly_chart(fig, use_container_width=True)

        with st.spinner("Training model and generating predictions..."):
            # Prepare data
            feature_data = create_features(hist_data)
            X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_ml_data(
                feature_data, forecast_days=forecast_days
            )

            # Train model
            model = train_model(X_train, y_train, model_type)

            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Predict future
            future_pred = predict_future(
                model, feature_data, scaler, feature_cols, days=future_days
            )

        # Model metrics
        st.subheader("Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"${mae:.2f}")
        col2.metric("RMSE", f"${rmse:.2f}")
        col3.metric("MSE", f"${mse:.2f}")
        col4.metric("R² Score", f"{r2:.4f}")

        # Test predictions visualization
        st.subheader("Test Predictions vs Actual")
        test_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        test_fig = go.Figure()
        test_fig.add_trace(
            go.Scatter(x=test_df.index, y=test_df["Actual"], name="Actual")
        )
        test_fig.add_trace(
            go.Scatter(x=test_df.index, y=test_df["Predicted"], name="Predicted")
        )
        st.plotly_chart(test_fig, use_container_width=True)

        # Future predictions
        st.subheader(f"Future Price Predictions ({future_days} days)")

        # Combine recent actual data with predictions
        last_30_days = hist_data["Close"][-30:]

        pred_fig = go.Figure()
        pred_fig.add_trace(
            go.Scatter(x=last_30_days.index, y=last_30_days, name="Historical Prices")
        )
        pred_fig.add_trace(
            go.Scatter(
                x=future_pred.index,
                y=future_pred["Predicted_Close"],
                name="Predictions",
                line=dict(dash="dash"),
            )
        )
        st.plotly_chart(pred_fig, use_container_width=True)

        # Show prediction table
        st.subheader("Prediction Values")
        st.dataframe(future_pred.style.format({"Predicted_Close": "${:.2f}"}))

        # Show feature importance for tree-based models
        if model_type in ["Random Forest", "XGBoost"]:
            st.subheader("Feature Importance")

            if model_type == "Random Forest":
                importance = model.feature_importances_
            else:  # XGBoost
                importance = model.feature_importances_

            importance_df = pd.DataFrame(
                {"Feature": feature_cols, "Importance": importance}
            ).sort_values("Importance", ascending=False)

            fig = go.Figure(
                go.Bar(
                    x=importance_df["Feature"],
                    y=importance_df["Importance"],
                    text=importance_df["Importance"].round(3),
                    textposition="auto",
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter a valid stock ticker symbol.")

# Disclaimer
st.markdown("---")
st.header("About this Application")
st.write(
    """
This application provides stock market analysis and machine learning-based price predictions.
The predictions are based on historical patterns and technical indicators.

The app uses four different algorithms:
- Linear Regression: A simple but effective linear approach
- Random Forest: An ensemble of decision trees for robust predictions
- Support Vector Regression (SVR): A kernel-based method for nonlinear relationships
- XGBoost: An advanced gradient boosting algorithm that often outperforms traditional methods
"""
)
st.warning(
    """
Disclaimer: Stock predictions are for educational purposes only. Past performance
is not indicative of future results. Invest at your own risk.
"""
)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Stock Algo Evaluator", layout="wide")
st.title("📈 Stock Trading Algorithm Evaluator")
st.caption(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")


# --- Helper Functions ---

@st.cache_data  # Cache data loading
def load_data(ticker_symbol, start_date, end_date):
    """Loads historical stock data using yfinance."""
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if data.empty:
            st.error(f"Could not download data for {ticker_symbol}. Check ticker or date range.")
            return None
        # Ensure datetime index
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker_symbol}: {e}")
        return None


def calculate_moving_averages(data, short_window, long_window):
    """Calculates short and long moving averages."""
    data_ma = data.copy()
    data_ma['Short_MA'] = data_ma['Close'].rolling(window=short_window, min_periods=1).mean()
    data_ma['Long_MA'] = data_ma['Close'].rolling(window=long_window, min_periods=1).mean()
    return data_ma


def simulate_ma_crossover_strategy(data, short_window, long_window, initial_capital):
    """Simulates a simple Moving Average Crossover strategy."""
    if short_window >= long_window:
        st.warning("Short MA window must be smaller than Long MA window.")
        return None, None  # Indicate error

    data_ma = calculate_moving_averages(data, short_window, long_window)
    data_ma = data_ma.dropna(subset=['Short_MA', 'Long_MA'])  # Remove initial NaNs for signal generation

    # Generate signals: 1 for Buy (short > long), -1 for Sell (short < long)
    data_ma['Signal'] = 0
    data_ma.loc[data_ma['Short_MA'] > data_ma['Long_MA'], 'Signal'] = 1
    data_ma.loc[data_ma['Short_MA'] < data_ma['Long_MA'], 'Signal'] = -1

    # Generate positions: 1 = Buy trigger, -1 = Sell trigger
    data_ma['Position'] = data_ma['Signal'].diff().fillna(0)
    data_ma.loc[data_ma['Position'] > 1, 'Position'] = 1  # Normalize if signal jumps from -1 to 1
    data_ma.loc[data_ma['Position'] < -1, 'Position'] = -1  # Normalize if signal jumps from 1 to -1

    # --- Backtesting Simulation ---
    portfolio = pd.DataFrame(index=data_ma.index)
    portfolio['Holdings'] = 0.0  # Value of stock held
    portfolio['Cash'] = float(initial_capital)
    portfolio['Total'] = float(initial_capital)
    portfolio['Action'] = ''  # 'Buy' or 'Sell'

    cash = float(initial_capital)
    shares = 0
    position_active = False  # Are we currently holding stock?

    buys = []
    sells = []

    for i in range(len(data_ma)):
        price = data_ma['Close'].iloc[i]
        signal = data_ma['Position'].iloc[i]
        current_index = data_ma.index[i]

        # --- Decision Logic ---
        if signal == 1 and not position_active and cash > 0:  # Buy Signal and not holding
            shares_to_buy = cash // price
            if shares_to_buy > 0:
                cash -= shares_to_buy * price
                shares += shares_to_buy
                position_active = True
                portfolio.loc[current_index, 'Action'] = 'Buy'
                buys.append(current_index)

        elif signal == -1 and position_active:  # Sell Signal and holding
            cash += shares * price
            shares = 0
            position_active = False
            portfolio.loc[current_index, 'Action'] = 'Sell'
            sells.append(current_index)

        # Update portfolio value daily
        portfolio.loc[current_index, 'Holdings'] = shares * price
        portfolio.loc[current_index, 'Cash'] = cash
        portfolio.loc[current_index, 'Total'] = cash + (shares * price)

    return data_ma, portfolio


def simulate_dummy_rl_strategy(data, initial_capital, buy_threshold=0.01, sell_threshold=-0.01):
    """Simulates a *very* basic rule-based agent (placeholder for RL)."""
    data_rl = data.copy()
    data_rl['Pct_Change'] = data_rl['Close'].pct_change().fillna(0)

    # Generate actions: 1 for Buy, -1 for Sell, 0 for Hold
    # This is a simple rule, a real RL agent learns a policy function
    data_rl['Action_RL'] = 0
    data_rl.loc[data_rl['Pct_Change'] > buy_threshold, 'Action_RL'] = 1
    data_rl.loc[data_rl['Pct_Change'] < sell_threshold, 'Action_RL'] = -1
    # Add some randomness for holding
    hold_mask = data_rl['Action_RL'] == 0
    data_rl.loc[hold_mask, 'Action_RL'] = np.random.choice([0, 1, -1], size=hold_mask.sum(), p=[0.85, 0.075, 0.075])

    # --- Backtesting Simulation (Similar to MA) ---
    portfolio = pd.DataFrame(index=data_rl.index)
    portfolio['Holdings'] = 0.0
    portfolio['Cash'] = float(initial_capital)
    portfolio['Total'] = float(initial_capital)
    portfolio['Action'] = ''  # 'Buy' or 'Sell'

    cash = float(initial_capital)
    shares = 0
    position_active = False

    buys = []
    sells = []

    for i in range(len(data_rl)):
        price = data_rl['Close'].iloc[i]
        action = data_rl['Action_RL'].iloc[i]  # RL Action
        current_index = data_rl.index[i]

        # --- Decision Logic (RL) ---
        if action == 1 and not position_active and cash > 0:  # Buy Action and not holding
            shares_to_buy = cash // price
            if shares_to_buy > 0:
                cash -= shares_to_buy * price
                shares += shares_to_buy
                position_active = True
                portfolio.loc[current_index, 'Action'] = 'Buy'
                buys.append(current_index)

        elif action == -1 and position_active:  # Sell Action and holding
            cash += shares * price
            shares = 0
            position_active = False
            portfolio.loc[current_index, 'Action'] = 'Sell'
            sells.append(current_index)

        # Action == 0 (Hold) -> Do nothing with shares/cash

        # Update portfolio value daily
        portfolio.loc[current_index, 'Holdings'] = shares * price
        portfolio.loc[current_index, 'Cash'] = cash
        portfolio.loc[current_index, 'Total'] = cash + (shares * price)

    return data_rl, portfolio


def plot_stock_data(data, ticker):
    """Plots candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name=ticker)])
    fig.update_layout(
        title=f'{ticker} Price Action',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        height=400
    )
    return fig


def plot_strategy(data, portfolio, buys, sells, title_suffix="", ma_short=None, ma_long=None):
    """Plots the stock price with buy/sell signals and optional MAs."""
    fig = go.Figure()

    # Price Line
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))

    # Moving Averages (Optional)
    if ma_short is not None and 'Short_MA' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['Short_MA'], mode='lines', name=f'MA {ma_short}',
                                 line=dict(color='orange', dash='dash')))
    if ma_long is not None and 'Long_MA' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['Long_MA'], mode='lines', name=f'MA {ma_long}',
                                 line=dict(color='purple', dash='dash')))

    # Buy Signals
    buy_points = data.loc[buys]
    fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points['Close'], mode='markers', name='Buy Signal',
                             marker=dict(color='green', size=10, symbol='triangle-up')))

    # Sell Signals
    sell_points = data.loc[sells]
    fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points['Close'], mode='markers', name='Sell Signal',
                             marker=dict(color='red', size=10, symbol='triangle-down')))

    fig.update_layout(
        title=f'Strategy Signals on Price Chart {title_suffix}',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=450,
        legend_title="Legend"
    )
    return fig


def plot_portfolio(portfolio, title_suffix=""):
    """Plots the portfolio value over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['Total'], mode='lines', name='Portfolio Value',
                             line=dict(color='green')))
    fig.update_layout(
        title=f'Portfolio Value Over Time {title_suffix}',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (USD)',
        height=350
    )
    return fig


def calculate_metrics(portfolio, initial_capital):
    """Calculates basic performance metrics."""
    final_value = portfolio['Total'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    num_trades = (portfolio['Action'] != '').sum()

    # Max Drawdown calculation
    portfolio['Peak'] = portfolio['Total'].cummax()
    portfolio['Drawdown'] = (portfolio['Total'] - portfolio['Peak']) / portfolio['Peak']
    max_drawdown = portfolio['Drawdown'].min() * 100  # Percentage

    # Simplified Sharpe Ratio (Annualized) - Assumes risk-free rate = 0
    # Needs daily returns for accurate calculation
    portfolio['Daily_Return'] = portfolio['Total'].pct_change().fillna(0)
    if portfolio['Daily_Return'].std() != 0:
        sharpe_ratio = (portfolio['Daily_Return'].mean() / portfolio['Daily_Return'].std()) * np.sqrt(
            252)  # Annualized (assuming 252 trading days)
    else:
        sharpe_ratio = 0.0

    return {
        "Final Portfolio Value": final_value,
        "Total Return (%)": total_return,
        "Number of Trades": num_trades,
        "Max Drawdown (%)": max_drawdown,
        "Annualized Sharpe Ratio (approx.)": sharpe_ratio
    }


# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
default_start = datetime(2020, 1, 1)
default_end = datetime.now()
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)

st.sidebar.subheader("Algorithm Selection")
algorithm = st.sidebar.selectbox(
    "Choose Algorithm to Evaluate",
    ["None", "Simple Moving Average Crossover (ML Example)", "Dummy Rule-Based Agent (RL Example)"]
)

# --- Algorithm Specific Parameters ---
params = {}
if algorithm == "Simple Moving Average Crossover (ML Example)":
    st.sidebar.markdown("**MA Crossover Parameters**")
    params['short_window'] = st.sidebar.slider("Short MA Window", 5, 50, 20)
    params['long_window'] = st.sidebar.slider("Long MA Window", 50, 250, 50)
elif algorithm == "Dummy Rule-Based Agent (RL Example)":
    st.sidebar.markdown("**Dummy RL Parameters**")
    params['buy_threshold'] = st.sidebar.slider("Buy Threshold (% change)", 0.0, 5.0, 1.0) / 100.0
    params['sell_threshold'] = st.sidebar.slider("Sell Threshold (% change)", -5.0, 0.0, -1.0) / 100.0
    st.sidebar.caption("Note: This RL agent uses simple rules, not actual learning.")

# --- Main Application Logic ---
if ticker:
    data = load_data(ticker, start_date, end_date)

    if data is not None:
        st.header(f"1. Historical Data: {ticker}")
        st.plotly_chart(plot_stock_data(data, ticker), use_container_width=True)

        # --- Evaluation Section ---
        if algorithm != "None":
            st.header(f"2. Evaluation: {algorithm}")
            st.markdown("---")

            portfolio_data = None
            strategy_data = None
            metrics = None
            fig_strategy = None
            fig_portfolio = None
            title_suffix = ""

            if algorithm == "Simple Moving Average Crossover (ML Example)":
                st.info("Simulating MA Crossover Strategy (Technical Indicator often used in ML baselines)...")
                title_suffix = "(MA Crossover)"
                strategy_data, portfolio_data = simulate_ma_crossover_strategy(
                    data, params['short_window'], params['long_window'], initial_capital
                )
                if strategy_data is not None and portfolio_data is not None:
                    buys = portfolio_data[portfolio_data['Action'] == 'Buy'].index
                    sells = portfolio_data[portfolio_data['Action'] == 'Sell'].index
                    fig_strategy = plot_strategy(
                        strategy_data, portfolio_data, buys, sells, title_suffix,
                        ma_short=params['short_window'], ma_long=params['long_window']
                    )

            elif algorithm == "Dummy Rule-Based Agent (RL Example)":
                st.warning("""
                 **Disclaimer:** This is a *highly simplified simulation* of an agent's actions based on fixed rules (price change thresholds + some randomness).
                 It does **not** represent a trained RL model. Real RL involves learning complex policies from interactions with an environment.
                 """)
                title_suffix = "(Dummy RL)"
                strategy_data, portfolio_data = simulate_dummy_rl_strategy(
                    data, initial_capital, params['buy_threshold'], params['sell_threshold']
                )
                if strategy_data is not None and portfolio_data is not None:
                    buys = portfolio_data[portfolio_data['Action'] == 'Buy'].index
                    sells = portfolio_data[portfolio_data['Action'] == 'Sell'].index
                    fig_strategy = plot_strategy(strategy_data, portfolio_data, buys, sells, title_suffix)

            # --- Display Results ---
            if strategy_data is not None and portfolio_data is not None:
                st.subheader("Strategy Performance")

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_strategy, use_container_width=True)
                with col2:
                    fig_portfolio = plot_portfolio(portfolio_data, title_suffix)
                    st.plotly_chart(fig_portfolio, use_container_width=True)

                st.subheader("Performance Metrics")
                metrics = calculate_metrics(portfolio_data, initial_capital)

                # Display metrics in columns
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Final Portfolio Value", f"${metrics['Final Portfolio Value']:,.2f}")
                m_col2.metric("Total Return", f"{metrics['Total Return (%)']:.2f}%")
                m_col3.metric("Number of Trades", f"{metrics['Number of Trades']}")

                m_col4, m_col5, m_col6 = st.columns(3)
                m_col4.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
                m_col5.metric("Annualized Sharpe Ratio (approx.)",
                              f"{metrics['Annualized Sharpe Ratio (approx.)']:.3f}")
                # Add another metric placeholder if needed in m_col6

                with st.expander("View Raw Portfolio Data"):
                    st.dataframe(portfolio_data.tail())

            elif algorithm != "None":  # Handle case where simulation failed (e.g., bad MA windows)
                st.error(f"Could not simulate {algorithm}. Check parameters.")

        else:
            st.info("Select an algorithm from the sidebar to start the evaluation.")
    else:
        st.warning("Failed to load data. Please check the ticker symbol and date range.")
else:
    st.info("Enter a stock ticker in the sidebar to begin.")

# --- Footer/Disclaimer ---
st.sidebar.markdown("---")
st.sidebar.warning("""
**Disclaimer:** This tool is for educational and demonstration purposes only.
Trading involves significant risk. The simulations here are highly simplified and do not account for real-world factors like slippage, fees, or market impact.
**Do not base investment decisions solely on this tool.** Consult a qualified financial advisor.
""")

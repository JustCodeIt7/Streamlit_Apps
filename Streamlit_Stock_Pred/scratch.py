#%%
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#%%
# Load a year of stock data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Example: Load Apple stock data
ticker = "AAPL"
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the data
print(stock_data.head())

# Plot the closing price
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'])
plt.title(f'{ticker} Stock Price (Past Year)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.tight_layout()
# %%

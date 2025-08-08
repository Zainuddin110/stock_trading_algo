# stock_trading_algo

# Algo Trading System with Yahoo Finance & Telegram Alerts

This project is a simple algorithmic trading system that:

- Fetches historical stock price data from Yahoo Finance (`yfinance`).
- Calculates key technical indicators: RSI, 20-day and 50-day moving averages.
- Generates buy and sell signals based on RSI and moving average crossovers.
- Backtests these signals to compute trades, profit/loss, and win ratio.
- Trains a basic Decision Tree model to predict next-day price movement.
- Logs trades to CSV files.
- Sends Telegram alerts for trade signals.
- Provides clear logging for progress and errors.

---

## Features

- Modular Python code with clear separation for data fetching, indicator calculation, strategy, ML, Telegram messaging, and logging.
- Supports multiple tickers in batch.
- Uses Telegram Bot API to send real-time alerts.
- Saves trade signals per ticker to CSV files.
- Handles exceptions and missing data gracefully.

---

## Requirements

- Python 3.7 or higher
- Required Python packages:


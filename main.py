!pip install pandas
!pip install yfinance
!pip install scikit-learn
!pip install telepot

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import telepot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
telegram_token = '8367885466:AAFEzRjBopWrRPx7ltEDoxCeR2AgDGQNvWk'
telegram_chat_id = '1076221731'

# Setup logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Data Ingestion ---
def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    Returns a dict: {ticker: DataFrame of 'Close' prices}.
    """
    logging.info("Fetching stock data...")
    data = yf.download(tickers, start=start_date, end=end_date)

    # Handle multi-ticker or single ticker case
    if isinstance(tickers, list) and len(tickers) > 1:
        close_data = {}
        for ticker in tickers:
            if ('Close', ticker) in data.columns:
                close_data[ticker] = data['Close'][ticker].to_frame(name='Close').dropna()
            else:
                logging.warning(f"Could not fetch 'Close' data for {ticker}. Skipping.")
        return close_data
    else:
        # Single ticker data
        key = tickers[0] if isinstance(tickers, list) else tickers
        if 'Close' in data.columns:
            return {key: data['Close'].to_frame(name='Close').dropna()}
        else:
            logging.warning(f"Could not fetch 'Close' data for {key}. Skipping.")
            return {}


# --- Indicator Calculations ---
def calculate_indicators(df):
    """
    Calculate RSI, 20-day MA, and 50-day MA for the stock DataFrame.
    Expects DataFrame with 'Close' column.
    Returns DataFrame with added indicator columns.
    """
    df = df.copy()
    if 'Close' not in df.columns:
        logging.error("Missing 'Close' column in data.")
        return None

    # Calculate moving averages
    df['20_DMA'] = df['Close'].rolling(window=20).mean()
    df['50_DMA'] = df['Close'].rolling(window=50).mean()

    # Calculate RSI manually
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


# --- Trading Strategy ---
def run_strategy(df):
    """
    Implement RSI < 30 + MA crossover buy signal,
    and close position by selling at the last day if still holding.
    Returns DataFrame of trade signals with date, signal, and price.
    """
    signals = []
    in_position = False

    if df is None or len(df) < 50 or not {'RSI', '20_DMA', '50_DMA'}.issubset(df.columns):
        logging.warning("Insufficient data or missing indicators to run the strategy.")
        return pd.DataFrame(signals)

    for i in range(1, len(df)):  # Start from 1 due to MA crossover check involving i-1
        # Buy condition
        if (df['RSI'].iloc[i] < 30 and
            df['20_DMA'].iloc[i] > df['50_DMA'].iloc[i] and
            df['20_DMA'].iloc[i-1] <= df['50_DMA'].iloc[i-1] and
            not in_position):
            signals.append({
                'date': df.index[i].strftime('%Y-%m-%d'),
                'signal': 'BUY',
                'price': df['Close'].iloc[i]
            })
            in_position = True

        # Sell condition - simplified to sell at last day if holding
        elif in_position and i == (len(df) - 1):
            signals.append({
                'date': df.index[i].strftime('%Y-%m-%d'),
                'signal': 'SELL',
                'price': df['Close'].iloc[i]
            })
            in_position = False

    return pd.DataFrame(signals)


# --- Performance Calculations ---
def calculate_pnl(trades_df):
    """
    Calculate total P&L and win ratio from trade signals DataFrame.
    """
    pnl = 0
    wins = 0
    losses = 0
    buy_price = None

    for _, trade in trades_df.iterrows():
        if trade['signal'] == 'BUY':
            buy_price = trade['price']
        elif trade['signal'] == 'SELL' and buy_price is not None:
            trade_pnl = trade['price'] - buy_price
            pnl += trade_pnl
            if trade_pnl > 0:
                wins += 1
            else:
                losses += 1

    total_trades = wins + losses
    win_ratio = (wins / total_trades) if total_trades > 0 else 0

    return pnl, win_ratio, wins, losses


# --- Machine Learning Automation ---
def train_and_predict_ml(df, ticker):
    """
    Train a Decision Tree classifier to predict next day’s positive return.
    Uses RSI, 20_DMA, 50_DMA, MACD, and Volume as features.
    """
    if df is None or len(df) < 51 or not {'RSI', '20_DMA', '50_DMA'}.issubset(df.columns):
        logging.warning(f"Insufficient data for ML on {ticker}.")
        return None

    df_ml = df.copy()
    df_ml['MACD'] = df_ml['Close'].ewm(span=12).mean() - df_ml['Close'].ewm(span=26).mean()

    # Fetch volume data aligned with df_ml index
    try:
        volume_data = yf.download(ticker, start=df_ml.index.min(), end=df_ml.index.max())['Volume']
        volume_data = volume_data[~volume_data.index.duplicated(keep='first')]
        df_ml = df_ml[~df_ml.index.duplicated(keep='first')]
        df_ml = df_ml.join(volume_data, how='left')
    except Exception as e:
        logging.warning(f"Failed to fetch volume for {ticker}: {e}")
        return None

    # Target variable: whether next day return is positive
    df_ml['Daily_Change'] = df_ml['Close'].pct_change().shift(-1)
    df_ml['Target'] = (df_ml['Daily_Change'] > 0).astype(int)

    features = ['RSI', '20_DMA', '50_DMA', 'MACD', 'Volume']
    df_ml.dropna(inplace=True)
    if len(df_ml) < 2:
        logging.warning(f"Not enough data after cleaning for ML on {ticker}.")
        return None

    X = df_ml[features]
    y = df_ml['Target']
    if len(X) < 2:
        logging.warning(f"Not enough data for train/test split on {ticker}.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy


# --- Logging to CSV ---
def save_to_csv(trade_log_df, ticker):
    """
    Save trade log DataFrame to CSV file.
    """
    try:
        filename = f"{ticker}_trade_log.csv"
        trade_log_df.to_csv(filename, index=False)
        logging.info(f"Trade log saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save to CSV: {e}")


# --- Telegram Alerts ---
def send_telegram_message(message):
    """
    Send a message using Telegram bot.
    """
    try:
        bot = telepot.Bot(telegram_token)
        bot.sendMessage(telegram_chat_id, message)
        logging.info("Telegram message sent.")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")


# --- Main Execution ---
def main_algo_component():
    """
    Main function to fetch data, apply strategy, do ML training,
    save logs and send alerts for multiple tickers.
    """
    logging.info("Starting algo-trading system...")

    tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year data

    all_data = fetch_stock_data(tickers, start_date, end_date)

    for ticker, df in all_data.items():
        try:
            indicators_df = calculate_indicators(df)
            if indicators_df is None:
                logging.warning(f"Skipping {ticker}: indicators not calculated.")
                continue

            trades_df = run_strategy(indicators_df)

            total_trades = 0
            win_ratio = 0.0
            total_pnl = 0.0

            if not trades_df.empty:
                total_pnl, win_ratio, wins, losses = calculate_pnl(trades_df)
                total_trades = len(trades_df[trades_df['signal'] == 'BUY'])

                save_to_csv(trades_df, ticker)
                
                last_signal = trades_df.iloc[-1]
                message = (f"New {last_signal['signal']} signal for {ticker} "
                           f"on {last_signal['date']} at price {last_signal['price']:.2f}")
                send_telegram_message(message)

            # Train ML model (result can be used or logged as desired)
            train_and_predict_ml(indicators_df, ticker)

            # Summary output
            print(f"📊 {ticker} Trading Summary")
            print(f"Total Trades: {total_trades}")
            print(f"Win Ratio: {win_ratio:.1%}")
            print(f"Total P&L: ₹{total_pnl:.2f}")
            print("-" * 20)

        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            send_telegram_message(f"Error processing {ticker}: {e}")
            print(f"📊 {ticker} Trading Summary")
            print(f"Error: {e}")
            print("-" * 20)

    logging.info("Algo-trading system completed.")


if __name__ == "__main__":
    main_algo_component()

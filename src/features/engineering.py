import pandas as pd
import numpy as np
import ta
import os

def load_and_merge_data(prices_path, macro_path, social_path):
    """
    Loads Prices, Macro, and Social data, merges them, and handles missing values.
    """
    print("Loading data...")
    
    # Load Prices
    # The CSV has a weird header structure:
    # Line 1: Price,Close,High,Low,Open,Volume
    # Line 2: Ticker,^NSEI,^NSEI,^NSEI,^NSEI,^NSEI
    # Line 3: Date,,,,,
    # Line 4: Data...
    # We read with header=0, then drop rows 0 and 1.
    prices_df = pd.read_csv(prices_path, header=0)
    prices_df = prices_df.iloc[2:].reset_index(drop=True)
    prices_df = prices_df.rename(columns={'Price': 'Date'})
    
    # Parse Date (DD-MM-YYYY)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'], dayfirst=True, utc=True).dt.tz_localize(None)
    prices_df = prices_df.sort_values('Date').set_index('Date')
    
    # Convert columns to numeric
    cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for c in cols:
        prices_df[c] = pd.to_numeric(prices_df[c], errors='coerce')
        
    print(f"Prices: {prices_df.shape}")
    
    # Load Macro
    if os.path.exists(macro_path):
        macro_df = pd.read_csv(macro_path)
        macro_df['Unnamed: 0'] = pd.to_datetime(macro_df['Unnamed: 0'])
        macro_df = macro_df.rename(columns={'Unnamed: 0': 'Date'}).set_index('Date')
        print(f"Macro: {macro_df.shape}")
    else:
        print("Macro data not found. Creating dummy.")
        macro_df = pd.DataFrame(index=prices_df.index)

    # Load Social
    if os.path.exists(social_path):
        social_df = pd.read_csv(social_path)
        social_df['Date'] = pd.to_datetime(social_df['Date'])
        social_df = social_df.set_index('Date')
        print(f"Social: {social_df.shape}")
    else:
        print("Social data not found. Creating dummy.")
        social_df = pd.DataFrame(index=prices_df.index)

    # Merge
    print("Merging data...")
    # Left join on Prices to keep trading days
    merged_df = prices_df.join(macro_df, how='left')
    merged_df = merged_df.join(social_df, how='left')
    
    # Forward fill Macro (since it might be daily including weekends, or have gaps)
    # Social data might be sparse, fill with 0 or mean?
    # For embeddings, 0 is reasonable (neutral/no info).
    # For sentiment, 0 (neutral).
    
    # Fill Macro
    cols_macro = [c for c in macro_df.columns if c in merged_df.columns]
    merged_df[cols_macro] = merged_df[cols_macro].ffill().bfill()
    
    # Fill Social
    cols_social = [c for c in social_df.columns if c in merged_df.columns]
    merged_df[cols_social] = merged_df[cols_social].fillna(0)
    
    return merged_df

def add_technical_indicators(df):
    """
    Adds Technical Indicators using 'ta' library.
    """
    print("Adding technical indicators...")
    
    # Ensure 'Close' is present
    if 'Close' not in df.columns:
        raise ValueError("Close column missing from data")
        
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Width'] = bollinger.bollinger_wband()
    
    # ATR (Requires High, Low, Close)
    if 'High' in df.columns and 'Low' in df.columns:
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # SMA / EMA
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    # Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (20-day rolling std of returns)
    df['Volatility'] = df['Log_Return'].rolling(window=20).std()
    
    # Target (Next Day Return or Direction)
    df['Target_Return'] = df['Log_Return'].shift(-1)
    df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
    
    return df

def process_features():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prices_path = os.path.join(base_dir, "Datasets", "nifty50_2015_to_2025.csv")
    macro_path = os.path.join(base_dir, "Datasets", "macro_data.csv")
    social_path = os.path.join(base_dir, "Datasets", "social_features.csv")
    output_path = os.path.join(base_dir, "Datasets", "processed_data.csv")
    
    df = load_and_merge_data(prices_path, macro_path, social_path)
    df = add_technical_indicators(df)
    
    # Drop initial NaNs from lookback periods (e.g. 200 days for SMA_200)
    df = df.dropna()
    
    print(f"Final processed data shape: {df.shape}")
    df.to_csv(output_path)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    process_features()

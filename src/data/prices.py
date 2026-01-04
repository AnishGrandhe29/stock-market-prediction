import pandas as pd
import os

def load_prices(filepath):
    """
    Loads and cleans NIFTY 50 price data.
    """
    print(f"Loading prices from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Ensure columns exist
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing columns. Expected {required_cols}, found {df.columns}")
        
    # Parse Date
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # Sort and drop duplicates
    df = df.sort_values('Date').drop_duplicates(subset=['Date'])
    
    # Set index
    df = df.set_index('Date')
    
    # Convert timezone if needed (keeping UTC for now to align with others)
    # df.index = df.index.tz_convert('Asia/Kolkata')
    
    print(f"Loaded {len(df)} price records.")
    return df

if __name__ == "__main__":
    # Test
    try:
        df = load_prices(os.path.join("Datasets", "nifty50_2015_to_2025.csv"))
        print(df.head())
    except Exception as e:
        print(e)

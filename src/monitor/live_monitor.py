import pandas as pd
import numpy as np
import yfinance as yf
import torch
import os
import sys
import ta

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.tcn import MultimodalTCN

class LiveDataFetcher:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path, index_col=0)
        self.df.index = pd.to_datetime(self.df.index)
        
    def get_live_window(self, window_size=30):
        # 1. Fetch Live NIFTY 50 Data
        ticker = "^NSEI"
        # Get last 5 days to ensure we have recent data and today's live candle
        live_data = yf.download(ticker, period="5d", interval="1d", progress=False)
        
        if live_data.empty:
            print("Warning: Could not fetch live data. Using historical data only.")
            return self.df.iloc[-window_size:]
            
        # 2. Update/Append to Historical Data
        # We need to ensure the columns match.
        # Historical columns: Open, High, Low, Close, Volume, etc.
        # We need to re-calculate technical indicators for the new/updated rows.
        
        # This is tricky because calculating TAs requires a lookback.
        # Strategy:
        # a. Take the last ~100 days of processed data (raw prices if possible, but we saved processed).
        #    Wait, 'processed_data.csv' has TAs already.
        #    We need the RAW prices to re-calc TAs for the live row.
        #    Let's load the raw prices file instead? 
        #    Or just assume we can append the live row and re-calc TAs for the last row?
        #    Re-calcing TAs like RSI requires previous values. 
        #    If we have the full history, we can just append and re-calc.
        
        # Let's load the RAW price data to be safe, or just use the columns from processed_data if they are raw.
        # processed_data has 'Close', 'Open', 'High', 'Low', 'Volume'.
        
        # Let's take the last 200 rows of processed_data, append the live row, re-calc indicators, then take the last 30.
        
        recent_history = self.df.iloc[-200:].copy()
        
        # Check if today is already in history
        last_date = recent_history.index[-1]
        today_date = pd.Timestamp.now().normalize()
        
        # yfinance returns data with timezone?
        live_data.index = live_data.index.tz_localize(None)
        
        # Get the latest row from live_data
        latest_row = live_data.iloc[-1]
        latest_date = live_data.index[-1]
        
        # Create a new row dataframe
        new_row = pd.DataFrame(index=[latest_date])
        new_row['Open'] = float(latest_row['Open'])
        new_row['High'] = float(latest_row['High'])
        new_row['Low'] = float(latest_row['Low'])
        new_row['Close'] = float(latest_row['Close'])
        new_row['Volume'] = float(latest_row['Volume'])
        
        # Fill other columns (Macro/Social) with the last known values (Forward Fill)
        for col in recent_history.columns:
            if col not in new_row.columns:
                new_row[col] = recent_history.iloc[-1][col]
                
        # Append or Update
        if latest_date == last_date:
            # Update last row
            recent_history.update(new_row)
        elif latest_date > last_date:
            # Append
            recent_history = pd.concat([recent_history, new_row])
            
        # 3. Re-calculate Technical Indicators for the window
        # We need to re-run the engineering logic for this small window.
        # Copy-paste logic from engineering.py or import it?
        # Importing is better but engineering.py is a script.
        # Let's just implement the necessary ones here using 'ta'.
        
        df_calc = recent_history.copy()
        
        # RSI
        df_calc['RSI'] = ta.momentum.rsi(df_calc['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df_calc['Close'])
        df_calc['MACD'] = macd.macd()
        df_calc['MACD_Signal'] = macd.macd_signal()
        df_calc['MACD_Diff'] = macd.macd_diff()
        
        # Bollinger
        bollinger = ta.volatility.BollingerBands(df_calc['Close'], window=20, window_dev=2)
        df_calc['BB_High'] = bollinger.bollinger_hband()
        df_calc['BB_Low'] = bollinger.bollinger_lband()
        df_calc['BB_Width'] = bollinger.bollinger_wband()
        
        # ATR
        df_calc['ATR'] = ta.volatility.average_true_range(df_calc['High'], df_calc['Low'], df_calc['Close'], window=14)
        
        # SMA/EMA
        df_calc['SMA_50'] = ta.trend.sma_indicator(df_calc['Close'], window=50)
        df_calc['SMA_200'] = ta.trend.sma_indicator(df_calc['Close'], window=200)
        df_calc['EMA_20'] = ta.trend.ema_indicator(df_calc['Close'], window=20)
        
        # Returns
        df_calc['Log_Return'] = np.log(df_calc['Close'] / df_calc['Close'].shift(1))
        df_calc['Volatility'] = df_calc['Log_Return'].rolling(window=20).std()
        
        # Clean up NaNs created by calc (though we started with 200 rows, so last 30 should be fine)
        # We just return the last window_size rows
        return df_calc.iloc[-window_size:]

class SignalGenerator:
    def __init__(self, model_path, price_cols, macro_cols, text_cols):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.price_cols = price_cols
        self.macro_cols = macro_cols
        self.text_cols = text_cols
        
        self.model = MultimodalTCN(
            price_input_size=len(price_cols),
            macro_input_size=len(macro_cols),
            text_input_size=len(text_cols),
            num_channels=[32, 32, 32],
            kernel_size=2,
            dropout=0.2
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def generate_signal(self, window_df):
        # Prepare inputs
        price_data = window_df[self.price_cols].values.astype('float32')
        macro_data = window_df[self.macro_cols].values.astype('float32')
        text_data = window_df[self.text_cols].values.astype('float32')
        
        # Tensorize [1, C, L]
        price_t = torch.tensor(price_data.T).unsqueeze(0).to(self.device)
        macro_t = torch.tensor(macro_data.T).unsqueeze(0).to(self.device)
        text_t = torch.tensor(text_data.T).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_preds, prob_preds = self.model(price_t, macro_t, text_t)
            
        prob = prob_preds.item()
        
        # Logic
        if prob > 0.6:
            signal = "BUY"
            confidence = "High"
        elif prob > 0.5:
            signal = "HOLD (Bias Up)"
            confidence = "Low"
        elif prob < 0.4:
            signal = "SELL"
            confidence = "High"
        else:
            signal = "HOLD (Bias Down)"
            confidence = "Low"
            
        return {
            "probability": prob,
            "signal": signal,
            "confidence": confidence,
            "current_price": window_df['Close'].iloc[-1],
            "last_updated": window_df.index[-1]
        }

def get_live_signal():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "Datasets", "processed_data.csv")
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    
    # Identify columns (Hardcoded for now to match training)
    # Ideally, we should save these metadata
    df = pd.read_csv(data_path, index_col=0)
    macro_cols = ['CPI', 'IIP', 'US10Y', 'USDINR', 'CrudeOil']
    text_cols = ['sentiment_mean', 'sentiment_count'] + [f'emb_{i}' for i in range(768)]
    target_cols = ['Target_Return', 'Target_Direction']
    exclude_cols = macro_cols + text_cols + target_cols + ['Date']
    price_cols = [c for c in df.columns if c not in exclude_cols]
    
    fetcher = LiveDataFetcher(data_path)
    window_df = fetcher.get_live_window()
    
    generator = SignalGenerator(checkpoint_path, price_cols, macro_cols, text_cols)
    result = generator.generate_signal(window_df)
    
    return result

if __name__ == "__main__":
    print(get_live_signal())

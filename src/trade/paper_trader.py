import pandas as pd
import requests
import time
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.train.online import online_train

def run_paper_trading_loop():
    print("Starting Paper Trading Loop...")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "Datasets", "processed_data.csv")
    trades_path = os.path.join(base_dir, "live_trades.csv")
    
    # Load Data
    df = pd.read_csv(data_path, index_col=0)
    
    # Simulation: We will iterate through the last 30 days of the dataset
    # treating each day as "today".
    simulation_days = 30
    dates = df.index[-simulation_days:]
    
    # Portfolio
    cash = 100000
    position = 0 # 0: Flat, 1: Long, -1: Short
    portfolio_history = []
    trades = []
    
    # API URL
    api_url = "http://127.0.0.1:8000"
    
    for i, date in enumerate(dates):
        print(f"\n--- Trading Day: {date} ---")
        
        # 1. Get Prediction
        try:
            response = requests.post(f"{api_url}/predict", json={"date": date})
            if response.status_code == 200:
                data = response.json()
                prob_up = data['probability_up']
                print(f"Prediction: P(Up) = {prob_up:.4f}")
                
                # 2. Execute Strategy
                current_price = df.loc[date, 'Close']
                action = "HOLD"
                
                if prob_up > 0.6 and position <= 0:
                    action = "BUY"
                    position = 1
                    cash -= current_price # Simplified: Buy 1 unit
                    trades.append({"Date": date, "Action": "BUY", "Price": current_price, "Prob": prob_up})
                    print(f"Action: BUY at {current_price}")
                    
                elif prob_up < 0.4 and position >= 0:
                    action = "SELL"
                    position = -1
                    cash += current_price # Simplified: Sell 1 unit (Short)
                    trades.append({"Date": date, "Action": "SELL", "Price": current_price, "Prob": prob_up})
                    print(f"Action: SELL at {current_price}")
                
                elif position == 1 and prob_up < 0.5:
                     action = "EXIT"
                     position = 0
                     cash += current_price
                     trades.append({"Date": date, "Action": "EXIT", "Price": current_price, "Prob": prob_up})
                     print(f"Action: EXIT LONG at {current_price}")

                elif position == -1 and prob_up > 0.5:
                     action = "EXIT"
                     position = 0
                     cash -= current_price
                     trades.append({"Date": date, "Action": "EXIT", "Price": current_price, "Prob": prob_up})
                     print(f"Action: EXIT SHORT at {current_price}")
                
                # Portfolio Value
                # Value = Cash + (Position * Current Price)
                # Note: Short position value is Cash - (Position * Current Price) ? No.
                # If Short (-1), we sold, so Cash increased. Liability is 1 unit.
                # Net Value = Cash - (1 * Current Price).
                # So Value = Cash + (Position * Current Price) works for Long (1) and Short (-1).
                
                port_value = cash + (position * current_price)
                portfolio_history.append({"Date": date, "Value": port_value})
                print(f"Portfolio Value: {port_value:.2f}")
                
            else:
                print(f"API Error: {response.text}")
                
        except Exception as e:
            print(f"Connection Error: {e}")
            continue
            
        # 3. Online Training (Simulated: Train on data up to THIS day)
        # In a real scenario, we'd wait for market close, get the true label (return), and train.
        # Here, we assume we just observed the close price and can train.
        # We take a window ending at 'date'.
        
        # Find index
        idx = df.index.get_loc(date)
        if idx > 60: # Ensure enough history
            # Train on last 60 days
            train_window = df.iloc[idx-60 : idx+1]
            print("Triggering online training...")
            online_train(train_window, epochs=1)
            
            # Reload API
            try:
                requests.post(f"{api_url}/reload")
                print("API Model Reloaded.")
            except:
                print("Failed to reload API.")

    # Save Results
    pd.DataFrame(trades).to_csv(trades_path, index=False)
    pd.DataFrame(portfolio_history).to_csv(os.path.join(base_dir, "portfolio_history.csv"), index=False)
    print("\nPaper Trading Simulation Complete.")

if __name__ == "__main__":
    run_paper_trading_loop()

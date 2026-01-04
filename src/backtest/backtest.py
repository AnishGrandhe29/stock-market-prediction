import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.tcn import MultimodalTCN
from src.train.train import TradingDataset
from torch.utils.data import DataLoader

def run_backtest():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "Datasets", "processed_data.csv")
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    results_path = os.path.join(base_dir, "backtest_results.csv")
    
    # Load Data
    print("Loading data for backtest...")
    df = pd.read_csv(data_path, index_col=0)
    df = df.dropna()
    
    # Identify columns (Same as train)
    macro_cols = ['CPI', 'IIP', 'US10Y', 'USDINR', 'CrudeOil']
    text_cols = ['sentiment_mean', 'sentiment_count'] + [f'emb_{i}' for i in range(768)]
    target_cols = ['Target_Return', 'Target_Direction']
    exclude_cols = macro_cols + text_cols + target_cols + ['Date']
    price_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Split (Test set)
    n = len(df)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    test_df = df.iloc[train_size + val_size :]
    
    # Dataset
    window_size = 30
    test_dataset = TradingDataset(test_df, price_cols, macro_cols, text_cols, window_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalTCN(
        price_input_size=len(price_cols),
        macro_input_size=len(macro_cols),
        text_input_size=len(text_cols),
        num_channels=[32, 32, 32],
        kernel_size=2,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Simulation
    print("Running simulation...")
    dates = test_df.index[window_size:]
    close_prices = test_df['Close'].values[window_size:]
    
    predictions = []
    probabilities = []
    actual_returns = []
    
    with torch.no_grad():
        for price, macro, text, target_ret, target_dir in test_loader:
            price, macro, text = price.to(device), macro.to(device), text.to(device)
            
            q_preds, prob_preds = model(price, macro, text)
            
            predictions.append(q_preds.cpu().numpy()) # Quantiles
            probabilities.append(prob_preds.item())
            actual_returns.append(target_ret.item())
            
    # Strategy
    # Buy if Prob > 0.6, Sell if Prob < 0.4
    # Position: 1 (Long), -1 (Short), 0 (Neutral)
    positions = []
    cash = 100000
    holdings = 0
    portfolio_values = []
    
    for i, prob in enumerate(probabilities):
        current_price = close_prices[i]
        
        if prob > 0.6:
            action = 1 # Long
        elif prob < 0.4:
            action = -1 # Short
        else:
            action = 0 # Neutral
            
        # Simple logic: 
        # If Long and no position: Buy
        # If Short and no position: Sell Short
        # If Neutral: Close position
        
        # This is simplified. Real backtest needs more state.
        # Let's assume we rebalance every day based on signal.
        
        # Daily Return Strategy
        # If signal is 1, we get the return of the asset.
        # If signal is -1, we get negative return of the asset.
        # If 0, we get 0.
        
        daily_ret = actual_returns[i]
        strategy_ret = action * daily_ret
        
        # Update portfolio
        portfolio_value = cash * (1 + strategy_ret) # Compounding
        # Wait, this is wrong. We need to track cumulative return.
        
        positions.append(action)
        
    # Calculate Cumulative Returns
    strategy_returns = np.array(positions) * np.array(actual_returns)
    cumulative_returns = (1 + strategy_returns).cumprod()
    market_returns = (1 + np.array(actual_returns)).cumprod()
    
    # Metrics
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
    
    # Save Results
    results_df = pd.DataFrame({
        'Date': dates,
        'Close': close_prices,
        'Probability': probabilities,
        'Position': positions,
        'Actual_Return': actual_returns,
        'Strategy_Return': strategy_returns,
        'Cumulative_Strategy': cumulative_returns,
        'Cumulative_Market': market_returns
    })
    
    results_df.to_csv(results_path)
    print(f"Backtest completed. Sharpe Ratio: {sharpe:.2f}")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    run_backtest()

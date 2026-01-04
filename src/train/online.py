import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.tcn import MultimodalTCN
from src.train.loss import CombinedLoss
from src.train.train import TradingDataset

def online_train(new_data_window_df, epochs=1, lr=0.0001):
    """
    Fine-tunes the model on the most recent window of data.
    """
    print("Starting online training...")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    
    # Identify columns (Same logic as train.py)
    macro_cols = ['CPI', 'IIP', 'US10Y', 'USDINR', 'CrudeOil']
    text_cols = ['sentiment_mean', 'sentiment_count'] + [f'emb_{i}' for i in range(768)]
    target_cols = ['Target_Return', 'Target_Direction']
    exclude_cols = macro_cols + text_cols + target_cols + ['Date']
    price_cols = [c for c in new_data_window_df.columns if c not in exclude_cols]
    
    # Dataset
    # We use a small window size for online training, or the same window size
    window_size = 30
    if len(new_data_window_df) <= window_size:
        print("Not enough data for online training.")
        return

    dataset = TradingDataset(new_data_window_df, price_cols, macro_cols, text_cols, window_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalTCN(
        price_input_size=len(price_cols),
        macro_input_size=len(macro_cols),
        text_input_size=len(text_cols),
        num_channels=[32, 32, 32],
        kernel_size=2,
        dropout=0.2
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for price, macro, text, target_ret, target_dir in loader:
            price, macro, text = price.to(device), macro.to(device), text.to(device)
            target_ret, target_dir = target_ret.to(device), target_dir.to(device)
            
            optimizer.zero_grad()
            q_preds, prob_preds = model(price, macro, text)
            
            loss = criterion(q_preds, prob_preds, target_ret, target_dir)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print(f"Online training complete. Loss: {total_loss:.4f}")
    
    # Save updated model
    torch.save(model.state_dict(), checkpoint_path)
    print("Updated model saved.")

if __name__ == "__main__":
    # Test run
    pass

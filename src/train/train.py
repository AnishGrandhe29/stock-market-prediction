import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.tcn import MultimodalTCN
from src.train.loss import CombinedLoss

class TradingDataset(Dataset):
    def __init__(self, data, price_cols, macro_cols, text_cols, window_size=30):
        self.data = data
        self.price_cols = price_cols
        self.macro_cols = macro_cols
        self.text_cols = text_cols
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Window: [idx, idx + window_size]
        window = self.data.iloc[idx : idx + self.window_size]
        
        # Inputs
        price_data = window[self.price_cols].values.astype(np.float32)
        macro_data = window[self.macro_cols].values.astype(np.float32)
        text_data = window[self.text_cols].values.astype(np.float32)
        
        # Targets (Next step after window)
        target_return = self.data.iloc[idx + self.window_size]['Target_Return']
        target_dir = self.data.iloc[idx + self.window_size]['Target_Direction']
        
        # Transpose for TCN [C, L]
        return (torch.tensor(price_data.T), 
                torch.tensor(macro_data.T), 
                torch.tensor(text_data.T), 
                torch.tensor(target_return, dtype=torch.float32), 
                torch.tensor(target_dir, dtype=torch.float32))

def train_model():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "Datasets", "processed_data.csv")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load Data
    print("Loading data...")
    df = pd.read_csv(data_path, index_col=0)
    df = df.dropna() # Ensure no NaNs
    
    # Identify columns
    macro_cols = ['CPI', 'IIP', 'US10Y', 'USDINR', 'CrudeOil']
    text_cols = ['sentiment_mean', 'sentiment_count'] + [f'emb_{i}' for i in range(768)]
    target_cols = ['Target_Return', 'Target_Direction']
    exclude_cols = macro_cols + text_cols + target_cols + ['Date'] # Date is index
    price_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Price features: {len(price_cols)}")
    print(f"Macro features: {len(macro_cols)}")
    print(f"Text features: {len(text_cols)}")
    
    # Split (Train 80%, Val 10%, Test 10%)
    # Time-series split
    n = len(df)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]
    
    # Datasets
    window_size = 30
    train_dataset = TradingDataset(train_df, price_cols, macro_cols, text_cols, window_size)
    val_dataset = TradingDataset(val_df, price_cols, macro_cols, text_cols, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False) # Shuffle=False for time series? Usually True for training windows is ok if independent, but False is safer for debugging.
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultimodalTCN(
        price_input_size=len(price_cols),
        macro_input_size=len(macro_cols),
        text_input_size=len(text_cols),
        num_channels=[32, 32, 32],
        kernel_size=2,
        dropout=0.2
    ).to(device)
    
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for price, macro, text, target_ret, target_dir in train_loader:
            price, macro, text = price.to(device), macro.to(device), text.to(device)
            target_ret, target_dir = target_ret.to(device), target_dir.to(device)
            
            optimizer.zero_grad()
            q_preds, prob_preds = model(price, macro, text)
            
            loss = criterion(q_preds, prob_preds, target_ret, target_dir)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for price, macro, text, target_ret, target_dir in val_loader:
                price, macro, text = price.to(device), macro.to(device), text.to(device)
                target_ret, target_dir = target_ret.to(device), target_dir.to(device)
                
                q_preds, prob_preds = model(price, macro, text)
                loss = criterion(q_preds, prob_preds, target_ret, target_dir)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print("Saved best model.")

if __name__ == "__main__":
    train_model()

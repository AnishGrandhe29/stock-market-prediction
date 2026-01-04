from fastapi import FastAPI, HTTPException
import pandas as pd
import torch
import os
import sys
from pydantic import BaseModel

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.tcn import MultimodalTCN

app = FastAPI(title="NIFTY 50 Trading System API")

# Load Model and Data Global
model = None
df = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window_size = 30

def load_resources():
    global model, df
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "Datasets", "processed_data.csv")
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    
    # Load Data
    df = pd.read_csv(data_path, index_col=0)
    df = df.dropna()
    
    # Identify columns
    macro_cols = ['CPI', 'IIP', 'US10Y', 'USDINR', 'CrudeOil']
    text_cols = ['sentiment_mean', 'sentiment_count'] + [f'emb_{i}' for i in range(768)]
    target_cols = ['Target_Return', 'Target_Direction']
    exclude_cols = macro_cols + text_cols + target_cols + ['Date']
    price_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Load Model
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
    
    return price_cols, macro_cols, text_cols

price_cols, macro_cols, text_cols = load_resources()

class PredictionRequest(BaseModel):
    date: str

@app.get("/")
def read_root():
    return {"message": "NIFTY 50 Trading System API is running"}

@app.post("/reload")
def reload_model():
    global model, df, price_cols, macro_cols, text_cols
    price_cols, macro_cols, text_cols = load_resources()
    return {"message": "Model and data reloaded successfully"}

@app.post("/predict")
def predict(request: PredictionRequest):
    date = request.date
    if date not in df.index:
        raise HTTPException(status_code=404, detail="Date not found in historical data")
    
    # Find index of date
    idx = df.index.get_loc(date)
    
    if idx < window_size:
        raise HTTPException(status_code=400, detail="Not enough historical data for this date (need 30 days prior)")
        
    # Get window
    window = df.iloc[idx - window_size : idx]
    
    # Prepare inputs
    price_data = window[price_cols].values.astype('float32')
    macro_data = window[macro_cols].values.astype('float32')
    text_data = window[text_cols].values.astype('float32')
    
    # Tensorize [1, C, L]
    price_t = torch.tensor(price_data.T).unsqueeze(0).to(device)
    macro_t = torch.tensor(macro_data.T).unsqueeze(0).to(device)
    text_t = torch.tensor(text_data.T).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_preds, prob_preds = model(price_t, macro_t, text_t)
        
    return {
        "date": date,
        "probability_up": prob_preds.item(),
        "quantiles": q_preds.cpu().numpy().tolist()[0]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

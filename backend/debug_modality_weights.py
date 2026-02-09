"""
Debug script to trace exactly what modality weights the model produces.
"""
import asyncio
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings
from app.models.stock import StockPrice, TechnicalIndicator


async def main():
    print("=" * 60)
    print("DEBUGGING MODEL MODALITY WEIGHTS")
    print("=" * 60)
    
    # 1. Check model file
    model_path = Path(settings.model_path)
    print(f"\nModel path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    if model_path.exists():
        print(f"Model size: {model_path.stat().st_size / 1024:.1f} KB")
    
    # 2. Load the model
    from app.ml.models.fusion import NIFTY50Predictor
    
    model = NIFTY50Predictor()
    
    if model_path.exists():
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print("[OK] Model loaded from file")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("Using random initialized model")
    else:
        print("[WARNING] Model file not found, using random weights")
    
    model.eval()
    
    # 3. Create dummy input to test
    print("\n--- Testing with dummy data ---")
    price_tensor = torch.randn(1, 60, 5)  # (batch, seq_len, features)
    sentiment_tensor = torch.randn(1, 3)  # (batch, features)
    technical_tensor = torch.randn(1, 15) # (batch, features)
    
    with torch.no_grad():
        output = model(price_tensor, sentiment_tensor, technical_tensor)
    
    print(f"\nModel output keys: {list(output.keys())}")
    
    if "modality_weights" in output:
        weights = output["modality_weights"]
        print(f"\nModality weights type: {type(weights)}")
        print(f"Modality weights value: {weights}")
        
        if isinstance(weights, dict):
            total = sum(weights.values())
            print(f"\nBreakdown (raw values):")
            for k, v in weights.items():
                pct = (v / total * 100) if total > 0 else 0
                print(f"  {k}: {v:.6f} ({pct:.1f}%)")
    else:
        print("[ERROR] No modality_weights in model output!")
    
    # 4. Now test with real data from DB
    print("\n--- Testing with real data ---")
    engine = create_async_engine(settings.database_url, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session_maker() as session:
        # Get latest prices
        result = await session.execute(
            select(StockPrice)
            .where(StockPrice.symbol == "^NSEI")
            .order_by(StockPrice.date.desc())
            .limit(60)
        )
        prices = result.scalars().all()[::-1]
        
        # Get latest technical indicators
        result = await session.execute(
            select(TechnicalIndicator)
            .where(TechnicalIndicator.symbol == "^NSEI")
            .order_by(TechnicalIndicator.date.desc())
            .limit(1)
        )
        tech = result.scalar_one_or_none()
        
        if prices and tech:
            # Build price tensor
            price_data = np.array([
                [p.open, p.high, p.low, p.close, p.volume or 0]
                for p in prices
            ])
            # Normalize
            price_mean = price_data.mean(axis=0)
            price_std = price_data.std(axis=0) + 1e-8
            price_normalized = (price_data - price_mean) / price_std
            price_tensor = torch.tensor(price_normalized, dtype=torch.float32).unsqueeze(0)
            
            # Build technical tensor
            technical_data = np.array([
                tech.rsi_14 or 50, tech.macd or 0, tech.macd_signal or 0,
                tech.macd_hist or 0, tech.stoch_k or 50, tech.stoch_d or 50,
                tech.ema_5 or 0, tech.ema_20 or 0, tech.ema_50 or 0,
                tech.sma_20 or 0, tech.adx or 25, tech.atr_14 or 0,
                tech.bb_upper or 0, tech.bb_middle or 0, tech.bb_lower or 0
            ]).reshape(1, -1)
            # Normalize
            tech_mean = technical_data.mean()
            tech_std = technical_data.std() + 1e-8
            tech_normalized = (technical_data - tech_mean) / tech_std
            technical_tensor = torch.tensor(tech_normalized, dtype=torch.float32)
            
            # Sentiment (dummy for now)
            sentiment_tensor = torch.tensor([[0.1, 0.05, 0.08]], dtype=torch.float32)
            
            with torch.no_grad():
                output = model(price_tensor, sentiment_tensor, technical_tensor)
            
            weights = output.get("modality_weights", {})
            print(f"\nWith real data - Modality weights:")
            total = sum(weights.values()) if weights else 0
            for k, v in weights.items():
                pct = (v / total * 100) if total > 0 else 0
                print(f"  {k}: {v:.6f} ({pct:.1f}%)")
        else:
            print("[ERROR] Could not load real data from DB")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

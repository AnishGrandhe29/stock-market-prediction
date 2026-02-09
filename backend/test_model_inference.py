"""
Test model inference with fresh load and real data.
"""
import torch
import numpy as np
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, '.')

from app.config import settings
from app.ml.models.fusion import NIFTY50Predictor
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.models.stock import StockPrice, TechnicalIndicator

async def main():
    print("=" * 60)
    print("TESTING MODEL INFERENCE")
    print("=" * 60)
    
    # 1. Load model fresh
    model = NIFTY50Predictor(
        price_seq_len=60, price_features=5, sentiment_features=3,
        technical_features=6, embedding_dim=128, dropout=0.2
    )
    
    model_path = Path(settings.model_path)
    print(f"Model path: {model_path.resolve()}")
    
    sd = torch.load(model_path, map_location='cpu', weights_only=True)
    result = model.load_state_dict(sd, strict=False)
    print(f"Missing keys: {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")
    
    model.eval()
    
    # 2. Get real data from DB
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
    
    # 3. Prepare tensors
    price_data = np.array([
        [p.open, p.high, p.low, p.close, p.volume or 0]
        for p in prices
    ])
    price_mean = price_data.mean(axis=0)
    price_std = price_data.std(axis=0) + 1e-8
    price_normalized = (price_data - price_mean) / price_std
    price_tensor = torch.tensor(price_normalized, dtype=torch.float32).unsqueeze(0)
    
    # Technical tensor (6 features)
    technical_data = np.array([
        tech.rsi_14 or 50,
        tech.macd or 0,
        tech.macd_signal or 0,
        tech.stoch_k or 50,
        tech.adx or 25,
        tech.atr_14 or 0,
    ]).reshape(1, -1)
    # Normalize
    tech_mean = np.array([50, 0, 0, 50, 25, 100])  # Approximate means
    tech_std = np.array([20, 50, 50, 20, 15, 100])  # Approximate stds
    tech_normalized = (technical_data - tech_mean) / tech_std
    technical_tensor = torch.tensor(tech_normalized, dtype=torch.float32)
    
    # Sentiment (mock)
    sentiment_tensor = torch.tensor([[0.1, 0.05, 0.08]], dtype=torch.float32)
    
    print(f"\nInput shapes:")
    print(f"  Price: {price_tensor.shape}")
    print(f"  Sentiment: {sentiment_tensor.shape}")
    print(f"  Technical: {technical_tensor.shape}")
    
    # 4. Run inference
    with torch.no_grad():
        output = model(price_tensor, sentiment_tensor, technical_tensor)
    
    print(f"\nModel output:")
    print(f"  Point prediction: {output['point_prediction'].item():.4f}")
    
    weights = output.get('modality_weights', {})
    print(f"\nModality weights:")
    for k, v in weights.items():
        print(f"  {k}: {v*100:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())

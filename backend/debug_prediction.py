
print("Initializing...", flush=True)
import asyncio
import sys
import os
from pathlib import Path

# Redirect stdout to file
sys.stdout = open('debug_result.txt', 'w', encoding='utf-8')

print("Imports 1...", flush=True)
import numpy as np
try:
    import torch
    print("Torch imported", flush=True)
except ImportError as e:
    print(f"Torch import failed: {e}", flush=True)

# Add backend to path (which contains the 'app' package)
# This assumes the script is located in backend/ and run from root
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))
print(f"Path modified: {sys.path[0]}", flush=True)

try:
    from app.config import settings
    print("Config imported", flush=True)
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    print("SQLAlchemy imported", flush=True)
    from app.services.ml_service import fetch_price_features, fetch_sentiment_features, fetch_technical_features, get_latest_close, load_model
    print("ML Service imported", flush=True)
except Exception as e:
    print(f"App imports failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

from app.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.services.ml_service import fetch_price_features, fetch_sentiment_features, fetch_technical_features, get_latest_close, load_model

async def debug_prediction():
    print("=" * 60)
    print("🔍 DEBUGGING PREDICTION INPUTS")
    print("=" * 60)
    
    engine = create_async_engine(settings.database_url, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session_maker() as session:
        symbol = "^NSEI"
        
        # 1. Check Latest Price
        latest_price = await get_latest_close(symbol, session)
        print(f"\n💰 Latest Close Price in DB: {latest_price}")
        
        # 2. Check Price Features
        price_data = await fetch_price_features(symbol, session)
        print(f"\n📉 Price Data (Last 5 steps):")
        print(price_data[-5:])
        print(f"Shape: {price_data.shape}")
        
        # 3. Check Technical Features
        tech_data = await fetch_technical_features(symbol, session)
        print(f"\n📊 Technical Features:")
        print(tech_data)
        
        # Check for Zeros
        if np.all(tech_data == 0):
            print("\n❌ CRITICAL: ALL TECHNICAL FEATURES ARE ZEROS!")
        elif tech_data[6] == 0: # EMA_5
            print(f"\n❌ CRITICAL: EMA_5 is ZERO! Price is {latest_price}")
            
        # 4. Check Sentiment
        sentiment_data = await fetch_sentiment_features(symbol, session)
        print(f"\n😊 Sentiment Features:")
        print(sentiment_data)
        
        # 5. Run Inference
        model = load_model()
        
        price_tensor = torch.tensor(price_data, dtype=torch.float32).unsqueeze(0)
        sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).unsqueeze(0)
        technical_tensor = torch.tensor(tech_data, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = model(price_tensor, sentiment_tensor, technical_tensor)
            
        print(f"\n🔮 Model Output:")
        print(f"Point Prediction (Change %): {output['point_prediction'].item():.4f}")
        pred_price = latest_price * (1 + output['point_prediction'].item() / 100)
        print(f"Predicted Price: {pred_price:.2f}")

    await engine.dispose()

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(debug_prediction())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error: {e}")

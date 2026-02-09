
import asyncio
import sys
import os
from pathlib import Path
from datetime import date, timedelta

# Add backend to path (which contains the 'app' package)
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from app.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, delete
from app.models.prediction import Prediction
from app.services.ml_service import get_prediction, get_next_trading_day

async def regenerate_prediction():
    print("=" * 60)
    print("🔄 REGENERATING NIFTY 50 PREDICTION")
    print("=" * 60)
    
    engine = create_async_engine(settings.database_url, echo=False)
    print(f"Connecting to database at: {settings.database_url}")
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session_maker() as session:
        symbol = "^NSEI"
        target_date = get_next_trading_day()
        
        print(f"Target Date: {target_date}")
        
        # 1. Delete existing prediction for today/target
        # We delete by target_date to ensure we replace the specific forecast
        print("🗑️  Deleting existing predictions for this target date...")
        await session.execute(
            delete(Prediction).where(
                Prediction.symbol == symbol,
                Prediction.target_date == target_date
            )
        )
        await session.commit()
        print("✅ Old predictions deleted.")
        
        # 2. Generate new prediction
        print("🔮 Generating new prediction...")
        try:
            prediction = await get_prediction(
                symbol=symbol,
                target_date=target_date,
                db=session
            )
            
            print("\n" + "=" * 60)
            print("✅ NEW PREDICTION GENERATED!")
            print("=" * 60)
            print(f"Predicted Close:  ₹{prediction.predicted_close:,.2f}")
            print(f"Change:           {prediction.predicted_change_pct:+.2f}%")
            print(f"Direction:        {prediction.predicted_direction.upper()}")
            print(f"Confidence:       {prediction.confidence_level.upper()}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error generating prediction: {e}")
            import traceback
            traceback.print_exc()

    await engine.dispose()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(regenerate_prediction())

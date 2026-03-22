import asyncio
import sys
from datetime import date
sys.path.insert(0, '.')

from app.core.database import async_session_maker
from app.services.ml_service import get_prediction

async def test_ml_service():
    print("Testing ML Service get_prediction...")
    try:
        async with async_session_maker() as session:
            prediction = await get_prediction("^NSEI", None, session)
            print("Successfully generated prediction!")
            print(f"Predicted Point: {prediction.predicted_change_pct}")
            print(f"Regime: {prediction.market_regime}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ml_service())

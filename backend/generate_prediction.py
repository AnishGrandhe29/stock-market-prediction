"""
Generate first prediction using the trained model.
Run this AFTER populate_real_data.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings
from app.services.ml_service import get_prediction


async def main():
    """Generate first prediction."""
    print("=" * 60)
    print("🤖 Generating First NIFTY 50 Prediction")
    print("=" * 60)
    
    engine = create_async_engine(settings.database_url, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session_maker() as session:
        try:
            print("\n📊 Generating prediction for ^NSEI...")
            print("This may take a few moments...")
            
            prediction = await get_prediction(
                symbol="^NSEI",
                target_date=None,  # Next trading day
                db=session
            )
            
            print("\n" + "=" * 60)
            print("✅ PREDICTION GENERATED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Symbol:           {prediction.symbol}")
            print(f"Target Date:      {prediction.target_date}")
            print(f"Predicted Open:   ₹{prediction.predicted_open:,.2f}")
            print(f"Change:           {prediction.predicted_change_pct:+.2f}%")
            print(f"Direction:        {prediction.predicted_direction.upper()}")
            print(f"Confidence:       {prediction.confidence_level.upper()}")
            print(f"Uncertainty:      {prediction.uncertainty_score:.2f}")
            print("=" * 60)
            print("\n✅ You can now view predictions in the frontend!")
            
        except Exception as e:
            print(f"\n❌ Error generating prediction: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

"""
Check the stored modality weights in the latest prediction.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings
from app.models.prediction import Prediction


async def main():
    print("=" * 60)
    print("CHECKING MODALITY WEIGHTS")
    print("=" * 60)
    
    engine = create_async_engine(settings.database_url, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session_maker() as session:
        result = await session.execute(
            select(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        pred = result.scalar_one_or_none()
        
        if not pred:
            print("[ERROR] No predictions found!")
            return
        
        print(f"Prediction ID:     {pred.id}")
        print(f"Target Date:       {pred.target_date}")
        print(f"Predicted Open:    {pred.predicted_open:.2f}")
        print(f"Direction:         {pred.predicted_direction}")
        print(f"Confidence:        {pred.confidence_level}")
        print()
        print("MODALITY WEIGHTS:")
        if pred.modality_weights:
            for key, val in pred.modality_weights.items():
                pct = val * 100 if isinstance(val, (int, float)) else val
                print(f"  {key}: {pct:.1f}%")
        else:
            print("  [NONE STORED]")
        
        print()
        print("TOP FEATURES:")
        if pred.top_features:
            for feat in pred.top_features[:5]:
                print(f"  - {feat}")
        else:
            print("  [NONE STORED]")
        
        print()
        print("SHAP VALUES (sample):")
        if pred.shap_values:
            # Print first few items
            count = 0
            for key, val in pred.shap_values.items():
                print(f"  {key}: {val}")
                count += 1
                if count >= 5:
                    break
        else:
            print("  [NONE STORED]")


if __name__ == "__main__":
    asyncio.run(main())

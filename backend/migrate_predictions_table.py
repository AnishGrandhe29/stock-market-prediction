"""
Migrate the predictions table from predicted_close to predicted_open.
This script drops and recreates the predictions table with the new column name.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings


async def main():
    print("=" * 60)
    print("MIGRATING PREDICTIONS TABLE")
    print("=" * 60)
    
    engine = create_async_engine(settings.database_url, echo=False)
    
    async with engine.begin() as conn:
        # Check if predictions table exists
        result = await conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        ))
        exists = result.fetchone()
        
        if exists:
            # Drop old predictions table
            print("[INFO] Dropping existing predictions table...")
            await conn.execute(text("DROP TABLE IF EXISTS predictions"))
            print("[OK] Old predictions table dropped.")
        
        # Recreate the predictions table with new schema
        print("[INFO] Creating new predictions table with predicted_open column...")
        await conn.execute(text("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT DEFAULT '^NSEI' NOT NULL,
                prediction_date DATE NOT NULL,
                target_date DATE NOT NULL,
                predicted_open FLOAT NOT NULL,
                predicted_change_pct FLOAT NOT NULL,
                quantile_5 FLOAT,
                quantile_50 FLOAT,
                quantile_95 FLOAT,
                uncertainty_score FLOAT,
                confidence_level TEXT,
                predicted_direction TEXT,
                direction_probability FLOAT,
                actual_close FLOAT,
                actual_change_pct FLOAT,
                prediction_error FLOAT,
                shap_values JSON,
                attention_weights JSON,
                modality_weights JSON,
                top_features JSON,
                input_features JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Create indexes
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(prediction_date)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_target_date ON predictions(target_date)"
        ))
        
        print("[OK] New predictions table created!")
        
    print("=" * 60)
    print("MIGRATION COMPLETE - Run regenerate_prediction.py next")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

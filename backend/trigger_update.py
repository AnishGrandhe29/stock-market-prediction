
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path (which contains the 'app' package)
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from app.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.services.indicators import compute_technical_indicators

async def trigger_update():
    print("=" * 60)
    print("🔄 TRIGGERING MANUAL INDICATOR UPDATE")
    print("=" * 60)
    
    engine = create_async_engine(settings.database_url, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session_maker() as session:
        symbol = "^NSEI"
        print(f"Calculating technical indicators for {symbol}...")
        
        try:
            indicators = await compute_technical_indicators(symbol, session)
            print(f"✅ Successfully computed {len(indicators)} indicator records.")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    await engine.dispose()
    print("\n✅ Update complete.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(trigger_update())

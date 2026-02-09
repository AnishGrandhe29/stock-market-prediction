import asyncio
import sys
from pathlib import Path
from datetime import date
from sqlalchemy import select, func

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.models.stock import StockPrice, TechnicalIndicator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

async def main():
    print("DIAGNOSTIC CHECK - DATA QUALITY")
    
    # 2. Check Database Data
    engine = create_async_engine(settings.database_url, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_maker() as session:
        # Check Stock Prices
        result = await session.execute(select(func.max(StockPrice.date)))
        latest_price_date = result.scalar()
        print(f"Latest Price Date: {latest_price_date}")

        if latest_price_date:
            days_diff = (date.today() - latest_price_date).days
            if days_diff > 5:
                print(f"[WARNING] Price data OLD: {days_diff} days")
            else:
                print("[OK] Price data recent.")

        # Check Technical Indicators
        result = await session.execute(select(func.max(TechnicalIndicator.date)))
        latest_tech_date = result.scalar()
        print(f"Latest Tech Date:  {latest_tech_date}")

        if latest_tech_date:
            if latest_price_date and latest_tech_date == latest_price_date:
                 print("[OK] Tech data matches price.")
            elif latest_price_date:
                 print(f"[WARNING] Tech lag: {(latest_price_date - latest_tech_date).days} days")
        else:
            print("[MISSING] No tech data found!")

if __name__ == "__main__":
    asyncio.run(main())

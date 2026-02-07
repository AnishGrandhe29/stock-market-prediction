"""
Background scheduler for periodic data fetching.
Reduces on-demand API calls by pre-populating cache.
"""
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import engine
from app.services.data_ingestion import fetch_stock_data, get_realtime_price
from app.core.redis import set_cache
import json


# Scheduler instance
scheduler = AsyncIOScheduler()


async def fetch_nifty_data_job():
    """Background job to fetch NIFTY data every 15 minutes."""
    print(f"\n🔄 [{datetime.now()}] Running scheduled NIFTY data fetch...")
    
    try:
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as session:
            # Fetch historical data (will use cache if recent)
            await fetch_stock_data("^NSEI", 60, session)
            
            # Fetch realtime price and update cache
            realtime = await get_realtime_price("^NSEI")
            
            print(f"✅ Scheduled fetch complete. Price: {realtime.get('price', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Scheduled fetch failed: {str(e)}")


async def fetch_constituents_data_job():
    """Background job to fetch top constituents data every 30 minutes."""
    print(f"\n🔄 [{datetime.now()}] Running scheduled constituents fetch...")
    
    constituents = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    ]
    
    try:
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as session:
            for symbol in constituents:
                try:
                    await fetch_stock_data(symbol, 60, session)
                    # Add delay to avoid rate limiting
                    await asyncio.sleep(5)
                except Exception as e:
                    print(f"⚠️ Failed to fetch {symbol}: {str(e)}")
                    continue
            
            print(f"✅ Constituents fetch complete")
            
    except Exception as e:
        print(f"❌ Constituents fetch failed: {str(e)}")


def start_scheduler():
    """Start the background scheduler."""
    
    # Fetch NIFTY data every 15 minutes
    scheduler.add_job(
        fetch_nifty_data_job,
        trigger=IntervalTrigger(minutes=15),
        id="fetch_nifty_data",
        name="Fetch NIFTY data",
        replace_existing=True
    )
    
    # Fetch constituents every 30 minutes
    scheduler.add_job(
        fetch_constituents_data_job,
        trigger=IntervalTrigger(minutes=30),
        id="fetch_constituents",
        name="Fetch constituents data",
        replace_existing=True
    )
    
    scheduler.start()
    print("🚀 Background scheduler started")
    print("📅 NIFTY data: every 15 minutes")
    print("📅 Constituents: every 30 minutes")


def stop_scheduler():
    """Stop the background scheduler."""
    scheduler.shutdown()
    print("⏹️ Background scheduler stopped")

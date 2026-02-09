"""
Background scheduler for periodic data fetching.
Reduces on-demand API calls by pre-populating cache.
"""
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import engine
from app.services.data_ingestion import fetch_stock_data, get_realtime_price
from app.core.redis import set_cache
import json


# Scheduler instance
scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")


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


async def daily_sentiment_collection_job():
    """Daily job to collect sentiment from all sources (4:30 PM IST after market close)."""
    print(f"\n📰 [{datetime.now()}] Running daily sentiment collection...")
    
    try:
        from app.services.sentiment_collector import collect_daily_sentiment
        
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as session:
            result = await collect_daily_sentiment("^NSEI", session)
            await session.commit()
            
            print(f"✅ Daily sentiment collected: {result}")
            
    except Exception as e:
        print(f"❌ Sentiment collection failed: {str(e)}")


async def daily_prediction_generation_job():
    """Daily job to generate next-day prediction (5:00 PM IST)."""
    print(f"\n🔮 [{datetime.now()}] Running daily prediction generation...")
    
    try:
        from app.services.ml_service import generate_prediction
        
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as session:
            prediction = await generate_prediction("^NSEI", session)
            
            if prediction:
                print(f"✅ Daily prediction generated:")
                print(f"   Predicted Open: ₹{prediction.predicted_open:,.2f}")
                print(f"   Direction: {prediction.predicted_direction}")
                print(f"   Confidence: {prediction.confidence_level}")
            else:
                print("⚠️ Prediction generation returned None")
            
    except Exception as e:
        print(f"❌ Prediction generation failed: {str(e)}")


async def daily_data_refresh_job():
    """Daily job to refresh all data (4:00 PM IST after market close)."""
    print(f"\n📊 [{datetime.now()}] Running daily data refresh...")
    
    try:
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as session:
            # Refresh NIFTY data with force
            await fetch_stock_data("^NSEI", 365, session, force_refresh=True)
            
            # Compute technical indicators
            from app.services.indicators import compute_technical_indicators
            await compute_technical_indicators("^NSEI", session)
            await session.commit()
            
            print(f"✅ Daily data refresh complete")
            
    except Exception as e:
        print(f"❌ Daily data refresh failed: {str(e)}")


def start_scheduler():
    """Start the background scheduler with all jobs."""
    
    # === INTRADAY JOBS (during market hours) ===
    
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
    
    # === DAILY JOBS (after market close at 3:30 PM IST) ===
    
    # 4:00 PM IST - Refresh all stock data and compute indicators
    scheduler.add_job(
        daily_data_refresh_job,
        trigger=CronTrigger(hour=16, minute=0),
        id="daily_data_refresh",
        name="Daily data refresh",
        replace_existing=True
    )
    
    # 4:30 PM IST - Collect sentiment from news sources
    scheduler.add_job(
        daily_sentiment_collection_job,
        trigger=CronTrigger(hour=16, minute=30),
        id="daily_sentiment",
        name="Daily sentiment collection",
        replace_existing=True
    )
    
    # 5:00 PM IST - Generate next-day prediction
    scheduler.add_job(
        daily_prediction_generation_job,
        trigger=CronTrigger(hour=17, minute=0),
        id="daily_prediction",
        name="Daily prediction generation",
        replace_existing=True
    )
    
    scheduler.start()
    print("🚀 Background scheduler started (Timezone: Asia/Kolkata)")
    print("📅 Intraday jobs:")
    print("   • NIFTY data: every 15 minutes")
    print("   • Constituents: every 30 minutes")
    print("📅 Daily jobs (after market close):")
    print("   • 4:00 PM - Data refresh & indicators")
    print("   • 4:30 PM - Sentiment collection")
    print("   • 5:00 PM - Prediction generation")


def stop_scheduler():
    """Stop the background scheduler."""
    scheduler.shutdown()
    print("⏹️ Background scheduler stopped")

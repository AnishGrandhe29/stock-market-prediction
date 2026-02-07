"""
Stock data API endpoints.
Provides OHLCV data, technical indicators, and sentiment scores.
"""
from typing import List, Optional
from datetime import date, datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.core.database import get_db
from app.models.stock import StockPrice, TechnicalIndicator, SentimentScore
from app.schemas import StockPriceResponse, TechnicalIndicatorResponse, SentimentResponse
from app.services.data_ingestion import fetch_stock_data, get_realtime_price

router = APIRouter()


# NIFTY 50 Symbol
NIFTY50_SYMBOL = "^NSEI"

# Top NIFTY 50 Constituents for watchlist
NIFTY50_CONSTITUENTS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "SBIN.NS", "BAJFINANCE.NS", "ITC.NS",
    "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
]


@router.get("/symbols")
async def get_available_symbols():
    """Get available stock symbols."""
    return {
        "index": NIFTY50_SYMBOL,
        "constituents": NIFTY50_CONSTITUENTS,
    }


@router.get("/realtime/{symbol}")
async def get_realtime_stock_price(symbol: str = NIFTY50_SYMBOL):
    """Get real-time price for a symbol (delayed by ~15 min for free API)."""
    try:
        data = await get_realtime_price(symbol)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch price: {str(e)}")


@router.get("/history/{symbol}", response_model=List[StockPriceResponse])
async def get_stock_history(
    symbol: str = NIFTY50_SYMBOL,
    days: int = Query(default=60, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """Get historical OHLCV data for a symbol."""
    start_date = date.today() - timedelta(days=days)
    
    result = await db.execute(
        select(StockPrice)
        .where(
            and_(
                StockPrice.symbol == symbol,
                StockPrice.date >= start_date
            )
        )
        .order_by(StockPrice.date.desc())
    )
    
    prices = result.scalars().all()
    
    # If no data in DB, fetch from source
    if not prices:
        try:
            prices = await fetch_stock_data(symbol, days, db)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")
    
    return prices


@router.get("/technical/{symbol}", response_model=List[TechnicalIndicatorResponse])
async def get_technical_indicators(
    symbol: str = NIFTY50_SYMBOL,
    days: int = Query(default=60, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """Get technical indicators for a symbol."""
    start_date = date.today() - timedelta(days=days)
    
    result = await db.execute(
        select(TechnicalIndicator)
        .where(
            and_(
                TechnicalIndicator.symbol == symbol,
                TechnicalIndicator.date >= start_date
            )
        )
        .order_by(TechnicalIndicator.date.desc())
    )
    
    return result.scalars().all()


@router.get("/sentiment/{symbol}", response_model=List[SentimentResponse])
async def get_sentiment_scores(
    symbol: str = NIFTY50_SYMBOL,
    days: int = Query(default=30, ge=1, le=90),
    db: AsyncSession = Depends(get_db)
):
    """Get sentiment scores for a symbol."""
    start_date = date.today() - timedelta(days=days)
    
    result = await db.execute(
        select(SentimentScore)
        .where(
            and_(
                SentimentScore.symbol == symbol,
                SentimentScore.date >= start_date
            )
        )
        .order_by(SentimentScore.date.desc())
    )
    
    return result.scalars().all()


@router.post("/refresh/{symbol}")
async def refresh_stock_data(
    symbol: str = NIFTY50_SYMBOL,
    db: AsyncSession = Depends(get_db)
):
    """Force refresh stock data from source."""
    try:
        prices = await fetch_stock_data(symbol, 60, db, force_refresh=True)
        return {"message": f"Refreshed {len(prices)} days of data for {symbol}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")


@router.get("/market-status")
async def get_market_status():
    """Get current market status (open/closed)."""
    now = datetime.now()
    
    # Indian market hours: 9:15 AM - 3:30 PM IST, Mon-Fri
    ist_now = now  # Assuming server is in IST
    
    is_weekday = ist_now.weekday() < 5
    market_open = ist_now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = ist_now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    is_open = is_weekday and market_open <= ist_now <= market_close
    
    return {
        "is_open": is_open,
        "current_time": ist_now.isoformat(),
        "market_open": "09:15 IST",
        "market_close": "15:30 IST",
        "next_open": None if is_open else "Next trading day 09:15 IST",
    }

"""
News API endpoints for market alerts.
"""
from typing import List, Optional
from fastapi import APIRouter, Query
from app.services.news_service import get_news_with_fallback

router = APIRouter()


@router.get("/market")
async def get_market_news(
    watchlist: Optional[str] = Query(None, description="Comma-separated list of watchlist symbols")
):
    """
    Get real-time market news affecting NIFTY 50.
    Optionally prioritize news for watchlist stocks.
    """
    # Parse watchlist
    watchlist_symbols = None
    if watchlist:
        watchlist_symbols = [s.strip() for s in watchlist.split(",") if s.strip()]
    
    news = await get_news_with_fallback(watchlist_symbols)
    
    return {
        "news": news,
        "total": len(news),
        "watchlistPrioritized": bool(watchlist_symbols)
    }


@router.get("/health")
async def news_health():
    """Check news service health."""
    return {"status": "ok", "service": "news"}

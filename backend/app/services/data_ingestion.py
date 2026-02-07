"""
Data ingestion service for stock data.
Fetches OHLCV data from multiple sources with fallback strategy.
Priority: NSE India -> Yahoo Finance -> Cached/Fallback
"""
import asyncio
import time
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.models.stock import StockPrice
from app.core.redis import get_cache, set_cache
import json


# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CACHE_TTL_REALTIME = 300  # 5 minutes for realtime data (increased from 2)
CACHE_TTL_HISTORICAL = 3600  # 1 hour for historical data
YAHOO_REQUEST_DELAY = 3  # 3 seconds between Yahoo requests
NSE_REQUEST_DELAY = 1  # 1 second between NSE requests

# Circuit breaker configuration
CIRCUIT_BREAKER = {
    "yahoo": {"failures": 0, "last_failure": None, "open_until": None},
    "nse": {"failures": 0, "last_failure": None, "open_until": None}
}
CIRCUIT_BREAKER_THRESHOLD = 3  # Open circuit after 3 failures
CIRCUIT_BREAKER_TIMEOUT = 300  # 5 minutes

# Rate limiting
LAST_REQUEST_TIME = {
    "yahoo": None,
    "nse": None
}


def is_circuit_open(source: str) -> bool:
    """Check if circuit breaker is open for a source."""
    breaker = CIRCUIT_BREAKER.get(source)
    if not breaker or breaker["open_until"] is None:
        return False
    
    if datetime.now() > breaker["open_until"]:
        # Reset circuit breaker
        breaker["failures"] = 0
        breaker["open_until"] = None
        print(f"🔄 Circuit breaker for {source} reset")
        return False
    
    return True


def record_failure(source: str):
    """Record a failure for circuit breaker."""
    breaker = CIRCUIT_BREAKER.get(source)
    if not breaker:
        return
    
    breaker["failures"] += 1
    breaker["last_failure"] = datetime.now()
    
    if breaker["failures"] >= CIRCUIT_BREAKER_THRESHOLD:
        breaker["open_until"] = datetime.now() + timedelta(seconds=CIRCUIT_BREAKER_TIMEOUT)
        print(f"⚠️ Circuit breaker opened for {source} until {breaker['open_until']}")


def record_success(source: str):
    """Record a success to reset failure count."""
    breaker = CIRCUIT_BREAKER.get(source)
    if breaker:
        breaker["failures"] = 0
        breaker["open_until"] = None


async def rate_limit_delay(source: str):
    """Apply rate limiting delay based on last request time."""
    global LAST_REQUEST_TIME
    
    last_time = LAST_REQUEST_TIME.get(source)
    delay = YAHOO_REQUEST_DELAY if source == "yahoo" else NSE_REQUEST_DELAY
    
    if last_time:
        elapsed = (datetime.now() - last_time).total_seconds()
        if elapsed < delay:
            wait_time = delay - elapsed
            print(f"⏳ Rate limiting: waiting {wait_time:.1f}s before {source} request")
            await asyncio.sleep(wait_time)
    
    LAST_REQUEST_TIME[source] = datetime.now()


async def fetch_from_nse(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch data from NSE India (primary source for Indian stocks).
    """
    if is_circuit_open("nse"):
        print("⚠️ NSE circuit breaker is open, skipping")
        return None
    
    try:
        await rate_limit_delay("nse")
        
        # Only works for NIFTY index
        if symbol != "^NSEI":
            return None
        
        from nsepython import index_history
        
        print(f"📊 Fetching {symbol} from NSE...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)
        
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: index_history(
                symbol="NIFTY 50",
                start_date=start_date.strftime("%d-%m-%Y"),
                end_date=end_date.strftime("%d-%m-%Y")
            )
        )
        
        if df is None or df.empty:
            record_failure("nse")
            return None
        
        # Rename columns
        df = df.rename(columns={
            'HistoricalDate': 'Date',
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOLUME': 'Volume'
        })
        
        # Parse dates
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y')
            except:
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        
        df = df.set_index('Date')
        df = df.sort_index()
        
        # Add Adj Close if not present
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        
        record_success("nse")
        print(f"✅ NSE: Fetched {len(df)} days for {symbol}")
        return df
        
    except Exception as e:
        print(f"❌ NSE fetch failed: {str(e)}")
        record_failure("nse")
        return None


async def fetch_from_yahoo(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch data from Yahoo Finance (backup source with rate limiting).
    """
    if is_circuit_open("yahoo"):
        print("⚠️ Yahoo Finance circuit breaker is open, skipping")
        return None
    
    try:
        await rate_limit_delay("yahoo")
        
        print(f"📊 Fetching {symbol} from Yahoo Finance...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)
        
        loop = asyncio.get_event_loop()
        
        # Use yfinance with retry
        for attempt in range(MAX_RETRIES):
            try:
                df = await loop.run_in_executor(
                    None,
                    lambda: yf.download(
                        symbol,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        progress=False
                    )
                )
                
                if not df.empty:
                    record_success("yahoo")
                    print(f"✅ Yahoo: Fetched {len(df)} days for {symbol}")
                    return df
                    
            except Exception as e:
                error_msg = str(e)
                
                # Check for rate limiting
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    print(f"⚠️ Yahoo rate limited (attempt {attempt + 1}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (2 ** attempt)
                        print(f"⏳ Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        record_failure("yahoo")
                        return None
                else:
                    raise e
        
        record_failure("yahoo")
        return None
        
    except Exception as e:
        print(f"❌ Yahoo fetch failed: {str(e)}")
        record_failure("yahoo")
        return None


async def fetch_stock_data(
    symbol: str,
    days: int,
    db: AsyncSession,
    force_refresh: bool = False
) -> List[StockPrice]:
    """
    Fetch historical OHLCV data using multi-source strategy.
    Priority: NSE India -> Yahoo Finance -> Cached data
    
    Args:
        symbol: Stock symbol (e.g., ^NSEI, RELIANCE.NS)
        days: Number of days of history to fetch
        db: Database session
        force_refresh: If True, delete existing data and refetch
    
    Returns:
        List of StockPrice objects
    """
    # Check cache first (unless forcing refresh)
    if not force_refresh:
        cache_key = f"stock_data:{symbol}:{days}"
        cached = await get_cache(cache_key)
        if cached:
            print(f"📦 Using cached data for {symbol}")
            return []  # Data already in DB
    
    # Delete existing data if refreshing
    if force_refresh:
        await db.execute(delete(StockPrice).where(StockPrice.symbol == symbol))
        await db.commit()
        print(f"🗑️ Cleared existing data for {symbol}")
    
    # Multi-source fetching strategy
    df = None
    
    # Try NSE first (for NIFTY index)
    if symbol == "^NSEI":
        df = await fetch_from_nse(symbol, days)
    
    # Fallback to Yahoo Finance
    if df is None or df.empty:
        df = await fetch_from_yahoo(symbol, days)
    
    # If still no data, return empty
    if df is None or df.empty:
        print(f"❌ Could not fetch data for {symbol} from any source")
        return []
    
    # Convert to StockPrice objects
    prices = []
    for idx, row in df.iterrows():
        price_date = idx.date() if hasattr(idx, 'date') else idx
        
        # Check if already exists
        existing = await db.execute(
            select(StockPrice).where(
                StockPrice.symbol == symbol,
                StockPrice.date == price_date
            )
        )
        if existing.scalar_one_or_none():
            continue
        
        price = StockPrice(
            symbol=symbol,
            date=price_date,
            open=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close']),
            volume=float(row['Volume']) if 'Volume' in row and pd.notna(row['Volume']) else 0,
            adj_close=float(row.get('Adj Close', row['Close'])),
        )
        db.add(price)
        prices.append(price)
    
    await db.commit()
    
    # Update cache
    await set_cache(f"stock_data:{symbol}:{days}", "1", expire=CACHE_TTL_HISTORICAL)
    
    print(f"✅ Stored {len(prices)} new price records for {symbol}")
    return prices


async def get_realtime_price_from_nse(symbol: str) -> Optional[Dict[str, Any]]:
    """Get realtime price from NSE."""
    if is_circuit_open("nse"):
        return None
    
    try:
        if symbol != "^NSEI":
            return None
        
        await rate_limit_delay("nse")
        
        from nsepython import nse_quote_ltp
        
        loop = asyncio.get_event_loop()
        price = await loop.run_in_executor(
            None,
            lambda: nse_quote_ltp("NIFTY 50")
        )
        
        if price:
            record_success("nse")
            return {
                "symbol": symbol,
                "price": float(price),
                "source": "nse"
            }
        
        record_failure("nse")
        return None
        
    except Exception as e:
        print(f"❌ NSE realtime fetch failed: {str(e)}")
        record_failure("nse")
        return None


async def get_realtime_price(symbol: str) -> dict:
    """
    Get real-time price for a symbol with multi-source fallback.
    Priority: Cache -> NSE -> Yahoo Finance -> Fallback
    """
    # Check cache (5 minute TTL to reduce API calls)
    cache_key = f"realtime:{symbol}"
    cached = await get_cache(cache_key)
    if cached:
        print(f"📦 Using cached realtime price for {symbol}")
        return json.loads(cached)
    
    # Try NSE first for NIFTY
    nse_data = await get_realtime_price_from_nse(symbol)
    if nse_data:
        # Get more details from database for previous close
        # For now, just use the price
        price_data = {
            "symbol": symbol,
            "price": nse_data["price"],
            "previous_close": nse_data["price"],  # Would need historical data
            "change": 0,
            "change_pct": 0,
            "high": nse_data["price"],
            "low": nse_data["price"],
            "open": nse_data["price"],
            "volume": 0,
            "timestamp": datetime.now().isoformat(),
            "source": "nse"
        }
        await set_cache(cache_key, json.dumps(price_data), expire=CACHE_TTL_REALTIME)
        return price_data
    
    # Fallback to Yahoo Finance with retry logic
    if is_circuit_open("yahoo"):
        print("⚠️ Yahoo circuit breaker open, using fallback data")
        return get_fallback_price_data(symbol)
    
    loop = asyncio.get_event_loop()
    
    for attempt in range(MAX_RETRIES):
        try:
            await rate_limit_delay("yahoo")
            
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            
            # Try fast_info first (fewer API calls)
            try:
                fast_info = await loop.run_in_executor(None, lambda: ticker.fast_info)
                current_price = getattr(fast_info, 'last_price', None)
                previous_close = getattr(fast_info, 'previous_close', None)
                
                if current_price:
                    price_data = {
                        "symbol": symbol,
                        "price": current_price,
                        "previous_close": previous_close or current_price,
                        "change": (current_price - previous_close) if previous_close else 0,
                        "change_pct": ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
                        "high": current_price,
                        "low": current_price,
                        "open": current_price,
                        "volume": 0,
                        "timestamp": datetime.now().isoformat(),
                        "source": "yahoo_fast"
                    }
                    
                    record_success("yahoo")
                    await set_cache(cache_key, json.dumps(price_data), expire=CACHE_TTL_REALTIME)
                    return price_data
                    
            except Exception:
                # Fallback to info (more API calls but more data)
                info = await loop.run_in_executor(None, lambda: ticker.info)
                
                current_price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
                previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
                
                if current_price:
                    price_data = {
                        "symbol": symbol,
                        "price": current_price,
                        "previous_close": previous_close or current_price,
                        "change": (current_price - previous_close) if previous_close else 0,
                        "change_pct": ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
                        "high": info.get('dayHigh') or info.get('regularMarketDayHigh') or current_price,
                        "low": info.get('dayLow') or info.get('regularMarketDayLow') or current_price,
                        "open": info.get('open') or info.get('regularMarketOpen') or current_price,
                        "volume": info.get('volume') or info.get('regularMarketVolume') or 0,
                        "timestamp": datetime.now().isoformat(),
                        "source": "yahoo_info"
                    }
                    
                    record_success("yahoo")
                    await set_cache(cache_key, json.dumps(price_data), expire=CACHE_TTL_REALTIME)
                    return price_data
                
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "Too Many Requests" in error_msg:
                print(f"⚠️ Yahoo rate limited (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(f"⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    record_failure("yahoo")
                    break
            else:
                print(f"❌ Yahoo realtime fetch failed: {error_msg}")
                record_failure("yahoo")
                break
    
    # All sources failed, return fallback
    print(f"⚠️ All sources failed for {symbol}, using fallback data")
    fallback_data = get_fallback_price_data(symbol)
    await set_cache(cache_key, json.dumps(fallback_data), expire=30)  # Short cache for fallback
    return fallback_data


def get_fallback_price_data(symbol: str) -> dict:
    """Return fallback price data when all sources fail."""
    return {
        "symbol": symbol,
        "price": 22000.0,  # Default NIFTY value
        "previous_close": 22000.0,
        "change": 0.0,
        "change_pct": 0.0,
        "high": 22000.0,
        "low": 22000.0,
        "open": 22000.0,
        "volume": 0,
        "timestamp": datetime.now().isoformat(),
        "source": "fallback",
        "error": "All data sources unavailable"
    }


async def fetch_nifty50_constituents(db: AsyncSession):
    """Fetch data for all NIFTY 50 constituent stocks."""
    constituents = [
        "^NSEI",  # NIFTY 50 Index
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "BHARTIARTL.NS", "SBIN.NS", "BAJFINANCE.NS", "ITC.NS",
    ]
    
    for symbol in constituents:
        try:
            print(f"\n📊 Processing {symbol}...")
            await fetch_stock_data(symbol, 365, db)
            # Add delay between constituents to avoid rate limiting
            await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            continue

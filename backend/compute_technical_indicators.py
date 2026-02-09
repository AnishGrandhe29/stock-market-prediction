"""
Compute and backfill technical indicators from existing price data.
Run this script to populate the technical_indicators table.
"""
import asyncio
import sys
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings
from app.models.stock import StockPrice, TechnicalIndicator


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD, Signal, and Histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    """Compute Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
    return stoch_k, stoch_d


def compute_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2):
    """Compute Bollinger Bands."""
    middle = prices.rolling(window=period, min_periods=1).mean()
    std = prices.rolling(window=period, min_periods=1).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (simplified)."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr = compute_atr(high, low, close, period)
    
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / (atr + 1e-10))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period, min_periods=1).mean()
    return adx


async def main():
    print("=" * 60)
    print("COMPUTING TECHNICAL INDICATORS")
    print("=" * 60)
    
    engine = create_async_engine(settings.database_url, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session_maker() as session:
        # Fetch all price data for NSEI
        result = await session.execute(
            select(StockPrice)
            .where(StockPrice.symbol == "^NSEI")
            .order_by(StockPrice.date.asc())
        )
        prices = result.scalars().all()
        
        if not prices:
            print("[ERROR] No price data found for ^NSEI")
            return
        
        print(f"[INFO] Found {len(prices)} price records")
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "date": p.date,
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume or 0,
            }
            for p in prices
        ])
        
        print("[INFO] Computing indicators...")
        
        # Compute all indicators
        df["rsi_14"] = compute_rsi(df["close"], 14)
        df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
        df["stoch_k"], df["stoch_d"] = compute_stochastic(df["high"], df["low"], df["close"])
        df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["adx"] = compute_adx(df["high"], df["low"], df["close"], 14)
        df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], 14)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = compute_bollinger_bands(df["close"])
        
        # Clear existing technical indicators for NSEI
        await session.execute(
            delete(TechnicalIndicator).where(TechnicalIndicator.symbol == "^NSEI")
        )
        
        # Insert new records
        count = 0
        for _, row in df.iterrows():
            ti = TechnicalIndicator(
                symbol="^NSEI",
                date=row["date"],
                rsi_14=float(row["rsi_14"]) if not pd.isna(row["rsi_14"]) else 50.0,
                macd=float(row["macd"]) if not pd.isna(row["macd"]) else 0.0,
                macd_signal=float(row["macd_signal"]) if not pd.isna(row["macd_signal"]) else 0.0,
                macd_hist=float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0.0,
                stoch_k=float(row["stoch_k"]) if not pd.isna(row["stoch_k"]) else 50.0,
                stoch_d=float(row["stoch_d"]) if not pd.isna(row["stoch_d"]) else 50.0,
                ema_5=float(row["ema_5"]) if not pd.isna(row["ema_5"]) else row["close"],
                ema_20=float(row["ema_20"]) if not pd.isna(row["ema_20"]) else row["close"],
                ema_50=float(row["ema_50"]) if not pd.isna(row["ema_50"]) else row["close"],
                sma_20=float(row["sma_20"]) if not pd.isna(row["sma_20"]) else row["close"],
                adx=float(row["adx"]) if not pd.isna(row["adx"]) else 25.0,
                atr_14=float(row["atr_14"]) if not pd.isna(row["atr_14"]) else 0.0,
                bb_upper=float(row["bb_upper"]) if not pd.isna(row["bb_upper"]) else row["close"],
                bb_middle=float(row["bb_middle"]) if not pd.isna(row["bb_middle"]) else row["close"],
                bb_lower=float(row["bb_lower"]) if not pd.isna(row["bb_lower"]) else row["close"],
            )
            session.add(ti)
            count += 1
        
        await session.commit()
        print(f"[OK] Inserted {count} technical indicator records")
        
        # Verify
        result = await session.execute(
            select(TechnicalIndicator)
            .where(TechnicalIndicator.symbol == "^NSEI")
            .order_by(TechnicalIndicator.date.desc())
            .limit(1)
        )
        latest = result.scalar_one_or_none()
        if latest:
            print(f"[OK] Latest tech date: {latest.date}")
            print(f"     RSI: {latest.rsi_14:.2f}, MACD: {latest.macd:.2f}")
        
        print("=" * 60)
        print("DONE - Technical indicators populated!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

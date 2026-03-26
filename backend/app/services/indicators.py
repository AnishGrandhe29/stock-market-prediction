"""
Technical indicators calculation service.
Computes real TA indicators using the `ta` library and persists them to DB.

Indicator groups:
  Momentum  : RSI(14/28), MACD, Stochastic (K/D)
  Trend     : EMA(5/20/50), SMA(20), ADX
  Volatility: ATR(14), Bollinger Bands (upper/middle/lower)
  Volume    : OBV, Volume SMA(20)
  GIFT Gap  : gap_abs, gap_pct  (stored for XAI display; sourced from
              gift_nifty_pipeline, NOT computed here)
"""
import logging
import asyncio
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.stock import StockPrice, TechnicalIndicator

logger = logging.getLogger(__name__)


def safe_float(value) -> Optional[float]:
    """Convert value to float, returning None for NaN / non-numeric."""
    if value is None:
        return None
    try:
        import math
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    except (ValueError, TypeError):
        return None


def _compute_ta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on a OHLCV DataFrame.

    Uses `ta` (preferred) with automatic fallback to `pandas-ta`.
    All indicators are causal — no future data used.

    Parameters
    ----------
    df : DataFrame with lowercase columns: open, high, low, close, volume.

    Returns
    -------
    df with additional indicator columns appended.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    try:
        from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
        from ta.momentum import RSIIndicator, StochasticOscillator
        from ta.volatility import BollingerBands, AverageTrueRange
        from ta.volume import OnBalanceVolumeIndicator

        # ── Momentum ──
        df["rsi_14"]     = RSIIndicator(close, window=14).rsi()
        df["rsi_28"]     = RSIIndicator(close, window=28).rsi()

        macd = MACD(close)
        df["macd"]       = macd.macd()
        df["macd_signal"]= macd.macd_signal()
        df["macd_hist"]  = macd.macd_diff()

        stoch = StochasticOscillator(high, low, close)
        df["stoch_k"]    = stoch.stoch()
        df["stoch_d"]    = stoch.stoch_signal()

        # ── Trend ──
        df["ema_5"]      = EMAIndicator(close, window=5).ema_indicator()
        df["ema_20"]     = EMAIndicator(close, window=20).ema_indicator()
        df["ema_50"]     = EMAIndicator(close, window=50).ema_indicator()
        df["sma_20"]     = SMAIndicator(close, window=20).sma_indicator()
        df["adx"]        = ADXIndicator(high, low, close).adx()

        # ── Volatility ──
        df["atr_14"]     = AverageTrueRange(high, low, close).average_true_range()
        bb = BollingerBands(close, window=20)
        df["bb_upper"]   = bb.bollinger_hband()
        df["bb_middle"]  = bb.bollinger_mavg()
        df["bb_lower"]   = bb.bollinger_lband()

        # ── Volume ──
        df["obv"]        = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df["volume_sma"] = SMAIndicator(volume, window=20).sma_indicator()

        logger.debug("Computed TA indicators using `ta` library.")

    except ImportError:
        logger.warning("`ta` library not installed; attempting `pandas_ta` fallback.")
        try:
            import pandas_ta as pta
            df["rsi_14"]     = pta.rsi(close, length=14)
            df["rsi_28"]     = pta.rsi(close, length=28)
            _macd = pta.macd(close)
            if _macd is not None:
                df["macd"]       = _macd.iloc[:, 0]
                df["macd_signal"]= _macd.iloc[:, 1]
                df["macd_hist"]  = _macd.iloc[:, 2]
            df["ema_5"]      = pta.ema(close, length=5)
            df["ema_20"]     = pta.ema(close, length=20)
            df["ema_50"]     = pta.ema(close, length=50)
            df["sma_20"]     = pta.sma(close, length=20)
            df["adx"]        = pta.adx(high, low, close).iloc[:, 0]
            df["atr_14"]     = pta.atr(high, low, close, length=14)
            _bb = pta.bbands(close)
            if _bb is not None:
                df["bb_upper"]  = _bb.iloc[:, 0]
                df["bb_middle"] = _bb.iloc[:, 2]
                df["bb_lower"]  = _bb.iloc[:, 4]
            df["obv"]        = pta.obv(close, volume)
            df["volume_sma"] = pta.sma(volume, length=20)
            logger.debug("Computed TA indicators using `pandas_ta` fallback.")
        except ImportError:
            logger.error(
                "No TA library available. Install: pip install ta  "
                "Falling back to neutral constants."
            )
            df["rsi_14"]     = 50.0
            df["rsi_28"]     = 50.0
            df["macd"]       = 0.0
            df["macd_signal"]= 0.0
            df["macd_hist"]  = 0.0
            df["stoch_k"]    = 50.0
            df["stoch_d"]    = 50.0
            df["ema_5"]      = close
            df["ema_20"]     = close
            df["ema_50"]     = close
            df["sma_20"]     = close
            df["adx"]        = 25.0
            df["atr_14"]     = (high - low)
            df["bb_upper"]   = close * 1.02
            df["bb_middle"]  = close
            df["bb_lower"]   = close * 0.98
            df["obv"]        = 0.0
            df["volume_sma"] = volume

    return df


async def compute_technical_indicators(
    symbol: str,
    db: AsyncSession,
    lookback_days: int = 100
) -> List[TechnicalIndicator]:
    """
    Compute technical indicators for a symbol and persist to the database.

    Parameters
    ----------
    symbol        : Ticker symbol, e.g. '^NSEI'.
    db            : Async SQLAlchemy session.
    lookback_days : How many trading days to process (default: last 100).

    Returns
    -------
    List of newly created TechnicalIndicator ORM objects.
    """
    start_date = date.today() - timedelta(days=lookback_days + 60)  # warmup buffer

    result = await db.execute(
        select(StockPrice)
        .where(
            and_(
                StockPrice.symbol == symbol,
                StockPrice.date >= start_date,
            )
        )
        .order_by(StockPrice.date)
    )
    prices = result.scalars().all()

    if len(prices) < 50:
        logger.warning(
            "Only %d price rows for %s – skipping indicator computation (need ≥50).",
            len(prices), symbol
        )
        return []

    # ── Build DataFrame ──
    df = pd.DataFrame([{
        "date"  : p.date,
        "open"  : p.open,
        "high"  : p.high,
        "low"   : p.low,
        "close" : p.close,
        "volume": float(p.volume or 0),
    } for p in prices])
    df.set_index("date", inplace=True)
    df = df.astype(float)

    # ── Compute indicators ──
    df = _compute_ta(df)

    # ── Persist last lookback_days rows ──
    indicators: List[TechnicalIndicator] = []
    recent_dates = df.index[-lookback_days:]

    for idx_date in recent_dates:
        row = df.loc[idx_date]

        existing = await db.execute(
            select(TechnicalIndicator).where(
                TechnicalIndicator.symbol == symbol,
                TechnicalIndicator.date   == idx_date,
            )
        )
        if existing.scalar_one_or_none():
            continue  # already computed

        indicator = TechnicalIndicator(
            symbol      = symbol,
            date        = idx_date,
            rsi_14      = safe_float(row.get("rsi_14")),
            macd        = safe_float(row.get("macd")),
            macd_signal = safe_float(row.get("macd_signal")),
            macd_hist   = safe_float(row.get("macd_hist")),
            stoch_k     = safe_float(row.get("stoch_k")),
            stoch_d     = safe_float(row.get("stoch_d")),
            ema_5       = safe_float(row.get("ema_5")),
            ema_20      = safe_float(row.get("ema_20")),
            ema_50      = safe_float(row.get("ema_50")),
            sma_20      = safe_float(row.get("sma_20")),
            adx         = safe_float(row.get("adx")),
            atr_14      = safe_float(row.get("atr_14")),
            bb_upper    = safe_float(row.get("bb_upper")),
            bb_middle   = safe_float(row.get("bb_middle")),
            bb_lower    = safe_float(row.get("bb_lower")),
            obv         = safe_float(row.get("obv")),
            volume_sma  = safe_float(row.get("volume_sma")),
        )
        db.add(indicator)
        indicators.append(indicator)

    if indicators:
        await db.commit()
        logger.info("Stored %d indicator rows for %s.", len(indicators), symbol)
    else:
        logger.debug("No new indicator rows needed for %s.", symbol)

    return indicators


async def get_latest_indicators(symbol: str, db: AsyncSession) -> dict:
    """
    Return the most recent technical indicator snapshot as a dict.
    Used by the prediction service to build the technical feature vector.
    """
    result = await db.execute(
        select(TechnicalIndicator)
        .where(TechnicalIndicator.symbol == symbol)
        .order_by(TechnicalIndicator.date.desc())
        .limit(1)
    )
    indicator = result.scalar_one_or_none()

    if not indicator:
        logger.warning("No technical indicators in DB for %s.", symbol)
        return {}

    return {
        "date"        : indicator.date,
        "rsi_14"      : indicator.rsi_14,
        "rsi_28"      : getattr(indicator, "rsi_28", None),
        "macd"        : indicator.macd,
        "macd_signal" : indicator.macd_signal,
        "macd_hist"   : indicator.macd_hist,
        "stoch_k"     : indicator.stoch_k,
        "stoch_d"     : indicator.stoch_d,
        "ema_5"       : indicator.ema_5,
        "ema_20"      : indicator.ema_20,
        "ema_50"      : indicator.ema_50,
        "sma_20"      : indicator.sma_20,
        "adx"         : indicator.adx,
        "atr_14"      : indicator.atr_14,
        "bb_upper"    : indicator.bb_upper,
        "bb_middle"   : indicator.bb_middle,
        "bb_lower"    : indicator.bb_lower,
        "obv"         : indicator.obv,
        "volume_sma"  : indicator.volume_sma,
    }

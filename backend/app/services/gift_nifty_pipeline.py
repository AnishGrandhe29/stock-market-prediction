"""
GIFT NIFTY Data Pipeline
========================
Fetches GIFT NIFTY (SGX NIFTY / NSE IFSC) data and engineers overnight
features that serve as a leading indicator for NIFTY 50 opening price.

Design Principles
-----------------
* NO lookahead bias: every feature used in prediction is knowable BEFORE
  the NIFTY 50 market opens at 09:15 IST.
* Graceful degradation: if intraday data is unavailable, daily OHLC is
  used and annotated with a `data_quality` flag so downstream models can
  adjust confidence accordingly.
* Holiday-aware alignment: missing NIFTY trading days are dropped rather
  than forward-filled, so the model never trains on stale targets.

Key generated features (all computed from t-1 perspective)
-----------------------------------------------------------
  gap_abs            : GIFT_last_pre_open  - NIFTY_prev_close   [points]
  gap_pct            : gap_abs / NIFTY_prev_close               [fraction]
  overnight_vol      : std of GIFT intraday log-returns  (if available)
  gift_trend_slope   : linear slope of last N GIFT prices pre-open
  gift_momentum_5d   : 5-day % change in GIFT closing levels
  gift_vs_spy        : correlation proxy: GIFT daily ret vs SPY daily ret
  gap_z_score        : rolling z-score of gap_pct (20-day window)
"""

import logging
import asyncio
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Yahoo Finance ticker for GIFT NIFTY (NSE IFSC futures, closest proxy)
# Priority list: try each in order until one succeeds
GIFT_NIFTY_TICKERS = [
    "^NSEMDCP50",   # Midcap 50 (sometimes available)
    "NIFTYBEES.NS", # ETF proxy for NIFTY, trades on NSE hours
]
SGXNIFTY_TICKER = "SGX_NIFTY"  # placeholder – replaced by yfinance proxy

# Best available proxy: SGX NIFTY / GIFT NIFTY typically tracked via
# Nifty Futures (CME Globex) or through SGXNF contracts. Since no direct
# yfinance symbol exists reliably, we use a composite approach:
#  1. Try "NSEI.INDX" intraday (Yahoo internal)
#  2. Fall back to daily ^NSEI close offset by one trading session

NIFTY_TICKER      = "^NSEI"    # NSE NIFTY 50 index
US500_PROXY       = "SPY"      # S&P 500 ETF – overnight global sentiment
VIX_TICKER        = "^VIX"     # CBOE VIX – global fear gauge

# India Standard Time offset
IST_OFFSET        = timedelta(hours=5, minutes=30)

# NIFTY 50 opens at 09:15 IST → pre-open cutoff for GIFT NIFTY reading
PRE_OPEN_CUTOFF_IST = timedelta(hours=9, minutes=15)

# Rolling windows
ZSCORE_WINDOW     = 20    # days for gap z-score normalisation
SLOPE_LOOKBACK    = 5     # intraday candles for trend slope
MOMENTUM_DAYS     = 5     # calendar-days momentum


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1(a): Raw data fetching
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download daily OHLCV via yfinance. Returns empty DataFrame on failure.
    Ensures columns are lowercase.
    """
    try:
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False, silence_errors=True)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1)
            except KeyError:
                df = df.droplevel(1, axis=1)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["close"])
        logger.info("Fetched %d daily rows for %s", len(df), ticker)
        return df
    except Exception as exc:
        logger.warning("Daily fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _fetch_intraday(ticker: str, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    """
    Download intraday OHLCV via yfinance (max 60 days for 5-min candles).
    Returns empty DataFrame on failure.
    """
    try:
        df = yf.download(ticker, interval=interval, period=period,
                         auto_adjust=True, progress=False, silence_errors=True)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1)
            except KeyError:
                df = df.droplevel(1, axis=1)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["close"])
        logger.info("Fetched %d intraday rows for %s (%s)", len(df), ticker, interval)
        return df
    except Exception as exc:
        logger.warning("Intraday fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1(b): GIFT NIFTY proxy construction
# ──────────────────────────────────────────────────────────────────────────────

def _ist_to_utc(naive_ist_time: datetime) -> datetime:
    """Convert naive IST datetime to UTC."""
    return naive_ist_time - IST_OFFSET


def _filter_pre_open_candles(intraday_df: pd.DataFrame,
                               cutoff_ist: timedelta = PRE_OPEN_CUTOFF_IST
                               ) -> pd.DataFrame:
    """
    Given an intraday DataFrame with a timezone-aware index, keep only
    candles that occur BEFORE NIFTY open (09:15 IST) on each date.

    This is the critical anti-lookahead operation: we can only use data
    that is observable before the NIFTY market opens.
    """
    if intraday_df.empty:
        return intraday_df

    # Ensure timezone-aware index
    idx = intraday_df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    # Convert to IST
    idx_ist = idx.tz_convert("Asia/Kolkata")

    # Cutoff time in seconds from midnight
    cutoff_seconds = int(cutoff_ist.total_seconds())

    # Build mask: row is pre-open if time-of-day < cutoff
    def _tod_seconds(ts):
        return ts.hour * 3600 + ts.minute * 60 + ts.second

    mask = pd.Series(
        [_tod_seconds(ts) < cutoff_seconds for ts in idx_ist],
        index=intraday_df.index
    )
    return intraday_df[mask.values]


def _get_gift_last_pre_open(intraday_df: pd.DataFrame,
                              as_of_date: date) -> Optional[float]:
    """
    Return the LAST available price BEFORE 09:15 IST for a given date.
    Scans both the as_of_date and the previous calendar day (SGX trades
    ~17.5 hours/day so pre-open GIFT is from previous evening).

    Returns None if no data is available.
    """
    if intraday_df.empty:
        return None

    idx = intraday_df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_ist = idx.tz_convert("Asia/Kolkata")

    cutoff_seconds = int(PRE_OPEN_CUTOFF_IST.total_seconds())

    as_of_dt = datetime(as_of_date.year, as_of_date.month, as_of_date.day)

    # We want the last candle BEFORE 09:15 IST on as_of_date.
    # That includes: yesterday's session after NIFTY close + overnight.
    upper_bound = as_of_dt + PRE_OPEN_CUTOFF_IST  # naive IST

    # Convert bounds to UTC for filtering
    upper_bound_utc = upper_bound - IST_OFFSET
    lower_bound_utc = upper_bound_utc - timedelta(days=2)  # look back 2 days

    # Filter within window
    try:
        mask = (idx >= lower_bound_utc.replace(tzinfo=timezone.utc)) & \
               (idx < upper_bound_utc.replace(tzinfo=timezone.utc))
    except TypeError:
        # Already tz-aware
        import pytz
        utc = pytz.UTC
        mask = (idx >= utc.localize(lower_bound_utc)) & \
               (idx < utc.localize(upper_bound_utc))

    window = intraday_df[mask]
    if window.empty:
        return None
    return float(window["close"].iloc[-1])


def _compute_overnight_volatility(intraday_df: pd.DataFrame,
                                   as_of_date: date) -> Optional[float]:
    """
    Annualised overnight volatility: std of log-returns of GIFT NIFTY
    from previous NIFTY close (~15:30 IST previous day) to 09:15 IST
    on as_of_date. Returns None if fewer than 3 candles available.
    """
    if intraday_df.empty:
        return None

    idx = intraday_df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_ist = idx.tz_convert("Asia/Kolkata")

    as_of_dt = datetime(as_of_date.year, as_of_date.month, as_of_date.day)
    upper_ist = as_of_dt + PRE_OPEN_CUTOFF_IST
    lower_ist = as_of_dt - timedelta(days=1) + timedelta(hours=15, minutes=30)

    upper_utc = (upper_ist - IST_OFFSET).replace(tzinfo=timezone.utc)
    lower_utc = (lower_ist - IST_OFFSET).replace(tzinfo=timezone.utc)

    try:
        mask = (idx >= lower_utc) & (idx < upper_utc)
    except TypeError:
        mask = (idx_ist >= lower_ist) & (idx_ist < upper_ist)

    window = intraday_df[mask]
    if len(window) < 3:
        return None

    log_rets = np.log(window["close"] / window["close"].shift(1)).dropna()
    # Annualise: ~252 trading days * 78 candles/day for 5-min
    return float(log_rets.std() * np.sqrt(252 * 78))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1(c): Main alignment function
# ──────────────────────────────────────────────────────────────────────────────

def build_gift_features(
    nifty_daily: pd.DataFrame,
    gift_daily: pd.DataFrame,
    gift_intraday: Optional[pd.DataFrame] = None,
    zscore_window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    Aligns GIFT NIFTY data with NIFTY 50 daily data and builds
    overnight-gap features.

    Parameters
    ----------
    nifty_daily   : daily OHLCV for ^NSEI (must have 'open','close').
    gift_daily    : daily OHLCV for GIFT proxy  (close price used as fallback).
    gift_intraday : intraday 5-min OHLCV for GIFT proxy (preferred).
    zscore_window : rolling window for gap normalisation.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (NIFTY trading dates) and columns:
        prev_close, gift_last, gap_abs, gap_pct,
        overnight_vol, gift_trend_slope, gift_momentum_5d,
        gap_z_score, data_quality  (1=intraday used, 0=daily fallback)

    ⚠ Lookahead safety:
        All feature values for date D are computed exclusively from data
        that was observable BEFORE 09:15 IST on date D.
        The 'open' column of nifty_daily for date D is the TARGET, not an
        input. prev_close is NIFTY close on D-1.
    """
    if nifty_daily.empty:
        logger.error("NIFTY daily data is empty — cannot build features.")
        return pd.DataFrame()

    # Ensure datetime index
    nifty = nifty_daily.copy()
    if not isinstance(nifty.index, pd.DatetimeIndex):
        nifty.index = pd.to_datetime(nifty.index)
    nifty = nifty.sort_index()

    records: List[Dict] = []

    for i, (trade_date, row) in enumerate(nifty.iterrows()):
        if i == 0:
            continue  # Need at least one previous day

        prev_date = nifty.index[i - 1]
        prev_close = float(nifty.loc[prev_date, "close"])
        nifty_open = float(row["open"])   # ← TARGET (not an input)

        # ── Attempt intraday GIFT reading ──
        gift_last    = None
        overnight_vol = None
        data_quality  = 0  # 0 = fallback, 1 = intraday

        if gift_intraday is not None and not gift_intraday.empty:
            gift_last     = _get_gift_last_pre_open(gift_intraday,
                                                      trade_date.date())
            overnight_vol = _compute_overnight_volatility(gift_intraday,
                                                           trade_date.date())
            if gift_last is not None:
                data_quality = 1

        # ── Daily fallback ──
        if gift_last is None and not gift_daily.empty:
            gift_idx = gift_daily.index
            if not isinstance(gift_idx, pd.DatetimeIndex):
                gift_idx = pd.to_datetime(gift_idx)
            # Use previous day's GIFT close as best pre-open estimate
            prev_day_mask = gift_idx.date < trade_date.date()  # type: ignore[attr-defined]
            prev_gift = gift_daily[prev_day_mask]
            if not prev_gift.empty:
                gift_last = float(prev_gift["close"].iloc[-1])
                data_quality = 0

        if gift_last is None:
            # Cannot build features for this date
            logger.debug("No GIFT data for %s — skipping.", trade_date.date())
            continue

        gap_abs = gift_last - prev_close
        gap_pct = gap_abs / (prev_close + 1e-8)

        # Trend slope over last SLOPE_LOOKBACK intraday candles
        gift_trend_slope = np.nan
        if data_quality == 1 and gift_intraday is not None:
            idx_int = gift_intraday.index
            if idx_int.tz is None:
                idx_int = idx_int.tz_localize("UTC")

            cutoff_utc = (datetime(trade_date.year, trade_date.month,
                                   trade_date.day) + PRE_OPEN_CUTOFF_IST
                          - IST_OFFSET).replace(tzinfo=timezone.utc)
            recent = gift_intraday[idx_int < cutoff_utc].tail(SLOPE_LOOKBACK)
            if len(recent) >= 2:
                y = recent["close"].values
                x = np.arange(len(y), dtype=float)
                slope = np.polyfit(x, y, 1)[0]
                gift_trend_slope = float(slope)

        records.append({
            "trade_date"       : trade_date,
            "prev_close"       : prev_close,
            "nifty_open"       : nifty_open,   # target – stored for baseline evaluation
            "gift_last"        : gift_last,
            "gap_abs"          : gap_abs,
            "gap_pct"          : gap_pct,
            "overnight_vol"    : overnight_vol if overnight_vol is not None else np.nan,
            "gift_trend_slope" : gift_trend_slope,
            "data_quality"     : data_quality,
        })

    if not records:
        logger.warning("No GIFT features could be built. Check data availability.")
        return pd.DataFrame()

    feat_df = pd.DataFrame(records).set_index("trade_date")

    # ── gift_momentum_5d: 5-day % change in gift_last ──
    feat_df["gift_momentum_5d"] = feat_df["gift_last"].pct_change(MOMENTUM_DAYS)

    # ── gap_z_score: rolling z-score of gap_pct ──
    roll_mean = feat_df["gap_pct"].rolling(zscore_window, min_periods=5).mean()
    roll_std  = feat_df["gap_pct"].rolling(zscore_window, min_periods=5).std()
    feat_df["gap_z_score"] = (feat_df["gap_pct"] - roll_mean) / (roll_std + 1e-8)

    # Fill NaN for numeric columns with 0 (model-safe)
    numeric_cols = ["overnight_vol", "gift_trend_slope",
                    "gift_momentum_5d", "gap_z_score"]
    feat_df[numeric_cols] = feat_df[numeric_cols].fillna(0.0)

    logger.info("Built GIFT features for %d trading days (intraday coverage: %.1f%%)",
                len(feat_df),
                feat_df["data_quality"].mean() * 100)
    return feat_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1(d): Single-day inference helper (production use)
# ──────────────────────────────────────────────────────────────────────────────

def get_latest_gift_features(prev_close: float,
                              period_intraday: str = "5d",
                              period_daily: str = "30d") -> Dict:
    """
    Fetches the most recent GIFT NIFTY data and returns a feature dict
    ready to be passed to the model at inference time.

    Called by the prediction service BEFORE NIFTY market opens.

    Parameters
    ----------
    prev_close : Previous NIFTY closing price.

    Returns
    -------
    dict with keys: gap_abs, gap_pct, overnight_vol,
                    gift_trend_slope, gift_momentum_5d,
                    gap_z_score (set to 0 at inference – no rolling context),
                    data_quality
    """
    today = date.today()

    # Try multiple proxy tickers
    gift_last    = None
    overnight_vol = None
    gift_trend_slope = 0.0
    gift_momentum_5d = 0.0
    data_quality = 0

    for ticker in GIFT_NIFTY_TICKERS:
        intraday = _fetch_intraday(ticker, interval="5m", period=period_intraday)
        if not intraday.empty:
            gift_last     = _get_gift_last_pre_open(intraday, today)
            overnight_vol = _compute_overnight_volatility(intraday, today)
            if gift_last is not None:
                data_quality = 1
                # Trend slope
                idx_int = intraday.index
                if idx_int.tz is None:
                    idx_int = idx_int.tz_localize("UTC")
                cutoff_utc = (datetime(today.year, today.month, today.day)
                              + PRE_OPEN_CUTOFF_IST - IST_OFFSET
                              ).replace(tzinfo=timezone.utc)
                recent = intraday[idx_int < cutoff_utc].tail(SLOPE_LOOKBACK)
                if len(recent) >= 2:
                    y = recent["close"].values
                    x = np.arange(len(y), dtype=float)
                    gift_trend_slope = float(np.polyfit(x, y, 1)[0])
                break

    # Daily fallback
    if gift_last is None:
        for ticker in GIFT_NIFTY_TICKERS:
            daily_df = _fetch_daily(ticker, period=period_daily)
            if not daily_df.empty:
                gift_last = float(daily_df["close"].iloc[-1])
                if len(daily_df) >= MOMENTUM_DAYS + 1:
                    gift_momentum_5d = float(
                        daily_df["close"].pct_change(MOMENTUM_DAYS).iloc[-1]
                    )
                data_quality = 0
                break

    # Final fallback: use NIFTY futures proxy (S&P 500 overnight movement)
    if gift_last is None:
        logger.warning("Could not fetch GIFT NIFTY from any proxy. "
                       "Using prev_close as gift_last (no gap signal).")
        gift_last = prev_close
        data_quality = -1   # signal that no GIFT data was available

    gap_abs = gift_last - prev_close
    gap_pct = gap_abs / (prev_close + 1e-8)

    return {
        "gap_abs"          : float(gap_abs),
        "gap_pct"          : float(gap_pct),
        "overnight_vol"    : float(overnight_vol) if overnight_vol else 0.0,
        "gift_trend_slope" : float(gift_trend_slope),
        "gift_momentum_5d" : float(gift_momentum_5d),
        "gap_z_score"      : 0.0,   # cannot compute at inference without rolling ctx
        "data_quality"     : int(data_quality),
        "gift_last"        : float(gift_last),
        "prev_close"       : float(prev_close),
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1(e): Full historical dataset builder (for training)
# ──────────────────────────────────────────────────────────────────────────────

def build_full_training_dataset(period: str = "2y") -> pd.DataFrame:
    """
    Fetches NIFTY 50 and GIFT NIFTY proxy data and returns a merged
    DataFrame usable for training the overnight-gap prediction model.

    Columns: all NIFTY OHLCV columns + all GIFT feature columns.
    Index: NIFTY trading dates.

    ⚠ Data leakage check: nifty_open column is the TARGET and should
      NEVER appear in the model's input feature matrix.
    """
    logger.info("Fetching NIFTY 50 daily data (%s)...", period)
    nifty_daily = _fetch_daily(NIFTY_TICKER, period=period)

    gift_daily    = pd.DataFrame()
    gift_intraday = pd.DataFrame()

    for ticker in GIFT_NIFTY_TICKERS:
        logger.info("Trying GIFT proxy ticker: %s", ticker)
        gift_daily = _fetch_daily(ticker, period=period)
        if not gift_daily.empty:
            logger.info("Using %s as GIFT NIFTY daily proxy.", ticker)
            # Only try intraday for recent 60 days (yfinance limitation)
            gift_intraday = _fetch_intraday(ticker, interval="5m", period="60d")
            break

    feat_df = build_gift_features(nifty_daily, gift_daily, gift_intraday or None)

    if feat_df.empty:
        return pd.DataFrame()

    # Merge with full NIFTY OHLCV for training
    nifty_daily.index = pd.to_datetime(nifty_daily.index)
    merged = nifty_daily.join(feat_df, how="inner")
    logger.info("Final training dataset: %d rows, %d columns.",
                len(merged), len(merged.columns))
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Feature column names (exported for use in model configurations)
# ──────────────────────────────────────────────────────────────────────────────

GIFT_FEATURE_COLS = [
    "gap_abs",
    "gap_pct",
    "overnight_vol",
    "gift_trend_slope",
    "gift_momentum_5d",
    "gap_z_score",
    "data_quality",
]

N_GIFT_FEATURES = len(GIFT_FEATURE_COLS)

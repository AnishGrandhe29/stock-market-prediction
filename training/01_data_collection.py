# %% [markdown]
# # Step 1 – Data Collection: NIFTY 50 + GIFT NIFTY
#
# Fetches:
#   * NIFTY 50 OHLCV  (^NSEI via Yahoo Finance)
#   * GIFT NIFTY proxy (best available via Yahoo Finance)
#   * Macro context: VIX, USD/INR, S&P 500
#   * News headlines (Google RSS)
#   * Synthetic sentiment baseline
#
# Output files (in DATA_DIR):
#   nifty50_ohlcv.csv
#   gift_nifty_raw.csv
#   gift_nifty_features.csv   ← aligned gap features (no lookahead)
#   macro_features.csv
#   sentiment_scores.csv
#   news_headlines.json
#
# ⚠ Run this ONCE before feature engineering.

# %% [markdown]
# ## Setup

# %%
# Google Colab setup – skip locally
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
    import subprocess
    subprocess.run(
        ["pip", "install", "yfinance", "pandas", "numpy", "feedparser",
         "pandas-ta", "scikit-learn", "ta", "-q"],
        check=True,
    )
except ImportError:
    IN_COLAB = False

# %%
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_collection")

# ──────────────────────────────────────────────────────────
# CONFIG  (change only these lines)
# ──────────────────────────────────────────────────────────
DATA_DIR    = Path(os.getenv("DATA_DIR", "data"))
YEARS_BACK  = int(os.getenv("YEARS_BACK", "3"))
NIFTY_SYM   = "^NSEI"

# GIFT NIFTY proxies – tried in order until one returns data
GIFT_PROXIES = [
    "NIFTYBEES.NS",   # NIFTY 50 BeES ETF – tracks NIFTY closely
    "JUNIORBEES.NS",  # Nifty Junior ETF
    "SETFNIF50.NS",   # SBI ETF Nifty 50
]

MACRO_SYMBOLS = {
    "vix"   : "^VIX",
    "usd_inr": "INR=X",
    "sp500" : "^GSPC",
    "gold"  : "GLD",
    "usdinr_fut": "DX-Y.NYB",
}
# ──────────────────────────────────────────────────────────

DATA_DIR.mkdir(parents=True, exist_ok=True)

end_dt   = datetime.now()
start_dt = end_dt - timedelta(days=365 * YEARS_BACK)
START    = start_dt.strftime("%Y-%m-%d")
END      = end_dt.strftime("%Y-%m-%d")

log.info("Fetching %d years of data: %s → %s", YEARS_BACK, START, END)


# %% [markdown]
# ## 1. NIFTY 50 OHLCV

# %%
def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Collapse MultiIndex columns returned by yfinance >= 0.2."""
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except KeyError:
            df = df.droplevel(1, axis=1)
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_nifty(symbol: str = NIFTY_SYM) -> pd.DataFrame:
    df = yf.download(symbol, start=START, end=END,
                     auto_adjust=True, progress=False)
    df = _flatten_columns(df, symbol)
    df = df.dropna(subset=["close"])
    log.info("NIFTY 50: %d rows", len(df))
    return df


df_nifty = fetch_nifty()
df_nifty.to_csv(DATA_DIR / "nifty50_ohlcv.csv")
log.info("Saved → %s", DATA_DIR / "nifty50_ohlcv.csv")
df_nifty.tail(3)

# %% [markdown]
# ## 2. GIFT NIFTY Proxy
#
# GIFT NIFTY (NSE IFSC futures) has no reliable direct yfinance ticker.
# We use the best available ETF proxy and compute the overnight gap from it.
#
# ⚠ LOOKAHEAD SAFETY:
#   gap_abs for date D = gift_close(D-1) - nifty_close(D-1)
#   This is knowable before NIFTY opens on day D.

# %%
def fetch_gift_proxy(proxies: list = GIFT_PROXIES) -> tuple[pd.DataFrame, str]:
    """Try each proxy ticker in order; return the first that succeeds."""
    for ticker in proxies:
        try:
            df = yf.download(ticker, start=START, end=END,
                             auto_adjust=True, progress=False)
            df = _flatten_columns(df, ticker)
            df = df.dropna(subset=["close"])
            if len(df) > 50:
                log.info("GIFT proxy: %s (%d rows)", ticker, len(df))
                return df, ticker
        except Exception as exc:
            log.warning("Proxy %s failed: %s", ticker, exc)
    log.warning("No GIFT proxy found – using empty DataFrame.")
    return pd.DataFrame(), "none"


df_gift_raw, gift_ticker = fetch_gift_proxy()
if not df_gift_raw.empty:
    df_gift_raw.to_csv(DATA_DIR / "gift_nifty_raw.csv")
    log.info("Saved → %s", DATA_DIR / "gift_nifty_raw.csv")


# %% [markdown]
# ## 3. Build Gap Features (Anti-Lookahead Aligned)

# %%
def build_gift_gap_features(
    nifty_df: pd.DataFrame,
    gift_df: pd.DataFrame,
    zscore_window: int = 20,
) -> pd.DataFrame:
    """
    For each NIFTY trading date D, compute overnight gap features using
    only data that existed BEFORE NIFTY opened on D.

    Specifically:
        prev_close = nifty_close(D-1)
        gift_close = gift_proxy_close(D-1)   ← last available, no lookahead
        gap_abs    = gift_close - prev_close
        gap_pct    = gap_abs / prev_close

    Returns DataFrame indexed by NIFTY trade dates.
    """
    if nifty_df.empty:
        return pd.DataFrame()

    nifty = nifty_df.copy()
    if not isinstance(nifty.index, pd.DatetimeIndex):
        nifty.index = pd.to_datetime(nifty.index)
    nifty = nifty.sort_index()

    # Shift NIFTY close by 1 to get prev_close for each row
    nifty["prev_close"] = nifty["close"].shift(1)

    # If gift data available, forward-fill to align with NIFTY dates
    if not gift_df.empty:
        gift = gift_df.copy()
        if not isinstance(gift.index, pd.DatetimeIndex):
            gift.index = pd.to_datetime(gift.index)
        gift = gift.sort_index()

        # Align gift close to NIFTY trading days, then shift by 1
        # (we use gift_close(D-1) for NIFTY open on D)
        gift_aligned = gift["close"].reindex(nifty.index, method="ffill")
        nifty["gift_close"] = gift_aligned.shift(1)
    else:
        # Fallback: use NIFTY prev-close as gift_close (zero gap signal)
        log.warning("No GIFT data – setting gift_close = prev_close (zero gap).")
        nifty["gift_close"] = nifty["prev_close"]

    # Drop the first row (no prev_close available)
    nifty = nifty.dropna(subset=["prev_close", "gift_close"])

    # Core gap features
    nifty["gap_abs"] = nifty["gift_close"] - nifty["prev_close"]
    nifty["gap_pct"] = nifty["gap_abs"] / (nifty["prev_close"] + 1e-8)

    # Rolling z-score of gap_pct (normalises extreme-gap days)
    roll_mean = nifty["gap_pct"].rolling(zscore_window, min_periods=5).mean()
    roll_std  = nifty["gap_pct"].rolling(zscore_window, min_periods=5).std()
    nifty["gap_z_score"] = (nifty["gap_pct"] - roll_mean) / (roll_std + 1e-8)

    # 5-day gift momentum
    nifty["gift_momentum_5d"] = nifty["gift_close"].pct_change(5)

    # Data-quality flag: 1 = real gift data, 0 = fallback
    nifty["data_quality"] = 0 if gift_df.empty else 1

    # Target: opening gap for NIFTY (used during training, NOT as input)
    nifty["target_gap"] = nifty["open"] - nifty["prev_close"]

    feat_cols = [
        "prev_close", "gift_close", "gap_abs", "gap_pct",
        "gap_z_score", "gift_momentum_5d", "data_quality",
        "target_gap",   # ← label only, never used as model input
    ]
    out = nifty[feat_cols].fillna(0.0)
    log.info("Gap features built: %d rows", len(out))
    return out


df_gift_feat = build_gift_gap_features(df_nifty, df_gift_raw)
df_gift_feat.to_csv(DATA_DIR / "gift_nifty_features.csv")
log.info("Saved → %s", DATA_DIR / "gift_nifty_features.csv")
df_gift_feat.tail(3)

# %% [markdown]
# ## 4. Macro Context Features

# %%
def fetch_macro(symbols: dict = MACRO_SYMBOLS) -> pd.DataFrame:
    """Fetch macro proxies and compute daily log-returns."""
    frames = []
    for name, ticker in symbols.items():
        try:
            df = yf.download(ticker, start=START, end=END,
                             auto_adjust=True, progress=False)
            df = _flatten_columns(df, ticker)
            if "close" in df.columns and len(df) > 10:
                s = df["close"].pct_change(1).rename(f"mac_{name}")
                frames.append(s)
                log.info("Macro %s (%s): %d rows", name, ticker, len(df))
        except Exception as exc:
            log.warning("Macro %s failed: %s", ticker, exc)

    if not frames:
        log.warning("No macro data fetched.")
        return pd.DataFrame()

    macro_df = pd.concat(frames, axis=1)
    macro_df.ffill(inplace=True)
    macro_df.fillna(0.0, inplace=True)
    return macro_df


df_macro = fetch_macro()
if not df_macro.empty:
    df_macro.to_csv(DATA_DIR / "macro_features.csv")
    log.info("Saved → %s", DATA_DIR / "macro_features.csv")

# %% [markdown]
# ## 5. News Headlines

# %%
def fetch_news(max_items: int = 100) -> list:
    """Fetch recent NIFTY news from Google RSS."""
    try:
        import feedparser
        url = (
            "https://news.google.com/rss/search"
            "?q=nifty+50+indian+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title"    : entry.get("title", ""),
                "published": entry.get("published", ""),
                "link"     : entry.get("link", ""),
            })
        log.info("News: fetched %d headlines", len(items))
        return items
    except Exception as exc:
        log.warning("News fetch failed: %s", exc)
        return []


news = fetch_news()
with open(DATA_DIR / "news_headlines.json", "w", encoding="utf-8") as f:
    json.dump(news, f, indent=2)

# %% [markdown]
# ## 6. Synthetic Sentiment (price-derived baseline)

# %%
def generate_synthetic_sentiment(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic sentiment scores from price returns.
    For initial training only — replace with real NLP sentiment in production.
    """
    rets = df_prices["close"].pct_change().fillna(0)
    base = np.tanh(rets * 50)   # squash to (-1, 1)
    rng  = np.random.default_rng(42)   # reproducible

    sentiment = pd.DataFrame({
        "news_sentiment"    : np.clip(base * 0.8  + rng.normal(0, 0.10, len(base)), -1, 1),
        "reddit_sentiment"  : np.clip(base * 0.6  + rng.normal(0, 0.15, len(base)), -1, 1),
        "combined_sentiment": np.clip(base,                                           -1, 1),
    }, index=df_prices.index)

    log.info("Sentiment: generated %d rows", len(sentiment))
    return sentiment


df_sentiment = generate_synthetic_sentiment(df_nifty)
df_sentiment.to_csv(DATA_DIR / "sentiment_scores.csv")
log.info("Saved → %s", DATA_DIR / "sentiment_scores.csv")

# %% [markdown]
# ## 7. Summary

# %%
print("\n" + "=" * 55)
print("  DATA COLLECTION COMPLETE")
print("=" * 55)
files_info = [
    (DATA_DIR / "nifty50_ohlcv.csv",      f"{len(df_nifty)} trading days"),
    (DATA_DIR / "gift_nifty_raw.csv",     f"{len(df_gift_raw)} rows ({gift_ticker})"),
    (DATA_DIR / "gift_nifty_features.csv",f"{len(df_gift_feat)} rows"),
    (DATA_DIR / "macro_features.csv",     f"{len(df_macro)} rows"),
    (DATA_DIR / "sentiment_scores.csv",   f"{len(df_sentiment)} rows"),
    (DATA_DIR / "news_headlines.json",    f"{len(news)} headlines"),
]
for path, info in files_info:
    exists = "✓" if path.exists() else "✗"
    print(f"  {exists}  {path.name:<35} {info}")
print(f"\n  Next: Run 02_feature_engineering.py")

# %%
# Colab: zip + download
if IN_COLAB:
    import shutil
    from google.colab import files  # type: ignore
    shutil.make_archive("nifty50_data", "zip", str(DATA_DIR))
    files.download("nifty50_data.zip")

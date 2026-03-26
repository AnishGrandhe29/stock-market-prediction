# %% [markdown]
# # Step 2 – Feature Engineering (with GIFT NIFTY)
#
# Reads raw CSVs from Step 1 and produces a model-ready dataset.
#
# Pipeline:
#   1. Compute 35 NIFTY technical indicators
#   2. Merge GIFT gap features (gap_abs, gap_pct, gap_z_score, gift_momentum_5d)
#   3. Merge macro features (VIX, USD/INR, S&P, Gold)
#   4. Merge sentiment features
#   5. Reformulate target: gap = open_t - close_{t-1}
#   6. Create 60-step sequences  (OHLCV + full feature matrix per timestep)
#   7. Build separate gift_feat array  (overnight snapshot, not sequenced)
#   8. Train/Val/Test split (time-ordered, no shuffling)
#   9. Save arrays + scalers
#
# ⚠ NO LOOKAHEAD: gift features use only D-1 data. `open` column is the
#   target and is NOT included in any input feature matrix.

# %% [markdown]
# ## Setup

# %%
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
    import subprocess
    subprocess.run(
        ["pip", "install", "pandas", "numpy", "pandas-ta", "scikit-learn", "ta", "-q"],
        check=True,
    )
except ImportError:
    IN_COLAB = False

# %%
import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_engineering")

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
DATA_DIR  = Path(os.getenv("DATA_DIR", "data"))
SEQ_LEN   = int(os.getenv("SEQ_LEN", "60"))
TRAIN_PCT = float(os.getenv("TRAIN_PCT", "0.70"))
VAL_PCT   = float(os.getenv("VAL_PCT",   "0.15"))
# test = 1 - TRAIN_PCT - VAL_PCT
# ──────────────────────────────────────────────────────────

DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Load Raw Data

# %%
def load_csv(path: Path, date_col: str = None) -> pd.DataFrame:
    """Load a CSV with a DatetimeIndex."""
    if not path.exists():
        log.warning("File not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    log.info("Loaded %s: %d rows", path.name, len(df))
    return df


df_nifty   = load_csv(DATA_DIR / "nifty50_ohlcv.csv")
df_gift    = load_csv(DATA_DIR / "gift_nifty_features.csv")
df_macro   = load_csv(DATA_DIR / "macro_features.csv")
df_sent    = load_csv(DATA_DIR / "sentiment_scores.csv")

assert not df_nifty.empty, "NIFTY data missing – run 01_data_collection.py first."
df_nifty.columns = [c.lower() for c in df_nifty.columns]
log.info("NIFTY columns: %s", list(df_nifty.columns))

# %% [markdown]
# ## 2. NIFTY Technical Indicators

# %%
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TA indicators using the `ta` library.
    All indicators are causal (no future leakage by construction).

    Returns DataFrame with same index as input, additional indicator columns.
    """
    try:
        import ta
        from ta.trend import MACD, EMAIndicator, ADXIndicator
        from ta.momentum import RSIIndicator, StochasticOscillator
        from ta.volatility import BollingerBands, AverageTrueRange
        from ta.volume import OnBalanceVolumeIndicator
    except ImportError:
        log.warning("ta library not found – using pandas-ta fallback.")
        try:
            import pandas_ta as pta
            df["rsi_14"]      = pta.rsi(df["close"], length=14)
            macd = pta.macd(df["close"])
            if macd is not None:
                df["macd"]        = macd.iloc[:, 0]
                df["macd_signal"] = macd.iloc[:, 1]
                df["macd_hist"]   = macd.iloc[:, 2]
            df["ema_5"]       = pta.ema(df["close"], length=5)
            df["ema_20"]      = pta.ema(df["close"], length=20)
            df["ema_50"]      = pta.ema(df["close"], length=50)
            df["atr_14"]      = pta.atr(df["high"], df["low"], df["close"], length=14)
            df["adx"]         = 25.0
            df["stoch_k"]     = 50.0
            df["stoch_d"]     = 50.0
            bb = pta.bbands(df["close"])
            if bb is not None:
                df["bb_upper"]  = bb.iloc[:, 0]
                df["bb_middle"] = bb.iloc[:, 2]
                df["bb_lower"]  = bb.iloc[:, 4]
            df["obv"]         = pta.obv(df["close"], df["volume"])
            return df
        except ImportError:
            log.error("Neither ta nor pandas_ta found. Install: pip install ta")
            # Stub all indicators with neutral values
            df["rsi_14"] = 50.0; df["macd"] = 0.0; df["macd_signal"] = 0.0
            df["macd_hist"] = 0.0; df["ema_5"] = df["close"]
            df["ema_20"] = df["close"]; df["ema_50"] = df["close"]
            df["atr_14"] = 0.0; df["adx"] = 25.0
            df["stoch_k"] = 50.0; df["stoch_d"] = 50.0
            df["bb_upper"] = df["close"]; df["bb_middle"] = df["close"]
            df["bb_lower"] = df["close"]; df["obv"] = 0.0
            return df

    c   = df["close"]
    h   = df["high"]
    lo  = df["low"]
    v   = df["volume"]

    # Returns
    df["ret_1d"]  = c.pct_change(1)
    df["ret_5d"]  = c.pct_change(5)
    df["ret_20d"] = c.pct_change(20)
    df["log_r1"]  = np.log(c / c.shift(1))

    # Normalised close/volume
    df["c_norm"]  = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)
    df["v_norm"]  = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-8)
    df["hl_rng"]  = (h - lo) / (c + 1e-8)
    df["oc_rng"]  = (c - df["open"]) / (c + 1e-8)

    # EMA ratios
    for w in [5, 10, 20, 50]:
        ema = EMAIndicator(c, window=w).ema_indicator()
        df[f"ema{w}"] = ema
        df[f"er{w}"]  = c / (ema + 1e-8) - 1

    # MACD
    m = MACD(c)
    df["macd"]     = m.macd()
    df["macd_sig"] = m.macd_signal()
    df["macd_dif"] = m.macd_diff()
    df["adx"]      = ADXIndicator(h, lo, c).adx()

    # Oscillators
    df["rsi_14"]  = RSIIndicator(c, window=14).rsi()
    df["rsi_28"]  = RSIIndicator(c, window=28).rsi()
    st = StochasticOscillator(h, lo, c)
    df["stoch_k"] = st.stoch()
    df["stoch_d"] = st.stoch_signal()

    # Volatility
    bb = BollingerBands(c, window=20)
    df["bb_hi"]  = bb.bollinger_hband()
    df["bb_lo"]  = bb.bollinger_lband()
    df["bb_wid"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()
    df["atr_14"] = AverageTrueRange(h, lo, c).average_true_range()
    df["vol_20"] = df["log_r1"].rolling(20).std() * np.sqrt(252)
    df["vol_60"] = df["log_r1"].rolling(60).std() * np.sqrt(252)

    # Volume
    df["obv"]       = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["vol_ratio"] = v / (v.rolling(20).mean() + 1e-8)

    # Distribution
    df["skew_20"] = df["log_r1"].rolling(20).skew()
    df["kurt_20"] = df["log_r1"].rolling(20).kurt()

    log.info("Computed %d technical indicators.", len(df.columns))
    return df


df_nifty = compute_technical_indicators(df_nifty)

# %% [markdown]
# ## 3. Merge GIFT Gap Features
#
# ⚠ These are D-1 features: safe to merge without shifting again
#    (shifting was already done in 01_data_collection.py).

# %%
GIFT_FEATURE_COLS = [
    "gap_abs",
    "gap_pct",
    "gap_z_score",
    "gift_momentum_5d",
    "data_quality",
]

if not df_gift.empty:
    # Standardise column names
    df_gift.columns = [c.lower() for c in df_gift.columns]

    # Mask target_gap – must NOT enter input features
    input_gift = df_gift[[c for c in GIFT_FEATURE_COLS if c in df_gift.columns]]
    df_nifty   = df_nifty.join(input_gift, how="left")

    # For dates where gift data is absent, fill with 0 (neutral gap)
    df_nifty[GIFT_FEATURE_COLS] = df_nifty[GIFT_FEATURE_COLS].fillna(0.0)
    log.info("Merged GIFT gap features: %s", GIFT_FEATURE_COLS)
else:
    # No gift data – add zero columns so downstream code doesn't break
    for col in GIFT_FEATURE_COLS:
        df_nifty[col] = 0.0
    log.warning("GIFT data unavailable – gap features set to 0.")

# %% [markdown]
# ## 4. Merge Macro and Sentiment

# %%
if not df_macro.empty:
    df_macro.columns = [c.lower() for c in df_macro.columns]
    df_nifty = df_nifty.join(df_macro, how="left")
    df_nifty[list(df_macro.columns)] = df_nifty[list(df_macro.columns)].ffill().fillna(0.0)
    log.info("Merged macro features: %s", list(df_macro.columns))

if not df_sent.empty:
    df_sent.columns = [c.lower() for c in df_sent.columns]
    sent_cols = ["news_sentiment", "reddit_sentiment", "combined_sentiment"]
    available_sent = [c for c in sent_cols if c in df_sent.columns]
    df_nifty = df_nifty.join(df_sent[available_sent], how="left")
    df_nifty[available_sent] = df_nifty[available_sent].fillna(0.0)
    log.info("Merged sentiment features: %s", available_sent)
else:
    for sc in ["news_sentiment", "reddit_sentiment", "combined_sentiment"]:
        df_nifty[sc] = 0.0


# %% [markdown]
# ## 5. Define Feature Column Groups

# %%
# Technical features (used in the 60-step temporal sequence)
TECH_FEATURE_COLS = [c for c in [
    "ret_1d", "ret_5d", "ret_20d", "log_r1",
    "c_norm", "v_norm", "hl_rng", "oc_rng",
    "ema5", "ema10", "ema20", "ema50",
    "er5", "er10", "er20", "er50",
    "macd", "macd_sig", "macd_dif", "adx",
    "rsi_14", "rsi_28", "stoch_k", "stoch_d",
    "bb_hi", "bb_lo", "bb_wid", "bb_pct",
    "atr_14", "vol_20", "vol_60", "obv", "vol_ratio",
    "skew_20", "kurt_20",
    # Macro (included in per-day snapshot; forward-filled)
    "mac_vix", "mac_usd_inr", "mac_sp500", "mac_gold",
    # Sentiment
    "news_sentiment", "reddit_sentiment", "combined_sentiment",
] if c in df_nifty.columns]

# GIFT features  (overnight snapshot – NOT sequenced, kept separate)
# These are already at D-1 resolution; using them as-is is safe.
GIFT_INPUT_COLS = [c for c in GIFT_FEATURE_COLS if c in df_nifty.columns]

log.info("Technical sequence features  : %d", len(TECH_FEATURE_COLS))
log.info("GIFT overnight feature cols  : %d  → %s", len(GIFT_INPUT_COLS), GIFT_INPUT_COLS)

# %% [markdown]
# ## 6. Target: Opening Gap

# %%
# gap = NIFTY_open_D - NIFTY_close_{D-1}
# This is the reformulated target (Step 5 of the architecture doc).
df_nifty["prev_close_target"] = df_nifty["close"].shift(1)
df_nifty["target_gap"]        = df_nifty["open"] - df_nifty["prev_close_target"]

# Drop rows that cannot form a full sequence or are missing core data
df_clean = df_nifty.dropna(subset=["target_gap", "prev_close_target"] + TECH_FEATURE_COLS[:5])
log.info("After dropna: %d usable rows.", len(df_clean))

# %% [markdown]
# ## 7. Normalisation

# %%
tech_scaler = StandardScaler()
gift_scaler = StandardScaler()

tech_arr   = tech_scaler.fit_transform(df_clean[TECH_FEATURE_COLS].fillna(0.0)).clip(-5, 5)
gift_arr   = df_clean[GIFT_INPUT_COLS].fillna(0.0).values.astype(np.float32)

# Normalise numeric gift features (all except data_quality flag)
numeric_gift_idx = [i for i, c in enumerate(GIFT_INPUT_COLS) if c != "data_quality"]
if numeric_gift_idx:
    gift_arr[:, numeric_gift_idx] = gift_scaler.fit_transform(
        gift_arr[:, numeric_gift_idx]
    ).clip(-5, 5)

target_gaps  = df_clean["target_gap"].values.astype(np.float32)
prev_closes  = df_clean["prev_close_target"].values.astype(np.float32)
actual_opens = df_clean["open"].values.astype(np.float32)

log.info("Tech array shape : %s", tech_arr.shape)
log.info("GIFT array shape : %s", gift_arr.shape)
log.info("Targets (gaps)   : %s", target_gaps.shape)

# %% [markdown]
# ## 8. Build Sequences

# %%
def create_sequences(
    tech: np.ndarray,
    gift: np.ndarray,
    targets: np.ndarray,
    prev_closes: np.ndarray,
    actual_opens: np.ndarray,
    seq_len: int,
):
    """
    Slide a window of size seq_len over the feature array.

    For sample i (starting from seq_len):
        seq_feat[i]  = tech[i-seq_len : i]   shape: (seq_len, n_tech)
        gift_feat[i] = gift[i]                shape: (n_gift,)   ← D-1 already
        target[i]    = targets[i]
        prev_c[i]    = prev_closes[i]

    ⚠ gift_feat is the snapshot for day i.  It uses gift data from i-1
       (ensured by the shift in Step 1).  No additional shifting needed.
    """
    X_seq, X_gift, y_gap, y_prev, y_open = [], [], [], [], []

    for i in range(seq_len, len(tech)):
        X_seq.append(tech[i - seq_len: i])
        X_gift.append(gift[i])
        y_gap.append(targets[i])
        y_prev.append(prev_closes[i])
        y_open.append(actual_opens[i])

    return (
        np.array(X_seq,  dtype=np.float32),
        np.array(X_gift, dtype=np.float32),
        np.array(y_gap,  dtype=np.float32),
        np.array(y_prev, dtype=np.float32),
        np.array(y_open, dtype=np.float32),
    )


X_seq, X_gift, y_gap, y_prev, y_open = create_sequences(
    tech_arr, gift_arr, target_gaps, prev_closes, actual_opens, SEQ_LEN
)

log.info("Sequence shape  : %s", X_seq.shape)
log.info("GIFT feat shape : %s", X_gift.shape)
log.info("Targets (gaps)  : %s", y_gap.shape)

# %% [markdown]
# ## 9. Baseline Evaluation (mandatory, Step 2 from architecture)

# %%
def evaluate_baseline(gift_feat: np.ndarray, y_gap_true: np.ndarray,
                       gift_input_cols: list) -> dict:
    """
    Baseline: pred_open = prev_close + gap_abs
    i.e. the model simply uses the GIFT gap signal directly.

    gap_abs is the first element in GIFT_INPUT_COLS.
    """
    if "gap_abs" not in gift_input_cols:
        log.warning("gap_abs not in gift features – baseline uses 0.")
        pred_gap = np.zeros(len(y_gap_true), dtype=np.float32)
    else:
        idx_gap_abs = gift_input_cols.index("gap_abs")
        pred_gap    = gift_feat[:, idx_gap_abs]   # already D-1

    errors     = np.abs(pred_gap - y_gap_true)
    sq_errors  = (pred_gap - y_gap_true) ** 2
    mae        = float(errors.mean())
    rmse       = float(np.sqrt(sq_errors.mean()))
    dir_pred   = np.sign(pred_gap)
    dir_true   = np.sign(y_gap_true)
    dir_acc    = float((dir_pred == dir_true).mean())

    result = {
        "model"       : "GIFT Carry-Forward Baseline",
        "n_samples"   : len(y_gap_true),
        "gap_mae_pts" : round(mae,  2),
        "gap_rmse_pts": round(rmse, 2),
        "direction_acc": round(dir_acc * 100, 1),
    }
    return result


baseline_result = evaluate_baseline(X_gift, y_gap, GIFT_INPUT_COLS)
print("\n" + "=" * 45)
print("  BASELINE: GIFT Carry-Forward")
print("=" * 45)
for k, v in baseline_result.items():
    print(f"  {k:<20s}: {v}")
print("=" * 45 + "\n")

# %% [markdown]
# ## 10. Train/Val/Test Split (time-ordered)

# %%
n = len(y_gap)
n_train = int(n * TRAIN_PCT)
n_val   = int(n * VAL_PCT)

def _split(arr):
    return (arr[:n_train],
            arr[n_train: n_train + n_val],
            arr[n_train + n_val:])

X_seq_tr,  X_seq_va,  X_seq_te  = _split(X_seq)
X_gift_tr, X_gift_va, X_gift_te = _split(X_gift)
y_gap_tr,  y_gap_va,  y_gap_te  = _split(y_gap)
y_prev_tr, y_prev_va, y_prev_te = _split(y_prev)
y_open_tr, y_open_va, y_open_te = _split(y_open)

log.info("Train: %d  |  Val: %d  |  Test: %d", n_train, n_val, n - n_train - n_val)

# %% [markdown]
# ## 11. Save All Outputs

# %%
# Sequences
np.save(DATA_DIR / "X_seq_train.npy",  X_seq_tr)
np.save(DATA_DIR / "X_seq_val.npy",    X_seq_va)
np.save(DATA_DIR / "X_seq_test.npy",   X_seq_te)

# GIFT overnight features
np.save(DATA_DIR / "X_gift_train.npy", X_gift_tr)
np.save(DATA_DIR / "X_gift_val.npy",   X_gift_va)
np.save(DATA_DIR / "X_gift_test.npy",  X_gift_te)

# Targets
np.save(DATA_DIR / "y_gap_train.npy",  y_gap_tr)
np.save(DATA_DIR / "y_gap_val.npy",    y_gap_va)
np.save(DATA_DIR / "y_gap_test.npy",   y_gap_te)

# Reconstruction helpers (not used as inputs)
np.save(DATA_DIR / "y_prev_train.npy", y_prev_tr)
np.save(DATA_DIR / "y_prev_val.npy",   y_prev_va)
np.save(DATA_DIR / "y_prev_test.npy",  y_prev_te)
np.save(DATA_DIR / "y_open_train.npy", y_open_tr)
np.save(DATA_DIR / "y_open_val.npy",   y_open_va)
np.save(DATA_DIR / "y_open_test.npy",  y_open_te)

# Scalers
with open(DATA_DIR / "tech_scaler.pkl", "wb") as f:
    pickle.dump(tech_scaler, f)
with open(DATA_DIR / "gift_scaler.pkl", "wb") as f:
    pickle.dump(gift_scaler, f)

# Metadata (column names for model config)
import json
meta = {
    "tech_feature_cols" : TECH_FEATURE_COLS,
    "gift_feature_cols" : GIFT_INPUT_COLS,
    "seq_len"           : SEQ_LEN,
    "n_tech_features"   : len(TECH_FEATURE_COLS),
    "n_gift_features"   : len(GIFT_INPUT_COLS),
    "baseline"          : baseline_result,
}
with open(DATA_DIR / "feature_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

log.info("All outputs saved to %s", DATA_DIR)

# %% [markdown]
# ## 12. Summary

# %%
print("\n" + "=" * 55)
print("  FEATURE ENGINEERING COMPLETE")
print("=" * 55)
print(f"  Sequence shape (train) : {X_seq_tr.shape}")
print(f"  GIFT feat  (train)     : {X_gift_tr.shape}")
print(f"  Target gaps (train)    : {y_gap_tr.shape}")
print(f"  Baseline gap MAE       : {baseline_result['gap_mae_pts']} pts")
print(f"  Baseline dir accuracy  : {baseline_result['direction_acc']}%")
print(f"\n  Next: Run 03_model_training.py (with GIFT integration)")
print("=" * 55)

# %%
if IN_COLAB:
    import shutil
    from google.colab import files  # type: ignore
    shutil.make_archive("processed_data", "zip", str(DATA_DIR))
    files.download("processed_data.zip")

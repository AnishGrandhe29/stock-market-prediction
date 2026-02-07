# %% [markdown]
# # NIFTY 50 Feature Engineering
# 
# This notebook computes technical indicators and prepares features for training.

# %% [markdown]
# ## Setup

# %%
!pip install pandas numpy pandas-ta scikit-learn -q

# %%
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import os
import pickle

# %% [markdown]
# ## 1. Load OHLCV Data

# %%
# Load NIFTY 50 data
df = pd.read_csv('data/nifty50_ohlcv.csv', index_col=0, parse_dates=True)
print(f"Loaded {len(df)} trading days")
df.head()

# %% [markdown]
# ## 2. Compute Technical Indicators

# %%
def compute_technical_indicators(df):
    """Compute all technical indicators using pandas-ta."""
    
    # Make a copy
    df = df.copy()
    
    # Momentum Indicators
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    
    macd = ta.macd(df['Close'])
    if macd is not None:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
    
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    if stoch is not None:
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    # Trend Indicators
    df['ema_5'] = ta.ema(df['Close'], length=5)
    df['ema_20'] = ta.ema(df['Close'], length=20)
    df['ema_50'] = ta.ema(df['Close'], length=50)
    df['sma_20'] = ta.sma(df['Close'], length=20)
    
    adx = ta.adx(df['High'], df['Low'], df['Close'])
    if adx is not None:
        df['adx'] = adx['ADX_14']
    
    # Volatility Indicators
    df['atr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    bbands = ta.bbands(df['Close'])
    if bbands is not None:
        df['bb_upper'] = bbands['BBU_5_2.0']
        df['bb_middle'] = bbands['BBM_5_2.0']
        df['bb_lower'] = bbands['BBL_5_2.0']
    
    # Volume Indicators
    df['obv'] = ta.obv(df['Close'], df['Volume'])
    df['volume_sma'] = ta.sma(df['Volume'], length=20)
    
    return df

# %%
df_with_ta = compute_technical_indicators(df)
print(f"Added {len(df_with_ta.columns) - len(df.columns)} technical indicators")
df_with_ta.tail()

# %% [markdown]
# ## 3. Create Target Variable

# %%
def create_targets(df, prediction_horizon=1):
    """
    Create target variables for prediction.
    
    Targets:
    - next_close: Next day's closing price
    - return_pct: Percentage return
    - direction: Up (1), Down (-1), Neutral (0)
    """
    df = df.copy()
    
    # Next day's close
    df['next_close'] = df['Close'].shift(-prediction_horizon)
    
    # Return percentage
    df['return_pct'] = (df['next_close'] - df['Close']) / df['Close'] * 100
    
    # Direction
    df['direction'] = 0  # Neutral
    df.loc[df['return_pct'] > 0.3, 'direction'] = 1   # Up
    df.loc[df['return_pct'] < -0.3, 'direction'] = -1  # Down
    
    return df

# %%
df_with_targets = create_targets(df_with_ta)
print(f"Target distribution:")
print(df_with_targets['direction'].value_counts())

# %% [markdown]
# ## 4. Merge with Sentiment Data

# %%
# Load sentiment data
df_sentiment = pd.read_csv('data/sentiment_scores.csv', parse_dates=['date'])
df_sentiment.set_index('date', inplace=True)

# Merge
df_merged = df_with_targets.join(df_sentiment, how='left')

# Fill missing sentiment with 0
df_merged[['news_sentiment', 'reddit_sentiment', 'combined_sentiment']] = \
    df_merged[['news_sentiment', 'reddit_sentiment', 'combined_sentiment']].fillna(0)

print(f"Merged dataset: {len(df_merged)} rows")

# %% [markdown]
# ## 5. Create Sequences for TCN

# %%
def create_sequences(df, seq_length=60):
    """
    Create sequences for time-series model.
    
    Returns:
    - price_sequences: (N, seq_length, 5) - OHLCV
    - technical_features: (N, 15) - Technical indicators
    - sentiment_features: (N, 3) - Sentiment scores
    - targets: (N,) - Return percentages
    """
    # Drop NaN rows
    df_clean = df.dropna()
    
    # Feature columns
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    tech_cols = [
        'rsi_14', 'macd', 'macd_signal', 'macd_hist', 
        'stoch_k', 'stoch_d', 'ema_5', 'ema_20', 'ema_50',
        'sma_20', 'adx', 'atr_14', 'bb_upper', 'bb_middle', 'bb_lower'
    ]
    sent_cols = ['news_sentiment', 'reddit_sentiment', 'combined_sentiment']
    
    # Normalize price data
    price_scaler = StandardScaler()
    price_data = price_scaler.fit_transform(df_clean[price_cols])
    
    # Normalize technical data
    tech_scaler = StandardScaler()
    tech_data = tech_scaler.fit_transform(df_clean[tech_cols].fillna(0))
    
    # Sentiment is already normalized
    sent_data = df_clean[sent_cols].values
    
    # Targets
    targets = df_clean['return_pct'].values
    
    # Create sequences
    price_sequences = []
    technical_features = []
    sentiment_features = []
    target_values = []
    
    for i in range(seq_length, len(df_clean) - 1):
        price_sequences.append(price_data[i-seq_length:i])
        technical_features.append(tech_data[i])
        sentiment_features.append(sent_data[i])
        target_values.append(targets[i])
    
    return (
        np.array(price_sequences),
        np.array(technical_features),
        np.array(sentiment_features),
        np.array(target_values),
        price_scaler,
        tech_scaler
    )

# %%
SEQ_LENGTH = 60

price_seq, tech_feat, sent_feat, targets, price_scaler, tech_scaler = create_sequences(
    df_merged, seq_length=SEQ_LENGTH
)

print(f"Dataset shapes:")
print(f"  Price sequences: {price_seq.shape}")
print(f"  Technical features: {tech_feat.shape}")
print(f"  Sentiment features: {sent_feat.shape}")
print(f"  Targets: {targets.shape}")

# %% [markdown]
# ## 6. Train/Validation/Test Split

# %%
# Time-based split (no shuffling for time series!)
train_size = int(len(targets) * 0.7)
val_size = int(len(targets) * 0.15)

X_price_train = price_seq[:train_size]
X_price_val = price_seq[train_size:train_size+val_size]
X_price_test = price_seq[train_size+val_size:]

X_tech_train = tech_feat[:train_size]
X_tech_val = tech_feat[train_size:train_size+val_size]
X_tech_test = tech_feat[train_size+val_size:]

X_sent_train = sent_feat[:train_size]
X_sent_val = sent_feat[train_size:train_size+val_size]
X_sent_test = sent_feat[train_size+val_size:]

y_train = targets[:train_size]
y_val = targets[train_size:train_size+val_size]
y_test = targets[train_size+val_size:]

print(f"Split sizes:")
print(f"  Train: {len(y_train)}")
print(f"  Validation: {len(y_val)}")
print(f"  Test: {len(y_test)}")

# %% [markdown]
# ## 7. Save Processed Data

# %%
# Save as numpy arrays
np.save('data/X_price_train.npy', X_price_train)
np.save('data/X_price_val.npy', X_price_val)
np.save('data/X_price_test.npy', X_price_test)

np.save('data/X_tech_train.npy', X_tech_train)
np.save('data/X_tech_val.npy', X_tech_val)
np.save('data/X_tech_test.npy', X_tech_test)

np.save('data/X_sent_train.npy', X_sent_train)
np.save('data/X_sent_val.npy', X_sent_val)
np.save('data/X_sent_test.npy', X_sent_test)

np.save('data/y_train.npy', y_train)
np.save('data/y_val.npy', y_val)
np.save('data/y_test.npy', y_test)

# Save scalers
with open('data/price_scaler.pkl', 'wb') as f:
    pickle.dump(price_scaler, f)
with open('data/tech_scaler.pkl', 'wb') as f:
    pickle.dump(tech_scaler, f)

print("\n✓ All data saved to 'data/' folder")
print("\nNext: Run 03_model_training.ipynb")

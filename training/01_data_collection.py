# %% [markdown]
# # NIFTY 50 Data Collection
# 
# This notebook collects historical OHLCV data, news, and sentiment for training.

# %% [markdown]
# ## Setup

# %%
!pip install yfinance pandas numpy feedparser praw transformers torch -q

# %%
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import feedparser
import json
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# %% [markdown]
# ## 1. Fetch NIFTY 50 OHLCV Data

# %%
# NIFTY 50 Index symbol
SYMBOL = "^NSEI"

# Fetch 3 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 3)

print(f"Fetching data from {start_date.date()} to {end_date.date()}")

df_nifty = yf.download(
    SYMBOL,
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    progress=True
)

print(f"Downloaded {len(df_nifty)} trading days")
df_nifty.head()

# %%
# Save OHLCV data
df_nifty.to_csv('data/nifty50_ohlcv.csv')
print(f"Saved to data/nifty50_ohlcv.csv")

# %% [markdown]
# ## 2. Fetch Top Constituents Data (for context)

# %%
# Top 10 NIFTY 50 constituents
CONSTITUENTS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "SBIN.NS", "BAJFINANCE.NS", "ITC.NS",
]

constituents_data = {}

for symbol in CONSTITUENTS:
    try:
        df = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        constituents_data[symbol] = df
        print(f"✓ {symbol}: {len(df)} days")
    except Exception as e:
        print(f"✗ {symbol}: {e}")

# %%
# Save constituents data
for symbol, df in constituents_data.items():
    filename = f"data/{symbol.replace('.NS', '')}_ohlcv.csv"
    df.to_csv(filename)

print(f"\nSaved {len(constituents_data)} constituent files")

# %% [markdown]
# ## 3. Fetch News Headlines (Google News RSS)

# %%
def fetch_news_historical():
    """
    Fetch recent news headlines from Google News RSS.
    Note: RSS only provides recent news, not historical.
    For training, we'll generate synthetic sentiment based on price movements.
    """
    rss_url = "https://news.google.com/rss/search?q=nifty+50+indian+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(rss_url)
    
    headlines = []
    for entry in feed.entries[:50]:
        headlines.append({
            'title': entry.get('title', ''),
            'published': entry.get('published', ''),
            'link': entry.get('link', ''),
        })
    
    return headlines

news = fetch_news_historical()
print(f"Fetched {len(news)} recent headlines")

# Save
with open('data/news_headlines.json', 'w') as f:
    json.dump(news, f, indent=2)

# %%
# Display sample headlines
for n in news[:5]:
    print(f"• {n['title'][:80]}...")

# %% [markdown]
# ## 4. Generate Synthetic Sentiment for Training
# 
# Since we don't have historical sentiment data, we'll generate synthetic sentiment
# based on price movements. This is a common approach for initial model training.

# %%
def generate_synthetic_sentiment(df_prices):
    """
    Generate synthetic sentiment scores based on price movements.
    
    Logic:
    - Positive returns → positive sentiment
    - Negative returns → negative sentiment
    - Add noise for realism
    """
    sentiment_data = []
    
    for i in range(1, len(df_prices)):
        current_date = df_prices.index[i]
        prev_close = df_prices['Close'].iloc[i-1]
        curr_close = df_prices['Close'].iloc[i]
        
        # Calculate return
        daily_return = (curr_close - prev_close) / prev_close
        
        # Convert to sentiment (-1 to 1) with noise
        base_sentiment = np.tanh(daily_return * 50)  # Scale and squash
        noise = np.random.normal(0, 0.1)
        sentiment = np.clip(base_sentiment + noise, -1, 1)
        
        sentiment_data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'news_sentiment': float(sentiment * 0.8 + np.random.normal(0, 0.1)),
            'reddit_sentiment': float(sentiment * 0.6 + np.random.normal(0, 0.15)),
            'combined_sentiment': float(sentiment),
        })
    
    return pd.DataFrame(sentiment_data)

# %%
df_sentiment = generate_synthetic_sentiment(df_nifty)
df_sentiment.to_csv('data/sentiment_scores.csv', index=False)
print(f"Generated {len(df_sentiment)} sentiment records")
df_sentiment.tail()

# %% [markdown]
# ## 5. Summary

# %%
print("\n" + "="*50)
print("DATA COLLECTION COMPLETE")
print("="*50)
print(f"\nFiles saved in 'data/' folder:")
print(f"  • nifty50_ohlcv.csv - {len(df_nifty)} trading days")
print(f"  • {len(constituents_data)} constituent OHLCV files")
print(f"  • news_headlines.json - {len(news)} headlines")
print(f"  • sentiment_scores.csv - {len(df_sentiment)} records")
print("\nNext: Run 02_feature_engineering.ipynb")

# %%
# Download data folder (for Colab)
from google.colab import files
import shutil

shutil.make_archive('nifty50_data', 'zip', 'data')
files.download('nifty50_data.zip')

"""
Real data fetching and population script for NIFTY 50.
Fetches real historical data from NSE and Yahoo Finance, calculates technical indicators,
and stores everything in the database.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, delete

from app.config import settings
from app.models.stock import StockPrice, TechnicalIndicator, SentimentScore
from app.models.prediction import Prediction


async def fetch_nifty_from_nse(days: int = 365):
    """
    Fetch NIFTY 50 historical data from NSE using nsepython.
    Returns DataFrame with OHLCV data.
    """
    try:
        from nsepython import index_history
        
        print(f"📊 Fetching NIFTY 50 data from NSE for last {days} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch NIFTY 50 index history
        df = index_history(
            symbol="NIFTY 50",
            start_date=start_date.strftime("%d-%m-%Y"),
            end_date=end_date.strftime("%d-%m-%Y")
        )
        
        if df is None or df.empty:
            print("⚠️  NSE returned empty data, will try Yahoo Finance...")
            return None
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'HistoricalDate': 'Date',
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOLUME': 'Volume'
        })
        
        # Try multiple date formats
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y')
            except:
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        
        df = df.set_index('Date')
        df = df.sort_index()
        
        print(f"✅ Fetched {len(df)} days of real NIFTY data from NSE")
        return df
        
    except Exception as e:
        print(f"❌ NSE fetch failed: {str(e)}")
        return None


async def fetch_from_csv(filepath: str = None):
    """
    Load NIFTY data from CSV file.
    CSV format should have: Date,Open,High,Low,Close,Volume
    """
    try:
        if filepath is None:
            filepath = Path(__file__).parent / "nifty_data.csv"
        
        if not Path(filepath).exists():
            print(f"⚠️  CSV file not found at {filepath}")
            return None
        
        print(f"📊 Loading NIFTY 50 data from CSV: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Try to parse date column
        date_col = None
        for col in ['Date', 'date', 'DATE', 'Timestamp']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            print("❌ Could not find date column in CSV")
            return None
        
        df['Date'] = pd.to_datetime(df[date_col])
        df = df.set_index('Date')
        df = df.sort_index()
        
        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                # Try case variations
                for c in df.columns:
                    if c.lower() == col.lower():
                        df[col] = df[c]
                        break
        
        # Add volume if missing
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        print(f"✅ Loaded {len(df)} days of data from CSV")
        return df
        
    except Exception as e:
        print(f"❌ CSV load failed: {str(e)}")
        return None


async def fetch_nifty_from_yahoo(days: int = 365):
    """
    Fallback: Fetch NIFTY data from Yahoo Finance with proper delays.
    """
    try:
        import yfinance as yf
        import time
        
        print(f"📊 Fetching NIFTY 50 data from Yahoo Finance (fallback)...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)
        
        # Add delay to avoid rate limiting
        time.sleep(2)
        
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        
        if df.empty:
            print("❌ Yahoo Finance returned empty data")
            return None
        
        print(f"✅ Fetched {len(df)} days of data from Yahoo Finance")
        return df
        
    except Exception as e:
        print(f"❌ Yahoo Finance fetch failed: {str(e)}")
        return None


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators using pandas-ta.
    """
    try:
        import pandas_ta as ta
        
        print("📈 Calculating technical indicators...")
        
        # RSI
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            df['MACD_hist'] = macd['MACDh_12_26_9']
        
        # Stochastic
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            df['STOCH_k'] = stoch['STOCHk_14_3_3']
            df['STOCH_d'] = stoch['STOCHd_14_3_3']
        
        # EMAs
        df['EMA_5'] = ta.ema(df['Close'], length=5)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        
        # SMA
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        
        # ADX
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        if adx is not None:
            df['ADX'] = adx['ADX_14']
        
        # ATR
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'])
        if bbands is not None:
            df['BB_upper'] = bbands['BBU_5_2.0']
            df['BB_middle'] = bbands['BBM_5_2.0']
            df['BB_lower'] = bbands['BBL_5_2.0']
        
        print("✅ Technical indicators calculated")
        return df
        
    except Exception as e:
        print(f"⚠️  Technical indicator calculation failed: {str(e)}")
        return df


async def store_stock_prices(df: pd.DataFrame, session: AsyncSession):
    """
    Store stock prices in database.
    """
    print("💾 Storing stock prices in database...")
    
    count = 0
    for idx, row in df.iterrows():
        try:
            price_date = idx.date() if hasattr(idx, 'date') else idx
            
            # Check if already exists
            result = await session.execute(
                select(StockPrice).where(
                    StockPrice.symbol == "^NSEI",
                    StockPrice.date == price_date
                )
            )
            if result.scalar_one_or_none():
                continue
            
            price = StockPrice(
                symbol="^NSEI",
                date=price_date,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume']) if pd.notna(row.get('Volume')) else 0,
                adj_close=float(row.get('Adj Close', row['Close']))
            )
            session.add(price)
            count += 1
            
        except Exception as e:
            print(f"⚠️  Error storing price for {idx}: {str(e)}")
            continue
    
    await session.commit()
    print(f"✅ Stored {count} stock price records")
    return count


async def store_technical_indicators(df: pd.DataFrame, session: AsyncSession):
    """
    Store technical indicators in database.
    """
    print("💾 Storing technical indicators in database...")
    
    count = 0
    for idx, row in df.iterrows():
        try:
            price_date = idx.date() if hasattr(idx, 'date') else idx
            
            # Skip if no technical data
            if pd.isna(row.get('RSI_14')):
                continue
            
            # Check if already exists
            result = await session.execute(
                select(TechnicalIndicator).where(
                    TechnicalIndicator.symbol == "^NSEI",
                    TechnicalIndicator.date == price_date
                )
            )
            if result.scalar_one_or_none():
                continue
            
            tech = TechnicalIndicator(
                symbol="^NSEI",
                date=price_date,
                rsi_14=float(row.get('RSI_14', 50)) if pd.notna(row.get('RSI_14')) else 50,
                macd=float(row.get('MACD', 0)) if pd.notna(row.get('MACD')) else 0,
                macd_signal=float(row.get('MACD_signal', 0)) if pd.notna(row.get('MACD_signal')) else 0,
                macd_hist=float(row.get('MACD_hist', 0)) if pd.notna(row.get('MACD_hist')) else 0,
                stoch_k=float(row.get('STOCH_k', 50)) if pd.notna(row.get('STOCH_k')) else 50,
                stoch_d=float(row.get('STOCH_d', 50)) if pd.notna(row.get('STOCH_d')) else 50,
                ema_5=float(row.get('EMA_5', row['Close'])) if pd.notna(row.get('EMA_5')) else float(row['Close']),
                ema_20=float(row.get('EMA_20', row['Close'])) if pd.notna(row.get('EMA_20')) else float(row['Close']),
                ema_50=float(row.get('EMA_50', row['Close'])) if pd.notna(row.get('EMA_50')) else float(row['Close']),
                sma_20=float(row.get('SMA_20', row['Close'])) if pd.notna(row.get('SMA_20')) else float(row['Close']),
                adx=float(row.get('ADX', 25)) if pd.notna(row.get('ADX')) else 25,
                atr_14=float(row.get('ATR_14', 0)) if pd.notna(row.get('ATR_14')) else 0,
                bb_upper=float(row.get('BB_upper', row['Close'])) if pd.notna(row.get('BB_upper')) else float(row['Close']),
                bb_middle=float(row.get('BB_middle', row['Close'])) if pd.notna(row.get('BB_middle')) else float(row['Close']),
                bb_lower=float(row.get('BB_lower', row['Close'])) if pd.notna(row.get('BB_lower')) else float(row['Close']),
            )
            session.add(tech)
            count += 1
            
        except Exception as e:
            print(f"⚠️  Error storing technical indicators for {idx}: {str(e)}")
            continue
    
    await session.commit()
    print(f"✅ Stored {count} technical indicator records")
    return count


async def store_basic_sentiment(session: AsyncSession):
    """
    Store neutral sentiment scores as baseline.
    Note: Real sentiment would require news/social media APIs.
    """
    print("💾 Storing baseline sentiment scores...")
    
    # Add neutral sentiment for recent days
    count = 0
    for i in range(30):
        sentiment_date = date.today() - timedelta(days=i)
        
        # Check if already exists
        result = await session.execute(
            select(SentimentScore).where(
                SentimentScore.symbol == "^NSEI",
                SentimentScore.date == sentiment_date
            )
        )
        if result.scalar_one_or_none():
            continue
        
        sentiment = SentimentScore(
            symbol="^NSEI",
            date=sentiment_date,
            news_sentiment=0.0,  # Neutral
            reddit_sentiment=0.0,  # Neutral
            combined_sentiment=0.0,  # Neutral
        )
        session.add(sentiment)
        count += 1
    
    await session.commit()
    print(f"✅ Stored {count} sentiment records (baseline neutral)")
    return count


async def main():
    """
    Main function to fetch and populate database with real data.
    """
    print("=" * 60)
    print("🚀 NIFTY 50 Real Data Fetching Script")
    print("=" * 60)
    
    # Create async engine
    engine = create_async_engine(settings.database_url, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session_maker() as session:
        try:
            # Step 1: Fetch historical data
            print("\n📥 Step 1: Fetching historical data...")
            df = await fetch_nifty_from_nse(days=365)
            
            if df is None or df.empty:
                print("⚠️  NSE failed, trying Yahoo Finance...")
                df = await fetch_nifty_from_yahoo(days=365)
            
            if df is None or df.empty:
                print("❌ Could not fetch data from any source. Exiting.")
                return
            
            # Step 2: Calculate technical indicators
            print("\n📊 Step 2: Calculating technical indicators...")
            df = calculate_technical_indicators(df)
            
            # Step 3: Store stock prices
            print("\n💾 Step 3: Storing stock prices...")
            price_count = await store_stock_prices(df, session)
            
            # Step 4: Store technical indicators
            print("\n💾 Step 4: Storing technical indicators...")
            tech_count = await store_technical_indicators(df, session)
            
            # Step 5: Store sentiment baselines
            print("\n💾 Step 5: Storing sentiment baselines...")
            sentiment_count = await store_basic_sentiment(session)
            
            # Summary
            print("\n" + "=" * 60)
            print("✅ DATA POPULATION COMPLETE!")
            print("=" * 60)
            print(f"📊 Stock Prices:        {price_count} records")
            print(f"📈 Technical Indicators: {tech_count} records")
            print(f"💬 Sentiment Scores:    {sentiment_count} records")
            print("=" * 60)
            print("\n✅ Database is now ready for predictions!")
            print("🚀 You can now generate predictions using the trained model.")
            
        except Exception as e:
            print(f"\n❌ Error during data population: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

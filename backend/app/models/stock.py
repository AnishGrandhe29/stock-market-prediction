"""
Stock and market data models.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, Index

from app.core.database import Base


class StockPrice(Base):
    """OHLCV price data for stocks/indices."""
    
    __tablename__ = "stock_prices"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)  # e.g., ^NSEI, RELIANCE.NS
    date = Column(Date, index=True, nullable=False)
    
    # OHLCV
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    
    # Adjusted close
    adj_close = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date', unique=True),
    )
    
    def __repr__(self):
        return f"<StockPrice {self.symbol} {self.date}>"


class TechnicalIndicator(Base):
    """Computed technical indicators for stocks."""
    
    __tablename__ = "technical_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    
    # Momentum
    rsi_14 = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_hist = Column(Float, nullable=True)
    stoch_k = Column(Float, nullable=True)
    stoch_d = Column(Float, nullable=True)
    
    # Trend
    ema_5 = Column(Float, nullable=True)
    ema_20 = Column(Float, nullable=True)
    ema_50 = Column(Float, nullable=True)
    sma_20 = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    
    # Volatility
    atr_14 = Column(Float, nullable=True)
    bb_upper = Column(Float, nullable=True)
    bb_middle = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    
    # Volume
    obv = Column(Float, nullable=True)
    volume_sma = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_tech_symbol_date', 'symbol', 'date', unique=True),
    )
    
    def __repr__(self):
        return f"<TechnicalIndicator {self.symbol} {self.date}>"


class SentimentScore(Base):
    """Aggregated sentiment scores from news and social media."""
    
    __tablename__ = "sentiment_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    
    # Sentiment scores (-1 to 1)
    news_sentiment = Column(Float, nullable=True)
    reddit_sentiment = Column(Float, nullable=True)
    combined_sentiment = Column(Float, nullable=True)
    
    # Metadata
    news_count = Column(Integer, default=0)
    reddit_count = Column(Integer, default=0)
    
    # Raw text for XAI
    top_headlines = Column(Text, nullable=True)  # JSON array
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_sentiment_symbol_date', 'symbol', 'date', unique=True),
    )
    
    def __repr__(self):
        return f"<SentimentScore {self.symbol} {self.date}>"

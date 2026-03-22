"""
Pydantic schemas for API request/response validation.
"""
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field


# ============ Auth Schemas ============

class UserCreate(BaseModel):
    """Schema for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Refresh token request."""
    refresh_token: str


class GoogleAuthCallback(BaseModel):
    """Google OAuth callback data."""
    code: str
    state: Optional[str] = None


# ============ User Schemas ============

class UserResponse(BaseModel):
    """User data response."""
    id: int
    email: str
    full_name: Optional[str]
    avatar_url: Optional[str]
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """User profile update."""
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None


# ============ Stock Schemas ============

class StockPriceResponse(BaseModel):
    """OHLCV price data."""
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]
    
    class Config:
        from_attributes = True


class TechnicalIndicatorResponse(BaseModel):
    """Technical indicator data."""
    symbol: str
    date: date
    rsi_14: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    ema_20: Optional[float]
    ema_50: Optional[float]
    atr_14: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    
    class Config:
        from_attributes = True


class SentimentResponse(BaseModel):
    """Sentiment score data."""
    symbol: str
    date: date
    news_sentiment: Optional[float]
    reddit_sentiment: Optional[float]
    combined_sentiment: Optional[float]
    news_count: int
    reddit_count: int
    
    class Config:
        from_attributes = True


# ============ Prediction Schemas ============

class HorizonDetail(BaseModel):
    point: Optional[float]
    direction: Optional[str]
    interval: Optional[List[float]]
    uncertainty: Optional[float]

class Horizons(BaseModel):
    h1d: Optional[HorizonDetail] = Field(None, alias="1d")
    h5d: Optional[HorizonDetail] = Field(None, alias="5d")
    h20d: Optional[HorizonDetail] = Field(None, alias="20d")
    h60d: Optional[HorizonDetail] = Field(None, alias="60d")

class PredictionResponse(BaseModel):
    """Model prediction with XAI data."""
    id: int
    symbol: str
    prediction_date: date
    target_date: date
    predicted_open: float
    predicted_change_pct: float
    
    # ACMI++ Multi-Horizon outputs
    horizons: Optional[Horizons] = None
    
    # Risk Metrics
    volatility_forecast: Optional[float] = None
    crash_probability: Optional[float] = None

    # Market Regime
    market_regime: Optional[str] = None
    regime_probabilities: Optional[Dict[str, float]] = None

    # Legacy fields
    uncertainty_score: Optional[float] = None
    confidence_level: Optional[str] = None
    predicted_direction: Optional[str] = None
    direction_probability: Optional[float] = None
    trend: Optional[str]  # Bullish / Bearish / Neutral
    signal: Optional[str]  # BUY / HOLD / SELL
    confidence_score: Optional[float]  # 0-1 numerical confidence
    
    # XAI
    shap_values: Optional[Dict[str, Any]] = None
    modality_weights: Optional[Dict[str, float]] = None
    top_features: Optional[List[Dict[str, Any]]] = None
    
    # Actual (if available)
    actual_close: Optional[float] = None
    prediction_error: Optional[float] = None
    
    class Config:
        from_attributes = True
        populate_by_name = True


class PredictionRequest(BaseModel):
    """Request for new prediction."""
    symbol: str = "^NSEI"
    target_date: Optional[date] = None  # None = next trading day


# ============ User Feature Schemas ============

class NoteCreate(BaseModel):
    """Create a new note."""
    title: Optional[str] = None
    content: str
    symbol: Optional[str] = None
    tags: Optional[str] = None


class NoteUpdate(BaseModel):
    """Update a note."""
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[str] = None


class NoteResponse(BaseModel):
    """Note response."""
    id: int
    title: Optional[str]
    content: str
    symbol: Optional[str]
    tags: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class WatchlistItemCreate(BaseModel):
    """Add to watchlist."""
    symbol: str
    display_name: Optional[str] = None


class WatchlistItemResponse(BaseModel):
    """Watchlist item response."""
    id: int
    symbol: str
    display_name: Optional[str]
    sort_order: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class AlertCreate(BaseModel):
    """Create price alert."""
    symbol: str
    alert_type: str  # price_above, price_below, change_pct
    target_value: float


class AlertResponse(BaseModel):
    """Alert response."""
    id: int
    symbol: str
    alert_type: str
    target_value: float
    is_active: bool
    is_triggered: bool
    triggered_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

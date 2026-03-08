"""
ML Service for model inference.
Loads trained model and generates predictions with XAI explanations.
"""
import time
import torch
import numpy as np
import joblib
from datetime import date, timedelta
from typing import Optional, Dict, Tuple
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.config import settings
from app.models.stock import StockPrice, TechnicalIndicator, SentimentScore
from app.models.prediction import Prediction
from app.ml.models.fusion import NIFTY50Predictor
from app.ml.models.technical import TECHNICAL_FEATURES
from app.ml.explainability.shap_explainer import MultimodalSHAP


# Global model instance
_model: Optional[NIFTY50Predictor] = None
_explainer: Optional[MultimodalSHAP] = None
_price_scaler = None
_tech_scaler = None

# Prediction cache: key -> (timestamp, result_dict)
_prediction_cache: Dict[str, Tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

# ── Realistic bounds for NIFTY 50 daily movement ──
MAX_DAILY_RETURN_PCT = 2.0   # ±2% for point prediction
MAX_QUANTILE_RANGE_PCT = 3.0 # ±3% for uncertainty band


def load_model() -> NIFTY50Predictor:
    """Load the trained model from disk."""
    global _model, _explainer
    
    if _model is not None:
        return _model
    
    model_path = Path(settings.model_path)
    
    # Initialize model
    _model = NIFTY50Predictor(
        price_seq_len=60,
        price_features=5,
        sentiment_features=3,
        technical_features=6,  # Updated to match retrained model
        embedding_dim=128,
        dropout=0.2
    )
    
    # Load weights if available
    if model_path.exists():
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            # Use strict=False to handle architecture mismatches
            result = _model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                print(f"Loaded model with {len(result.missing_keys)} missing keys")
            if result.unexpected_keys:
                print(f"Loaded model with {len(result.unexpected_keys)} unexpected keys")
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Using random initialization.")
    else:
        print(f"Model not found at {model_path}. Using random initialization.")
    
    _model.eval()

    # Load scalers
    global _price_scaler, _tech_scaler
    price_scaler_path = Path(settings.price_scaler_path)
    tech_scaler_path = Path(settings.tech_scaler_path)

    if price_scaler_path.exists():
        try:
            _price_scaler = joblib.load(price_scaler_path)
            print(f"Loaded price scaler from {price_scaler_path}")
        except Exception as e:
            print(f"Error loading price scaler: {e}")

    if tech_scaler_path.exists():
        try:
            _tech_scaler = joblib.load(tech_scaler_path)
            print(f"Loaded tech scaler from {tech_scaler_path}")
        except Exception as e:
            print(f"Error loading tech scaler: {e}")
    
    # Initialize explainer
    _explainer = MultimodalSHAP(
        model=_model,
        feature_names={
            "technical": TECHNICAL_FEATURES,
            "sentiment": ["news_sentiment", "reddit_sentiment", "combined_sentiment"],
            "price": ["open", "high", "low", "close", "volume"],
        }
    )
    
    return _model


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))


def _classify_trend(return_pct: float) -> str:
    """Classify trend based on predicted return."""
    if return_pct > 0.3:
        return "Bullish"
    elif return_pct < -0.3:
        return "Bearish"
    return "Neutral"


def _classify_signal(return_pct: float) -> str:
    """Generate trading signal based on predicted return."""
    if return_pct > 0.5:
        return "BUY"
    elif return_pct < -0.5:
        return "SELL"
    return "HOLD"


def _compute_confidence(uncertainty: float, direction_prob: float) -> tuple:
    """
    Compute a robust confidence score from model uncertainty and direction probability.

    Returns (confidence_score 0-1, confidence_level str).
    """
    # Normalise uncertainty to 0-1 where lower uncertainty = higher confidence
    uncertainty_confidence = max(0.0, 1.0 - abs(uncertainty))

    # Blend: 60% direction probability, 40% uncertainty-based confidence
    score = 0.6 * direction_prob + 0.4 * uncertainty_confidence
    score = max(0.0, min(1.0, score))  # clamp to [0, 1]

    if score >= 0.70:
        level = "high"
    elif score >= 0.45:
        level = "medium"
    else:
        level = "low"

    return score, level


async def get_prediction(
    symbol: str,
    target_date: Optional[date],
    db: AsyncSession
) -> Prediction:
    """
    Generate a prediction for the given symbol.

    Pipeline:
      1. Run model inference (cached for 5 min)
      2. Clamp predicted return to ±MAX_DAILY_RETURN_PCT
      3. Derive predicted_price = current_price * (1 + clamped_return / 100)
      4. Classify trend & signal
      5. Compute confidence from uncertainty + direction probability
      6. Persist and return
    """
    # Load model
    model = load_model()

    # Determine target date
    if target_date is None:
        target_date = get_next_trading_day()

    prediction_date = date.today()

    # ── Check cache ──
    cache_key = f"{symbol}:{target_date}"
    now = time.time()
    if cache_key in _prediction_cache:
        cached_ts, cached_pred = _prediction_cache[cache_key]
        if now - cached_ts < _CACHE_TTL_SECONDS:
            # Return a fresh DB row from the cached numbers
            return cached_pred

    # ── Fetch input data ──
    price_data = await fetch_price_features(symbol, db)
    sentiment_data = await fetch_sentiment_features(symbol, db)
    technical_data = await fetch_technical_features(symbol, db)

    # Get latest price for reference
    latest_price = await get_latest_close(symbol, db)

    # Convert to tensors
    price_tensor = torch.tensor(price_data, dtype=torch.float32).unsqueeze(0)
    sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).unsqueeze(0)
    technical_tensor = torch.tensor(technical_data, dtype=torch.float32).unsqueeze(0)

    # ── Run inference ──
    model.eval()
    with torch.no_grad():
        output = model(price_tensor, sentiment_tensor, technical_tensor)

    # ── Extract & clamp predictions ──
    raw_point = output["point_prediction"].item()
    raw_q5 = output["quantile_5"].item()
    raw_q50 = output["quantile_50"].item()
    raw_q95 = output["quantile_95"].item()
    uncertainty = output["uncertainty"].item()

    # Clamp to realistic daily movement
    point_pred = _clamp(raw_point, -MAX_DAILY_RETURN_PCT, MAX_DAILY_RETURN_PCT)
    quantile_5 = _clamp(raw_q5, -MAX_QUANTILE_RANGE_PCT, MAX_QUANTILE_RANGE_PCT)
    quantile_50 = _clamp(raw_q50, -MAX_DAILY_RETURN_PCT, MAX_DAILY_RETURN_PCT)
    quantile_95 = _clamp(raw_q95, -MAX_QUANTILE_RANGE_PCT, MAX_QUANTILE_RANGE_PCT)

    # Ensure quantile ordering: q5 <= q50 <= q95
    quantile_5 = min(quantile_5, quantile_50)
    quantile_95 = max(quantile_95, quantile_50)

    # ── Derived values ──
    predicted_open = latest_price * (1 + point_pred / 100)

    # Direction from probabilities
    direction_probs = output["direction_probs"][0].numpy()
    direction_idx = np.argmax(direction_probs)
    direction = ["down", "neutral", "up"][direction_idx]
    direction_prob = float(direction_probs[direction_idx])

    # Trend & signal
    trend = _classify_trend(point_pred)
    signal = _classify_signal(point_pred)

    # Confidence
    confidence_score, confidence_level = _compute_confidence(uncertainty, direction_prob)

    # ── XAI explanations ──
    xai_data = _explainer.explain(price_tensor, sentiment_tensor, technical_tensor)

    # ── Create prediction record ──
    prediction = Prediction(
        symbol=symbol,
        prediction_date=prediction_date,
        target_date=target_date,
        predicted_open=round(predicted_open, 2),
        predicted_change_pct=round(point_pred, 4),
        quantile_5=round(latest_price * (1 + quantile_5 / 100), 2),
        quantile_50=round(latest_price * (1 + quantile_50 / 100), 2),
        quantile_95=round(latest_price * (1 + quantile_95 / 100), 2),
        uncertainty_score=round(uncertainty, 4),
        confidence_level=confidence_level,
        confidence_score=round(confidence_score, 4),
        predicted_direction=direction,
        direction_probability=round(direction_prob, 4),
        trend=trend,
        signal=signal,
        shap_values=xai_data["shap_values"],
        modality_weights=xai_data["modality_weights"],
        top_features=xai_data["top_features"],
        input_features={
            "price_shape": list(price_data.shape),
            "latest_close": latest_price,
            "sentiment": sentiment_data.tolist(),
            "raw_point_pred": raw_point,
            "clamped_point_pred": point_pred,
        },
    )

    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)

    # ── Cache the result ──
    _prediction_cache[cache_key] = (now, prediction)

    return prediction


async def fetch_price_features(symbol: str, db: AsyncSession) -> np.ndarray:
    """Fetch last 60 days of OHLCV data."""
    start_date = date.today() - timedelta(days=90)
    
    result = await db.execute(
        select(StockPrice)
        .where(
            and_(
                StockPrice.symbol == symbol,
                StockPrice.date >= start_date
            )
        )
        .order_by(StockPrice.date.desc())
        .limit(60)
    )
    
    prices = result.scalars().all()
    
    if len(prices) < 60:
        # Pad with zeros or repeat last value
        prices = list(prices) + [prices[-1]] * (60 - len(prices)) if prices else []
    
    # Reverse to chronological order
    prices = list(reversed(prices))
    
    # Convert to numpy array (seq_len, 5)
    data = np.array([
        [p.open, p.high, p.low, p.close, p.volume or 0]
        for p in prices[-60:]
    ], dtype=np.float32)
    
    # Normalize using loaded scaler if available
    if _price_scaler is not None and data.shape[0] > 0:
        try:
            # Scaler expects (N, features)
            data = _price_scaler.transform(data).astype(np.float32)
        except Exception as e:
            print(f"Error applying price scaler: {e}")
            # Fallback to manual normalization
            close_mean = data[:, 3].mean()
            close_std = data[:, 3].std() + 1e-8
            data = (data - close_mean) / close_std
    elif data.shape[0] > 0:
        # Fallback to manual normalization
        close_mean = data[:, 3].mean()
        close_std = data[:, 3].std() + 1e-8
        data = (data - close_mean) / close_std
    
    return data


async def fetch_sentiment_features(symbol: str, db: AsyncSession) -> np.ndarray:
    """Fetch latest sentiment scores."""
    result = await db.execute(
        select(SentimentScore)
        .where(SentimentScore.symbol == symbol)
        .order_by(SentimentScore.date.desc())
        .limit(1)
    )
    
    sentiment = result.scalar_one_or_none()
    
    if sentiment:
        return np.array([
            sentiment.news_sentiment or 0,
            sentiment.reddit_sentiment or 0,
            sentiment.combined_sentiment or 0,
        ], dtype=np.float32)
    
    return np.zeros(3, dtype=np.float32)


async def fetch_technical_features(symbol: str, db: AsyncSession) -> np.ndarray:
    """Fetch latest technical indicators (6 features to match retrained model)."""
    result = await db.execute(
        select(TechnicalIndicator)
        .where(TechnicalIndicator.symbol == symbol)
        .order_by(TechnicalIndicator.date.desc())
        .limit(1)
    )
    
    tech = result.scalar_one_or_none()
    
    if tech:
        # Return 6 features to match model architecture
        data = np.array([
            tech.rsi_14 or 50,
            tech.macd or 0,
            tech.macd_signal or 0,
            tech.stoch_k or 50,
            tech.adx or 25,
            tech.atr_14 or 0,
        ], dtype=np.float32)
        
        if _tech_scaler is not None:
             try:
                data = _tech_scaler.transform(data.reshape(1, -1)).flatten().astype(np.float32)
             except Exception as e:
                print(f"Error applying tech scaler: {e}")
        
        return data
    else:
        # Fallback: Default "neutral" values (6 features)
        print(f"[WARNING] Missing technical indicators for {symbol}, using defaults")
        latest_price = await get_latest_close(symbol, db)
        
        data = np.array([
            50.0,  # RSI (neutral)
            0.0,   # MACD (neutral)
            0.0,   # MACD Signal
            50.0,  # Stoch K (neutral)
            25.0,  # ADX (weak trend)
            latest_price * 0.01 if latest_price else 100,  # ATR
        ], dtype=np.float32)
        
        if _tech_scaler is not None:
             try:
                data = _tech_scaler.transform(data.reshape(1, -1)).flatten().astype(np.float32)
             except Exception as e:
                print(f"Error applying tech scaler: {e}")
                
        return data


async def get_latest_close(symbol: str, db: AsyncSession) -> float:
    """Get the latest closing price from database."""
    result = await db.execute(
        select(StockPrice)
        .where(StockPrice.symbol == symbol)
        .order_by(StockPrice.date.desc())
        .limit(1)
    )
    
    price = result.scalar_one_or_none()
    if price:
        return price.close
    
    # No fake data - raise exception if no real data exists
    raise ValueError(f"No price data available for {symbol}. Please run data population first.")


def get_next_trading_day() -> date:
    """Get the next trading day (skip weekends)."""
    today = date.today()
    next_day = today + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    
    return next_day

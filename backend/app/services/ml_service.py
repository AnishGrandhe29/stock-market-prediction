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
from app.ml.models.acmi import ACMIPredictor


# Global model instance
_model: Optional[ACMIPredictor] = None

# Prediction cache: key -> (timestamp, result_dict)
_prediction_cache: Dict[str, Tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

# ── Realistic bounds for NIFTY 50 daily movement ──
MAX_DAILY_RETURN_PCT = 2.0   # ±2% for point prediction
MAX_QUANTILE_RANGE_PCT = 3.0 # ±3% for uncertainty band


def load_model() -> ACMIPredictor:
    """Load the trained model from disk."""
    global _model
    
    if _model is not None:
        return _model
    
    model_path = Path("a:/Project/stock-market-prediction/models/acmi_best_model.pt")
    scaler_path = Path("a:/Project/stock-market-prediction/models/scalers.pkl")
    ensemble_path = Path("a:/Project/stock-market-prediction/models/acmi_ensemble.pt")
    
    # Initialize robust model wrapper
    try:
        _model = ACMIPredictor(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            ensemble_path=str(ensemble_path) if ensemble_path.exists() else None,
            device="cpu"  # Force CPU for API inference
        )
        print(f"Loaded ACMI++ predictor from {model_path}")
    except Exception as e:
        print(f"Error loading ACMI++ model: {e}")
        _model = None
    
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
    db: AsyncSession,
    use_gift: bool = True,
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

    Parameters
    ----------
    use_gift : If True (default), inject GIFT NIFTY overnight features into
               the model's overnight encoder.  Set False for ablation studies
               or when GIFT data is unavailable.
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
            return cached_pred

    # ── Run inference using ACMIPredictor ──
    try:
        # ACMIPredictor handles internal historical feature download and engineering.
        # use_gift=True injects the GIFT NIFTY overnight signal.
        result_dict = model.predict(symbol, use_gift=use_gift)
    except Exception as e:
        print(f"Error in ACMIPredictor for {symbol}: {e}")
        raise ValueError(f"Could not generate prediction: {e}")

    latest_price = result_dict.get("latest_price", 1.0)
    horizon_1d   = result_dict["horizons"]["1d"]
    horizon_5d   = result_dict["horizons"]["5d"]
    horizon_20d  = result_dict["horizons"]["20d"]
    horizon_60d  = result_dict["horizons"]["60d"]

    # 1d Base outputs
    point_pred  = horizon_1d["point"]
    uncertainty = horizon_1d["uncertainty"]

    # ── Gap-based open prediction (NEW) ──
    # If the model ran the gap_head successfully, use that directly.
    # Otherwise fall back to the legacy return-based formula.
    gap_pred       = result_dict.get("gap_pred", None)
    predicted_open = result_dict.get("predicted_open", None)
    if predicted_open is None:
        predicted_open = latest_price * (1 + point_pred)
    
    direction = "up" if horizon_1d["direction"] == "UP" else "down"
    # Basic direction probability mapped from uncertainty
    direction_prob = 1.0 - min(uncertainty * 10, 0.5)
    
    trend = _classify_trend(point_pred * 100)
    signal = _classify_signal(point_pred * 100)
    confidence_score, confidence_level = _compute_confidence(uncertainty, direction_prob)

    # ── Create prediction record ──
    prediction = Prediction(
        symbol=symbol,
        prediction_date=prediction_date,
        target_date=target_date,
        predicted_open=round(predicted_open, 2),
        predicted_change_pct=round(point_pred * 100, 4),

        horizon_1d_point=round(horizon_1d["point"] * 100, 4),
        horizon_1d_interval=horizon_1d["interval"],
        horizon_5d_point=round(horizon_5d["point"] * 100, 4),
        horizon_5d_interval=horizon_5d["interval"],
        horizon_20d_point=round(horizon_20d["point"] * 100, 4),
        horizon_20d_interval=horizon_20d["interval"],
        horizon_60d_point=round(horizon_60d["point"] * 100, 4),
        horizon_60d_interval=horizon_60d["interval"],

        volatility_forecast=round(result_dict["vol_fcast"], 4),
        crash_probability=round(result_dict["crash_prob"], 4),

        market_regime=result_dict["regime"],
        regime_probabilities=result_dict["regime_p"],

        uncertainty_score=round(uncertainty, 4),
        confidence_level=confidence_level,
        confidence_score=round(confidence_score, 4),
        predicted_direction=direction,
        direction_probability=round(direction_prob, 4),
        trend=trend,
        signal=signal,

        # Disable XAI temporarily for production speed
        shap_values=None,
        modality_weights=None,
        top_features=None,

        input_features={
            "latest_close"       : latest_price,
            "acmi_point_pred"    : point_pred,
            # NEW: expose GIFT NIFTY context in input features
            "gap_pred_pts"       : round(gap_pred, 2) if gap_pred is not None else None,
            "gap_pred_pct"       : round(result_dict.get("gap_pred_pct", 0.0) * 100, 4),
            "gift_features"      : result_dict.get("gift_features"),
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

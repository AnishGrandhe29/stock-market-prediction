"""
ML Service for model inference.
Loads trained model and generates predictions with XAI explanations.
"""
import torch
import numpy as np
import joblib
from datetime import date, timedelta
from typing import Optional, Dict
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
        technical_features=15,
        embedding_dim=128,
        dropout=0.2
    )
    
    # Load weights if available
    if model_path.exists():
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            _model.load_state_dict(state_dict)
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


async def get_prediction(
    symbol: str,
    target_date: Optional[date],
    db: AsyncSession
) -> Prediction:
    """
    Generate a prediction for the given symbol.
    
    Args:
        symbol: Stock symbol (default ^NSEI)
        target_date: Date to predict (None = next trading day)
        db: Database session
        
    Returns:
        Prediction object with XAI data
    """
    # Load model
    model = load_model()
    
    # Determine target date
    if target_date is None:
        target_date = get_next_trading_day()
    
    prediction_date = date.today()
    
    # Fetch input data
    price_data = await fetch_price_features(symbol, db)
    sentiment_data = await fetch_sentiment_features(symbol, db)
    technical_data = await fetch_technical_features(symbol, db)
    
    # Get latest price for reference
    latest_price = await get_latest_close(symbol, db)
    
    # Convert to tensors
    price_tensor = torch.tensor(price_data, dtype=torch.float32).unsqueeze(0)
    sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).unsqueeze(0)
    technical_tensor = torch.tensor(technical_data, dtype=torch.float32).unsqueeze(0)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(price_tensor, sentiment_tensor, technical_tensor)
    
    # Extract predictions
    point_pred = output["point_prediction"].item()
    quantile_5 = output["quantile_5"].item()
    quantile_50 = output["quantile_50"].item()
    quantile_95 = output["quantile_95"].item()
    uncertainty = output["uncertainty"].item()
    
    # Direction from probabilities
    direction_probs = output["direction_probs"][0].numpy()
    direction_idx = np.argmax(direction_probs)
    direction = ["down", "neutral", "up"][direction_idx]
    direction_prob = float(direction_probs[direction_idx])
    
    # Calculate predicted close price
    predicted_close = latest_price * (1 + point_pred / 100)
    
    # Compute XAI explanations
    xai_data = _explainer.explain(price_tensor, sentiment_tensor, technical_tensor)
    
    # Determine confidence level
    if uncertainty < 0.5:
        confidence_level = "high"
    elif uncertainty < 1.0:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # Create prediction record
    prediction = Prediction(
        symbol=symbol,
        prediction_date=prediction_date,
        target_date=target_date,
        predicted_close=predicted_close,
        predicted_change_pct=point_pred,
        quantile_5=latest_price * (1 + quantile_5 / 100),
        quantile_50=latest_price * (1 + quantile_50 / 100),
        quantile_95=latest_price * (1 + quantile_95 / 100),
        uncertainty_score=uncertainty,
        confidence_level=confidence_level,
        predicted_direction=direction,
        direction_probability=direction_prob,
        shap_values=xai_data["shap_values"],
        modality_weights=xai_data["modality_weights"],
        top_features=xai_data["top_features"],
        input_features={
            "price_shape": list(price_data.shape),
            "latest_close": latest_price,
            "sentiment": sentiment_data.tolist(),
        },
    )
    
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)
    
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
    """Fetch latest technical indicators."""
    result = await db.execute(
        select(TechnicalIndicator)
        .where(TechnicalIndicator.symbol == symbol)
        .order_by(TechnicalIndicator.date.desc())
        .limit(1)
    )
    
    tech = result.scalar_one_or_none()
    
    if tech:
        data = np.array([
            tech.rsi_14 or 50,
            tech.macd or 0,
            tech.macd_signal or 0,
            tech.macd_hist or 0,
            tech.stoch_k or 50,
            tech.stoch_d or 50,
            tech.ema_5 or 0,
            tech.ema_20 or 0,
            tech.ema_50 or 0,
            tech.sma_20 or 0,
            tech.adx or 25,
            tech.atr_14 or 0,
            tech.bb_upper or 0,
            tech.bb_middle or 0,
            tech.bb_lower or 0,
        ], dtype=np.float32)

        if _tech_scaler is not None:
             try:
                data = _tech_scaler.transform(data.reshape(1, -1)).flatten().astype(np.float32)
             except Exception as e:
                print(f"Error applying tech scaler: {e}")
        
        return data
    
    return np.zeros(15, dtype=np.float32)


async def get_latest_close(symbol: str, db: AsyncSession) -> float:
    """Get the latest closing price."""
    result = await db.execute(
        select(StockPrice)
        .where(StockPrice.symbol == symbol)
        .order_by(StockPrice.date.desc())
        .limit(1)
    )
    
    price = result.scalar_one_or_none()
    return price.close if price else 22000.0  # Default NIFTY level


def get_next_trading_day() -> date:
    """Get the next trading day (skip weekends)."""
    today = date.today()
    next_day = today + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    
    return next_day

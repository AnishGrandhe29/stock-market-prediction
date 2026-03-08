"""
Prediction and XAI explanation models.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, DateTime, Date, Text, JSON

from app.core.database import Base


class Prediction(Base):
    """Model predictions with uncertainty and XAI data."""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # What we're predicting
    symbol = Column(Text, default="^NSEI", nullable=False)
    prediction_date = Column(Date, index=True, nullable=False)  # Date prediction was made
    target_date = Column(Date, index=True, nullable=False)  # Date being predicted
    
    # Predictions
    predicted_open = Column(Float, nullable=False)
    predicted_change_pct = Column(Float, nullable=False)  # % change from current
    
    # Quantiles for uncertainty
    quantile_5 = Column(Float, nullable=True)  # 5th percentile
    quantile_50 = Column(Float, nullable=True)  # Median
    quantile_95 = Column(Float, nullable=True)  # 95th percentile
    
    # Confidence
    uncertainty_score = Column(Float, nullable=True)  # 0-1
    confidence_level = Column(Text, nullable=True)  # low/medium/high
    
    # Direction
    predicted_direction = Column(Text, nullable=True)  # up/down/neutral
    direction_probability = Column(Float, nullable=True)  # 0-1
    
    # Trend & Signal
    trend = Column(Text, nullable=True)  # Bullish/Bearish/Neutral
    signal = Column(Text, nullable=True)  # BUY/HOLD/SELL
    confidence_score = Column(Float, nullable=True)  # 0-1 numerical confidence
    
    # Actual values (filled after market close)
    actual_close = Column(Float, nullable=True)
    actual_change_pct = Column(Float, nullable=True)
    prediction_error = Column(Float, nullable=True)
    
    # XAI Data (stored as JSON)
    shap_values = Column(JSON, nullable=True)  # Feature importance
    attention_weights = Column(JSON, nullable=True)  # For sentiment
    modality_weights = Column(JSON, nullable=True)  # Fusion gate weights
    top_features = Column(JSON, nullable=True)  # Top contributing features
    
    # Input snapshot (for reproducibility)
    input_features = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Prediction {self.target_date} -> {self.predicted_open}>"


class PredictionAccuracy(Base):
    """Aggregated prediction accuracy metrics."""
    
    __tablename__ = "prediction_accuracy"
    
    id = Column(Integer, primary_key=True, index=True)
    period = Column(Text, nullable=False)  # daily, weekly, monthly
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    
    # Metrics
    total_predictions = Column(Integer, default=0)
    correct_direction = Column(Integer, default=0)
    direction_accuracy = Column(Float, nullable=True)
    
    mae = Column(Float, nullable=True)  # Mean Absolute Error
    rmse = Column(Float, nullable=True)  # Root Mean Square Error
    mape = Column(Float, nullable=True)  # Mean Absolute Percentage Error
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<PredictionAccuracy {self.period} {self.start_date}-{self.end_date}>"

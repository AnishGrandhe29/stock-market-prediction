"""
Prediction API endpoints.
Provides ML model predictions with XAI explanations.
"""
from typing import List, Optional
from datetime import date, datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.utils import get_next_trading_day
from app.models.prediction import Prediction, PredictionAccuracy
from app.schemas import PredictionResponse, PredictionRequest
from app.services.ml_service import get_prediction

router = APIRouter()


@router.get("/latest", response_model=None)
async def get_latest_prediction(
    symbol: str = Query(default="^NSEI"),
    db: AsyncSession = Depends(get_db)
):
    """Get the latest prediction for a symbol."""
    result = await db.execute(
        select(Prediction)
        .where(Prediction.symbol == symbol)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    )
    
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        # Return "pending" status instead of mock data
        # Frontend should show "Prediction generating..." or similar
        
        return {
            "id": None,
            "symbol": symbol,
            "prediction_date": date.today().isoformat(),
            "target_date": get_next_trading_day(date.today()).isoformat(),
            "status": "pending",
            "message": "Prediction not yet available. Please run generate_prediction.py or wait for scheduled update.",
            "is_pending": True,
            "created_at": datetime.now().isoformat()
        }

    # Enrich response with generated explanation text
    prediction_dict = {
        col.name: getattr(prediction, col.name)
        for col in prediction.__table__.columns
    }
    prediction_dict["explanation_text"] = generate_explanation_text(prediction)

    return prediction_dict


@router.get("/history", response_model=List[PredictionResponse])
async def get_prediction_history(
    symbol: str = Query(default="^NSEI"),
    days: int = Query(default=30, ge=1, le=90),
    db: AsyncSession = Depends(get_db)
):
    """Get prediction history for a symbol."""
    start_date = date.today() - timedelta(days=days)
    
    result = await db.execute(
        select(Prediction)
        .where(
            and_(
                Prediction.symbol == symbol,
                Prediction.prediction_date >= start_date
            )
        )
        .order_by(Prediction.prediction_date.desc())
    )
    
    return result.scalars().all()


@router.post("/generate", response_model=PredictionResponse)
async def generate_prediction(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate a new prediction using the ML model."""
    try:
        prediction = await get_prediction(
            symbol=request.symbol,
            target_date=request.target_date,
            db=db
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/xai/{prediction_id}")
async def get_xai_explanation(
    prediction_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed XAI explanation for a prediction."""
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return {
        "prediction_id": prediction.id,
        "target_date": prediction.target_date,
        "predicted_value": prediction.predicted_open,
        "shap_values": prediction.shap_values,
        "modality_weights": prediction.modality_weights,
        "top_features": prediction.top_features,
        "attention_weights": prediction.attention_weights,
        "explanation_text": generate_explanation_text(prediction),
    }


@router.get("/accuracy")
async def get_prediction_accuracy(
    period: str = Query(default="weekly", pattern="^(daily|weekly|monthly)$"),
    db: AsyncSession = Depends(get_db)
):
    """Get prediction accuracy metrics."""
    result = await db.execute(
        select(PredictionAccuracy)
        .where(PredictionAccuracy.period == period)
        .order_by(PredictionAccuracy.end_date.desc())
        .limit(10)
    )
    
    accuracies = result.scalars().all()
    
    if not accuracies:
        raise HTTPException(status_code=404, detail="Accuracy history not found for the specified period")
    
    return {
        "period": period,
        "metrics": {
            "direction_accuracy": accuracies[0].direction_accuracy,
            "mae": accuracies[0].mae,
            "rmse": accuracies[0].rmse,
            "mape": accuracies[0].mape,
        },
        "history": [
            {
                "start_date": acc.start_date,
                "end_date": acc.end_date,
                "direction_accuracy": acc.direction_accuracy,
            }
            for acc in accuracies
        ],
    }


def generate_explanation_text(prediction: Prediction) -> str:
    """Generate human-readable explanation for a prediction."""
    direction = prediction.predicted_direction or "up"
    confidence = prediction.confidence_level or "medium"
    change_pct = abs(prediction.predicted_change_pct or 0)
    
    direction_text = "increase" if direction == "up" else "decrease"
    
    explanation = f"The model predicts NIFTY 50 will {direction_text} by approximately {change_pct:.2f}% "
    explanation += f"with {confidence} confidence. "
    
    if prediction.modality_weights:
        dominant = max(prediction.modality_weights, key=prediction.modality_weights.get)
        explanation += f"The {dominant} data had the strongest influence on this prediction."
    
    return explanation

"""
Model information API endpoint.
Serves static model architecture metadata for the frontend.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/info")
async def get_model_info():
    """Return model architecture and training metadata."""
    return {
        "name": "NIFTY50-Multimodal-TCN",
        "version": "1.0.0",
        "description": "Multimodal deep learning model for NIFTY 50 Index prediction using Temporal Convolutional Networks with Adaptive Fusion.",
        "parameters": 847_000,
        "architecture": {
            "type": "Multimodal Fusion Network",
            "components": [
                {
                    "name": "Price Encoder (TCN)",
                    "description": "6-layer Temporal Convolutional Network with dilated causal convolutions. Processes 60-day OHLCV sequences to capture multi-scale temporal patterns.",
                    "input": "(batch, 60, 5) — 60 days × 5 OHLCV features",
                    "output": "(batch, 128) — dense price embedding",
                    "key_innovation": "Dilated convolutions (d=1,2,4,8,16,32) give a 189-day receptive field without recurrence, enabling parallel training."
                },
                {
                    "name": "Sentiment Encoder",
                    "description": "2-layer MLP encoding aggregated news and social media sentiment scores.",
                    "input": "(batch, 3) — news, reddit, combined sentiment",
                    "output": "(batch, 128) — sentiment embedding",
                    "key_innovation": "Lightweight encoder avoids overfitting on sparse sentiment data."
                },
                {
                    "name": "Technical Encoder",
                    "description": "3-layer MLP with BatchNorm encoding 6 technical indicators (RSI, MACD, MACD Signal, Stochastic K, ADX, ATR).",
                    "input": "(batch, 6) — normalised technical indicators",
                    "output": "(batch, 128) — technical embedding",
                    "key_innovation": "Feature importance extraction via gradient-based attribution."
                },
                {
                    "name": "Adaptive Fusion Gate",
                    "description": "Context-aware gating mechanism that dynamically weights each modality based on current market conditions. Uses learned temperature parameter for softmax.",
                    "input": "3 × (batch, 128) — modality embeddings",
                    "output": "(batch, 384) — fused representation + modality weights",
                    "key_innovation": "Dynamic fusion weights provide built-in explainability — no post-hoc analysis needed."
                },
                {
                    "name": "Prediction Head",
                    "description": "Multi-output head producing point prediction (daily return %), quantile estimates (5th/50th/95th), and direction probabilities.",
                    "input": "(batch, 384) — fused representation",
                    "output": "Point prediction, 3 quantiles, 3-class direction probabilities, uncertainty score",
                    "key_innovation": "Simultaneous regression + classification + uncertainty quantification."
                }
            ],
            "flow": "Price (TCN) + Sentiment (MLP) + Technical (MLP) → Adaptive Fusion Gate → Multi-Output Prediction Head"
        },
        "training": {
            "dataset": "NIFTY 50 historical data (2010-2025)",
            "epochs": 50,
            "optimizer": "AdamW (lr=1e-3, weight_decay=0.01)",
            "scheduler": "Cosine Annealing",
            "loss": "MSE + Quantile Loss",
            "gradient_clipping": 1.0,
            "train_val_test_split": "70% / 15% / 15% (time-based, no shuffling)"
        },
        "inference": {
            "prediction_target": "Next trading day's return (%)",
            "clamping": "±2% daily return (±3% for quantiles)",
            "confidence": "Blended score: 60% direction probability + 40% uncertainty",
            "caching": "5-minute TTL to avoid redundant inference"
        },
        "explainability": {
            "methods": [
                "Perturbation-based SHAP approximation",
                "Adaptive Fusion Gate modality weights",
                "Gradient-based feature importance",
                "Natural language explanation generation"
            ],
            "what_it_explains": "Which data modality (price/sentiment/technical) and which specific features drove the prediction"
        },
        "novelty_points": [
            "Multimodal fusion with adaptive gating (vs single-modality approaches)",
            "TCN architecture for price series (parallelisable, unlike LSTM/GRU)",
            "Built-in explainability via fusion gate weights (not post-hoc)",
            "Simultaneous point + quantile + direction prediction",
            "Real-time sentiment integration from news and Reddit",
            "Realistic prediction clamping to prevent unrealistic forecasts",
            "Full XAI pipeline: SHAP + modality weights + natural language explanations"
        ]
    }

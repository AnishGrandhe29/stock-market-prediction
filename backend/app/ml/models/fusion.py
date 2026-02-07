"""
Adaptive Fusion Gate and Prediction Head for multimodal integration.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


class AdaptiveFusionGate(nn.Module):
    """
    Adaptive fusion gate that learns to weight each modality dynamically.
    
    The gate learns context-dependent weights based on input features,
    allowing the model to emphasize different modalities based on
    market conditions.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_modalities: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.embedding_dim = embedding_dim
        
        # Context encoder for dynamic weighting
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_modalities),
        )
        
        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        price_emb: torch.Tensor,
        sentiment_emb: torch.Tensor,
        technical_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            price_emb: (batch, embedding_dim) - Price embedding
            sentiment_emb: (batch, embedding_dim) - Sentiment embedding
            technical_emb: (batch, embedding_dim) - Technical embedding
        Returns:
            fused: (batch, embedding_dim * 3) - Fused representation
            weights: Dict with modality weights for XAI
        """
        # Stack embeddings
        embeddings = torch.stack([price_emb, sentiment_emb, technical_emb], dim=1)
        # (batch, 3, embedding_dim)
        
        # Compute context from all modalities
        context = torch.cat([price_emb, sentiment_emb, technical_emb], dim=-1)
        # (batch, embedding_dim * 3)
        
        # Compute dynamic weights
        logits = self.context_encoder(context)  # (batch, 3)
        weights = torch.softmax(logits / self.temperature, dim=-1)  # (batch, 3)
        
        # Weighted combination
        weighted_embeddings = embeddings * weights.unsqueeze(-1)  # (batch, 3, embedding_dim)
        
        # Flatten to fused representation
        fused = weighted_embeddings.view(weights.size(0), -1)  # (batch, embedding_dim * 3)
        
        # Average weights for XAI (across batch)
        avg_weights = weights.mean(dim=0).detach().cpu()
        modality_weights = {
            "price": float(avg_weights[0]),
            "sentiment": float(avg_weights[1]),
            "technical": float(avg_weights[2]),
        }
        
        return fused, modality_weights


class PredictionHead(nn.Module):
    """
    Prediction head that outputs price prediction with uncertainty.
    
    Outputs:
    - Point prediction (predicted close price / change)
    - Quantiles (5%, 50%, 95%) for uncertainty
    - Uncertainty score
    """
    
    def __init__(
        self,
        input_dim: int = 384,  # embedding_dim * 3
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Point prediction head
        self.point_head = nn.Linear(hidden_dim // 2, 1)
        
        # Quantile heads (5%, 50%, 95%)
        self.quantile_head = nn.Linear(hidden_dim // 2, 3)
        
        # Direction classification head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),  # down, neutral, up
            nn.Softmax(dim=-1),
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) - Fused representation
        Returns:
            Dict with predictions
        """
        # Shared representation
        shared = self.shared(x)
        
        # Point prediction (as percentage change)
        point_pred = self.point_head(shared).squeeze(-1)
        
        # Quantiles
        quantiles = self.quantile_head(shared)  # (batch, 3)
        
        # Direction probabilities
        direction = self.direction_head(shared)  # (batch, 3)
        
        # Uncertainty from quantile spread
        uncertainty = (quantiles[:, 2] - quantiles[:, 0]) / 2  # Half of 90% interval
        
        return {
            "point_prediction": point_pred,
            "quantile_5": quantiles[:, 0],
            "quantile_50": quantiles[:, 1],
            "quantile_95": quantiles[:, 2],
            "direction_probs": direction,
            "uncertainty": uncertainty,
        }


class NIFTY50Predictor(nn.Module):
    """
    Complete multimodal prediction model for NIFTY 50 Index.
    
    Combines:
    - TCN for price time-series
    - Sentiment encoder for news/social sentiment
    - MLP for technical indicators
    - Adaptive fusion gate
    - Prediction head with uncertainty
    """
    
    def __init__(
        self,
        price_seq_len: int = 60,
        price_features: int = 5,
        sentiment_features: int = 3,
        technical_features: int = 15,
        embedding_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        
        from app.ml.models.tcn import TCNEncoder
        from app.ml.models.sentiment import SimpleSentimentEncoder
        from app.ml.models.technical import TechnicalEncoder
        
        # Encoders
        self.price_encoder = TCNEncoder(
            input_features=price_features,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        self.sentiment_encoder = SimpleSentimentEncoder(
            input_features=sentiment_features,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        self.technical_encoder = TechnicalEncoder(
            input_features=technical_features,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Fusion
        self.fusion = AdaptiveFusionGate(
            embedding_dim=embedding_dim,
            num_modalities=3,
            dropout=dropout
        )
        
        # Prediction
        self.prediction_head = PredictionHead(
            input_dim=embedding_dim * 3,
            dropout=dropout
        )
        
    def forward(
        self,
        price_data: torch.Tensor,
        sentiment_data: torch.Tensor,
        technical_data: torch.Tensor
    ) -> Dict:
        """
        Args:
            price_data: (batch, seq_len, 5) - OHLCV
            sentiment_data: (batch, 3) - Sentiment scores
            technical_data: (batch, 15) - Technical indicators
        Returns:
            Dict with predictions and XAI data
        """
        # Encode each modality
        price_emb = self.price_encoder(price_data)
        sentiment_emb = self.sentiment_encoder(sentiment_data)
        technical_emb = self.technical_encoder(technical_data)
        
        # Fuse modalities
        fused, modality_weights = self.fusion(price_emb, sentiment_emb, technical_emb)
        
        # Get predictions
        predictions = self.prediction_head(fused)
        
        # Add modality weights for XAI
        predictions["modality_weights"] = modality_weights
        
        # Add technical feature importance
        predictions["technical_importance"] = self.technical_encoder.get_feature_importance()
        
        return predictions

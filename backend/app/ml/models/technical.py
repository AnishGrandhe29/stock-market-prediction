"""
Technical indicators encoder (MLP-based).
"""
import torch
import torch.nn as nn


class TechnicalEncoder(nn.Module):
    """
    MLP encoder for technical indicators.
    
    Input: (batch, num_indicators) - Technical indicator values
    Output: (batch, embedding_dim) - Technical embedding
    """
    
    def __init__(
        self,
        input_features: int = 15,
        hidden_dim: int = 64,
        embedding_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # First layer
            nn.Linear(input_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second layer
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(),
        )
        
        # Feature importance tracking
        self.feature_weights = nn.Parameter(torch.ones(input_features))
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_indicators) - Technical indicator values
        Returns:
            (batch, embedding_dim) - Technical embedding
        """
        # Apply learned feature weighting
        weighted_x = x * torch.softmax(self.feature_weights, dim=0)
        
        return self.encoder(weighted_x)
    
    def get_feature_importance(self):
        """Get learned feature importance weights for XAI."""
        return torch.softmax(self.feature_weights, dim=0).detach()


# Define which indicators we use
TECHNICAL_FEATURES = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "stoch_k",
    "stoch_d",
    "ema_5",
    "ema_20",
    "ema_50",
    "sma_20",
    "adx",
    "atr_14",
    "bb_upper",
    "bb_middle",
    "bb_lower",
]

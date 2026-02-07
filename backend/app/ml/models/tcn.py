"""
Temporal Convolutional Network (TCN) for price time-series encoding.
"""
import torch
import torch.nn as nn
from typing import List


class TemporalBlock(nn.Module):
    """
    A single TCN block with dilated causal convolution.
    
    Architecture:
    - Dilated causal conv → BatchNorm → ReLU → Dropout
    - Dilated causal conv → BatchNorm → ReLU → Dropout
    - Residual connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # Causal: trim future
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]  # Causal: trim future
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual
        res = self.downsample(x) if self.downsample else x
        
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network for encoding price time-series.
    
    Input: (batch, seq_len, features) - OHLCV data
    Output: (batch, embedding_dim) - Price embedding
    """
    
    def __init__(
        self,
        input_features: int = 5,
        hidden_channels: int = 64,
        embedding_dim: int = 128,
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8, 16, 32],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_projection = nn.Conv1d(input_features, hidden_channels, 1)
        
        # TCN blocks with increasing dilation
        layers = []
        for dilation in dilations:
            layers.append(
                TemporalBlock(
                    hidden_channels, hidden_channels,
                    kernel_size, dilation, dropout
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        
        # Global pooling + projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features) - OHLCV data
        Returns:
            (batch, embedding_dim) - Price embedding
        """
        # Transpose for conv: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # TCN blocks
        x = self.tcn(x)
        
        # Global pooling: (batch, channels, 1) -> (batch, channels)
        x = self.global_pool(x).squeeze(-1)
        
        # Project to embedding
        x = self.projection(x)
        
        return x

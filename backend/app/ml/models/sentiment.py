"""
Sentiment encoder using DistilBERT for text embedding.
"""
import torch
import torch.nn as nn
from typing import Optional


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over sequence dimension.
    Learns to weight important tokens.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
        Returns:
            (batch, hidden_size) - Pooled output
        """
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax over sequence
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return pooled, weights


class SentimentEncoder(nn.Module):
    """
    Sentiment encoder using DistilBERT with attention pooling.
    
    Input: Pre-computed DistilBERT embeddings or raw text
    Output: (batch, embedding_dim) - Sentiment embedding
    
    Note: For inference, we use pre-computed embeddings to save memory.
    During training in Colab, we can fine-tune DistilBERT.
    """
    
    def __init__(
        self,
        bert_hidden_size: int = 768,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        use_pretrained: bool = False
    ):
        super().__init__()
        
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            # Load DistilBERT (only in Colab training)
            try:
                from transformers import DistilBertModel
                self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
                # Freeze BERT layers
                for param in self.bert.parameters():
                    param.requires_grad = False
            except ImportError:
                self.bert = None
        else:
            self.bert = None
        
        # Attention pooling
        self.attention_pool = AttentionPooling(bert_hidden_size)
        
        # Projection to embedding
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden_size, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        
    def forward(
        self,
        embeddings: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            embeddings: (batch, hidden_size) - Pre-computed BERT embeddings
            input_ids: (batch, seq_len) - For live BERT encoding
            attention_mask: (batch, seq_len) - For live BERT encoding
        Returns:
            embedding: (batch, embedding_dim) - Sentiment embedding
            attention_weights: (batch, seq_len) or None - For XAI
        """
        attention_weights = None
        
        if embeddings is not None:
            # Use pre-computed embeddings (inference mode)
            # Assume pooled embeddings: (batch, hidden_size)
            pooled = embeddings
        elif self.bert is not None and input_ids is not None:
            # Live BERT encoding (training mode)
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
            pooled, attention_weights = self.attention_pool(hidden_states, attention_mask)
        else:
            raise ValueError("Either embeddings or input_ids must be provided")
        
        # Project to embedding dim
        embedding = self.projection(pooled)
        
        return embedding, attention_weights


class SimpleSentimentEncoder(nn.Module):
    """
    Simplified sentiment encoder for inference without DistilBERT.
    Takes pre-computed sentiment scores as input.
    """
    
    def __init__(
        self,
        input_features: int = 3,  # news_sentiment, reddit_sentiment, combined
        embedding_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_features) - Sentiment scores
        Returns:
            (batch, embedding_dim) - Sentiment embedding
        """
        return self.encoder(x)

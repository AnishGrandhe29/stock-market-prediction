"""
Overnight Encoder Module
========================
A dedicated neural encoder for GIFT NIFTY overnight features.  Designed
to integrate cleanly into the existing ACMI++ architecture.

Architecture
------------
    Input  : (batch, N_GIFT_FEATURES)   [7 float features]
    Layer 1: Linear(7  → d//2) + GELU + LayerNorm(d//2) + Dropout
    Layer 2: Linear(d//2 → d)  + GELU + LayerNorm(d)    + Dropout
    Output : (batch, d)

Design rationale
----------------
* Two-layer depth is sufficient: the features are already numerically
  informative (gap_pct is the dominant signal) and the shallow MLP avoids
  overfitting on a small, potentially noisy overnight dataset.
* LayerNorm (not BatchNorm) is used deliberately: at inference we often
  run with batch_size = 1, where BN statistics are unreliable.
* GELU activation matches the rest of ACMI++ for activation homogeneity.
* `quality_gate` (learnable scalar) down-weights the encoder's output
  when `data_quality == 0` (daily fallback rather than intraday reading).
  This is a form of learned data-quality awareness.

Integration point
-----------------
    In ACMIPlusPlus.forward():
        [BEFORE]   fused = self.fusion([temporal, tech, graph_f], reg_probs)
        [AFTER]    fused = self.fusion([temporal, tech, graph_f, overnight], reg_probs)
"""

import logging
import torch
import torch.nn as nn
from typing import Optional

from app.services.gift_nifty_pipeline import N_GIFT_FEATURES, GIFT_FEATURE_COLS

logger = logging.getLogger(__name__)


class OvernightEncoder(nn.Module):
    """
    Encodes GIFT NIFTY overnight features into an embedding of size `d`.

    Parameters
    ----------
    n_features : int   – number of input features (default = N_GIFT_FEATURES = 7)
    d          : int   – embedding dimension, must match ACMIPlusPlus D_MODEL
    dropout    : float – dropout probability
    """

    def __init__(
        self,
        n_features: int = N_GIFT_FEATURES,
        d: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d = d

        # Two-layer MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, d // 2),
            nn.GELU(),
            nn.LayerNorm(d // 2),
            nn.Dropout(dropout),
            nn.Linear(d // 2, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(dropout),
        )

        # Learned quality gate: scalar weight in [0, 1] that scales the
        # output when data_quality is passed in.  Initialised at 1.0 so
        # the gate is effectively open by default.
        self.quality_gate_weight = nn.Parameter(torch.tensor(1.0))

        # Feature importance weights (for XAI)
        self.feature_importance = nn.Parameter(torch.ones(n_features))

    def forward(
        self,
        x: torch.Tensor,
        data_quality: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : (batch, n_features) – standardised GIFT features.
        data_quality : (batch,) int tensor, values {-1, 0, 1}.
                       1 = intraday used, 0 = daily fallback, -1 = no data.
                       When provided, the output is down-scaled for lower
                       quality inputs via the learned quality_gate_weight.

        Returns
        -------
        (batch, d) – overnight embedding
        """
        # Apply learned feature importance weighting (for XAI)
        fi = torch.softmax(self.feature_importance, dim=0)  # (n_features,)
        x_weighted = x * fi.unsqueeze(0)                    # (batch, n_features)

        # Encode
        h = self.encoder(x_weighted)   # (batch, d)

        # Quality-aware gating
        if data_quality is not None:
            # Normalise quality to [0, 1]: -1→0, 0→0.5, 1→1
            q = ((data_quality.float() + 1.0) / 2.0).clamp(0.0, 1.0)  # (batch,)
            # Apply learned gate: gate ∈ [0, 1] scaled by learned weight
            gate = q * torch.sigmoid(self.quality_gate_weight)  # (batch,)
            h = h * gate.unsqueeze(-1)                          # (batch, d)

        return h

    def get_feature_importance(self) -> dict:
        """Return feature importance scores for XAI display."""
        weights = torch.softmax(self.feature_importance, dim=0).detach().cpu().numpy()
        return dict(zip(GIFT_FEATURE_COLS, weights.tolist()))

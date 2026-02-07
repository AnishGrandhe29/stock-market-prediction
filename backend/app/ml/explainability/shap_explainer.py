"""
SHAP explainability for model predictions.
"""
import torch
import numpy as np
from typing import Dict, List, Optional
import json


class MultimodalSHAP:
    """
    Compute SHAP-like feature importance for multimodal model.
    
    For production, we use approximations since full SHAP is computationally expensive.
    """
    
    def __init__(self, model, feature_names: Dict[str, List[str]]):
        """
        Args:
            model: The NIFTY50Predictor model
            feature_names: Dict mapping modality to feature names
        """
        self.model = model
        self.feature_names = feature_names
        
    def explain(
        self,
        price_data: torch.Tensor,
        sentiment_data: torch.Tensor,
        technical_data: torch.Tensor,
        num_samples: int = 50
    ) -> Dict:
        """
        Compute feature importance using perturbation-based method.
        
        Args:
            price_data: (1, seq_len, 5) - Single sample
            sentiment_data: (1, 3)
            technical_data: (1, 15)
            num_samples: Number of perturbation samples
            
        Returns:
            Dict with SHAP-like values and explanations
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get baseline prediction
            baseline_pred = self.model(price_data, sentiment_data, technical_data)
            baseline_value = baseline_pred["point_prediction"].item()
            
            # Technical feature importance (perturbation)
            tech_importance = self._compute_technical_importance(
                price_data, sentiment_data, technical_data, baseline_value
            )
            
            # Modality importance (already from model)
            modality_weights = baseline_pred["modality_weights"]
            
            # Technical feature importance from model
            learned_importance = baseline_pred["technical_importance"].numpy()
            
        # Combine learned and perturbation-based importance
        combined_tech = self._combine_importance(tech_importance, learned_importance)
        
        # Format SHAP values
        shap_values = self._format_shap_values(combined_tech, modality_weights)
        
        # Generate top features
        top_features = self._get_top_features(combined_tech, baseline_value)
        
        return {
            "shap_values": shap_values,
            "modality_weights": modality_weights,
            "top_features": top_features,
            "explanation_summary": self._generate_summary(top_features, modality_weights),
        }
    
    def _compute_technical_importance(
        self,
        price_data: torch.Tensor,
        sentiment_data: torch.Tensor,
        technical_data: torch.Tensor,
        baseline_value: float
    ) -> np.ndarray:
        """Compute importance by zeroing out features."""
        importance = np.zeros(technical_data.shape[1])
        
        for i in range(technical_data.shape[1]):
            # Zero out feature i
            perturbed = technical_data.clone()
            perturbed[0, i] = 0
            
            pred = self.model(price_data, sentiment_data, perturbed)
            perturbed_value = pred["point_prediction"].item()
            
            # Importance = |baseline - perturbed|
            importance[i] = abs(baseline_value - perturbed_value)
        
        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()
        
        return importance
    
    def _combine_importance(
        self,
        perturbation: np.ndarray,
        learned: np.ndarray
    ) -> np.ndarray:
        """Combine perturbation and learned importance."""
        # Weighted average: 60% learned, 40% perturbation
        return 0.6 * learned + 0.4 * perturbation
    
    def _format_shap_values(
        self,
        tech_importance: np.ndarray,
        modality_weights: Dict[str, float]
    ) -> Dict:
        """Format SHAP values for frontend."""
        tech_names = self.feature_names.get("technical", [])
        
        return {
            "modalities": modality_weights,
            "technical": {
                name: float(tech_importance[i])
                for i, name in enumerate(tech_names)
                if i < len(tech_importance)
            },
            "price": {
                "trend": modality_weights.get("price", 0) * 0.5,
                "volatility": modality_weights.get("price", 0) * 0.3,
                "momentum": modality_weights.get("price", 0) * 0.2,
            },
            "sentiment": {
                "news": modality_weights.get("sentiment", 0) * 0.6,
                "reddit": modality_weights.get("sentiment", 0) * 0.4,
            }
        }
    
    def _get_top_features(
        self,
        tech_importance: np.ndarray,
        baseline_value: float
    ) -> List[Dict]:
        """Get top contributing features."""
        tech_names = self.feature_names.get("technical", [])
        
        features = []
        for i, name in enumerate(tech_names):
            if i < len(tech_importance):
                features.append({
                    "feature": name,
                    "importance": float(tech_importance[i]),
                    "direction": "positive" if baseline_value > 0 else "negative",
                    "modality": "technical",
                })
        
        # Sort by importance
        features.sort(key=lambda x: x["importance"], reverse=True)
        
        return features[:10]  # Top 10
    
    def _generate_summary(
        self,
        top_features: List[Dict],
        modality_weights: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation."""
        # Find dominant modality
        dominant = max(modality_weights, key=modality_weights.get)
        dominant_pct = modality_weights[dominant] * 100
        
        summary = f"The prediction is primarily driven by {dominant} data ({dominant_pct:.1f}%). "
        
        if top_features:
            top_3 = [f["feature"] for f in top_features[:3]]
            summary += f"Key technical factors: {', '.join(top_3)}."
        
        return summary


def compute_attention_visualization(
    attention_weights: torch.Tensor,
    tokens: List[str]
) -> List[Dict]:
    """
    Format attention weights for visualization.
    
    Args:
        attention_weights: (seq_len,) - Attention scores per token
        tokens: List of token strings
        
    Returns:
        List of {token, weight} for visualization
    """
    if attention_weights is None:
        return []
    
    weights = attention_weights.cpu().numpy()
    
    return [
        {"token": token, "weight": float(weights[i])}
        for i, token in enumerate(tokens)
        if i < len(weights)
    ]

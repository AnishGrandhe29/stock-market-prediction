"""ML explainability module initialization."""
from app.ml.explainability.shap_explainer import MultimodalSHAP, compute_attention_visualization

__all__ = ["MultimodalSHAP", "compute_attention_visualization"]

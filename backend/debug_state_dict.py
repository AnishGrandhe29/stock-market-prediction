"""
Debug script to check state dict compatibility.
"""
import torch
from pathlib import Path
import sys
sys.path.insert(0, '.')
from app.config import settings
from app.ml.models.fusion import NIFTY50Predictor

# Load model with correct architecture
model = NIFTY50Predictor(
    price_seq_len=60, price_features=5, sentiment_features=3,
    technical_features=6, embedding_dim=128, dropout=0.2
)

model_path = Path(settings.model_path)
sd = torch.load(model_path, map_location='cpu', weights_only=True)

print("SAVED MODEL STATE DICT (technical/fusion keys):")
for k in sorted(sd.keys()):
    if 'technical' in k or 'fusion' in k:
        shape = sd[k].shape if hasattr(sd[k], 'shape') else type(sd[k])
        print(f"  {k}: {shape}")

print()
print("CURRENT MODEL STATE DICT (technical/fusion keys):")
for k, v in model.state_dict().items():
    if 'technical' in k or 'fusion' in k:
        print(f"  {k}: {v.shape}")

# Try to load with strict=False
print()
print("Attempting load with strict=False...")
try:
    result = model.load_state_dict(sd, strict=False)
    print(f"Missing keys: {result.missing_keys}")
    print(f"Unexpected keys: {result.unexpected_keys}")
except Exception as e:
    print(f"Error: {e}")

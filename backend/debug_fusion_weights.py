"""
Debug the fusion gate weights directly.
"""
import torch
from pathlib import Path
import sys
sys.path.insert(0, '.')
from app.config import settings

model_path = Path(settings.model_path)
sd = torch.load(model_path, map_location='cpu', weights_only=True)

print("FUSION GATE WEIGHTS IN SAVED MODEL:")
print("=" * 50)
for k in sorted(sd.keys()):
    if 'fusion' in k:
        v = sd[k]
        print(f"\n{k}:")
        print(f"  Shape: {v.shape}")
        if v.numel() < 100:
            print(f"  Values: {v}")
        else:
            print(f"  Mean: {v.mean().item():.4f}, Std: {v.std().item():.4f}")
            print(f"  Min: {v.min().item():.4f}, Max: {v.max().item():.4f}")

print("\n" + "=" * 50)
print("TECHNICAL ENCODER WEIGHTS:")
for k in sorted(sd.keys()):
    if 'technical' in k:
        v = sd[k]
        print(f"\n{k}:")
        print(f"  Shape: {v.shape}")

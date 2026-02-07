# %% [markdown]
# # Export Model for Deployment
# 
# This notebook exports the trained model for local inference.

# %% [markdown]
# ## 1. Load Trained Model

# %%
import torch
import os

# Load model (using the same architecture from training notebook)
# Note: Run cell with model definitions from 03_model_training.py first

model = NIFTY50Predictor(embedding_dim=128, dropout=0.2)
model.load_state_dict(torch.load('nifty50_model.pt', map_location='cpu'))
model.eval()

print("Model loaded successfully")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 2. Test Inference

# %%
# Test with dummy data
batch_size = 1
price_dummy = torch.randn(batch_size, 60, 5)
sentiment_dummy = torch.randn(batch_size, 3)
technical_dummy = torch.randn(batch_size, 15)

with torch.no_grad():
    output = model(price_dummy, sentiment_dummy, technical_dummy)

print("Test inference output:")
print(f"  Point prediction: {output['point'].item():.4f}")
print(f"  Quantile 5%: {output['quantile_5'].item():.4f}")
print(f"  Quantile 95%: {output['quantile_95'].item():.4f}")
print(f"  Modality weights: {output['modality_weights'].numpy()}")

# %% [markdown]
# ## 3. Export Model

# %%
# Save for CPU inference
model_cpu = model.cpu()

# Save state dict
torch.save(model_cpu.state_dict(), 'nifty50_model.pt')
print("✓ Saved state dict: nifty50_model.pt")

# Save complete model (with architecture)
torch.save(model_cpu, 'nifty50_model_complete.pt')
print("✓ Saved complete model: nifty50_model_complete.pt")

# %% [markdown]
# ## 4. Export Scalers

# %%
import pickle
import shutil

# Copy scalers
files_to_export = [
    'nifty50_model.pt',
    'nifty50_model_complete.pt',
    'data/price_scaler.pkl',
    'data/tech_scaler.pkl',
    'training_history.png',
]

os.makedirs('export', exist_ok=True)

for f in files_to_export:
    if os.path.exists(f):
        shutil.copy(f, f'export/{os.path.basename(f)}')
        print(f"✓ Copied {f}")

# %% [markdown]
# ## 5. Create Model Info File

# %%
import json

model_info = {
    "name": "NIFTY50Predictor",
    "version": "1.0.0",
    "architecture": {
        "price_encoder": "TCN (6 blocks, dilations 1-32)",
        "sentiment_encoder": "MLP (3 → 64 → 128)",
        "technical_encoder": "MLP (15 → 64 → 128)",
        "fusion": "Adaptive Gate (3 modalities)",
        "prediction_head": "MLP with quantile outputs",
    },
    "input_shapes": {
        "price": [60, 5],
        "sentiment": [3],
        "technical": [15],
    },
    "outputs": {
        "point": "Predicted % change",
        "quantile_5": "5th percentile",
        "quantile_50": "Median",
        "quantile_95": "95th percentile",
        "modality_weights": "Fusion gate weights [price, sent, tech]",
    },
    "parameters": sum(p.numel() for p in model.parameters()),
}

with open('export/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n✓ Created model_info.json")

# %% [markdown]
# ## 6. Download for Local Use

# %%
# Zip export folder
shutil.make_archive('nifty50_export', 'zip', 'export')

# Download
from google.colab import files
files.download('nifty50_export.zip')

print("\n" + "="*50)
print("EXPORT COMPLETE")
print("="*50)
print("\nDownloading nifty50_export.zip...")
print("\nAfter download:")
print("1. Extract the zip file")
print("2. Copy nifty50_model.pt to your project's models/ folder")
print("3. Copy scalers to models/ folder")
print("4. Start the backend server")

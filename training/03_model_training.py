# %% [markdown]
# # NIFTY 50 Model Training
# 
# This notebook trains the multimodal TCN model on Google Colab with GPU.

# %% [markdown]
# ## Setup

# %%
!pip install torch torchvision -q

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 1. Define Model Architecture

# %%
class TemporalBlock(nn.Module):
    """TCN block with dilated causal convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.relu(self.bn1(out))
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        
        res = self.downsample(x) if self.downsample else x
        if res.size(2) > out.size(2):
            res = res[:, :, :out.size(2)]
        elif res.size(2) < out.size(2):
            out = out[:, :, :res.size(2)]
            
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """TCN for price time-series encoding."""
    
    def __init__(self, input_features=5, hidden_channels=64, embedding_dim=128, dropout=0.2):
        super().__init__()
        
        self.input_projection = nn.Conv1d(input_features, hidden_channels, 1)
        
        dilations = [1, 2, 4, 8, 16, 32]
        layers = [TemporalBlock(hidden_channels, hidden_channels, 3, d, dropout) for d in dilations]
        self.tcn = nn.Sequential(*layers)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        return self.projection(x)


class TechnicalEncoder(nn.Module):
    """MLP for technical indicators."""
    
    def __init__(self, input_features=15, embedding_dim=128, dropout=0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.encoder(x)


class SentimentEncoder(nn.Module):
    """Simple encoder for sentiment scores."""
    
    def __init__(self, input_features=3, embedding_dim=128, dropout=0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.encoder(x)


class AdaptiveFusionGate(nn.Module):
    """Dynamic modality weighting."""
    
    def __init__(self, embedding_dim=128, num_modalities=3, dropout=0.2):
        super().__init__()
        
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_modalities),
        )
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, price_emb, sentiment_emb, technical_emb):
        embeddings = torch.stack([price_emb, sentiment_emb, technical_emb], dim=1)
        context = torch.cat([price_emb, sentiment_emb, technical_emb], dim=-1)
        
        logits = self.context_encoder(context)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        
        weighted = embeddings * weights.unsqueeze(-1)
        fused = weighted.view(weights.size(0), -1)
        
        return fused, weights


class PredictionHead(nn.Module):
    """Multi-output prediction head."""
    
    def __init__(self, input_dim=384, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.point_head = nn.Linear(hidden_dim // 2, 1)
        self.quantile_head = nn.Linear(hidden_dim // 2, 3)
        
    def forward(self, x):
        shared = self.shared(x)
        point = self.point_head(shared).squeeze(-1)
        quantiles = self.quantile_head(shared)
        
        return {
            'point': point,
            'quantile_5': quantiles[:, 0],
            'quantile_50': quantiles[:, 1],
            'quantile_95': quantiles[:, 2],
        }


class NIFTY50Predictor(nn.Module):
    """Complete multimodal model."""
    
    def __init__(self, embedding_dim=128, dropout=0.2):
        super().__init__()
        
        self.price_encoder = TCNEncoder(5, 64, embedding_dim, dropout)
        self.sentiment_encoder = SentimentEncoder(3, embedding_dim, dropout)
        self.technical_encoder = TechnicalEncoder(15, embedding_dim, dropout)
        self.fusion = AdaptiveFusionGate(embedding_dim, 3, dropout)
        self.prediction_head = PredictionHead(embedding_dim * 3, 256, dropout)
        
    def forward(self, price, sentiment, technical):
        price_emb = self.price_encoder(price)
        sent_emb = self.sentiment_encoder(sentiment)
        tech_emb = self.technical_encoder(technical)
        
        fused, weights = self.fusion(price_emb, sent_emb, tech_emb)
        predictions = self.prediction_head(fused)
        predictions['modality_weights'] = weights
        
        return predictions

# %% [markdown]
# ## 2. Load Data

# %%
class NIFTY50Dataset(Dataset):
    def __init__(self, price, technical, sentiment, targets):
        self.price = torch.tensor(price, dtype=torch.float32)
        self.technical = torch.tensor(technical, dtype=torch.float32)
        self.sentiment = torch.tensor(sentiment, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'price': self.price[idx],
            'technical': self.technical[idx],
            'sentiment': self.sentiment[idx],
            'target': self.targets[idx],
        }

# %%
# Load data
X_price_train = np.load('data/X_price_train.npy')
X_price_val = np.load('data/X_price_val.npy')
X_tech_train = np.load('data/X_tech_train.npy')
X_tech_val = np.load('data/X_tech_val.npy')
X_sent_train = np.load('data/X_sent_train.npy')
X_sent_val = np.load('data/X_sent_val.npy')
y_train = np.load('data/y_train.npy')
y_val = np.load('data/y_val.npy')

print(f"Training samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")

# %%
# Create datasets and dataloaders
train_dataset = NIFTY50Dataset(X_price_train, X_tech_train, X_sent_train, y_train)
val_dataset = NIFTY50Dataset(X_price_val, X_tech_val, X_sent_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# %% [markdown]
# ## 3. Training

# %%
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        price = batch['price'].to(device)
        technical = batch['technical'].to(device)
        sentiment = batch['sentiment'].to(device)
        target = batch['target'].to(device)
        
        output = model(price, sentiment, technical)
        
        # Combined loss
        loss = criterion(output['point'], target)
        loss += 0.5 * criterion(output['quantile_50'], target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in loader:
            price = batch['price'].to(device)
            technical = batch['technical'].to(device)
            sentiment = batch['sentiment'].to(device)
            target = batch['target'].to(device)
            
            output = model(price, sentiment, technical)
            loss = criterion(output['point'], target)
            
            total_loss += loss.item()
            predictions.extend(output['point'].cpu().numpy())
            actuals.extend(target.cpu().numpy())
    
    # Calculate direction accuracy
    pred_dir = np.sign(predictions)
    actual_dir = np.sign(actuals)
    direction_acc = (pred_dir == actual_dir).mean()
    
    return total_loss / len(loader), direction_acc

# %%
# Initialize model
model = NIFTY50Predictor(embedding_dim=128, dropout=0.2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# Training loop
EPOCHS = 50
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Dir Acc={val_acc:.2%}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'nifty50_model.pt')
        print("  ✓ Saved best model")

# %% [markdown]
# ## 4. Visualize Training

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss')
ax1.legend()

ax2.plot(history['val_acc'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Direction Accuracy')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# %% [markdown]
# ## 5. Evaluate on Test Set

# %%
# Load test data
X_price_test = np.load('data/X_price_test.npy')
X_tech_test = np.load('data/X_tech_test.npy')
X_sent_test = np.load('data/X_sent_test.npy')
y_test = np.load('data/y_test.npy')

test_dataset = NIFTY50Dataset(X_price_test, X_tech_test, X_sent_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load best model
model.load_state_dict(torch.load('nifty50_model.pt'))
test_loss, test_acc = validate(model, test_loader, criterion, device)

print(f"\n{'='*50}")
print(f"TEST RESULTS")
print(f"{'='*50}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Direction Accuracy: {test_acc:.2%}")
print(f"{'='*50}")

# %%
print("\n✓ Training complete!")
print("Model saved as 'nifty50_model.pt'")
print("\nNext: Run 04_export_model.ipynb to prepare for deployment")

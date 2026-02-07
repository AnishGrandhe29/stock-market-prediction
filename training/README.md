# NIFTY 50 Model Training - Google Colab

This folder contains the training pipeline for the multimodal deep learning model.

---

## 🚀 Quick Start (Recommended)

### **Use the All-in-One Notebook:**

**`NIFTY50_Complete_Training.ipynb`**

This single notebook combines everything:
- ✅ Data Collection (OHLCV + Sentiment)
- ✅ Feature Engineering (Technical Indicators)  
- ✅ Model Training (Multimodal TCN)
- ✅ Export & Download

### Usage:

1. Upload `NIFTY50_Complete_Training.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Go to `Runtime` → `Change runtime type` → Select `T4 GPU`
3. Go to `Runtime` → `Run all` (or press `Ctrl+F9`)
4. Wait ~30-60 minutes for training to complete
5. Download `nifty50_model_export.zip` (automatic download at end)
6. Extract and copy `nifty50_model.pt` to `models/` folder

---

## 📁 Individual Scripts (Reference)

For advanced users, the individual Python scripts are also available:

| File                        | Description                                       |
|-----------------------------|---------------------------------------------------|
| `01_data_collection.py`     | Fetch historical OHLCV, news, and sentiment data  |
| `02_feature_engineering.py` | Compute technical indicators and prepare features |
| `03_model_training.py`      | Train the multimodal TCN model                    |
| `04_export_model.py`        | Export trained model for local inference          |

---

## 💻 GPU Requirements

- ✅ Free Google Colab T4 GPU is sufficient
- ⏱️ Training takes approximately 30-60 minutes
- 💾 Model uses ~500MB GPU memory

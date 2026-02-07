# NIFTY 50 Index Prediction System

A production-ready multimodal deep learning platform for predicting NIFTY 50 Index movements with Explainable AI.

## Features

- 📊 **Live NIFTY 50 Charts** - Real-time price visualization
- 🤖 **AI Predictions** - Multimodal deep learning model
- 🔍 **Explainable AI** - SHAP values and feature importance
- 👤 **User Features** - Notes, watchlist, alerts
- 🔐 **Secure Auth** - Google OAuth + Email/Password

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis (optional)

### Installation

1. **Clone and setup backend:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Setup frontend:**
```bash
cd frontend
npm install
```

3. **Configure environment:**
```bash
copy backend\.env.example backend\.env
# Edit .env with your settings
```

4. **Run the application:**
```bash
run_all.bat
```

## Project Structure

```
├── backend/          # FastAPI backend
├── frontend/         # Next.js frontend
├── training/         # Colab notebooks
├── models/           # Trained models
└── data/            # Cached data
```

## Tech Stack

- **Backend**: FastAPI, PostgreSQL, Redis
- **Frontend**: Next.js 14, TypeScript, TailwindCSS
- **ML**: PyTorch, DistilBERT, SHAP
- **Data**: Yahoo Finance, Reddit, Google News

## License

MIT

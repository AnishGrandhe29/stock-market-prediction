# Stock Market Prediction using TCN (Temporal Convolutional Networks)

This project implements a modular, multimodal trading system designed for the NIFTY 50 index. It leverages Temporal Convolutional Networks (TCN) to predict market movements by integrating various data sources including historical prices, macroeconomic indicators, and social sentiment.

## Codebase Structure & Components

The source code is organized clearly within the `src/` directory. Here is a breakdown of what each component does:

### 1. Data Collection (`src/data/`)
- **`fetch_macro.py`**: Fetches macroeconomic indicators (e.g., interest rates, inflation) relevant to market analysis.
- **`social.py`**: Scrapes and processes social media sentiment data (e.g., from Reddit or Twitter) to gauge market sentiment.
- **`prices.py`**: Handles fetching of historical price data (OHLCV) for the NIFTY 50 index and constituents.

### 2. Feature Engineering (`src/features/`)
- **`engineering.py`**: Processes raw data into model-ready features. This includes calculating technical indicators (RSI, MACD, etc.), normalizing data, and aligning timestamps across different data sources.

### 3. Model (`src/model/`)
- **`tcn.py`**: Defines the Temporal Convolutional Network (TCN) architecture. This deep learning model is chosen for its ability to capture long-range dependencies in time-series data better than traditional RNNs/LSTMs in many contexts.

### 4. Training (`src/train/`)
- **`train.py`**: The main training script. it loads processed features, initializes the TCN model, runs the training loop, and saves model checkpoints.
- **`online.py`**: Implements online learning capabilities, allowing the model to adapt incrementally to new data without a full retrain.
- **`loss.py`**: Contains custom loss functions optimized for financial time-series prediction.

### 5. Backtesting (`src/backtest/`)
- **`backtest.py`**: Simulates trading strategies using historical data to evaluate performance. It calculates metrics like Sharpe Ratio, Maximum Drawdown, and cumulative returns.

### 6. Trading & Monitoring (`src/trade/` & `src/monitor/`)
- **`paper_trader.py`**: Executes paper trades (simulated live trading) to test the system in real-time market conditions without risking real capital.
- **`live_monitor.py`**: A monitoring script to track the health of the system and live trading performance.

### 7. Serving & Dashboard (`src/serve/`)
- **`app.py`**: A FastAPI application that serves the model predictions via a REST API.
- **`dashboard.py`**: A Streamlit-based dashboard that visualizes data, model predictions, and backtest results for user interaction.

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Docker (optional)

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd FinalYear_Project
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Running the Project

The project uses a `Makefile` to simplify running various stages of the pipeline.

### 1. Data Pipeline
To fetch data and generate features:
```bash
# Fetch latest data (macro & social)
make data

# Run feature engineering
make features
```

### 2. Training
To train the TCN model:
```bash
make train
```
*Cmd falls back to `python src/train/train.py`*

### 3. Backtesting
To run backtests on historical data:
```bash
make backtest
```

### 4. Deployment & Visualization
To run the API and Dashboard:

**API (FastAPI):**
```bash
make serve
```
*Access API docs at `http://localhost:8000/docs`*

**Dashboard (Streamlit):**
```bash
make dashboard
```
*Access dashboard at `http://localhost:8501`*

### 5. Run All
To run the complete pipeline (Data -> Features -> Train -> Backtest):
```bash
make all
```

---

## Docker Support
You can also run the application using Docker.

1. **Build the image:**
   ```bash
   docker build -t stock-prediction-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 -p 8501:8501 stock-prediction-app
   ```

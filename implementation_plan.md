# ACMI++ Model Implementation Plan

This document outlines the step-by-step approach to replacing the current `NIFTY50Predictor` with the new **ACMI++** (Adaptive Contextual Market Intelligence) model from `ACMI_PlusPlus_Full_Pipeline.ipynb`.

## 🧠 1. Architectural Differences & Understanding

### Current Setup (Legacy)
- **Model:** TCN + Transformer for 1-day ahead prediction.
- **Data:** ~14 features total (5 price, 3 sentiment, 6 technical). Data is routinely ingested via background jobs into SQLite.
- **Output:** Next day `% change` point prediction + simple quantiles.

### New Setup (ACMI++)
- **Model:** Regime-Conditioned, Multi-Horizon stock prediction leveraging Temporal (TCN/TF), Technical, and Structural (GNN) embeddings. Integrates Conformal Predictions & Deep Ensembles.
- **Data:** 40+ technical features generated using the `ta` library, plus historical equity and macroeconomic OHLCV data. 
- **Outputs:**
  - Multi-Horizon limits: 1d, 5d, 20d, 60d point predictions + 90% confidence bands.
  - Volatility forecast.
  - Crash probability probability.
  - Market Regime Classification (Bull, Bear, Sideways, HighVol, Unknown).

---

## 🛠️ 2. Step-by-Step Implementation Approach

DO NOT proceed to coding until this plan is verbally approved.

### Phase 1: Model & State Migration
1. **Transfer Model Weights:** Load the user-provided `acmi_best_model.pt`, `scalers.pkl`, and `acmi_ensemble.pt` into the backend directory (e.g., `backend/app/ml/weights/acmi/`).
2. **Port PyTorch Architecture:** Create `backend/app/ml/models/acmi.py` and cleanly port over the model classes: `TCNBlock`, `TCNEncoder`, `TFEncoder`, `GNNEncoder`, `RegimeEngine`, `CrossModalFusion`, and `ACMIPlusPlus`.
3. **Port Prediction Wrapper:** Implement the `ACMIPredictor` class as a singleton in `backend/app/services/ml_service.py` to handle ensemble inference and conformal bounded ranges.

### Phase 2: Database & Schema Evolution
1. **Update SQLAlchemy Models (`app/models/prediction.py`):**
   - Accommodate multi-horizon fields: _(e.g. `horizon_1d_point`, `horizon_1d_interval`, `horizon_5d_...`)_
   - Add new metrics: `crash_probability` (Float), `volatility_forecast` (Float).
   - Add regime fields: `market_regime` (String), `regime_probabilities` (JSON).
2. **Update Pydantic Schemas (`app/schemas.py`):**
   - Refactor `PredictionResponse` to map these nested objects cleanly to the frontend JSON structure.
3. **Database Migration:** Generate an Alembic script or manual SQL to alter the existing SQLite `predictions` table structure.

### Phase 3: Data Pipeline Adaptation
The ACMI++ model requires complex dynamically generated technical features (~40+) unlike the current DB-driven approach. 
1. **Dynamic Feature Generation:** Implement the ACMI++ `DataPipeline` inside `ml_service.py` for inference.
   - When a prediction is requested, query historical OHLCV from the SQLite DB (or fetch the remainder from `yfinance`).
   - Run the `ta` library feature generation precisely as the colab notebook did prior to `transform()` scaling, ensuring identical metric creation. 
   - Note: We will need to decide if we want to run `ta` on historical DB data, or fetch history directly from Yahoo Finance at inference time (and rely heavily on Redis for caching to avoid rate-limiting). **Recommendation:** Rely on DB historical data + calculate features dynamically on inference.

### Phase 4: Backend Logic Update (`ml_service.py`)
1. **Refactor `get_prediction()`:** 
   - Instantiate the `ACMIPredictor` on service start.
   - Inside `get_prediction`, delegate sequence creation to the `ACMIPredictor`.
   - Update mapping logic to persist multi-horizon, regime, and risk data into the updated `Prediction` DB entity.
   - Keep the existing `Redis` inference cache, perhaps extending TTL slightly given the new heavier inference load.
2. **Modify/Disable SHAP (Optional):** Ensure XAI still works gracefully over the expanded 40+ feature set. It might require adapting `MultimodalSHAP` to correctly map the updated input dimensions.

### Phase 5: Frontend Enrichment (Future Steps)
1. Ensure the frontend pulls the updated API schema correctly.
2. Build UI widgets to visualize `Market Regime` badges.
3. Hook up a "Risk Meter" or warning component for `Crash Probability`.
4. Shift from a 1-day prediction text field to a multi-horizon timeline or table view.

---

## 🚦 3. Next Steps & Approval
Once you confirm this plan aligns with your expectations, we will begin sequentially:
1. Translating the ACMI Architecture to the backend.
2. Expanding the schemas and database definitions.
3. Porting the inference wrapper and prediction workflows.

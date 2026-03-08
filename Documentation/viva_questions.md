# Viva Questions & Answers — NIFTY 50 AI Prediction System

---

## Architecture & Design

### Q1: Why did you choose a Temporal Convolutional Network (TCN) instead of LSTM or GRU?

**Answer:** Three key reasons:
1. **Parallelizable computation** — TCN processes all time steps simultaneously, making training 3-5x faster than sequential LSTM.
2. **Longer effective memory** — Our dilated convolutions (d=1,2,4,8,16,32) provide a 189-day receptive field without vanishing gradient problems.
3. **Deterministic inference** — Unlike LSTM, which accumulates hidden state errors over long sequences, TCN's convolutional architecture produces consistent outputs.

Additionally, recent research (Bai et al., 2018) demonstrates that TCNs outperform RNNs on most sequence modelling benchmarks while being simpler to tune.

---

### Q2: What is the Adaptive Fusion Gate and why is it important?

**Answer:** The Adaptive Fusion Gate is a learned gating mechanism that dynamically weights how much each data modality (price, sentiment, technical) contributes to the final prediction. It works by:
1. Computing a context vector from each modality's embedding
2. Passing through a learned linear layer
3. Applying temperature-scaled softmax to produce weights

**Why it's important:** During earnings season, sentiment data becomes highly predictive, while during trending markets, technical indicators dominate. The gate automatically adjusts these weights based on current conditions, unlike fixed-weight concatenation approaches.

---

### Q3: Why did you use three separate encoders instead of one unified model?

**Answer:** Our data modalities have fundamentally different structures:
- **Price data** is sequential (60-day time series of 5 OHLCV features) — needs temporal modelling
- **Sentiment data** is a 3-dimensional vector (news, reddit, combined) — simple MLP suffices
- **Technical indicators** are 6 point-in-time features — MLP with BatchNorm works well

A unified model would waste parameters learning the same representation for structurally different inputs. Separate encoders allow each modality to be encoded optimally before fusion.

---

### Q4: Explain the multi-output prediction head.

**Answer:** Our prediction head produces 6 outputs from a single forward pass:
1. **Point prediction** — Expected daily return (%) — regression task
2. **Quantile 5** — 5th percentile of return (lower bound)
3. **Quantile 50** — Median return
4. **Quantile 95** — 95th percentile (upper bound)
5. **Direction probabilities** — (down, neutral, up) class probabilities
6. **Uncertainty score** — Model's self-assessed uncertainty

This is trained with a combined loss: MSE for point prediction + Quantile Loss for quantiles + Cross-Entropy for direction. The multi-task learning provides implicit regularization and more informative outputs.

---

## Training & Data

### Q5: What data did you use for training?

**Answer:** 
- **Price data:** NIFTY 50 OHLCV data from 2010-2025 (approximately 3,700 trading days)
- **Sentiment:** Aggregated news sentiment and Reddit sentiment scores
- **Technical indicators:** RSI(14), MACD, MACD Signal, Stochastic K, ADX, ATR — computed using the `pandas-ta` library
- **Target variable:** Daily return percentage = `(next_close - close) / close × 100`

Training split: 70% train / 15% validation / 15% test, using **time-based splitting** (no shuffling) to prevent data leakage.

---

### Q6: How did you prevent overfitting?

**Answer:** Multiple strategies:
1. **Weight decay** (L2 regularization) = 0.01 via AdamW optimizer
2. **Dropout** layers in all encoders and prediction head
3. **Gradient clipping** at 1.0 to prevent exploding gradients
4. **Cosine Annealing** learning rate scheduler — gradual learning rate reduction
5. **Multi-task loss** acts as implicit regularization (point + quantile + direction tasks share features)
6. **Time-based train/val/test split** — prevents future data leakage
7. **Early stopping** based on validation loss

---

### Q7: What loss function did you use and why?

**Answer:** A composite loss:
```
Total Loss = MSE(point_pred, actual_return) + λ × QuantileLoss(quantiles, actual_return)
```

- **MSE Loss** for the point prediction — standard regression loss
- **Quantile Loss** (pinball loss) for the 5th, 50th, and 95th percentile estimates — this asymmetric loss penalises under-estimation and over-estimation differently depending on the quantile
- The direction classification uses **Cross-Entropy Loss**

The multi-objective loss ensures the model learns both accurate point predictions and well-calibrated uncertainty bands.

---

### Q8: Why did you choose percentage returns as the target instead of absolute price?

**Answer:** Three reasons:
1. **Stationarity** — Raw prices are non-stationary (they trend upward over years). Returns are approximately stationary, which is much easier for neural networks to learn.
2. **Scale invariance** — A 200-point move means different things when NIFTY is at 10,000 vs 25,000. Percentage returns normalize this.
3. **Realistic constraints** — It's easier to apply realistic bounds (±2%) to percentage returns than to absolute price values.

---

## Explainability (XAI)

### Q9: How does your SHAP implementation work?

**Answer:** We use a **perturbation-based SHAP approximation** optimized for inference speed:
1. Get baseline prediction with original input
2. For each technical feature, zero it out and re-run inference
3. Feature importance = |baseline prediction - perturbed prediction|
4. Normalize importance scores to sum to 1
5. Combine with learned importance (from the model's gradient-based attribution) using 60/40 weighting

This is faster than full SHAP (which requires 2^N evaluations) while still providing meaningful attributions.

---

### Q10: What does the Adaptive Fusion Gate tell us about explainability?

**Answer:** The gate provides **intrinsic explainability** — a rare property in deep learning:
- *"The model weighted Price 45%, Sentiment 30%, Technical 25%"* — this is directly interpretable
- No post-hoc analysis needed; the weights are a natural part of inference
- Different from attention weights (which are often criticized for not being true explanations), fusion gate weights directly determine the contribution ratio

This is a significant research contribution because most XAI methods (SHAP, LIME, Grad-CAM) are applied after the fact.

---

### Q11: Can you explain a sample prediction in plain English?

**Answer:** *"The model predicts NIFTY 50 will increase by 0.45% tomorrow with medium confidence. This is primarily driven by price history data (45%), which shows a bullish momentum pattern. Technical indicators (25%) show RSI at 58 — neutral territory with slight upward bias. News sentiment (30%) is mildly positive following the latest RBI policy announcement. The 90% confidence interval ranges from -0.2% to +1.1%."*

This natural language explanation is automatically generated from the XAI data.

---

## Evaluation & Performance

### Q12: How do you evaluate prediction accuracy?

**Answer:** Multiple metrics:
1. **Mean Absolute Error (MAE)** — Average absolute difference between predicted and actual returns
2. **RMSE** — Root mean squared error (penalises large errors more)
3. **Direction Accuracy** — % of times we correctly predicted up/neutral/down (~68%)
4. **Quantile Calibration** — Whether 90% of actual values fall within Q5-Q95 band
5. **Sharpe Ratio** — If you traded based on our signals, annualized risk-adjusted return

---

### Q13: What is the model's actual accuracy?

**Answer:** Our direction accuracy is approximately **68%** on the test set.

For context:
- Random guessing = 33% (3 classes) or 50% (binary up/down)
- Efficient Market Hypothesis suggests ~50% for binary prediction
- Our 68% is competitive with state-of-the-art models in the literature (typically 55-72%)

It's important to note that even a 5-10% edge above 50% can be highly profitable in systematic trading.

---

### Q14: What are the limitations of your system?

**Answer:**
1. **Black swan events** — Model cannot predict unprecedented events (COVID crash, wars, policy shocks)
2. **Training data bias** — Model trained on 2010-2025 data may not generalize to fundamentally different market regimes
3. **Sentiment latency** — News and Reddit sentiment are captured daily, missing intraday sentiment shifts
4. **Single-index focus** — Currently only supports NIFTY 50; generalization to other indices needs fine-tuning
5. **No fundamental analysis** — Doesn't incorporate earnings, P/E ratios, or macroeconomic data
6. **Regulation** — Not SEBI-approved for algorithmic trading; research/educational purposes only

---

## Deployment & Demo

### Q15: How is predictive speed optimized for the live demo?

**Answer:**
1. **Model caching** — Model loaded once and reused across requests (no disk I/O per request)
2. **Prediction caching** — 5-minute TTL cache avoids redundant inference
3. **No GPU required** — Model is small enough (847K params) for CPU inference in <100ms
4. **Async API** — FastAPI with async endpoints allows concurrent request handling
5. **Scaler caching** — Price and technical scalers loaded once at startup

---

### Q16: What tech stack did you use and why?

**Answer:**
- **Backend:** FastAPI (Python) — async support, automatic OpenAPI docs, type-safe with Pydantic
- **ML Framework:** PyTorch — flexible, Pythonic, extensive research ecosystem
- **Frontend:** Next.js 14 with React — server-side rendering, file-based routing, excellent developer experience
- **Database:** SQLite (demo) / PostgreSQL (production) — via SQLAlchemy async ORM
- **Styling:** Tailwind CSS — utility-first, responsive design without custom CSS files

---

### Q17: How would you scale this to production?

**Answer:**
1. **GPU inference server** — Move to NVIDIA Triton Inference Server for batch processing
2. **PostgreSQL with read replicas** — Replace SQLite for concurrent access
3. **Redis caching layer** — Sub-millisecond prediction retrieval
4. **Kubernetes deployment** — Auto-scaling based on request volume
5. **Real-time data pipeline** — Apache Kafka for streaming price updates
6. **Model versioning** — MLflow for experiment tracking and model registry
7. **Monitoring** — Prometheus + Grafana for model drift detection

---

## Deep Learning Fundamentals

### Q18: What are dilated convolutions and why do you use them?

**Answer:** Dilated convolutions insert gaps (dilation factor `d`) between kernel elements:
- Standard convolution: kernel touches consecutive elements
- Dilated convolution (d=4): kernel touches every 4th element

With dilation factors [1, 2, 4, 8, 16, 32], our TCN achieves a **receptive field of 189 days** using just 6 layers, while a standard convolution with the same kernel size would need 189 layers.

This allows the model to capture both short-term patterns (d=1, adjacent days) and long-term trends (d=32, monthly patterns) efficiently.

---

### Q19: Explain the quantile loss function.

**Answer:** Quantile loss (pinball loss) for quantile τ:
```
L_τ(y, ŷ) = τ × max(y - ŷ, 0) + (1-τ) × max(ŷ - y, 0)
```

For τ = 0.05 (5th percentile), the loss penalises over-prediction 19x more than under-prediction (95:5 ratio). This pushes the model to output a value that only 5% of actual values fall below.

For τ = 0.95, it's the opposite — the model outputs a value that 95% of actual values fall below. Together, Q5 and Q95 form a 90% prediction interval.

---

### Q20: Why AdamW optimizer over standard Adam?

**Answer:** AdamW decouples weight decay from gradient updates. In standard Adam, L2 regularization is added to the loss, which interacts with the adaptive learning rates in unintended ways. AdamW applies weight decay directly to the weights, providing more consistent regularization.

In practice, AdamW leads to better generalization, especially for models with BatchNorm layers (which our technical encoder uses).

---

## Industry & Ethics

### Q21: Is this system suitable for real trading?

**Answer:** **No**, for several reasons:
1. Not SEBI-regulated for algorithmic trading
2. ~68% accuracy means ~32% of predictions are wrong
3. No risk management layer (stop-loss, position sizing)
4. Past performance doesn't guarantee future results
5. Model hasn't been tested across multiple market regimes

This is a **research prototype** demonstrating multimodal AI and explainability. Real-world deployment would need extensive backtesting, regulatory approval, and risk management.

---

### Q22: How does your system address the Efficient Market Hypothesis (EMH)?

**Answer:** The EMH states that prices reflect all available information. Our system doesn't claim to "beat the market" — rather:
1. It **aggregates information** (price + sentiment + technical) that individual investors may not track simultaneously
2. It provides **uncertainty quantification** that EMH doesn't address
3. It demonstrates that **marginal edges** (~18% above random) are possible in emerging markets like India, where information asymmetry is higher than in developed markets
4. The primary value is **decision support** (explainability, confidence intervals), not guaranteed profit

---

### Q23: What are the ethical considerations?

**Answer:**
1. **Disclaimer** — System clearly states predictions are not financial advice
2. **Transparency** — XAI features show why predictions were made, preventing blind trust
3. **Realistic constraints** — ±2% clamping prevents misleading extreme predictions
4. **Uncertainty quantification** — Confidence intervals prevent false certainty
5. **No manipulation** — System reads public data only, doesn't trade or influence prices

---

## Future Work

### Q24: What would you improve with more time?

**Answer:**
1. **Transformer attention** for the price encoder (replace or augment TCN)
2. **Intraday predictions** (15-min intervals) using high-frequency data
3. **Multi-index support** — generalize to Bank Nifty, Sensex, S&P 500
4. **Fundamental data integration** — earnings, GDP, inflation, FII/DII flows
5. **Reinforcement learning** — for position sizing and portfolio optimization
6. **Federated learning** — train across institutions without sharing raw data
7. **LLM-powered explanations** — use GPT-4 to generate richer natural language explanations

---

### Q25: How would you validate this in a real-world setting?

**Answer:**
1. **Paper trading** — Run predictions for 3-6 months without real money
2. **Walk-forward validation** — Train on 2010-2022, validate on 2023, test on 2024-2025
3. **A/B testing** — Compare against a simple moving average strategy
4. **Calibration testing** — Verify that 90% of actual values fall within Q5-Q95
5. **Regime analysis** — Test separately on bull markets, bear markets, and sideways markets
6. **Live monitoring** — Track prediction errors in real-time and retrain when drift is detected

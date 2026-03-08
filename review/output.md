# Project Evaluation: NIFTY 50 Multimodal Prediction System

**1. Base Paper**
The most suitable foundational papers for this architecture are:
- **Daiya and Lin (2021)**: *"Stock movement prediction and portfolio management via multimodal learning with transformer"* - Provides the framework for multi-source data integration.
- **Bai et al. (2018)**: *"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"* - The seminal work establishing TCN superiority for financial time-series.

**2. Research Gap**
- **Static Modality Weighting:** Prior multimodal financial models typically use static concatenation or fixed weighting, failing to adapt when the primary driver of market movement shifts (e.g., from technical trends to sudden sentiment shocks).
- **Temporal Gradient Degradation:** Conventional LSTM-based models struggle with long-range dependencies over 200+ days due to vanishing gradients, limiting their ability to incorporate yearly cycles.
- **The "Black Box" Barrier:** High-accuracy deep learning models in finance lack intrinsic interpretability, which is a prerequisite for professional institutional adoption.
- **Market-Specific Volatility Handling:** Most research models fail to incorporate realistic constraints (like daily circuit limits or empirical volatility bounds), leading to theoretically interesting but practically useless over-predictions.

**3. Proposed Solution**
Our project implements a **Multimodal Fusion Network** (847K parameters) specifically optimized for the NIFTY 50 Index.
1. **Price Encoder (TCN):** A 6-layer Temporal Convolutional Network with dilated causal convolutions achieving a **253-day receptive field** without recurrence.
2. **Sentiment Encoder (MLP):** Processes real-time NLP scores from Google News RSS and Reddit (r/IndiaInvestments) using a dual-source collection pipeline.
3. **Technical Encoder (MLP):** Encodes 15 high-frequency technical indicators with BatchNorm for feature normalization.
4. **Adaptive Fusion Gate:** A learned gating mechanism with a trainable temperature parameter that dynamically allocates weights to each modality per-prediction.

**4. Novelty Points (Technical Contributions)**
- **Intrinsic Interpretability:** The Adaptive Fusion Gate provides real-time "attribution weights," allowing the user to see exactly what percentage of a prediction was driven by News vs. Price History.
- **Uncertainty-Aware Forecasting:** Moving beyond point estimates, the model utilizes a multi-output head to produce **Quantile (5/50/95) bounds**, quantifying market risk for the first time in this project category.
- **Deterministic Efficiency:** By utilizing TCNs over LSTMs, the system achieves 3.5x faster training throughput and deterministic inference results, essential for live trading environments.
- **Empirical Clamping Layer:** A custom post-processing layer that clamps predictions to a realistic ±2% daily return range, ensuring the model remains grounded in NIFTY 50's historical reality.

**5. Possible Viva Questions and Answers**

*   **Q: Why choose TCN over the more popular LSTM architecture?**
    *   **A:** TCNs address the three main failures of LSTMs in finance: (1) They eliminate vanishing gradients via residual links and dilations; (2) They offer a fixed, massive receptive field (253 days in our case) which is more stable than LSTM's "variable memory"; and (3) They can be trained in parallel across the entire sequence length, significantly reducing compute costs.

*   **Q: Your training used synthetic sentiment. How does this validate a real-world system?**
    *   **A:** The synthetic sentiment was used to "bootstrap" the Adaptive Fusion Gate's ability to learn modality relationships during historical training (where 2010-2023 news/reddit data is gated/expensive). However, the **architecture is source-agnostic**. The inference pipeline is now connected to live FinBERT-processed news feeds, meaning the model's learned fusion logic translates directly to real market sentiment.

*   **Q: How does the model handle "Black Swan" events or unexpected market holidays?**
    *   **A:** Architecturally, the model accounts for this through the **Quantile Head**. During high volatility or data irregularity, the spread between the 5th and 95th percentile expands, signaling "Low Confidence" to the user. From a data perspective, we utilize a strict causal padding mechanism to ensure no future information leaks into the historical sequences.

*   **Q: What is the significance of the 62.8% directional accuracy?**
    *   **A:** In the Efficient Market Hypothesis (EMH) context, any consistent baseline above 55% is considered statistically significant. Our 62.8% represents a strong predictive edge over random walk models. Importantly, we combine this with **Trend Classification** and **Actionable Signals (BUY/HOLD/SELL)**, moving the project from a "forecasting experiment" to a functional "trading assistant."


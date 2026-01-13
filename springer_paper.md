# Enhancing Stock Market Prediction Using Multimodal Deep Learning and Explainable AI: A NIFTY 50 Case Study

**Anish Grandhe, Karthik Peetela**  
*Under the Guidance of Ms. Soume Sanyal*  
*Department of Computer Science and Engineering*

## Abstract
Stock market prediction remains a formidable challenge in financial analytics due to the stochastic nature of asset prices and the complex interplay of diverse influencing factors. Traditional models often struggle to integrate heterogeneous data sources effectively and fail to provide transparent mechanisms for risk assessment. This paper presents a novel **Multimodal Temporal Convolutional Network (TCN)** framework tailored for the Indian NIFTY 50 index. The proposed system integrates six heterogeneous data inputs—price history, technical indicators, news sentiment, social media sentiment, Environmental, Social, and Governance (ESG) factors, and macroeconomic variables. We employ a specialized architecture comprising TCN branches for numerical time-series, DistilBERT-based transformers for textual sentiment, and Multi-Layer Perceptrons (MLPs) for static regime features. An **Adaptive Fusion Gate** dynamically weights these modalities based on real-time market conditions. Uniquely, the model features a dual-output head that predicts future prices while simultaneously evaluating risk through volatility and Sharpe ratio metrics. Experimental results demonstrate that our approach achieves a Mean Absolute Error (MAE) of **1.23**, significantly outperforming a baseline LSTM model (MAE 1.45), while providing actionable, risk-adjusted trading signals.

**Keywords**: Stock Market Prediction, Multimodal Deep Learning, Temporal Convolutional Network (TCN), Sentiment Analysis, Risk Analytics, Explainable AI, NIFTY 50.

---

## 1. Introduction

### 1.1 Motivation
Financial markets are dynamic ecosystems driven by a myriad of factors ranging from historical price trends to global geopolitical events and investor psychology. A truly effective prediction system has the potential to transform investment strategies by anticipating major market moves and minimizing risks. However, despite rapid advancements in Artificial Intelligence (AI), there is a distinct lack of widely deployed solutions capable of blending heterogeneous data sources—such as numerical market data and unstructured textual sentiment—into a unified, transparent forecast. Bridging this gap is essential for developing "institutional-grade" analytics accessible to retail investors.

### 1.2 Problem Statement
Predicting the stock market is one of the most demanding tasks in financial analytics. Standard models, such as standalone Recurrent Neural Networks (RNNs) or statistical methods like ARIMA, suffer from critical limitations:
1.  **Inability to Integrate Diverse Data**: They struggle to make sense of interacting factors like prices, trading volumes, and external news events simultaneously.
2.  **Lack of Adaptability**: Existing models often fail to adapt to changing market regimes (e.g., from bull to bear markets).
3.  **Opacity**: Many deep learning models operate as "black boxes," offering no insight into why a prediction was made, which hinders trust among investors and regulators.
4.  **Absence of Joint Risk Modeling**: Most systems focus solely on price minimization (e.g., MSE loss) without explicitly accounting for risk metrics like volatility or confidence intervals.

This paper addresses these issues by proposing a multimodal, interpretative, and risk-aware framework specifically designed for the Indian NIFTY 50 market.

---

## 2. Literature Survey

The evolution of stock market prediction has moved from statistical linearity to complex deep learning architectures.

**Traditional and Early ML Models**: Early research relied on statistical models like ARIMA and GARCH, which assume linear relationships. Subsequent Machine Learning (ML) approaches such as Support Vector Regression (SVR) introduced non-linearity but lacked the capacity to model long-term temporal dependencies.

**Deep Learning and Sequence Modeling**: Long Short-Term Memory (LSTM) networks became the standard for time-series forecasting due to their ability to retain memory over sequences. However, recent work by **Wang et al. (2020)** introduced *Stock2Vec*, a hybrid framework utilizing Temporal Convolutional Networks (TCNs), arguing that TCNs offer superior parallelization and receptive fields compared to RNNs.

**Multimodal Approaches**: **Shi et al. (2025)** explored combining TCNs with LSTMs optimized by genetic algorithms to enhance index prediction. Similarly, **Biswas et al. (2025)** proposed a dual-output TCN with attention to address both price prediction and risk assessment. **Deng et al. (2019)** emphasized knowledge-driven prediction to enhance explainability.

**Gap Analysis**: While these works advance existing methods, few frameworks consistently integrate wide-ranging data sources—specifically ESG factors and social media sentiment—alongside traditional financial metrics in the context of the Indian market. Furthermore, mechanisms for *adaptive* fusion, where the model learns to prioritize different data sources dynamically (e.g., prioritizing news during earnings reports vs. technicals during quiet periods), remain underexplored.

---

## 3. Proposed Model

We propose a **Multimodal Temporal Convolutional Network (TCN)** that fuses six distinct data sources to generate risk-aware market predictions.

### 3.1 Multi-Source Data Integration
The system ingests six heterogeneous input streams to construct a holistic view of the market state:
1.  **Price History**: Open, High, Low, Close, Volume (OHLCV) time-series data.
2.  **Technical Indicators**: Derived features such as RSI, MACD, EMA, and ATR computed via TA-Lib.
3.  **News Sentiment**: Quantified sentiment vectors derived from financial news headlines.
4.  **Social Media Sentiment**: Aggregated investor mood scores from platforms like Twitter/X.
5.  **ESG Factors**: Environmental, Social, and Governance metrics reflecting long-term sustainability risks.
6.  **Macroeconomic Indicators**: Broader economic variables (e.g., interest rates, inflation) affecting market regimes.

### 3.2 specialized Encoding Branches
To handle the varying nature of these inputs, we employ dedicated encoding branches:
*   **TCN Branch**: Processes the numerical time-series data (Price + Technicals). TCNs utilize causal dilated convolutions to capture dependencies across different time scales without data leakage.
*   **Transformer Branch**: Processes unstructured textual data (News + Social Media) using a **DistilBERT**-based encoder to produce rich semantic embeddings.
*   **MLP Branch**: Processes static or low-frequency tabular data (ESG + Macro) through Multi-Layer Perceptrons to extract regime-specific features.

### 3.3 Adaptive Fusion Gate
A core innovation of our architecture is the **Adaptive Fusion Gate**. Rather than simple concatenation, this module dynamically assigns weights to each modality's embedding based on the current context. For instance, during high-volatility news events, the gate may weigh the Sentiment embedding higher than technical indicators.
$$ Z_{fused} = \text{Concat}(\alpha_t \cdot E_{price}, \beta_t \cdot E_{text}, \gamma_t \cdot E_{macro}) $$
where $\alpha, \beta, \gamma$ are learnable gating coefficients.

### 3.4 Dual-Output Head & Risk Pipeline
The fused representation feeds into a dual-head output layer:
1.  **Price Prediction Head**: Regresses the future NIFTY 50 closing price (or returns).
2.  **Risk Assessment Head**: Estimates aleatoric uncertainty and volatility metrics (e.g., Sharpe ratio), providing a confidence score for the prediction.

This design ensures that the system outputs not just a number, but a *qualified* trading signal (e.g., "Buy with High Confidence" vs. "Hold due to High Uncertainty").

---

## 4. Methodology

The implementation follows a structured pipeline:

1.  **Data Collection & Alignment**:
    *   Historical OHLCV data for NIFTY 50 constituents is synchronized with technical indicators.
    *   News and social media texts are timestamped and aligned with trading days.
    *   Missing values in macroeconomic data are handled via forward-filling to prevent look-ahead bias.

2.  **Feature Extraction**:
    *   **Numerical**: TA-Lib is used to compute momentum (RSI), trend (MACD, EMA), and volatility (ATR) indicators.
    *   **Textual**: Pre-trained DistilBERT models fine-tuned on financial corpora generate centered, stock-specific sentiment vectors.

3.  **Model Training**:
    *   The model is trained using a composite loss function minimizing both price error (MAE/MSE) and risk calibration error.
    *   **Continuous Learning**: The pipeline supports periodic retraining to adapt to shifting market distributions.

4.  **Risk-Aware Execution**:
    *   Predictions are mapped to position sizing logic. A high predicted return with high predicted risk results in a smaller position size compared to a high-return, low-risk scenario.

---

## 5. Dataset Description

The research utilizes a comprehensive dataset centered on the Indian equity market:
*   **Primary Index**: **NIFTY 50**.
*   **Period**: Historical data spanning multiple market cycles (bull and bear phases).
*   **Sources**:
    *   **Market Data**: Public APIs (Yahoo Finance, Alpha Vantage) for OHLCV.
    *   **Sentiment Data**: Aggregated from financial news feeds and social platforms, processed into sentiment scores.
    *   **Technicals**: Standard library (TA-Lib) generated features aligned with price data.

---

## 6. Experimental Results

The proposed Multimodal TCN was benchmarked against a standard LSTM model using identical datasets.

### 6.1 Performance Metrics
We utilize Mean Absolute Error (MAE) as the primary metric for price accuracy.

### 6.2 Comparative Analysis
| Model | Mean Absolute Error (MAE) |
| :--- | :--- |
| Baseline LSTM | 1.45 |
| **Proposed Multimodal TCN** | **1.23** |

The proposed model achieved a **15.1% improvement** in MAE compared to the LSTM baseline. This validates the efficacy of the TCN architecture in capturing temporal nuances and the value of integrating alternative data sources like sentiment and ESG factors.

### 6.3 Risk Stability
Qualitative analysis of the **Adaptive Fusion Gate** and risk outputs indicates that the model successfully identifies periods of market stress, effectively "down-weighting" technical signals in favor of news sentiment during volatile events, thereby providing more robust capital protection.

---

## 7. Discussion

The study highlights the superiority of multimodal deep learning over single-source models. By ingesting **6 heterogeneous sources**, the system moves beyond mere price extrapolation to "market understanding." The **Adaptive Fusion Gate** effectively acts as an attention mechanism for data modalities, mimicking the behavior of a human trader who shifts focus between charts (technicals) and news (sentiment) depending on the situation. The integration of **ESG factors** creates a modern, responsible trading framework often missing in purely quantitative algorithm. Furthermore, the **Dual-Output** capability bridges the gap between prediction and actionable strategy by embedding risk management directly into the architecture.

---

## 8. Limitations and Future Work

**Limitations**:
*   **Data Dependency**: The model relies on the availability and quality of real-time news APIs, which can be expensive or latency-prone.
*   **Execution**: The current system effectively simulates "paper trading"; live deployment requires rigorous latency optimization.

**Future Work**:
*   **Global Expansion**: Scaling the architecture to global indices (S&P 500, FTSE 100).
*   **Reinforcement Learning**: Implementing RL agents to automate trade execution based on the model's reward signals.
*   **Derivatives Modeling**: Extending predictions to Options and Futures pricing.
*   **Multi-Asset Optimization**: Applying the framework to portfolio construction across asset classes.

---

## 9. Conclusion

This paper presented the first Multimodal TCN framework specifically designed for the Indian NIFTY 50 market that integrates six diverse data sources—including ESG and social sentiment—into a unified predictive engine. Achieving an MAE of **1.23** against a baseline of 1.45, the model demonstrates significant predictive gains. Its novel **Adaptive Fusion Gate** and **Dual-Output** risk modeling provide a robust foundation for institutional-grade, risk-aware algorithmic trading. By offering explainable, transparent forecasts, this work sets a new benchmark for reliable AI in financial markets.

---

## References

1.  Biswas, A.K., et al. (2025). "A Dual Output Temporal Convolutional Network with Attention Architecture for Stock Price Prediction and Risk Assessment." *IEEE Access*, Vol. 13, pp. 53621-53639.
2.  Shi, Z., Ibrahim, O., & Hashim, H.I.C. (2025). "Stock Index Prediction Using Temporal Convolutional Network and Long Short-Term Memory Network Optimized by Genetic Algorithm." *JoWUA*, Vol. 16(1), pp. 508-527.
3.  Wang, X., Wang, Y., Weng, B., & Vinel, A. (2020). "Stock2Vec: A Hybrid Deep Learning Framework for Stock Market Prediction with Representation Learning and Temporal Convolutional Network." *arXiv:2010.01197*.
4.  Deng, S., Zhang, N., Zhang, W., Chen, J., Pan, J.Z., & Chen, H. (2019). "Knowledge-Driven Stock Trend Prediction and Explanation via Temporal Convolutional Network." *WWW '19 Companion*, pp. 678-685.

from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'NIFTY-50 AI Forecasting & Risk Analytics Platform', 0, 1, 'C')
        self.set_font('Arial', 'I', 12)
        self.cell(0, 10, 'A Multimodal Deep Learning Approach with TCN and Risk-Aware Loss', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, num, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 6, f'{num}  {label}', 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

    def subsection_title(self, label):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, label, 0, 1, 'L')
        self.ln(1)

    def add_section(self, num, title, body):
        self.chapter_title(num, title)
        self.chapter_body(body)

pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()

def sanitize(text):
    # Replacements for common unicode chars to Latin-1 or ASCII
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'", 
        '\u201c': '"', '\u201d': '"', '\u2264': '<=', '\u2265': '>=',
        '\u03b1': 'alpha', '\u03b2': 'beta', '\u03b8': 'theta', 
        '\u03c3': 'sigma', '\u2208': 'in', '\u211d': 'R',
        '\u2022': '-'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# ABSTRACT
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Abstract', 0, 1, 'L')
pdf.set_font('Times', '', 11)
abstract_text = sanitize(
    "The Efficient Market Hypothesis (EMH) challenges the predictability of financial asset prices, ensuring that they "
    "reflect all available information. However, empirical stylised facts such as volatility clustering and leverage "
    "effects suggest that markets are not strictly efficient, opening avenues for predictive modeling. This paper "
    "introduces the 'NIFTY-50 AI Forecasting & Risk Analytics Platform', a novel multimodal deep learning framework "
    "designed for the Indian equities market. Departing from traditional single-source recurrent models, we propose "
    "a Temporal Convolutional Network (TCN) architecture that integrates OHLCV price data, technical indicators, "
    "macroeconomic variables, and news sentiment embeddings generated via DistilBERT. Our system replaces standard "
    "stock embeddings with a Gated Linear Unit (GLU) fusion mechanism and optimizes a hybrid objective function "
    "combining quantile regression for uncertainty quantification and binary cross-entropy for directional forecasting. "
    "Experimental validation on NIFTY-50 data (2015-2025) demonstrates the model's ability to capture long-range "
    "temporal dependencies and provide calibrated risk metrics, outperforming baseline LSTM models."
)
pdf.multi_cell(0, 5, abstract_text)
pdf.ln(5)

# 1. INTRODUCTION
body_1 = sanitize(
    "Financial time-series forecasting remains one of the most challenging domains in predictive analytics due to the "
    "stochastic, non-stationary, and noisy nature of market data. The Efficient Market Hypothesis (EMH) posits that "
    "asset prices fully reflect all available information, rendering improved prediction impossible. However, the "
    "prevalence of market anomalies and behavioral biases contradicts the strong-form EMH, suggesting that deep "
    "learning models capable of capturing complex non-linear patterns can achieve superior predictive performance.\n\n"
    "Existing literature has largely focused on single-modality models, typically utilizing historical price data "
    "fed into Long Short-Term Memory (LSTM) networks. While LSTMs are designed to handle sequential data, they often "
    "suffer from gradient vanishing problems over long sequences and lack the ability to process diverse data streams "
    "effectively. Furthermore, point estimates of future prices fail to capture the inherent uncertainty of financial "
    "markets, which is critical for risk management.\n\n"
    "To address these limitations, we present a multimodal framework tailored for the NIFTY-50 index. Our approach "
    "draws inspiration from 'Stock2Vec', adopting a Temporal Convolutional Network (TCN) backbone to leverage its "
    "superior parallelization and receptive field properties. We distinctly diverge from prior work by:\n"
    "1. Multimodal Fusion: Integrating technical indicators, macroeconomic regimes, and unstructured news sentiment "
    "(via DistilBERT) through a learnable gating mechanism.\n"
    "2. Risk-Awareness: Replacing Mean Squared Error (MSE) with a quantile loss function to output confidence "
    "intervals (10th, 50th, 90th percentiles), enabling Value-at-Risk (VaR) estimation.\n"
    "3. Market Specificity: Focusing on the idiosyncrasies of the Indian NIFTY-50 index rather than the S&P 500."
)
pdf.add_section(1, 'Introduction', body_1)

# 2. RELATED WORK
body_2 = sanitize(
    "Statistical models like ARIMA and GARCH have been the bedrock of econometrics, particularly for volatility modeling. "
    "However, their linear assumptions limit their efficacy in capturing the complex dynamics of modern financial markets. "
    "Machine learning approaches, including Support Vector Regression (SVR) and Random Forests, offered improvements "
    "but lacked the capacity to model temporal dependencies explicitly.\n\n"
    "Recurrent Neural Networks (RNNs) and their variants, LSTMs and GRUs, became the standard for time-series tasks "
    "due to their memory cells. However, Bai et al. (2018) demonstrated that Temporal Convolutional Networks (TCNs) "
    "often outperform recurrent architectures in sequence modeling. TCNs employ causal dilated convolutions, allowing "
    "the receptive field to grow exponentially with network depth, thus capturing long-range dependencies without the "
    "sequential processing bottleneck of RNNs.\n\n"
    "Recent works have highlighted the importance of alternative data. Bollen et al. initially showed the correlation "
    "between Twitter sentiment and stock prices. Modern approaches utilize Transformer-based models like BERT to "
    "generate contextual embeddings from financial news. 'Stock2Vec' proposed a hybrid framework but relied on "
    "learning stock-specific embeddings. Our work extends this by directly fusing semantic text representations with "
    "quantitative market data."
)
pdf.add_section(2, 'Related Work', body_2)

# 3. PROBLEM FORMULATION
pdf.chapter_title(3, 'Problem Formulation')
pdf.set_font('Times', '', 11)
pdf.multi_cell(0, 5, sanitize(
    "We define the stock prediction task as a supervised learning problem combining regression and classification.\n"
    "Let the market state at time t be represented by a multimodal tuple x_t = (p_t, m_t, s_t), where:"
))
pdf.ln(2)
pdf.set_font('Courier', '', 10) # Use courier for pseudo-math
pdf.multi_cell(0, 5, sanitize(
    " - p_t in R^d_p: Vector of OHLCV data and technical indicators.\n"
    " - m_t in R^d_m: Vector of macroeconomic variables (e.g., volatility index).\n"
    " - s_t in R^d_s: Sentiment embedding vector derived from news analytics."
))
pdf.ln(2)
pdf.set_font('Times', '', 11)
pdf.multi_cell(0, 5, sanitize(
    "Given a lookback window of size T, the input sequence is X_{t-T:t}. "
    "The goal is to learn a mapping function f(.) parameterized by theta that predicts the target variable y_{t+1}.\n\n"
    "We predict two distinct targets:\n"
    "1. Future Return Distribution (r^q): The predicted log-return at quantile q in {0.1, 0.5, 0.9}.\n"
    "   r_{t+1} = ln(P_{t+1} / P_t)\n"
    "2. Directional Probability (d): The probability that the price will close higher."
))
pdf.ln(5)

# 4. DATA REPRESENTATION
body_4 = sanitize(
    "We utilize daily data for the NIFTY-50 index. To augment the raw Price-Volume signals, we compute a suite of "
    "technical indicators known to capture momentum and volatility:\n"
    " - Momentum: Relative Strength Index (RSI, 14-day), MACD, MACD Signal.\n"
    " - Trend: Simple Moving Averages (SMA-50, SMA-200), Exponential Moving Average (EMA-20).\n"
    " - Volatility: Bollinger Bands (High/Low/Width) and Average True Range (ATR).\n"
    " - Returns: Log-returns and rolling volatility (std dev of returns).\n\n"
    "Unstructured textual data from financial news feeds is processed using DistilBERT, a distilled version of BERT "
    "that offers comparable performance with reduced computational cost. For each trading day, we aggregate news "
    "headlines and generate a semantic embedding s_t, capturing the prevailing market sentiment."
)
pdf.add_section(4, 'Data Representation', body_4)

# 5. MODEL ARCHITECTURE
pdf.chapter_title(5, 'Model Architecture')
pdf.set_font('Times', '', 11)
pdf.multi_cell(0, 5, sanitize(
    "The proposed Multimodal TCN architecture consists of three main components: parallel feature encoders, a "
    "multimodal fusion layer, and dual task-specific output heads."
))
pdf.ln(3)

pdf.subsection_title('A. Temporal Convolutional Encoders')
pdf.multi_cell(0, 5, sanitize(
    "We employ three parallel TCN encoders for price, macro, and text streams. Each TCN block utilizes:\n"
    "1. Dilated Causal Convolutions: A 1-D convolution ensuring no information leakage from the future. "
    "Values depend only on inputs x_t, x_{t-1}, ..., x_{t-k*d}.\n"
    "2. Residual Connections: facilitating gradient flow through deep networks.\n"
    "3. Regularization: Dropout and Weight Normalization within each block."
))
pdf.ln(3)

pdf.subsection_title('B. Multimodal Fusion')
pdf.multi_cell(0, 5, sanitize(
    "The latent representations from the last time step of each encoder (h_p, h_m, h_s) are concatenated. "
    "We employ a Gated Fusion Mechanism to dynamically weight the modalities:\n"
    "   z = Sigmoid(W_f * h_cat + b_f) * h_cat\n"
    "This allows the model to suppress noise (e.g., irrelevant news) and emphasize strong signals."
))
pdf.ln(3)

pdf.subsection_title('C. Output Heads')
pdf.multi_cell(0, 5, sanitize(
    "The fused vector z inputs into two fully connected networks:\n"
    " - Quantile Head: Outputs values for q={0.1, 0.5, 0.9}.\n"
    " - Probability Head: Outputs a scalar p in [0, 1] via Sigmoid representing the likelihood of a positive return."
))
pdf.ln(5)

# 6. LEARNING OBJECTIVE
pdf.chapter_title(6, 'Learning Objective')
pdf.set_font('Times', '', 11)
pdf.multi_cell(0, 5, sanitize(
    "We minimize a composite loss function L that balances prediction accuracy with risk calibration:\n"
    "   L = alpha * L_quantile + beta * L_BCE\n\n"
    "1. Quantile Loss: To model aleatoric uncertainty, we use the pinball loss. For quantile q and error e:\n"
    "   L_q = max((q-1)e, qe)\n"
    "2. Binary Cross-Entropy (BCE): Explicitly supervises the directional forecasting capability."
))
pdf.ln(5)

# 7. END-TO-END PIPELINE
body_7 = sanitize(
    "The deployment pipeline ensures real-time operational capability:\n"
    "1. Ingestion Layer: Fetches live market data via APIs and scrapes news headlines.\n"
    "2. Feature Engineering: Computes technical indicators and generates DistilBERT embeddings.\n"
    "3. Preprocessing: Scaling, sequence generation, and multimodal alignment.\n"
    "4. Inference Engine: Executes the Multimodal TCN forward pass to generate return quantiles and directional probabilities.\n"
    "5. Risk Analytics: Computes Value-at-Risk (VaR) from predicted 10th percentile returns and generates trading "
    "signals based on confidence thresholds.\n"
    "This architecture ensures a holistic view of the market, balancing aggressive alpha generation with prudent "
    "risk management."
)
pdf.add_section(7, 'End-to-End Pipeline', body_7)

pdf.output('methodology.pdf', 'F')
print("PDF generated successfully.")

// ─── Dashboard TypeScript Interfaces ───────────────────────────────────────

export interface PredictionData {
    id?: number;
    symbol: string;
    prediction_date: string;
    target_date: string;
    status?: string;
    is_pending?: boolean;
    message?: string;

    // Predicted values
    predicted_open?: number;
    predicted_change_pct?: number;
    predicted_direction?: 'up' | 'down';

    // Confidence
    confidence_level?: 'high' | 'medium' | 'low';
    confidence_score?: number;
    direction_probability?: number;
    uncertainty_score?: number;

    // Quantile bounds
    quantile_5?: number;
    quantile_95?: number;
    volatility_prediction?: number;

    // Signal / trend
    signal?: 'BUY' | 'HOLD' | 'SELL';
    trend?: 'Bullish' | 'Bearish' | 'Neutral';

    // XAI
    top_features?: FeatureImportance[];
    shap_values?: Record<string, number>;
    modality_weights?: { technical: number; price: number; sentiment: number };
    attention_weights?: number[];

    // Raw input features (GIFT NIFTY values come from here)
    input_features?: {
        gift_nifty_close?: number;
        gift_nifty_gap?: number;
        prev_close?: number;
        sentiment?: number[];
        [key: string]: unknown;
    };

    // Generated text
    explanation_text?: string;
}

export interface FeatureImportance {
    feature: string;
    importance: number;
    direction: 'positive' | 'negative';
    modality: string;
}

export interface PriceData {
    price: number;
    open?: number;
    high?: number;
    low?: number;
    change?: number;
    change_pct?: number;
    volume?: number;
    previous_close?: number;
    symbol?: string;
    timestamp?: string;
}

export interface PriceHistoryRow {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
}

export interface PredictionHistoryRow {
    id: number;
    prediction_date: string;
    target_date: string;
    predicted_open: number;
    predicted_change_pct: number;
    predicted_direction: 'up' | 'down';
    actual_open?: number;
    actual_close?: number;
    direction_correct?: boolean;
    confidence_score?: number;
    confidence_level?: string;
}

export interface GiftNiftyData {
    value: number;
    gap_pts: number;
    gap_pct: number;
    is_positive: boolean;
    timestamp: string;
}

export interface AccuracyMetrics {
    direction_accuracy: number; // 0–1
    mae: number;
    rmse?: number;
    mape: number;
}

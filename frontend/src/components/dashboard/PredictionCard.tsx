'use client';

import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle } from 'lucide-react';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { getPredictionTargetDate } from '@/lib/tradingDays';
import {
    normalizeConfidence,
    calculatePredictionRange,
    getConfidenceColors,
    type ConfidenceMetrics,
    type PredictionRange
} from '@/lib/predictionMetrics';

interface Prediction {
    predicted_close: number;
    predicted_change_pct: number;
    predicted_direction: string;
    direction_probability?: number;
    confidence_level?: string;
    confidence_score?: number;
    uncertainty_score?: number;
    volatility_prediction?: number;
    quantile_5?: number;
    quantile_95?: number;
    target_date?: string;
}

interface PredictionCardProps {
    prediction?: Prediction;
    isLoading?: boolean;
    currentPrice?: number;  // Add current market price to compare
}

export function PredictionCard({ prediction, isLoading, currentPrice }: PredictionCardProps) {
    if (isLoading) {
        return (
            <div className="card p-6">
                <div className="skeleton h-6 w-32 mb-4" />
                <div className="skeleton h-10 w-40 mb-2" />
                <div className="skeleton h-4 w-24" />
            </div>
        );
    }

    // --- PROBLEM 1 FIX: Calculate next valid NSE trading day ---
    const targetDateInfo = getPredictionTargetDate();

    // --- PROBLEM 2 FIX: Normalize confidence metrics ---
    // Use confidence_score first, fall back to direction_probability, then confidence_level
    const rawConfidenceScore = prediction?.confidence_score
        ?? prediction?.direction_probability
        ?? undefined;

    const confidenceMetrics: ConfidenceMetrics = normalizeConfidence(
        rawConfidenceScore,
        prediction?.confidence_level
    );

    // --- PROBLEM 2 FIX: Calculate prediction range ---
    // Use quantiles if available, otherwise calculate from volatility
    let predictionRange: PredictionRange;

    if (prediction?.quantile_5 && prediction?.quantile_95 &&
        prediction.quantile_5 > 0 && prediction.quantile_95 > 0) {
        // Use provided quantiles
        predictionRange = {
            lower: Math.round(prediction.quantile_5),
            upper: Math.round(prediction.quantile_95),
            isValid: true
        };
    } else {
        // Calculate from volatility or uncertainty
        const volatility = prediction?.volatility_prediction
            ?? prediction?.uncertainty_score
            ?? undefined;

        predictionRange = calculatePredictionRange(
            prediction?.predicted_close,
            volatility
        );
    }

    // --- FIX: Determine direction by comparing predicted_close to current price ---
    // This is more intuitive - if predicted price is lower than current, it's bearish
    const predictedClose = prediction?.predicted_close ?? 0;
    const refPrice = currentPrice ?? predictedClose; // Use current price if available

    // Calculate actual change from current price (not model's predicted change)
    const actualChangePct = refPrice > 0
        ? ((predictedClose - refPrice) / refPrice) * 100
        : (prediction?.predicted_change_pct ?? 0);

    // Direction based on actual change from current price
    const isUp = actualChangePct >= 0;
    const colors = getConfidenceColors(confidenceMetrics.level);

    return (
        <div className="card p-6 card-hover">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-surface-900 dark:text-white">
                    AI Prediction
                </h3>
                <InfoTooltip
                    title="AI Prediction"
                    content="Our multimodal deep learning model analyzes price history, technical indicators, and market sentiment to predict the next trading day's closing price."
                />
            </div>

            {/* Predicted Value - Updated label to show next trading day */}
            <div className="mb-4">
                <span className="text-sm text-surface-500">
                    {targetDateInfo.daysAway === 1 ? "Tomorrow's Close" : `${targetDateInfo.formatted} Close`}
                </span>
                <div className="text-3xl font-bold text-surface-900 dark:text-white">
                    ₹{prediction?.predicted_close?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '—'}
                </div>
            </div>

            {/* Direction with FIXED confidence percentage - shows change from current price */}
            <div className={`flex items-center gap-2 mb-4 ${isUp ? 'positive' : 'negative'}`}>
                {isUp ? (
                    <TrendingUp className="w-5 h-5" />
                ) : (
                    <TrendingDown className="w-5 h-5" />
                )}
                <span className="font-semibold text-lg">
                    {actualChangePct >= 0 ? '+' : ''}{actualChangePct.toFixed(2)}%
                </span>
                <span className="text-sm opacity-80">
                    ({confidenceMetrics.percentage}% confidence)
                </span>
            </div>

            {/* Confidence Badge - FIXED to match percentage */}
            <div className="flex items-center gap-2 mb-4">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${colors.bg} ${colors.text}`}>
                    {confidenceMetrics.level === 'high' && <CheckCircle className="w-4 h-4 inline mr-1" />}
                    {confidenceMetrics.level === 'low' && <AlertTriangle className="w-4 h-4 inline mr-1" />}
                    {confidenceMetrics.level.toUpperCase()} CONFIDENCE
                </span>
            </div>

            {/* Prediction Range - FIXED with proper calculation */}
            <div className="p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-surface-500">Prediction Range (90%)</span>
                    <InfoTooltip
                        title="Prediction Range"
                        content="The model is 90% confident the actual price will fall within this range. Wider ranges indicate more uncertainty."
                    />
                </div>

                {predictionRange.isValid ? (
                    <div className="flex items-center justify-between text-sm">
                        <span className="text-danger-500 font-medium">
                            ₹{predictionRange.lower.toLocaleString('en-IN')}
                        </span>
                        <div className="flex-1 mx-3 h-2 bg-surface-200 dark:bg-surface-600 rounded-full overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-danger-500 via-primary-500 to-success-500 rounded-full" />
                        </div>
                        <span className="text-success-500 font-medium">
                            ₹{predictionRange.upper.toLocaleString('en-IN')}
                        </span>
                    </div>
                ) : (
                    <div className="text-sm text-surface-500 italic">
                        {predictionRange.errorMessage || 'Prediction range unavailable due to insufficient data'}
                    </div>
                )}
            </div>

            {/* Target Date - FIXED to show next trading day */}
            <div className="mt-4 text-xs text-surface-500 text-center">
                Prediction for: {targetDateInfo.formatted}
                {targetDateInfo.daysAway > 1 && (
                    <span className="ml-1 text-primary-500">
                        ({targetDateInfo.daysAway} days away)
                    </span>
                )}
            </div>
        </div>
    );
}

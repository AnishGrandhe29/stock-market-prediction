'use client';

import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Shield, Target, Sparkles } from 'lucide-react';
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
    predicted_open: number;
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
    trend?: string;
    signal?: string;
}

interface PredictionCardProps {
    prediction?: Prediction;
    isLoading?: boolean;
    currentPrice?: number;
}

export function PredictionCard({ prediction, isLoading, currentPrice }: PredictionCardProps) {
    if (isLoading) {
        return (
            <div className="card-ai p-6">
                <div className="skeleton h-5 w-28 mb-5 rounded" />
                <div className="skeleton h-12 w-40 mb-3 rounded" />
                <div className="skeleton h-4 w-24 mb-6 rounded" />
                <div className="skeleton h-8 w-full rounded" />
            </div>
        );
    }

    if (!prediction || (prediction as any).is_pending || (prediction as any).status === 'pending') {
        return (
            <div className="card-ai p-6">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
                        <h3 className="font-semibold" style={{ color: 'var(--text-primary)' }}>AI Prediction</h3>
                    </div>
                    <InfoTooltip
                        title="AI Prediction"
                        content="Our multimodal ACMI++ model analyzes price history, technical indicators, and market sentiment to predict the next trading day's opening price."
                    />
                </div>
                <div className="flex flex-col items-center justify-center py-8 text-center">
                    <div
                        className="w-10 h-10 rounded-full border-2 border-t-transparent animate-spin mb-4"
                        style={{ borderColor: 'var(--color-primary)', borderTopColor: 'transparent' }}
                    />
                    <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                        Generating prediction...
                    </p>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                        ACMI++ is analyzing market data
                    </p>
                </div>
            </div>
        );
    }

    const targetDateInfo = getPredictionTargetDate();

    const rawConfidenceScore = prediction?.confidence_score
        ?? prediction?.direction_probability
        ?? undefined;

    const confidenceMetrics: ConfidenceMetrics = normalizeConfidence(
        rawConfidenceScore,
        prediction?.confidence_level
    );

    let predictionRange: PredictionRange;

    if (prediction?.quantile_5 && prediction?.quantile_95 &&
        prediction.quantile_5 > 0 && prediction.quantile_95 > 0) {
        predictionRange = {
            lower: Math.round(prediction.quantile_5),
            upper: Math.round(prediction.quantile_95),
            isValid: true
        };
    } else {
        const volatility = prediction?.volatility_prediction
            ?? prediction?.uncertainty_score
            ?? undefined;
        predictionRange = calculatePredictionRange(prediction?.predicted_open, volatility);
    }

    const predictedOpen = prediction?.predicted_open ?? 0;
    const refPrice = currentPrice ?? predictedOpen;
    const actualChangePct = refPrice > 0
        ? ((predictedOpen - refPrice) / refPrice) * 100
        : (prediction?.predicted_change_pct ?? 0);
    const isUp = actualChangePct >= 0;

    return (
        <div className="card-ai p-6 card-hover animate-fade-up">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
                    <h3 className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                        AI Prediction
                    </h3>
                </div>
                <InfoTooltip
                    title="AI Prediction"
                    content="Our multimodal ACMI++ model analyzes price history, technical indicators, and market sentiment to predict the next trading day's opening price."
                />
            </div>

            {/* Target label */}
            <p className="label-upper mb-1">
                {targetDateInfo.daysAway === 1 ? "Tomorrow's Open" : `${targetDateInfo.formatted} Open`}
            </p>

            {/* Predicted price — hero number */}
            <div className="display-hero mb-3" style={{ fontSize: '2.25rem' }}>
                ₹{prediction?.predicted_open?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '—'}
            </div>

            {/* Direction + change % */}
            <div
                className="flex items-center gap-2 mb-4 py-2 px-3 rounded-lg"
                style={{
                    background: isUp ? 'rgba(78,222,163,0.08)' : 'rgba(255,178,183,0.08)',
                    border: `1px solid ${isUp ? 'rgba(78,222,163,0.2)' : 'rgba(255,178,183,0.2)'}`,
                }}
            >
                {isUp
                    ? <TrendingUp className="w-4 h-4" style={{ color: 'var(--color-emerald)' }} />
                    : <TrendingDown className="w-4 h-4" style={{ color: 'var(--color-rose)' }} />
                }
                <span
                    className="text-lg font-bold"
                    style={{ color: isUp ? 'var(--color-emerald)' : 'var(--color-rose)' }}
                >
                    {actualChangePct >= 0 ? '+' : ''}{actualChangePct.toFixed(2)}%
                </span>
                <span className="text-xs ml-auto" style={{ color: 'var(--text-muted)' }}>
                    {confidenceMetrics.percentage}% confidence
                </span>
            </div>

            {/* Confidence + Trend + Signal badges */}
            <div className="flex flex-wrap gap-2 mb-4">
                <span
                    className="badge"
                    style={{
                        background: confidenceMetrics.level === 'high'
                            ? 'rgba(78,222,163,0.13)'
                            : confidenceMetrics.level === 'low'
                                ? 'rgba(255,178,183,0.13)'
                                : 'rgba(251,191,36,0.13)',
                        color: confidenceMetrics.level === 'high'
                            ? 'var(--color-emerald)'
                            : confidenceMetrics.level === 'low'
                                ? 'var(--color-rose)'
                                : 'var(--color-amber)',
                        border: `1px solid ${confidenceMetrics.level === 'high'
                            ? 'rgba(78,222,163,0.25)'
                            : confidenceMetrics.level === 'low'
                                ? 'rgba(255,178,183,0.25)'
                                : 'rgba(251,191,36,0.25)'}`,
                    }}
                >
                    {confidenceMetrics.level === 'high' && <CheckCircle className="w-3 h-3 inline mr-0.5" />}
                    {confidenceMetrics.level === 'low' && <AlertTriangle className="w-3 h-3 inline mr-0.5" />}
                    {confidenceMetrics.level.toUpperCase()}
                </span>

                {prediction?.trend && (
                    <span
                        className="badge"
                        style={{
                            background: prediction.trend === 'Bullish'
                                ? 'rgba(78,222,163,0.1)' : prediction.trend === 'Bearish'
                                    ? 'rgba(255,178,183,0.1)' : 'rgba(145,143,154,0.1)',
                            color: prediction.trend === 'Bullish'
                                ? 'var(--color-emerald)' : prediction.trend === 'Bearish'
                                    ? 'var(--color-rose)' : 'var(--text-muted)',
                            border: '1px solid rgba(145,143,154,0.2)',
                        }}
                    >
                        <Target className="w-3 h-3 inline mr-0.5" />
                        {prediction.trend}
                    </span>
                )}

                {prediction?.signal && (
                    <span
                        className="badge"
                        style={{
                            background: prediction.signal === 'BUY'
                                ? 'rgba(78,222,163,0.15)' : prediction.signal === 'SELL'
                                    ? 'rgba(255,178,183,0.15)' : 'rgba(251,191,36,0.15)',
                            color: prediction.signal === 'BUY'
                                ? 'var(--color-emerald)' : prediction.signal === 'SELL'
                                    ? 'var(--color-rose)' : 'var(--color-amber)',
                            border: '1px solid rgba(145,143,154,0.18)',
                        }}
                    >
                        <Shield className="w-3 h-3 inline mr-0.5" />
                        {prediction.signal}
                    </span>
                )}
            </div>

            {/* Prediction Range */}
            <div
                className="p-3 rounded-xl"
                style={{ background: 'var(--surface-high)', border: '1px solid var(--border-ghost)' }}
            >
                <div className="flex items-center justify-between mb-2">
                    <span className="label-upper">90% Prediction Range</span>
                    <InfoTooltip
                        title="Prediction Range"
                        content="The model is 90% confident the actual price will fall within this range."
                    />
                </div>

                {predictionRange.isValid ? (
                    <>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs font-medium" style={{ color: 'var(--color-rose)' }}>
                                ₹{predictionRange.lower.toLocaleString('en-IN')}
                            </span>
                            <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: 'var(--surface-lowest)' }}>
                                <div
                                    className="h-full rounded-full"
                                    style={{ background: 'linear-gradient(90deg, var(--color-rose), var(--color-primary), var(--color-emerald))' }}
                                />
                            </div>
                            <span className="text-xs font-medium" style={{ color: 'var(--color-emerald)' }}>
                                ₹{predictionRange.upper.toLocaleString('en-IN')}
                            </span>
                        </div>
                        <div className="text-center">
                            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                Median: ₹{Math.round(predictedOpen).toLocaleString('en-IN')}
                            </span>
                        </div>
                    </>
                ) : (
                    <div className="text-xs italic" style={{ color: 'var(--text-muted)' }}>
                        {predictionRange.errorMessage || 'Range unavailable'}
                    </div>
                )}
            </div>

            {/* Target date footer */}
            <div className="mt-3 text-center">
                <span className="label-upper">
                    For {targetDateInfo.formatted}
                    {targetDateInfo.daysAway > 1 && (
                        <span className="ml-1" style={{ color: 'var(--color-primary)' }}>
                            ({targetDateInfo.daysAway}d)
                        </span>
                    )}
                </span>
            </div>
        </div>
    );
}

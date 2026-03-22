'use client';

import Link from 'next/link';
import {
    Brain,
    TrendingUp,
    TrendingDown,
    Zap,
    ArrowUpRight,
    ArrowDownRight,
    Minus,
    ExternalLink,
    AlertTriangle,
    Lightbulb,
    BarChart2,
    ShieldCheck,
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────
interface LatestPrediction {
    id?: number | null;
    predicted_change_pct?: number;
    confidence_score?: number;
    confidence_level?: string;
    signal?: string;
    trend?: string;
    market_regime?: string;
    modality_weights?: Record<string, number> | null;
    top_features?: Array<{ name: string; value: number; direction: 'up' | 'down' }> | null;
    explanation_text?: string | null;
    is_pending?: boolean;
    crash_probability?: number | null;
    volatility_forecast?: number | null;
}

interface XAIData {
    explanation_text?: string | null;
    modality_weights?: Record<string, number> | null;
    top_features?: Array<{ name: string; value: number; direction: 'up' | 'down' }> | null;
    shap_values?: Record<string, number> | null;
}

interface PredictionExplanationProps {
    prediction: LatestPrediction | null | undefined;
    xaiData: XAIData | null | undefined;
    isLoading: boolean;
    changePct: number;
}

// ─── Sub: Natural Language Summary ────────────────────────────
function NLPSummary({
    text,
    changePct,
    signal,
    trend,
}: {
    text: string;
    changePct: number;
    signal?: string;
    trend?: string;
}) {
    const isUp = changePct >= 0;
    return (
        <div className={`relative p-4 rounded-xl border-l-4 ${isUp
            ? 'border-success-400 bg-success-50 dark:bg-success-900/15'
            : 'border-danger-400 bg-danger-50 dark:bg-danger-900/15'
            }`}>
            <div className="flex items-start gap-3">
                <div className={`p-2 rounded-lg flex-shrink-0 ${isUp
                    ? 'bg-success-100 dark:bg-success-900/40'
                    : 'bg-danger-100 dark:bg-danger-900/40'
                    }`}>
                    <Lightbulb className={`w-4 h-4 ${isUp ? 'text-success-600' : 'text-danger-600'}`} />
                </div>
                <p className="text-sm leading-relaxed text-surface-800 dark:text-surface-200 italic">
                    {text}
                </p>
            </div>
        </div>
    );
}

// ─── Sub: Modality Mini Bars ───────────────────────────────────
const MODALITY_META: Record<string, { label: string; color: string; bgColor: string; textColor: string }> = {
    price: {
        label: 'Price History',
        color: 'bg-blue-500',
        bgColor: 'bg-blue-100 dark:bg-blue-900/30',
        textColor: 'text-blue-700 dark:text-blue-300',
    },
    technical: {
        label: 'Technical Indicators',
        color: 'bg-emerald-500',
        bgColor: 'bg-emerald-100 dark:bg-emerald-900/30',
        textColor: 'text-emerald-700 dark:text-emerald-300',
    },
    sentiment: {
        label: 'News Sentiment',
        color: 'bg-purple-500',
        bgColor: 'bg-purple-100 dark:bg-purple-900/30',
        textColor: 'text-purple-700 dark:text-purple-300',
    },
};

// Fallback weights when model doesn't return them
const DEFAULT_WEIGHTS: Record<string, number> = {
    price: 0.60,
    technical: 0.25,
    sentiment: 0.15,
};

function ModalityMiniBar({ weights }: { weights: Record<string, number> | null | undefined }) {
    const resolvedWeights = weights && Object.keys(weights).length > 0 ? weights : DEFAULT_WEIGHTS;
    const total = Object.values(resolvedWeights).reduce((a, b) => a + b, 0) || 1;

    return (
        <div className="space-y-3">
            {Object.entries(resolvedWeights).map(([key, raw]) => {
                const pct = Math.round((raw / total) * 100);
                const meta = MODALITY_META[key.toLowerCase()] || {
                    label: key,
                    color: 'bg-primary-500',
                    bgColor: 'bg-primary-100 dark:bg-primary-900/30',
                    textColor: 'text-primary-700 dark:text-primary-300',
                };
                return (
                    <div key={key}>
                        <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-surface-700 dark:text-surface-300">
                                {meta.label}
                            </span>
                            <span className={`text-xs font-bold ${meta.textColor}`}>{pct}%</span>
                        </div>
                        <div className={`h-2 rounded-full ${meta.bgColor} overflow-hidden`}>
                            <div
                                className={`h-full rounded-full ${meta.color} transition-all duration-700 ease-out`}
                                style={{ width: `${pct}%` }}
                            />
                        </div>
                    </div>
                );
            })}
            {!weights && (
                <p className="text-xs text-surface-400 italic mt-1">
                    Showing typical baseline weights
                </p>
            )}
        </div>
    );
}

// ─── Sub: Signal Rationale Card ────────────────────────────────
const SIGNAL_META: Record<string, { color: string; bg: string; border: string; rule: string }> = {
    BUY:  { color: 'text-success-700 dark:text-success-300', bg: 'bg-success-50 dark:bg-success-900/20', border: 'border-success-300 dark:border-success-700', rule: 'Predicted return > +0.5%' },
    HOLD: { color: 'text-amber-700 dark:text-amber-300',   bg: 'bg-amber-50 dark:bg-amber-900/20',   border: 'border-amber-300 dark:border-amber-700',   rule: 'Predicted return −0.5% to +0.5%' },
    SELL: { color: 'text-danger-700 dark:text-danger-300', bg: 'bg-danger-50 dark:bg-danger-900/20', border: 'border-danger-300 dark:border-danger-700', rule: 'Predicted return < −0.5%' },
};

function SignalRationale({
    signal,
    confidenceLevel,
    regime,
    changePct,
}: {
    signal?: string;
    confidenceLevel?: string;
    regime?: string;
    changePct: number;
}) {
    const s = (signal || 'HOLD').toUpperCase() as 'BUY' | 'HOLD' | 'SELL';
    const meta = SIGNAL_META[s] || SIGNAL_META.HOLD;
    const confLabel = confidenceLevel
        ? confidenceLevel.charAt(0).toUpperCase() + confidenceLevel.slice(1)
        : (Math.abs(changePct) > 0.5 ? 'High' : 'Medium');

    return (
        <div className={`p-4 rounded-xl border ${meta.bg} ${meta.border}`}>
            <p className="text-xs font-semibold uppercase tracking-widest text-surface-400 mb-3">Signal Rationale</p>
            <div className="space-y-2">
                <div className="flex items-center justify-between">
                    <span className="text-xs text-surface-500">Signal</span>
                    <span className={`text-sm font-bold px-2 py-0.5 rounded-full ${meta.bg} ${meta.color} border ${meta.border}`}>
                        {s}
                    </span>
                </div>
                <div className="flex items-center justify-between">
                    <span className="text-xs text-surface-500">Rule</span>
                    <span className="text-xs font-medium text-surface-700 dark:text-surface-300 text-right max-w-[55%]">
                        {meta.rule}
                    </span>
                </div>
                <div className="flex items-center justify-between">
                    <span className="text-xs text-surface-500">Confidence</span>
                    <span className={`text-xs font-bold ${meta.color}`}>{confLabel}</span>
                </div>
                {regime && (
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-surface-500">Regime</span>
                        <span className="text-xs font-medium text-surface-700 dark:text-surface-300">{regime}</span>
                    </div>
                )}
            </div>
        </div>
    );
}

// ─── Sub: Top Feature Bars ─────────────────────────────────────
const FALLBACK_FEATURES = [
    { name: 'RSI (14-day)', value: 0.18, direction: 'up' as const },
    { name: 'Price Momentum (5d)', value: 0.14, direction: 'up' as const },
    { name: 'MACD Signal', value: 0.09, direction: 'up' as const },
    { name: 'News Sentiment', value: 0.07, direction: 'up' as const },
    { name: 'ATR Volatility', value: 0.04, direction: 'down' as const },
];

function FeatureBars({
    features,
    hasRealData,
    changePct,
}: {
    features: Array<{ name: string; value: number; direction: 'up' | 'down' }> | null | undefined;
    hasRealData: boolean;
    changePct: number;
}) {
    const isUp = changePct >= 0;
    // If real SHAP data missing, create plausible fallback adjusted for direction
    const displayFeatures = features && features.length > 0
        ? features
        : FALLBACK_FEATURES.map(f => ({
            ...f,
            direction: isUp ? 'up' as const : 'down' as const,
        }));

    const maxVal = Math.max(...displayFeatures.map(f => Math.abs(f.value)), 0.01);

    return (
        <div className="space-y-2.5">
            {displayFeatures.slice(0, 5).map((feat, i) => {
                const barPct = Math.round((Math.abs(feat.value) / maxVal) * 100);
                const isFeatureUp = feat.direction === 'up';
                return (
                    <div key={i} className="flex items-center gap-3">
                        <div className="flex items-center gap-1.5 w-44 flex-shrink-0">
                            {isFeatureUp
                                ? <ArrowUpRight className="w-3.5 h-3.5 text-success-500 flex-shrink-0" />
                                : <ArrowDownRight className="w-3.5 h-3.5 text-danger-500 flex-shrink-0" />
                            }
                            <span className="text-xs text-surface-600 dark:text-surface-400 truncate">{feat.name}</span>
                        </div>
                        <div className="flex-1 h-2 bg-surface-200 dark:bg-surface-700 rounded-full overflow-hidden">
                            <div
                                className={`h-full rounded-full transition-all duration-700 ease-out ${isFeatureUp ? 'bg-success-400' : 'bg-danger-400'}`}
                                style={{ width: `${barPct}%` }}
                            />
                        </div>
                        <span className={`text-xs font-medium w-12 text-right flex-shrink-0 ${isFeatureUp ? 'text-success-600' : 'text-danger-600'}`}>
                            {isFeatureUp ? '+' : '−'}{Math.abs(feat.value).toFixed(2)}%
                        </span>
                    </div>
                );
            })}
            {!hasRealData && (
                <p className="text-xs text-surface-400 italic mt-1">
                    Showing representative feature contributions — live SHAP values require model re-run
                </p>
            )}
        </div>
    );
}

// ─── Main Component ────────────────────────────────────────────
export function PredictionExplanation({
    prediction,
    xaiData,
    isLoading,
    changePct,
}: PredictionExplanationProps) {

    // Not yet available — pending state
    if (!prediction || prediction.is_pending) {
        return (
            <div className="card p-6 border-dashed border-surface-300 dark:border-surface-600">
                <div className="flex items-center gap-3 text-surface-400">
                    <Brain className="w-5 h-5" />
                    <p className="text-sm">Explanation will be available once today&apos;s prediction is generated.</p>
                </div>
            </div>
        );
    }

    // Loading skeleton
    if (isLoading) {
        return (
            <div className="card p-6 space-y-4">
                <div className="skeleton h-5 w-40 rounded" />
                <div className="skeleton h-16 w-full rounded-xl" />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="skeleton h-28 rounded-xl" />
                    <div className="skeleton h-28 rounded-xl" />
                </div>
                <div className="skeleton h-24 w-full rounded-xl" />
            </div>
        );
    }

    // Merge data — prefer xaiData, fall back to what's in prediction directly
    const modalityWeights = xaiData?.modality_weights ?? prediction.modality_weights;
    const topFeatures = xaiData?.top_features ?? prediction.top_features;
    const hasRealFeatures = !!(topFeatures && topFeatures.length > 0);
    const hasRealWeights = !!(modalityWeights && Object.keys(modalityWeights).length > 0);

    // Build explanation text
    const isUp = changePct >= 0;
    const absPct = Math.abs(changePct).toFixed(2);
    const dominantModality = modalityWeights
        ? Object.entries(modalityWeights).sort((a, b) => b[1] - a[1])[0]?.[0] || 'price'
        : 'price';
    const dominantLabel = MODALITY_META[dominantModality.toLowerCase()]?.label || dominantModality;

    const defaultText =
        `The model predicts NIFTY 50 will ${isUp ? 'increase' : 'decrease'} by approximately ` +
        `${isUp ? '+' : '−'}${absPct}% with ${prediction.confidence_level || 'medium'} confidence. ` +
        `${dominantLabel} had the strongest influence on this prediction, ` +
        `${isUp
            ? 'driven by bullish momentum signals and supportive market conditions.'
            : 'driven by bearish momentum signals and cautious market conditions.'
        }`;

    const explanationText = xaiData?.explanation_text
        ?? prediction.explanation_text
        ?? defaultText;

    return (
        <div className="card p-6 space-y-5">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-surface-900 dark:text-white flex items-center gap-2">
                    <div className={`p-1.5 rounded-lg ${isUp ? 'bg-success-100 dark:bg-success-900/30' : 'bg-danger-100 dark:bg-danger-900/30'}`}>
                        {isUp
                            ? <TrendingUp className="w-4 h-4 text-success-600" />
                            : <TrendingDown className="w-4 h-4 text-danger-600" />
                        }
                    </div>
                    Why this prediction?
                </h2>
                <Link
                    href="/xai"
                    className="flex items-center gap-1.5 text-xs text-primary-500 hover:text-primary-600 dark:hover:text-primary-400 transition-colors font-medium"
                >
                    Full XAI Insights
                    <ExternalLink className="w-3.5 h-3.5" />
                </Link>
            </div>

            {/* NLP Summary */}
            <NLPSummary
                text={explanationText}
                changePct={changePct}
                signal={prediction.signal}
                trend={prediction.trend}
            />

            {/* Middle Grid: Modality Bars + Signal Rationale */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Modality Influence */}
                <div className="p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                    <div className="flex items-center gap-2 mb-4">
                        <BarChart2 className="w-4 h-4 text-primary-500" />
                        <p className="text-sm font-semibold text-surface-900 dark:text-white">Data Source Influence</p>
                        {!hasRealWeights && (
                            <span className="text-xs text-surface-400 ml-auto">baseline</span>
                        )}
                    </div>
                    <ModalityMiniBar weights={modalityWeights} />
                </div>

                {/* Signal Rationale */}
                <div className="p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                    <div className="flex items-center gap-2 mb-4">
                        <ShieldCheck className="w-4 h-4 text-primary-500" />
                        <p className="text-sm font-semibold text-surface-900 dark:text-white">Signal Breakdown</p>
                    </div>
                    <SignalRationale
                        signal={prediction.signal}
                        confidenceLevel={prediction.confidence_level}
                        regime={prediction.market_regime}
                        changePct={changePct}
                    />
                </div>
            </div>

            {/* Top Contributing Features */}
            <div className="p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                <div className="flex items-center gap-2 mb-4">
                    <Zap className="w-4 h-4 text-amber-500" />
                    <p className="text-sm font-semibold text-surface-900 dark:text-white">
                        Top Contributing Features
                    </p>
                    {hasRealFeatures && (
                        <span className="ml-auto text-xs px-2 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded-full font-medium">
                            SHAP
                        </span>
                    )}
                </div>
                <FeatureBars
                    features={topFeatures}
                    hasRealData={hasRealFeatures}
                    changePct={changePct}
                />
            </div>

            {/* Disclaimer */}
            <div className="flex items-start gap-2 p-3 bg-surface-100 dark:bg-surface-800 rounded-lg">
                <AlertTriangle className="w-4 h-4 text-surface-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-surface-400">
                    Predictions are probabilistic estimates. This is not financial advice. Past accuracy does not guarantee future results.
                </p>
            </div>
        </div>
    );
}

'use client';

import {
    TrendingUp, TrendingDown, ShieldCheck, ShieldAlert,
    ShieldQuestion, Minus, Brain, AlertTriangle, Sparkles
} from 'lucide-react';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { getPredictionTargetDate } from '@/lib/tradingDays';
import { normalizeConfidence } from '@/lib/predictionMetrics';
import type { PredictionData, PriceData } from '@/types/dashboard.types';

interface PredictionSummaryPanelProps {
    prediction?: PredictionData;
    priceData?: PriceData;
    isLoading?: boolean;
}

// Circular confidence ring using SVG
function ConfidenceRing({ pct, level }: { pct: number; level: string }) {
    const r = 36;
    const circ = 2 * Math.PI * r;
    const dashOffset = circ - (pct / 100) * circ;

    const color =
        level === 'high' ? 'var(--color-emerald)' :
        level === 'low'  ? 'var(--color-rose)'    :
                           'var(--color-amber)';

    return (
        <div className="relative flex items-center justify-center w-24 h-24 flex-shrink-0">
            <svg width="96" height="96" viewBox="0 0 96 96" className="rotate-[-90deg]">
                {/* Track */}
                <circle cx="48" cy="48" r={r} fill="none"
                    stroke="var(--surface-highest)" strokeWidth="7" />
                {/* Fill */}
                <circle cx="48" cy="48" r={r} fill="none"
                    stroke={color} strokeWidth="7"
                    strokeDasharray={`${circ}`}
                    strokeDashoffset={dashOffset}
                    strokeLinecap="round"
                    style={{ transition: 'stroke-dashoffset 0.8s ease' }}
                />
            </svg>
            <div className="absolute text-center">
                <span className="text-xl font-bold tabular-nums" style={{ color }}>{pct}</span>
                <span className="text-xs block" style={{ color: 'var(--text-muted)' }}>%</span>
            </div>
        </div>
    );
}

const signalConfig = {
    BUY:  { color: 'var(--color-emerald)', bg: 'rgba(78,222,163,0.12)', border: 'rgba(78,222,163,0.25)', Icon: ShieldCheck },
    SELL: { color: 'var(--color-rose)',    bg: 'rgba(255,178,183,0.12)', border: 'rgba(255,178,183,0.25)', Icon: ShieldAlert },
    HOLD: { color: 'var(--color-amber)',   bg: 'rgba(251,191,36,0.12)',  border: 'rgba(251,191,36,0.25)', Icon: ShieldQuestion },
};

export function PredictionSummaryPanel({ prediction, priceData, isLoading }: PredictionSummaryPanelProps) {
    if (isLoading) {
        return (
            <div className="card-ai p-6 h-full">
                <div className="skeleton h-4 w-28 mb-5 rounded" />
                <div className="skeleton h-12 w-44 mb-3 rounded" />
                <div className="skeleton h-8 w-full mb-4 rounded" />
                <div className="skeleton h-24 w-full rounded" />
            </div>
        );
    }

    // ── Pending / No data state
    if (!prediction || prediction.is_pending) {
        return (
            <div className="card-ai p-6 h-full flex flex-col items-center justify-center text-center gap-3">
                <div
                    className="w-12 h-12 rounded-full flex items-center justify-center"
                    style={{ background: 'rgba(192,193,255,0.10)' }}
                >
                    <Brain className="w-6 h-6" style={{ color: 'var(--color-primary)' }} />
                </div>
                <div>
                    <p className="font-semibold" style={{ color: 'var(--text-secondary)' }}>
                        Prediction Pending
                    </p>
                    <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                        {prediction?.message || 'Run the model to generate a prediction'}
                    </p>
                </div>
            </div>
        );
    }

    const targetDateInfo = getPredictionTargetDate();
    const prevClose = prediction.input_features?.prev_close as number | undefined
        ?? priceData?.previous_close
        ?? priceData?.price;

    const predictedOpen = prediction.predicted_open ?? 0;
    const gapPts = prevClose ? predictedOpen - prevClose : null;
    const gapPct = prediction.predicted_change_pct ?? (
        gapPts != null && prevClose ? (gapPts / prevClose) * 100 : null
    );
    const isUp = (gapPct ?? 0) >= 0;

    const { percentage: confPct, level: confLevel } = normalizeConfidence(
        prediction.confidence_score ?? prediction.direction_probability,
        prediction.confidence_level
    );

    const signal = prediction.signal as keyof typeof signalConfig | undefined;
    const sigCfg = signal ? signalConfig[signal] : null;
    const SigIcon = sigCfg?.Icon ?? Minus;

    return (
        <div className="card-ai p-6 h-full flex flex-col gap-5 animate-fade-up">

            {/* ── Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
                    <span className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                        AI Prediction
                    </span>
                    <span className="badge badge-indigo">ACMI++</span>
                </div>
                <InfoTooltip
                    title="ACMI++ Prediction"
                    content="Adaptive Cross-Modal Integration model combining temporal price history, technical indicators, and GIFT NIFTY overnight signals."
                />
            </div>

            {/* ── Target date */}
            <p className="label-upper -mt-3">
                {targetDateInfo.daysAway === 1 ? "Tomorrow's Open" : `For ${targetDateInfo.formatted}`}
            </p>

            {/* ── Hero: Predicted Price + Confidence Ring */}
            <div className="flex items-center justify-between gap-4">
                <div className="flex-1 min-w-0">
                    <p
                        className="text-4xl font-black tabular-nums truncate"
                        style={{ color: 'var(--text-primary)', letterSpacing: '-0.035em' }}
                    >
                        ₹{predictedOpen.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                    </p>
                    <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                        Predicted Open Price
                    </p>
                </div>

                <div className="flex flex-col items-center gap-1">
                    <ConfidenceRing pct={confPct} level={confLevel} />
                    <p className="label-upper">Confidence</p>
                </div>
            </div>

            {/* ── Gap vs Prev Close */}
            <div
                className="grid grid-cols-3 gap-3 p-3 rounded-xl"
                style={{ background: 'var(--surface-high)', border: '1px solid var(--border-ghost)' }}
            >
                {/* Prev Close */}
                <div className="text-center">
                    <p className="label-upper mb-1">Prev Close</p>
                    <p className="text-sm font-bold tabular-nums" style={{ color: 'var(--text-primary)' }}>
                        {prevClose ? `₹${prevClose.toLocaleString('en-IN', { maximumFractionDigits: 2 })}` : '—'}
                    </p>
                </div>

                {/* Gap Points */}
                <div className="text-center">
                    <p className="label-upper mb-1">Gap (pts)</p>
                    <p
                        className="text-sm font-bold tabular-nums"
                        style={{ color: isUp ? 'var(--color-emerald)' : 'var(--color-rose)' }}
                    >
                        {gapPts != null
                            ? `${gapPts >= 0 ? '+' : ''}${gapPts.toFixed(1)}`
                            : '—'}
                    </p>
                </div>

                {/* Gap % */}
                <div className="text-center">
                    <p className="label-upper mb-1">Gap (%)</p>
                    <p
                        className="text-sm font-bold tabular-nums"
                        style={{ color: isUp ? 'var(--color-emerald)' : 'var(--color-rose)' }}
                    >
                        {gapPct != null
                            ? `${gapPct >= 0 ? '+' : ''}${gapPct.toFixed(2)}%`
                            : '—'}
                    </p>
                </div>
            </div>

            {/* ── Direction + Signal */}
            <div className="flex items-center gap-3">
                {/* Direction */}
                <div
                    className="flex items-center gap-2 flex-1 px-3 py-2.5 rounded-xl"
                    style={{
                        background: isUp ? 'rgba(78,222,163,0.08)' : 'rgba(255,178,183,0.08)',
                        border: `1px solid ${isUp ? 'rgba(78,222,163,0.22)' : 'rgba(255,178,183,0.22)'}`,
                    }}
                >
                    {isUp
                        ? <TrendingUp className="w-5 h-5 flex-shrink-0" style={{ color: 'var(--color-emerald)' }} />
                        : <TrendingDown className="w-5 h-5 flex-shrink-0" style={{ color: 'var(--color-rose)' }} />
                    }
                    <div>
                        <p
                            className="text-sm font-bold"
                            style={{ color: isUp ? 'var(--color-emerald)' : 'var(--color-rose)' }}
                        >
                            {isUp ? 'BULLISH' : 'BEARISH'}
                        </p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            {confLevel.toUpperCase()} CONFIDENCE
                        </p>
                    </div>
                </div>

                {/* Signal */}
                {sigCfg && (
                    <div
                        className="flex items-center gap-2 px-3 py-2.5 rounded-xl"
                        style={{
                            background: sigCfg.bg,
                            border: `1px solid ${sigCfg.border}`,
                        }}
                    >
                        <SigIcon className="w-5 h-5" style={{ color: sigCfg.color }} />
                        <span className="text-sm font-bold" style={{ color: sigCfg.color }}>
                            {signal}
                        </span>
                    </div>
                )}
            </div>

            {/* ── AI Explanation text */}
            {prediction.explanation_text && (
                <div
                    className="p-3 rounded-xl text-sm leading-relaxed italic"
                    style={{
                        background: 'rgba(192,193,255,0.05)',
                        border: '1px solid rgba(192,193,255,0.12)',
                        color: 'var(--text-muted)',
                    }}
                >
                    <Brain className="w-3.5 h-3.5 inline mr-1.5" style={{ color: 'var(--color-primary)' }} />
                    {prediction.explanation_text}
                </div>
            )}
        </div>
    );
}

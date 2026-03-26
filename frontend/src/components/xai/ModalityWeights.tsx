'use client';

import { TrendingUp, Zap, BarChart3 } from 'lucide-react';

interface ModalityWeightsProps {
    weights?: {
        price?: number;
        sentiment?: number;
        technical?: number;
        // Some backends use these alternate keys
        temporal?: number;
        overnight?: number;
    };
}

// ACMI++ modality definitions — Temporal / Technical / Overnight(GIFT)
const modalities = [
    {
        key: 'technical',
        label: 'Technical Indicators',
        subLabel: 'RSI · MACD · Bollinger',
        Icon: BarChart3,
        gradient: 'linear-gradient(90deg, rgba(192,193,255,0.5), var(--color-primary))',
        textColor: 'var(--color-primary)',
    },
    {
        key: 'temporal',  // maps to 'price' as fallback
        altKey: 'price',
        label: 'Temporal (Price History)',
        subLabel: 'OHLCV sequences · 60-day window',
        Icon: TrendingUp,
        gradient: 'linear-gradient(90deg, rgba(78,222,163,0.5), var(--color-emerald))',
        textColor: 'var(--color-emerald)',
    },
    {
        key: 'overnight',  // maps to 'sentiment' as fallback
        altKey: 'sentiment',
        label: 'Overnight (GIFT + News)',
        subLabel: 'GIFT NIFTY · News sentiment',
        Icon: Zap,
        gradient: 'linear-gradient(90deg, rgba(251,191,36,0.5), var(--color-amber))',
        textColor: 'var(--color-amber)',
    },
];

export function ModalityWeights({ weights }: ModalityWeightsProps) {
    const fallback = { technical: 0.35, temporal: 0.45, overnight: 0.20 };
    const data = weights ?? fallback;

    const resolve = (m: typeof modalities[0]) => {
        // Try primary key, then altKey, then fallback
        const w = (data as any)[m.key]
            ?? (m.altKey ? (data as any)[m.altKey] : undefined)
            ?? (fallback as any)[m.key]
            ?? (m.altKey ? (fallback as any)[m.altKey] : 0);
        return Math.min(1, Math.max(0, w));
    };

    // Normalise so they sum to 1 even if backend returns unnormalised values
    const rawVals = modalities.map(resolve);
    const total = rawVals.reduce((s, v) => s + v, 0) || 1;
    const normVals = rawVals.map(v => v / total);

    return (
        <div className="space-y-4">
            {modalities.map((m, i) => {
                const pct = Math.round(normVals[i] * 100);

                return (
                    <div key={m.key} className="space-y-1.5">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <m.Icon className="w-3.5 h-3.5" style={{ color: m.textColor }} />
                                <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                                    {m.label}
                                </span>
                            </div>
                            <span className="text-sm font-bold tabular-nums" style={{ color: m.textColor }}>
                                {pct}%
                            </span>
                        </div>

                        {/* Progress bar */}
                        <div
                            className="h-1.5 rounded-full overflow-hidden"
                            style={{ background: 'var(--surface-highest)' }}
                        >
                            <div
                                className="h-full rounded-full"
                                style={{
                                    width: `${pct}%`,
                                    background: m.gradient,
                                    transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)',
                                }}
                            />
                        </div>

                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            {m.subLabel}
                        </p>
                    </div>
                );
            })}
        </div>
    );
}

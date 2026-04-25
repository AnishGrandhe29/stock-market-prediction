'use client';

import { InfoTooltip } from '@/components/ui/InfoTooltip';

interface ConfidenceIntervalChartProps {
    currentPrice?: number;
    predictedPrice?: number;
    quantile5?: number;
    quantile95?: number;
    changePct?: number;
}

export function ConfidenceIntervalChart({
    currentPrice,
    predictedPrice,
    quantile5,
    quantile95,
    changePct,
}: ConfidenceIntervalChartProps) {
    const price = currentPrice || 0;
    const predicted = predictedPrice || price;
    const q5  = quantile5  || predicted * 0.98;
    const q95 = quantile95 || predicted * 1.02;

    if (!price || price <= 0) {
        return (
            <div className="card p-5">
                <h3 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
                    Confidence Interval
                </h3>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    Waiting for price data...
                </p>
            </div>
        );
    }

    const spread   = q95 - q5;
    const padding  = spread * 0.3;
    const vizMin   = q5 - padding;
    const vizMax   = q95 + padding;
    const vizRange = vizMax - vizMin;

    const q5Pct        = ((q5 - vizMin) / vizRange) * 100;
    const q95Pct       = ((q95 - vizMin) / vizRange) * 100;
    const predictedPct = ((predicted - vizMin) / vizRange) * 100;
    const currentPct   = ((price - vizMin) / vizRange) * 100;

    const isUp = (changePct ?? 0) >= 0;

    return (
        <div className="card p-5 animate-fade-up animate-fade-up-2">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                    Confidence Interval
                </h3>
                <InfoTooltip
                    title="90% Confidence Interval"
                    content="The model is 90% confident the actual NIFTY 50 opening price will land in this range. Wider bands = more uncertainty."
                />
            </div>

            {/* Legend */}
            <div className="flex items-center gap-4 mb-4 text-xs" style={{ color: 'var(--text-muted)' }}>
                <span className="flex items-center gap-1">
                    <span
                        className="w-2.5 h-2.5 rounded-full inline-block border-2"
                        style={{ background: 'var(--surface-bright)', borderColor: 'var(--outline-color)' }}
                    />
                    Current
                </span>
                <span className="flex items-center gap-1">
                    <span
                        className="w-2.5 h-2.5 rounded-full inline-block"
                        style={{ background: isUp ? 'var(--color-emerald)' : 'var(--color-rose)' }}
                    />
                    Predicted
                </span>
                <span className="flex items-center gap-1">
                    <span
                        className="w-5 h-2.5 rounded inline-block"
                        style={{ background: 'rgba(192,193,255,0.20)' }}
                    />
                    90% Band
                </span>
            </div>

            {/* Visual bar */}
            <div className="relative h-14 mb-2">
                {/* Track */}
                <div
                    className="absolute top-5 left-0 right-0 h-4 rounded-full bg-surface-200 dark:bg-surface-800"
                />

                {/* Confidence band */}
                <div
                    className="absolute top-3 h-8 rounded-lg"
                    style={{
                        left: `${q5Pct}%`,
                        width: `${q95Pct - q5Pct}%`,
                        background: 'rgba(99, 102, 241, 0.25)',
                        border: '1px solid rgba(99, 102, 241, 0.4)',
                    }}
                />

                {/* Current price marker */}
                <div
                    className="absolute top-2 flex flex-col items-center"
                    style={{ left: `${currentPct}%`, transform: 'translateX(-50%)' }}
                >
                    <div
                        className="w-px h-10"
                        style={{ background: 'var(--outline-color)' }}
                    />
                    <div
                        className="w-3 h-3 rounded-full border-2 -mt-7"
                        style={{
                            background: 'var(--surface-bright)',
                            borderColor: 'var(--outline-color)',
                        }}
                    />
                </div>

                {/* Predicted price marker */}
                <div
                    className="absolute top-2 flex flex-col items-center"
                    style={{ left: `${predictedPct}%`, transform: 'translateX(-50%)' }}
                >
                    <div
                        className="w-px h-10"
                        style={{ background: isUp ? 'var(--color-emerald)' : 'var(--color-rose)' }}
                    />
                    <div
                        className="w-3.5 h-3.5 rounded-full -mt-8 border-2"
                        style={{
                            background: isUp ? 'var(--color-emerald)' : 'var(--color-rose)',
                            borderColor: 'var(--surface-card)',
                            boxShadow: isUp
                                ? '0 0 8px rgba(78,222,163,0.5)'
                                : '0 0 8px rgba(255,178,183,0.5)',
                        }}
                    />
                </div>
            </div>

            {/* Price labels */}
            <div className="relative h-5 text-xs">
                <span
                    className="absolute font-medium tabular-nums"
                    style={{ left: `${q5Pct}%`, transform: 'translateX(-50%)', color: 'var(--text-muted)' }}
                >
                    ₹{Math.round(q5).toLocaleString('en-IN')}
                </span>
                <span
                    className="absolute font-bold tabular-nums"
                    style={{
                        left: `${predictedPct}%`,
                        transform: 'translateX(-50%)',
                        color: isUp ? 'var(--color-emerald)' : 'var(--color-rose)',
                    }}
                >
                    ₹{Math.round(predicted).toLocaleString('en-IN')}
                </span>
                <span
                    className="absolute font-medium tabular-nums"
                    style={{ left: `${q95Pct}%`, transform: 'translateX(-50%)', color: 'var(--text-muted)' }}
                >
                    ₹{Math.round(q95).toLocaleString('en-IN')}
                </span>
            </div>

            {/* Stats row */}
            <div
                className="grid grid-cols-3 gap-2 mt-4 pt-3"
                style={{ borderTop: '1px solid var(--border-ghost)' }}
            >
                {[
                    { label: 'P5',      val: Math.round(q5),        color: 'var(--color-rose)' },
                    { label: 'Median',  val: Math.round(predicted),  color: isUp ? 'var(--color-emerald)' : 'var(--color-rose)' },
                    { label: 'P95',     val: Math.round(q95),        color: 'var(--color-emerald)' },
                ].map(({ label, val, color }) => (
                    <div key={label} className="text-center">
                        <p className="label-upper mb-0.5">{label}</p>
                        <p className="text-sm font-bold tabular-nums" style={{ color }}>
                            ₹{val.toLocaleString('en-IN')}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
}

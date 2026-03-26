'use client';

import { InfoTooltip } from '@/components/ui/InfoTooltip';

interface SentimentGaugeProps {
    newsSentiment?: number;
    redditSentiment?: number;
    combinedSentiment?: number;
}

export function SentimentGauge({
    newsSentiment = 0,
    redditSentiment = 0,
    combinedSentiment,
}: SentimentGaugeProps) {
    const combined = combinedSentiment ?? ((newsSentiment + redditSentiment) / 2);

    // Map -1..1 → 0..180 degrees
    const rotation = ((combined + 1) / 2) * 180;

    const getLabel = (val: number) => {
        if (val > 0.4)  return 'Very Bullish';
        if (val > 0.15) return 'Bullish';
        if (val > -0.15)return 'Neutral';
        if (val > -0.4) return 'Bearish';
        return 'Very Bearish';
    };

    const getLabelColor = (val: number) => {
        if (val > 0.15)  return 'var(--color-emerald)';
        if (val > -0.15) return 'var(--color-amber)';
        return 'var(--color-rose)';
    };

    const formatScore = (val: number) => val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2);

    return (
        <div className="card p-6 animate-fade-up animate-fade-up-3">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                    Market Sentiment
                </h3>
                <InfoTooltip
                    title="Market Sentiment"
                    content="Aggregated sentiment from financial news and Reddit communities. Ranges from -1 (extremely bearish) to +1 (extremely bullish). Updated daily."
                />
            </div>

            {/* Semicircle gauge */}
            <div className="flex flex-col items-center">
                <div className="relative w-44 h-[88px] overflow-hidden">
                    {/* Gradient arc background */}
                    <div
                        className="absolute inset-0 rounded-t-full"
                        style={{
                            background:
                                'conic-gradient(from 180deg at 50% 100%, var(--color-rose) 0deg, var(--color-amber) 72deg, var(--color-amber) 108deg, var(--color-emerald) 180deg)',
                        }}
                    />
                    {/* Inner cutout */}
                    <div
                        className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[130px] h-[65px] rounded-t-full"
                        style={{ background: 'var(--surface-card)' }}
                    />
                    {/* Needle */}
                    <div
                        className="absolute bottom-0 left-1/2 origin-bottom transition-transform duration-700 ease-out"
                        style={{ transform: `translateX(-50%) rotate(${rotation - 90}deg)` }}
                    >
                        <div
                            className="w-0.5 h-[75px] rounded-full mx-auto"
                            style={{ background: 'var(--text-primary)' }}
                        />
                        <div
                            className="w-2.5 h-2.5 rounded-full -mt-1 mx-auto"
                            style={{ background: 'var(--text-primary)', boxShadow: '0 0 6px rgba(255,255,255,0.3)' }}
                        />
                    </div>
                </div>

                {/* Axis labels */}
                <div className="flex justify-between w-44 mt-1">
                    <span className="text-xs" style={{ color: 'var(--color-rose)' }}>Bearish</span>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Neutral</span>
                    <span className="text-xs" style={{ color: 'var(--color-emerald)' }}>Bullish</span>
                </div>

                {/* Label */}
                <div className="mt-3 text-center">
                    <span className="text-lg font-bold" style={{ color: getLabelColor(combined) }}>
                        {getLabel(combined)}
                    </span>
                    <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                        Score: <span style={{ color: getLabelColor(combined), fontWeight: 700 }}>
                            {formatScore(combined)}
                        </span>
                    </p>
                </div>
            </div>

            {/* Source breakdown */}
            <div
                className="grid grid-cols-2 gap-3 mt-4 pt-4"
                style={{ borderTop: '1px solid var(--border-ghost)' }}
            >
                {[
                    { label: '📰 News', val: newsSentiment },
                    { label: '💬 Reddit', val: redditSentiment },
                ].map(({ label, val }) => (
                    <div
                        key={label}
                        className="text-center p-3 rounded-xl"
                        style={{ background: 'var(--surface-high)', border: '1px solid var(--border-ghost)' }}
                    >
                        <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>{label}</p>
                        <p
                            className="text-sm font-bold"
                            style={{ color: getLabelColor(val) }}
                        >
                            {formatScore(val)}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
}

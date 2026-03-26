'use client';

import { InfoTooltip } from '@/components/ui/InfoTooltip';

interface Feature {
    feature: string;
    importance: number;
    direction: 'positive' | 'negative';
    modality: string;
}

interface TopFeaturesProps {
    features?: Feature[];
}

// Modality accent colors (Stitch design tokens)
const modalityAccent: Record<string, string> = {
    Technical:  'var(--color-primary)',
    Sentiment:  'var(--color-amber)',
    Price:      'var(--color-emerald)',
};

const modalityBadge: Record<string, { bg: string; color: string }> = {
    Technical: { bg: 'rgba(192,193,255,0.12)', color: 'var(--color-primary)' },
    Sentiment: { bg: 'rgba(251,191,36,0.12)',  color: 'var(--color-amber)' },
    Price:     { bg: 'rgba(78,222,163,0.12)',  color: 'var(--color-emerald)' },
};

export function TopFeatures({ features }: TopFeaturesProps) {
    const defaultFeatures: Feature[] = [
        { feature: 'RSI (14)',         importance: 0.042, direction: 'positive', modality: 'Technical' },
        { feature: 'MACD Signal',      importance: 0.038, direction: 'positive', modality: 'Technical' },
        { feature: 'GIFT NIFTY Spread',importance: 0.031, direction: 'positive', modality: 'Technical' },
        { feature: 'Volume Ratio',     importance: 0.024, direction: 'positive', modality: 'Price' },
        { feature: 'VIX Level',        importance: 0.019, direction: 'negative', modality: 'Technical' },
        { feature: 'Close / SMA20',    importance: 0.018, direction: 'positive', modality: 'Technical' },
        { feature: 'BB Width',         importance: 0.015, direction: 'positive', modality: 'Technical' },
        { feature: 'News Sentiment',   importance: 0.009, direction: 'positive', modality: 'Sentiment' },
    ];

    const data = features || defaultFeatures;
    const maxImportance = Math.max(...data.map(f => f.importance));

    return (
        <div className="space-y-3">
            {data.slice(0, 8).map((feature, index) => {
                const pct = (feature.importance / maxImportance) * 100;
                const accent = modalityAccent[feature.modality] || 'var(--color-primary)';
                const badge = modalityBadge[feature.modality] || modalityBadge.Technical;
                const isPos = feature.direction === 'positive';

                return (
                    <div
                        key={feature.feature}
                        className="flex items-center gap-3 animate-fade-up"
                        style={{ animationDelay: `${index * 40}ms` }}
                    >
                        {/* Rank */}
                        <span
                            className="w-5 text-xs font-bold text-center flex-shrink-0"
                            style={{ color: 'var(--text-disabled)' }}
                        >
                            {index + 1}
                        </span>

                        {/* Feature name */}
                        <div className="w-36 flex-shrink-0">
                            <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                                {feature.feature}
                            </span>
                        </div>

                        {/* SHAP bar */}
                        <div className="flex-1 relative">
                            <div
                                className="h-2 rounded-full overflow-hidden"
                                style={{ background: 'var(--surface-highest)' }}
                            >
                                <div
                                    className="h-full rounded-full transition-all duration-700 ease-out"
                                    style={{
                                        width: `${pct}%`,
                                        background: isPos
                                            ? `linear-gradient(90deg, rgba(78,222,163,0.35), var(--color-emerald))`
                                            : `linear-gradient(90deg, rgba(255,178,183,0.35), var(--color-rose))`,
                                    }}
                                />
                            </div>
                        </div>

                        {/* SHAP value */}
                        <span
                            className="w-14 text-right text-xs font-bold flex-shrink-0 tabular-nums"
                            style={{ color: isPos ? 'var(--color-emerald)' : 'var(--color-rose)' }}
                        >
                            {isPos ? '+' : '-'}{feature.importance.toFixed(3)}
                        </span>

                        {/* Direction arrow */}
                        <span
                            className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                            style={{
                                background: isPos ? 'rgba(78,222,163,0.15)' : 'rgba(255,178,183,0.15)',
                                color: isPos ? 'var(--color-emerald)' : 'var(--color-rose)',
                            }}
                        >
                            {isPos ? '↑' : '↓'}
                        </span>

                        {/* Modality badge */}
                        <span
                            className="px-2 py-0.5 rounded-full text-xs font-semibold flex-shrink-0"
                            style={{ background: badge.bg, color: badge.color }}
                        >
                            {feature.modality}
                        </span>

                        <InfoTooltip
                            title={feature.feature}
                            content={getFeatureDescription(feature.feature)}
                        />
                    </div>
                );
            })}
        </div>
    );
}

function getFeatureDescription(feature: string): string {
    const desc: Record<string, string> = {
        'RSI (14)':          'Relative Strength Index — measures momentum. Above 70 = overbought, below 30 = oversold.',
        'MACD Signal':       'Moving Average Convergence Divergence signal line — shows trend direction strength.',
        'GIFT NIFTY Spread': 'Spread between GIFT Nifty futures and NIFTY 50 spot — a leading pre-market indicator.',
        'Volume Ratio':      'Current volume vs 20-day average — spikes indicate institutional activity.',
        'VIX Level':         'India VIX volatility index — higher = more uncertainty / market fear.',
        'Close / SMA20':     'Current close relative to 20-day simple moving average — position in trend.',
        'BB Width':          'Bollinger Band width — narrow bands often precede large moves.',
        'News Sentiment':    'Aggregated sentiment from financial news targeting NIFTY 50 and Indian markets.',
    };
    return desc[feature] || 'Feature importance in the ACMI++ prediction model.';
}

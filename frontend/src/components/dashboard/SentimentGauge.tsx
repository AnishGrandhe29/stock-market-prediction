'use client';

import { InfoTooltip } from '@/components/ui/InfoTooltip';

interface SentimentGaugeProps {
    newsSentiment?: number;   // -1 to 1
    redditSentiment?: number; // -1 to 1
    combinedSentiment?: number; // -1 to 1
}

export function SentimentGauge({
    newsSentiment = 0,
    redditSentiment = 0,
    combinedSentiment,
}: SentimentGaugeProps) {
    const combined = combinedSentiment ?? ((newsSentiment + redditSentiment) / 2);

    // Map -1..1 to 0..180 degrees rotation
    const rotation = ((combined + 1) / 2) * 180;

    // Derive label
    const getLabel = (val: number) => {
        if (val > 0.4) return 'Very Bullish';
        if (val > 0.15) return 'Bullish';
        if (val > -0.15) return 'Neutral';
        if (val > -0.4) return 'Bearish';
        return 'Very Bearish';
    };

    const getColor = (val: number) => {
        if (val > 0.15) return 'text-emerald-600 dark:text-emerald-400';
        if (val > -0.15) return 'text-amber-600 dark:text-amber-400';
        return 'text-rose-600 dark:text-rose-400';
    };

    const formatScore = (val: number) => {
        if (val > 0) return `+${val.toFixed(2)}`;
        return val.toFixed(2);
    };

    return (
        <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-surface-900 dark:text-white">
                    Market Sentiment
                </h3>
                <InfoTooltip
                    title="Market Sentiment"
                    content="Aggregated sentiment from financial news and Reddit (r/IndiaInvestments, r/IndianStreetBets). Ranges from -1 (extremely bearish) to +1 (extremely bullish). Updated daily."
                />
            </div>

            {/* Gauge */}
            <div className="flex flex-col items-center">
                {/* Semicircle gauge */}
                <div className="relative w-48 h-24 overflow-hidden">
                    {/* Background arc */}
                    <div
                        className="absolute inset-0 rounded-t-full"
                        style={{
                            background: 'conic-gradient(from 180deg at 50% 100%, #ef4444 0deg, #f59e0b 72deg, #eab308 90deg, #84cc16 108deg, #10b981 180deg)',
                        }}
                    />
                    {/* Inner cutout */}
                    <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-36 h-[72px] bg-white dark:bg-surface-800 rounded-t-full" />

                    {/* Needle */}
                    <div
                        className="absolute bottom-0 left-1/2 origin-bottom transition-transform duration-700 ease-out"
                        style={{ transform: `translateX(-50%) rotate(${rotation - 90}deg)` }}
                    >
                        <div className="w-1 h-20 bg-surface-800 dark:bg-white rounded-full mx-auto" />
                        <div className="w-3 h-3 bg-surface-800 dark:bg-white rounded-full -mt-1 mx-auto" />
                    </div>
                </div>

                {/* Labels under gauge */}
                <div className="flex justify-between w-48 mt-1 text-xs text-surface-500">
                    <span>Bearish</span>
                    <span>Neutral</span>
                    <span>Bullish</span>
                </div>

                {/* Main label */}
                <div className="mt-3 text-center">
                    <span className={`text-xl font-bold ${getColor(combined)}`}>
                        {getLabel(combined)}
                    </span>
                    <p className="text-sm text-surface-500 mt-1">
                        Score: {formatScore(combined)}
                    </p>
                </div>
            </div>

            {/* Source breakdown */}
            <div className="grid grid-cols-2 gap-3 mt-5 pt-4 border-t border-surface-200 dark:border-surface-700">
                <div className="text-center p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                    <p className="text-xs text-surface-500 mb-1">📰 News</p>
                    <p className={`text-sm font-bold ${getColor(newsSentiment)}`}>
                        {formatScore(newsSentiment)}
                    </p>
                </div>
                <div className="text-center p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                    <p className="text-xs text-surface-500 mb-1">💬 Reddit</p>
                    <p className={`text-sm font-bold ${getColor(redditSentiment)}`}>
                        {formatScore(redditSentiment)}
                    </p>
                </div>
            </div>
        </div>
    );
}

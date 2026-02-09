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

// Color schemes for different modalities
const modalityColors: Record<string, { bg: string; text: string; bar: string }> = {
    Technical: {
        bg: 'bg-blue-100 dark:bg-blue-900/40',
        text: 'text-blue-700 dark:text-blue-300',
        bar: 'bg-gradient-to-r from-blue-400 to-blue-600 dark:from-blue-500 dark:to-blue-400'
    },
    Sentiment: {
        bg: 'bg-purple-100 dark:bg-purple-900/40',
        text: 'text-purple-700 dark:text-purple-300',
        bar: 'bg-gradient-to-r from-purple-400 to-purple-600 dark:from-purple-500 dark:to-purple-400'
    },
    Price: {
        bg: 'bg-amber-100 dark:bg-amber-900/40',
        text: 'text-amber-700 dark:text-amber-300',
        bar: 'bg-gradient-to-r from-amber-400 to-amber-600 dark:from-amber-500 dark:to-amber-400'
    }
};

export function TopFeatures({ features }: TopFeaturesProps) {
    const defaultFeatures: Feature[] = [
        { feature: 'RSI_14', importance: 0.18, direction: 'positive', modality: 'Technical' },
        { feature: 'News Sentiment', importance: 0.15, direction: 'positive', modality: 'Sentiment' },
        { feature: 'EMA_20 Trend', importance: 0.12, direction: 'positive', modality: 'Technical' },
        { feature: 'MACD Histogram', importance: 0.10, direction: 'negative', modality: 'Technical' },
        { feature: 'Price Momentum', importance: 0.09, direction: 'positive', modality: 'Price' },
        { feature: 'Volume Surge', importance: 0.08, direction: 'positive', modality: 'Price' },
        { feature: 'ATR Volatility', importance: 0.07, direction: 'negative', modality: 'Technical' },
        { feature: 'Reddit Sentiment', importance: 0.06, direction: 'positive', modality: 'Sentiment' },
    ];

    const data = features || defaultFeatures;

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {data.slice(0, 8).map((feature, index) => {
                const colors = modalityColors[feature.modality] || modalityColors.Technical;
                // Convert importance to percentage (0.18 = 18%)
                const percentage = feature.importance * 100;
                // Bar width equals the actual percentage displayed (18% importance = 18% bar width)
                const barWidth = percentage;

                return (
                    <div
                        key={feature.feature}
                        className="p-4 bg-white dark:bg-surface-800 rounded-xl border border-surface-200 dark:border-surface-700 shadow-sm hover:shadow-md transition-shadow animate-slide-up"
                        style={{ animationDelay: `${index * 50}ms` }}
                    >
                        {/* Feature Name */}
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-sm font-semibold text-gray-900 dark:text-white">
                                {feature.feature}
                            </span>
                            <InfoTooltip
                                title={feature.feature}
                                content={getFeatureDescription(feature.feature)}
                            />
                        </div>

                        {/* Progress Bar - width matches displayed percentage */}
                        <div className="flex items-center gap-3 mb-3">
                            <div className="flex-1">
                                <div className="h-2.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all duration-700 ease-out ${colors.bar}`}
                                        style={{ width: `${barWidth}%` }}
                                    />
                                </div>
                            </div>

                            {/* Direction indicator */}
                            <span
                                className={`text-sm font-bold w-6 h-6 flex items-center justify-center rounded-full ${feature.direction === 'positive'
                                        ? 'bg-emerald-100 text-emerald-600 dark:bg-emerald-900/50 dark:text-emerald-400'
                                        : 'bg-rose-100 text-rose-600 dark:bg-rose-900/50 dark:text-rose-400'
                                    }`}
                            >
                                {feature.direction === 'positive' ? '↑' : '↓'}
                            </span>
                        </div>

                        {/* Modality tag and importance */}
                        <div className="flex items-center justify-between">
                            <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${colors.bg} ${colors.text}`}>
                                {feature.modality}
                            </span>
                            <span className="text-sm font-bold text-gray-700 dark:text-gray-300">
                                {percentage.toFixed(0)}%
                            </span>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

function getFeatureDescription(feature: string): string {
    const descriptions: Record<string, string> = {
        RSI_14: 'Relative Strength Index (14-period) measures momentum. Above 70 = overbought, below 30 = oversold.',
        'News Sentiment': 'Aggregated sentiment from financial news articles about NIFTY 50 and Indian markets.',
        'EMA_20 Trend': '20-day Exponential Moving Average trend direction indicates short-term momentum.',
        'MACD Histogram': 'Moving Average Convergence Divergence histogram shows trend strength and potential reversals.',
        'Price Momentum': 'Rate of price change over recent trading sessions.',
        'Volume Surge': 'Unusual increase in trading volume compared to average.',
        'ATR Volatility': 'Average True Range measures market volatility over 14 periods.',
        'Reddit Sentiment': 'Sentiment from r/IndiaInvestments and r/IndianStreetBets communities.',
    };

    return descriptions[feature] || 'Feature importance in the prediction model.';
}

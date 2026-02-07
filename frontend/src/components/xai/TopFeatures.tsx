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

export function TopFeatures({ features }: TopFeaturesProps) {
    const defaultFeatures: Feature[] = [
        { feature: 'RSI_14', importance: 0.18, direction: 'positive', modality: 'technical' },
        { feature: 'News Sentiment', importance: 0.15, direction: 'positive', modality: 'sentiment' },
        { feature: 'EMA_20 Trend', importance: 0.12, direction: 'positive', modality: 'technical' },
        { feature: 'MACD Histogram', importance: 0.10, direction: 'negative', modality: 'technical' },
        { feature: 'Price Momentum', importance: 0.09, direction: 'positive', modality: 'price' },
        { feature: 'Volume Surge', importance: 0.08, direction: 'positive', modality: 'price' },
        { feature: 'ATR Volatility', importance: 0.07, direction: 'negative', modality: 'technical' },
        { feature: 'Reddit Sentiment', importance: 0.06, direction: 'positive', modality: 'sentiment' },
    ];

    const data = features || defaultFeatures;

    return (
        <div className= "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4" >
        {
            data.slice(0, 8).map((feature, index) => (
                <div
          key= { feature.feature }
          className = "p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl animate-slide-up"
          style = {{ animationDelay: `${index * 50}ms` }}
        >
        <div className="flex items-center justify-between mb-2" >
            <span className="text-sm font-medium text-surface-700 dark:text-surface-300" >
                { feature.feature }
                </span>
                < InfoTooltip
    title = { feature.feature }
    content = { getFeatureDescription(feature.feature) }
        />
        </div>

        < div className = "flex items-center gap-3" >
            {/* Impact bar */ }
            < div className = "flex-1" >
                <div className="h-3 bg-surface-200 dark:bg-surface-600 rounded-full overflow-hidden" >
                    <div
                  className={
        `h-full rounded-full transition-all duration-500 ${feature.direction === 'positive'
            ? 'bg-gradient-to-r from-success-400 to-success-500'
            : 'bg-gradient-to-r from-danger-400 to-danger-500'
        }`
    }
    style = {{ width: `${feature.importance * 100 * 5}%` }
}
                />
    </div>
    </div>

{/* Direction indicator */ }
<span
              className={
    `text-xs font-bold px-2 py-0.5 rounded ${feature.direction === 'positive'
        ? 'bg-success-100 text-success-700 dark:bg-success-900/50 dark:text-success-300'
        : 'bg-danger-100 text-danger-700 dark:bg-danger-900/50 dark:text-danger-300'
    }`
}
            >
    { feature.direction === 'positive' ? '↑' : '↓' }
    </span>
    </div>

{/* Modality tag */ }
<div className="mt-2" >
    <span className="text-xs text-surface-500 capitalize" >
        { feature.modality }
        </span>
        </div>
        </div>
      ))}
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

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
    // Use realistic defaults if data not available
    const price = currentPrice || 0;
    const predicted = predictedPrice || price;
    const q5 = quantile5 || predicted * 0.98;
    const q95 = quantile95 || predicted * 1.02;

    if (!price || price <= 0) {
        return (
            <div className="card p-6">
                <h3 className="font-semibold text-surface-900 dark:text-white mb-2">Confidence Interval</h3>
                <p className="text-sm text-surface-500">Waiting for price data...</p>
            </div>
        );
    }

    // Compute range for the visualization
    const spread = q95 - q5;
    const padding = spread * 0.3;
    const vizMin = q5 - padding;
    const vizMax = q95 + padding;
    const vizRange = vizMax - vizMin;

    // Positions as percentages
    const q5Pct = ((q5 - vizMin) / vizRange) * 100;
    const q95Pct = ((q95 - vizMin) / vizRange) * 100;
    const predictedPct = ((predicted - vizMin) / vizRange) * 100;
    const currentPct = ((price - vizMin) / vizRange) * 100;

    const isUp = (changePct ?? 0) >= 0;

    return (
        <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-surface-900 dark:text-white">
                    Prediction Confidence Interval
                </h3>
                <InfoTooltip
                    title="90% Confidence Interval"
                    content="The shaded region shows the range within which the actual price is expected to fall with 90% probability. Narrower bands indicate higher model confidence."
                />
            </div>

            {/* Legend */}
            <div className="flex items-center gap-4 mb-4 text-xs text-surface-500">
                <span className="flex items-center gap-1">
                    <span className="w-3 h-3 rounded-full bg-surface-400 border-2 border-surface-600 inline-block" />
                    Current
                </span>
                <span className="flex items-center gap-1">
                    <span className={`w-3 h-3 rounded-full inline-block ${isUp ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                    Predicted
                </span>
                <span className="flex items-center gap-1">
                    <span className="w-6 h-3 rounded bg-primary-200 dark:bg-primary-800 inline-block" />
                    90% Band
                </span>
            </div>

            {/* Visual bar */}
            <div className="relative h-16 mb-2">
                {/* Background track */}
                <div className="absolute top-6 left-0 right-0 h-4 bg-surface-100 dark:bg-surface-700 rounded-full" />

                {/* Confidence band */}
                <div
                    className="absolute top-4 h-8 bg-gradient-to-r from-primary-200 via-primary-300 to-primary-200 dark:from-primary-800 dark:via-primary-700 dark:to-primary-800 rounded-lg border border-primary-300 dark:border-primary-600 opacity-70"
                    style={{ left: `${q5Pct}%`, width: `${q95Pct - q5Pct}%` }}
                />

                {/* Current price marker */}
                <div
                    className="absolute top-3 flex flex-col items-center"
                    style={{ left: `${currentPct}%`, transform: 'translateX(-50%)' }}
                >
                    <div className="w-0.5 h-10 bg-surface-400 dark:bg-surface-500" />
                    <div className="w-3 h-3 rounded-full bg-surface-400 border-2 border-surface-600 dark:border-surface-300 -mt-7" />
                </div>

                {/* Predicted price marker */}
                <div
                    className="absolute top-3 flex flex-col items-center"
                    style={{ left: `${predictedPct}%`, transform: 'translateX(-50%)' }}
                >
                    <div className={`w-0.5 h-10 ${isUp ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                    <div className={`w-4 h-4 rounded-full -mt-8 border-2 border-white dark:border-surface-800 shadow-lg ${isUp ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                </div>
            </div>

            {/* Labels */}
            <div className="relative h-8 text-xs">
                <span
                    className="absolute text-surface-500 font-medium"
                    style={{ left: `${q5Pct}%`, transform: 'translateX(-50%)' }}
                >
                    ₹{Math.round(q5).toLocaleString('en-IN')}
                </span>
                <span
                    className={`absolute font-bold ${isUp ? 'text-emerald-600' : 'text-rose-600'}`}
                    style={{ left: `${predictedPct}%`, transform: 'translateX(-50%)' }}
                >
                    ₹{Math.round(predicted).toLocaleString('en-IN')}
                </span>
                <span
                    className="absolute text-surface-500 font-medium"
                    style={{ left: `${q95Pct}%`, transform: 'translateX(-50%)' }}
                >
                    ₹{Math.round(q95).toLocaleString('en-IN')}
                </span>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-surface-200 dark:border-surface-700">
                <div className="text-center">
                    <p className="text-xs text-surface-500">Lower Bound</p>
                    <p className="text-sm font-semibold text-rose-600">₹{Math.round(q5).toLocaleString('en-IN')}</p>
                </div>
                <div className="text-center">
                    <p className="text-xs text-surface-500">Predicted</p>
                    <p className={`text-sm font-bold ${isUp ? 'text-emerald-600' : 'text-rose-600'}`}>
                        ₹{Math.round(predicted).toLocaleString('en-IN')}
                    </p>
                </div>
                <div className="text-center">
                    <p className="text-xs text-surface-500">Upper Bound</p>
                    <p className="text-sm font-semibold text-emerald-600">₹{Math.round(q95).toLocaleString('en-IN')}</p>
                </div>
            </div>
        </div>
    );
}

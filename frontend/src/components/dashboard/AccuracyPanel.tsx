'use client';

import { CheckCircle, XCircle, Minus, TrendingUp, BarChart3, AlertTriangle } from 'lucide-react';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import type { PredictionHistoryRow, PriceHistoryRow, AccuracyMetrics } from '@/types/dashboard.types';

interface AccuracyPanelProps {
    predictions?: PredictionHistoryRow[];
    priceHistory?: PriceHistoryRow[];
    metrics?: AccuracyMetrics;
    isLoading?: boolean;
}

// Build merged rows: match each prediction's target_date to actual open in price history
function buildAccuracyRows(
    predictions: PredictionHistoryRow[],
    priceHistory: PriceHistoryRow[]
) {
    const priceMap = new Map(priceHistory.map(p => [p.date, p]));

    return predictions
        .filter(p => p.target_date)
        .map(pred => {
            const actual = priceMap.get(pred.target_date);
            const actualOpen = actual?.open;
            const errorPts = actualOpen != null ? actualOpen - pred.predicted_open : null;
            const errorPct = actualOpen != null && pred.predicted_open > 0
                ? ((actualOpen - pred.predicted_open) / pred.predicted_open) * 100
                : null;

            // Direction correct: predicted up and actual was up, or predicted down and actual down
            let directionCorrect: boolean | null = null;
            if (actualOpen != null && pred.predicted_open > 0) {
                const actualDir = actualOpen >= (actual?.open ?? pred.predicted_open) ? 'up' : 'down';
                // Compare predicted_change_pct sign with actual price direction
                const predDir = pred.predicted_change_pct >= 0 ? 'up' : 'down';
                directionCorrect = pred.direction_correct ?? (predDir === actualDir);
            }

            return {
                target_date: pred.target_date,
                predicted: pred.predicted_open,
                actual: actualOpen,
                errorPts,
                errorPct,
                directionCorrect,
                confidence: pred.confidence_score,
                confidenceLevel: pred.confidence_level,
            };
        })
        .slice(0, 14); // last 14 days
}

// Compute metrics if not provided by API
function computeMetrics(rows: ReturnType<typeof buildAccuracyRows>): AccuracyMetrics | null {
    const valid = rows.filter(r => r.actual != null && r.errorPts != null);
    if (valid.length === 0) return null;

    const correctDir = valid.filter(r => r.directionCorrect === true).length;
    const mae = valid.reduce((s, r) => s + Math.abs(r.errorPts!), 0) / valid.length;
    const mape = valid.reduce((s, r) => s + Math.abs(r.errorPct!), 0) / valid.length;

    return {
        direction_accuracy: correctDir / valid.length,
        mae: Math.round(mae * 100) / 100,
        mape: Math.round(mape * 100) / 100,
    };
}

export function AccuracyPanel({ predictions, priceHistory, metrics: propMetrics, isLoading }: AccuracyPanelProps) {

    if (isLoading) {
        return (
            <div className="card p-6">
                <div className="skeleton h-5 w-40 mb-5 rounded" />
                <div className="grid grid-cols-3 gap-3 mb-5">
                    {[0, 1, 2].map(i => <div key={i} className="skeleton h-16 rounded-xl" />)}
                </div>
                <div className="space-y-2">
                    {[0, 1, 2, 3, 4].map(i => <div key={i} className="skeleton h-10 rounded" />)}
                </div>
            </div>
        );
    }

    const rows = (predictions && priceHistory)
        ? buildAccuracyRows(predictions, priceHistory)
        : [];
    const hasData = rows.length > 0;
    const metrics = propMetrics ?? (hasData ? computeMetrics(rows) : null);

    const dirAccPct = metrics ? Math.round(metrics.direction_accuracy * 100) : null;

    return (
        <div className="card p-6 animate-fade-up animate-fade-up-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" style={{ color: 'var(--color-primary)' }} />
                    <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
                        Prediction Accuracy
                    </h2>
                    <span className="badge badge-muted">Last 14 days</span>
                </div>
                <InfoTooltip
                    title="Prediction Accuracy"
                    content="Shows how ACMI++ performed on recent predictions. Direction accuracy = did the model correctly predict up vs. down? MAE = average absolute error in points."
                />
            </div>

            {/* Metrics Row */}
            {metrics ? (
                <div className="grid grid-cols-3 gap-3 mb-5">
                    {/* Direction Accuracy */}
                    <div
                        className="p-3 rounded-xl text-center"
                        style={{ background: 'var(--surface-high)', border: '1px solid var(--border-ghost)' }}
                    >
                        <p className="label-upper mb-2">Direction Accuracy</p>
                        {/* Mini arc bar */}
                        <div className="flex justify-center mb-2">
                            <svg width="60" height="32" viewBox="0 0 60 32">
                                <path d="M 5 30 A 25 25 0 0 1 55 30" fill="none"
                                    stroke="var(--surface-highest)" strokeWidth="6" strokeLinecap="round" />
                                <path d="M 5 30 A 25 25 0 0 1 55 30" fill="none"
                                    stroke={dirAccPct! >= 65 ? 'var(--color-emerald)' : dirAccPct! >= 50 ? 'var(--color-amber)' : 'var(--color-rose)'}
                                    strokeWidth="6" strokeLinecap="round"
                                    strokeDasharray="78.5"
                                    strokeDashoffset={78.5 - (78.5 * (dirAccPct! / 100))}
                                    style={{ transition: 'stroke-dashoffset 1s ease' }}
                                />
                            </svg>
                        </div>
                        <p
                            className="text-2xl font-black tabular-nums"
                            style={{
                                color: dirAccPct! >= 65
                                    ? 'var(--color-emerald)'
                                    : dirAccPct! >= 50
                                        ? 'var(--color-amber)'
                                        : 'var(--color-rose)',
                            }}
                        >
                            {dirAccPct}%
                        </p>
                    </div>

                    {/* MAE */}
                    <div
                        className="p-3 rounded-xl text-center"
                        style={{ background: 'var(--surface-high)', border: '1px solid var(--border-ghost)' }}
                    >
                        <p className="label-upper mb-2">Avg Error (MAE)</p>
                        <div className="flex items-end justify-center gap-1 mb-1">
                            <TrendingUp className="w-4 h-4 mb-1" style={{ color: 'var(--color-primary)' }} />
                        </div>
                        <p className="text-2xl font-black tabular-nums" style={{ color: 'var(--text-primary)' }}>
                            {metrics.mae.toFixed(0)}
                        </p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>pts avg error</p>
                    </div>

                    {/* MAPE */}
                    <div
                        className="p-3 rounded-xl text-center"
                        style={{ background: 'var(--surface-high)', border: '1px solid var(--border-ghost)' }}
                    >
                        <p className="label-upper mb-2">MAPE</p>
                        <p className="text-2xl font-black tabular-nums" style={{ color: 'var(--text-primary)' }}>
                            {metrics.mape.toFixed(2)}
                        </p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>% mean abs error</p>
                    </div>
                </div>
            ) : (
                <div
                    className="flex items-center gap-2 p-3 rounded-xl mb-5"
                    style={{ background: 'var(--surface-high)', border: '1px solid var(--border-ghost)' }}
                >
                    <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--text-muted)' }} />
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                        Accuracy metrics available once predictions have verified actuals.
                    </p>
                </div>
            )}

            {/* Table */}
            {hasData ? (
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-ghost)' }}>
                                {['Date', 'Predicted', 'Actual', 'Error (pts)', 'Error (%)', 'Direction'].map(h => (
                                    <th key={h} className="label-upper py-2 px-3 text-left font-semibold">
                                        {h}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row, i) => {
                                const isCorrect = row.directionCorrect;
                                const hasActual = row.actual != null;

                                return (
                                    <tr
                                        key={row.target_date}
                                        className="transition-colors"
                                        style={{
                                            borderBottom: '1px solid var(--border-faint)',
                                            background: hasActual
                                                ? isCorrect
                                                    ? 'rgba(78,222,163,0.03)'
                                                    : 'rgba(255,178,183,0.03)'
                                                : 'transparent',
                                            borderLeft: hasActual
                                                ? `3px solid ${isCorrect ? 'rgba(78,222,163,0.35)' : 'rgba(255,178,183,0.35)'}`
                                                : '3px solid transparent',
                                        }}
                                    >
                                        {/* Date */}
                                        <td className="py-2.5 px-3 tabular-nums" style={{ color: 'var(--text-muted)' }}>
                                            {new Date(row.target_date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })}
                                        </td>
                                        {/* Predicted */}
                                        <td className="py-2.5 px-3 font-semibold tabular-nums" style={{ color: 'var(--text-primary)' }}>
                                            ₹{row.predicted.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                                        </td>
                                        {/* Actual */}
                                        <td className="py-2.5 px-3 tabular-nums" style={{ color: hasActual ? 'var(--text-primary)' : 'var(--text-disabled)' }}>
                                            {hasActual
                                                ? `₹${row.actual!.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
                                                : <span className="italic">Pending</span>}
                                        </td>
                                        {/* Error pts */}
                                        <td
                                            className="py-2.5 px-3 tabular-nums font-medium"
                                            style={{
                                                color: row.errorPts == null
                                                    ? 'var(--text-disabled)'
                                                    : Math.abs(row.errorPts) < 50
                                                        ? 'var(--color-emerald)'
                                                        : Math.abs(row.errorPts) < 150
                                                            ? 'var(--color-amber)'
                                                            : 'var(--color-rose)',
                                            }}
                                        >
                                            {row.errorPts != null
                                                ? `${row.errorPts >= 0 ? '+' : ''}${row.errorPts.toFixed(0)}`
                                                : '—'}
                                        </td>
                                        {/* Error % */}
                                        <td
                                            className="py-2.5 px-3 tabular-nums"
                                            style={{ color: row.errorPct == null ? 'var(--text-disabled)' : 'var(--text-secondary)' }}
                                        >
                                            {row.errorPct != null
                                                ? `${row.errorPct >= 0 ? '+' : ''}${row.errorPct.toFixed(2)}%`
                                                : '—'}
                                        </td>
                                        {/* Direction */}
                                        <td className="py-2.5 px-3">
                                            {isCorrect === true && (
                                                <CheckCircle className="w-4 h-4" style={{ color: 'var(--color-emerald)' }} />
                                            )}
                                            {isCorrect === false && (
                                                <XCircle className="w-4 h-4" style={{ color: 'var(--color-rose)' }} />
                                            )}
                                            {isCorrect === null && (
                                                <Minus className="w-4 h-4" style={{ color: 'var(--text-disabled)' }} />
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            ) : (
                <div
                    className="flex flex-col items-center justify-center py-8 text-center rounded-xl"
                    style={{ background: 'var(--surface-high)', border: '1px solid var(--border-ghost)' }}
                >
                    <BarChart3 className="w-8 h-8 mb-2" style={{ color: 'var(--text-disabled)' }} />
                    <p className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>
                        No history yet
                    </p>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-disabled)' }}>
                        Predictions will appear here as the model runs daily
                    </p>
                </div>
            )}
        </div>
    );
}

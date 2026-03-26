'use client';

import { Zap, TrendingUp, TrendingDown, Clock, AlertTriangle } from 'lucide-react';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import type { PredictionData } from '@/types/dashboard.types';

interface GiftNiftyPanelProps {
    prediction?: PredictionData;
    currentNiftyClose?: number;
}

export function GiftNiftyPanel({ prediction, currentNiftyClose }: GiftNiftyPanelProps) {
    // Extract GIFT NIFTY data from prediction's input_features
    const giftClose = prediction?.input_features?.gift_nifty_close as number | undefined;
    const prevClose = prediction?.input_features?.prev_close as number | undefined
        ?? currentNiftyClose;

    const hasData = typeof giftClose === 'number' && giftClose > 0;
    const gapPts = hasData && prevClose ? giftClose - prevClose : null;
    const gapPct = gapPts && prevClose ? (gapPts / prevClose) * 100 : null;
    const isPositive = (gapPts ?? 0) >= 0;

    return (
        <div
            className="rounded-2xl p-4 animate-fade-up"
            style={{
                background: 'linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.03) 100%)',
                border: '1px solid rgba(245,158,11,0.22)',
                boxShadow: '0 0 24px rgba(245,158,11,0.06)',
            }}
        >
            <div className="flex items-center justify-between">
                {/* Left: Label + values */}
                <div className="flex items-center gap-4">
                    {/* Icon */}
                    <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                        style={{ background: 'rgba(245,158,11,0.15)' }}
                    >
                        <Zap className="w-5 h-5" style={{ color: 'var(--color-amber)' }} />
                    </div>

                    {/* Label */}
                    <div>
                        <div className="flex items-center gap-2">
                            <span
                                className="text-xs font-bold uppercase tracking-widest"
                                style={{ color: 'var(--color-amber)' }}
                            >
                                GIFT NIFTY
                            </span>
                            <span
                                className="text-xs px-2 py-0.5 rounded-full font-semibold"
                                style={{
                                    background: 'rgba(245,158,11,0.15)',
                                    color: 'var(--color-amber)',
                                    border: '1px solid rgba(245,158,11,0.25)',
                                }}
                            >
                                Overnight Signal
                            </span>
                            <InfoTooltip
                                title="GIFT NIFTY"
                                content="GIFT NIFTY (formerly SGX Nifty) is a futures contract traded in Gujarat International Finance Tec-City that mirrors NIFTY 50. It trades before NSE opens, making it the primary leading indicator for the Indian market's opening direction."
                            />
                        </div>
                        <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                            Pre-market indicator · ACMI++ overnight modality
                        </p>
                    </div>
                </div>

                {/* Right: Values */}
                {hasData ? (
                    <div className="flex items-center gap-6">
                        {/* GIFT NIFTY value */}
                        <div className="text-right">
                            <p className="label-upper mb-0.5">GIFT NIFTY Close</p>
                            <p
                                className="text-2xl font-bold tabular-nums"
                                style={{ color: 'var(--color-amber)', letterSpacing: '-0.02em' }}
                            >
                                {giftClose?.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                            </p>
                        </div>

                        {/* Gap vs NSE prev close */}
                        <div
                            className="px-4 py-2 rounded-xl text-center"
                            style={{
                                background: isPositive
                                    ? 'rgba(78,222,163,0.10)'
                                    : 'rgba(255,178,183,0.10)',
                                border: `1px solid ${isPositive
                                    ? 'rgba(78,222,163,0.22)'
                                    : 'rgba(255,178,183,0.22)'}`,
                            }}
                        >
                            <p className="label-upper mb-0.5">Gap vs NSE</p>
                            <div className="flex items-center gap-1.5">
                                {isPositive
                                    ? <TrendingUp className="w-4 h-4" style={{ color: 'var(--color-emerald)' }} />
                                    : <TrendingDown className="w-4 h-4" style={{ color: 'var(--color-rose)' }} />
                                }
                                <span
                                    className="text-lg font-bold tabular-nums"
                                    style={{
                                        color: isPositive ? 'var(--color-emerald)' : 'var(--color-rose)',
                                    }}
                                >
                                    {gapPts !== null && (gapPts >= 0 ? '+' : '')}{gapPts?.toFixed(1)} pts
                                </span>
                            </div>
                            {gapPct !== null && (
                                <p
                                    className="text-xs font-semibold mt-0.5"
                                    style={{
                                        color: isPositive ? 'var(--color-emerald)' : 'var(--color-rose)',
                                    }}
                                >
                                    ({gapPct >= 0 ? '+' : ''}{gapPct.toFixed(2)}%)
                                </p>
                            )}
                        </div>

                        {/* NSE prev close reference */}
                        {prevClose && (
                            <div className="text-right">
                                <p className="label-upper mb-0.5">NSE Prev Close</p>
                                <p
                                    className="text-xl font-semibold tabular-nums"
                                    style={{ color: 'var(--text-primary)' }}
                                >
                                    {prevClose?.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                                </p>
                            </div>
                        )}
                    </div>
                ) : (
                    /* No data state */
                    <div
                        className="flex items-center gap-2 px-4 py-2 rounded-xl"
                        style={{
                            background: 'rgba(145,143,154,0.08)',
                            border: '1px solid rgba(145,143,154,0.15)',
                        }}
                    >
                        <AlertTriangle className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                        <div>
                            <p className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>
                                Awaiting GIFT NIFTY data
                            </p>
                            <p className="text-xs" style={{ color: 'var(--text-disabled)' }}>
                                Available in prediction payload after market close
                            </p>
                        </div>
                    </div>
                )}

                {/* Timestamp */}
                <div
                    className="flex items-center gap-1 text-xs flex-shrink-0"
                    style={{ color: 'var(--text-disabled)' }}
                >
                    <Clock className="w-3 h-3" />
                    <span>Used in ACMI++ prediction</span>
                </div>
            </div>
        </div>
    );
}

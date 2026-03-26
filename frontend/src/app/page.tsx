'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    Activity, BarChart3, Brain, Clock,
    TrendingUp, TrendingDown, BarChart2,
} from 'lucide-react';

import { stocksAPI, predictionsAPI } from '@/lib/api';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

// Layout
import { GiftNiftyPanel }         from '@/components/dashboard/GiftNiftyPanel';
import { PredictionSummaryPanel } from '@/components/dashboard/PredictionSummaryPanel';
import { AccuracyPanel }          from '@/components/dashboard/AccuracyPanel';

// Charts
import { SimplePriceChart }         from '@/components/charts/SimplePriceChart';
import { ConfidenceIntervalChart }  from '@/components/dashboard/ConfidenceIntervalChart';
import { SentimentGauge }           from '@/components/dashboard/SentimentGauge';

// XAI
import { ModalityWeights } from '@/components/xai/ModalityWeights';
import { TopFeatures }     from '@/components/xai/TopFeatures';

import type { PredictionData, PriceData, PriceHistoryRow, PredictionHistoryRow } from '@/types/dashboard.types';

// ─── Reusable mini stat card ──────────────────────────────────────────────────
interface StatCardProps {
    label: string;
    children: React.ReactNode;
    tooltip?: string;
    tooltipContent?: string;
    accent?: 'emerald' | 'rose' | 'indigo' | 'amber';
    delay?: number;
}

function StatCard({ label, children, tooltip, tooltipContent, accent, delay = 0 }: StatCardProps) {
    const accentBorder: Record<string, string> = {
        emerald: '3px solid var(--color-emerald)',
        rose:    '3px solid var(--color-rose)',
        indigo:  '3px solid var(--color-primary)',
        amber:   '3px solid var(--color-amber)',
    };
    return (
        <div
            className="card p-4 card-hover animate-fade-up"
            style={{
                borderLeft: accent ? accentBorder[accent] : undefined,
                animationDelay: `${delay}ms`,
            }}
        >
            <div className="flex items-center justify-between mb-2.5">
                <span className="label-upper">{label}</span>
                {tooltip && tooltipContent && (
                    <InfoTooltip title={tooltip} content={tooltipContent} />
                )}
            </div>
            {children}
        </div>
    );
}

// ─── Section heading ──────────────────────────────────────────────────────────
function SectionHeading({ icon: Icon, label }: { icon: React.ElementType; label: string }) {
    return (
        <div className="flex items-center gap-2 mb-4">
            <Icon className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
            <h2 className="text-sm font-bold uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>
                {label}
            </h2>
            <div className="flex-1 h-px" style={{ background: 'var(--border-ghost)' }} />
        </div>
    );
}

// ─── Dashboard Page ───────────────────────────────────────────────────────────
export default function Dashboard() {
    const [lastUpdated, setLastUpdated] = useState('');

    useEffect(() => {
        const upd = () => setLastUpdated(
            new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })
        );
        upd();
        const t = setInterval(upd, 60_000);
        return () => clearInterval(t);
    }, []);

    // ── Data fetching
    const { data: priceRes, isLoading: priceLoading } = useQuery({
        queryKey: ['realtime-price'],
        queryFn: () => stocksAPI.getRealtime('^NSEI'),
        refetchInterval: 60_000,
    });

    const { data: predRes, isLoading: predLoading } = useQuery({
        queryKey: ['latest-prediction'],
        queryFn: () => predictionsAPI.getLatest('^NSEI'),
    });

    const { data: predHistRes } = useQuery({
        queryKey: ['pred-history'],
        queryFn: () => predictionsAPI.getHistory('^NSEI', 14),
    });

    const { data: priceHistRes } = useQuery({
        queryKey: ['price-history-acc'],
        queryFn: () => stocksAPI.getHistory('^NSEI', 20),
    });

    const { data: sentimentRes } = useQuery({
        queryKey: ['sentiment'],
        queryFn: () => stocksAPI.getSentiment('^NSEI', 1),
    });

    const { data: marketRes } = useQuery({
        queryKey: ['market-status'],
        queryFn: () => stocksAPI.getMarketStatus(),
    });

    // ── Derive values
    const price  = priceRes?.data    as PriceData | undefined;
    const pred   = predRes?.data     as PredictionData | undefined;
    const predHistory = predHistRes?.data as PredictionHistoryRow[] | undefined;
    const priceHistory = priceHistRes?.data as PriceHistoryRow[] | undefined;
    const sentiment = sentimentRes?.data?.[0];
    const isMarketOpen = marketRes?.data?.is_open as boolean | undefined;
    const isPositive = (price?.change_pct ?? 0) >= 0;

    // Day range position %
    const rangePct = (price?.high && price?.low && price?.price)
        ? ((price.price - price.low) / (price.high - price.low)) * 100
        : 50;

    return (
        <div className="space-y-6">

            {/* ══ Page Header */}
            <div className="flex items-center justify-between animate-fade-up">
                <div>
                    <h1 className="text-2xl font-black tracking-tight" style={{ color: 'var(--text-primary)', letterSpacing: '-0.03em' }}>
                        NIFTY 50 Dashboard
                    </h1>
                    <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>
                        ACMI++ · Multimodal AI Prediction System
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    {isMarketOpen !== undefined && (
                        <div
                            className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold"
                            style={{
                                background: isMarketOpen ? 'rgba(78,222,163,0.10)' : 'rgba(70,69,84,0.12)',
                                border: `1px solid ${isMarketOpen ? 'rgba(78,222,163,0.25)' : 'rgba(70,69,84,0.2)'}`,
                                color: isMarketOpen ? 'var(--color-emerald)' : 'var(--text-muted)',
                            }}
                        >
                            {isMarketOpen
                                ? <><span className="pulse-green" /> Market Open</>
                                : <><span className="w-2 h-2 rounded-full inline-block" style={{ background: 'var(--outline-color)' }} /> Market Closed</>
                            }
                        </div>
                    )}
                    <div
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs"
                        style={{
                            background: 'var(--surface-high)',
                            border: '1px solid var(--border-ghost)',
                            color: 'var(--text-muted)',
                        }}
                    >
                        <Clock className="w-3 h-3" />
                        <span suppressHydrationWarning>Updated {lastUpdated || '—'}</span>
                    </div>
                </div>
            </div>

            {/* ══ Row 1: Stat Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {/* NIFTY Price */}
                <StatCard
                    label="NIFTY 50"
                    tooltip="NIFTY 50 Price"
                    tooltipContent="Live/delayed NIFTY 50 index price from NSE."
                    accent={isPositive ? 'emerald' : 'rose'}
                    delay={0}
                >
                    {priceLoading ? <div className="skeleton h-9 w-32 rounded" /> : (
                        <>
                            <p className="metric-value">
                                ₹{price?.price?.toLocaleString('en-IN', { maximumFractionDigits: 2 }) || '—'}
                            </p>
                            <div
                                className="flex items-center gap-1 mt-1 text-sm font-semibold"
                                style={{ color: isPositive ? 'var(--color-emerald)' : 'var(--color-rose)' }}
                            >
                                {isPositive ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                                {isPositive ? '+' : ''}{price?.change?.toFixed(1)} ({price?.change_pct?.toFixed(2)}%)
                            </div>
                        </>
                    )}
                </StatCard>

                {/* Day Range */}
                <StatCard label="Day Range" delay={50}>
                    <div className="space-y-1.5">
                        <div className="flex justify-between text-xs">
                            <span style={{ color: 'var(--color-rose)' }}>
                                L: ₹{price?.low?.toLocaleString('en-IN') || '—'}
                            </span>
                            <span style={{ color: 'var(--color-emerald)' }}>
                                H: ₹{price?.high?.toLocaleString('en-IN') || '—'}
                            </span>
                        </div>
                        <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--surface-highest)' }}>
                            <div
                                className="h-full rounded-full"
                                style={{
                                    width: `${rangePct}%`,
                                    background: 'linear-gradient(90deg, var(--color-rose), var(--color-primary), var(--color-emerald))',
                                }}
                            />
                        </div>
                        <p className="text-xs text-center font-semibold tabular-nums" style={{ color: 'var(--text-primary)' }}>
                            Current: ₹{price?.price?.toLocaleString('en-IN') || '—'}
                        </p>
                    </div>
                </StatCard>

                {/* Volume */}
                <StatCard
                    label="Volume"
                    tooltip="Trading Volume"
                    tooltipContent="Total shares traded today. Available only during market hours."
                    delay={100}
                >
                    <div className="flex items-center gap-2">
                        <BarChart2 className="w-5 h-5" style={{ color: 'var(--color-primary)' }} />
                        <span className="metric-value" style={{ fontSize: '1.5rem' }}>
                            {price?.volume && price.volume > 0
                                ? price.volume >= 1e9 ? (price.volume / 1e9).toFixed(1) + 'B'
                                : price.volume >= 1e6 ? (price.volume / 1e6).toFixed(1) + 'M'
                                    : price.volume.toLocaleString('en-IN')
                                : 'N/A'}
                        </span>
                    </div>
                    {(!price?.volume || price.volume === 0) && (
                        <p className="text-xs mt-1" style={{ color: 'var(--text-disabled)' }}>
                            Available during market hours
                        </p>
                    )}
                </StatCard>

                {/* Prev Close */}
                <StatCard
                    label="Prev Close"
                    tooltip="Previous Close"
                    tooltipContent="NIFTY 50 closing price from the previous trading session."
                    delay={150}
                >
                    <p className="metric-value" style={{ fontSize: '1.5rem' }}>
                        ₹{(price?.previous_close ?? pred?.input_features?.prev_close as number | undefined)
                            ?.toLocaleString('en-IN', { maximumFractionDigits: 2 }) || '—'}
                    </p>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>NSE close</p>
                </StatCard>
            </div>

            {/* ══ Row 2: GIFT NIFTY Banner */}
            <GiftNiftyPanel
                prediction={pred}
                currentNiftyClose={price?.previous_close ?? price?.price}
            />

            {/* ══ Row 3: Price Chart (2/3) + Prediction Hero (1/3) */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Chart — full featured */}
                <div className="lg:col-span-2 card p-5 animate-fade-up animate-fade-up-1">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
                            <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                                NIFTY 50 — 60 Day Chart
                            </h2>
                            <span className="badge badge-indigo">+ AI Overlay</span>
                        </div>
                        <InfoTooltip
                            title="Price Chart"
                            content="60-day NIFTY 50 candlestick chart. The dashed indigo line shows the ACMI++ model's predicted opening price for the next trading day."
                        />
                    </div>
                    <SimplePriceChart prediction={pred} />

                    {/* Chart legend */}
                    <div className="flex items-center gap-5 mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                        <span className="flex items-center gap-1.5">
                            <span className="w-3 h-3 rounded-sm inline-block" style={{ background: 'var(--color-emerald)' }} />
                            Bullish candle
                        </span>
                        <span className="flex items-center gap-1.5">
                            <span className="w-3 h-3 rounded-sm inline-block" style={{ background: 'var(--color-rose)' }} />
                            Bearish candle
                        </span>
                        <span className="flex items-center gap-1.5">
                            <span className="w-5 h-0.5 inline-block" style={{ background: 'var(--color-primary)', borderTop: '2px dashed var(--color-primary)' }} />
                            AI Prediction
                        </span>
                    </div>
                </div>

                {/* Prediction Summary Hero */}
                <PredictionSummaryPanel
                    prediction={pred}
                    priceData={price}
                    isLoading={predLoading}
                />
            </div>

            {/* ══ Row 4: Confidence Band | Model Focus | Sentiment */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

                {/* Confidence Interval */}
                <ConfidenceIntervalChart
                    currentPrice={price?.price}
                    predictedPrice={pred?.predicted_open}
                    quantile5={pred?.quantile_5}
                    quantile95={pred?.quantile_95}
                    changePct={pred?.predicted_change_pct}
                />

                {/* ACMI++ Model Focus */}
                <div className="card p-5 animate-fade-up animate-fade-up-2">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <Brain className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
                            <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                                ACMI++ Model Focus
                            </h3>
                        </div>
                        <InfoTooltip
                            title="Modality Weights"
                            content="How much the model relied on each data stream for this prediction. Temporal = price history sequences, Technical = RSI/MACD/etc, Overnight = GIFT NIFTY & news."
                        />
                    </div>
                    <ModalityWeights weights={pred?.modality_weights} />
                </div>

                {/* Sentiment Gauge */}
                <SentimentGauge
                    newsSentiment={sentiment?.news_sentiment}
                    redditSentiment={sentiment?.reddit_sentiment}
                    combinedSentiment={sentiment?.combined_sentiment}
                />
            </div>

            {/* ══ Row 5: Accuracy History (full width) */}
            <div>
                <SectionHeading icon={BarChart3} label="Prediction Accuracy History" />
                <AccuracyPanel
                    predictions={predHistory}
                    priceHistory={priceHistory}
                    isLoading={false}
                />
            </div>

            {/* ══ Row 6: SHAP Feature Importance (full width) */}
            <div>
                <SectionHeading icon={Brain} label="AI Explanation — SHAP Feature Importance" />
                <div className="card-ai p-6 animate-fade-up animate-fade-up-4">
                    <div className="flex items-center justify-between mb-5">
                        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                            Features that pushed today's prediction <span style={{ color: 'var(--color-emerald)' }}>up ↑</span> or{' '}
                            <span style={{ color: 'var(--color-rose)' }}>down ↓</span>.
                            Values = mean absolute SHAP impact on predicted return.
                        </p>
                        <InfoTooltip
                            title="SHAP Values"
                            content="SHapley Additive exPlanations (SHAP) show how each feature contributed to the model's prediction. Larger bars = greater influence on this specific prediction."
                        />
                    </div>
                    <TopFeatures features={pred?.top_features} />
                </div>
            </div>

        </div>
    );
}

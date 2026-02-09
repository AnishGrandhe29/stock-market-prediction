'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    TrendingUp,
    TrendingDown,
    Activity,
    BarChart3,
    Brain,
    AlertTriangle,
    CheckCircle,
    Clock
} from 'lucide-react';
import { stocksAPI, predictionsAPI } from '@/lib/api';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { PredictionCard } from '@/components/dashboard/PredictionCard';
import { PriceChart } from '@/components/charts/PriceChart';
import { ModalityWeights } from '@/components/xai/ModalityWeights';
import { TopFeatures } from '@/components/xai/TopFeatures';

export default function Dashboard() {
    // Client-side time to avoid hydration mismatch
    const [lastUpdated, setLastUpdated] = useState<string>('');

    useEffect(() => {
        const updateTime = () => setLastUpdated(new Date().toLocaleTimeString());
        updateTime();
        const interval = setInterval(updateTime, 60000);
        return () => clearInterval(interval);
    }, []);

    // Fetch real-time price
    const { data: priceData, isLoading: priceLoading } = useQuery({
        queryKey: ['realtime-price'],
        queryFn: () => stocksAPI.getRealtime('^NSEI'),
        refetchInterval: 60000, // Refresh every minute
    });

    // Fetch latest prediction
    const { data: predictionData, isLoading: predictionLoading } = useQuery({
        queryKey: ['latest-prediction'],
        queryFn: () => predictionsAPI.getLatest('^NSEI'),
    });

    // Fetch market status
    const { data: marketStatus } = useQuery({
        queryKey: ['market-status'],
        queryFn: () => stocksAPI.getMarketStatus(),
    });

    const price = priceData?.data;
    const prediction = predictionData?.data;
    const isPositive = price?.change_pct >= 0;

    return (
        <div className="space-y-6" >
            {/* Header */}
            < div className="flex items-center justify-between" >
                <div>
                    <h1 className="text-3xl font-bold text-surface-900 dark:text-white" >
                        Dashboard
                    </h1>
                    < p className="text-surface-500 mt-1" >
                        NIFTY 50 Index Predictions & Analysis
                    </p>
                </div>
                < div className="flex items-center gap-2" >
                    <Clock className="w-4 h-4 text-surface-500" />
                    <span className="text-sm text-surface-500" suppressHydrationWarning>
                        Last updated: {lastUpdated || '...'}
                    </span>
                </div>
            </div>

            {/* Price Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4" >
                {/* Current Price */}
                < div className="card p-6" >
                    <div className="flex items-center justify-between mb-2" >
                        <span className="text-sm text-surface-500" > NIFTY 50 Index </span>
                        < InfoTooltip
                            title="NIFTY 50"
                            content="The NIFTY 50 is India's benchmark stock market index representing the weighted average of 50 of the largest Indian companies listed on the National Stock Exchange."
                        />
                    </div>
                    {
                        priceLoading ? (
                            <div className="skeleton h-8 w-32" />
                        ) : (
                            <>
                                <div className="text-3xl font-bold text-surface-900 dark:text-white" >
                                    ₹{price?.price?.toLocaleString('en-IN') || '—'}
                                </div>
                                < div className={`flex items-center gap-1 mt-2 ${isPositive ? 'positive' : 'negative'}`
                                }>
                                    {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                                    <span className="font-medium" >
                                        {isPositive ? '+' : ''}{price?.change?.toFixed(2)} ({price?.change_pct?.toFixed(2)} %)
                                    </span>
                                </div>
                            </>
                        )}
                </div>

                {/* Day Range */}
                <div className="card p-6" >
                    <div className="flex items-center justify-between mb-2" >
                        <span className="text-sm text-surface-500" > Day Range </span>
                        < InfoTooltip
                            title="Day Range"
                            content="The high and low prices for NIFTY 50 during today's trading session."
                        />
                    </div>
                    < div className="space-y-2" >
                        <div className="flex items-center justify-between" >
                            <span className="text-surface-500" > Low </span>
                            < span className="font-semibold" >₹{price?.low?.toLocaleString('en-IN') || '—'} </span>
                        </div>
                        < div className="h-2 bg-surface-100 dark:bg-surface-700 rounded-full overflow-hidden" >
                            <div className="h-full bg-gradient-to-r from-danger-500 via-primary-500 to-success-500 rounded-full"
                                style={{ width: `${((price?.price - price?.low) / (price?.high - price?.low)) * 100 || 50}%` }} />
                        </div>
                        < div className="flex items-center justify-between" >
                            <span className="text-surface-500" > High </span>
                            < span className="font-semibold" >₹{price?.high?.toLocaleString('en-IN') || '—'} </span>
                        </div>
                    </div>
                </div>

                {/* Volume */}
                <div className="card p-6" >
                    <div className="flex items-center justify-between mb-2" >
                        <span className="text-sm text-surface-500" > Volume </span>
                        < InfoTooltip
                            title="Trading Volume"
                            content="The total number of shares traded today. Volume data is only available during market hours."
                        />
                    </div>
                    < div className="flex items-center gap-2" >
                        <BarChart3 className="w-6 h-6 text-primary-500" />
                        <span className="text-2xl font-bold text-surface-900 dark:text-white" >
                            {price?.volume && price.volume > 0
                                ? (price.volume >= 1e9
                                    ? (price.volume / 1e9).toFixed(2) + 'B'
                                    : price.volume >= 1e6
                                        ? (price.volume / 1e6).toFixed(2) + 'M'
                                        : price.volume.toLocaleString('en-IN'))
                                : 'N/A'}
                        </span>
                    </div>
                    {(!price?.volume || price.volume === 0) && (
                        <p className="text-xs text-surface-400 mt-2">
                            Volume unavailable (market may be closed)
                        </p>
                    )}
                </div>

                {/* Market Status */}
                <div className="card p-6" >
                    <div className="flex items-center justify-between mb-2" >
                        <span className="text-sm text-surface-500" > Market Status </span>
                        < InfoTooltip
                            title="Market Status"
                            content="NSE trading hours are 9:15 AM to 3:30 PM IST, Monday through Friday."
                        />
                    </div>
                    < div className="flex items-center gap-2" >
                        {
                            marketStatus?.data?.is_open ? (
                                <>
                                    <div className="w-3 h-3 bg-success-500 rounded-full animate-pulse" />
                                    <span className="text-lg font-semibold text-success-600" > Open </span>
                                </>
                            ) : (
                                <>
                                    <div className="w-3 h-3 bg-surface-400 rounded-full" />
                                    <span className="text-lg font-semibold text-surface-500" > Closed </span>
                                </>
                            )}
                    </div>
                </div>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6" >
                {/* Chart - Takes 2 columns */}
                < div className="lg:col-span-2 card p-6" >
                    <div className="flex items-center justify-between mb-4" >
                        <h2 className="text-xl font-semibold text-surface-900 dark:text-white" >
                            Price Chart
                        </h2>
                        < InfoTooltip
                            title="Live Chart"
                            content="Real-time candlestick chart showing NIFTY 50 price movements. Green candles indicate price increase, red indicates decrease."
                        />
                    </div>
                    < PriceChart />
                </div>

                {/* Prediction Card */}
                <div className="space-y-6" >
                    <PredictionCard prediction={prediction} isLoading={predictionLoading} currentPrice={price?.price} />

                    {/* Modality Weights */}
                    < div className="card p-6" >
                        <div className="flex items-center justify-between mb-4" >
                            <h3 className="font-semibold text-surface-900 dark:text-white" >
                                Model Focus
                            </h3>
                            < InfoTooltip
                                title="Modality Weights"
                                content="Shows how much the AI model is relying on each data type for this prediction. Higher weights mean more influence."
                            />
                        </div>
                        < ModalityWeights weights={prediction?.modality_weights} />
                    </div>
                </div>
            </div>

            {/* XAI Section */}
            <div className="card p-6" >
                <div className="flex items-center justify-between mb-6" >
                    <div className="flex items-center gap-3" >
                        <Brain className="w-6 h-6 text-primary-500" />
                        <h2 className="text-xl font-semibold text-surface-900 dark:text-white" >
                            AI Explanation
                        </h2>
                    </div>
                    < InfoTooltip
                        title="Explainable AI"
                        content="This section shows WHY the AI made its prediction. Understanding the reasoning helps you make better informed decisions."
                    />
                </div>
                < TopFeatures features={prediction?.top_features} />
            </div>
        </div>
    );
}

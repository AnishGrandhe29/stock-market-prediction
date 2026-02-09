'use client';

import { PriceChart } from '@/components/charts/PriceChart';
import { useQuery } from '@tanstack/react-query';
import { stocksAPI } from '@/lib/api';
import { TrendingUp, TrendingDown, BarChart3, Calendar, Activity } from 'lucide-react';

export default function ChartPage() {
    const { data: realtimeData, isLoading } = useQuery({
        queryKey: ['realtime-price'],
        queryFn: () => stocksAPI.getRealtime('^NSEI'),
        refetchInterval: 30000,
    });

    const { data: historyData } = useQuery({
        queryKey: ['price-history'],
        queryFn: () => stocksAPI.getHistory('^NSEI', 60),
    });

    const price = realtimeData?.data?.price || 0;
    const change = realtimeData?.data?.change || 0;
    const changePercent = realtimeData?.data?.change_percent || 0;
    const isUp = change >= 0;

    // Calculate stats from history
    const prices = historyData?.data || [];
    const high52Week = prices.length > 0 ? Math.max(...prices.map((p: any) => p.high)) : 0;
    const low52Week = prices.length > 0 ? Math.min(...prices.map((p: any) => p.low)) : 0;
    const avgVolume = prices.length > 0
        ? prices.reduce((sum: number, p: any) => sum + (p.volume || 0), 0) / prices.length
        : 0;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-surface-900 dark:text-white">
                        NIFTY 50 Chart
                    </h1>
                    <p className="text-surface-500 mt-1">
                        Interactive candlestick chart with volume analysis
                    </p>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
                    <BarChart3 className="w-5 h-5 text-primary-500" />
                    <span className="text-sm font-medium text-primary-700 dark:text-primary-300">
                        Technical Analysis
                    </span>
                </div>
            </div>

            {/* Price Summary Card */}
            <div className="card p-6">
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-4">
                        <div className={`p-3 rounded-xl ${isUp
                            ? 'bg-success-100 dark:bg-success-900/30'
                            : 'bg-danger-100 dark:bg-danger-900/30'
                            }`}>
                            {isUp ? (
                                <TrendingUp className="w-8 h-8 text-success-600" />
                            ) : (
                                <TrendingDown className="w-8 h-8 text-danger-600" />
                            )}
                        </div>
                        <div>
                            <p className="text-sm text-surface-500">Current Price</p>
                            <p className="text-3xl font-bold text-surface-900 dark:text-white">
                                ₹{price.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                            </p>
                        </div>
                    </div>
                    <div className="text-right">
                        <p className={`text-2xl font-bold ${isUp ? 'text-success-600' : 'text-danger-600'}`}>
                            {isUp ? '+' : ''}{change.toFixed(2)}
                        </p>
                        <p className={`text-lg ${isUp ? 'text-success-600' : 'text-danger-600'}`}>
                            ({isUp ? '+' : ''}{changePercent.toFixed(2)}%)
                        </p>
                    </div>
                </div>

                {/* Stats Row */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="text-center p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                        <p className="text-xs text-surface-500 mb-1">52 Week High</p>
                        <p className="text-lg font-bold text-success-600">
                            ₹{high52Week.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                        </p>
                    </div>
                    <div className="text-center p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                        <p className="text-xs text-surface-500 mb-1">52 Week Low</p>
                        <p className="text-lg font-bold text-danger-600">
                            ₹{low52Week.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                        </p>
                    </div>
                    <div className="text-center p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                        <p className="text-xs text-surface-500 mb-1">Avg Volume</p>
                        <p className="text-lg font-bold text-surface-700 dark:text-surface-300">
                            {(avgVolume / 1000000).toFixed(1)}M
                        </p>
                    </div>
                </div>

                {/* Chart */}
                <div className="border border-surface-200 dark:border-surface-700 rounded-xl p-4 bg-surface-50 dark:bg-surface-800/50">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <Calendar className="w-4 h-4 text-surface-500" />
                            <span className="text-sm text-surface-600 dark:text-surface-400">
                                Last 60 trading days
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-surface-500" />
                            <span className="text-sm text-surface-600 dark:text-surface-400">
                                Candlestick + Volume
                            </span>
                        </div>
                    </div>
                    <PriceChart />
                </div>
            </div>

            {/* Chart Legend */}
            <div className="card p-4">
                <h3 className="font-semibold text-surface-900 dark:text-white mb-3">Chart Legend</h3>
                <div className="flex flex-wrap gap-6">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-success-500 rounded"></div>
                        <span className="text-sm text-surface-700 dark:text-surface-300">Bullish Candle (Close &gt; Open)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-danger-500 rounded"></div>
                        <span className="text-sm text-surface-700 dark:text-surface-300">Bearish Candle (Close &lt; Open)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-primary-500/50 rounded"></div>
                        <span className="text-sm text-surface-700 dark:text-surface-300">Volume Bars</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-1 bg-amber-500 rounded"></div>
                        <span className="text-sm text-surface-700 dark:text-surface-300">30-Day MA</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

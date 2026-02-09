'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    Bell,
    AlertTriangle,
    TrendingUp,
    TrendingDown,
    Newspaper,
    Globe,
    Clock,
    ExternalLink,
    RefreshCw,
    Filter,
    ChevronDown,
    Star,
    Loader2
} from 'lucide-react';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { newsAPI, usersAPI } from '@/lib/api';
import { useAuthStore } from '@/stores/authStore';

interface NewsItem {
    id: string;
    title: string;
    summary: string;
    source: string;
    url: string;
    publishedAt: string;
    sentiment: 'positive' | 'negative' | 'neutral';
    impact: 'high' | 'medium' | 'low';
    category: string;
    relatedStocks?: string[];
    isWatchlist?: boolean;
}

export default function AlertsPage() {
    const { isAuthenticated } = useAuthStore();
    const [filter, setFilter] = useState<'all' | 'positive' | 'negative' | 'neutral'>('all');
    const [impactFilter, setImpactFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');
    const [showFilters, setShowFilters] = useState(false);
    const [watchlistSymbols, setWatchlistSymbols] = useState<string[]>([]);

    // Fetch watchlist for prioritization
    const { data: watchlistData } = useQuery({
        queryKey: ['watchlist'],
        queryFn: () => usersAPI.getWatchlist(),
        enabled: isAuthenticated,
    });

    // Extract watchlist symbols
    useEffect(() => {
        if (watchlistData?.data) {
            const symbols = watchlistData.data.map((item: any) => item.symbol);
            setWatchlistSymbols(symbols);
        }
    }, [watchlistData]);

    // Fetch market news
    const { data: newsData, isLoading, refetch, isRefetching } = useQuery({
        queryKey: ['market-news', watchlistSymbols],
        queryFn: () => newsAPI.getMarketNews(watchlistSymbols.length > 0 ? watchlistSymbols : undefined),
        refetchInterval: 300000, // Refresh every 5 minutes
        staleTime: 60000, // Consider fresh for 1 minute
    });

    const news: NewsItem[] = newsData?.data?.news || [];

    const filteredNews = news.filter(item => {
        if (filter !== 'all' && item.sentiment !== filter) return false;
        if (impactFilter !== 'all' && item.impact !== impactFilter) return false;
        return true;
    });

    const formatTime = (dateStr: string) => {
        const date = new Date(dateStr);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / (1000 * 60));
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 5) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays === 1) return 'Yesterday';
        return `${diffDays} days ago`;
    };

    const getSentimentColor = (sentiment: string) => {
        switch (sentiment) {
            case 'positive': return 'text-success-500 bg-success-100 dark:bg-success-900/30';
            case 'negative': return 'text-danger-500 bg-danger-100 dark:bg-danger-900/30';
            default: return 'text-surface-500 bg-surface-100 dark:bg-surface-700';
        }
    };

    const getImpactColor = (impact: string) => {
        switch (impact) {
            case 'high': return 'border-danger-500 bg-danger-50 dark:bg-danger-900/20';
            case 'medium': return 'border-warning-500 bg-warning-50 dark:bg-warning-900/20';
            default: return 'border-surface-300 bg-surface-50 dark:bg-surface-800';
        }
    };

    const positiveCount = news.filter(n => n.sentiment === 'positive').length;
    const negativeCount = news.filter(n => n.sentiment === 'negative').length;
    const neutralCount = news.filter(n => n.sentiment === 'neutral').length;
    const watchlistNewsCount = news.filter(n => n.isWatchlist).length;

    return (
        <div className="max-w-5xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Bell className="w-8 h-8 text-primary-500" />
                    <div>
                        <h1 className="text-3xl font-bold text-surface-900 dark:text-white">
                            Market Alerts
                        </h1>
                        <p className="text-surface-500">
                            Real-time news affecting NIFTY 50
                        </p>
                    </div>
                </div>
                <button
                    onClick={() => refetch()}
                    disabled={isLoading || isRefetching}
                    className="flex items-center gap-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 transition-colors"
                >
                    <RefreshCw className={`w-4 h-4 ${isRefetching ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </div>

            {/* Watchlist Alert */}
            {watchlistSymbols.length > 0 && watchlistNewsCount > 0 && (
                <div className="card p-4 border-l-4 border-primary-500 bg-primary-50 dark:bg-primary-900/20">
                    <div className="flex items-center gap-3">
                        <Star className="w-5 h-5 text-primary-500 fill-primary-500" />
                        <div>
                            <p className="font-medium text-primary-700 dark:text-primary-300">
                                {watchlistNewsCount} news items about your watchlist stocks
                            </p>
                            <p className="text-sm text-primary-600 dark:text-primary-400">
                                News related to your watchlist is prioritized at the top
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Sentiment Overview */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="card p-4 border-l-4 border-success-500">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <TrendingUp className="w-6 h-6 text-success-500" />
                            <div>
                                <p className="text-sm text-surface-500">Positive News</p>
                                <p className="text-2xl font-bold text-success-600 dark:text-success-400">
                                    {isLoading ? '—' : positiveCount}
                                </p>
                            </div>
                        </div>
                        <InfoTooltip title="Positive Sentiment" content="News likely to have a positive impact on NIFTY 50" />
                    </div>
                </div>
                <div className="card p-4 border-l-4 border-danger-500">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <TrendingDown className="w-6 h-6 text-danger-500" />
                            <div>
                                <p className="text-sm text-surface-500">Negative News</p>
                                <p className="text-2xl font-bold text-danger-600 dark:text-danger-400">
                                    {isLoading ? '—' : negativeCount}
                                </p>
                            </div>
                        </div>
                        <InfoTooltip title="Negative Sentiment" content="News likely to have a negative impact on NIFTY 50" />
                    </div>
                </div>
                <div className="card p-4 border-l-4 border-surface-400">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Newspaper className="w-6 h-6 text-surface-500" />
                            <div>
                                <p className="text-sm text-surface-500">Neutral News</p>
                                <p className="text-2xl font-bold text-surface-600 dark:text-surface-300">
                                    {isLoading ? '—' : neutralCount}
                                </p>
                            </div>
                        </div>
                        <InfoTooltip title="Neutral Sentiment" content="News with mixed or uncertain market impact" />
                    </div>
                </div>
            </div>

            {/* Filters */}
            <div className="card p-4">
                <button
                    onClick={() => setShowFilters(!showFilters)}
                    className="flex items-center gap-2 text-surface-600 dark:text-surface-300 hover:text-primary-500 transition-colors"
                >
                    <Filter className="w-5 h-5" />
                    <span className="font-medium">Filters</span>
                    <ChevronDown className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                </button>

                {showFilters && (
                    <div className="mt-4 flex flex-wrap gap-4">
                        <div>
                            <label className="block text-sm text-surface-500 mb-2">Sentiment</label>
                            <div className="flex gap-2">
                                {['all', 'positive', 'negative', 'neutral'].map((s) => (
                                    <button
                                        key={s}
                                        onClick={() => setFilter(s as any)}
                                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${filter === s
                                                ? 'bg-primary-500 text-white'
                                                : 'bg-surface-100 dark:bg-surface-700 text-surface-600 dark:text-surface-300 hover:bg-surface-200 dark:hover:bg-surface-600'
                                            }`}
                                    >
                                        {s.charAt(0).toUpperCase() + s.slice(1)}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div>
                            <label className="block text-sm text-surface-500 mb-2">Impact</label>
                            <div className="flex gap-2">
                                {['all', 'high', 'medium', 'low'].map((i) => (
                                    <button
                                        key={i}
                                        onClick={() => setImpactFilter(i as any)}
                                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${impactFilter === i
                                                ? 'bg-primary-500 text-white'
                                                : 'bg-surface-100 dark:bg-surface-700 text-surface-600 dark:text-surface-300 hover:bg-surface-200 dark:hover:bg-surface-600'
                                            }`}
                                    >
                                        {i.charAt(0).toUpperCase() + i.slice(1)}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Loading State */}
            {isLoading && (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
                    <span className="ml-3 text-surface-500">Fetching latest news...</span>
                </div>
            )}

            {/* News List */}
            {!isLoading && (
                <div className="space-y-4">
                    {filteredNews.map((item) => (
                        <div
                            key={item.id}
                            className={`card p-5 border-l-4 ${getImpactColor(item.impact)} hover:shadow-lg transition-all ${item.isWatchlist ? 'ring-2 ring-primary-500/50' : ''
                                }`}
                        >
                            <div className="flex items-start justify-between gap-4">
                                <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                                        {item.isWatchlist && (
                                            <span className="px-2 py-0.5 rounded text-xs font-medium bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 flex items-center gap-1">
                                                <Star className="w-3 h-3 fill-current" />
                                                Watchlist
                                            </span>
                                        )}
                                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${getSentimentColor(item.sentiment)}`}>
                                            {item.sentiment === 'positive' && <TrendingUp className="w-3 h-3 inline mr-1" />}
                                            {item.sentiment === 'negative' && <TrendingDown className="w-3 h-3 inline mr-1" />}
                                            {item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1)}
                                        </span>
                                        <span className="px-2 py-0.5 rounded text-xs font-medium bg-surface-100 dark:bg-surface-700 text-surface-600 dark:text-surface-400">
                                            {item.category}
                                        </span>
                                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${item.impact === 'high'
                                                ? 'bg-danger-100 dark:bg-danger-900/30 text-danger-600 dark:text-danger-400'
                                                : item.impact === 'medium'
                                                    ? 'bg-warning-100 dark:bg-warning-900/30 text-warning-600 dark:text-warning-400'
                                                    : 'bg-surface-100 dark:bg-surface-700 text-surface-500'
                                            }`}>
                                            {item.impact.toUpperCase()} Impact
                                        </span>
                                    </div>
                                    <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">
                                        {item.title}
                                    </h3>
                                    <p className="text-surface-600 dark:text-surface-400 text-sm mb-3 line-clamp-2">
                                        {item.summary}
                                    </p>
                                    {item.relatedStocks && item.relatedStocks.length > 0 && (
                                        <div className="flex items-center gap-2 mb-3">
                                            <span className="text-xs text-surface-500">Related:</span>
                                            {item.relatedStocks.slice(0, 5).map(stock => (
                                                <span key={stock} className="px-2 py-0.5 rounded text-xs font-medium bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400">
                                                    {stock}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                    <div className="flex items-center gap-4 text-xs text-surface-500">
                                        <span className="flex items-center gap-1">
                                            <Globe className="w-3 h-3" />
                                            {item.source}
                                        </span>
                                        <span className="flex items-center gap-1">
                                            <Clock className="w-3 h-3" />
                                            {formatTime(item.publishedAt)}
                                        </span>
                                    </div>
                                </div>
                                <a
                                    href={item.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="p-2 rounded-lg bg-surface-100 dark:bg-surface-700 hover:bg-primary-100 dark:hover:bg-primary-900/30 text-surface-500 hover:text-primary-500 transition-colors"
                                >
                                    <ExternalLink className="w-5 h-5" />
                                </a>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {!isLoading && filteredNews.length === 0 && (
                <div className="text-center py-12">
                    <AlertTriangle className="w-16 h-16 mx-auto text-surface-300 mb-4" />
                    <p className="text-surface-500">
                        {news.length === 0 ? 'Unable to fetch news. Please try again later.' : 'No news matching your filters'}
                    </p>
                </div>
            )}

            {/* Auto-refresh indicator */}
            <div className="text-center text-xs text-surface-400">
                News refreshes automatically every 5 minutes
            </div>

            {/* Disclaimer */}
            <div className="card p-4 bg-surface-50 dark:bg-surface-800/50 border-l-4 border-warning-500">
                <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-warning-500 flex-shrink-0 mt-0.5" />
                    <div>
                        <p className="text-sm font-medium text-surface-700 dark:text-surface-300">Disclaimer</p>
                        <p className="text-xs text-surface-500 mt-1">
                            News sentiment analysis is AI-generated and for informational purposes only.
                            This does not constitute financial advice. Always conduct your own research before making investment decisions.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}

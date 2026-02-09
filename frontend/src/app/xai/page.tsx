'use client';

import { useQuery } from '@tanstack/react-query';
import {
    Brain,
    Lightbulb,
    TrendingUp,
    TrendingDown,
    BarChart3,
    PieChart,
    Activity,
    Zap
} from 'lucide-react';
import { predictionsAPI, stocksAPI } from '@/lib/api';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { ModalityWeights } from '@/components/xai/ModalityWeights';
import { TopFeatures } from '@/components/xai/TopFeatures';

export default function XAIPage() {
    // Fetch latest prediction for XAI data
    const { data: predictionData, isLoading } = useQuery({
        queryKey: ['latest-prediction-xai'],
        queryFn: () => predictionsAPI.getLatest('^NSEI'),
    });

    // Fetch current price to compare prediction against
    const { data: priceData } = useQuery({
        queryKey: ['realtime-price'],
        queryFn: () => stocksAPI.getRealtime('^NSEI'),
    });

    const prediction = predictionData?.data;
    const currentPrice = priceData?.data?.price;

    // Calculate actual direction by comparing predicted_close to current price
    const predictedClose = prediction?.predicted_close ?? 0;
    const actualChangePct = currentPrice && currentPrice > 0
        ? ((predictedClose - currentPrice) / currentPrice) * 100
        : (prediction?.predicted_change_pct ?? 0);
    const isActuallyUp = actualChangePct >= 0;

    // Default explanation text
    const explanationText = prediction?.explanation_text ||
        "The model predicts NIFTY 50 will increase by approximately 0.30% with medium confidence. " +
        "Technical indicators show bullish momentum with RSI in neutral territory. " +
        "News sentiment is mildly positive, contributing to the upward prediction. " +
        "The price modality had the strongest influence on this prediction.";

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-surface-900 dark:text-white">
                        XAI Insights
                    </h1>
                    <p className="text-surface-500 mt-1">
                        Understand why the AI made its predictions
                    </p>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
                    <Lightbulb className="w-5 h-5 text-primary-500" />
                    <span className="text-sm font-medium text-primary-700 dark:text-primary-300">
                        Explainable AI
                    </span>
                </div>
            </div>

            {/* What is XAI Card */}
            <div className="card p-6 bg-gradient-to-r from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 border-primary-200 dark:border-primary-800">
                <div className="flex items-start gap-4">
                    <div className="p-3 bg-primary-500 rounded-xl">
                        <Brain className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">
                            What is Explainable AI?
                        </h2>
                        <p className="text-surface-600 dark:text-surface-400">
                            Traditional AI models are "black boxes" - they give predictions but don't explain why.
                            Our XAI system uses SHAP values, attention weights, and feature importance analysis
                            to show you exactly which factors influenced each prediction and by how much.
                        </p>
                    </div>
                </div>
            </div>

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Modality Weights - Left Column */}
                <div className="card p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="font-semibold text-surface-900 dark:text-white flex items-center gap-2">
                            <PieChart className="w-5 h-5 text-primary-500" />
                            Data Source Weights
                        </h3>
                        <InfoTooltip
                            title="Modality Weights"
                            content="Shows how much each data source contributed to the final prediction. The model dynamically adjusts these weights based on market conditions."
                        />
                    </div>
                    <ModalityWeights weights={prediction?.modality_weights} />

                    <div className="mt-6 p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                        <h4 className="text-sm font-medium text-surface-900 dark:text-white mb-2">
                            What this means:
                        </h4>
                        <ul className="text-sm text-surface-600 dark:text-surface-400 space-y-1">
                            <li>• <strong>Price History:</strong> Past OHLCV patterns</li>
                            <li>• <strong>Sentiment:</strong> News & social signals</li>
                            <li>• <strong>Technical:</strong> RSI, MACD, Bollinger</li>
                        </ul>
                    </div>
                </div>

                {/* Prediction Explanation - Middle Column (spans 2) */}
                <div className="lg:col-span-2 card p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="font-semibold text-surface-900 dark:text-white flex items-center gap-2">
                            <Lightbulb className="w-5 h-5 text-warning-500" />
                            AI Explanation
                        </h3>
                        <InfoTooltip
                            title="Natural Language Explanation"
                            content="A human-readable summary of why the model made this prediction, generated from the underlying XAI analysis."
                        />
                    </div>

                    {/* Direction Indicator */}
                    <div className="flex items-center gap-4 mb-6 p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                        <div className={`p-3 rounded-xl ${isActuallyUp
                            ? 'bg-success-100 dark:bg-success-900/30'
                            : 'bg-danger-100 dark:bg-danger-900/30'
                            }`}>
                            {isActuallyUp ? (
                                <TrendingUp className="w-8 h-8 text-success-600" />
                            ) : (
                                <TrendingDown className="w-8 h-8 text-danger-600" />
                            )}
                        </div>
                        <div>
                            <p className="text-sm text-surface-500">Predicted Direction</p>
                            <p className={`text-xl font-bold ${isActuallyUp
                                ? 'text-success-600'
                                : 'text-danger-600'
                                }`}>
                                {isActuallyUp ? 'BULLISH' : 'BEARISH'}
                                <span className="text-sm font-normal ml-2">
                                    ({actualChangePct >= 0 ? '+' : ''}
                                    {actualChangePct.toFixed(2)}%)
                                </span>
                            </p>
                        </div>
                    </div>

                    {/* Explanation Text */}
                    <div className="prose dark:prose-invert max-w-none">
                        <p className="text-surface-700 dark:text-surface-300 leading-relaxed">
                            {explanationText}
                        </p>
                    </div>

                    {/* Confidence Breakdown */}
                    <div className="mt-6 grid grid-cols-3 gap-4">
                        <div className="text-center p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                            <p className="text-xs text-surface-500 mb-1">Confidence</p>
                            <p className="text-lg font-bold text-primary-600">
                                {((prediction?.confidence_score || 0.72) * 100).toFixed(0)}%
                            </p>
                        </div>
                        <div className="text-center p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                            <p className="text-xs text-surface-500 mb-1">Risk Score</p>
                            <p className="text-lg font-bold text-warning-600">
                                {((prediction?.risk_score || 0.35) * 100).toFixed(0)}%
                            </p>
                        </div>
                        <div className="text-center p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                            <p className="text-xs text-surface-500 mb-1">Volatility</p>
                            <p className="text-lg font-bold text-surface-600 dark:text-surface-400">
                                {(prediction?.volatility_prediction || 0.85).toFixed(2)}
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Feature Importance Section */}
            <div className="card p-6">
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                        <Zap className="w-6 h-6 text-warning-500" />
                        <h2 className="text-xl font-semibold text-surface-900 dark:text-white">
                            Top Contributing Features
                        </h2>
                    </div>
                    <InfoTooltip
                        title="Feature Importance"
                        content="These are the specific features (indicators, news signals, price patterns) that had the most influence on the prediction. Arrows indicate if the feature pushed the prediction up or down."
                    />
                </div>
                <TopFeatures features={prediction?.top_features} />
            </div>

            {/* How to Use This Section */}
            <div className="card p-6">
                <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-primary-500" />
                    How to Use XAI Insights
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                        <div className="text-2xl mb-2">1️⃣</div>
                        <h4 className="font-medium text-surface-900 dark:text-white mb-1">Check Feature Alignment</h4>
                        <p className="text-sm text-surface-600 dark:text-surface-400">
                            If the top features align with your own analysis, the prediction may be more reliable.
                        </p>
                    </div>
                    <div className="p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                        <div className="text-2xl mb-2">2️⃣</div>
                        <h4 className="font-medium text-surface-900 dark:text-white mb-1">Review Modality Weights</h4>
                        <p className="text-sm text-surface-600 dark:text-surface-400">
                            During volatile markets, sentiment may dominate. In stable periods, technical indicators matter more.
                        </p>
                    </div>
                    <div className="p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                        <div className="text-2xl mb-2">3️⃣</div>
                        <h4 className="font-medium text-surface-900 dark:text-white mb-1">Consider Risk Metrics</h4>
                        <p className="text-sm text-surface-600 dark:text-surface-400">
                            High confidence with low risk suggests a stronger signal. High risk suggests caution.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}

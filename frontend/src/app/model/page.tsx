'use client';

import { useQuery } from '@tanstack/react-query';
import {
    Brain,
    Cpu,
    Layers,
    Zap,
    Award,
    ArrowRight,
    BookOpen,
    Shield,
    BarChart3,
    GitBranch,
} from 'lucide-react';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

export default function ModelInfoPage() {
    const { data: modelData, isLoading } = useQuery({
        queryKey: ['model-info'],
        queryFn: async () => {
            const res = await fetch(
                (process.env.NEXT_PUBLIC_API_BASE_URL || '/api/v1') + '/model/info'
            );
            return res.json();
        },
        staleTime: Infinity,
    });

    const info = modelData || null;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-surface-900 dark:text-white">
                        Model Architecture
                    </h1>
                    <p className="text-surface-500 mt-1">
                        NIFTY50-Multimodal-TCN — Multimodal fusion with Adaptive Gating
                    </p>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
                    <Cpu className="w-5 h-5 text-primary-500" />
                    <span className="text-sm font-medium text-primary-700 dark:text-primary-300">
                        {((info?.parameters || 847000) / 1000).toFixed(0)}K Parameters
                    </span>
                </div>
            </div>

            {/* Overview Card */}
            <div className="card p-6 bg-gradient-to-r from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 border-primary-200 dark:border-primary-800">
                <div className="flex items-start gap-4">
                    <div className="p-3 bg-primary-500 rounded-xl flex-shrink-0">
                        <Brain className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">
                            {info?.name || 'NIFTY50-Multimodal-TCN'}
                        </h2>
                        <p className="text-surface-600 dark:text-surface-400">
                            {info?.description || 'Multimodal deep learning model for NIFTY 50 Index prediction using Temporal Convolutional Networks with Adaptive Fusion.'}
                        </p>
                    </div>
                </div>
            </div>

            {/* Architecture Flow */}
            <div className="card p-6">
                <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-primary-500" />
                    Architecture Pipeline
                </h3>
                <div className="flex flex-wrap items-center gap-2 p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                    {[
                        { label: 'Price History (TCN)', color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300', desc: '60-day OHLCV sequences' },
                        { label: 'Sentiment (MLP)', color: 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300', desc: 'News + Reddit scores' },
                        { label: 'Technical (MLP)', color: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300', desc: 'RSI, MACD, ADX, ATR' },
                    ].map((mod, i) => (
                        <span key={mod.label}>
                            <span className={`inline-block px-3 py-2 rounded-lg text-sm font-medium ${mod.color}`}>
                                {mod.label}
                                <span className="block text-xs opacity-70 font-normal">{mod.desc}</span>
                            </span>
                            {i < 2 && <span className="text-surface-400 mx-1">+</span>}
                        </span>
                    ))}
                    <ArrowRight className="w-5 h-5 text-surface-400 mx-2" />
                    <span className="px-3 py-2 rounded-lg text-sm font-medium bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
                        Adaptive Fusion Gate
                        <span className="block text-xs opacity-70 font-normal">Dynamic modality weighting</span>
                    </span>
                    <ArrowRight className="w-5 h-5 text-surface-400 mx-2" />
                    <span className="px-3 py-2 rounded-lg text-sm font-medium bg-primary-100 text-primary-700 dark:bg-primary-900/40 dark:text-primary-300">
                        Multi-Output Head
                        <span className="block text-xs opacity-70 font-normal">Return % + Quantiles + Direction</span>
                    </span>
                </div>
            </div>

            {/* Component Details */}
            <div className="card p-6">
                <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                    <Cpu className="w-5 h-5 text-primary-500" />
                    Component Details
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {(info?.architecture?.components || []).map((comp: any) => (
                        <div
                            key={comp.name}
                            className="p-5 bg-surface-50 dark:bg-surface-700/50 rounded-xl border border-surface-200 dark:border-surface-600 hover:shadow-lg transition-shadow"
                        >
                            <h4 className="font-semibold text-surface-900 dark:text-white text-sm mb-2">
                                {comp.name}
                            </h4>
                            <p className="text-xs text-surface-500 mb-3">
                                {comp.description}
                            </p>
                            <div className="text-xs space-y-1.5 mb-3">
                                <div>
                                    <span className="text-surface-400">Input: </span>
                                    <code className="text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/30 px-1.5 py-0.5 rounded text-xs">
                                        {comp.input}
                                    </code>
                                </div>
                                <div>
                                    <span className="text-surface-400">Output: </span>
                                    <code className="text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 px-1.5 py-0.5 rounded text-xs">
                                        {comp.output}
                                    </code>
                                </div>
                            </div>
                            {comp.key_innovation && (
                                <div className="p-2.5 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                                    <p className="text-xs text-amber-700 dark:text-amber-300 flex items-start gap-1.5">
                                        <Zap className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                                        <strong>Innovation:</strong> {comp.key_innovation}
                                    </p>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Training + Inference */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Training */}
                <div className="card p-6">
                    <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                        <BarChart3 className="w-5 h-5 text-primary-500" />
                        Training Configuration
                    </h3>
                    <div className="space-y-3">
                        {info?.training && Object.entries(info.training).map(([key, value]) => (
                            <div key={key} className="flex items-center justify-between p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                                <span className="text-sm text-surface-500 capitalize">{key.replace(/_/g, ' ')}</span>
                                <span className="text-sm font-semibold text-surface-900 dark:text-white">
                                    {String(value)}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Inference */}
                <div className="card p-6">
                    <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                        <Shield className="w-5 h-5 text-primary-500" />
                        Inference Pipeline
                    </h3>
                    <div className="space-y-3">
                        {info?.inference && Object.entries(info.inference).map(([key, value]) => (
                            <div key={key} className="flex items-center justify-between p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                                <span className="text-sm text-surface-500 capitalize">{key.replace(/_/g, ' ')}</span>
                                <span className="text-sm font-semibold text-surface-900 dark:text-white text-right max-w-[60%]">
                                    {String(value)}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* XAI Methods */}
            <div className="card p-6">
                <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-primary-500" />
                    Explainability Methods
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {(info?.explainability?.methods || []).map((method: string, i: number) => (
                        <div key={i} className="flex items-center gap-3 p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                            <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900/40 text-primary-600 dark:text-primary-400 text-sm font-bold">
                                {i + 1}
                            </span>
                            <span className="text-sm text-surface-700 dark:text-surface-300">{method}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Novelty Points */}
            <div className="card p-6">
                <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                    <Award className="w-5 h-5 text-amber-500" />
                    Novelty &amp; Research Contributions
                </h3>
                <p className="text-sm text-surface-500 mb-4">
                    Key differentiators that make this system novel compared to existing stock prediction approaches:
                </p>
                <div className="space-y-3">
                    {(info?.novelty_points || []).map((point: string, i: number) => (
                        <div
                            key={i}
                            className="flex items-start gap-3 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-xl border border-amber-200 dark:border-amber-800"
                        >
                            <span className="flex-shrink-0 w-7 h-7 flex items-center justify-center rounded-full bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200 text-xs font-bold">
                                {i + 1}
                            </span>
                            <p className="text-sm text-surface-700 dark:text-surface-300 pt-0.5">
                                {point}
                            </p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Comparison Table */}
            <div className="card p-6">
                <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                    <GitBranch className="w-5 h-5 text-primary-500" />
                    Comparison with Existing Approaches
                </h3>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-surface-200 dark:border-surface-700">
                                <th className="text-left py-3 px-4 text-surface-500 font-medium">Feature</th>
                                <th className="text-center py-3 px-4 text-surface-500 font-medium">Traditional ML</th>
                                <th className="text-center py-3 px-4 text-surface-500 font-medium">LSTM-only</th>
                                <th className="text-center py-3 px-4 font-medium text-primary-600 dark:text-primary-400">Our System</th>
                            </tr>
                        </thead>
                        <tbody>
                            {[
                                ['Multimodal Input', '❌', '❌', '✅'],
                                ['Parallelizable Training', '✅', '❌', '✅ (TCN)'],
                                ['Dynamic Feature Weighting', '❌', '❌', '✅ (Fusion Gate)'],
                                ['Uncertainty Quantification', '❌', '❌', '✅ (Quantiles)'],
                                ['Built-in Explainability', '❌', '❌', '✅ (SHAP + Gate)'],
                                ['Sentiment Integration', '❌', 'Sometimes', '✅ (News + Reddit)'],
                                ['Realistic Constraints', '❌', '❌', '✅ (±2% clamp)'],
                                ['Trading Signals', '❌', '❌', '✅ (BUY/HOLD/SELL)'],
                            ].map(([feature, trad, lstm, ours]) => (
                                <tr key={feature} className="border-b border-surface-100 dark:border-surface-700/50">
                                    <td className="py-2.5 px-4 text-surface-900 dark:text-white font-medium">{feature}</td>
                                    <td className="py-2.5 px-4 text-center">{trad}</td>
                                    <td className="py-2.5 px-4 text-center">{lstm}</td>
                                    <td className="py-2.5 px-4 text-center font-medium text-primary-600 dark:text-primary-400">{ours}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

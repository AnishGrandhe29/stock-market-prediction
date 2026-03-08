'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    Brain,
    ChevronDown,
    ChevronUp,
    Cpu,
    Layers,
    Zap,
    Award,
    ArrowRight,
} from 'lucide-react';

export function ModelArchitecturePanel() {
    const [isExpanded, setIsExpanded] = useState(false);

    const { data: modelData } = useQuery({
        queryKey: ['model-info'],
        queryFn: async () => {
            const res = await fetch(
                (process.env.NEXT_PUBLIC_API_BASE_URL || '/api/v1') + '/model/info'
            );
            return res.json();
        },
        staleTime: Infinity, // Static data, never refetch
    });

    const info = modelData || null;

    return (
        <div className="card p-6">
            {/* Header — always visible */}
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full flex items-center justify-between"
            >
                <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl">
                        <Brain className="w-6 h-6 text-white" />
                    </div>
                    <div className="text-left">
                        <h3 className="font-semibold text-surface-900 dark:text-white">
                            Model Architecture
                        </h3>
                        <p className="text-sm text-surface-500">
                            {info?.name || 'NIFTY50-Multimodal-TCN'} &bull; {((info?.parameters || 847000) / 1000).toFixed(0)}K params
                        </p>
                    </div>
                </div>
                {isExpanded ? (
                    <ChevronUp className="w-5 h-5 text-surface-400" />
                ) : (
                    <ChevronDown className="w-5 h-5 text-surface-400" />
                )}
            </button>

            {/* Expandable content */}
            {isExpanded && (
                <div className="mt-6 space-y-6 animate-slide-up">
                    {/* Architecture Flow */}
                    <div>
                        <h4 className="text-sm font-medium text-surface-900 dark:text-white mb-3 flex items-center gap-2">
                            <Layers className="w-4 h-4 text-primary-500" />
                            Pipeline Flow
                        </h4>
                        <div className="flex flex-wrap items-center gap-2 p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl">
                            {[
                                { label: 'Price (TCN)', color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300' },
                                { label: 'Sentiment (MLP)', color: 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300' },
                                { label: 'Technical (MLP)', color: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300' },
                            ].map((mod, i) => (
                                <span key={mod.label}>
                                    <span className={`inline-block px-3 py-1.5 rounded-lg text-sm font-medium ${mod.color}`}>
                                        {mod.label}
                                    </span>
                                    {i < 2 && <span className="text-surface-400 mx-1">+</span>}
                                </span>
                            ))}
                            <ArrowRight className="w-4 h-4 text-surface-400 mx-1" />
                            <span className="px-3 py-1.5 rounded-lg text-sm font-medium bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
                                Adaptive Fusion Gate
                            </span>
                            <ArrowRight className="w-4 h-4 text-surface-400 mx-1" />
                            <span className="px-3 py-1.5 rounded-lg text-sm font-medium bg-primary-100 text-primary-700 dark:bg-primary-900/40 dark:text-primary-300">
                                Prediction Head
                            </span>
                        </div>
                    </div>

                    {/* Component Details */}
                    {info?.architecture?.components && (
                        <div>
                            <h4 className="text-sm font-medium text-surface-900 dark:text-white mb-3 flex items-center gap-2">
                                <Cpu className="w-4 h-4 text-primary-500" />
                                Component Details
                            </h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                                {info.architecture.components.map((comp: any) => (
                                    <div
                                        key={comp.name}
                                        className="p-4 bg-surface-50 dark:bg-surface-700/50 rounded-xl border border-surface-200 dark:border-surface-600 hover:shadow-md transition-shadow"
                                    >
                                        <h5 className="font-medium text-surface-900 dark:text-white text-sm mb-1">
                                            {comp.name}
                                        </h5>
                                        <p className="text-xs text-surface-500 mb-2 line-clamp-2">
                                            {comp.description}
                                        </p>
                                        <div className="text-xs space-y-1">
                                            <div>
                                                <span className="text-surface-400">In: </span>
                                                <code className="text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/30 px-1 rounded">
                                                    {comp.input}
                                                </code>
                                            </div>
                                            <div>
                                                <span className="text-surface-400">Out: </span>
                                                <code className="text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 px-1 rounded">
                                                    {comp.output}
                                                </code>
                                            </div>
                                        </div>
                                        {comp.key_innovation && (
                                            <p className="mt-2 text-xs text-amber-600 dark:text-amber-400 flex items-start gap-1">
                                                <Zap className="w-3 h-3 mt-0.5 flex-shrink-0" />
                                                {comp.key_innovation}
                                            </p>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Training Details */}
                    {info?.training && (
                        <div>
                            <h4 className="text-sm font-medium text-surface-900 dark:text-white mb-3 flex items-center gap-2">
                                <Zap className="w-4 h-4 text-primary-500" />
                                Training Configuration
                            </h4>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                {Object.entries(info.training).map(([key, value]) => (
                                    <div key={key} className="p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg text-center">
                                        <p className="text-xs text-surface-500 mb-1 capitalize">
                                            {key.replace(/_/g, ' ')}
                                        </p>
                                        <p className="text-sm font-semibold text-surface-900 dark:text-white truncate">
                                            {String(value)}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Novelty Points */}
                    {info?.novelty_points && (
                        <div>
                            <h4 className="text-sm font-medium text-surface-900 dark:text-white mb-3 flex items-center gap-2">
                                <Award className="w-4 h-4 text-amber-500" />
                                Novelty &amp; Contributions
                            </h4>
                            <div className="space-y-2">
                                {info.novelty_points.map((point: string, i: number) => (
                                    <div
                                        key={i}
                                        className="flex items-start gap-3 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800"
                                    >
                                        <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center rounded-full bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200 text-xs font-bold">
                                            {i + 1}
                                        </span>
                                        <p className="text-sm text-surface-700 dark:text-surface-300">
                                            {point}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

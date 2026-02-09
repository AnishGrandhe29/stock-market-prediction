'use client';

import { TrendingUp, MessageSquare, BarChart3 } from 'lucide-react';

interface ModalityWeightsProps {
    weights?: {
        price: number;
        sentiment: number;
        technical: number;
    };
}

export function ModalityWeights({ weights }: ModalityWeightsProps) {
    const data = weights || {
        price: 0.45,
        sentiment: 0.30,
        technical: 0.25,
    };

    const modalities = [
        {
            name: 'Price History',
            key: 'price',
            icon: TrendingUp,
            color: 'from-blue-500 to-blue-600',
            description: 'OHLCV patterns & trends',
        },
        {
            name: 'Market Sentiment',
            key: 'sentiment',
            icon: MessageSquare,
            color: 'from-purple-500 to-purple-600',
            description: 'News & social media',
        },
        {
            name: 'Technical Indicators',
            key: 'technical',
            icon: BarChart3,
            color: 'from-emerald-500 to-emerald-600',
            description: 'RSI, MACD, Bollinger',
        },
    ];

    return (
        <div className="space-y-4" >
            {
                modalities.map((modality) => {
                    const weight = data[modality.key as keyof typeof data];
                    const percentage = (weight * 100).toFixed(0);

                    return (
                        <div key={modality.key} className="space-y-2" >
                            <div className="flex items-center justify-between" >
                                <div className="flex items-center gap-2" >
                                    <modality.icon className="w-4 h-4 text-surface-700 dark:text-surface-300" />
                                    <span className="text-sm font-medium text-surface-900 dark:text-white" >
                                        {modality.name}
                                    </span>
                                </div>
                                < span className="text-sm font-bold text-surface-900 dark:text-white" >
                                    {percentage} %
                                </span>
                            </div>
                            < div className="h-2 bg-surface-100 dark:bg-surface-700 rounded-full overflow-hidden" >
                                <div
                                    className={`h-full bg-gradient-to-r ${modality.color} rounded-full transition-all duration-500`}
                                    style={{ width: `${percentage}%` }
                                    }
                                />
                            </div>
                            <p className="text-xs text-surface-700 dark:text-surface-300" > {modality.description} </p>
                        </div>
                    );
                })
            }
        </div>
    );
}

'use client';

import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle } from 'lucide-react';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

interface Prediction {
    predicted_close: number;
    predicted_change_pct: number;
    predicted_direction: string;
    direction_probability: number;
    confidence_level: string;
    uncertainty_score: number;
    quantile_5: number;
    quantile_95: number;
    target_date: string;
}

interface PredictionCardProps {
    prediction?: Prediction;
    isLoading?: boolean;
}

export function PredictionCard({ prediction, isLoading }: PredictionCardProps) {
    if (isLoading) {
        return (
            <div className= "card p-6" >
            <div className="skeleton h-6 w-32 mb-4" />
                <div className="skeleton h-10 w-40 mb-2" />
                    <div className="skeleton h-4 w-24" />
                        </div>
    );
    }

    const isUp = prediction?.predicted_direction === 'up';
    const confidenceColor = {
        high: 'text-success-500 bg-success-50 dark:bg-success-500/20',
        medium: 'text-primary-500 bg-primary-50 dark:bg-primary-500/20',
        low: 'text-danger-500 bg-danger-50 dark:bg-danger-500/20',
    }[prediction?.confidence_level || 'medium'];

    return (
        <div className= "card p-6 card-hover" >
        <div className="flex items-center justify-between mb-4" >
            <h3 className="font-semibold text-surface-900 dark:text-white" >
                AI Prediction
                    </h3>
                    < InfoTooltip
    title = "AI Prediction"
    content = "Our multimodal deep learning model analyzes price history, technical indicators, and market sentiment to predict the next trading day's closing price."
        />
        </div>

    {/* Predicted Value */ }
    <div className="mb-4" >
        <span className="text-sm text-surface-500" > Tomorrow's Close</span>
            < div className = "text-3xl font-bold text-surface-900 dark:text-white" >
          ₹{ prediction?.predicted_close?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '—' }
    </div>
        </div>

    {/* Direction */ }
    <div className={ `flex items-center gap-2 mb-4 ${isUp ? 'positive' : 'negative'}` }>
        {
            isUp?(
          <TrendingUp className = "w-5 h-5" />
        ): (
                    <TrendingDown className = "w-5 h-5" />
        )
}
<span className="font-semibold text-lg" >
    { isUp? '+': '' }{ prediction?.predicted_change_pct?.toFixed(2) || 0 }%
        </span>
        < span className = "text-sm opacity-80" >
            ({(prediction?.direction_probability || 0) * 100}% confidence)
</span>
    </div>

{/* Confidence Badge */ }
<div className="flex items-center gap-2 mb-4" >
    <span className={ `px-3 py-1 rounded-full text-sm font-medium ${confidenceColor}` }>
        { prediction?.confidence_level === 'high' && <CheckCircle className="w-4 h-4 inline mr-1" />}
{ prediction?.confidence_level === 'low' && <AlertTriangle className="w-4 h-4 inline mr-1" />}
{ prediction?.confidence_level?.toUpperCase() || 'MEDIUM' } CONFIDENCE
    </span>
    </div>

{/* Range */ }
<div className="p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg" >
    <div className="flex items-center justify-between mb-2" >
        <span className="text-sm text-surface-500" > Prediction Range(90 %) </span>
            < InfoTooltip
title = "Prediction Range"
content = "The model is 90% confident the actual price will fall within this range. Wider ranges indicate more uncertainty."
    />
    </div>
    < div className = "flex items-center justify-between text-sm" >
        <span className="text-danger-500" >
            ₹{ prediction?.quantile_5?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '—' }
</span>
    < div className = "flex-1 mx-3 h-2 bg-surface-200 dark:bg-surface-600 rounded-full overflow-hidden" >
        <div className="h-full bg-gradient-to-r from-danger-500 via-primary-500 to-success-500 rounded-full" />
            </div>
            < span className = "text-success-500" >
            ₹{ prediction?.quantile_95?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '—' }
</span>
    </div>
    </div>

{/* Target Date */ }
<div className="mt-4 text-xs text-surface-500 text-center" >
    Prediction for: { prediction?.target_date || 'Next Trading Day' }
</div>
    </div>
  );
}

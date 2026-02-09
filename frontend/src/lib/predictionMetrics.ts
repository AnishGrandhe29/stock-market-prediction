/**
 * Prediction Metrics Utilities
 * 
 * Provides functions to calculate and validate prediction confidence levels,
 * prediction ranges, and ensure consistency between displayed values.
 * 
 * Confidence Level Thresholds:
 * - LOW: 0-40%
 * - MEDIUM: 41-70%
 * - HIGH: 71-100%
 */

export type ConfidenceLevel = 'low' | 'medium' | 'high';

export interface ConfidenceMetrics {
    score: number;           // 0-1 normalized score
    percentage: number;      // 0-100 percentage
    level: ConfidenceLevel;  // Categorical label
    isValid: boolean;        // Whether metrics are consistent
}

export interface PredictionRange {
    lower: number;           // Lower bound (5th percentile)
    upper: number;           // Upper bound (95th percentile)
    isValid: boolean;        // Whether range could be computed
    errorMessage?: string;   // Error message if invalid
}

/**
 * Confidence level thresholds
 */
const CONFIDENCE_THRESHOLDS = {
    low: { min: 0, max: 40 },
    medium: { min: 41, max: 70 },
    high: { min: 71, max: 100 }
};

/**
 * Z-score for 90% confidence interval (two-tailed)
 * P(−z < Z < z) = 0.90 → z ≈ 1.645
 */
const Z_SCORE_90_PERCENT = 1.645;

/**
 * Determine confidence level from percentage
 */
export function getConfidenceLevel(percentage: number): ConfidenceLevel {
    if (percentage <= CONFIDENCE_THRESHOLDS.low.max) {
        return 'low';
    } else if (percentage <= CONFIDENCE_THRESHOLDS.medium.max) {
        return 'medium';
    } else {
        return 'high';
    }
}

/**
 * Validate and normalize confidence metrics
 * Ensures percentage and level are consistent
 * 
 * @param score - Raw confidence score (0-1 or 0-100)
 * @param level - Optional categorical level (if provided, used for validation)
 * @returns Normalized and validated confidence metrics
 */
export function normalizeConfidence(
    score?: number,
    level?: string
): ConfidenceMetrics {
    // Handle missing score
    if (score === undefined || score === null || isNaN(score)) {
        // If level is provided, derive a reasonable score
        if (level) {
            const defaultScores: Record<string, number> = {
                low: 0.30,
                medium: 0.55,
                high: 0.85
            };
            const normalizedLevel = level.toLowerCase() as ConfidenceLevel;
            const derivedScore = defaultScores[normalizedLevel] ?? 0.55;
            return {
                score: derivedScore,
                percentage: derivedScore * 100,
                level: normalizedLevel,
                isValid: true
            };
        }
        // Default to medium confidence
        return {
            score: 0.55,
            percentage: 55,
            level: 'medium',
            isValid: false
        };
    }

    // Normalize score to 0-1 range
    let normalizedScore = score;
    if (score > 1) {
        normalizedScore = score / 100;
    }

    // Clamp to valid range
    normalizedScore = Math.max(0, Math.min(1, normalizedScore));
    const percentage = Math.round(normalizedScore * 100);

    // Determine correct level from percentage
    const computedLevel = getConfidenceLevel(percentage);

    // Check if provided level matches computed level
    const isValid = !level || level.toLowerCase() === computedLevel;

    return {
        score: normalizedScore,
        percentage,
        level: computedLevel,
        isValid
    };
}

/**
 * Calculate 90% prediction range using volatility
 * 
 * @param predictedClose - The predicted closing price
 * @param volatility - Volatility measure (ATR, rolling std, or uncertainty score)
 * @param currentPrice - Current price (optional, for sanity checks)
 * @returns Prediction range with lower and upper bounds
 */
export function calculatePredictionRange(
    predictedClose?: number,
    volatility?: number,
    currentPrice?: number
): PredictionRange {
    // Validate inputs
    if (!predictedClose || predictedClose <= 0) {
        return {
            lower: 0,
            upper: 0,
            isValid: false,
            errorMessage: 'Prediction range unavailable: missing predicted price'
        };
    }

    // If volatility is not provided, estimate from typical NIFTY volatility
    // NIFTY 50 typically has ~1% daily volatility
    let vol = volatility;
    if (!vol || vol <= 0 || isNaN(vol)) {
        // Use 1% of predicted price as default volatility (conservative estimate)
        vol = predictedClose * 0.01;
    }

    // For very small volatility values (likely normalized 0-1), convert to price terms
    if (vol < 1) {
        vol = predictedClose * vol;
    }

    // Calculate bounds using z-score for 90% CI
    const lower = predictedClose - (Z_SCORE_90_PERCENT * vol);
    const upper = predictedClose + (Z_SCORE_90_PERCENT * vol);

    // Sanity check: bounds should be positive and reasonable
    if (lower <= 0 || upper <= 0) {
        return {
            lower: 0,
            upper: 0,
            isValid: false,
            errorMessage: 'Prediction range unavailable: calculation error'
        };
    }

    // Sanity check: range shouldn't be more than 10% of price
    const rangePercent = ((upper - lower) / predictedClose) * 100;
    if (rangePercent > 10) {
        // Cap at reasonable bounds
        const cappedRange = predictedClose * 0.05; // 5% range on each side
        return {
            lower: Math.round(predictedClose - cappedRange),
            upper: Math.round(predictedClose + cappedRange),
            isValid: true
        };
    }

    return {
        lower: Math.round(lower),
        upper: Math.round(upper),
        isValid: true
    };
}

/**
 * Format confidence display string
 */
export function formatConfidenceDisplay(metrics: ConfidenceMetrics): string {
    return `${metrics.percentage}% confidence`;
}

/**
 * Get color classes for confidence level
 */
export function getConfidenceColors(level: ConfidenceLevel): {
    text: string;
    bg: string;
    border: string;
} {
    switch (level) {
        case 'high':
            return {
                text: 'text-emerald-600 dark:text-emerald-400',
                bg: 'bg-emerald-50 dark:bg-emerald-500/20',
                border: 'border-emerald-200 dark:border-emerald-800'
            };
        case 'medium':
            return {
                text: 'text-blue-600 dark:text-blue-400',
                bg: 'bg-blue-50 dark:bg-blue-500/20',
                border: 'border-blue-200 dark:border-blue-800'
            };
        case 'low':
            return {
                text: 'text-rose-600 dark:text-rose-400',
                bg: 'bg-rose-50 dark:bg-rose-500/20',
                border: 'border-rose-200 dark:border-rose-800'
            };
    }
}

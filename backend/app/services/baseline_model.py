"""
Baseline Model: GIFT NIFTY Gap Predictor
=========================================
Implements the mandatory simple baseline BEFORE any deep learning changes.

Baseline formula
----------------
    pred_open = prev_close + (gift_last - prev_close)
             ≡ gift_last

This is the "naive carry-forward" of the GIFT NIFTY signal: if GIFT NIFTY
sits at a level L before market open, NIFTY 50 will open at L.

This is a strong baseline because GIFT NIFTY is a direct NIFTY futures
contract — market participants *price in* the opening level through arbitrage.

Evaluation metrics
------------------
    MAE             : Mean Absolute Error in index points
    RMSE            : Root Mean Squared Error in index points
    MAPE            : Mean Absolute Percentage Error
    Direction Acc.  : % of days where predicted direction == actual direction
                      (positive gap → open > prev_close, and vice-versa)
    R²              : Coefficient of determination

Usage
-----
    from app.services.baseline_model import GIFTNiftyBaseline, evaluate_baseline
    baseline = GIFTNiftyBaseline()
    result = evaluate_baseline(training_df)
    print(result.summary())
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline predictor
# ──────────────────────────────────────────────────────────────────────────────

class GIFTNiftyBaseline:
    """
    Dead-simple GIFT NIFTY carry-forward baseline.

    Prediction: pred_open = prev_close + gap_abs
                          = prev_close + (gift_last - prev_close)
                          = gift_last

    The baseline assigns 100% weight to the GIFT signal and zero to all
    other information. The deep-learning model should beat this on most
    metrics, particularly on high-volatility days where the relationship
    between GIFT and NIFTY open breaks down.
    """

    name = "GIFT-Carry Baseline"

    def predict_open(self, prev_close: float, gift_last: float) -> float:
        """
        Single-sample prediction.

        Parameters
        ----------
        prev_close : NIFTY 50 close on the previous trading day.
        gift_last  : Last GIFT NIFTY price before 09:15 IST.

        Returns
        -------
        Predicted NIFTY 50 opening price (in index points).
        """
        return float(gift_last)

    def predict_gap(self, prev_close: float, gift_last: float) -> float:
        """Returns the predicted opening gap in index points."""
        return float(gift_last - prev_close)

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorised prediction on a DataFrame with columns
        ['prev_close', 'gift_last'].
        """
        return df["gift_last"].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineEvalResult:
    model_name       : str
    n_samples        : int
    mae_points       : float
    rmse_points      : float
    mape_pct         : float
    direction_acc    : float   # 0–1
    r2               : float
    mae_gap_points   : float   # gap prediction MAE
    rmse_gap_points  : float
    # Breakdowns
    mae_high_vol     : Optional[float] = None   # MAE on volatile days
    mae_low_vol      : Optional[float] = None
    direction_high   : Optional[float] = None   # direction acc on large-gap days
    direction_low    : Optional[float] = None
    # Per-day arrays (optional, for plotting)
    pred_opens       : Optional[np.ndarray] = field(default=None, repr=False)
    actual_opens     : Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            f"{'=' * 55}",
            f"  Evaluation: {self.model_name}",
            f"{'=' * 55}",
            f"  Samples                : {self.n_samples}",
            f"  --- Open Price Errors ---",
            f"  MAE          (pts)     : {self.mae_points:.2f}",
            f"  RMSE         (pts)     : {self.rmse_points:.2f}",
            f"  MAPE                   : {self.mape_pct:.3f}%",
            f"  R²                     : {self.r2:.4f}",
            f"  --- Gap Errors ---",
            f"  Gap MAE      (pts)     : {self.mae_gap_points:.2f}",
            f"  Gap RMSE     (pts)     : {self.rmse_gap_points:.2f}",
            f"  --- Directional ---",
            f"  Direction Accuracy     : {self.direction_acc * 100:.1f}%",
        ]
        if self.mae_high_vol is not None:
            lines += [
                f"  --- Volatility Breakdown ---",
                f"  MAE (high-vol days)    : {self.mae_high_vol:.2f}",
                f"  MAE (low-vol  days)    : {self.mae_low_vol:.2f}",
                f"  Dir Acc (large gap)    : {self.direction_high * 100:.1f}%",
                f"  Dir Acc (small gap)    : {self.direction_low * 100:.1f}%",
            ]
        lines.append("=" * 55)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model"           : self.model_name,
            "n_samples"       : self.n_samples,
            "mae_pts"         : round(self.mae_points, 2),
            "rmse_pts"        : round(self.rmse_points, 2),
            "mape_pct"        : round(self.mape_pct, 4),
            "direction_acc"   : round(self.direction_acc, 4),
            "r2"              : round(self.r2, 4),
            "mae_gap_pts"     : round(self.mae_gap_points, 2),
            "rmse_gap_pts"    : round(self.rmse_gap_points, 2),
            "mae_high_vol"    : round(self.mae_high_vol, 2) if self.mae_high_vol else None,
            "mae_low_vol"     : round(self.mae_low_vol, 2) if self.mae_low_vol else None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_baseline(
    feat_df: pd.DataFrame,
    volatility_threshold_pct: float = 0.5
) -> BaselineEvalResult:
    """
    Evaluate the GIFT carry-forward baseline on the full historical dataset.

    Parameters
    ----------
    feat_df : DataFrame produced by build_full_training_dataset().
              Must contain columns: nifty_open, prev_close, gift_last.
    volatility_threshold_pct : Gap percentile threshold to define
                                'high volatility' days. Default: 0.5
                                (above median gap is 'high').

    Returns
    -------
    BaselineEvalResult
    """
    required = {"nifty_open", "prev_close", "gift_last"}
    missing  = required - set(feat_df.columns)
    if missing:
        raise ValueError(f"feat_df missing required columns: {missing}")

    df = feat_df[list(required)].dropna()
    if df.empty:
        raise ValueError("No valid rows after dropping NaN.")

    baseline = GIFTNiftyBaseline()
    pred_opens  = baseline.predict_batch(df).values
    actual_opens = df["nifty_open"].values
    prev_closes  = df["prev_close"].values

    # ── Open price metrics ──
    errors   = np.abs(pred_opens - actual_opens)
    mae_pts  = float(errors.mean())
    rmse_pts = float(np.sqrt(((pred_opens - actual_opens) ** 2).mean()))
    mape     = float((errors / (np.abs(actual_opens) + 1e-8)).mean() * 100)
    r2       = float(r2_score(actual_opens, pred_opens))

    # ── Gap metrics ──
    actual_gaps = actual_opens - prev_closes
    pred_gaps   = pred_opens - prev_closes
    mae_gap  = float(np.abs(pred_gaps - actual_gaps).mean())
    rmse_gap = float(np.sqrt(((pred_gaps - actual_gaps) ** 2).mean()))

    # ── Direction accuracy ──
    pred_dir   = np.sign(pred_gaps)
    actual_dir = np.sign(actual_gaps)
    direction_acc = float((pred_dir == actual_dir).mean())

    # ── Volatility breakdown ──
    abs_gaps   = np.abs(actual_gaps)
    median_gap = np.median(abs_gaps)

    high_vol_mask = abs_gaps >= median_gap
    low_vol_mask  = ~high_vol_mask

    mae_high_vol     = float(errors[high_vol_mask].mean()) if high_vol_mask.any() else None
    mae_low_vol      = float(errors[low_vol_mask].mean())  if low_vol_mask.any()  else None
    direction_high   = float((pred_dir[high_vol_mask] == actual_dir[high_vol_mask]).mean()) \
                       if high_vol_mask.any() else None
    direction_low    = float((pred_dir[low_vol_mask] == actual_dir[low_vol_mask]).mean()) \
                       if low_vol_mask.any() else None

    result = BaselineEvalResult(
        model_name      = baseline.name,
        n_samples       = len(df),
        mae_points      = mae_pts,
        rmse_points     = rmse_pts,
        mape_pct        = mape,
        direction_acc   = direction_acc,
        r2              = r2,
        mae_gap_points  = mae_gap,
        rmse_gap_points = rmse_gap,
        mae_high_vol    = mae_high_vol,
        mae_low_vol     = mae_low_vol,
        direction_high  = direction_high,
        direction_low   = direction_low,
        pred_opens      = pred_opens,
        actual_opens    = actual_opens,
    )

    logger.info("Baseline evaluation complete.\n%s", result.summary())
    return result


def compare_models(
    baseline_result: BaselineEvalResult,
    model_results: "dict[str, BaselineEvalResult]"
) -> pd.DataFrame:
    """
    Returns a clean comparison table: baseline vs all model variants.
    Useful for ablation studies (Step 5).

    Parameters
    ----------
    baseline_result : Output of evaluate_baseline().
    model_results   : Dict of {model_name: BaselineEvalResult}.

    Returns
    -------
    pd.DataFrame with rows = models, columns = metrics.
    """
    rows = [baseline_result.to_dict()]
    for name, result in model_results.items():
        d = result.to_dict()
        d["model"] = name
        rows.append(d)

    df = pd.DataFrame(rows).set_index("model")

    # Compute % improvement over baseline
    base_mae = rows[0]["mae_pts"]
    df["mae_improvement_pct"] = (base_mae - df["mae_pts"]) / (base_mae + 1e-8) * 100

    return df.round(4)

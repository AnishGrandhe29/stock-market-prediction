"""
GIFT NIFTY: Gap-Target Training & Ablation Experiments
=======================================================
This script trains ACMI++ variants with and without GIFT NIFTY features
and runs the comparison study described in Step 4 (target reformulation)
and Step 5 (ablation / validation).

How to run
----------
    cd training/
    python gift_nifty_ablation.py

What it does
------------
1. Builds the full training dataset (NIFTY + GIFT features).
2. Runs the GIFT carry-forward baseline and prints metrics.
3. Defines gap-prediction loss (Pinball + MSE blend).
4. Trains two ACMIPlusPlus variants:
     Model A – WITHOUT GIFT (use_gift=False, null token forced)
     Model B – WITH GIFT    (use_gift=True, full overnight encoder)
5. Evaluates both on a holdout set and prints a comparison table.
6. Saves trained weights to models/ directory.

Reformulated target
-------------------
    Instead of predicting next-day log-return, the dedicated `gap_head`
    predicts:

        gap = NIFTY_open_{t} - NIFTY_close_{t-1}       [index points]

    Then:
        pred_open = NIFTY_close_{t-1} + predicted_gap

    Loss:
        gap_loss = λ₁ * PinballLoss(quantiles, true_gap)
                 + λ₂ * MSELoss(point_pred, true_gap)
                 + existing multi-task losses (unchanged)

    Setting λ₂ >> λ₁ during early training helps the gap head converge
    quickly before the quantile heads regularise the distribution.
"""

import sys
import logging
import json
from pathlib import Path
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Project path setup ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gift_ablation")

# ── Project imports ──
from backend.app.ml.models.acmi import ACMIPlusPlus
from backend.app.services.gift_nifty_pipeline import (
    build_full_training_dataset,
    GIFT_FEATURE_COLS,
    N_GIFT_FEATURES,
)
from backend.app.services.baseline_model import evaluate_baseline, compare_models

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters & Config
# ──────────────────────────────────────────────────────────────────────────────

CFG = {
    "TICKERS"        : ["AAPL","MSFT","NVDA","GOOGL","AMZN"],
    "MACRO_TICKERS"  : ["^VIX","^TNX","DX-Y.NYB","GLD","SPY"],
    "SEQ_LEN"        : 60,
    "D_MODEL"        : 128,
    "N_HEADS"        : 4,
    "TCN_LAYERS"     : 6,
    "N_REGIMES"      : 5,
    "HORIZONS"       : [1, 5, 20, 60],
    "QUANTILES"      : [0.1, 0.25, 0.5, 0.75, 0.9],
    "N_GIFT_FEATURES": N_GIFT_FEATURES,
}

TRAIN_CFG = {
    "EPOCHS"       : 40,
    "BATCH_SIZE"   : 32,
    "LR"           : 3e-4,
    "WEIGHT_DECAY" : 1e-4,
    "TRAIN_SPLIT"  : 0.80,
    "VAL_SPLIT"    : 0.10,
    # test = remaining 10%
    "LAMBDA_MSE"   : 1.0,    # weight on MSE gap loss
    "LAMBDA_PIN"   : 0.5,    # weight on pinball quantile loss
    "LAMBDA_MULTI" : 0.3,    # weight on existing multi-horizon return heads
    "PATIENCE"     : 7,      # early stopping patience
}

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

NIFTY_TECH_FEATURE_COLS = [
    "ret_1d","ret_5d","ret_20d","log_r1",
    "c_norm","v_norm","hl_rng","oc_rng",
    "ema5","ema10","ema20","ema50",
    "er5","er10","er20","er50",
    "macd","macd_sig","macd_dif","adx",
    "rsi14","rsi28","stoch_k","stoch_d",
    "bb_hi","bb_lo","bb_wid","bb_pct",
    "atr14","vol20","vol60","obv","vol_ratio",
    "skew20","kurt20",
]

QUANTILES_T = torch.tensor(CFG["QUANTILES"], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class NiftyGiftDataset(Dataset):
    """
    Dataset for gap-prediction training.

    Returns per-sample:
        seq_feat   : (SEQ_LEN, n_tech_features) – normalised NIFTY features
        gift_feat  : (N_GIFT_FEATURES,) – overnight GIFT features
        data_qual  : int  – data quality flag
        gap        : float – true opening gap = open_t - close_{t-1}
        prev_close : float – for reconstructing predicted open
    """

    def __init__(
        self,
        merged_df: pd.DataFrame,
        feature_cols: list,
        seq_len: int = 60,
        use_gift: bool = True,
    ):
        self.seq_len = seq_len
        self.use_gift = use_gift
        self.feature_cols = feature_cols

        # Require: open, prev_close, gap columns + gift features
        df = merged_df.copy()
        df = df.dropna(subset=["nifty_open", "prev_close"] + GIFT_FEATURE_COLS)

        # ── Standardise technical features ──
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        available_cols = [c for c in feature_cols if c in df.columns]
        tech_data = df[available_cols].fillna(0.0).values.astype(np.float32)
        tech_data = scaler.fit_transform(tech_data).clip(-5, 5)
        self.scaler = scaler
        self.tech_arr = tech_data          # (N, n_features)
        self.n_features = tech_data.shape[1]

        # ── Gap target ──
        self.gaps = (df["nifty_open"] - df["prev_close"]).values.astype(np.float32)
        self.prev_closes = df["prev_close"].values.astype(np.float32)

        # ── GIFT features ──
        gift_data = df[GIFT_FEATURE_COLS].fillna(0.0).values.astype(np.float32)
        # Standardise gift features (except data_quality which stays as int)
        gift_numeric = gift_data[:, :-1]
        gift_qual    = gift_data[:, -1:]
        from sklearn.preprocessing import StandardScaler as SS
        self.gift_scaler   = SS()
        gift_numeric_scaled = self.gift_scaler.fit_transform(gift_numeric).clip(-5, 5)
        self.gift_arr = np.concatenate([gift_numeric_scaled, gift_qual], axis=1)

        # ── Valid indices (need seq_len history) ──
        self.valid_idx = list(range(seq_len, len(tech_data)))

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        idx = self.valid_idx[i]

        seq = torch.tensor(
            self.tech_arr[idx - self.seq_len: idx],
            dtype=torch.float32
        )  # (seq_len, n_features)

        gap = torch.tensor(self.gaps[idx], dtype=torch.float32)
        pc  = torch.tensor(self.prev_closes[idx], dtype=torch.float32)

        if self.use_gift:
            gift = torch.tensor(self.gift_arr[idx], dtype=torch.float32)
            qual = torch.tensor(int(self.gift_arr[idx, -1]), dtype=torch.long)
        else:
            gift = torch.zeros(N_GIFT_FEATURES, dtype=torch.float32)
            qual = torch.tensor(-1, dtype=torch.long)

        return {
            "seq"       : seq,
            "gift"      : gift,
            "quality"   : qual,
            "gap"       : gap,
            "prev_close": pc,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Loss Functions (Step 4: target reformulation)
# ──────────────────────────────────────────────────────────────────────────────

def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor,
                 quantiles: torch.Tensor) -> torch.Tensor:
    """
    Asymmetric Pinball / Quantile loss.

    pred_q  : (batch, n_quantiles) – raw quantile predictions
    target  : (batch,) – true gap values
    """
    target_exp = target.unsqueeze(1).expand_as(pred_q)  # (batch, Q)
    q_exp      = quantiles.unsqueeze(0).to(pred_q.device)

    errors = target_exp - pred_q
    loss   = torch.max(q_exp * errors, (q_exp - 1.0) * errors)
    return loss.mean()


def monotone_quantile_penalty(pred_q: torch.Tensor) -> torch.Tensor:
    """
    Penalises quantile crossing: ensures q₁ ≤ q₂ ≤ … ≤ qₙ.
    This is critical for valid uncertainty bands.
    """
    penalty = torch.clamp(pred_q[:, :-1] - pred_q[:, 1:], min=0.0)
    return penalty.mean()


def combined_gap_loss(
    gap_pt: torch.Tensor,
    gap_q : torch.Tensor,
    true_gap: torch.Tensor,
    lam_mse: float = 1.0,
    lam_pin: float = 0.5,
) -> torch.Tensor:
    """
    Total gap loss:
        lam_mse * MSE(point_pred, true_gap)
      + lam_pin * Pinball(quantiles, true_gap)
      + 0.01   * MonotonePenalty(quantiles)
    """
    mse  = nn.functional.mse_loss(gap_pt, true_gap)
    pin  = pinball_loss(gap_q, true_gap, QUANTILES_T)
    mono = monotone_quantile_penalty(gap_q)
    return lam_mse * mse + lam_pin * pin + 0.01 * mono


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ACMIPlusPlus,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    cfg: dict,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        seq     = batch["seq"]           # (B, seq, features)
        gift    = batch["gift"]          # (B, N_GIFT)
        quality = batch["quality"]       # (B,)
        true_gap = batch["gap"]          # (B,)

        optimizer.zero_grad()

        out = model(
            seq,
            gift_features=gift,
            data_quality=quality,
            ret_regime=True,
        )

        # -- Gap head loss (primary target reformulation) --
        gap_loss = combined_gap_loss(
            out["gap_pt"], out["gap_q"], true_gap,
            lam_mse=cfg["LAMBDA_MSE"],
            lam_pin=cfg["LAMBDA_PIN"],
        )

        # -- Multi-horizon return heads (kept as auxiliary task) --
        # Proxy: 1d return ~ gap / prev_close (normalised)
        # The full dataset would have actual horizon returns.
        # Here we use gap direction as weak supervision.
        gap_sign = (true_gap > 0).float()
        aux_loss = torch.tensor(0.0)
        for h_key in [f"h{h}_pt" for h in cfg["HORIZONS"]]:
            aux_loss = aux_loss + nn.functional.binary_cross_entropy_with_logits(
                out[h_key], gap_sign
            )
        aux_loss = aux_loss / len(cfg["HORIZONS"])

        loss = gap_loss + cfg["LAMBDA_MULTI"] * aux_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(seq)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(
    model: ACMIPlusPlus,
    loader: DataLoader,
    cfg: dict,
) -> Dict[str, float]:
    model.eval()
    all_preds, all_true, all_prev = [], [], []

    for batch in loader:
        seq     = batch["seq"]
        gift    = batch["gift"]
        quality = batch["quality"]
        true_gap = batch["gap"]
        prev_close = batch["prev_close"]

        out = model(seq, gift_features=gift, data_quality=quality)

        all_preds.append(out["gap_pt"].cpu().numpy())
        all_true.append(true_gap.cpu().numpy())
        all_prev.append(prev_close.cpu().numpy())

    preds  = np.concatenate(all_preds)
    trues  = np.concatenate(all_true)
    prevs  = np.concatenate(all_prev)

    pred_opens   = prevs + preds
    actual_opens = prevs + trues

    mae      = float(np.abs(pred_opens - actual_opens).mean())
    rmse     = float(np.sqrt(((pred_opens - actual_opens) ** 2).mean()))
    gap_mae  = float(np.abs(preds - trues).mean())

    pred_dir   = np.sign(preds)
    actual_dir = np.sign(trues)
    dir_acc    = float((pred_dir == actual_dir).mean())

    return {
        "mae_pts"    : mae,
        "rmse_pts"   : rmse,
        "gap_mae"    : gap_mae,
        "dir_acc"    : dir_acc,
    }


def train_model(
    model: ACMIPlusPlus,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    save_path: Path,
    label: str = "model",
) -> Dict[str, float]:
    """
    Full training loop with:
    - cosine annealing LR schedule
    - early stopping on val MAE
    - gradient clipping
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["LR"],
        weight_decay=cfg["WEIGHT_DECAY"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["EPOCHS"], eta_min=1e-6
    )

    best_val_mae = float("inf")
    patience_ctr = 0
    best_state   = None

    for epoch in range(1, cfg["EPOCHS"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg)
        val_metrics = evaluate_epoch(model, val_loader, cfg)
        scheduler.step()

        val_mae = val_metrics["mae_pts"]
        logger.info(
            "[%s] Epoch %02d/%02d  train_loss=%.4f  val_mae=%.2f  "
            "dir_acc=%.1f%%  lr=%.2e",
            label, epoch, cfg["EPOCHS"], train_loss, val_mae,
            val_metrics["dir_acc"] * 100,
            optimizer.param_groups[0]["lr"],
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["PATIENCE"]:
                logger.info("[%s] Early stopping at epoch %d.", label, epoch)
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    torch.save({"model_state": model.state_dict(), "cfg": CFG}, save_path)
    logger.info("[%s] Saved best model to %s  (val_mae=%.2f)", label, save_path, best_val_mae)

    return val_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Ablation runner
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation():
    """
    Entry point that:
    1. Fetches & builds training data
    2. Evaluates baseline
    3. Trains Model A (no GIFT)
    4. Trains Model B (with GIFT)
    5. Prints comparison table
    """
    logger.info("=" * 60)
    logger.info("GIFT NIFTY Ablation Study")
    logger.info("=" * 60)

    # ── Step 1: Data ──
    logger.info("Fetching training data (NIFTY + GIFT)...")
    full_df = build_full_training_dataset(period="2y")
    if full_df.empty:
        logger.error("Dataset is empty. Check data pipeline.")
        return

    logger.info("Dataset: %d rows, %d cols", len(full_df), len(full_df.columns))

    # ── Step 2: Baseline ──
    logger.info("\n--- Step 2: Baseline Evaluation ---")
    baseline_result = evaluate_baseline(full_df)
    print(baseline_result.summary())

    # ── Train/Val/Test split ──
    n = len(full_df)
    n_train = int(n * TRAIN_CFG["TRAIN_SPLIT"])
    n_val   = int(n * TRAIN_CFG["VAL_SPLIT"])
    train_df = full_df.iloc[:n_train]
    val_df   = full_df.iloc[n_train: n_train + n_val]
    test_df  = full_df.iloc[n_train + n_val:]
    logger.info("Split → train:%d  val:%d  test:%d", len(train_df), len(val_df), len(test_df))

    # Feature columns to use as temporal sequence
    feature_cols = [c for c in NIFTY_TECH_FEATURE_COLS if c in full_df.columns]
    n_features = len(feature_cols)
    logger.info("Using %d technical feature columns.", n_features)

    # ── Model A: WITHOUT GIFT (ablation) ──
    logger.info("\n--- Step 5a: Training Model A (NO GIFT) ---")
    ds_train_a = NiftyGiftDataset(train_df, feature_cols, CFG["SEQ_LEN"], use_gift=False)
    ds_val_a   = NiftyGiftDataset(val_df,   feature_cols, CFG["SEQ_LEN"], use_gift=False)
    ds_test_a  = NiftyGiftDataset(test_df,  feature_cols, CFG["SEQ_LEN"], use_gift=False)

    dl_train_a = DataLoader(ds_train_a, batch_size=TRAIN_CFG["BATCH_SIZE"], shuffle=True, drop_last=True)
    dl_val_a   = DataLoader(ds_val_a,   batch_size=TRAIN_CFG["BATCH_SIZE"], shuffle=False)
    dl_test_a  = DataLoader(ds_test_a,  batch_size=TRAIN_CFG["BATCH_SIZE"], shuffle=False)

    cfg_a = {**CFG, "N_GIFT_FEATURES": N_GIFT_FEATURES}
    model_a = ACMIPlusPlus(n_features=n_features, cfg=cfg_a)
    train_model(model_a, dl_train_a, dl_val_a, TRAIN_CFG,
                save_path=MODELS_DIR / "acmi_no_gift.pt", label="NO-GIFT")
    test_metrics_a = evaluate_epoch(model_a, dl_test_a, TRAIN_CFG)

    # ── Model B: WITH GIFT ──
    logger.info("\n--- Step 5b: Training Model B (WITH GIFT) ---")
    ds_train_b = NiftyGiftDataset(train_df, feature_cols, CFG["SEQ_LEN"], use_gift=True)
    ds_val_b   = NiftyGiftDataset(val_df,   feature_cols, CFG["SEQ_LEN"], use_gift=True)
    ds_test_b  = NiftyGiftDataset(test_df,  feature_cols, CFG["SEQ_LEN"], use_gift=True)

    dl_train_b = DataLoader(ds_train_b, batch_size=TRAIN_CFG["BATCH_SIZE"], shuffle=True, drop_last=True)
    dl_val_b   = DataLoader(ds_val_b,   batch_size=TRAIN_CFG["BATCH_SIZE"], shuffle=False)
    dl_test_b  = DataLoader(ds_test_b,  batch_size=TRAIN_CFG["BATCH_SIZE"], shuffle=False)

    model_b = ACMIPlusPlus(n_features=n_features, cfg=cfg_a)
    train_model(model_b, dl_train_b, dl_val_b, TRAIN_CFG,
                save_path=MODELS_DIR / "acmi_with_gift.pt", label="WITH-GIFT")
    test_metrics_b = evaluate_epoch(model_b, dl_test_b, TRAIN_CFG)

    # ── Build comparison table ──
    from backend.app.services.baseline_model import BaselineEvalResult
    def _to_result(name, metrics, n):
        return BaselineEvalResult(
            model_name      = name,
            n_samples       = n,
            mae_points      = metrics["mae_pts"],
            rmse_points     = metrics["rmse_pts"],
            mape_pct        = 0.0,   # not computed in epoch eval
            direction_acc   = metrics["dir_acc"],
            r2              = 0.0,
            mae_gap_points  = metrics["gap_mae"],
            rmse_gap_points = 0.0,
        )

    n_test = len(ds_test_a)
    model_results = {
        "ACMI++ (no GIFT)" : _to_result("ACMI++ (no GIFT)", test_metrics_a, n_test),
        "ACMI++ (GIFT)"    : _to_result("ACMI++ (GIFT)",    test_metrics_b, n_test),
    }

    print("\n" + "=" * 60)
    print("ABLATION COMPARISON TABLE (TEST SET)")
    print("=" * 60)
    comp_df = compare_models(baseline_result, model_results)
    print(comp_df.to_string())

    print("\nKEY INSIGHT TABLE")
    print("-" * 60)
    for name, m in [
        ("Baseline (GIFT carry)", baseline_result),
        ("ACMI++ no GIFT",         _to_result("ACMI++ (no GIFT)", test_metrics_a, n_test)),
        ("ACMI++ with GIFT",       _to_result("ACMI++ (GIFT)",    test_metrics_b, n_test)),
    ]:
        improvement = ((baseline_result.mae_points - m.mae_points)
                       / (baseline_result.mae_points + 1e-8) * 100)
        print(f"  {name:<28}  MAE={m.mae_points:.1f}pts  DirAcc={m.direction_acc*100:.1f}%  "
              f"Δbaseline={improvement:+.1f}%")

    # Save results
    results_path = MODELS_DIR / "ablation_results.json"
    results_dict = {
        "baseline"    : baseline_result.to_dict(),
        "no_gift"     : _to_result("ACMI++ (no GIFT)", test_metrics_a, n_test).to_dict(),
        "with_gift"   : _to_result("ACMI++ (GIFT)",    test_metrics_b, n_test).to_dict(),
        "run_date"    : str(date.today()),
    }
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info("Ablation results saved to %s", results_path)


if __name__ == "__main__":
    run_ablation()

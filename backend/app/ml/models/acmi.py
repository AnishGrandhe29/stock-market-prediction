import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import joblib
from pathlib import Path
from collections import defaultdict
import ta
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# GIFT NIFTY overnight encoder & pipeline
try:
    from app.ml.models.overnight_encoder import OvernightEncoder
    from app.services.gift_nifty_pipeline import (
        get_latest_gift_features, N_GIFT_FEATURES, GIFT_FEATURE_COLS
    )
    _GIFT_OK = True
except ImportError:
    _GIFT_OK = False
    OvernightEncoder = None

logger = logging.getLogger(__name__)

# ----- PyTorch Geometric (GNN) Fallback -------------------------
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data as GraphData
    GNN_OK = True
except ImportError:
    GNN_OK = False

# ================================================================
# ACMI++ MODEL ARCHITECTURE
# ================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_c, out_c, ks, dil):
        super().__init__()
        pad = (ks - 1) * dil
        self.conv = nn.Conv1d(in_c, out_c, ks, padding=pad, dilation=dil)
        self.crop = pad
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.crop] if self.crop > 0 else out

class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, dil=1, drop=0.1):
        super().__init__()
        self.c1   = CausalConv1d(in_c, out_c, ks, dil)
        self.c2   = CausalConv1d(out_c, out_c, ks, dil)
        self.n1   = nn.LayerNorm(out_c)
        self.n2   = nn.LayerNorm(out_c)
        self.drop = nn.Dropout(drop)
        self.act  = nn.GELU()
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x):
        h = self.act(self.n1(self.c1(x).permute(0,2,1)).permute(0,2,1))
        h = self.drop(self.act(self.n2(self.c2(h).permute(0,2,1)).permute(0,2,1)))
        return h + self.skip(x)

class TCNEncoder(nn.Module):
    def __init__(self, in_f, d, n_layers=6, drop=0.1):
        super().__init__()
        self.proj   = nn.Linear(in_f, d)
        self.blocks = nn.ModuleList(
            [TCNBlock(d, d, ks=3, dil=2**i, drop=drop) for i in range(n_layers)]
        )
        self.norm   = nn.LayerNorm(d)
    def forward(self, x):
        h = self.proj(x).permute(0,2,1)
        for b in self.blocks:
            h = b(h)
        return self.norm(h.permute(0,2,1))

class TFEncoder(nn.Module):
    def __init__(self, d, heads=4, layers=2, drop=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d*4,
            dropout=drop, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.enc  = nn.TransformerEncoder(layer, num_layers=layers)
        self.pos  = nn.Embedding(512, d)
    def forward(self, x):
        T   = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return self.enc(x + self.pos(pos))

class GNNEncoder(nn.Module):
    def __init__(self, in_f, d, use_gnn=True):
        super().__init__()
        self.use_gnn = use_gnn and GNN_OK
        if self.use_gnn:
            self.g1 = GATConv(in_f, d//2, heads=2, dropout=0.1, concat=True)
            self.g2 = GATConv(d, d, heads=1, dropout=0.1, concat=False)
            self.n1 = nn.LayerNorm(d)
            self.n2 = nn.LayerNorm(d)
        else:
            self.fb = nn.Sequential(
                nn.Linear(in_f, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)
            )
    def forward(self, x, edge_index):
        if self.use_gnn:
            h = F.gelu(self.n1(self.g1(x, edge_index)))
            return self.n2(self.g2(h, edge_index))
        return self.fb(x)

class RegimeEngine(nn.Module):
    def __init__(self, d, n_reg=5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d, d//2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d//2, n_reg)
        )
    def forward(self, tf_out):
        logits = self.head(tf_out[:, -1, :])
        return logits, F.softmax(logits, dim=-1)

class CrossModalFusion(nn.Module):
    """
    Cross-modal attention fusion supporting N modalities.
    When n_mod=4 (Temporal, Technical, Graph, Overnight) the Overnight
    modality attends to all others, letting the model learn how strongly
    the GIFT signal should override the historical TCN context.
    """
    def __init__(self, d, heads=4, n_mod=4, drop=0.1):
        super().__init__()
        self.n_mod = n_mod
        self.attn  = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm  = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(drop), nn.Linear(d*2, d)
        )
        self.reg_proj = nn.Linear(5, n_mod)

    def forward(self, mods, reg_probs):
        """
        mods      : list of (batch, d) tensors – one per modality.
        reg_probs : (batch, 5) softmax regime probabilities.
        """
        tokens = torch.stack(mods, dim=1)            # (B, n_mod, d)
        bias   = self.reg_proj(reg_probs).unsqueeze(-1)  # (B, n_mod, 1)
        tokens = tokens + bias
        out, attn_weights = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + out)
        tokens = tokens + self.ffn(tokens)
        return tokens.mean(dim=1), attn_weights   # (B, d), (B, n_mod, n_mod)

class ReturnHead(nn.Module):
    def __init__(self, d, n_q=5, drop=0.1):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(d, d//2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d//2, 1 + n_q)
        )
    def forward(self, x):
        out = self.h(x)
        return out[:,0], out[:,1:]

class VolHead(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(d, d//4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d//4, 1), nn.Softplus()
        )
    def forward(self, x): return self.h(x).squeeze(-1)

class CrashHead(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(d, d//4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d//4, 1)
        )
    def forward(self, x): return self.h(x).squeeze(-1)

class ACMIPlusPlus(nn.Module):
    """
    Adaptive Cross-Modal Integration Plus Plus (ACMI++)

    Modalities fused (in order):
      0 – Temporal  : TCN → Transformer (price history)
      1 – Technical : MLP on last-day technical snapshot
      2 – Graph     : GNN / ticker embedding (relational)
      3 – Overnight : OvernightEncoder on GIFT NIFTY features  ← NEW

    The Overnight modality is injected AFTER the temporal encoder and
    BEFORE cross-modal fusion so that the attention mechanism can learn
    how much weight to assign to the GIFT signal per regime.

    When gift_features is None the overnight embedding falls back to a
    zero vector so existing checkpoints remain usable without retraining.
    """

    def __init__(self, n_features, cfg):
        super().__init__()
        D = cfg["D_MODEL"]
        H = cfg["N_HEADS"]
        Q = len(cfg["QUANTILES"])

        # ── Existing encoders (unchanged) ──
        self.tcn  = TCNEncoder(n_features, D, n_layers=cfg["TCN_LAYERS"])
        self.tf   = TFEncoder(D, heads=H, layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.tech = nn.Sequential(
            nn.Linear(n_features, D), nn.GELU(), nn.LayerNorm(D),
            nn.Dropout(0.1), nn.Linear(D, D)
        )

        self.gnn      = GNNEncoder(4, D, use_gnn=GNN_OK)
        self.tick_emb = nn.Embedding(50, D)
        self.gnn_proj = nn.Linear(D, D)

        self.regime_eng = RegimeEngine(D, cfg["N_REGIMES"])

        # ── NEW: Overnight encoder (GIFT NIFTY) ──
        n_gift = cfg.get("N_GIFT_FEATURES", 7)   # default 7 features
        if _GIFT_OK and OvernightEncoder is not None:
            self.overnight_enc = OvernightEncoder(n_features=n_gift, d=D,
                                                   dropout=0.1)
        else:
            # Fallback: simple zero-pad projection
            self.overnight_enc = nn.Sequential(
                nn.Linear(n_gift, D), nn.GELU(), nn.LayerNorm(D)
            )
        self.overnight_null = nn.Parameter(torch.zeros(1, D))  # learned null token

        # ── Fusion: 4 modalities now ──
        self.fusion = CrossModalFusion(D, heads=H, n_mod=4)
        self.res_w  = nn.Parameter(torch.ones(1) * 0.1)

        # ── Output heads (unchanged) ──
        self.ret_heads = nn.ModuleDict({
            f"h{h}": ReturnHead(D, n_q=Q) for h in cfg["HORIZONS"]
        })
        self.vol_head   = VolHead(D)
        self.crash_head = CrashHead(D)

        # ── NEW: Dedicated gap head for opening prediction ──
        # Outputs: (point_gap, [quantile_gaps]) using gap = open - prev_close
        self.gap_head = ReturnHead(D, n_q=Q)

    def forward(
        self,
        x,
        ticker_ids=None,
        graph_data=None,
        gift_features=None,     # (B, N_GIFT_FEATURES) or None
        data_quality=None,      # (B,) int {-1,0,1} or None
        ret_regime=False,
    ):
        """
        Parameters
        ----------
        x             : (B, SEQ_LEN, n_features) – normalised feature matrix
        ticker_ids    : (B,) long – ticker index for GNN embedding
        graph_data    : torch_geometric.data.Data or None
        gift_features : (B, N_GIFT_FEATURES) – overnight GIFT features; None
                        means no GIFT signal available → overnight embedding
                        is zeroed-out (graceful degradation).
        data_quality  : (B,) int tensor; used by OvernightEncoder quality gate.
        ret_regime    : bool – include regime logits/probs in output dict.

        Returns
        -------
        dict with all existing keys plus:
            'gap_pt'   : (B,) predicted opening gap (open - prev_close)
            'gap_q'    : (B, Q) quantiles of the gap distribution
            'overnight_attn' : cross-modal attention weights (B, 4, 4)
        """
        B, T, F = x.shape

        # ── Temporal pathway (unchanged) ──
        tcn_out  = self.tcn(x)
        tf_out   = self.tf(tcn_out)
        temporal = self.pool(tf_out.permute(0, 2, 1)).squeeze(-1)  # (B, D)

        # ── Technical snapshot (unchanged) ──
        tech = self.tech(x[:, -1, :])   # (B, D)

        # ── Graph / ticker pathway (unchanged) ──
        if graph_data is not None and GNN_OK:
            gnn_nodes = self.gnn(graph_data.x.to(x.device),
                                 graph_data.edge_index.to(x.device))
            if ticker_ids is not None:
                t_ids   = ticker_ids.clamp(0, gnn_nodes.size(0) - 1)
                graph_f = gnn_nodes[t_ids]
            else:
                graph_f = gnn_nodes.mean(0).unsqueeze(0).expand(B, -1)
        else:
            t_ids   = ticker_ids.clamp(0, self.tick_emb.num_embeddings - 1) \
                      if ticker_ids is not None \
                      else torch.zeros(B, dtype=torch.long, device=x.device)
            graph_f = self.tick_emb(t_ids)
        graph_f = self.gnn_proj(graph_f)   # (B, D)

        # ── Regime classification (unchanged) ──
        reg_logits, reg_probs = self.regime_eng(tf_out)

        # ── OVERNIGHT ENCODER (NEW) ──
        # Injected AFTER temporal encoding, BEFORE fusion.
        if gift_features is not None:
            gift_t = gift_features.to(x.device)
            if isinstance(self.overnight_enc, OvernightEncoder if _GIFT_OK else type(None)):
                overnight = self.overnight_enc(gift_t, data_quality)
            else:
                overnight = self.overnight_enc(gift_t)   # fallback Sequential
        else:
            # No GIFT data: use the learned null token (batch-expanded)
            overnight = self.overnight_null.expand(B, -1)   # (B, D)

        # ── 4-modality fusion ──
        fused, attn_weights = self.fusion(
            [temporal, tech, graph_f, overnight], reg_probs
        )                                                 # (B, D)
        fused = fused + self.res_w * temporal            # residual shortcut

        # ── Output heads ──
        outputs = {}
        for h_key, head in self.ret_heads.items():
            pt, q = head(fused)
            outputs[f"{h_key}_pt"] = pt
            outputs[f"{h_key}_q"]  = q

        # Gap head: predicts open - prev_close
        gap_pt, gap_q = self.gap_head(fused)
        outputs["gap_pt"]          = gap_pt
        outputs["gap_q"]           = gap_q

        outputs["vol"]             = self.vol_head(fused)
        outputs["crash"]           = self.crash_head(fused)
        outputs["overnight_attn"]  = attn_weights   # for XAI

        if ret_regime:
            outputs["reg_logits"] = reg_logits
            outputs["reg_probs"]  = reg_probs

        return outputs

# ================================================================
# PREDICTOR WRAPPER AND DATA PIPELINE
# ================================================================

class DataPipelineFallback:
    def __init__(self, cfg):
        self.cfg = cfg

    def macro_features(self, start_date, end_date):
        # Fallback pseudo macro features if not available to download during inference
        # Real pipeline downloading macro during fast API inference might be too slow.
        # But we will use yf to try. Let's wrap yf inside a try-except
        raw_macro = {}
        try:
            raw = yf.download(self.cfg["MACRO_TICKERS"], start=start_date, end=end_date, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                for t in self.cfg["MACRO_TICKERS"]:
                    try:
                        df = raw.xs(t, axis=1, level=1)[["Close"]].copy()
                        df.columns = ["close"]
                        df.dropna(inplace=True)
                        if len(df) >= 10:
                            raw_macro[t] = df
                    except Exception:
                        pass
        except Exception:
            pass

        frames = []
        for t, df in raw_macro.items():
            name = "mac_" + t.replace("^","").replace("-","_").replace(".","_")
            s = df["close"].pct_change(1).rename(name)
            frames.append(s)
            
        if not frames:
            # Create a dummy DataFrame if no macro downloaded
            return pd.DataFrame()
            
        out = pd.concat(frames, axis=1)
        out.ffill(inplace=True)
        out.bfill(inplace=True)
        return out

    def technical(self, df):
        c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
        f = pd.DataFrame(index=df.index)

        f["ret_1d"]   = c.pct_change(1)
        f["ret_5d"]   = c.pct_change(5)
        f["ret_20d"]  = c.pct_change(20)
        f["log_r1"]   = np.log(c / c.shift(1))

        f["c_norm"]   = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)
        f["v_norm"]   = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-8)
        f["hl_rng"]   = (h - lo) / (c + 1e-8)
        f["oc_rng"]   = (c - df["open"]) / (c + 1e-8)

        for w in [5, 10, 20, 50]:
            ema = EMAIndicator(c, window=w).ema_indicator()
            f[f"ema{w}"]  = ema
            f[f"er{w}"]   = c / (ema + 1e-8) - 1

        m = MACD(c)
        f["macd"]     = m.macd()
        f["macd_sig"] = m.macd_signal()
        f["macd_dif"] = m.macd_diff()
        f["adx"]      = ADXIndicator(h, lo, c).adx()

        f["rsi14"]    = RSIIndicator(c, window=14).rsi()
        f["rsi28"]    = RSIIndicator(c, window=28).rsi()
        st = StochasticOscillator(h, lo, c)
        f["stoch_k"]  = st.stoch()
        f["stoch_d"]  = st.stoch_signal()

        bb = BollingerBands(c, window=20)
        f["bb_hi"]    = bb.bollinger_hband()
        f["bb_lo"]    = bb.bollinger_lband()
        f["bb_wid"]   = bb.bollinger_wband()
        f["bb_pct"]   = bb.bollinger_pband()
        f["atr14"]    = AverageTrueRange(h, lo, c).average_true_range()
        f["vol20"]    = f["log_r1"].rolling(20).std() * np.sqrt(252)
        f["vol60"]    = f["log_r1"].rolling(60).std() * np.sqrt(252)

        f["obv"]       = OnBalanceVolumeIndicator(c, v).on_balance_volume()
        f["vol_ratio"] = v / (v.rolling(20).mean() + 1e-8)

        f["skew20"]  = f["log_r1"].rolling(20).skew()
        f["kurt20"]  = f["log_r1"].rolling(20).kurt()

        return f

class ACMIPredictor:
    REGIME_NAMES = ["Bull Trending","Bear Trending","High-Volatility",
                    "Sideways","Unknown"]

    def __init__(self, model_path, scaler_path, ensemble_path=None, device="cpu"):
        self.device = torch.device(device)
        sc = joblib.load(scaler_path)
        self.feat_scaler  = sc["feat"]
        self.feature_cols = sc["feature_cols"]
        self.n_features = len(self.feature_cols)

        # Restore CFG structure from the training notebook
        self.cfg = {
            "TICKERS"        : ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","JPM","V","UNH"],
            "MACRO_TICKERS"  : ["^VIX","^TNX","DX-Y.NYB","GLD","TLT","HYG","USO","SPY"],
            "SEQ_LEN"        : 60,
            "D_MODEL"        : 128,
            "N_HEADS"        : 4,
            "TCN_LAYERS"     : 6,
            "N_REGIMES"      : 5,
            "HORIZONS"       : [1, 5, 20, 60],
            "QUANTILES"      : [0.1, 0.25, 0.5, 0.75, 0.9],
            "N_GIFT_FEATURES": 7,   # gap_abs, gap_pct, overnight_vol,
                                    # gift_trend_slope, gift_momentum_5d,
                                    # gap_z_score, data_quality
        }

        # Primary model
        ckpt  = torch.load(model_path, map_location=self.device)
        m = ACMIPlusPlus(n_features=self.n_features, cfg=self.cfg)
        m.load_state_dict(ckpt["model_state"], strict=False)
        m.to(self.device).eval()
        self.ensemble = [m]
        self.conf_alphas = {}

        # Ensemble
        if ensemble_path and Path(ensemble_path).exists():
            ec = torch.load(ensemble_path, map_location=self.device)
            self.conf_alphas = ec.get("conformal_alphas", {})
            for sd in ec["models"][1:]:  # skip first model if it's the same
                em = ACMIPlusPlus(n_features=self.n_features, cfg=self.cfg)
                em.load_state_dict(sd, strict=False)
                em.to(self.device).eval()
                self.ensemble.append(em)

    def _features(self, ticker, period="2y"):
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        
        # Determine actual valid columns (sometimes yfinance returns multi-index)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten or select the right one
            try:
                df = df.xs(ticker, axis=1, level=1).copy()
            except Exception:
                # If ticker not in level 1, try dropping level 1
                try:
                    df = df.droplevel(1, axis=1)
                except Exception:
                    pass
        
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns:
            raise ValueError(f"Could not find valid price columns for {ticker}")

        df.dropna(subset=["close"], inplace=True)

        pipe_tmp = DataPipelineFallback(self.cfg)
        tech = pipe_tmp.technical(df)
        
        # Macro data (optional but recommended in notebook)
        start_date = df.index[0].strftime("%Y-%m-%d")
        end_date = df.index[-1].strftime("%Y-%m-%d")
        macro_df = pipe_tmp.macro_features(start_date=start_date, end_date=end_date)
        
        df2 = pd.concat([df, tech], axis=1)
        if not macro_df.empty:
            df2 = df2.join(macro_df, how="left")
            for c in macro_df.columns:
                df2[c] = df2[c].ffill().bfill()
        
        df2.ffill(inplace=True); df2.fillna(0, inplace=True)
        
        # Ensure all columns exist
        for c in self.feature_cols:
            if c not in df2.columns:
                df2[c] = 0.0
                
        return df2[self.feature_cols].values, df["close"].iloc[-1]

    @torch.no_grad()
    def predict(self, ticker, use_gift: bool = True):
        """
        Generate a prediction for the given ticker.

        Parameters
        ----------
        ticker   : Yahoo Finance ticker symbol (e.g. '^NSEI').
        use_gift : If True (default), fetches GIFT NIFTY features and
                   injects them as the overnight modality.  Set False
                   to run the ablation 'without GIFT' variant.

        Returns
        -------
        dict – all existing keys plus:
            'gift_features'   : raw GIFT feature dict used (or None)
            'predicted_open'  : prev_close + predicted_gap
            'gap_pred'        : predicted opening gap (points)
            'gap_pred_pct'    : predicted opening gap (%)
        """
        feats, latest_price = self._features(ticker)

        if len(feats) < self.cfg["SEQ_LEN"]:
            raise ValueError(f"Not enough history for {ticker}")

        seq = np.clip(self.feat_scaler.transform(feats[-self.cfg["SEQ_LEN"]:]), -5, 5)
        x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        # ── Fetch & encode GIFT NIFTY features ──
        gift_tensor    = None
        quality_tensor = None
        gift_feat_dict = None

        if use_gift and _GIFT_OK:
            try:
                gift_feat_dict = get_latest_gift_features(prev_close=latest_price)
                n_gift = self.cfg["N_GIFT_FEATURES"]
                gift_vec = np.array([
                    gift_feat_dict["gap_abs"],
                    gift_feat_dict["gap_pct"],
                    gift_feat_dict["overnight_vol"],
                    gift_feat_dict["gift_trend_slope"],
                    gift_feat_dict["gift_momentum_5d"],
                    gift_feat_dict["gap_z_score"],
                    float(gift_feat_dict["data_quality"]),
                ], dtype=np.float32)
                gift_tensor = torch.tensor(gift_vec, dtype=torch.float32
                                           ).unsqueeze(0).to(self.device)
                quality_tensor = torch.tensor(
                    [gift_feat_dict["data_quality"]], dtype=torch.long
                ).to(self.device)
                logger.debug("GIFT features loaded: gap_pct=%.4f, quality=%d",
                             gift_feat_dict["gap_pct"],
                             gift_feat_dict["data_quality"])
            except Exception as exc:
                logger.warning("GIFT feature fetch failed (%s) – proceeding without.", exc)
                gift_feat_dict = None

        # ── Ensemble inference ──
        pool = defaultdict(list)
        for m in self.ensemble:
            out = m(
                x,
                gift_features=gift_tensor,
                data_quality=quality_tensor,
                ret_regime=True,
            )
            for h in self.cfg["HORIZONS"]:
                pool[h].append(out[f"h{h}_pt"].item())
            pool["gap"].append(out["gap_pt"].item())
            pool["vol"].append(out["vol"].item())
            pool["crash"].append(torch.sigmoid(out["crash"]).item())
            pool["reg"].append(out["reg_probs"].cpu().numpy())

        mean_reg = np.mean(pool["reg"], axis=0)[0]
        regime   = self.REGIME_NAMES[int(np.argmax(mean_reg))]

        # ── Gap prediction (primary output for open) ──
        mean_gap     = float(np.mean(pool["gap"]))
        std_gap      = float(np.std(pool["gap"]))
        predicted_open = latest_price + mean_gap
        gap_pct        = mean_gap / (latest_price + 1e-8)

        result = {
            "ticker"         : ticker,
            "latest_price"   : latest_price,
            "date"           : datetime.date.today().isoformat(),
            "regime"         : regime,
            "regime_p"       : {r: float(p) for r, p in zip(self.REGIME_NAMES, mean_reg)},
            "vol_fcast"      : float(np.mean(pool["vol"])),
            "crash_prob"     : float(np.mean(pool["crash"])),
            # ── NEW: Gap-based opening prediction ──
            "gap_pred"       : mean_gap,
            "gap_pred_pct"   : gap_pct,
            "gap_uncertainty": std_gap,
            "predicted_open" : predicted_open,
            "gift_features"  : gift_feat_dict,
            "horizons"       : {},
        }

        for h in self.cfg["HORIZONS"]:
            mu  = float(np.mean(pool[h]))
            sig = float(np.std(pool[h]))
            al  = self.conf_alphas.get(h, abs(mu) * 1.5 + 0.005)
            result["horizons"][f"{h}d"] = {
                "point"      : mu,
                "direction"  : "UP" if mu > 0 else "DOWN",
                "interval"   : [mu - al, mu + al],
                "uncertainty": sig,
            }

        return result

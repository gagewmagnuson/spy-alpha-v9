"""
SPY Alpha V9 — Step 5: Strategy 1 State-Responsive Allocator
strategy_state_allocator.py
 
Cross-sectional LightGBM trained on (state, asset) pairs.
 
Architecture (locked before implementation, per design review):
  Target:    Tail-aware reward per asset, 63-day forward window.
             Ported exactly from V8 meta_allocator.compute_tail_aware_reward.
             Turnover set to 0.0 (no portfolio-level turnover cost per asset).
  Features:  Shared state context + asset-specific momentum / vol.
  Universe:  Full TRADING_UNIVERSE (~33 assets: TIER1-4).
  Selection: Top 12 assets per refit by predicted reward score.
  Weights:   max(score, ε) × (1 / trailing_21d_vol); caps: TIER4 ≤ 20%,
             ETFs ≤ 30%; normalized to 1.0 via iterative redistribution.
  Analog:    Option A — coherence/reliability/hazard as features only.
             Concentration control reserved for Step 6 (Conviction Governor).
 
Walk-forward discipline:
  Training cutoff: position i − FORWARD_WINDOW (prevents target look-ahead).
  Minimum training: 504 (date × asset) pair dates.
  Refit frequency:  every 21 trading days.
 
Ablation models:
  A: pillars only         (5 state + 5 asset = 10 features)
  B: state_vector         (25 state + 5 asset = 30 features)
  C: state + analog       (28 state + 5 asset = 33 features)
 
Replaces V8's regime_model + return_forecaster + asset_selector
+ portfolio_optimizer with a single unified model.
"""
 
from __future__ import annotations
 
import logging
from typing import Any, Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
 
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
 
logger = logging.getLogger(__name__)
 
# ------------------------------------------------------------------ #
# Trading Universe & Tier Map                                          #
# ------------------------------------------------------------------ #
 
try:
    from data_pipeline import (
        TIER1_MACRO, TIER2_SECTORS, TIER3_THEMATIC, TIER4_STOCKS,
    )
    TRADING_UNIVERSE: List[str] = (
        TIER1_MACRO + TIER2_SECTORS + TIER3_THEMATIC + TIER4_STOCKS
    )
    ASSET_TIER_MAP: Dict[str, int] = {}
    for _a in TIER1_MACRO:   ASSET_TIER_MAP[_a] = 1
    for _a in TIER2_SECTORS: ASSET_TIER_MAP[_a] = 2
    for _a in TIER3_THEMATIC:ASSET_TIER_MAP[_a] = 3
    for _a in TIER4_STOCKS:  ASSET_TIER_MAP[_a] = 4
except ImportError:
    TRADING_UNIVERSE = []
    ASSET_TIER_MAP   = {}
 
# ------------------------------------------------------------------ #
# Feature Columns                                                       #
# ------------------------------------------------------------------ #
 
PILLAR_NAMES:     List[str] = [
    "growth_momentum", "inflation_pressure", "financial_stress",
    "trend_persistence", "participation_quality",
]
TRAJECTORY_TYPES: List[str] = ["velocity", "persistence", "divergence", "stability"]
ANALOG_NAMES:     List[str] = ["coherence", "reliability", "transition_hazard"]
 
STATE_COLS_A: List[str] = PILLAR_NAMES                                             # 5
STATE_COLS_B: List[str] = (                                                        # 25
    PILLAR_NAMES
    + [f"{p}_{t}" for p in PILLAR_NAMES for t in TRAJECTORY_TYPES]
)
STATE_COLS_C: List[str] = STATE_COLS_B + ANALOG_NAMES                              # 28
 
ASSET_FEATURE_COLS: List[str] = [
    "asset_ret_21d", "asset_ret_63d",
    "asset_vol_21d", "asset_vol_63d",
    "asset_tier",                          # normalized tier [0.25, 1.0]
]                                                                                  # 5
 
# ------------------------------------------------------------------ #
# Constants                                                             #
# ------------------------------------------------------------------ #
 
MIN_TRAIN:       int   = 504    # ~2 years minimum training rows (date-level)
REFIT_EVERY:     int   = 21     # monthly refit
FORWARD_WINDOW:  int   = 63     # reward horizon (matching V8 meta-allocator)
N_TOP:           int   = 12     # assets selected per refit
CAP_TIER4:       float = 0.20   # max weight, individual stocks
CAP_ETF:         float = 0.30   # max weight, ETFs / macro
MIN_WEIGHT_EPS:  float = 1e-4   # floor weight for selected assets
 
DEFAULT_LGB_PARAMS: Dict[str, Any] = {
    "objective":         "regression",
    "metric":            "mse",
    "n_estimators":      200,
    "learning_rate":     0.05,
    "max_depth":         4,
    "num_leaves":        15,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "verbose":          -1,
    "n_jobs":           -1,
}
 
# Reward weights — ported exactly from V8 meta_allocator.REWARD_WEIGHTS
REWARD_WEIGHTS: Dict[str, float] = {
    "differential_sharpe": 1.0,
    "sortino_component":   0.5,
    "drawdown_penalty":    2.0,
    "turnover_penalty":    0.3,
    "tail_risk_penalty":   1.0,
}
 
 
# ------------------------------------------------------------------ #
# Tail-Aware Reward Function  (ported exactly from V8 meta_allocator) #
# ------------------------------------------------------------------ #
 
def compute_tail_aware_reward(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover:          float = 0.0,
    window:            int   = FORWARD_WINDOW,
) -> float:
    """
    Tail-aware reward for a single asset held over a forward window.
 
    Ported exactly from V8 meta_allocator.compute_tail_aware_reward.
    Per-asset application: portfolio_returns = single asset daily returns.
    Turnover = 0.0 (no portfolio turnover cost for raw per-asset scoring).
 
    reward = 1.0 × differential_sharpe  (asset Sharpe − SPY Sharpe)
           + 0.5 × sortino_component    (downside-weighted return, capped 2.0)
           − 2.0 × drawdown_penalty     (max_DD above 10%, heavy penalty)
           − 0.3 × turnover_penalty     (set to 0.0 for per-asset use)
           − 1.0 × tail_risk_penalty    (excess kurtosis above 3.0)
 
    This is materially different from return/vol:
      • Heavy drawdown penalty distinguishes UPRO from SHY in bear markets.
      • Tail penalty penalizes fat-tailed assets (leveraged ETFs, volatile stocks).
      • Differential Sharpe vs SPY captures relative attractiveness.
    """
    if len(portfolio_returns) < 21:
        return np.nan
 
    port  = portfolio_returns.dropna()
    bench = benchmark_returns.reindex(port.index).dropna()
    if len(port) < 21:
        return np.nan
 
    # ---- Differential Sharpe ----
    port_sharpe  = port.mean()  / port.std()  * np.sqrt(252) if port.std()  > 0 else 0.0
    bench_sharpe = bench.mean() / bench.std() * np.sqrt(252) if bench.std() > 0 else 0.0
    diff_sharpe  = port_sharpe - bench_sharpe
 
    # ---- Sortino Component ----
    downside = port[port < 0]
    if len(downside) > 0 and downside.std() > 0:
        downside_vol = float(downside.std() * np.sqrt(252))
    elif port.std() > 0:
        downside_vol = float(port.std() * np.sqrt(252))  # fallback: total vol
    else:
        downside_vol = 0.01                               # final fallback: 1%/day
    sortino      = float(np.clip((port.mean() * 252) / downside_vol, -10.0, 10.0))
    sortino_comp = float(np.clip(sortino / 2.0, -2.0, 2.0))  # symmetric clip
 
    # ---- Drawdown Penalty ----
    cum     = (1.0 + port).cumprod()
    peak    = cum.expanding().max()
    max_dd  = float(abs((cum / peak - 1.0).min()))
    dd_pen  = max(max_dd - 0.10, 0.0)
 
    # ---- Turnover Penalty ----
    to_pen  = max(turnover - 0.20, 0.0)
 
    # ---- Tail Risk Penalty (excess kurtosis) ----
    tail_pen = 0.0
    if len(port) > 10:
        kurt = port.kurtosis()
        if pd.notna(kurt):
            tail_pen = float(np.clip(max(float(kurt) - 3.0, 0.0) / 10.0, 0.0, 2.0))
 
    w = REWARD_WEIGHTS
    reward = (
        w["differential_sharpe"] * diff_sharpe
        + w["sortino_component"]  * sortino_comp
        - w["drawdown_penalty"]   * dd_pen
        - w["turnover_penalty"]   * to_pen
        - w["tail_risk_penalty"]  * tail_pen
    )
    return float(reward)
 
 
# ------------------------------------------------------------------ #
# StateAllocator                                                        #
# ------------------------------------------------------------------ #
 
class StateAllocator:
    """
    Step 5: Strategy 1 — State-Responsive Allocator.
 
    Single cross-sectional LightGBM model replaces V8's
    regime_model + return_forecaster + asset_selector + portfolio_optimizer.
 
    Usage::
 
        from data_pipeline import SnapshotManager, get_adj_close
        snap      = SnapshotManager().load_snapshot('baseline_v7')
        adj_close = get_adj_close(snap)          # or snap.get('adj_close')
 
        allocator = StateAllocator()
        allocator.build(
            pillars       = engine.pillars,
            state_vector  = engine.state_vector,
            analog_scores = memory.analog_scores,
            raw_close     = engine._raw_close,
            adj_close     = adj_close,
        )
        allocator.validate()
    """
 
    def __init__(
        self,
        n_top: int = N_TOP,
        min_train: int = MIN_TRAIN,
        refit_every: int = REFIT_EVERY,
        forward_window: int = FORWARD_WINDOW,
        lgb_params: Optional[Dict[str, Any]] = None,
        target_type: str = "excess_return",
        weighting_type: str = "inverse_vol",
    ) -> None:
        if not LGB_AVAILABLE:
            raise ImportError(
                "lightgbm required. "
                "pip install lightgbm --break-system-packages"
            )
        self.n_top = n_top
        self.min_train = min_train
        self.refit_every = refit_every
        self.forward_window = forward_window
        self.lgb_params = lgb_params or DEFAULT_LGB_PARAMS.copy()
        self.target_type = target_type    # "tail_aware" | "raw_return" | "excess_return"
        self.weighting_type = weighting_type  # "inverse_vol" | "score_prop"
 
        # Populated by build()
        self.predictions:   Optional[Dict[str, pd.DataFrame]] = None
        self.daily_returns: Optional[pd.DataFrame]            = None
        self.metrics:       Optional[pd.DataFrame]            = None
        self._targets:      Optional[pd.DataFrame]            = None
        self._portfolios:   Optional[Dict[str, Dict]]         = None
 
    # ---------------------------------------------------------------- #
    # Public: build                                                     #
    # ---------------------------------------------------------------- #
 
    def build(
        self,
        pillars:       pd.DataFrame,
        state_vector:  pd.DataFrame,
        analog_scores: pd.DataFrame,
        raw_close:     pd.DataFrame,
        adj_close:     pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build and evaluate all three ablation models.
 
        Args:
            pillars       : 5-col DataFrame from StateEngine.pillars
            state_vector  : 25-col DataFrame (pillars + trajectory features)
            analog_scores : 3-col DataFrame (coherence, reliability, hazard)
            raw_close     : raw (unadjusted) prices — feature engineering
            adj_close     : adjusted close — reward targets + return simulation
 
        Returns:
            pd.DataFrame of performance metrics (models + benchmarks).
        """
        logger.info("=" * 66)
        logger.info("StateAllocator.build() — Step 5")
        logger.info("=" * 66)
 
        # Validate SPY presence
        if "SPY" not in adj_close.columns:
            raise ValueError("adj_close must contain 'SPY'.")
 
        # ---- 1. Merge state features ----
        logger.info("Step 1/6: Merging state feature frames...")
        all_state = pd.concat(
            [pillars, state_vector, analog_scores], axis=1
        ).loc[:, ~pd.concat([pillars, state_vector, analog_scores],
                             axis=1).columns.duplicated()]
 
        # ---- 2. Asset-specific features ----
        logger.info("Step 2/6: Computing per-asset features (momentum, vol)...")
        asset_feats = self._compute_asset_features(raw_close)
 
        # ---- 3. Target rewards ----
        logger.info("Step 3/6: Computing tail-aware reward targets "
                    f"({FORWARD_WINDOW}-day forward)...")
        spy_ret = adj_close["SPY"].pct_change().dropna()
        self._targets = self._compute_target_rewards(adj_close, spy_ret)
        n_valid = int(self._targets.notna().sum().sum())
        logger.info(f"  {n_valid:,} valid (date, asset) reward targets")
 
        # ---- 4. Walk-forward prediction per ablation model ----
        logger.info("Step 4/6: Walk-forward cross-sectional training...")
        model_defs: Dict[str, List[str]] = {
            "A": [c for c in STATE_COLS_A if c in all_state.columns],
            "B": [c for c in STATE_COLS_B if c in all_state.columns],
            "C": [c for c in STATE_COLS_C if c in all_state.columns],
        }
        self.predictions = {}
        for name, scols in model_defs.items():
            n_feat = len(scols) + len(ASSET_FEATURE_COLS)
            logger.info(f"  Model {name}: {len(scols)} state + "
                        f"{len(ASSET_FEATURE_COLS)} asset = {n_feat} features...")
            scores = self._walk_forward_predict(all_state, asset_feats, scols)
            self.predictions[name] = scores
            n = int(scores.notna().sum().sum()) if not scores.empty else 0
            n_dates = len(scores) if not scores.empty else 0
            logger.info(f"  Model {name}: {n:,} predictions across "
                        f"{n_dates} refit dates")
 
        # ---- 5. Portfolios & simulated returns ----
        logger.info("Step 5/6: Constructing portfolios and simulating returns...")
        self.daily_returns = pd.DataFrame(index=adj_close.index)
        self._portfolios = {}
 
        for name, scores in self.predictions.items():
            wbd = self._scores_to_weights(scores, raw_close)
            self._portfolios[name] = wbd
            col = f"model_{name}"
            self.daily_returns[col] = self._simulate_daily_returns(
                wbd, adj_close
            ).reindex(self.daily_returns.index)
 
        # Benchmarks
        self.daily_returns["spy_buyhold"] = adj_close["SPY"].pct_change()
        avail = [a for a in TRADING_UNIVERSE if a in adj_close.columns]
        if avail:
            self.daily_returns["equal_weight"] = (
                adj_close[avail].pct_change().mean(axis=1)
            )
 
        # ---- 6. Metrics ----
        logger.info("Step 6/6: Computing performance metrics...")
        model_cols = [c for c in self.daily_returns.columns
                      if c.startswith("model_")]
        has_pred = self.daily_returns[model_cols].notna().any(axis=1)
        start = has_pred[has_pred].index[0] if has_pred.any() else self.daily_returns.index[0]
        eval_df = self.daily_returns.loc[start:]
 
        metrics = {}
        for col in eval_df.columns:
            s = eval_df[col].dropna()
            if len(s) >= 126:
                metrics[col] = self._compute_metrics(s)
        self.metrics = pd.DataFrame(metrics).T
 
        self._log_summary()
        return self.metrics
 
    # ---------------------------------------------------------------- #
    # Private: target reward computation                                 #
    # ---------------------------------------------------------------- #
 
    def _compute_target_rewards(
        self,
        adj_close:   pd.DataFrame,
        spy_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute per-(date, asset) tail-aware reward targets.
 
        For date t and asset a:
            reward[a][t] = compute_tail_aware_reward(
                adj_close[a].pct_change()[t : t + FORWARD_WINDOW],
                spy_returns[t : t + FORWARD_WINDOW],
            )
 
        Returns DataFrame[dates × assets].
        """
        daily_ret = adj_close.pct_change()
        assets    = [a for a in TRADING_UNIVERSE if a in adj_close.columns]
        dates     = daily_ret.index.tolist()
        N         = len(dates)
 
        spy_vals  = spy_returns.reindex(daily_ret.index).values  # numpy array
 
        reward_dict: Dict[str, Dict] = {a: {} for a in assets}
 
        for i in range(N - self.forward_window):
            date   = dates[i]
            end_i  = i + self.forward_window
            s_win  = pd.Series(spy_vals[i:end_i], dtype=float)
 
            for asset in assets:
                if asset not in daily_ret.columns:
                    continue
                p_win = daily_ret[asset].iloc[i:end_i]
                r = self._compute_single_target(p_win, s_win)
                if r is not None and not np.isnan(r):
                    reward_dict[asset][date] = r
 
        return pd.DataFrame(reward_dict)
 
    # ---------------------------------------------------------------- #
    # Private: single-window target computation                         #
    # ---------------------------------------------------------------- #

    def _compute_single_target(
        self,
        asset_returns: pd.Series,
        spy_returns: pd.Series,
    ) -> Optional[float]:
        """
        Compute a single target value for one (date, asset) window.

        Dispatches on self.target_type:
            "tail_aware"    — original V8 meta-allocator reward (portfolio-level)
            "raw_return"    — annualized forward return (V8 return_forecaster style)
            "excess_return" — annualized forward excess return vs SPY
        """
        if self.target_type == "tail_aware":
            r = compute_tail_aware_reward(asset_returns, spy_returns, turnover=0.0)
            return float(r) if not np.isnan(r) else None

        port = asset_returns.dropna()
        if len(port) < 21:
            return None

        scale = 252.0 / len(port)

        if self.target_type == "raw_return":
            total_ret = float((1.0 + port).prod() - 1.0)
            return total_ret * scale

        if self.target_type == "excess_return":
            # spy_returns has integer index from numpy slice;
            # port has DatetimeIndex — reindex by label fails silently.
            # Use values directly and mask NaN by position.
            a_arr = asset_returns.values
            s_arr = spy_returns.values
            valid   = ~(np.isnan(a_arr) | np.isnan(s_arr))
            a_clean = a_arr[valid]
            s_clean = s_arr[valid]
            if len(a_clean) < 21:
                return None
            scale_adj = 252.0 / len(a_clean)
            asset_ret = float((1.0 + a_clean).prod() - 1.0)
            spy_ret   = float((1.0 + s_clean).prod() - 1.0)
            return (asset_ret - spy_ret) * scale_adj

        return None

    # ---------------------------------------------------------------- #
    # Private: asset features                                            #
    # ---------------------------------------------------------------- #
 
    def _compute_asset_features(
        self,
        raw_close: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Per-asset momentum and vol features.
 
        Per V8 rule: raw (unadjusted) prices for all feature engineering.
 
        Returns Dict[asset → DataFrame(ASSET_FEATURE_COLS, date-indexed)].
        """
        daily_ret = raw_close.pct_change()
        feats: Dict[str, pd.DataFrame] = {}
 
        for asset in TRADING_UNIVERSE:
            if asset not in raw_close.columns:
                continue
            price = raw_close[asset]
            ret   = daily_ret[asset]
            tier  = ASSET_TIER_MAP.get(asset, 1)
 
            feats[asset] = pd.DataFrame({
                "asset_ret_21d": price.pct_change(21),
                "asset_ret_63d": price.pct_change(63),
                "asset_vol_21d": ret.rolling(21, min_periods=10).std() * np.sqrt(252),
                "asset_vol_63d": ret.rolling(63, min_periods=30).std() * np.sqrt(252),
                "asset_tier":    tier / 4.0,   # normalize to [0.25, 1.0]
            }, index=raw_close.index)
 
        return feats
 
    # ---------------------------------------------------------------- #
    # Private: panel builder                                             #
    # ---------------------------------------------------------------- #
 
    def _build_panel(
        self,
        state_complete: pd.DataFrame,
        asset_feats:    Dict[str, pd.DataFrame],
        state_cols:     List[str],
        assets:         List[str],
    ) -> pd.DataFrame:
        """
        Build flat cross-sectional panel: one row per (date, asset).
 
        Columns: state_cols + ASSET_FEATURE_COLS + ['target', 'asset'].
        Sorted by date to enable efficient walk-forward slicing.
        """
        blocks: List[pd.DataFrame] = []
 
        for asset in assets:
            block = state_complete[state_cols].copy()
 
            # Asset features (aligned to state dates)
            af = asset_feats.get(asset)
            if af is not None:
                af_aligned = af.reindex(state_complete.index)
                for col in ASSET_FEATURE_COLS:
                    block[col] = af_aligned[col] if col in af_aligned.columns else np.nan
            else:
                for col in ASSET_FEATURE_COLS:
                    block[col] = np.nan
 
            # Ensure asset_tier is populated even if af is None
            if af is None:
                block["asset_tier"] = ASSET_TIER_MAP.get(asset, 1) / 4.0
 
            # Target
            if self._targets is not None and asset in self._targets.columns:
                block["target"] = self._targets[asset].reindex(state_complete.index)
            else:
                block["target"] = np.nan
 
            block["asset"] = asset
            blocks.append(block)
 
        panel = pd.concat(blocks, axis=0).sort_index()
        return panel
 
    # ---------------------------------------------------------------- #
    # Private: walk-forward cross-sectional prediction                   #
    # ---------------------------------------------------------------- #
 
    def _walk_forward_predict(
        self,
        all_state:   pd.DataFrame,
        asset_feats: Dict[str, pd.DataFrame],
        state_cols:  List[str],
    ) -> pd.DataFrame:
        """
        Walk-forward cross-sectional LightGBM prediction.
 
        At each refit position i (in the sorted unique-date array):
          Train:   all (date, asset) rows where date ≤ dates[i - forward_window]
          Predict: score all assets at dates[i]
 
        Walk-forward boundary:
          train_end = dates[i - forward_window]
          This prevents any target observation at date ≤ train_end from
          requiring future adj_close data at the time of training.
 
        Returns pd.DataFrame[dates × assets] of predicted reward scores.
        """
        # Align state to complete rows only
        state_ok = all_state[state_cols].dropna()
        dates     = state_ok.index          # unique sorted dates
        N         = len(dates)
 
        min_need = self.min_train + self.forward_window + self.refit_every
        if N < min_need:
            logger.warning(
                f"  Insufficient data: {N} dates, need {min_need}. "
                "Returning empty."
            )
            return pd.DataFrame()
 
        assets = [a for a in TRADING_UNIVERSE if a in asset_feats]
        logger.info(
            f"    Building panel: {N} dates × {len(assets)} assets "
            f"= {N * len(assets):,} rows..."
        )
        panel = self._build_panel(state_ok, asset_feats, state_cols, assets)
 
        all_feat_cols = state_cols + ASSET_FEATURE_COLS
 
        # Precompute which rows are fully valid (no NaN in features or target)
        feat_valid  = panel[all_feat_cols].notna().all(axis=1)
        valid_panel = panel[feat_valid & panel["target"].notna()]
        pred_panel  = panel[feat_valid]   # for prediction (no target needed)
 
        logger.info(
            f"    Panel: {len(panel):,} rows total, "
            f"{len(valid_panel):,} valid for training"
        )
 
        # Walk-forward loop
        score_records: List[Dict] = []
 
        first_refit = self.min_train + self.forward_window
        refit_positions = list(range(first_refit, N - 1, self.refit_every))
        if not refit_positions or refit_positions[-1] < N - 1:
            refit_positions.append(N - 1)
 
        logger.info(
            f"    {len(refit_positions)} refit dates "
            f"(first at position {first_refit}/{N})"
        )
 
        for k, refit_pos in enumerate(refit_positions):
            train_end_date = dates[refit_pos - self.forward_window]
            refit_date     = dates[refit_pos]
 
            # Training: all valid rows with date ≤ train_end_date
            train_mask = valid_panel.index <= train_end_date
            X_tr = valid_panel.loc[train_mask, all_feat_cols].values
            y_tr = valid_panel.loc[train_mask, "target"].values
 
            if len(y_tr) < self.min_train:
                continue
 
            # Train
            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(X_tr, y_tr)
 
            # Predict: all valid rows for the refit date
            pred_mask = pred_panel.index == refit_date
            pred_block = pred_panel.loc[pred_mask]
            if len(pred_block) == 0:
                continue
 
            scores = model.predict(pred_block[all_feat_cols].values)
            for asset, score in zip(pred_block["asset"].values, scores):
                score_records.append({
                    "date":  refit_date,
                    "asset": str(asset),
                    "score": float(score),
                })
 
            if (k + 1) % 20 == 0:
                logger.info(
                    f"    Refit {k+1}/{len(refit_positions)}: "
                    f"trained on {len(y_tr):,} pairs, "
                    f"scored {len(scores)} assets at {refit_date.date()}"
                )
 
        if not score_records:
            return pd.DataFrame()
 
        scores_df = pd.DataFrame(score_records)
        pivot = scores_df.pivot(index="date", columns="asset", values="score")
        pivot.columns.name = None
        return pivot
 
    # ---------------------------------------------------------------- #
    # Private: portfolio construction                                     #
    # ---------------------------------------------------------------- #
 
    def _scores_to_weights(
        self,
        scores_pivot: pd.DataFrame,
        raw_close:    pd.DataFrame,
    ) -> Dict[pd.Timestamp, Dict[str, float]]:
        """
        Convert per-asset scores to portfolio weights for each refit date.
 
        1. Select top N_TOP assets by predicted reward score.
        2. Raw weight = max(score, ε) × (1 / trailing_21d_vol).
        3. Apply position caps: TIER4 ≤ CAP_TIER4, ETFs ≤ CAP_ETF.
        4. Iterative redistribution (≤ 5 passes) to handle cap overflow.
        5. Normalize to sum 1.0.
        """
        if scores_pivot.empty:
            return {}
 
        daily_ret = raw_close.pct_change()
        weights_by_date: Dict[pd.Timestamp, Dict[str, float]] = {}
 
        for date, row in scores_pivot.iterrows():
            valid = row.dropna()
            if len(valid) < 3:
                continue
 
            top_assets = valid.nlargest(self.n_top).index.tolist()
 
            # Trailing volatility for selected assets
            vols: Dict[str, float] = {}
            for asset in top_assets:
                if asset in raw_close.columns:
                    hist = daily_ret[asset].loc[:date].tail(21)
                    v = float(hist.std()) if hist.notna().sum() >= 5 else 0.02
                    vols[asset] = max(v, 1e-4)
                else:
                    vols[asset] = 0.02
 
            # Raw weight based on weighting_type
            if self.weighting_type == "inverse_vol":
                raw_w: Dict[str, float] = {
                    a: max(float(valid[a]), MIN_WEIGHT_EPS) / vols[a]
                    for a in top_assets
                }
            else:  # score_prop: weight proportional to predicted score only
                raw_w: Dict[str, float] = {
                    a: max(float(valid[a]), MIN_WEIGHT_EPS)
                    for a in top_assets
                }
 
            # Initial normalization
            total = sum(raw_w.values())
            if total <= 0:
                continue
            w = {k: v / total for k, v in raw_w.items()}
 
            # Iterative cap enforcement
            for _ in range(5):
                excess   = 0.0
                uncapped: List[str] = []
                for asset, wt in w.items():
                    cap = CAP_TIER4 if ASSET_TIER_MAP.get(asset, 1) == 4 else CAP_ETF
                    if wt > cap:
                        excess += wt - cap
                        w[asset] = cap
                    else:
                        uncapped.append(asset)
 
                if excess < 1e-6 or not uncapped:
                    break
 
                add = excess / len(uncapped)
                for a in uncapped:
                    w[a] += add
 
                t2 = sum(w.values())
                if t2 > 0:
                    w = {k: v / t2 for k, v in w.items()}
 
            weights_by_date[date] = w
 
        return weights_by_date
 
    # ---------------------------------------------------------------- #
    # Private: return simulation                                         #
    # ---------------------------------------------------------------- #
 
    def _simulate_daily_returns(
        self,
        weights_by_date: Dict[pd.Timestamp, Dict[str, float]],
        adj_close:       pd.DataFrame,
    ) -> pd.Series:
        """
        Simulate daily portfolio returns by forward-filling weights.
 
        Per V8 rule: adjusted close for return calculations.
        Portfolio weights are updated at each refit date and held constant
        until the next refit (no daily rebalancing).
        """
        if not weights_by_date:
            return pd.Series(dtype=float)
 
        daily_ret   = adj_close.pct_change()
        refit_dates = sorted(weights_by_date.keys())
        first_refit = refit_dates[0]
 
        strategy_ret = pd.Series(np.nan, index=adj_close.index)
        current_w: Optional[Dict[str, float]] = None
        refit_idx = 0
 
        for date in adj_close.index:
            while (refit_idx < len(refit_dates)
                   and refit_dates[refit_idx] <= date):
                current_w = weights_by_date[refit_dates[refit_idx]]
                refit_idx += 1
 
            if current_w is None or date < first_refit:
                continue
 
            port_ret = 0.0
            for asset, weight in current_w.items():
                if asset in daily_ret.columns:
                    r = daily_ret.loc[date, asset]
                    if pd.notna(r):
                        port_ret += weight * r
 
            strategy_ret[date] = port_ret
 
        return strategy_ret
 
    # ---------------------------------------------------------------- #
    # Private: metrics                                                   #
    # ---------------------------------------------------------------- #
 
    def _compute_metrics(self, daily_returns: pd.Series) -> Dict[str, Any]:
        n   = len(daily_returns)
        if n < 63:
            return {}
        cagr    = float((1.0 + daily_returns).prod() ** (252.0 / n) - 1.0)
        sharpe  = float(daily_returns.mean() / (daily_returns.std() + 1e-10)
                        * np.sqrt(252))
        cum     = (1.0 + daily_returns).cumprod()
        peak    = cum.expanding().max()
        max_dd  = float((cum / peak - 1.0).min())
        calmar  = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else np.nan
        down    = daily_returns[daily_returns < 0]
        sortino = float(
            cagr / (down.std() * np.sqrt(252) + 1e-10)
        ) if len(down) > 0 else np.nan
        return {
            "CAGR":    round(cagr,   4),
            "Sharpe":  round(sharpe, 4),
            "Max_DD":  round(max_dd, 4),
            "Calmar":  round(calmar, 4) if not np.isnan(calmar) else np.nan,
            "Sortino": round(sortino, 4) if not np.isnan(sortino) else np.nan,
            "N_Days":  n,
        }
 
    def _log_summary(self) -> None:
        if self.metrics is None or self.metrics.empty:
            return
        logger.info("=" * 66)
        logger.info("StateAllocator summary:")
        for idx, row in self.metrics.iterrows():
            logger.info(
                f"  {idx:<24} Sharpe={row.get('Sharpe', np.nan):.3f}  "
                f"CAGR={row.get('CAGR', np.nan):.2%}  "
                f"MaxDD={row.get('Max_DD', np.nan):.2%}"
            )
 
    # ---------------------------------------------------------------- #
    # Public: validate                                                   #
    # ---------------------------------------------------------------- #
 
    def validate(self) -> bool:
        """
        Validate the state-responsive allocator.
 
        Pass criteria:
          1. Model C Sharpe > SPY buy-and-hold Sharpe
          2. Model C Sharpe ≥ Model A Sharpe  (analog memory adds value)
          3. Model C Max Drawdown > −30%
 
        Ablation analysis quantifies trajectory and analog contributions.
        Holdings analysis shows which assets the model selects most often.
        """
        if self.metrics is None:
            logger.error("validate(): call build() first.")
            return False
 
        m  = self.metrics
        dr = self.daily_returns
 
        print("\n" + "=" * 76)
        print("STRATEGY 1: STATE-RESPONSIVE ALLOCATOR — VALIDATION")
        print("=" * 76)
        print(f"\n  Parameters:")
        print(f"    Min training samples : {self.min_train} days")
        print(f"    Refit frequency      : every {self.refit_every} days")
        print(f"    Forward window       : {self.forward_window} days")
        print(f"    Assets selected      : top {self.n_top} per refit")
        print(f"    Target               : tail-aware reward "
              f"(V8 meta-allocator formula)")
        print(f"    Universe             : {len(TRADING_UNIVERSE)} assets "
              f"(TIER1-4)")
 
        # ---- Performance table ----
        print(f"\n  {'Strategy':<28} {'Sharpe':>7} {'CAGR':>8} "
              f"{'Max DD':>8} {'Calmar':>8} {'Sortino':>8}")
        print("  " + "─" * 72)
 
        row_order = [
            ("model_A",      "Model A  (pillars)"),
            ("model_B",      "Model B  (+ trajectory)"),
            ("model_C",      "Model C  (+ analog scores)"),
            ("spy_buyhold",  "SPY buy-and-hold"),
            ("equal_weight", "Equal weight (~33 assets)"),
        ]
        for key, label in row_order:
            if key not in m.index:
                continue
            r = m.loc[key]
            sh = r.get("Sharpe",  np.nan)
            ca = r.get("CAGR",    np.nan)
            dd = r.get("Max_DD",  np.nan)
            cl = r.get("Calmar",  np.nan)
            so = r.get("Sortino", np.nan)
            print(
                f"  {label:<28} "
                f"{sh:>7.3f} {ca:>8.2%} {dd:>8.2%} "
                f"{cl:>8.3f} {so:>8.3f}"
            )
 
        # ---- Pass / Fail ----
        def _g(key: str, col: str) -> float:
            return float(m.loc[key, col]) if key in m.index and col in m.columns else np.nan
 
        c_sharpe   = _g("model_C",    "Sharpe")
        a_sharpe   = _g("model_A",    "Sharpe")
        b_sharpe   = _g("model_B",    "Sharpe")
        spy_sharpe = _g("spy_buyhold","Sharpe")
        c_dd       = _g("model_C",    "Max_DD")
 
        print(f"\n  {'─' * 72}")
        print("  Pass Criteria")
        print(f"  {'─' * 72}")
 
        all_pass = True
 
        crit1 = not np.isnan(c_sharpe) and c_sharpe > spy_sharpe
        all_pass = all_pass and crit1
        print(f"  1. Model C ({c_sharpe:.3f}) > SPY ({spy_sharpe:.3f})  "
              f"{'✓' if crit1 else '✗'}")
 
        delta_ac = c_sharpe - a_sharpe
        crit2 = not np.isnan(delta_ac) and delta_ac >= 0.0
        all_pass = all_pass and crit2
        print(f"  2. Model C ({c_sharpe:.3f}) ≥ Model A ({a_sharpe:.3f}), "
              f"Δ={delta_ac:+.3f}  {'✓' if crit2 else '✗'}")
 
        crit3 = not np.isnan(c_dd) and c_dd > -0.30
        all_pass = all_pass and crit3
        print(f"  3. Model C Max DD ({c_dd:.2%}) > −30.00%  "
              f"{'✓' if crit3 else '✗'}")
 
        # ---- Ablation breakdown ----
        print(f"\n  {'─' * 72}")
        print("  Ablation Analysis")
        print(f"  {'─' * 72}")
        traj_delta  = b_sharpe - a_sharpe
        analog_delta = c_sharpe - b_sharpe
        print(f"  Trajectory contribution : {traj_delta:+.3f} Sharpe "
              f"(A={a_sharpe:.3f} → B={b_sharpe:.3f})")
        print(f"  Analog contribution     : {analog_delta:+.3f} Sharpe "
              f"(B={b_sharpe:.3f} → C={c_sharpe:.3f})")
 
        if not np.isnan(analog_delta):
            if analog_delta > 0.10:
                print("  Analog memory: MATERIAL CONTRIBUTION ✓")
            elif analog_delta >= 0.0:
                print("  Analog memory: marginal positive")
            else:
                print("  Analog memory: NEGATIVE contribution ✗")
 
        # ---- Cumulative return ----
        model_c_col = "model_C"
        if model_c_col in dr.columns and "spy_buyhold" in dr.columns:
            c_cum  = float((1 + dr[model_c_col].dropna()).prod() - 1)
            sp_cum = float((1 + dr["spy_buyhold"].dropna()).prod() - 1)
            print(f"\n  Cumulative return (evaluation period):")
            print(f"    Model C    : {c_cum:.1%}")
            print(f"    SPY        : {sp_cum:.1%}")
 
        # ---- Holdings frequency ----
        if self.predictions and "C" in self.predictions:
            scores = self.predictions["C"]
            if not scores.empty:
                counts: Dict[str, int] = {}
                n_refits = len(scores)
                for _, row in scores.iterrows():
                    for a in row.dropna().nlargest(self.n_top).index:
                        counts[a] = counts.get(a, 0) + 1
 
                print(f"\n  Most frequently selected assets "
                      f"(Model C, top {self.n_top}):")
                for asset, cnt in sorted(counts.items(),
                                         key=lambda x: -x[1])[:15]:
                    tier = ASSET_TIER_MAP.get(asset, 0)
                    pct  = cnt / n_refits * 100
                    print(f"    {asset:<10} {pct:5.1f}%  [Tier {tier}]")
 
        # ---- Summary ----
        print(f"\n  {'=' * 72}")
        verdict = (
            "PASS ✓  — allocator demonstrates value over baselines"
            if all_pass else
            "FAIL ✗  — review failing criteria above"
        )
        print(f"  Overall: {verdict}")
        print("=" * 76)
 
        return all_pass